use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::io::Cursor;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::ipc::reader::{FileReader, StreamReader};
use datafusion_expr::Expr;
use datafusion_sql::unparser::dialect::CustomDialectBuilder;
use datafusion_sql::unparser::Unparser;
use lance_core::Error as LanceError;
use lance_namespace::models::{
    QueryTableRequest, QueryTableRequestColumns, QueryTableRequestFullTextQuery,
    QueryTableRequestVector, StringFtsQuery,
};
use lance_namespace::{LanceNamespace, NamespaceError};
use lance_namespace_impls::{DirectoryNamespaceBuilder, RestNamespaceBuilder};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::StreamHandle;
use super::util::{
    cstr_to_str, parse_headers_tsv, parse_optional_filter_ir, slice_from_ptr, FfiError, FfiResult,
};

const NAMESPACE_KIND_DIRECTORY: u8 = 0;
const NAMESPACE_KIND_REST: u8 = 1;

#[repr(C)]
pub struct LanceNamespaceQueryConfig {
    namespace_kind: u8,
    root: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    columns: *const *const c_char,
    columns_len: usize,
    expected_columns: *const *const c_char,
    expected_columns_len: usize,
    filter: *const c_char,
    dataset_version: u64,
    k: u64,
    prefilter: u8,
}

#[repr(C)]
pub struct LanceNamespaceVectorSearchOptions {
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    nprobes: u64,
    refine_factor: u64,
    use_index: u8,
}

#[repr(C)]
pub struct LanceNamespaceFtsSearchOptions {
    text_column: *const c_char,
    query: *const c_char,
}

enum NamespaceBackend {
    Directory {
        root: String,
        storage_options: HashMap<String, String>,
    },
    Rest {
        endpoint: String,
        bearer_token: Option<String>,
        api_key: Option<String>,
        delimiter: Option<String>,
        headers: Vec<(String, String)>,
    },
}

struct ParsedNamespaceQueryConfig {
    backend: NamespaceBackend,
    table_id: String,
    columns: Vec<String>,
    expected_columns: Vec<String>,
    filter: Option<String>,
    dataset_version: i64,
    k: i32,
    prefilter: bool,
}

unsafe fn optional_cstr_to_string(
    ptr: *const c_char,
    what: &'static str,
) -> FfiResult<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = unsafe { cstr_to_str(ptr, what)? };
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s.to_string()))
    }
}

unsafe fn parse_string_array(
    ptr: *const *const c_char,
    len: usize,
    what: &'static str,
) -> FfiResult<Vec<String>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null with non-zero length"),
        ));
    }

    let values = unsafe { slice_from_ptr(ptr, len, what)? };
    let mut out = Vec::with_capacity(len);
    for (idx, &value_ptr) in values.iter().enumerate() {
        if value_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("{what}[{idx}] is null"),
            ));
        }
        let value = unsafe { CStr::from_ptr(value_ptr) }
            .to_str()
            .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("{what}[{idx}] utf8: {err}")))?;
        out.push(value.to_string());
    }
    Ok(out)
}

unsafe fn parse_storage_options(
    keys: *const *const c_char,
    values: *const *const c_char,
    len: usize,
) -> FfiResult<HashMap<String, String>> {
    let keys = unsafe { parse_string_array(keys, len, "option_keys")? };
    let values = unsafe { parse_string_array(values, len, "option_values")? };
    if keys.len() != values.len() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option key/value length mismatch",
        ));
    }
    Ok(keys.into_iter().zip(values).collect())
}

fn validate_rest_table_id(table_id: &str, delimiter: Option<&str>) -> FfiResult<()> {
    if delimiter == Some("") {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace delimiter must not be empty",
        ));
    }
    let delimiter = delimiter.unwrap_or("$");
    if !table_id.split(delimiter).any(|segment| !segment.is_empty()) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace table id must identify a table",
        ));
    }
    Ok(())
}

fn validate_namespace_location(value: &str, what: &'static str) -> FfiResult<()> {
    if value.trim().is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} must not be empty"),
        ));
    }
    Ok(())
}

unsafe fn parse_config(
    config: *const LanceNamespaceQueryConfig,
) -> FfiResult<ParsedNamespaceQueryConfig> {
    if config.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace query config is null",
        ));
    }
    let config = unsafe { &*config };
    let table_id = unsafe { cstr_to_str(config.table_id, "table_id")? }.to_string();
    if table_id.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace table id must not be empty",
        ));
    }
    let columns = unsafe { parse_string_array(config.columns, config.columns_len, "columns")? };
    let expected_columns = unsafe {
        parse_string_array(
            config.expected_columns,
            config.expected_columns_len,
            "expected_columns",
        )?
    };
    let filter = unsafe { optional_cstr_to_string(config.filter, "filter")? };
    let dataset_version = i64::try_from(config.dataset_version).map_err(|_| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            "dataset version must fit in i64",
        )
    })?;
    if dataset_version <= 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "dataset version must be greater than zero",
        ));
    }
    let k = if config.k > i32::MAX as u64 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "k must fit in i32",
        ));
    } else {
        config.k as i32
    };

    let backend = match config.namespace_kind {
        NAMESPACE_KIND_DIRECTORY => {
            let root = unsafe { cstr_to_str(config.root, "root")? }.to_string();
            validate_namespace_location(&root, "namespace root")?;
            let storage_options = unsafe {
                parse_storage_options(config.option_keys, config.option_values, config.options_len)?
            };
            NamespaceBackend::Directory {
                root,
                storage_options,
            }
        }
        NAMESPACE_KIND_REST => {
            let endpoint = unsafe { cstr_to_str(config.endpoint, "endpoint")? }.to_string();
            validate_namespace_location(&endpoint, "namespace endpoint")?;
            let bearer_token =
                unsafe { optional_cstr_to_string(config.bearer_token, "bearer_token")? };
            let api_key = unsafe { optional_cstr_to_string(config.api_key, "api_key")? };
            // Unlike optional credentials, an explicitly empty delimiter is
            // not equivalent to omission: `str::split("")` would corrupt the
            // table identity into character-sized segments.
            let delimiter = if config.delimiter.is_null() {
                None
            } else {
                Some(unsafe { cstr_to_str(config.delimiter, "delimiter")? }.to_string())
            };
            validate_rest_table_id(&table_id, delimiter.as_deref())?;
            let headers_tsv =
                unsafe { optional_cstr_to_string(config.headers_tsv, "headers_tsv")? };
            let headers = parse_headers_tsv(headers_tsv.as_deref())?;
            NamespaceBackend::Rest {
                endpoint,
                bearer_token,
                api_key,
                delimiter,
                headers,
            }
        }
        other => {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("unknown namespace kind: {other}"),
            ))
        }
    };

    Ok(ParsedNamespaceQueryConfig {
        backend,
        table_id,
        columns,
        expected_columns,
        filter,
        dataset_version,
        k,
        prefilter: config.prefilter != 0,
    })
}

fn apply_base_request(config: &ParsedNamespaceQueryConfig, request: &mut QueryTableRequest) {
    request.id = Some(match &config.backend {
        NamespaceBackend::Rest { delimiter, .. } => {
            let delimiter = delimiter.as_deref().unwrap_or("$");
            config
                .table_id
                .split(delimiter)
                .filter(|segment| !segment.is_empty())
                .map(str::to_string)
                .collect()
        }
        NamespaceBackend::Directory { .. } => vec![config.table_id.clone()],
    });
    request.prefilter = Some(config.prefilter);
    request.version = Some(config.dataset_version);
    if !config.columns.is_empty() {
        let mut columns = QueryTableRequestColumns::new();
        columns.column_names = Some(config.columns.clone());
        request.columns = Some(Box::new(columns));
    }
    if let Some(filter) = &config.filter {
        request.filter = Some(filter.clone());
    }
}

const NAMESPACE_SCAN_PAGE_ROWS: i32 = 16_384;
const REST_QUERY_ATTEMPTS: usize = 3;
const REST_QUERY_TIMEOUT: Duration = Duration::from_secs(30);

struct PreparedNamespace {
    namespace: Arc<dyn LanceNamespace>,
    is_rest: bool,
}

fn is_retryable_rest_query_error(error: &LanceError) -> bool {
    match error {
        LanceError::Timeout { .. } => true,
        LanceError::Namespace { source, .. } => source
            .downcast_ref::<NamespaceError>()
            .is_some_and(|error| {
                matches!(
                    error,
                    NamespaceError::ServiceUnavailable { .. }
                        | NamespaceError::Throttling { .. }
                        | NamespaceError::Internal { .. }
                )
            }),
        _ => false,
    }
}

async fn prepare_namespace(backend: &NamespaceBackend) -> FfiResult<PreparedNamespace> {
    match backend {
        NamespaceBackend::Directory {
            root,
            storage_options,
        } => {
            let mut builder = DirectoryNamespaceBuilder::new(root.as_str()).manifest_enabled(false);
            if !storage_options.is_empty() {
                builder = builder.storage_options(storage_options.clone());
            }
            let namespace = builder.build().await.map_err(|err| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    format!("dir namespace build '{root}': {err}"),
                )
            })?;
            Ok(PreparedNamespace {
                namespace: Arc::new(namespace),
                is_rest: false,
            })
        }
        NamespaceBackend::Rest {
            endpoint,
            bearer_token,
            api_key,
            delimiter,
            headers,
        } => {
            let mut builder = RestNamespaceBuilder::new(endpoint.as_str());
            if let Some(token) = bearer_token {
                builder = builder.header("Authorization", format!("Bearer {token}"));
            }
            if let Some(key) = api_key {
                builder = builder.header("x-api-key", key);
            }
            for (key, value) in headers {
                builder = builder.header(key, value);
            }
            if let Some(delimiter) = delimiter {
                builder = builder.delimiter(delimiter);
            }
            Ok(PreparedNamespace {
                namespace: Arc::new(builder.build()),
                is_rest: true,
            })
        }
    }
}

async fn execute_query_table(
    prepared: &PreparedNamespace,
    request: &QueryTableRequest,
) -> FfiResult<Vec<u8>> {
    let attempts = if prepared.is_rest {
        REST_QUERY_ATTEMPTS
    } else {
        1
    };
    let mut last_error = String::new();

    for attempt in 1..=attempts {
        let query = prepared.namespace.query_table(request.clone());
        let result = if prepared.is_rest {
            match tokio::time::timeout(REST_QUERY_TIMEOUT, query).await {
                Ok(result) => result,
                Err(_) => {
                    last_error = format!(
                        "REST namespace query_table timed out after {} seconds",
                        REST_QUERY_TIMEOUT.as_secs()
                    );
                    if attempt < attempts {
                        log::warn!("{last_error}; retrying ({}/{})", attempt + 1, attempts);
                        tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
                        continue;
                    }
                    break;
                }
            }
        } else {
            query.await
        };

        match result {
            Ok(bytes) => return Ok(bytes.to_vec()),
            Err(err) => {
                last_error = if prepared.is_rest {
                    format!("REST namespace query_table: {err}")
                } else {
                    format!("directory namespace query_table: {err}")
                };
                if prepared.is_rest && !is_retryable_rest_query_error(&err) {
                    return Err(FfiError::new(ErrorCode::NamespaceQueryTable, last_error));
                }
                if attempt < attempts {
                    log::warn!(
                        "{last_error}; retrying idempotent query ({}/{})",
                        attempt + 1,
                        attempts
                    );
                    tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
                }
            }
        }
    }

    Err(FfiError::new(ErrorCode::NamespaceQueryTable, last_error))
}

fn reorder_response_batches(
    batches: Vec<RecordBatch>,
    schema: &arrow::datatypes::Schema,
    expected_columns: &[String],
) -> FfiResult<Vec<RecordBatch>> {
    if expected_columns.is_empty() {
        return Ok(batches);
    }

    if schema.fields().len() != expected_columns.len() {
        return Err(FfiError::new(
            ErrorCode::NamespaceQueryTable,
            format!(
                "namespace query_table schema mismatch: expected columns {:?} ({}), got {} columns {:?}",
                expected_columns,
                expected_columns.len(),
                schema.fields().len(),
                schema
                    .fields()
                    .iter()
                    .map(|field| field.name())
                    .collect::<Vec<_>>()
            ),
        ));
    }

    let mut actual_indices = HashMap::with_capacity(schema.fields().len());
    for (index, field) in schema.fields().iter().enumerate() {
        if actual_indices
            .insert(field.name().as_str(), index)
            .is_some()
        {
            return Err(FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!(
                    "namespace query_table returned duplicate column '{}'",
                    field.name()
                ),
            ));
        }
    }

    let mut seen_expected = std::collections::HashSet::with_capacity(expected_columns.len());
    let mut order = Vec::with_capacity(expected_columns.len());
    for column in expected_columns {
        if !seen_expected.insert(column.as_str()) {
            return Err(FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("namespace query_table expected duplicate column '{column}'"),
            ));
        }
        let index = actual_indices
            .get(column.as_str())
            .copied()
            .ok_or_else(|| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    format!(
                        "namespace query_table schema mismatch: did not return expected column '{column}'"
                    ),
                )
            })?;
        order.push(index);
    }

    if order.iter().copied().eq(0..order.len()) {
        return Ok(batches);
    }

    let reordered_schema = Arc::new(arrow::datatypes::Schema::new_with_metadata(
        order
            .iter()
            .map(|index| schema.field(*index).clone())
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    ));
    batches
        .into_iter()
        .map(|batch| {
            let columns = order
                .iter()
                .map(|index| batch.column(*index).clone())
                .collect::<Vec<_>>();
            RecordBatch::try_new(reordered_schema.clone(), columns).map_err(|error| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    format!("reorder namespace query_table response: {error}"),
                )
            })
        })
        .collect()
}

fn ipc_bytes_to_batches(
    bytes: Vec<u8>,
    expected_columns: &[String],
) -> FfiResult<Vec<RecordBatch>> {
    if bytes.starts_with(b"ARROW1") {
        let reader = FileReader::try_new(Cursor::new(bytes), None).map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("open Arrow IPC file response: {err}"),
            )
        })?;
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>().map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("read Arrow IPC file: {err}"),
            )
        })?;
        reorder_response_batches(batches, schema.as_ref(), expected_columns)
    } else {
        let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("open Arrow IPC stream response: {err}"),
            )
        })?;
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>().map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("read Arrow IPC stream: {err}"),
            )
        })?;
        reorder_response_batches(batches, schema.as_ref(), expected_columns)
    }
}

fn execute_to_stream(
    config: ParsedNamespaceQueryConfig,
    request: QueryTableRequest,
) -> FfiResult<StreamHandle> {
    let prepared = runtime::block_on(prepare_namespace(&config.backend))
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;
    let bytes = runtime::block_on(execute_query_table(&prepared, &request))
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;
    let batches = ipc_bytes_to_batches(bytes, &config.expected_columns)?;
    Ok(StreamHandle::Batches(batches.into_iter()))
}

pub(crate) struct NamespaceQueryStream {
    prepared: PreparedNamespace,
    request: QueryTableRequest,
    expected_columns: Vec<String>,
    next_offset: i64,
    remaining: Option<u64>,
    pending: std::vec::IntoIter<RecordBatch>,
    done: bool,
}

impl NamespaceQueryStream {
    fn try_new(config: ParsedNamespaceQueryConfig, request: QueryTableRequest) -> FfiResult<Self> {
        if request.k < 0 {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "namespace scan k must be non-negative",
            ));
        }
        let prepared = runtime::block_on(prepare_namespace(&config.backend))
            .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;
        let mut stream = Self {
            prepared,
            next_offset: i64::from(request.offset.unwrap_or(0)),
            remaining: (request.k != 0).then_some(request.k as u64),
            request,
            expected_columns: config.expected_columns,
            pending: Vec::new().into_iter(),
            done: false,
        };
        // Execute the first page while constructing the stream.  The C++ scan
        // layer can safely fall back to DuckDB-side filtering only when a
        // rejected pushed filter is reported by stream creation; deferring the
        // request until lance_stream_next would turn that recoverable bind-time
        // failure into a mid-scan query failure.
        stream.fetch_page()?;
        Ok(stream)
    }

    fn fetch_page(&mut self) -> FfiResult<()> {
        let page_rows = self
            .remaining
            .map(|remaining| remaining.min(NAMESPACE_SCAN_PAGE_ROWS as u64))
            .unwrap_or(NAMESPACE_SCAN_PAGE_ROWS as u64) as i32;
        if page_rows == 0 {
            self.done = true;
            return Ok(());
        }

        let mut request = self.request.clone();
        request.k = page_rows;
        request.offset = if self.next_offset == 0 {
            None
        } else {
            Some(i32::try_from(self.next_offset).map_err(|_| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    "namespace scan offset exceeds the query_table i32 limit",
                )
            })?)
        };
        let bytes = runtime::block_on(execute_query_table(&self.prepared, &request))
            .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;
        let batches = ipc_bytes_to_batches(bytes, &self.expected_columns)?;
        let returned_rows = batches.iter().try_fold(0_u64, |rows, batch| {
            rows.checked_add(batch.num_rows() as u64).ok_or_else(|| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    "namespace query_table response row count overflow",
                )
            })
        })?;
        if returned_rows > page_rows as u64 {
            return Err(FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!(
                    "namespace query_table returned {returned_rows} rows for a {page_rows}-row page"
                ),
            ));
        }
        if returned_rows == 0 {
            self.done = true;
            return Ok(());
        }

        self.next_offset = self
            .next_offset
            .checked_add(returned_rows as i64)
            .ok_or_else(|| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    "namespace scan offset overflow",
                )
            })?;
        if let Some(remaining) = self.remaining.as_mut() {
            *remaining -= returned_rows;
            if *remaining == 0 {
                self.done = true;
            }
        }
        self.pending = batches.into_iter();
        Ok(())
    }

    pub(crate) fn next_batch(&mut self) -> FfiResult<Option<RecordBatch>> {
        loop {
            if let Some(batch) = self.pending.next() {
                return Ok(Some(batch));
            }
            if self.done {
                return Ok(None);
            }
            self.fetch_page()?;
        }
    }
}

fn filter_expr_to_sql(expr: &Expr) -> FfiResult<String> {
    let dialect = CustomDialectBuilder::new()
        .with_identifier_quote_style('`')
        .build();
    Unparser::new(&dialect)
        .expr_to_sql(expr)
        .map(|sql| sql.to_string())
        .map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("unparse namespace scan filter: {err}"),
            )
        })
}

fn build_namespace_scan_request(
    config: &ParsedNamespaceQueryConfig,
    filter_expr: Option<&Expr>,
    limit: i64,
    offset: i64,
    with_row_id: bool,
) -> FfiResult<QueryTableRequest> {
    if limit < -1 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace scan limit must be >= -1",
        ));
    }
    if offset < 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace scan offset must be non-negative",
        ));
    }
    if limit == -1 && offset != 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace query_table cannot apply OFFSET without LIMIT",
        ));
    }

    let k = if limit == -1 {
        0
    } else {
        i32::try_from(limit).map_err(|_| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                "namespace scan limit must fit in i32",
            )
        })?
    };
    let offset = i32::try_from(offset).map_err(|_| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace scan offset must fit in i32",
        )
    })?;

    let mut request = QueryTableRequest::new(k, QueryTableRequestVector::new());
    apply_base_request(config, &mut request);
    request.prefilter = None;
    if offset != 0 {
        request.offset = Some(offset);
    }
    if with_row_id {
        request.with_row_id = Some(true);
        let mut clear_columns = false;
        if let Some(columns) = request.columns.as_mut() {
            if let Some(column_names) = columns.column_names.as_mut() {
                column_names.retain(|name| name != "_rowid");
                clear_columns = column_names.is_empty();
            }
        }
        if clear_columns {
            request.columns = None;
        }
    }

    if let Some(filter_expr) = filter_expr {
        let filter_sql = filter_expr_to_sql(filter_expr)?;
        request.filter = Some(match request.filter.take() {
            Some(existing) => format!("({existing}) AND ({filter_sql})"),
            None => filter_sql,
        });
    }
    Ok(request)
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_create_namespace_scan_stream_ir(
    config: *const LanceNamespaceQueryConfig,
    filter_ir: *const u8,
    filter_ir_len: usize,
    limit: i64,
    offset: i64,
    with_row_id: u8,
) -> *mut c_void {
    match unsafe {
        create_namespace_scan_stream_ir_inner(
            config,
            filter_ir,
            filter_ir_len,
            limit,
            offset,
            with_row_id,
        )
    } {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

unsafe fn create_namespace_scan_stream_ir_inner(
    config: *const LanceNamespaceQueryConfig,
    filter_ir: *const u8,
    filter_ir_len: usize,
    limit: i64,
    offset: i64,
    with_row_id: u8,
) -> FfiResult<StreamHandle> {
    let config = unsafe { parse_config(config)? };
    let filter_expr = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::NamespaceQueryTable,
            "namespace scan filter_ir",
        )?
    };
    let request = build_namespace_scan_request(
        &config,
        filter_expr.as_ref(),
        limit,
        offset,
        with_row_id != 0,
    )?;
    if limit == 0 {
        return Ok(StreamHandle::Batches(Vec::new().into_iter()));
    }
    Ok(StreamHandle::Namespace(Box::new(
        NamespaceQueryStream::try_new(config, request)?,
    )))
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_create_namespace_vector_search_stream(
    config: *const LanceNamespaceQueryConfig,
    options: *const LanceNamespaceVectorSearchOptions,
) -> *mut c_void {
    match create_namespace_vector_search_stream_inner(config, options) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

unsafe fn create_namespace_vector_search_stream_inner(
    config: *const LanceNamespaceQueryConfig,
    options: *const LanceNamespaceVectorSearchOptions,
) -> FfiResult<StreamHandle> {
    if options.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace vector search options is null",
        ));
    }
    let config = unsafe { parse_config(config)? };
    if config.k == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace vector search k must be positive",
        ));
    }
    let options = unsafe { &*options };
    let vector_column = unsafe { cstr_to_str(options.vector_column, "vector_column")? };
    let query_values =
        unsafe { slice_from_ptr(options.query_values, options.query_len, "query_values")? };
    if query_values.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "query vector must be non-empty",
        ));
    }

    let mut vector = QueryTableRequestVector::new();
    vector.single_vector = Some(query_values.to_vec());
    let mut request = QueryTableRequest::new(config.k, vector);
    apply_base_request(&config, &mut request);
    request.vector_column = Some(vector_column.to_string());
    if options.nprobes != 0 {
        request.nprobes =
            Some(options.nprobes.try_into().map_err(|_| {
                FfiError::new(ErrorCode::InvalidArgument, "nprobes must fit in i32")
            })?);
    }
    if options.refine_factor != 0 {
        request.refine_factor = Some(options.refine_factor.try_into().map_err(|_| {
            FfiError::new(ErrorCode::InvalidArgument, "refine_factor must fit in i32")
        })?);
    }
    request.bypass_vector_index = Some(options.use_index == 0);

    execute_to_stream(config, request)
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_create_namespace_fts_search_stream(
    config: *const LanceNamespaceQueryConfig,
    options: *const LanceNamespaceFtsSearchOptions,
) -> *mut c_void {
    match create_namespace_fts_search_stream_inner(config, options) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

unsafe fn create_namespace_fts_search_stream_inner(
    config: *const LanceNamespaceQueryConfig,
    options: *const LanceNamespaceFtsSearchOptions,
) -> FfiResult<StreamHandle> {
    if options.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace FTS search options is null",
        ));
    }
    let config = unsafe { parse_config(config)? };
    if config.k == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace FTS search k must be positive",
        ));
    }
    let options = unsafe { &*options };
    let text_column = unsafe { cstr_to_str(options.text_column, "text_column")? };
    let query = unsafe { cstr_to_str(options.query, "query")? };

    let mut request = QueryTableRequest::new(config.k, QueryTableRequestVector::new());
    apply_base_request(&config, &mut request);

    let mut string_query = StringFtsQuery::new(query.to_string());
    string_query.columns = Some(vec![text_column.to_string()]);
    let mut fts_query = QueryTableRequestFullTextQuery::new();
    fts_query.string_query = Some(Box::new(string_query));
    request.full_text_query = Some(Box::new(fts_query));

    execute_to_stream(config, request)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::{FileWriter, StreamWriter};
    use datafusion_expr::{col, lit};

    use super::*;

    fn rest_config(columns: &[&str]) -> ParsedNamespaceQueryConfig {
        ParsedNamespaceQueryConfig {
            backend: NamespaceBackend::Rest {
                endpoint: "http://localhost:8000".to_string(),
                bearer_token: None,
                api_key: None,
                delimiter: Some("$".to_string()),
                headers: Vec::new(),
            },
            table_id: "parent$child$table".to_string(),
            columns: columns.iter().map(|column| (*column).to_string()).collect(),
            expected_columns: columns.iter().map(|column| (*column).to_string()).collect(),
            filter: None,
            dataset_version: 42,
            k: 0,
            prefilter: true,
        }
    }

    #[test]
    fn namespace_query_retries_only_transient_rest_errors() {
        let transient: LanceError = NamespaceError::ServiceUnavailable {
            message: "temporary".to_string(),
        }
        .into();
        let deterministic: LanceError = NamespaceError::InvalidInput {
            message: "invalid request".to_string(),
        }
        .into();

        assert!(is_retryable_rest_query_error(&transient));
        assert!(!is_retryable_rest_query_error(&deterministic));
    }

    #[test]
    fn namespace_query_table_ids_drop_empty_segments() {
        let mut config = rest_config(&["id"]);
        config.table_id = "$parent$$table$".to_string();
        let mut request = QueryTableRequest::new(1, QueryTableRequestVector::new());
        apply_base_request(&config, &mut request);
        assert_eq!(
            request.id,
            Some(vec!["parent".to_string(), "table".to_string()])
        );
    }

    #[test]
    fn namespace_query_rejects_empty_delimiter_and_root_table_id() {
        assert!(validate_rest_table_id("table", Some("")).is_err());
        assert!(validate_rest_table_id("$$", Some("$")).is_err());
        assert!(validate_rest_table_id("$table$", Some("$")).is_ok());
    }

    #[test]
    fn namespace_query_rejects_empty_locations() {
        assert!(validate_namespace_location("", "namespace root").is_err());
        assert!(validate_namespace_location("  ", "namespace endpoint").is_err());
        assert!(validate_namespace_location("/", "namespace root").is_ok());
        assert!(validate_namespace_location("file:///", "namespace root").is_ok());
    }

    #[test]
    fn namespace_scan_request_pushes_query_shape() {
        let config = rest_config(&["name", "_rowid"]);
        let filter = col("age").gt(lit(21));
        let request = build_namespace_scan_request(&config, Some(&filter), 5, 2, true).unwrap();

        assert_eq!(
            request.id,
            Some(vec![
                "parent".to_string(),
                "child".to_string(),
                "table".to_string()
            ])
        );
        assert_eq!(request.k, 5);
        assert_eq!(request.version, Some(42));
        assert_eq!(request.offset, Some(2));
        assert_eq!(request.with_row_id, Some(true));
        assert_eq!(request.prefilter, None);
        assert_eq!(
            request.columns.unwrap().column_names,
            Some(vec!["name".to_string()])
        );
        let filter = request.filter.unwrap();
        assert!(filter.contains("age"));
        assert!(filter.contains("21"));
    }

    #[test]
    fn namespace_scan_request_rejects_offset_without_limit() {
        let config = rest_config(&["name"]);
        let error = build_namespace_scan_request(&config, None, -1, 1, false).unwrap_err();
        assert!(error.message.contains("cannot apply OFFSET without LIMIT"));
    }

    fn ipc_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2]))]).unwrap()
    }

    #[test]
    fn namespace_query_reads_stream_and_validates_schema() {
        let batch = ipc_batch();
        let mut bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let batches = ipc_bytes_to_batches(bytes.clone(), &["id".to_string()]).unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);

        let error = ipc_bytes_to_batches(bytes, &["wrong".to_string()]).unwrap_err();
        assert!(error.message.contains("schema mismatch"));
        assert!(!error.message.contains("localhost"));
    }

    #[test]
    fn namespace_query_reads_file_and_validates_column_order() {
        let batch = ipc_batch();
        let mut bytes = Vec::new();
        {
            let mut writer = FileWriter::try_new(&mut bytes, batch.schema().as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let batches = ipc_bytes_to_batches(bytes.clone(), &["id".to_string()]).unwrap();
        assert_eq!(batches.len(), 1);

        let error =
            ipc_bytes_to_batches(bytes, &["id".to_string(), "extra".to_string()]).unwrap_err();
        assert!(error.message.contains("expected columns"));
    }

    #[test]
    fn namespace_query_reorders_response_columns_by_name() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("_rowid", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![10, 20])),
                Arc::new(Int64Array::from(vec![100, 200])),
            ],
        )
        .unwrap();
        let mut bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, schema.as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let batches =
            ipc_bytes_to_batches(bytes, &["_rowid".to_string(), "id".to_string()]).unwrap();
        let batch = &batches[0];
        assert_eq!(batch.schema().field(0).name(), "_rowid");
        assert_eq!(batch.schema().field(1).name(), "id");
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[100, 200]
        );
    }
}
