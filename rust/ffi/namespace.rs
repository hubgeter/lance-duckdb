use std::collections::HashSet;
use std::ffi::{c_char, c_void, CString};
use std::future::Future;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

use arrow::ffi::FFI_ArrowSchema;
use lance::dataset::builder::DatasetBuilder;
use lance_core::Error as LanceError;

use lance_namespace::models::{
    DeclareTableRequest, DescribeTableRequest, DropTableRequest, ListTablesRequest,
};
use lance_namespace::schema::convert_json_arrow_schema;
use lance_namespace::{LanceNamespace, NamespaceError};
use lance_namespace_impls::RestNamespaceBuilder;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::session::{record_dataset_open, record_namespace_describe};
use super::types::DatasetHandle;
use super::util::{
    cstr_to_str, ffi_output_string, join_ffi_lines, optional_session_handle, parse_headers_tsv,
    schema_to_ffi_arrow_schema, to_c_string, FfiError, FfiResult,
};

const REST_READ_ATTEMPTS: usize = 3;
const REST_CALL_TIMEOUT: Duration = Duration::from_secs(30);

fn is_retryable_rest_read_error(error: &LanceError) -> bool {
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

fn is_rest_mutation_outcome_unknown_error(error: &LanceError) -> bool {
    match error {
        LanceError::Timeout { .. }
        | LanceError::IO { .. }
        | LanceError::Internal { .. }
        | LanceError::PrerequisiteFailed { .. }
        | LanceError::Wrapped { .. }
        | LanceError::Cloned { .. }
        | LanceError::Cleanup { .. }
        | LanceError::External { .. }
        | LanceError::Fenced { .. } => true,
        LanceError::Namespace { source, .. } => {
            source.downcast_ref::<NamespaceError>().is_none_or(|error| {
                matches!(
                    error,
                    NamespaceError::ServiceUnavailable { .. }
                        | NamespaceError::Internal { .. }
                        | NamespaceError::Throttling { .. }
                )
            })
        }
        _ => false,
    }
}

fn rest_mutation_error(code: ErrorCode, operation: &str, error: LanceError) -> FfiError {
    if is_rest_mutation_outcome_unknown_error(&error) {
        FfiError::new(
            ErrorCode::NamespaceMutationOutcomeUnknown,
            format!("namespace {operation}: {error}; outcome is unknown"),
        )
    } else {
        FfiError::new(code, format!("namespace {operation}: {error}"))
    }
}

async fn execute_rest_call<T, F, Fut>(
    code: ErrorCode,
    operation: &'static str,
    attempts: usize,
    mut call: F,
) -> FfiResult<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LanceError>>,
{
    debug_assert!(attempts > 0);
    let mut last_error = String::new();
    for attempt in 1..=attempts {
        match tokio::time::timeout(REST_CALL_TIMEOUT, call()).await {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(error)) => {
                last_error = format!("namespace {operation}: {error}");
                if !is_retryable_rest_read_error(&error) {
                    return Err(FfiError::new(code, last_error));
                }
            }
            Err(_) => {
                last_error = format!(
                    "namespace {operation} timed out after {} seconds",
                    REST_CALL_TIMEOUT.as_secs()
                );
            }
        }
        if attempt < attempts {
            log::warn!(
                "{last_error}; retrying idempotent REST request ({}/{})",
                attempt + 1,
                attempts
            );
            tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
        }
    }
    Err(FfiError::new(code, last_error))
}

async fn execute_rest_mutation_once<T, Fut>(
    code: ErrorCode,
    operation: &'static str,
    call: Fut,
) -> FfiResult<T>
where
    Fut: Future<Output = Result<T, LanceError>>,
{
    execute_rest_mutation_once_with_timeout(code, operation, REST_CALL_TIMEOUT, call).await
}

async fn execute_rest_mutation_once_with_timeout<T, Fut>(
    code: ErrorCode,
    operation: &'static str,
    timeout: Duration,
    call: Fut,
) -> FfiResult<T>
where
    Fut: Future<Output = Result<T, LanceError>>,
{
    match tokio::time::timeout(timeout, call).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(rest_mutation_error(code, operation, error)),
        Err(_) => Err(FfiError::new(
            ErrorCode::NamespaceMutationOutcomeUnknown,
            format!(
                "namespace {operation} timed out after {} seconds; outcome is unknown",
                timeout.as_secs_f64()
            ),
        )),
    }
}

fn is_missing_table(error: &LanceError) -> bool {
    match error {
        LanceError::DatasetNotFound { .. } | LanceError::NotFound { .. } => true,
        LanceError::Namespace { source, .. } => source
            .downcast_ref::<NamespaceError>()
            .is_some_and(|error| matches!(error, NamespaceError::TableNotFound { .. })),
        _ => error.is_not_found(),
    }
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
        return Ok(None);
    }
    Ok(Some(s.to_string()))
}

fn build_config(
    endpoint: &str,
    bearer_token: Option<&str>,
    api_key: Option<&str>,
    headers_tsv: Option<&str>,
) -> FfiResult<RestNamespaceBuilder> {
    if endpoint.trim().is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace endpoint must not be empty",
        ));
    }
    let mut builder = RestNamespaceBuilder::new(endpoint);
    if let Some(token) = bearer_token {
        builder = builder.header("Authorization", format!("Bearer {token}"));
    }
    if let Some(key) = api_key {
        builder = builder.header("x-api-key", key.to_string());
    }
    // Add custom headers from TSV
    for (key, value) in parse_headers_tsv(headers_tsv)? {
        builder = builder.header(key, value);
    }
    Ok(builder)
}

fn storage_options_to_tsv(
    storage_options: std::collections::HashMap<String, String>,
    code: ErrorCode,
    operation: &str,
) -> FfiResult<String> {
    if storage_options.is_empty() {
        return Ok(String::new());
    }
    let mut items: Vec<(String, String)> = storage_options.into_iter().collect();
    items.sort_by(|(a, _), (b, _)| a.cmp(b));
    let mut lines = Vec::with_capacity(items.len());
    for (key, value) in items {
        if key.is_empty() {
            return Err(FfiError::new(
                code,
                format!("{operation}: storage option key must not be empty"),
            ));
        }
        for (label, component) in [("key", &key), ("value", &value)] {
            if component
                .bytes()
                .any(|byte| matches!(byte, b'\0' | b'\t' | b'\n' | b'\r'))
            {
                return Err(FfiError::new(
                    code,
                    format!("{operation}: storage option {label} contains an unsupported control character"),
                ));
            }
        }
        lines.push(format!("{key}\t{value}"));
    }
    Ok(lines.join("\n"))
}

fn namespace_id_segments(value: &str, delimiter: &str) -> Vec<String> {
    value
        .split(delimiter)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

fn table_id_segments(value: &str, delimiter: &str) -> FfiResult<Vec<String>> {
    let segments = namespace_id_segments(value, delimiter);
    if segments.is_empty() {
        Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace table id must contain at least one non-empty segment",
        ))
    } else {
        Ok(segments)
    }
}

unsafe fn parse_namespace_delimiter(delimiter: *const c_char) -> FfiResult<String> {
    if delimiter.is_null() {
        return Ok("$".to_string());
    }
    let delimiter = unsafe { cstr_to_str(delimiter, "delimiter")? }.to_string();
    if delimiter.is_empty() {
        Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace delimiter must not be empty",
        ))
    } else {
        Ok(delimiter)
    }
}

fn list_tables_inner(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<Vec<String>> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let namespace_id = unsafe { cstr_to_str(namespace_id, "namespace_id")? };
    let delimiter = unsafe { parse_namespace_delimiter(delimiter)? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let namespace_segments = namespace_id_segments(namespace_id, &delimiter);
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )?
    .delimiter(delimiter)
    .build();

    let tables = runtime::block_on(async move {
        let mut out = Vec::new();
        let mut page_token: Option<String> = None;
        let mut seen_page_tokens = HashSet::new();
        loop {
            let mut req = ListTablesRequest::new();
            req.id = Some(namespace_segments.clone());
            req.page_token = page_token.clone();
            req.limit = Some(1000);
            let resp = execute_rest_call(
                ErrorCode::NamespaceListTables,
                "list_tables",
                REST_READ_ATTEMPTS,
                || namespace.list_tables(req.clone()),
            )
            .await?;
            out.extend(resp.tables);
            match resp.page_token {
                Some(token) if !token.is_empty() => {
                    if !seen_page_tokens.insert(token.clone()) {
                        return Err(FfiError::new(
                            ErrorCode::NamespaceListTables,
                            "namespace list_tables returned a repeated page token",
                        ));
                    }
                    page_token = Some(token);
                }
                _ => break,
            }
        }
        Ok::<_, FfiError>(out)
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok(tables)
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_namespace_list_tables(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> *const c_char {
    match list_tables_inner(
        endpoint,
        namespace_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok(tables) => {
            match join_ffi_lines(
                &tables,
                ErrorCode::NamespaceListTables,
                "namespace list_tables response",
            ) {
                Ok(joined) => {
                    clear_last_error();
                    to_c_string(joined).into_raw() as *const c_char
                }
                Err(err) => {
                    set_last_error(err.code, err.message);
                    ptr::null()
                }
            }
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null()
        }
    }
}

fn describe_table_info_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<(CString, CString)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { parse_namespace_delimiter(delimiter)? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )?
    .delimiter(delimiter.clone())
    .build();

    let (location, storage_options_tsv) = runtime::block_on(async move {
        record_namespace_describe();
        let mut req = DescribeTableRequest::new();
        // FIX: a qualified table id (e.g. "catalog.schema.table") must be sent as
        // its multi-segment namespace path, not a single segment. Split on the
        // delimiter so the server sees the full 3-level id instead of "got: 1".
        req.id = Some(table_id_segments(table_id, &delimiter)?);
        req.with_table_uri = Some(true);
        let resp = execute_rest_call(
            ErrorCode::NamespaceDescribeTableInfo,
            "describe_table",
            REST_READ_ATTEMPTS,
            || namespace.describe_table(req.clone()),
        )
        .await?;
        let location = resp.table_uri.or(resp.location).ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTableInfo,
                "namespace describe_table: missing location and table_uri",
            )
        })?;
        if location.is_empty() || location.contains('\0') {
            return Err(FfiError::new(
                ErrorCode::NamespaceDescribeTableInfo,
                "namespace describe_table: location is empty or contains a NUL byte",
            ));
        }
        let storage_options_tsv = storage_options_to_tsv(
            resp.storage_options.unwrap_or_default(),
            ErrorCode::NamespaceDescribeTableInfo,
            "namespace describe_table",
        )?;
        Ok::<_, FfiError>((location, storage_options_tsv))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok((
        ffi_output_string(
            location,
            ErrorCode::NamespaceDescribeTableInfo,
            "namespace table location",
        )?,
        ffi_output_string(
            storage_options_tsv,
            ErrorCode::NamespaceDescribeTableInfo,
            "namespace storage options",
        )?,
    ))
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_namespace_describe_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_location: *mut *const c_char,
    out_storage_options_tsv: *mut *const c_char,
) -> i32 {
    if super::util::output_regions_overlap(out_location, out_storage_options_tsv) {
        set_last_error(
            ErrorCode::InvalidArgument,
            "out_location and out_storage_options_tsv must not alias",
        );
        return -1;
    }
    if !out_location.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_location, ptr::null());
        }
    }
    if !out_storage_options_tsv.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_storage_options_tsv, ptr::null());
        }
    }

    match describe_table_info_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok((location, storage_options_tsv)) => {
            clear_last_error();
            if !out_location.is_null() {
                unsafe {
                    std::ptr::write_unaligned(out_location, location.into_raw() as *const c_char);
                }
            }
            if !out_storage_options_tsv.is_null() {
                unsafe {
                    std::ptr::write_unaligned(
                        out_storage_options_tsv,
                        storage_options_tsv.into_raw() as *const c_char,
                    );
                }
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn create_empty_table_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<(CString, CString)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { parse_namespace_delimiter(delimiter)? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )?
    .delimiter(delimiter.clone())
    .build();

    let table_id_segments = table_id_segments(table_id, &delimiter)?;
    let (location, storage_options_tsv) = runtime::block_on(async move {
        let mut req = DeclareTableRequest::new();
        req.id = Some(table_id_segments);
        // Declare is not safe to retry blindly: a timeout can mean the table
        // was created but the response was lost. Bound it, but execute once.
        let resp = execute_rest_mutation_once(
            ErrorCode::NamespaceCreateEmptyTable,
            "declare_table",
            namespace.declare_table(req),
        )
        .await?;
        let location = resp.location.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceMutationOutcomeUnknown,
                "namespace declare_table succeeded but returned no location; outcome is unknown",
            )
        })?;
        if location.is_empty() || location.contains('\0') {
            return Err(FfiError::new(
                ErrorCode::NamespaceMutationOutcomeUnknown,
                "namespace declare_table succeeded but returned an unusable location; outcome is unknown",
            ));
        }
        let storage_options_tsv = storage_options_to_tsv(
            resp.storage_options.unwrap_or_default(),
            ErrorCode::NamespaceMutationOutcomeUnknown,
            "namespace declare_table succeeded but returned unusable storage options; outcome is unknown",
        )?;
        Ok::<_, FfiError>((location, storage_options_tsv))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok((
        ffi_output_string(
            location,
            ErrorCode::NamespaceMutationOutcomeUnknown,
            "declared namespace table location",
        )?,
        ffi_output_string(
            storage_options_tsv,
            ErrorCode::NamespaceMutationOutcomeUnknown,
            "declared namespace table storage options",
        )?,
    ))
}

#[ffi_guard_macro::ffi_guard(namespace_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_namespace_create_empty_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_location: *mut *const c_char,
    out_storage_options_tsv: *mut *const c_char,
) -> i32 {
    // Validate required outputs before the non-idempotent declaration. Once
    // the server accepts it, returning without the location/credentials would
    // strand a table that the caller cannot safely finish or retry.
    if out_location.is_null() || out_storage_options_tsv.is_null() {
        set_last_error(
            ErrorCode::InvalidArgument,
            "out_location/out_storage_options_tsv is null",
        );
        return -1;
    }
    if super::util::output_regions_overlap(out_location, out_storage_options_tsv) {
        set_last_error(
            ErrorCode::InvalidArgument,
            "out_location and out_storage_options_tsv must not alias",
        );
        return -1;
    }
    unsafe {
        std::ptr::write_unaligned(out_location, ptr::null());
        std::ptr::write_unaligned(out_storage_options_tsv, ptr::null());
    }

    match create_empty_table_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok((location, storage_options_tsv)) => {
            clear_last_error();
            unsafe {
                std::ptr::write_unaligned(out_location, location.into_raw() as *const c_char);
            }
            unsafe {
                std::ptr::write_unaligned(
                    out_storage_options_tsv,
                    storage_options_tsv.into_raw() as *const c_char,
                );
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn drop_table_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<()> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { parse_namespace_delimiter(delimiter)? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )?
    .delimiter(delimiter.clone())
    .build();

    let table_id_segments = table_id_segments(table_id, &delimiter)?;
    runtime::block_on(async move {
        let mut req = DropTableRequest::new();
        req.id = Some(table_id_segments);
        // Do not retry a timed-out drop. A prior attempt may have succeeded,
        // and an external client may have recreated the same table name before
        // a retry, which would make a second DELETE target different data.
        match tokio::time::timeout(REST_CALL_TIMEOUT, namespace.drop_table(req)).await {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) if is_missing_table(&error) => Ok(()),
            Ok(Err(error)) => Err(rest_mutation_error(
                ErrorCode::NamespaceDropTable,
                &format!("drop_table '{table_id}'"),
                error,
            )),
            Err(_) => Err(FfiError::new(
                ErrorCode::NamespaceMutationOutcomeUnknown,
                format!(
                    "namespace drop_table '{table_id}' timed out after {} seconds; outcome is unknown",
                    REST_CALL_TIMEOUT.as_secs()
                ),
            )),
        }
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?
}

#[ffi_guard_macro::ffi_guard(namespace_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_namespace_drop_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> i32 {
    match drop_table_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn namespace_table_version_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<u64> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { parse_namespace_delimiter(delimiter)? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )?
    .delimiter(delimiter.clone())
    .build();

    runtime::block_on(async move {
        record_namespace_describe();
        let mut req = DescribeTableRequest::new();
        req.id = Some(table_id_segments(table_id, &delimiter)?);
        req.load_detailed_metadata = Some(true);
        let resp = execute_rest_call(
            ErrorCode::NamespaceDescribeTable,
            "describe_table version",
            REST_READ_ATTEMPTS,
            || namespace.describe_table(req.clone()),
        )
        .await?;
        let version = resp.version.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                "namespace describe_table version: missing version",
            )
        })?;
        u64::try_from(version).map_err(|_| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                format!("namespace describe_table version: invalid version {version}"),
            )
        })
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_namespace_get_table_version(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> u64 {
    match namespace_table_version_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok(version) if version > 0 => {
            clear_last_error();
            version
        }
        Ok(_) => {
            set_last_error(
                ErrorCode::NamespaceDescribeTable,
                "namespace describe_table version: version must be greater than zero",
            );
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            0
        }
    }
}

/// Describe a table with `load_detailed_metadata=true` and return the schema
/// as a JSON string. This avoids opening the dataset from S3.
fn describe_table_with_schema_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<CString> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { parse_namespace_delimiter(delimiter)? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )?
    .delimiter(delimiter.clone())
    .build();

    let schema_json = runtime::block_on(async move {
        let mut req = DescribeTableRequest::new();
        // FIX: split the qualified id into its namespace segments (see describe_table_info_inner).
        req.id = Some(table_id_segments(table_id, &delimiter)?);
        req.with_table_uri = Some(true);
        req.load_detailed_metadata = Some(true);
        let resp = execute_rest_call(
            ErrorCode::NamespaceDescribeTable,
            "describe_table schema",
            REST_READ_ATTEMPTS,
            || namespace.describe_table(req.clone()),
        )
        .await?;

        let schema = resp.schema.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                "namespace describe_table: missing schema in response",
            )
        })?;

        serde_json::to_string(&schema).map_err(|err| {
            FfiError::new(
                ErrorCode::SchemaExport,
                format!("failed to serialize schema: {err}"),
            )
        })
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    ffi_output_string(
        schema_json,
        ErrorCode::SchemaExport,
        "namespace table schema JSON",
    )
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_namespace_describe_table_with_schema(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_schema_json: *mut *const c_char,
) -> i32 {
    if out_schema_json.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "out_schema_json is null");
        return -1;
    }
    if !out_schema_json.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_schema_json, ptr::null());
        }
    }

    match describe_table_with_schema_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok(schema_json) => {
            clear_last_error();
            unsafe {
                std::ptr::write_unaligned(out_schema_json, schema_json.into_raw() as *const c_char);
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn open_dataset_in_namespace_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    session: *mut c_void,
) -> FfiResult<(DatasetHandle, CString)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { parse_namespace_delimiter(delimiter)? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )?
    .delimiter(delimiter.clone())
    .build();
    let session = unsafe { optional_session_handle(session)? };
    // FIX: split the qualified id into namespace segments so the crate's internal
    // describe (DatasetBuilder::from_namespace) gets the full 3-level id, not 1.
    let table_id_segments = table_id_segments(table_id, &delimiter)?;

    let (dataset, table_uri) = runtime::block_on(async move {
        record_namespace_describe();
        let namespace = Arc::new(namespace);
        let mut builder = execute_rest_call(
            ErrorCode::NamespaceDescribeTable,
            "describe_table for dataset open",
            REST_READ_ATTEMPTS,
            || DatasetBuilder::from_namespace(namespace.clone(), table_id_segments.clone()),
        )
        .await?;
        if let Some(session) = session {
            builder = builder.with_session(session);
        }
        let dataset = builder.load().await.map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetOpen,
                format!("namespace dataset open: {err}"),
            )
        })?;
        let table_uri = dataset.uri().to_string();
        Ok::<_, FfiError>((Arc::new(dataset), table_uri))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    let table_uri = ffi_output_string(table_uri, ErrorCode::DatasetOpen, "dataset URI")?;
    record_dataset_open();
    Ok((DatasetHandle::new(dataset), table_uri))
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_in_namespace(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_table_uri: *mut *const c_char,
) -> *mut c_void {
    if !out_table_uri.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_table_uri, ptr::null());
        }
    }

    match open_dataset_in_namespace_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
        ptr::null_mut(),
    ) {
        Ok((handle, table_uri)) => {
            clear_last_error();
            if !out_table_uri.is_null() {
                unsafe {
                    std::ptr::write_unaligned(out_table_uri, table_uri.into_raw() as *const c_char);
                }
            }
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_in_namespace_with_session(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    session: *mut c_void,
    out_table_uri: *mut *const c_char,
) -> *mut c_void {
    if !out_table_uri.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_table_uri, ptr::null());
        }
    }

    match open_dataset_in_namespace_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
        session,
    ) {
        Ok((handle, table_uri)) => {
            clear_last_error();
            if !out_table_uri.is_null() {
                unsafe {
                    std::ptr::write_unaligned(out_table_uri, table_uri.into_raw() as *const c_char);
                }
            }
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

/// Convert a JSON Arrow schema string to Arrow C Data Interface ArrowSchema.
#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_json_arrow_schema_to_c(
    json_schema: *const c_char,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    let result = (|| -> FfiResult<()> {
        if out_schema.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "out_schema is null",
            ));
        }
        let json_str = unsafe { cstr_to_str(json_schema, "json_schema")? };
        let json_arrow: lance_namespace::models::JsonArrowSchema = serde_json::from_str(json_str)
            .map_err(|err| {
            FfiError::new(
                ErrorCode::SchemaExport,
                format!("failed to parse JSON arrow schema: {err}"),
            )
        })?;
        let arrow_schema = convert_json_arrow_schema(&json_arrow).map_err(|err| {
            FfiError::new(
                ErrorCode::SchemaExport,
                format!("failed to convert JSON arrow schema: {err}"),
            )
        })?;
        let ffi_schema = schema_to_ffi_arrow_schema(&arrow_schema)?;
        unsafe {
            std::ptr::write_unaligned(out_schema, ffi_schema);
        }
        Ok(())
    })();

    match result {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn namespace_ids_are_split_into_every_segment() {
        assert_eq!(
            namespace_id_segments("org$team$table", "$"),
            vec!["org", "team", "table"]
        );
        assert_eq!(
            namespace_id_segments("$org$$team$table$", "$"),
            vec!["org", "team", "table"]
        );
        assert!(namespace_id_segments("", "$").is_empty());
    }

    #[test]
    fn namespace_delimiter_must_not_be_empty() {
        let empty = CString::new("").unwrap();
        let error = unsafe { parse_namespace_delimiter(empty.as_ptr()) }.unwrap_err();
        assert_eq!(error.code as i32, ErrorCode::InvalidArgument as i32);
        assert_eq!(
            unsafe { parse_namespace_delimiter(ptr::null()) }.unwrap(),
            "$"
        );
    }

    #[test]
    fn namespace_endpoint_must_not_be_empty() {
        let error = match build_config("  ", None, None, None) {
            Ok(_) => panic!("an empty namespace endpoint unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.code as i32, ErrorCode::InvalidArgument as i32);
        assert!(error.message.contains("endpoint"));
    }

    #[test]
    fn namespace_table_id_must_not_resolve_to_root() {
        let error = table_id_segments("$$", "$").unwrap_err();
        assert_eq!(error.code as i32, ErrorCode::InvalidArgument as i32);
    }

    #[test]
    fn storage_option_transport_rejects_ambiguous_control_characters() {
        for options in [
            std::collections::HashMap::from([("".to_string(), "value".to_string())]),
            std::collections::HashMap::from([(
                "session_token".to_string(),
                "first\nsecond".to_string(),
            )]),
        ] {
            let error = storage_options_to_tsv(
                options,
                ErrorCode::NamespaceMutationOutcomeUnknown,
                "test mutation outcome is unknown",
            )
            .unwrap_err();

            assert_eq!(
                error.code as i32,
                ErrorCode::NamespaceMutationOutcomeUnknown as i32
            );
        }
    }

    #[test]
    fn idempotent_rest_call_retries_boundedly() {
        let calls = AtomicUsize::new(0);
        let value = runtime::block_on(execute_rest_call(
            ErrorCode::NamespaceDescribeTable,
            "test read",
            REST_READ_ATTEMPTS,
            || {
                let attempt = calls.fetch_add(1, Ordering::Relaxed) + 1;
                async move {
                    if attempt < REST_READ_ATTEMPTS {
                        Err(NamespaceError::ServiceUnavailable {
                            message: "temporary test failure".to_string(),
                        }
                        .into())
                    } else {
                        Ok(42_u64)
                    }
                }
            },
        ))
        .unwrap()
        .unwrap();

        assert_eq!(value, 42);
        assert_eq!(calls.load(Ordering::Relaxed), REST_READ_ATTEMPTS);
    }

    #[test]
    fn idempotent_rest_call_does_not_retry_deterministic_errors() {
        let calls = AtomicUsize::new(0);
        let error = runtime::block_on(execute_rest_call(
            ErrorCode::NamespaceDescribeTable,
            "test read",
            REST_READ_ATTEMPTS,
            || {
                calls.fetch_add(1, Ordering::Relaxed);
                async {
                    Err::<(), LanceError>(
                        NamespaceError::InvalidInput {
                            message: "invalid test request".to_string(),
                        }
                        .into(),
                    )
                }
            },
        ))
        .unwrap()
        .unwrap_err();

        assert!(error.message.contains("invalid test request"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn non_idempotent_rest_call_is_executed_once_and_reports_ambiguity() {
        let calls = AtomicUsize::new(0);
        let call = async {
            calls.fetch_add(1, Ordering::Relaxed);
            Err::<(), _>(LanceError::internal("ambiguous test failure"))
        };
        let error = runtime::block_on(execute_rest_mutation_once(
            ErrorCode::NamespaceCreateEmptyTable,
            "test mutation",
            call,
        ))
        .unwrap()
        .unwrap_err();

        assert!(error.message.contains("ambiguous test failure"));
        assert_eq!(
            error.code as i32,
            ErrorCode::NamespaceMutationOutcomeUnknown as i32
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn non_idempotent_rest_call_preserves_a_definitive_validation_error() {
        let calls = AtomicUsize::new(0);
        let call = async {
            calls.fetch_add(1, Ordering::Relaxed);
            Err::<(), _>(LanceError::invalid_input("invalid test request"))
        };
        let error = runtime::block_on(execute_rest_mutation_once(
            ErrorCode::NamespaceCreateEmptyTable,
            "test mutation",
            call,
        ))
        .unwrap()
        .unwrap_err();

        assert_eq!(
            error.code as i32,
            ErrorCode::NamespaceCreateEmptyTable as i32
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn timed_out_mutation_has_a_machine_readable_unknown_outcome_code() {
        let call = async {
            std::future::pending::<()>().await;
            Ok::<(), LanceError>(())
        };
        let error = runtime::block_on(execute_rest_mutation_once_with_timeout(
            ErrorCode::NamespaceCreateEmptyTable,
            "test mutation",
            Duration::from_millis(1),
            call,
        ))
        .unwrap()
        .unwrap_err();

        assert_eq!(
            error.code as i32,
            ErrorCode::NamespaceMutationOutcomeUnknown as i32
        );
        assert!(error.message.contains("outcome is unknown"));
    }

    #[test]
    fn disconnected_mutation_has_a_machine_readable_unknown_outcome_code() {
        let call = async {
            Err::<(), LanceError>(
                NamespaceError::ServiceUnavailable {
                    message: "connection closed before the response".to_string(),
                }
                .into(),
            )
        };
        let error = runtime::block_on(execute_rest_mutation_once(
            ErrorCode::NamespaceCreateEmptyTable,
            "test mutation",
            call,
        ))
        .unwrap()
        .unwrap_err();

        assert_eq!(
            error.code as i32,
            ErrorCode::NamespaceMutationOutcomeUnknown as i32
        );
        assert!(error.message.contains("outcome is unknown"));
    }

    #[test]
    fn namespace_table_not_found_is_recognized_as_missing() {
        let error: LanceError = NamespaceError::TableNotFound {
            message: "missing".to_string(),
        }
        .into();
        assert!(is_missing_table(&error));
    }

    #[test]
    fn json_schema_export_rejects_null_output() {
        let rc = unsafe { lance_json_arrow_schema_to_c(ptr::null(), ptr::null_mut()) };
        assert_eq!(rc, -1);
        assert_eq!(
            crate::error::lance_last_error_code(),
            ErrorCode::InvalidArgument as i32
        );
    }

    #[test]
    fn table_declaration_rejects_null_outputs_before_the_request() {
        let rc = unsafe {
            lance_namespace_create_empty_table(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(rc, -1);
        assert_eq!(
            crate::error::lance_last_error_code(),
            ErrorCode::InvalidArgument as i32
        );
        let message = crate::error::lance_last_error_message();
        assert!(!message.is_null());
        let text = unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        unsafe { crate::error::lance_free_string(message) };
        assert!(text.contains("out_location/out_storage_options_tsv"));
    }

    #[test]
    fn table_declaration_rejects_aliased_outputs_before_the_request() {
        let mut output = ptr::null();
        let output_ptr = &mut output as *mut *const c_char;
        let rc = unsafe {
            lance_namespace_create_empty_table(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                output_ptr,
                output_ptr,
            )
        };
        assert_eq!(rc, -1);
        assert_eq!(
            crate::error::lance_last_error_code(),
            ErrorCode::InvalidArgument as i32
        );
        let message = crate::error::lance_last_error_message();
        assert!(!message.is_null());
        let text = unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        unsafe { crate::error::lance_free_string(message) };
        assert!(text.contains("must not alias"));
    }
}
