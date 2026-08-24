use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;
use arrow::compute;
use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use chrono::{DateTime, Duration, Utc};
use futures::TryStreamExt;
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::optimize::{compact_files, CompactionOptions};
use lance::dataset::{BatchUDF, ColumnAlteration, NewColumnTransform};
use lance::index::DatasetIndexExt;
use lance::Dataset;
use lance_index::scalar::ScalarIndexParams;
use lance_index::IndexType;
use serde::{Deserialize, Serialize};

use super::util::{
    canonicalize_lance_field_path, cstr_to_str, lance_mutation_error, optional_cstr_array,
    to_c_string, FfiError, FfiResult,
};

fn parse_batch_size_from_config(dataset: &Dataset) -> Option<u32> {
    dataset
        .config()
        .get("lance.add_columns.batch_size")
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v > 0)
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CompactFilesOptionsInput {
    target_rows_per_fragment: Option<usize>,
    max_rows_per_group: Option<usize>,
    max_bytes_per_file: Option<usize>,
    materialize_deletions: Option<bool>,
    materialize_deletions_threshold: Option<f32>,
    num_threads: Option<usize>,
    batch_size: Option<usize>,
    defer_index_remap: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CleanupOldVersionsOptionsInput {
    older_than_seconds: i64,
    delete_unverified: bool,
    error_if_tagged_old_versions: bool,
    retain_n_versions: Option<u64>,
}

#[derive(Debug, Serialize)]
struct CleanupOldVersionsMetricsOutput {
    bytes_removed: u64,
    old_versions: u64,
}

pub(super) async fn cleanup_vane_staging_files(
    dataset: &Dataset,
    cutoff: DateTime<Utc>,
) -> Result<(), lance::Error> {
    // Durable operation markers are the source of truth for idempotent replay
    // after Lance has vacuumed the transaction that originally recorded the
    // operation.  VACUUM must therefore only remove abandoned staging files.
    // Match Lance's safety window for those unverified files so a cleanup
    // running without the caller's coordination cannot erase an active write.
    // An explicitly unsafe cleanup may use the requested threshold directly.
    let store = dataset.object_store(None).await?;
    let base = dataset.branch_location().path;

    let prefix = if base.as_ref().is_empty() {
        object_store::path::Path::parse("_vane_staging")?
    } else {
        object_store::path::Path::parse(format!("{}/_vane_staging", base.as_ref()))?
    };
    let objects = match store.list(Some(prefix)).try_collect::<Vec<_>>().await {
        Ok(objects) => objects,
        Err(error) if error.is_not_found() => return Ok(()),
        Err(error) => return Err(error),
    };
    for object in objects {
        if object.last_modified < cutoff {
            store.delete(&object.location).await?;
        }
    }
    Ok(())
}

fn checked_cleanup_cutoff(
    now: DateTime<Utc>,
    older_than_seconds: i64,
    label: &str,
) -> FfiResult<DateTime<Utc>> {
    let retention = Duration::try_seconds(older_than_seconds).ok_or_else(|| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{label} is too large to represent"),
        )
    })?;
    now.checked_sub_signed(retention).ok_or_else(|| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{label} produces a timestamp outside the supported range"),
        )
    })
}

impl Default for CleanupOldVersionsOptionsInput {
    fn default() -> Self {
        Self {
            older_than_seconds: 0,
            delete_unverified: false,
            error_if_tagged_old_versions: true,
            retain_n_versions: None,
        }
    }
}

fn parse_compaction_options_json(options_json: *const c_char) -> FfiResult<CompactionOptions> {
    if options_json.is_null() {
        return Ok(CompactionOptions::default());
    }
    let text = unsafe { CStr::from_ptr(options_json) }
        .to_str()
        .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("options_json utf8: {err}")))?;
    if text.trim().is_empty() {
        return Ok(CompactionOptions::default());
    }

    let input: CompactFilesOptionsInput = serde_json::from_str(text).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("compact_files options_json parse: {err}"),
        )
    })?;

    for (name, value) in [
        ("target_rows_per_fragment", input.target_rows_per_fragment),
        ("max_rows_per_group", input.max_rows_per_group),
        ("max_bytes_per_file", input.max_bytes_per_file),
        ("num_threads", input.num_threads),
        ("batch_size", input.batch_size),
    ] {
        if value == Some(0) {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("compact_files option '{name}' must be greater than zero"),
            ));
        }
    }
    if input
        .materialize_deletions_threshold
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "compact_files option 'materialize_deletions_threshold' must be finite and non-negative",
        ));
    }

    let mut options = CompactionOptions::default();
    if let Some(v) = input.target_rows_per_fragment {
        options.target_rows_per_fragment = v;
    }
    if let Some(v) = input.max_rows_per_group {
        options.max_rows_per_group = v;
    }
    if let Some(v) = input.max_bytes_per_file {
        options.max_bytes_per_file = Some(v);
    }
    if let Some(v) = input.materialize_deletions {
        options.materialize_deletions = v;
    }
    if let Some(v) = input.materialize_deletions_threshold {
        options.materialize_deletions_threshold = v;
    }
    if let Some(v) = input.num_threads {
        options.num_threads = Some(v);
    }
    if let Some(v) = input.batch_size {
        options.batch_size = Some(v);
    }
    if let Some(v) = input.defer_index_remap {
        options.defer_index_remap = v;
    }
    options.validate();
    Ok(options)
}

fn parse_cleanup_options_json(
    options_json: *const c_char,
) -> FfiResult<CleanupOldVersionsOptionsInput> {
    if options_json.is_null() {
        return Ok(CleanupOldVersionsOptionsInput::default());
    }
    let text = unsafe { CStr::from_ptr(options_json) }
        .to_str()
        .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("options_json utf8: {err}")))?;
    if text.trim().is_empty() {
        return Ok(CleanupOldVersionsOptionsInput::default());
    }
    serde_json::from_str(text).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("cleanup_old_versions options_json parse: {err}"),
        )
    })
}

fn write_metrics_json<T: Serialize>(
    value: &T,
    out_metrics_json: *mut *const c_char,
    context: &'static str,
    code: ErrorCode,
) -> FfiResult<()> {
    if out_metrics_json.is_null() {
        return Ok(());
    }
    let payload = serde_json::to_string(value)
        .map_err(|err| FfiError::new(code, format!("{context} serialize: {err}")))?;
    unsafe {
        ptr::write_unaligned(
            out_metrics_json,
            to_c_string(payload).into_raw() as *const c_char,
        );
    }
    Ok(())
}

fn parse_arrow_schema(schema: *const c_void, what: &'static str) -> FfiResult<Arc<ArrowSchema>> {
    if schema.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null"),
        ));
    }

    let ffi_schema = unsafe { &*(schema as *const arrow_schema::ffi::FFI_ArrowSchema) };
    let schema = ArrowSchema::try_from(ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::InvalidArgument, format!("{what} import: {err}"))
    })?;
    Ok(Arc::new(schema))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_schema::ffi::FFI_ArrowSchema;
    use arrow_schema::{DataType, Field};

    use super::*;

    #[test]
    fn rejects_unrepresentable_cleanup_interval() {
        let error = checked_cleanup_cutoff(Utc::now(), i64::MAX, "older_than_seconds")
            .expect_err("an i64::MAX-second chrono duration is not representable");
        assert_eq!(error.code as i32, ErrorCode::InvalidArgument as i32);
        assert!(error.message.contains("too large"));
    }

    #[test]
    fn arrow_schema_import_preserves_schema_metadata() {
        let expected = ArrowSchema::new_with_metadata(
            vec![Field::new("value", DataType::Int64, false)],
            HashMap::from([("owner".to_string(), "vane".to_string())]),
        );
        let ffi = FFI_ArrowSchema::try_from(&expected).unwrap();
        let actual = parse_arrow_schema(
            &ffi as *const FFI_ArrowSchema as *const c_void,
            "test_schema",
        )
        .unwrap();
        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    fn rejects_unsafe_zero_compaction_options() {
        for options in [
            r#"{"target_rows_per_fragment":0}"#,
            r#"{"max_rows_per_group":0}"#,
            r#"{"max_bytes_per_file":0}"#,
            r#"{"num_threads":0}"#,
            r#"{"batch_size":0}"#,
            r#"{"materialize_deletions_threshold":-0.1}"#,
        ] {
            let options = std::ffi::CString::new(options).unwrap();
            let error = parse_compaction_options_json(options.as_ptr())
                .expect_err("unsafe compaction option must fail at the FFI boundary");
            assert_eq!(error.code as i32, ErrorCode::InvalidArgument as i32);
        }
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_add_columns(
    dataset: *mut c_void,
    new_columns_schema: *const c_void,
    expressions: *const *const c_char,
    expressions_len: usize,
    batch_size: u32,
) -> i32 {
    match dataset_add_columns_inner(
        dataset,
        new_columns_schema,
        expressions,
        expressions_len,
        batch_size,
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

fn dataset_add_columns_inner(
    dataset: *mut c_void,
    new_columns_schema: *const c_void,
    expressions: *const *const c_char,
    expressions_len: usize,
    batch_size: u32,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let mut ds = (*handle.dataset).clone();

    let output_schema = parse_arrow_schema(new_columns_schema, "new_columns_schema")?;
    if output_schema.fields().is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "new_columns_schema must have at least one field",
        ));
    }

    let batch_size = if batch_size == 0 {
        parse_batch_size_from_config(&ds)
    } else {
        Some(batch_size)
    };

    if expressions_len == 0 {
        let transforms = NewColumnTransform::AllNulls(output_schema);
        match runtime::block_on(ds.add_columns(transforms, None, batch_size)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(lance_mutation_error(
                ErrorCode::DatasetAddColumns,
                ErrorCode::DatasetCommitOutcomeUnknown,
                "dataset add_columns(all_nulls)",
                err,
            )),
            Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        }
    } else {
        let exprs = unsafe { optional_cstr_array(expressions, expressions_len, "expressions")? };
        if exprs.len() != output_schema.fields().len() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "expressions_len must match new_columns_schema field count",
            ));
        }

        let full_read_schema = Arc::new(ArrowSchema::from(ds.schema()));
        let planner = lance::io::exec::Planner::new(full_read_schema);

        let parsed = exprs
            .iter()
            .enumerate()
            .map(|(idx, expr)| {
                let expr = planner.parse_expr(expr).map_err(|err| {
                    FfiError::new(
                        ErrorCode::DatasetAddColumns,
                        format!("expression[{idx}] parse: {err}"),
                    )
                })?;
                let expr = planner.optimize_expr(expr).map_err(|err| {
                    FfiError::new(
                        ErrorCode::DatasetAddColumns,
                        format!("expression[{idx}] optimize: {err}"),
                    )
                })?;
                Ok(expr)
            })
            .collect::<FfiResult<Vec<_>>>()?;

        let mut needed = HashSet::<String>::new();
        for expr in parsed.iter() {
            for col in lance::io::exec::Planner::column_names_in_expr(expr) {
                needed.insert(col);
            }
        }
        let needed_columns = needed.into_iter().collect::<Vec<_>>();
        let read_schema = ds.schema().project(&needed_columns).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetAddColumns,
                format!("read schema project: {err}"),
            )
        })?;
        let read_schema = Arc::new(ArrowSchema::from(&read_schema));
        let planner = lance::io::exec::Planner::new(read_schema.clone());

        let physical = parsed
            .into_iter()
            .enumerate()
            .map(|(idx, expr)| {
                planner.create_physical_expr(&expr).map_err(|err| {
                    FfiError::new(
                        ErrorCode::DatasetAddColumns,
                        format!("expression[{idx}] physical: {err}"),
                    )
                })
            })
            .collect::<FfiResult<Vec<_>>>()?;

        // This FFI is used by DuckDB ALTER TABLE ADD COLUMN, where every SQL
        // expression is also the persisted default expression.  Put that
        // metadata into the new field before the single add-columns commit;
        // a follow-up metadata commit would make ADD COLUMN partially atomic.
        let output_schema = Arc::new(ArrowSchema::new_with_metadata(
            output_schema
                .fields()
                .iter()
                .zip(exprs.iter())
                .map(|(field, expression)| {
                    let mut metadata = field.metadata().clone();
                    metadata.insert("duckdb_default_expr".to_string(), expression.clone());
                    Arc::new(field.as_ref().clone().with_metadata(metadata))
                })
                .collect::<Vec<_>>(),
            output_schema.metadata().clone(),
        ));

        let output_fields = output_schema.fields().to_vec();
        let schema_ref = output_schema.clone();
        let mapper = move |batch: &RecordBatch| {
            let num_rows = batch.num_rows();
            let mut arrays = Vec::with_capacity(physical.len());
            for (idx, (field, expr)) in output_fields.iter().zip(physical.iter()).enumerate() {
                let arr = expr
                    .evaluate(batch)
                    .map_err(|err| {
                        lance::Error::invalid_input(format!("expression[{idx}] evaluate: {err}"))
                    })?
                    .into_array(num_rows)
                    .map_err(|err| {
                        lance::Error::invalid_input(format!("expression[{idx}] into_array: {err}"))
                    })?;
                let arr = if arr.data_type() != field.data_type() {
                    compute::cast(&arr, field.data_type()).map_err(|err| {
                        lance::Error::invalid_input(format!("expression[{idx}] cast: {err}"))
                    })?
                } else {
                    arr
                };
                arrays.push(arr);
            }
            RecordBatch::try_new(schema_ref.clone(), arrays)
                .map_err(|err| lance::Error::invalid_input(format!("output batch: {err}")))
        };

        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: None,
        });

        let read_columns = Some(
            read_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>(),
        );
        match runtime::block_on(ds.add_columns(transforms, read_columns, batch_size)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(lance_mutation_error(
                ErrorCode::DatasetAddColumns,
                ErrorCode::DatasetCommitOutcomeUnknown,
                "dataset add_columns(sql)",
                err,
            )),
            Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        }
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_drop_columns(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
) -> i32 {
    match dataset_drop_columns_inner(dataset, columns, columns_len) {
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

fn dataset_drop_columns_inner(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
) -> FfiResult<()> {
    if columns_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "columns_len must be > 0",
        ));
    }
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let cols = unsafe { optional_cstr_array(columns, columns_len, "columns")? };

    let mut ds = (*handle.dataset).clone();
    let col_refs = cols.iter().map(|c| c.as_str()).collect::<Vec<_>>();
    match runtime::block_on(ds.drop_columns(&col_refs)) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetDropColumns,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset drop_columns",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_alter_columns_rename(
    dataset: *mut c_void,
    path: *const c_char,
    new_name: *const c_char,
) -> i32 {
    match dataset_alter_columns_rename_inner(dataset, path, new_name) {
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

fn dataset_alter_columns_rename_inner(
    dataset: *mut c_void,
    path: *const c_char,
    new_name: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let new_name = unsafe { cstr_to_str(new_name, "new_name")? }.to_string();

    let mut ds = (*handle.dataset).clone();
    let alteration = ColumnAlteration::new(path).rename(new_name);
    match runtime::block_on(ds.alter_columns(&[alteration])) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetAlterColumns,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset alter_columns(rename)",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_alter_columns_set_nullable(
    dataset: *mut c_void,
    path: *const c_char,
    nullable: u8,
) -> i32 {
    match dataset_alter_columns_set_nullable_inner(dataset, path, nullable != 0) {
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

fn dataset_alter_columns_set_nullable_inner(
    dataset: *mut c_void,
    path: *const c_char,
    nullable: bool,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();

    let mut ds = (*handle.dataset).clone();
    let alteration = ColumnAlteration::new(path).set_nullable(nullable);
    match runtime::block_on(ds.alter_columns(&[alteration])) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetAlterColumns,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset alter_columns(nullable)",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_alter_columns_cast(
    dataset: *mut c_void,
    path: *const c_char,
    new_type_schema: *const c_void,
) -> i32 {
    match dataset_alter_columns_cast_inner(dataset, path, new_type_schema) {
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

fn dataset_alter_columns_cast_inner(
    dataset: *mut c_void,
    path: *const c_char,
    new_type_schema: *const c_void,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();

    let type_schema = parse_arrow_schema(new_type_schema, "new_type_schema")?;
    if type_schema.fields().len() != 1 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "new_type_schema must have exactly one field",
        ));
    }
    let new_type = type_schema.fields()[0].data_type().clone();

    let mut ds = (*handle.dataset).clone();
    let alteration = ColumnAlteration::new(path).cast_to(new_type);
    match runtime::block_on(ds.alter_columns(&[alteration])) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetAlterColumns,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset alter_columns(cast)",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_table_metadata(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    match dataset_update_table_metadata_inner(dataset, key, value) {
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

fn dataset_update_table_metadata_inner(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let key = unsafe { cstr_to_str(key, "key")? }.to_string();
    if key.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "metadata key must be non-empty",
        ));
    }
    let value = if value.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(value) }
                .to_str()
                .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("value utf8: {err}")))?,
        )
    };

    let mut ds = (*handle.dataset).clone();
    let updates = [(key.as_str(), value)];
    match runtime::block_on(async { ds.update_metadata(updates).await }) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetUpdateMetadata,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset update_metadata",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_config(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    match dataset_update_config_inner(dataset, key, value) {
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

fn dataset_update_config_inner(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> FfiResult<()> {
    let key = unsafe { cstr_to_str(key, "key")? }.to_string();
    if key.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "config key must be non-empty",
        ));
    }
    let value = if value.is_null() {
        None
    } else {
        Some(unsafe { cstr_to_str(value, "value")? }.to_string())
    };

    dataset_update_config_entries_inner(dataset, vec![(key, value)])
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_config_entries(
    dataset: *mut c_void,
    keys: *const *const c_char,
    values: *const *const c_char,
    entries_len: usize,
) -> i32 {
    match dataset_update_config_entries_from_ffi(dataset, keys, values, entries_len) {
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

fn dataset_update_config_entries_from_ffi(
    dataset: *mut c_void,
    keys: *const *const c_char,
    values: *const *const c_char,
    entries_len: usize,
) -> FfiResult<()> {
    if entries_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "config entries must be non-empty",
        ));
    }
    let keys = unsafe { super::util::slice_from_ptr(keys, entries_len, "config keys")? };
    let values = unsafe { super::util::slice_from_ptr(values, entries_len, "config values")? };
    let mut seen = HashSet::with_capacity(entries_len);
    let mut updates = Vec::with_capacity(entries_len);
    for (index, (&key, &value)) in keys.iter().zip(values.iter()).enumerate() {
        let key = unsafe { cstr_to_str(key, "config key")? };
        if key.is_empty() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("config key at index {index} is empty"),
            ));
        }
        if !seen.insert(key.to_string()) {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("duplicate config key: {key}"),
            ));
        }
        let value = if value.is_null() {
            None
        } else {
            Some(unsafe { cstr_to_str(value, "config value")? }.to_string())
        };
        updates.push((key.to_string(), value));
    }
    dataset_update_config_entries_inner(dataset, updates)
}

fn dataset_update_config_entries_inner(
    dataset: *mut c_void,
    updates: Vec<(String, Option<String>)>,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let mut ds = (*handle.dataset).clone();
    match runtime::block_on(async {
        ds.update_config(
            updates
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_deref())),
        )
        .await
    }) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetUpdateConfig,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset update_config",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_schema_metadata(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    match dataset_update_schema_metadata_inner(dataset, key, value) {
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

fn dataset_update_schema_metadata_inner(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let key = unsafe { cstr_to_str(key, "key")? }.to_string();
    if key.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "schema metadata key must be non-empty",
        ));
    }
    let value = if value.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(value) }
                .to_str()
                .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("value utf8: {err}")))?,
        )
    };

    let mut ds = (*handle.dataset).clone();
    let updates = [(key.as_str(), value)];
    match runtime::block_on(async { ds.update_schema_metadata(updates).await }) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetUpdateSchemaMetadata,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset update_schema_metadata",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_field_metadata(
    dataset: *mut c_void,
    field_path: *const c_char,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    match dataset_update_field_metadata_inner(dataset, field_path, key, value) {
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

fn dataset_update_field_metadata_inner(
    dataset: *mut c_void,
    field_path: *const c_char,
    key: *const c_char,
    value: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let field_path = unsafe { cstr_to_str(field_path, "field_path")? }.to_string();
    let key = unsafe { cstr_to_str(key, "key")? }.to_string();
    if field_path.is_empty() || key.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "field path and metadata key must be non-empty",
        ));
    }
    let value = if value.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(value) }
                .to_str()
                .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("value utf8: {err}")))?,
        )
    };

    let mut ds = (*handle.dataset).clone();
    let mut builder = ds.update_field_metadata();
    builder = builder
        .update(field_path.as_str(), [(key.as_str(), value)])
        .map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetUpdateFieldMetadata,
                format!("update_field_metadata: {err}"),
            )
        })?;

    match runtime::block_on(async { builder.await }) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetUpdateFieldMetadata,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset update_field_metadata",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_compact_files(dataset: *mut c_void) -> i32 {
    match dataset_compact_files_with_options_inner(dataset, ptr::null(), ptr::null_mut()) {
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

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_compact_files_with_options(
    dataset: *mut c_void,
    options_json: *const c_char,
    out_metrics_json: *mut *const c_char,
) -> i32 {
    if !out_metrics_json.is_null() {
        unsafe {
            ptr::write_unaligned(out_metrics_json, ptr::null());
        }
    }
    match dataset_compact_files_with_options_inner(dataset, options_json, out_metrics_json) {
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

fn dataset_compact_files_with_options_inner(
    dataset: *mut c_void,
    options_json: *const c_char,
    out_metrics_json: *mut *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let mut ds = (*handle.dataset).clone();
    let options = parse_compaction_options_json(options_json)?;

    match runtime::block_on(compact_files(&mut ds, options, None)) {
        Ok(Ok(metrics)) => write_metrics_json(
            &metrics,
            out_metrics_json,
            "compact_files metrics_json",
            ErrorCode::DatasetCommitOutcomeUnknown,
        ),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetCompactFiles,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset compact_files",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_cleanup_old_versions(
    dataset: *mut c_void,
    older_than_seconds: i64,
    delete_unverified: u8,
) -> i32 {
    let options = CleanupOldVersionsOptionsInput {
        older_than_seconds,
        delete_unverified: delete_unverified != 0,
        ..Default::default()
    };
    match dataset_cleanup_old_versions_with_options_struct(dataset, options, ptr::null_mut()) {
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

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_cleanup_old_versions_with_options(
    dataset: *mut c_void,
    options_json: *const c_char,
    out_metrics_json: *mut *const c_char,
) -> i32 {
    if !out_metrics_json.is_null() {
        unsafe {
            ptr::write_unaligned(out_metrics_json, ptr::null());
        }
    }
    match parse_cleanup_options_json(options_json).and_then(|options| {
        dataset_cleanup_old_versions_with_options_struct(dataset, options, out_metrics_json)
    }) {
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

fn dataset_cleanup_old_versions_with_options_struct(
    dataset: *mut c_void,
    options: CleanupOldVersionsOptionsInput,
    out_metrics_json: *mut *const c_char,
) -> FfiResult<()> {
    if options.older_than_seconds < 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "older_than_seconds must be >= 0",
        ));
    }
    if options.retain_n_versions == Some(0) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "retain_n_versions must be > 0",
        ));
    }

    const UNVERIFIED_GRACE_SECONDS: i64 = 7 * 24 * 60 * 60;
    let now = Utc::now();
    let before_timestamp =
        checked_cleanup_cutoff(now, options.older_than_seconds, "older_than_seconds")?;
    let staging_retention_seconds = if options.delete_unverified {
        options.older_than_seconds
    } else {
        options.older_than_seconds.max(UNVERIFIED_GRACE_SECONDS)
    };
    let staging_cutoff =
        checked_cleanup_cutoff(now, staging_retention_seconds, "staging retention interval")?;

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let mut builder = CleanupPolicyBuilder::default()
        .before_timestamp(before_timestamp)
        .delete_unverified(options.delete_unverified)
        .error_if_tagged_old_versions(options.error_if_tagged_old_versions);

    if let Some(retain_n_versions) = options.retain_n_versions {
        let retain_n_versions = usize::try_from(retain_n_versions).map_err(|err| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                format!("retain_n_versions too large: {err}"),
            )
        })?;
        builder = match runtime::block_on(builder.retain_n_versions(&ds, retain_n_versions)) {
            Ok(Ok(updated)) => updated,
            Ok(Err(err)) => {
                return Err(FfiError::new(
                    ErrorCode::DatasetCleanupOldVersions,
                    format!("cleanup_old_versions retain_n_versions: {err}"),
                ))
            }
            Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        };
    }

    let policy = builder.build();
    match runtime::block_on(async {
        let stats = ds.cleanup_with_policy(policy).await?;
        cleanup_vane_staging_files(&ds, staging_cutoff).await?;
        Ok::<_, lance::Error>(stats)
    }) {
        Ok(Ok(stats)) => {
            let metrics = CleanupOldVersionsMetricsOutput {
                bytes_removed: stats.bytes_removed,
                old_versions: stats.old_versions,
            };
            write_metrics_json(
                &metrics,
                out_metrics_json,
                "cleanup_old_versions metrics_json",
                ErrorCode::DatasetCommitOutcomeUnknown,
            )
        }
        // Cleanup is deliberately non-transactional: an error can arrive after
        // some old manifests, data files, or Vane staging files were removed.
        // Never describe that state as a definitive no-op, regardless of the
        // concrete Lance error variant.
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            format!("dataset cleanup_old_versions may be incomplete: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_config(dataset: *mut c_void) -> *const c_char {
    match dataset_list_kv_inner(dataset, "config") {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_table_metadata(dataset: *mut c_void) -> *const c_char {
    match dataset_list_kv_inner(dataset, "metadata") {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_schema_metadata(dataset: *mut c_void) -> *const c_char {
    match dataset_list_kv_inner(dataset, "schema_metadata") {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_field_metadata(
    dataset: *mut c_void,
    field_path: *const c_char,
) -> *const c_char {
    match dataset_list_field_metadata_inner(dataset, field_path) {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

fn dataset_list_field_metadata_inner(
    dataset: *mut c_void,
    field_path: *const c_char,
) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let field_path = unsafe { cstr_to_str(field_path, "field_path")? };

    let field = ds.schema().field(field_path).ok_or_else(|| {
        FfiError::new(
            ErrorCode::DatasetListKeyValues,
            format!("field not found: '{field_path}'"),
        )
    })?;
    let mut out = String::new();
    for (k, v) in field.metadata.iter() {
        super::util::push_ffi_key_value_row(&mut out, k, v);
    }
    Ok(out)
}

fn dataset_list_kv_inner(dataset: *mut c_void, which: &'static str) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let mut out = String::new();

    match which {
        "config" => {
            for (k, v) in ds.config().iter() {
                super::util::push_ffi_key_value_row(&mut out, k, v);
            }
        }
        "metadata" => {
            for (k, v) in ds.metadata().iter() {
                super::util::push_ffi_key_value_row(&mut out, k, v);
            }
        }
        "schema_metadata" => {
            for (k, v) in ds.schema().metadata.iter() {
                super::util::push_ffi_key_value_row(&mut out, k, v);
            }
        }
        _ => return Err(FfiError::new(ErrorCode::InvalidArgument, "unknown kv type")),
    }

    Ok(out)
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_indices(dataset: *mut c_void) -> *const c_char {
    match dataset_list_indices_inner(dataset) {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

fn dataset_list_indices_inner(dataset: *mut c_void) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();

    let indices = match runtime::block_on(ds.load_indices()) {
        Ok(Ok(v)) => v,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetListIndices,
                format!("dataset load_indices: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let schema = ds.schema();
    let mut out = String::new();
    for idx in indices.iter() {
        let cols = idx
            .fields
            .iter()
            .map(|id| {
                schema.field_path(*id).map_err(|err| {
                    FfiError::new(
                        ErrorCode::DatasetListIndices,
                        format!(
                            "index '{}' references invalid field id {id}: {err}",
                            idx.name
                        ),
                    )
                })
            })
            .collect::<FfiResult<Vec<_>>>()?
            .join(",");
        super::util::push_ffi_key_value_row(&mut out, &idx.name, &cols);
    }
    Ok(out)
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_create_scalar_index(
    dataset: *mut c_void,
    column: *const c_char,
    index_name: *const c_char,
    replace: u8,
) -> i32 {
    match dataset_create_scalar_index_inner(dataset, column, index_name, replace != 0) {
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

fn dataset_create_scalar_index_inner(
    dataset: *mut c_void,
    column: *const c_char,
    index_name: *const c_char,
    replace: bool,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let column = unsafe { cstr_to_str(column, "column")? };
    let index_name = unsafe { cstr_to_str(index_name, "index_name")? };
    if column.is_empty() || index_name.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "index column and name must be non-empty",
        ));
    }

    let mut ds = (*handle.dataset).clone();
    let canonical_column = canonicalize_lance_field_path(ds.schema(), column, "index column")?;
    let cols = [canonical_column.as_str()];
    match runtime::block_on(ds.create_index(
        &cols,
        IndexType::Scalar,
        Some(index_name.to_string()),
        &ScalarIndexParams::default(),
        replace,
    )) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(lance_mutation_error(
            ErrorCode::DatasetCreateScalarIndex,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "dataset create_index(scalar)",
            err,
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}
