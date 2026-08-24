use std::ffi::{c_char, c_void};
use std::ptr;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_array::builder::Float32Builder;
use arrow_array::{Float32Array, UInt32Array, UInt64Array};
use lance::dataset::ProjectionRequest;
use lance_index::scalar::FullTextSearchQuery;

use crate::constants::{DISTANCE_COLUMN, HYBRID_SCORE_COLUMN, ROW_ID_COLUMN, SCORE_COLUMN};
use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;
use crate::scanner::LanceStream;

use super::projection;
use super::types::{SchemaHandle, StreamHandle};
use super::util::{
    cstr_to_str, nonzero_u64_to_i64, nonzero_u64_to_usize, optional_cstr_to_string,
    parse_optional_filter_ir, slice_from_ptr, FfiError, FfiResult,
};

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_get_fts_schema(
    dataset: *mut c_void,
    text_column: *const c_char,
    query: *const c_char,
    k: u64,
    prefilter: u8,
) -> *mut c_void {
    match get_fts_schema_inner(dataset, text_column, query, k, prefilter) {
        Ok(schema) => {
            clear_last_error();
            Box::into_raw(Box::new(schema)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn get_fts_schema_inner(
    dataset: *mut c_void,
    text_column: *const c_char,
    query: *const c_char,
    k: u64,
    prefilter: u8,
) -> FfiResult<SchemaHandle> {
    let text_column = unsafe { cstr_to_str(text_column, "text_column")? };
    let query = unsafe { cstr_to_str(query, "query")? };
    let k_i64 = nonzero_u64_to_i64(k, "k")?;

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let projection = handle.fts_projection.clone();

    let fts_query = FullTextSearchQuery::new(query.to_string())
        .with_column(text_column.to_string())
        .map_err(|err| FfiError::new(ErrorCode::FtsSchema, format!("fts query: {err}")))?
        .limit(Some(k_i64));

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    scan.full_text_search(fts_query)
        .map_err(|err| FfiError::new(ErrorCode::FtsSchema, format!("fts schema search: {err}")))?;
    scan.disable_scoring_autoprojection();
    scan.project(projection.as_ref())
        .map_err(|err| FfiError::new(ErrorCode::FtsSchema, format!("fts schema project: {err}")))?;
    scan.scan_in_order(false);

    let schema = LanceStream::from_scanner(scan)
        .map_err(|err| FfiError::new(ErrorCode::FtsSchema, format!("fts schema: {err}")))?
        .schema();
    Ok(schema)
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_create_fts_stream_ir(
    dataset: *mut c_void,
    text_column: *const c_char,
    query: *const c_char,
    k: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    filter_sql: *const c_char,
    prefilter: u8,
) -> *mut c_void {
    match create_fts_stream_ir_inner(
        dataset,
        text_column,
        query,
        k,
        filter_ir,
        filter_ir_len,
        filter_sql,
        prefilter,
    ) {
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

#[allow(clippy::too_many_arguments)]
fn create_fts_stream_ir_inner(
    dataset: *mut c_void,
    text_column: *const c_char,
    query: *const c_char,
    k: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    filter_sql: *const c_char,
    prefilter: u8,
) -> FfiResult<StreamHandle> {
    let text_column = unsafe { cstr_to_str(text_column, "text_column")? };
    let query = unsafe { cstr_to_str(query, "query")? };
    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::FtsStreamCreate,
            "fts filter_ir",
        )?
    };
    let filter_sql = unsafe { optional_cstr_to_string(filter_sql, "fts filter_sql")? };
    let k_i64 = nonzero_u64_to_i64(k, "k")?;

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let projection = handle.fts_projection.clone();

    let fts_query = FullTextSearchQuery::new(query.to_string())
        .with_column(text_column.to_string())
        .map_err(|err| FfiError::new(ErrorCode::FtsStreamCreate, format!("fts query: {err}")))?
        .limit(Some(k_i64));

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter_sql) = filter_sql.as_deref() {
        scan.filter(filter_sql).map_err(|err| {
            FfiError::new(ErrorCode::FtsStreamCreate, format!("fts filter_sql: {err}"))
        })?;
    }
    if let Some(filter) = filter {
        let filter = match scan.get_expr_filter().map_err(|err| {
            FfiError::new(
                ErrorCode::FtsStreamCreate,
                format!("fts filter_sql parse: {err}"),
            )
        })? {
            Some(sql_filter) => sql_filter.and(filter),
            None => filter,
        };
        scan.filter_expr(filter);
    }
    scan.full_text_search(fts_query).map_err(|err| {
        FfiError::new(
            ErrorCode::FtsStreamCreate,
            format!("fts scan search: {err}"),
        )
    })?;
    scan.disable_scoring_autoprojection();
    scan.project(projection.as_ref()).map_err(|err| {
        FfiError::new(
            ErrorCode::FtsStreamCreate,
            format!("fts scan project: {err}"),
        )
    })?;
    scan.scan_in_order(false);

    let stream = LanceStream::from_scanner(scan).map_err(|err| {
        FfiError::new(
            ErrorCode::FtsStreamCreate,
            format!("fts stream create: {err}"),
        )
    })?;
    Ok(StreamHandle::Lance(stream))
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_get_hybrid_schema(dataset: *mut c_void) -> *mut c_void {
    match get_hybrid_schema_inner(dataset) {
        Ok(schema) => {
            clear_last_error();
            Box::into_raw(Box::new(schema)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn get_hybrid_schema_inner(dataset: *mut c_void) -> FfiResult<SchemaHandle> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    Ok(projection::build_hybrid_schema(&handle.arrow_schema))
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_create_hybrid_stream_ir(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    text_column: *const c_char,
    text_query: *const c_char,
    k: u64,
    nprobes: u64,
    refine_factor: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    prefilter: u8,
    use_index: u8,
    alpha: f32,
    oversample_factor: u32,
) -> *mut c_void {
    match create_hybrid_stream_ir_inner(
        dataset,
        vector_column,
        query_values,
        query_len,
        text_column,
        text_query,
        k,
        nprobes,
        refine_factor,
        filter_ir,
        filter_ir_len,
        prefilter,
        use_index,
        alpha,
        oversample_factor,
    ) {
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

#[allow(clippy::too_many_arguments)]
fn create_hybrid_stream_ir_inner(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    text_column: *const c_char,
    text_query: *const c_char,
    k: u64,
    nprobes: u64,
    refine_factor: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    prefilter: u8,
    use_index: u8,
    alpha: f32,
    oversample_factor: u32,
) -> FfiResult<StreamHandle> {
    if query_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "query vector must be non-empty",
        ));
    }
    let vector_column = unsafe { cstr_to_str(vector_column, "vector_column")? };
    let text_column = unsafe { cstr_to_str(text_column, "text_column")? };
    let text_query = unsafe { cstr_to_str(text_query, "text_query")? };
    let query_values = unsafe { slice_from_ptr(query_values, query_len, "query_values")? };
    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::HybridStreamCreate,
            "hybrid filter_ir",
        )?
    };
    let k_usize = nonzero_u64_to_usize(k, "k")?;

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let rows = create_hybrid_batch(
        handle,
        vector_column,
        query_values,
        text_column,
        text_query,
        k_usize,
        nprobes,
        refine_factor,
        filter,
        prefilter,
        use_index,
        alpha,
        oversample_factor,
    )?;

    Ok(StreamHandle::Batches(vec![rows].into_iter()))
}

#[allow(clippy::too_many_arguments)]
fn create_hybrid_batch(
    handle: &super::types::DatasetHandle,
    vector_column: &str,
    query_values: &[f32],
    text_column: &str,
    text_query: &str,
    k: usize,
    nprobes: u64,
    refine_factor: u64,
    filter: Option<datafusion_expr::Expr>,
    prefilter: u8,
    use_index: u8,
    alpha: f32,
    oversample_factor: u32,
) -> FfiResult<RecordBatch> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "hybrid alpha must be finite and between 0 and 1",
        ));
    }
    if oversample_factor == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "hybrid oversample_factor must be greater than zero",
        ));
    }
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    let oversample = k.checked_mul(oversample_factor as usize).ok_or_else(|| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            "hybrid candidate count overflow",
        )
    })?;
    let oversample_i64 = i64::try_from(oversample).map_err(|_| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            "hybrid candidate count must fit in i64",
        )
    })?;

    let mut vector_scan = handle.dataset.scan();
    vector_scan.prefilter(prefilter != 0);
    if let Some(expr) = filter.clone() {
        vector_scan.filter_expr(expr);
    }
    vector_scan
        .nearest(vector_column, &query, oversample)
        .map_err(|err| {
            FfiError::new(
                ErrorCode::HybridStreamCreate,
                format!("hybrid vector nearest: {err}"),
            )
        })?;
    if nprobes != 0 {
        let nprobes_usize = nonzero_u64_to_usize(nprobes, "nprobes")?;
        vector_scan.nprobes(nprobes_usize);
    }
    if refine_factor != 0 {
        let refine_factor_u32: u32 = refine_factor.try_into().map_err(|_| {
            FfiError::new(ErrorCode::InvalidArgument, "refine_factor must fit in u32")
        })?;
        vector_scan.refine(refine_factor_u32);
    }
    vector_scan.use_index(use_index != 0);
    vector_scan.with_row_id();
    vector_scan.disable_scoring_autoprojection();
    vector_scan
        .project(&[ROW_ID_COLUMN, DISTANCE_COLUMN])
        .map_err(|err| {
            FfiError::new(
                ErrorCode::HybridStreamCreate,
                format!("hybrid vector project: {err}"),
            )
        })?;
    vector_scan.scan_in_order(false);

    let mut vector_stream = LanceStream::from_scanner(vector_scan).map_err(|err| {
        FfiError::new(
            ErrorCode::HybridStreamCreate,
            format!("hybrid vector stream: {err}"),
        )
    })?;

    let vector_rows = collect_row_f32_pairs(&mut vector_stream, ROW_ID_COLUMN, DISTANCE_COLUMN)
        .map_err(|err| FfiError::new(ErrorCode::HybridStreamCreate, err))?;

    let fts_query = FullTextSearchQuery::new(text_query.to_string())
        .with_column(text_column.to_string())
        .map_err(|err| {
            FfiError::new(
                ErrorCode::HybridStreamCreate,
                format!("hybrid fts query: {err}"),
            )
        })?
        .limit(Some(oversample_i64));

    let mut fts_scan = handle.dataset.scan();
    fts_scan.prefilter(prefilter != 0);
    if let Some(expr) = filter {
        fts_scan.filter_expr(expr);
    }
    fts_scan.full_text_search(fts_query).map_err(|err| {
        FfiError::new(
            ErrorCode::HybridStreamCreate,
            format!("hybrid fts search: {err}"),
        )
    })?;
    fts_scan.with_row_id();
    fts_scan.disable_scoring_autoprojection();
    fts_scan
        .project(&[ROW_ID_COLUMN, SCORE_COLUMN])
        .map_err(|err| {
            FfiError::new(
                ErrorCode::HybridStreamCreate,
                format!("hybrid fts project: {err}"),
            )
        })?;
    fts_scan.scan_in_order(false);

    let mut fts_stream = LanceStream::from_scanner(fts_scan).map_err(|err| {
        FfiError::new(
            ErrorCode::HybridStreamCreate,
            format!("hybrid fts stream: {err}"),
        )
    })?;
    let fts_rows = collect_row_f32_pairs(&mut fts_stream, ROW_ID_COLUMN, SCORE_COLUMN)
        .map_err(|err| FfiError::new(ErrorCode::HybridStreamCreate, err))?;

    let (dist_min, dist_max) = finite_min_max(vector_rows.iter().map(|(_, v)| *v));
    let (score_min, score_max) = finite_min_max(fts_rows.iter().map(|(_, v)| *v));

    let mut merged = std::collections::HashMap::<u64, (Option<f32>, Option<f32>)>::new();
    for (rowid, dist) in vector_rows {
        merged
            .entry(rowid)
            .and_modify(|e| {
                e.0 = match e.0 {
                    Some(old) => Some(old.min(dist)),
                    None => Some(dist),
                }
            })
            .or_insert((Some(dist), None));
    }
    for (rowid, score) in fts_rows {
        merged
            .entry(rowid)
            .and_modify(|e| {
                e.1 = match e.1 {
                    Some(old) => Some(old.max(score)),
                    None => Some(score),
                }
            })
            .or_insert((None, Some(score)));
    }

    let mut ranked: Vec<(u64, Option<f32>, Option<f32>, f32)> = merged
        .into_iter()
        .map(|(rowid, (dist, score))| {
            let mut hybrid = 0.0_f32;
            if let Some(d) = dist {
                hybrid += alpha * distance_similarity(d, dist_min, dist_max);
            }
            if let Some(s) = score {
                hybrid += (1.0 - alpha) * normalize_value(s, score_min, score_max);
            }
            (rowid, dist, score, hybrid)
        })
        .collect();

    ranked.sort_by(|a, b| cmp_desc_f32(b.3, a.3).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);

    let row_ids: Vec<u64> = ranked.iter().map(|(rowid, _, _, _)| *rowid).collect();
    let rows = if row_ids.is_empty() {
        // Lance's take_rows fast path expects at least one row address and
        // panics for an empty request.  Preserve the typed empty result
        // contract for filters/searches that produce no candidates.
        RecordBatch::new_empty(handle.arrow_schema.clone())
    } else {
        let mut take_columns = handle.base_projection.to_vec();
        take_columns.push(ROW_ID_COLUMN.to_string());
        let projection = ProjectionRequest::from_columns(&take_columns, handle.dataset.schema());
        let unordered_rows = match runtime::block_on(handle.dataset.take_rows(&row_ids, projection))
        {
            Ok(Ok(batch)) => batch,
            Ok(Err(err)) => {
                return Err(FfiError::new(
                    ErrorCode::HybridStreamCreate,
                    format!("hybrid take_rows: {err}"),
                ))
            }
            Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        };
        order_hybrid_take_rows(unordered_rows, &row_ids)?
    };

    let mut dist_builder = Float32Builder::with_capacity(rows.num_rows());
    let mut score_builder = Float32Builder::with_capacity(rows.num_rows());
    let mut hybrid_builder = Float32Builder::with_capacity(rows.num_rows());
    for (_, dist, score, hybrid) in &ranked {
        match dist {
            Some(v) if v.is_finite() => dist_builder.append_value(*v),
            _ => dist_builder.append_null(),
        }
        match score {
            Some(v) if v.is_finite() => score_builder.append_value(*v),
            _ => score_builder.append_null(),
        }
        if hybrid.is_finite() {
            hybrid_builder.append_value(*hybrid);
        } else {
            hybrid_builder.append_null();
        }
    }

    let mut cols: Vec<Arc<dyn Array>> = rows.columns().to_vec();
    cols.push(Arc::new(dist_builder.finish()) as Arc<dyn Array>);
    cols.push(Arc::new(score_builder.finish()) as Arc<dyn Array>);
    cols.push(Arc::new(hybrid_builder.finish()) as Arc<dyn Array>);

    let mut fields = rows.schema().fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(Field::new(
        DISTANCE_COLUMN,
        DataType::Float32,
        true,
    )));
    fields.push(Arc::new(Field::new(SCORE_COLUMN, DataType::Float32, true)));
    fields.push(Arc::new(Field::new(
        HYBRID_SCORE_COLUMN,
        DataType::Float32,
        true,
    )));

    let out_schema = Arc::new(Schema::new_with_metadata(
        fields,
        rows.schema().metadata().clone(),
    ));
    RecordBatch::try_new(out_schema, cols).map_err(|err| {
        FfiError::new(
            ErrorCode::HybridStreamCreate,
            format!("hybrid batch: {err}"),
        )
    })
}

fn order_hybrid_take_rows(rows: RecordBatch, row_ids: &[u64]) -> FfiResult<RecordBatch> {
    let rowid_index = rows.schema().index_of(ROW_ID_COLUMN).map_err(|_| {
        FfiError::new(
            ErrorCode::HybridStreamCreate,
            "hybrid take_rows response is missing _rowid",
        )
    })?;
    let returned_row_ids = rows
        .column(rowid_index)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            FfiError::new(
                ErrorCode::HybridStreamCreate,
                "hybrid take_rows _rowid column is not UInt64",
            )
        })?;

    let mut positions = std::collections::HashMap::with_capacity(returned_row_ids.len());
    for index in 0..returned_row_ids.len() {
        if returned_row_ids.is_null(index) {
            return Err(FfiError::new(
                ErrorCode::HybridStreamCreate,
                "hybrid take_rows returned a NULL _rowid",
            ));
        }
        let row_id = returned_row_ids.value(index);
        if positions.insert(row_id, index).is_some() {
            return Err(FfiError::new(
                ErrorCode::HybridStreamCreate,
                format!("hybrid take_rows returned duplicate _rowid {row_id}"),
            ));
        }
    }

    let ordered_indices = row_ids
        .iter()
        .map(|row_id| {
            let index = positions.get(row_id).ok_or_else(|| {
                FfiError::new(
                    ErrorCode::HybridStreamCreate,
                    format!("hybrid take_rows omitted requested _rowid {row_id}"),
                )
            })?;
            u32::try_from(*index).map_err(|_| {
                FfiError::new(
                    ErrorCode::HybridStreamCreate,
                    "hybrid take_rows response exceeds UInt32 row indexing",
                )
            })
        })
        .collect::<FfiResult<Vec<_>>>()?;
    if positions.len() != row_ids.len() {
        return Err(FfiError::new(
            ErrorCode::HybridStreamCreate,
            "hybrid take_rows returned rows that were not requested",
        ));
    }
    let indices = UInt32Array::from(ordered_indices);

    let mut fields = Vec::with_capacity(rows.num_columns().saturating_sub(1));
    let mut columns = Vec::with_capacity(rows.num_columns().saturating_sub(1));
    for (index, field) in rows.schema().fields().iter().enumerate() {
        if index == rowid_index {
            continue;
        }
        fields.push(field.clone());
        columns.push(
            arrow::compute::take(rows.column(index).as_ref(), &indices, None).map_err(|err| {
                FfiError::new(
                    ErrorCode::HybridStreamCreate,
                    format!("hybrid take_rows reorder: {err}"),
                )
            })?,
        );
    }
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            fields,
            rows.schema().metadata().clone(),
        )),
        columns,
    )
    .map_err(|err| {
        FfiError::new(
            ErrorCode::HybridStreamCreate,
            format!("hybrid reordered batch: {err}"),
        )
    })
}

fn collect_row_f32_pairs(
    stream: &mut LanceStream,
    rowid_col: &str,
    value_col: &str,
) -> Result<Vec<(u64, f32)>, String> {
    let mut out = Vec::<(u64, f32)>::new();
    loop {
        match stream.next() {
            Ok(Some(batch)) => {
                let idx_rowid = batch
                    .schema()
                    .index_of(rowid_col)
                    .map_err(|_| format!("batch missing {rowid_col}"))?;
                let idx_val = batch
                    .schema()
                    .index_of(value_col)
                    .map_err(|_| format!("batch missing {value_col}"))?;

                let rowids = batch
                    .column(idx_rowid)
                    .as_any()
                    .downcast_ref::<UInt64Array>();
                let values = batch
                    .column(idx_val)
                    .as_any()
                    .downcast_ref::<Float32Array>();
                let (Some(rowids), Some(values)) = (rowids, values) else {
                    return Err("batch has unexpected column types".to_string());
                };

                for i in 0..batch.num_rows() {
                    if rowids.is_null(i) || values.is_null(i) {
                        continue;
                    }
                    out.push((rowids.value(i), values.value(i)));
                }
            }
            Ok(None) => break,
            Err(err) => return Err(format!("stream next: {err}")),
        }
    }
    Ok(out)
}

fn finite_min_max(values: impl Iterator<Item = f32>) -> (f32, f32) {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for v in values {
        if !v.is_finite() {
            continue;
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if !min_v.is_finite() {
        (0.0, 0.0)
    } else {
        (min_v, max_v)
    }
}

fn normalize_value(v: f32, min_v: f32, max_v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    if (max_v - min_v).abs() < f32::EPSILON {
        return 0.5;
    }
    ((v - min_v) / (max_v - min_v)).clamp(0.0, 1.0)
}

fn distance_similarity(dist: f32, min_v: f32, max_v: f32) -> f32 {
    if !dist.is_finite() {
        return 0.0;
    }
    1.0 - normalize_value(dist, min_v, max_v)
}

fn cmp_desc_f32(a: f32, b: f32) -> std::cmp::Ordering {
    match a.partial_cmp(&b) {
        Some(ord) => ord,
        None => {
            if a.is_nan() && b.is_nan() {
                std::cmp::Ordering::Equal
            } else if a.is_nan() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;

    fn hybrid_rows(values: Vec<i64>, row_ids: Vec<u64>) -> RecordBatch {
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("value", DataType::Int64, false),
                Field::new(ROW_ID_COLUMN, DataType::UInt64, false),
            ],
            std::collections::HashMap::from([(
                "schema-key".to_string(),
                "schema-value".to_string(),
            )]),
        ));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(values)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .expect("valid hybrid rows")
    }

    #[test]
    fn hybrid_take_rows_are_reordered_to_match_ranked_row_ids() {
        let rows = hybrid_rows(vec![20, 10], vec![2, 1]);

        let ordered = order_hybrid_take_rows(rows, &[1, 2]).expect("reordered rows");

        assert_eq!(ordered.num_columns(), 1);
        assert_eq!(ordered.schema().field(0).name(), "value");
        assert_eq!(
            ordered
                .schema()
                .metadata()
                .get("schema-key")
                .map(String::as_str),
            Some("schema-value")
        );
        let values = ordered
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 values");
        assert_eq!(values.values(), &[10, 20]);
    }

    #[test]
    fn hybrid_take_rows_rejects_missing_requested_row_ids() {
        let rows = hybrid_rows(vec![10], vec![1]);

        let err = order_hybrid_take_rows(rows, &[1, 2]).expect_err("missing row ID must fail");

        assert!(err.message.contains("omitted requested _rowid 2"));
    }
}
