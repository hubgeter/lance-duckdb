use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::JoinHandle;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, Float64Builder};
use arrow_array::{
    make_array, Array, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
    RecordBatch, RecordBatchReader, StructArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, Dataset, InsertBuilder, WriteMode, WriteParams};
use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor};
use lance::session::Session;
use lance_core::datatypes::{Field as LanceField, Schema as LanceSchema, SchemaCompareOptions};
use lance_table::format::{pb, Fragment, RowIdMeta};
use lance_table::io::deletion::relative_deletion_file_path;
use lance_table::rowids::version::RowDatasetVersionMeta;
use object_store::path::Path;
use prost::Message;
use serde::{Deserialize, Serialize};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::session::record_commit;
use super::util::{cstr_to_str, optional_session_handle, slice_from_ptr, FfiError, FfiResult};

const VANE_OPERATION_ID_PROPERTY: &str = "vane.distributed.operation_id";
const VANE_ROW_COUNT_PROPERTY: &str = "vane.distributed.row_count";
const VANE_WRITE_MODE_PROPERTY: &str = "vane.distributed.write_mode";
const VANE_NULL_VECTOR_FIELDS_PROPERTY: &str = "vane.distributed.null_vector_fields";
const VANE_OPERATION_MARKER_FORMAT_VERSION: u32 = 1;

#[repr(C)]
struct RawArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut RawArrowArray,
    dictionary: *mut RawArrowArray,
    release: Option<unsafe extern "C" fn(arg1: *mut RawArrowArray)>,
    private_data: *mut c_void,
}

struct ReceiverRecordBatchReader {
    schema: SchemaRef,
    receiver: Receiver<RecordBatch>,
    aborted: Arc<AtomicBool>,
}

impl ReceiverRecordBatchReader {
    fn new(schema: SchemaRef, receiver: Receiver<RecordBatch>, aborted: Arc<AtomicBool>) -> Self {
        Self {
            schema,
            receiver,
            aborted,
        }
    }
}

impl Iterator for ReceiverRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(batch) => Some(Ok(batch)),
            Err(_) if self.aborted.load(Ordering::Acquire) => Some(Err(
                ArrowError::InvalidArgumentError("Lance writer was aborted".to_string()),
            )),
            Err(_) => None,
        }
    }
}

impl RecordBatchReader for ReceiverRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

struct WriterHandle {
    input_schema: SchemaRef,
    data_type: DataType,
    state: Mutex<WriterState>,
    batches_sent: AtomicU64,
    aborted: Arc<AtomicBool>,
}

enum WriterResult {
    Committed,
    Uncommitted(Box<lance::dataset::transaction::Transaction>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterKind {
    Committed,
    Uncommitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorListKind {
    List,
    LargeList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorElementType {
    Float32,
    Float64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VectorConversion {
    col_idx: usize,
    field_name: String,
    dim: usize,
    list_kind: VectorListKind,
    element_type: VectorElementType,
}

struct WriterState {
    kind: WriterKind,
    path: String,
    params: WriteParams,

    vector_candidates: Vec<VectorConversion>,
    buffered_batches: Vec<RecordBatch>,
    buffered_rows: usize,

    output_schema: Option<SchemaRef>,
    output_sender: Option<SyncSender<RecordBatch>>,
    output_join: Option<JoinHandle<Result<WriterResult, String>>>,
}

impl Drop for WriterHandle {
    fn drop(&mut self) {
        // Closing a writer without calling one of the finish entry points is an
        // abort, not a clean end-of-stream.  Publish that decision before the
        // last sender is dropped so the background reader cannot commit a
        // partial COPY/INSERT on cancellation or upstream failure.
        self.aborted.store(true, Ordering::Release);
        let (sender, join) = {
            let mut guard = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (guard.output_sender.take(), guard.output_join.take())
        };
        drop(sender);
        if let Some(join) = join {
            let _ = join.join();
        }
    }
}

// Buffer by row count, not Arrow batch count: batch boundaries are an execution
// detail and must not decide whether a LIST<FLOAT> column becomes a vector.
// A late value after this bound fails loudly instead of silently freezing the
// physical schema as a variable list.  Users can make the decision explicit
// with VECTOR_DIMS or disable conversion with INFER_VECTOR_DIMS false.
const MAX_VECTOR_DIM_INFERENCE_ROWS: usize = 65_536;

fn is_variable_list_vector_type(dt: &DataType) -> Option<(VectorListKind, VectorElementType)> {
    match dt {
        DataType::List(field) => match field.data_type() {
            DataType::Float32 => Some((VectorListKind::List, VectorElementType::Float32)),
            DataType::Float64 => Some((VectorListKind::List, VectorElementType::Float64)),
            _ => None,
        },
        DataType::LargeList(field) => match field.data_type() {
            DataType::Float32 => Some((VectorListKind::LargeList, VectorElementType::Float32)),
            DataType::Float64 => Some((VectorListKind::LargeList, VectorElementType::Float64)),
            _ => None,
        },
        _ => None,
    }
}

fn parse_vector_candidates(
    schema: &SchemaRef,
    vector_dims: *const c_char,
    infer_vector_dims: bool,
) -> FfiResult<Vec<VectorConversion>> {
    let mut explicit_dims = if vector_dims.is_null() {
        HashMap::new()
    } else {
        let raw = unsafe { cstr_to_str(vector_dims, "vector_dims")? };
        serde_json::from_str::<HashMap<String, usize>>(raw).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetWriteOpen,
                format!(
                    "vector_dims must be a JSON object mapping column names to positive dimensions: {err}"
                ),
            )
        })?
    };

    if let Some((field_name, _)) = explicit_dims.iter().find(|(_, dim)| **dim == 0) {
        return Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("vector_dims for column '{field_name}' must be greater than zero"),
        ));
    }

    let mut candidates = Vec::new();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let explicit_dim = explicit_dims.remove(field.name());
        let Some((list_kind, element_type)) = is_variable_list_vector_type(field.data_type())
        else {
            if explicit_dim.is_some() {
                return Err(FfiError::new(
                    ErrorCode::DatasetWriteOpen,
                    format!(
                        "vector_dims column '{}' must have type LIST(FLOAT) or LIST(DOUBLE)",
                        field.name()
                    ),
                ));
            }
            continue;
        };
        if explicit_dim.is_none() && !infer_vector_dims {
            continue;
        }
        candidates.push(VectorConversion {
            col_idx,
            field_name: field.name().clone(),
            dim: explicit_dim.unwrap_or(0),
            list_kind,
            element_type,
        });
    }

    if let Some(field_name) = explicit_dims.keys().next() {
        return Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("vector_dims references unknown column '{field_name}'"),
        ));
    }
    Ok(candidates)
}

fn infer_vector_dim_from_array(
    array: &dyn Array,
    list_kind: VectorListKind,
) -> Option<Result<usize, String>> {
    match list_kind {
        VectorListKind::List => {
            let list = array.as_any().downcast_ref::<ListArray>()?;
            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }
                let dim = list.value_length(i) as usize;
                if dim == 0 {
                    return Some(Err("vector dim must be non-zero".to_string()));
                }
                return Some(Ok(dim));
            }
            None
        }
        VectorListKind::LargeList => {
            let list = array.as_any().downcast_ref::<LargeListArray>()?;
            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }
                let dim = list.value_length(i) as usize;
                if dim == 0 {
                    return Some(Err("vector dim must be non-zero".to_string()));
                }
                return Some(Ok(dim));
            }
            None
        }
    }
}

fn validate_list_vector_dim(
    array: &dyn Array,
    list_kind: VectorListKind,
    expected_dim: usize,
) -> Result<(), String> {
    match list_kind {
        VectorListKind::List => {
            let list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| "vector column is not ListArray".to_string())?;
            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }
                let dim = list.value_length(i) as usize;
                if dim != expected_dim {
                    return Err(format!(
                        "vector dim mismatch: expected {expected_dim} got {dim}"
                    ));
                }
            }
            Ok(())
        }
        VectorListKind::LargeList => {
            let list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| "vector column is not LargeListArray".to_string())?;
            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }
                let dim = list.value_length(i) as usize;
                if dim != expected_dim {
                    return Err(format!(
                        "vector dim mismatch: expected {expected_dim} got {dim}"
                    ));
                }
            }
            Ok(())
        }
    }
}

fn convert_list_array_to_fixed_size(
    array: &dyn Array,
    list_kind: VectorListKind,
    element_type: VectorElementType,
    dim: usize,
) -> Result<FixedSizeListArray, String> {
    let dim_i32 = i32::try_from(dim).map_err(|_| "vector dim is too large".to_string())?;

    match (list_kind, element_type) {
        (VectorListKind::List, VectorElementType::Float32) => {
            let list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| "vector column is not ListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| "vector values are not Float32".to_string())?;
            let field = match list.data_type() {
                DataType::List(field) => field.clone(),
                _ => return Err("vector column has unexpected data type".to_string()),
            };

            let mut builder =
                FixedSizeListBuilder::with_capacity(Float32Builder::new(), dim_i32, list.len())
                    .with_field(field);
            let offsets = list.value_offsets();
            for (i, start) in offsets.iter().take(list.len()).enumerate() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!("vector dim mismatch: expected {dim} got {len}"));
                }
                let start = *start as usize;
                for j in 0..dim {
                    let idx = start + j;
                    if idx >= values.len() {
                        return Err("vector offsets are out of bounds".to_string());
                    }
                    if values.is_null(idx) {
                        builder.values().append_null();
                    } else {
                        builder.values().append_value(values.value(idx));
                    }
                }
                builder.append(true);
            }
            Ok(builder.finish())
        }
        (VectorListKind::List, VectorElementType::Float64) => {
            let list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| "vector column is not ListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| "vector values are not Float64".to_string())?;
            let field = match list.data_type() {
                DataType::List(field) => field.clone(),
                _ => return Err("vector column has unexpected data type".to_string()),
            };

            let mut builder =
                FixedSizeListBuilder::with_capacity(Float64Builder::new(), dim_i32, list.len())
                    .with_field(field);
            let offsets = list.value_offsets();
            for (i, start) in offsets.iter().take(list.len()).enumerate() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!("vector dim mismatch: expected {dim} got {len}"));
                }
                let start = *start as usize;
                for j in 0..dim {
                    let idx = start + j;
                    if idx >= values.len() {
                        return Err("vector offsets are out of bounds".to_string());
                    }
                    if values.is_null(idx) {
                        builder.values().append_null();
                    } else {
                        builder.values().append_value(values.value(idx));
                    }
                }
                builder.append(true);
            }
            Ok(builder.finish())
        }
        (VectorListKind::LargeList, VectorElementType::Float32) => {
            let list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| "vector column is not LargeListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| "vector values are not Float32".to_string())?;
            let field = match list.data_type() {
                DataType::LargeList(field) => field.clone(),
                _ => return Err("vector column has unexpected data type".to_string()),
            };

            let mut builder =
                FixedSizeListBuilder::with_capacity(Float32Builder::new(), dim_i32, list.len())
                    .with_field(field);
            let offsets = list.value_offsets();
            for (i, start) in offsets.iter().take(list.len()).enumerate() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!("vector dim mismatch: expected {dim} got {len}"));
                }
                let start = *start as usize;
                for j in 0..dim {
                    let idx = start + j;
                    if idx >= values.len() {
                        return Err("vector offsets are out of bounds".to_string());
                    }
                    if values.is_null(idx) {
                        builder.values().append_null();
                    } else {
                        builder.values().append_value(values.value(idx));
                    }
                }
                builder.append(true);
            }
            Ok(builder.finish())
        }
        (VectorListKind::LargeList, VectorElementType::Float64) => {
            let list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| "vector column is not LargeListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| "vector values are not Float64".to_string())?;
            let field = match list.data_type() {
                DataType::LargeList(field) => field.clone(),
                _ => return Err("vector column has unexpected data type".to_string()),
            };

            let mut builder =
                FixedSizeListBuilder::with_capacity(Float64Builder::new(), dim_i32, list.len())
                    .with_field(field);
            let offsets = list.value_offsets();
            for (i, start) in offsets.iter().take(list.len()).enumerate() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!("vector dim mismatch: expected {dim} got {len}"));
                }
                let start = *start as usize;
                for j in 0..dim {
                    let idx = start + j;
                    if idx >= values.len() {
                        return Err("vector offsets are out of bounds".to_string());
                    }
                    if values.is_null(idx) {
                        builder.values().append_null();
                    } else {
                        builder.values().append_value(values.value(idx));
                    }
                }
                builder.append(true);
            }
            Ok(builder.finish())
        }
    }
}

fn build_output_schema(
    input_schema: &SchemaRef,
    conversions: &[VectorConversion],
) -> Result<SchemaRef, String> {
    if conversions.is_empty() {
        return Ok(input_schema.clone());
    }
    let mut fields = input_schema.fields().as_ref().to_vec();
    for conv in conversions {
        let idx = conv.col_idx;
        if idx >= fields.len() {
            return Err(format!(
                "vector column '{}': column index is out of bounds",
                conv.field_name
            ));
        }
        let original = fields[idx].as_ref();
        let (list_kind, element_type) = is_variable_list_vector_type(original.data_type())
            .ok_or_else(|| {
                format!(
                    "vector column '{}': unexpected input data type",
                    conv.field_name
                )
            })?;
        if list_kind != conv.list_kind || element_type != conv.element_type {
            return Err(format!(
                "vector column '{}': unexpected input data type",
                conv.field_name
            ));
        }
        let child_field = match original.data_type() {
            DataType::List(field) | DataType::LargeList(field) => field.clone(),
            _ => {
                return Err(format!(
                    "vector column '{}': unexpected input data type",
                    conv.field_name
                ))
            }
        };
        let dim_i32 = i32::try_from(conv.dim).map_err(|_| {
            format!(
                "vector column '{}': dimension is too large",
                conv.field_name
            )
        })?;
        fields[idx] = Arc::new(Field::new(
            original.name(),
            DataType::FixedSizeList(child_field, dim_i32),
            original.is_nullable(),
        ));
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn convert_record_batch(
    input_batch: &RecordBatch,
    output_schema: &SchemaRef,
    conversions: &[VectorConversion],
) -> Result<RecordBatch, String> {
    if conversions.is_empty() {
        return RecordBatch::try_new(output_schema.clone(), input_batch.columns().to_vec())
            .map_err(|e| e.to_string());
    }
    let mut cols = input_batch.columns().to_vec();
    for conv in conversions {
        let arr = cols
            .get(conv.col_idx)
            .ok_or_else(|| {
                format!(
                    "vector column '{}': column index is out of bounds",
                    conv.field_name
                )
            })?
            .as_ref();
        validate_list_vector_dim(arr, conv.list_kind, conv.dim)
            .map_err(|err| format!("vector column '{}': {err}", conv.field_name))?;
        let fixed =
            convert_list_array_to_fixed_size(arr, conv.list_kind, conv.element_type, conv.dim)
                .map_err(|err| format!("vector column '{}': {err}", conv.field_name))?;
        cols[conv.col_idx] = Arc::new(fixed);
    }
    RecordBatch::try_new(output_schema.clone(), cols).map_err(|e| e.to_string())
}

fn spawn_writer_thread(
    kind: WriterKind,
    path: String,
    params: WriteParams,
    schema: SchemaRef,
    receiver: Receiver<RecordBatch>,
    aborted: Arc<AtomicBool>,
) -> JoinHandle<Result<WriterResult, String>> {
    std::thread::spawn(move || -> Result<WriterResult, String> {
        let reader = ReceiverRecordBatchReader::new(schema, receiver, aborted);
        match kind {
            WriterKind::Committed => {
                let fut = Dataset::write(reader, &path, Some(params));
                match runtime::block_on(fut) {
                    Ok(Ok(_)) => Ok(WriterResult::Committed),
                    Ok(Err(err)) => Err(err.to_string()),
                    Err(err) => Err(format!("runtime: {err}")),
                }
            }
            WriterKind::Uncommitted => {
                let source: Box<dyn RecordBatchReader + Send> = Box::new(reader);
                let builder = InsertBuilder::new(path.as_str()).with_params(&params);
                let fut = builder.execute_uncommitted_stream(source);
                match runtime::block_on(fut) {
                    Ok(Ok(txn)) => Ok(WriterResult::Uncommitted(Box::new(txn))),
                    Ok(Err(err)) => Err(err.to_string()),
                    Err(err) => Err(format!("runtime: {err}")),
                }
            }
        }
    })
}

fn writer_channel_failure(
    join: Option<JoinHandle<Result<WriterResult, String>>>,
    code: ErrorCode,
) -> FfiError {
    match join.map(JoinHandle::join) {
        Some(Ok(Err(message))) => FfiError::new(code, message),
        Some(Err(_)) => FfiError::new(code, "writer thread panicked"),
        Some(Ok(Ok(_))) => FfiError::new(
            code,
            "writer background task exited before accepting all input",
        ),
        None => FfiError::new(code, "writer background task exited"),
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_open_writer_with_storage_options(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    data_storage_version: *const c_char,
    vector_dims: *const c_char,
    infer_vector_dims: u8,
    session: *mut c_void,
    schema: *const c_void,
) -> *mut c_void {
    match open_writer_inner(
        path,
        mode,
        option_keys,
        option_values,
        options_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        data_storage_version,
        vector_dims,
        infer_vector_dims,
        session,
        schema,
    ) {
        Ok(handle) => {
            clear_last_error();
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
pub unsafe extern "C" fn lance_open_uncommitted_writer_with_storage_options(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    data_storage_version: *const c_char,
    vector_dims: *const c_char,
    infer_vector_dims: u8,
    session: *mut c_void,
    schema: *const c_void,
) -> *mut c_void {
    match open_uncommitted_writer_inner(
        path,
        mode,
        option_keys,
        option_values,
        options_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        data_storage_version,
        vector_dims,
        infer_vector_dims,
        session,
        schema,
    ) {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn parse_data_storage_version_arg(
    data_storage_version: *const c_char,
) -> FfiResult<Option<String>> {
    if data_storage_version.is_null() {
        return Ok(None);
    }

    let raw = unsafe { cstr_to_str(data_storage_version, "data_storage_version")? };
    let token = raw.trim();
    if token.is_empty() {
        return Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            "data_storage_version cannot be empty",
        ));
    }

    let lower = token.to_ascii_lowercase();
    let normalized = match lower.as_str() {
        "v2_0" | "v2.0" | "2_0" => "2.0".to_string(),
        "v2_1" | "v2.1" | "2_1" => "2.1".to_string(),
        "v2_2" | "v2.2" | "2_2" => "2.2".to_string(),
        _ => lower,
    };
    Ok(Some(normalized))
}

#[allow(clippy::too_many_arguments)]
fn open_uncommitted_writer_inner(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    data_storage_version: *const c_char,
    vector_dims: *const c_char,
    infer_vector_dims: u8,
    session: *mut c_void,
    schema: *const c_void,
) -> FfiResult<WriterHandle> {
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let mode = unsafe { cstr_to_str(mode, "mode")? };

    if schema.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "schema is null"));
    }

    if options_len > 0 && (option_keys.is_null() || option_values.is_null()) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        ));
    }

    let keys = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_keys, options_len, "option_keys")? }
    };
    let values = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_values, options_len, "option_values")? }
    };

    let mut storage_options = HashMap::<String, String>::new();
    for (idx, (&key_ptr, &val_ptr)) in keys.iter().zip(values.iter()).enumerate() {
        if key_ptr.is_null() || val_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("option key/value is null at index {idx}"),
            ));
        }
        let key = unsafe { CStr::from_ptr(key_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_keys[{idx}] utf8: {err}"))
        })?;
        let value = unsafe { CStr::from_ptr(val_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_values[{idx}] utf8: {err}"))
        })?;
        storage_options.insert(key.to_string(), value.to_string());
    }

    let ffi_schema = unsafe { &*(schema as *const arrow_schema::ffi::FFI_ArrowSchema) };
    let data_type = DataType::try_from(ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::DatasetWriteOpen, format!("schema import: {err}"))
    })?;
    let DataType::Struct(fields) = &data_type else {
        return Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            "schema must be a struct",
        ));
    };
    let schema: SchemaRef = std::sync::Arc::new(Schema::new(fields.clone()));

    let write_mode = WriteMode::try_from(mode).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid write mode '{mode}': {err}"),
        )
    })?;

    let max_rows_per_file = usize::try_from(max_rows_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_rows_per_file: {err}"),
        )
    })?;
    let max_rows_per_group = usize::try_from(max_rows_per_group).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_rows_per_group: {err}"),
        )
    })?;
    let max_bytes_per_file = usize::try_from(max_bytes_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_bytes_per_file: {err}"),
        )
    })?;
    let data_storage_version = parse_data_storage_version_arg(data_storage_version)?
        .map(|value| {
            value.parse().map_err(|err| {
                FfiError::new(
                    ErrorCode::DatasetWriteOpen,
                    format!("invalid data_storage_version '{value}': {err}"),
                )
            })
        })
        .transpose()?;
    let session = unsafe { optional_session_handle(session)? };

    preflight_create_target(&path, write_mode, &storage_options, session.clone())?;

    let mut store_params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        store_params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options),
        ));
    }

    let params = WriteParams {
        mode: write_mode,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        data_storage_version,
        session,
        store_params: Some(store_params),
        ..Default::default()
    };
    let vector_candidates = parse_vector_candidates(&schema, vector_dims, infer_vector_dims != 0)?;

    let aborted = Arc::new(AtomicBool::new(false));
    Ok(WriterHandle {
        input_schema: schema.clone(),
        data_type,
        state: Mutex::new(WriterState {
            kind: WriterKind::Uncommitted,
            path,
            params,
            vector_candidates,
            buffered_batches: Vec::new(),
            buffered_rows: 0,
            output_schema: None,
            output_sender: None,
            output_join: None,
        }),
        batches_sent: AtomicU64::new(0),
        aborted,
    })
}

#[allow(clippy::too_many_arguments)]
fn open_writer_inner(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    data_storage_version: *const c_char,
    vector_dims: *const c_char,
    infer_vector_dims: u8,
    session: *mut c_void,
    schema: *const c_void,
) -> FfiResult<WriterHandle> {
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let mode = unsafe { cstr_to_str(mode, "mode")? };

    if schema.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "schema is null"));
    }

    if options_len > 0 && (option_keys.is_null() || option_values.is_null()) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        ));
    }

    let keys = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_keys, options_len, "option_keys")? }
    };
    let values = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_values, options_len, "option_values")? }
    };

    let mut storage_options = HashMap::<String, String>::new();
    for (idx, (&key_ptr, &val_ptr)) in keys.iter().zip(values.iter()).enumerate() {
        if key_ptr.is_null() || val_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("option key/value is null at index {idx}"),
            ));
        }
        let key = unsafe { CStr::from_ptr(key_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_keys[{idx}] utf8: {err}"))
        })?;
        let value = unsafe { CStr::from_ptr(val_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_values[{idx}] utf8: {err}"))
        })?;
        storage_options.insert(key.to_string(), value.to_string());
    }

    let ffi_schema = unsafe { &*(schema as *const arrow_schema::ffi::FFI_ArrowSchema) };
    let data_type = DataType::try_from(ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::DatasetWriteOpen, format!("schema import: {err}"))
    })?;
    let DataType::Struct(fields) = &data_type else {
        return Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            "schema must be a struct",
        ));
    };
    let schema: SchemaRef = std::sync::Arc::new(Schema::new(fields.clone()));

    let write_mode = WriteMode::try_from(mode).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid write mode '{mode}': {err}"),
        )
    })?;

    let max_rows_per_file = usize::try_from(max_rows_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_rows_per_file: {err}"),
        )
    })?;
    let max_rows_per_group = usize::try_from(max_rows_per_group).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_rows_per_group: {err}"),
        )
    })?;
    let max_bytes_per_file = usize::try_from(max_bytes_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_bytes_per_file: {err}"),
        )
    })?;
    let data_storage_version = parse_data_storage_version_arg(data_storage_version)?
        .map(|value| {
            value.parse().map_err(|err| {
                FfiError::new(
                    ErrorCode::DatasetWriteOpen,
                    format!("invalid data_storage_version '{value}': {err}"),
                )
            })
        })
        .transpose()?;
    let session = unsafe { optional_session_handle(session)? };

    preflight_create_target(&path, write_mode, &storage_options, session.clone())?;

    let mut store_params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        store_params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options),
        ));
    }

    let params = WriteParams {
        mode: write_mode,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        data_storage_version,
        session,
        store_params: Some(store_params),
        ..Default::default()
    };

    let vector_candidates = parse_vector_candidates(&schema, vector_dims, infer_vector_dims != 0)?;

    let aborted = Arc::new(AtomicBool::new(false));
    Ok(WriterHandle {
        input_schema: schema.clone(),
        data_type,
        state: Mutex::new(WriterState {
            kind: WriterKind::Committed,
            path,
            params,
            vector_candidates,
            buffered_batches: Vec::new(),
            buffered_rows: 0,
            output_schema: None,
            output_sender: None,
            output_join: None,
        }),
        batches_sent: AtomicU64::new(0),
        aborted,
    })
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_writer_write_batch(writer: *mut c_void, array: *mut c_void) -> i32 {
    match writer_write_batch_inner(writer, array) {
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

fn writer_write_batch_inner(writer: *mut c_void, array: *mut c_void) -> FfiResult<()> {
    if writer.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "writer is null"));
    }
    if array.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "array is null"));
    }

    let handle = unsafe { &*(writer as *const WriterHandle) };

    let raw_array = unsafe { ptr::read(array as *mut RawArrowArray) };
    unsafe {
        (*(array as *mut RawArrowArray)).release = None;
    }

    let ffi_array: arrow::ffi::FFI_ArrowArray = unsafe { std::mem::transmute(raw_array) };

    let array_data =
        unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, handle.data_type.clone()) }
            .map_err(|err| {
                FfiError::new(ErrorCode::DatasetWriteBatch, format!("array import: {err}"))
            })?;
    let array = make_array(array_data);
    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| FfiError::new(ErrorCode::DatasetWriteBatch, "array is not a struct"))?;

    let input_batch =
        RecordBatch::try_new(handle.input_schema.clone(), struct_array.columns().to_vec())
            .map_err(|err| {
                FfiError::new(ErrorCode::DatasetWriteBatch, format!("record batch: {err}"))
            })?;

    let (sender, to_send) = {
        let mut guard = handle
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Once the writer starts, its physical schema cannot change.  A value
        // that appears only after the bounded inference window must therefore
        // fail instead of silently producing a non-vector LIST schema.
        if guard.output_sender.is_some() {
            for candidate in guard.vector_candidates.iter_mut() {
                if candidate.dim != 0 {
                    continue;
                }
                let array = input_batch.column(candidate.col_idx).as_ref();
                match infer_vector_dim_from_array(array, candidate.list_kind) {
                    Some(Ok(dim)) => {
                        return Err(FfiError::new(
                            ErrorCode::DatasetWriteBatch,
                            format!(
                                "vector column '{}' first non-NULL value (dimension {dim}) appeared after the {}-row inference window; set VECTOR_DIMS to an explicit JSON mapping, use a fixed-size ARRAY type, or set INFER_VECTOR_DIMS false for a ragged list",
                                candidate.field_name, MAX_VECTOR_DIM_INFERENCE_ROWS
                            ),
                        ));
                    }
                    Some(Err(message)) => {
                        return Err(FfiError::new(
                            ErrorCode::DatasetWriteBatch,
                            format!("vector column '{}': {message}", candidate.field_name),
                        ));
                    }
                    None => {}
                }
            }
        }

        if guard.output_sender.is_none() {
            guard.buffered_rows = guard
                .buffered_rows
                .checked_add(input_batch.num_rows())
                .ok_or_else(|| {
                    FfiError::new(
                        ErrorCode::DatasetWriteBatch,
                        "vector inference buffered row count overflow",
                    )
                })?;
            guard.buffered_batches.push(input_batch);

            if !guard.vector_candidates.is_empty() {
                let batches = guard.buffered_batches.clone();
                for cand in guard.vector_candidates.iter_mut() {
                    if cand.dim != 0 {
                        continue;
                    }
                    for batch in batches.iter() {
                        let arr = batch.column(cand.col_idx).as_ref();
                        match infer_vector_dim_from_array(arr, cand.list_kind) {
                            Some(Ok(dim)) => {
                                cand.dim = dim;
                                break;
                            }
                            Some(Err(e)) => {
                                return Err(FfiError::new(
                                    ErrorCode::DatasetWriteBatch,
                                    format!("vector column '{}': {e}", cand.field_name),
                                ));
                            }
                            None => {}
                        }
                    }
                }

                let batches = guard.buffered_batches.clone();
                for cand in guard.vector_candidates.iter() {
                    if cand.dim == 0 {
                        continue;
                    }
                    for batch in batches.iter() {
                        let arr = batch.column(cand.col_idx).as_ref();
                        if let Err(e) = validate_list_vector_dim(arr, cand.list_kind, cand.dim) {
                            return Err(FfiError::new(
                                ErrorCode::DatasetWriteBatch,
                                format!("vector column '{}': {e}", cand.field_name),
                            ));
                        }
                    }
                }
            }

            let can_start = guard.vector_candidates.iter().all(|c| c.dim != 0)
                || guard.buffered_rows >= MAX_VECTOR_DIM_INFERENCE_ROWS;
            if can_start {
                let conversions: Vec<VectorConversion> = guard
                    .vector_candidates
                    .iter()
                    .filter(|c| c.dim != 0)
                    .cloned()
                    .collect();

                let output_schema = build_output_schema(&handle.input_schema, &conversions)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
                let (sender, receiver) = sync_channel::<RecordBatch>(2);
                let join = spawn_writer_thread(
                    guard.kind,
                    guard.path.clone(),
                    guard.params.clone(),
                    output_schema.clone(),
                    receiver,
                    handle.aborted.clone(),
                );

                let buffered = std::mem::take(&mut guard.buffered_batches);
                guard.buffered_rows = 0;
                let mut out_batches = Vec::with_capacity(buffered.len());
                for b in buffered.iter() {
                    let out = convert_record_batch(b, &output_schema, &conversions)
                        .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
                    out_batches.push(out);
                }

                guard.output_schema = Some(output_schema);
                guard.output_sender = Some(sender.clone());
                guard.output_join = Some(join);
                (Some(sender), out_batches)
            } else {
                (None, Vec::new())
            }
        } else {
            let sender = guard.output_sender.as_ref().cloned();
            let schema = guard
                .output_schema
                .as_ref()
                .ok_or_else(|| {
                    FfiError::new(ErrorCode::DatasetWriteBatch, "writer is not initialized")
                })?
                .clone();
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let out = convert_record_batch(&input_batch, &schema, &conversions)
                .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
            (sender, vec![out])
        }
    };

    if let Some(sender) = sender {
        for batch in to_send {
            if sender.send(batch).is_err() {
                drop(sender);
                let join = {
                    let mut guard = handle
                        .state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    guard.output_sender = None;
                    guard.output_join.take()
                };
                return Err(writer_channel_failure(join, ErrorCode::DatasetWriteBatch));
            }
        }
    }

    handle.batches_sent.fetch_add(1, Ordering::Relaxed);

    Ok(())
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_writer_finish(writer: *mut c_void) -> i32 {
    match writer_finish_inner(writer) {
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

fn writer_finish_inner(writer: *mut c_void) -> FfiResult<()> {
    if writer.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "writer is null"));
    }

    let handle = unsafe { &*(writer as *const WriterHandle) };
    let (sender, join, to_send) = {
        let mut guard = handle
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.output_sender.is_none() {
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let output_schema = build_output_schema(&handle.input_schema, &conversions)
                .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinish, e))?;
            let (sender, receiver) = sync_channel::<RecordBatch>(2);
            let join = spawn_writer_thread(
                guard.kind,
                guard.path.clone(),
                guard.params.clone(),
                output_schema.clone(),
                receiver,
                handle.aborted.clone(),
            );
            let buffered = std::mem::take(&mut guard.buffered_batches);
            guard.buffered_rows = 0;
            let mut out_batches = Vec::with_capacity(buffered.len() + 1);
            for b in buffered.iter() {
                let out = convert_record_batch(b, &output_schema, &conversions)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinish, e))?;
                out_batches.push(out);
            }
            if handle.batches_sent.load(Ordering::Acquire) == 0 {
                out_batches.push(RecordBatch::new_empty(output_schema.clone()));
            }
            guard.output_schema = Some(output_schema);
            guard.output_sender = Some(sender.clone());
            guard.output_join = Some(join);
            guard.buffered_batches = out_batches;
        }

        let sender = guard.output_sender.as_ref().cloned().ok_or_else(|| {
            FfiError::new(ErrorCode::DatasetWriteFinish, "writer is not initialized")
        })?;
        let join = guard.output_join.take().ok_or_else(|| {
            FfiError::new(ErrorCode::DatasetWriteFinish, "writer is already finished")
        })?;
        let to_send = std::mem::take(&mut guard.buffered_batches);
        guard.output_sender = None;
        (sender, join, to_send)
    };

    for b in to_send {
        if sender.send(b).is_err() {
            drop(sender);
            return Err(writer_channel_failure(
                Some(join),
                ErrorCode::DatasetWriteFinish,
            ));
        }
    }
    drop(sender);

    match join.join() {
        Ok(Ok(WriterResult::Committed)) => Ok(()),
        Ok(Ok(WriterResult::Uncommitted(_))) => Err(FfiError::new(
            ErrorCode::DatasetWriteFinish,
            "writer returned an uncommitted transaction",
        )),
        Ok(Err(message)) => Err(FfiError::new(ErrorCode::DatasetWriteFinish, message)),
        Err(_) => Err(FfiError::new(
            ErrorCode::DatasetWriteFinish,
            "writer thread panicked",
        )),
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_writer_finish_uncommitted(
    writer: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> i32 {
    match writer_finish_uncommitted_inner(writer, out_transaction) {
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

fn writer_finish_uncommitted_inner(
    writer: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> FfiResult<()> {
    if writer.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "writer is null"));
    }
    if out_transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction is null",
        ));
    }

    let handle = unsafe { &*(writer as *const WriterHandle) };
    let (sender, join, to_send, null_vector_fields) = {
        let mut guard = handle
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.output_sender.is_none() {
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let output_schema = build_output_schema(&handle.input_schema, &conversions)
                .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinishUncommitted, e))?;
            let (sender, receiver) = sync_channel::<RecordBatch>(2);
            let join = spawn_writer_thread(
                guard.kind,
                guard.path.clone(),
                guard.params.clone(),
                output_schema.clone(),
                receiver,
                handle.aborted.clone(),
            );
            let buffered = std::mem::take(&mut guard.buffered_batches);
            guard.buffered_rows = 0;
            let mut out_batches = Vec::with_capacity(buffered.len() + 1);
            for b in buffered.iter() {
                let out = convert_record_batch(b, &output_schema, &conversions)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinishUncommitted, e))?;
                out_batches.push(out);
            }
            if handle.batches_sent.load(Ordering::Acquire) == 0 {
                out_batches.push(RecordBatch::new_empty(output_schema.clone()));
            }
            guard.output_schema = Some(output_schema);
            guard.output_sender = Some(sender.clone());
            guard.output_join = Some(join);
            guard.buffered_batches = out_batches;
        }

        let sender = guard.output_sender.as_ref().cloned().ok_or_else(|| {
            FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer is not initialized",
            )
        })?;
        let join = guard.output_join.take().ok_or_else(|| {
            FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer is already finished",
            )
        })?;
        let to_send = std::mem::take(&mut guard.buffered_batches);
        let null_vector_fields = guard
            .vector_candidates
            .iter()
            .filter(|candidate| candidate.dim == 0)
            .map(|candidate| handle.input_schema.field(candidate.col_idx).name().clone())
            .collect::<Vec<_>>();
        guard.output_sender = None;
        (sender, join, to_send, null_vector_fields)
    };

    for b in to_send {
        if sender.send(b).is_err() {
            drop(sender);
            return Err(writer_channel_failure(
                Some(join),
                ErrorCode::DatasetWriteFinishUncommitted,
            ));
        }
    }
    drop(sender);

    let mut txn = match join.join() {
        Ok(Ok(WriterResult::Uncommitted(txn))) => txn,
        Ok(Ok(WriterResult::Committed)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer did not return an uncommitted transaction",
            ))
        }
        Ok(Err(message)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                message,
            ))
        }
        Err(_) => {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer thread panicked",
            ))
        }
    };

    if !null_vector_fields.is_empty() {
        let encoded = serde_json::to_string(&null_vector_fields).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                format!("serialize all-NULL vector fields: {err}"),
            )
        })?;
        let mut properties = txn
            .transaction_properties
            .as_deref()
            .cloned()
            .unwrap_or_default();
        properties.insert(VANE_NULL_VECTOR_FIELDS_PROPERTY.to_string(), encoded);
        txn.transaction_properties = Some(Arc::new(properties));
    }

    unsafe {
        *out_transaction = Box::into_raw(txn) as *mut c_void;
    }

    Ok(())
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_close_writer(writer: *mut c_void) {
    if writer.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(writer as *mut WriterHandle);
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_commit_transaction_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    transaction: *mut c_void,
) -> i32 {
    match commit_transaction_inner(
        path,
        option_keys,
        option_values,
        options_len,
        session,
        transaction,
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

fn commit_transaction_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    transaction: *mut c_void,
) -> FfiResult<()> {
    if transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "transaction is null",
        ));
    }
    // The C ABI transfers ownership on every non-null call, including calls
    // that fail validation.  Consume first so malformed paths/options cannot
    // leak a transaction and so the C++ ownership contract stays symmetric.
    let txn = unsafe { Box::from_raw(transaction as *mut Transaction) };
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();

    if options_len > 0 && (option_keys.is_null() || option_values.is_null()) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        ));
    }

    let keys = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_keys, options_len, "option_keys")? }
    };
    let values = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_values, options_len, "option_values")? }
    };

    let mut storage_options = HashMap::<String, String>::new();
    for (idx, (&key_ptr, &val_ptr)) in keys.iter().zip(values.iter()).enumerate() {
        if key_ptr.is_null() || val_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("option key/value is null at index {idx}"),
            ));
        }
        let key = unsafe { CStr::from_ptr(key_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_keys[{idx}] utf8: {err}"))
        })?;
        let value = unsafe { CStr::from_ptr(val_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_values[{idx}] utf8: {err}"))
        })?;
        storage_options.insert(key.to_string(), value.to_string());
    }

    let mut store_params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        store_params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options),
        ));
    }
    let session = unsafe { optional_session_handle(session)? };

    let mut builder = CommitBuilder::new(path.as_str()).with_store_params(store_params);
    if let Some(session) = session {
        builder = builder.with_session(session);
    }
    let fut = builder.execute(*txn);
    match runtime::block_on(fut) {
        Ok(Ok(_)) => {
            record_commit();
            Ok(())
        }
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            err.to_string(),
        )),
        Err(err) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            format!("runtime: {err}"),
        )),
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_free_transaction(transaction: *mut c_void) {
    if transaction.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(transaction as *mut lance::dataset::transaction::Transaction);
    }
}

fn collect_fragment_owned_paths(fragment: &Fragment, paths: &mut HashSet<String>) {
    for file in &fragment.files {
        if file.base_id.is_none() {
            paths.insert(format!("data/{}", file.path));
        }
    }
    for overlay in &fragment.overlays {
        if overlay.data_file.base_id.is_none() {
            paths.insert(format!("data/{}", overlay.data_file.path));
        }
    }
    if let Some(deletion_file) = &fragment.deletion_file {
        if deletion_file.base_id.is_none() {
            paths.insert(relative_deletion_file_path(fragment.id, deletion_file));
        }
    }
    if let Some(RowIdMeta::External(file)) = &fragment.row_id_meta {
        paths.insert(file.path.clone());
    }
    for version_meta in [
        fragment.last_updated_at_version_meta.as_ref(),
        fragment.created_at_version_meta.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if let RowDatasetVersionMeta::External(file) = version_meta {
            paths.insert(file.path.clone());
        }
    }
}

fn collect_transaction_owned_paths(transaction: &Transaction) -> HashSet<String> {
    let fragments: Vec<&Fragment> = match &transaction.operation {
        Operation::Append { fragments } | Operation::Overwrite { fragments, .. } => {
            fragments.iter().collect()
        }
        Operation::Delete {
            updated_fragments, ..
        } => updated_fragments.iter().collect(),
        Operation::Update {
            updated_fragments,
            new_fragments,
            ..
        } => updated_fragments.iter().chain(new_fragments).collect(),
        Operation::Rewrite { groups, .. } => groups
            .iter()
            .flat_map(|group| group.new_fragments.iter())
            .collect(),
        Operation::Merge { fragments, .. } => fragments.iter().collect(),
        _ => Vec::new(),
    };
    let mut paths = HashSet::new();
    for fragment in fragments {
        collect_fragment_owned_paths(fragment, &mut paths);
    }
    paths
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_abort_transaction_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    transaction: *mut c_void,
) -> i32 {
    match abort_transaction_with_storage_options_inner(
        path,
        option_keys,
        option_values,
        options_len,
        session,
        transaction,
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

fn abort_transaction_with_storage_options_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    transaction: *mut c_void,
) -> FfiResult<()> {
    if transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "transaction is null",
        ));
    }
    // Match commit ownership: every non-null call consumes the transaction,
    // even when a later argument is invalid.
    let transaction = unsafe { Box::from_raw(transaction as *mut Transaction) };
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let storage_options =
        unsafe { distributed_storage_options_from_ffi(option_keys, option_values, options_len)? };
    let session = unsafe { optional_session_handle(session)? };
    let candidate_paths = collect_transaction_owned_paths(&transaction);
    if candidate_paths.is_empty() {
        return Ok(());
    }

    match runtime::block_on(async {
        let dataset =
            load_optional_distributed_dataset(&path, &storage_options, session.clone()).await?;
        let (store, base) =
            resolve_distributed_object_store(&path, &storage_options, session.as_ref()).await?;
        let mut referenced_paths = HashSet::new();
        if let Some(dataset) = dataset.as_ref() {
            for fragment in dataset.manifest().fragments.iter() {
                collect_fragment_owned_paths(fragment, &mut referenced_paths);
            }
        }
        for relative in candidate_paths.difference(&referenced_paths) {
            let object = path_join(&base, relative)?;
            match store.delete(&object).await {
                Ok(()) => {}
                Err(error) if error.is_not_found() => {}
                Err(error) => return Err(error),
            }
        }
        Ok::<(), lance::Error>(())
    }) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            format!("abort uncommitted Lance transaction: {error}"),
        )),
        Err(error) => Err(FfiError::new(
            ErrorCode::Runtime,
            format!("runtime: {error}"),
        )),
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_serialize_transaction(
    transaction: *mut c_void,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    match serialize_transaction_inner(transaction, out_data, out_len) {
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

fn serialize_transaction_inner(
    transaction: *mut c_void,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> FfiResult<()> {
    if transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "transaction is null",
        ));
    }
    if out_data.is_null() || out_len.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_data/out_len is null",
        ));
    }

    let transaction = unsafe { &*(transaction as *const Transaction) };
    let message = pb::Transaction::from(transaction);
    let bytes = message.encode_to_vec().into_boxed_slice();
    let len = bytes.len();
    let data = Box::into_raw(bytes) as *mut u8;
    unsafe {
        *out_data = data;
        *out_len = len;
    }
    Ok(())
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_free_bytes(data: *mut u8, len: usize) {
    if data.is_null() {
        return;
    }
    unsafe {
        let slice = ptr::slice_from_raw_parts_mut(data, len);
        let _ = Box::from_raw(slice);
    }
}

unsafe fn distributed_storage_options_from_ffi(
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> FfiResult<HashMap<String, String>> {
    if options_len > 0 && (option_keys.is_null() || option_values.is_null()) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        ));
    }
    let keys = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_keys, options_len, "option_keys")? }
    };
    let values = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_values, options_len, "option_values")? }
    };

    let mut storage_options = HashMap::with_capacity(options_len);
    for (idx, (&key_ptr, &value_ptr)) in keys.iter().zip(values.iter()).enumerate() {
        if key_ptr.is_null() || value_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("option key/value is null at index {idx}"),
            ));
        }
        let key = unsafe { CStr::from_ptr(key_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_keys[{idx}] utf8: {err}"))
        })?;
        let value = unsafe { CStr::from_ptr(value_ptr) }
            .to_str()
            .map_err(|err| {
                FfiError::new(ErrorCode::Utf8, format!("option_values[{idx}] utf8: {err}"))
            })?;
        storage_options.insert(key.to_string(), value.to_string());
    }
    Ok(storage_options)
}

fn distributed_store_params(storage_options: &HashMap<String, String>) -> ObjectStoreParams {
    let mut params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options.clone()),
        ));
    }
    params
}

async fn load_optional_distributed_dataset(
    path: &str,
    storage_options: &HashMap<String, String>,
    session: Option<Arc<Session>>,
) -> Result<Option<Dataset>, lance::Error> {
    let mut builder = DatasetBuilder::from_uri(path).with_storage_options(storage_options.clone());
    if let Some(session) = session {
        builder = builder.with_session(session);
    }
    match builder.load().await {
        Ok(dataset) => Ok(Some(dataset)),
        Err(lance::Error::DatasetNotFound { .. } | lance::Error::NotFound { .. }) => Ok(None),
        Err(err) if err.is_not_found() => Ok(None),
        Err(err) => Err(err),
    }
}

fn preflight_create_target(
    path: &str,
    mode: WriteMode,
    storage_options: &HashMap<String, String>,
    session: Option<Arc<Session>>,
) -> FfiResult<()> {
    if !matches!(mode, WriteMode::Create) {
        return Ok(());
    }
    match runtime::block_on(load_optional_distributed_dataset(
        path,
        storage_options,
        session,
    )) {
        Ok(Ok(Some(_))) => Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("dataset already exists: {path}"),
        )),
        Ok(Ok(None)) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("dataset create preflight '{path}': {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DistributedOperationMarker {
    format_version: u32,
    operation_id: String,
    row_count: u64,
    write_mode: String,
}

#[derive(Debug)]
enum DistributedCommitError {
    Known(lance::Error),
    OutcomeUnknown(String),
}

impl From<lance::Error> for DistributedCommitError {
    fn from(value: lance::Error) -> Self {
        Self::Known(value)
    }
}

fn is_definitive_commit_failure(error: &lance::Error) -> bool {
    matches!(
        error,
        lance::Error::InvalidInput { .. }
            | lance::Error::DatasetAlreadyExists { .. }
            | lance::Error::SchemaMismatch { .. }
            | lance::Error::CommitConflict { .. }
            | lance::Error::IncompatibleTransaction { .. }
    )
}

fn validate_distributed_operation_marker(
    marker: &DistributedOperationMarker,
    operation_id: &str,
    selected_rows: u64,
    mode: &str,
) -> Result<u64, lance::Error> {
    if marker.row_count != selected_rows {
        return Err(lance::Error::internal(format!(
            "distributed Lance operation '{operation_id}' was already committed with {} rows, not {selected_rows}",
            marker.row_count
        )));
    }
    if marker.write_mode != mode {
        return Err(lance::Error::internal(format!(
            "distributed Lance operation '{operation_id}' was already committed in '{}' mode, not '{mode}'",
            marker.write_mode
        )));
    }
    Ok(marker.row_count)
}

async fn find_committed_distributed_operation_in_history(
    dataset: Option<&Dataset>,
    operation_id: &str,
) -> Result<Option<DistributedOperationMarker>, lance::Error> {
    let Some(dataset) = dataset else {
        return Ok(None);
    };
    let transaction_count = usize::try_from(dataset.version_id()).unwrap_or(usize::MAX);
    for transaction in dataset.get_transactions(transaction_count).await? {
        let Some(transaction) = transaction else {
            continue;
        };
        let Some(properties) = transaction.transaction_properties else {
            continue;
        };
        if properties
            .get(VANE_OPERATION_ID_PROPERTY)
            .is_some_and(|value| value == operation_id)
        {
            let value = properties.get(VANE_ROW_COUNT_PROPERTY).ok_or_else(|| {
                lance::Error::internal(format!(
                    "distributed Lance transaction for operation '{operation_id}' has no row count"
                ))
            })?;
            let row_count = value.parse::<u64>().map_err(|err| {
                lance::Error::internal(format!(
                    "invalid distributed Lance row count '{value}' for operation '{operation_id}': {err}"
                ))
            })?;
            let write_mode = properties
                .get(VANE_WRITE_MODE_PROPERTY)
                .ok_or_else(|| {
                    lance::Error::internal(format!(
                        "distributed Lance transaction for operation '{operation_id}' has no write mode"
                    ))
                })?
                .clone();
            return Ok(Some(DistributedOperationMarker {
                format_version: VANE_OPERATION_MARKER_FORMAT_VERSION,
                operation_id: operation_id.to_string(),
                row_count,
                write_mode,
            }));
        }
    }
    Ok(None)
}

fn validate_distributed_write_mode(mode: &str) -> FfiResult<()> {
    if matches!(mode, "create" | "append" | "overwrite") {
        Ok(())
    } else {
        Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("invalid distributed Lance write mode '{mode}'"),
        ))
    }
}

fn hex_identity(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

fn path_join(base: &Path, suffix: &str) -> Result<Path, lance::Error> {
    let joined = if base.as_ref().is_empty() {
        suffix.to_string()
    } else if suffix.is_empty() {
        base.as_ref().to_string()
    } else {
        format!("{}/{suffix}", base.as_ref())
    };
    Path::parse(joined).map_err(Into::into)
}

fn distributed_operation_marker_path(
    base: &Path,
    operation_id: &str,
) -> Result<Path, lance::Error> {
    path_join(
        base,
        &format!("_vane_operations/{}.json", hex_identity(operation_id)),
    )
}

async fn read_distributed_operation_marker(
    store: &ObjectStore,
    base: &Path,
    operation_id: &str,
) -> Result<Option<DistributedOperationMarker>, lance::Error> {
    let path = distributed_operation_marker_path(base, operation_id)?;
    let bytes = match store.read_one_all(&path).await {
        Ok(bytes) => bytes,
        Err(err) if err.is_not_found() => return Ok(None),
        Err(err) => return Err(err),
    };
    let marker: DistributedOperationMarker = serde_json::from_slice(&bytes).map_err(|err| {
        lance::Error::internal(format!(
            "invalid distributed Lance operation marker '{}': {err}",
            path.as_ref()
        ))
    })?;
    if marker.format_version != VANE_OPERATION_MARKER_FORMAT_VERSION {
        return Err(lance::Error::internal(format!(
            "unsupported distributed Lance operation marker version {} for operation '{operation_id}'",
            marker.format_version
        )));
    }
    if marker.operation_id != operation_id {
        return Err(lance::Error::internal(format!(
            "distributed Lance operation marker identity mismatch for operation '{operation_id}'"
        )));
    }
    if !matches!(
        marker.write_mode.as_str(),
        "create" | "append" | "overwrite"
    ) {
        return Err(lance::Error::internal(format!(
            "invalid distributed Lance write mode '{}' in marker for operation '{operation_id}'",
            marker.write_mode
        )));
    }
    Ok(Some(marker))
}

async fn write_distributed_operation_marker(
    store: &ObjectStore,
    base: &Path,
    marker: &DistributedOperationMarker,
) -> Result<(), lance::Error> {
    let path = distributed_operation_marker_path(base, &marker.operation_id)?;
    let bytes = serde_json::to_vec(marker).map_err(|err| {
        lance::Error::internal(format!(
            "serialize distributed Lance operation marker for '{}': {err}",
            marker.operation_id
        ))
    })?;
    store.put(&path, &bytes).await?;
    Ok(())
}

fn dataset_references_distributed_operation(dataset: &Dataset, operation_id: &str) -> bool {
    let prefix = distributed_destination_operation_prefix(operation_id);
    dataset.manifest().fragments.iter().any(|fragment| {
        fragment
            .files
            .iter()
            .any(|file| file.base_id.is_none() && file.path.starts_with(&prefix))
    })
}

async fn find_committed_distributed_operation(
    dataset: Option<&Dataset>,
    store: &ObjectStore,
    base: &Path,
    operation_id: &str,
) -> Result<Option<DistributedOperationMarker>, lance::Error> {
    let Some(dataset) = dataset else {
        return Ok(None);
    };
    if let Some(marker) = read_distributed_operation_marker(store, base, operation_id).await? {
        return Ok(Some(marker));
    }
    // A new operation has neither a marker nor referenced destination files.
    // Avoid scanning every historical transaction in that overwhelmingly
    // common case; history is only a recovery path for an operation whose
    // deterministic files are visible in the current manifest.
    if !dataset_references_distributed_operation(dataset, operation_id) {
        return Ok(None);
    }
    if let Some(marker) =
        find_committed_distributed_operation_in_history(Some(dataset), operation_id).await?
    {
        // Backfill the durable marker while the vacuumable transaction is still
        // available.  A failure is fail-closed: callers must not replay or clean
        // up an operation whose commit outcome is already known.
        write_distributed_operation_marker(store, base, &marker).await?;
        return Ok(Some(marker));
    }
    Err(lance::Error::internal(format!(
        "distributed Lance operation '{operation_id}' has referenced data files but no durable commit marker; refusing unsafe replay"
    )))
}

async fn resolve_distributed_object_store(
    path: &str,
    storage_options: &HashMap<String, String>,
    session: Option<&Arc<Session>>,
) -> Result<(Arc<ObjectStore>, Path), lance::Error> {
    let registry = session
        .map(|session| session.store_registry())
        .unwrap_or_else(|| Arc::new(ObjectStoreRegistry::default()));
    ObjectStore::from_uri_and_params(registry, path, &distributed_store_params(storage_options))
        .await
}

async fn remove_distributed_dir_if_exists(
    store: &ObjectStore,
    path: &Path,
) -> Result<(), lance::Error> {
    match store.remove_dir_all(path.clone()).await {
        Ok(()) => Ok(()),
        Err(err) if err.is_not_found() => Ok(()),
        Err(err) => Err(err),
    }
}

fn distributed_staging_operation_path(
    base: &Path,
    operation_id: &str,
) -> Result<Path, lance::Error> {
    path_join(
        base,
        &format!("_vane_staging/{}", hex_identity(operation_id)),
    )
}

fn distributed_destination_operation_prefix(operation_id: &str) -> String {
    format!("vane_{}_", hex_identity(operation_id))
}

fn distributed_destination_task_prefix(operation_id: &str, task_attempt_id: &str) -> String {
    format!(
        "{}{}_",
        distributed_destination_operation_prefix(operation_id),
        hex_identity(task_attempt_id)
    )
}

async fn remove_distributed_destination_files(
    store: &ObjectStore,
    base: &Path,
    operation_id: &str,
    dataset: Option<&Dataset>,
) -> Result<(), lance::Error> {
    let data = path_join(base, "data")?;
    let prefix = distributed_destination_operation_prefix(operation_id);
    let referenced = dataset
        .into_iter()
        .flat_map(|dataset| dataset.manifest().fragments.iter())
        .flat_map(|fragment| fragment.files.iter())
        .filter(|file| file.base_id.is_none())
        .map(|file| file.path.as_str())
        .collect::<HashSet<_>>();
    let objects = match store.list(Some(data.clone())).try_collect::<Vec<_>>().await {
        Ok(objects) => objects,
        Err(err) if err.is_not_found() => return Ok(()),
        Err(err) => return Err(err),
    };
    let data_prefix = data.as_ref();
    for object in objects {
        let location = object.location.as_ref();
        let relative = location
            .strip_prefix(data_prefix)
            .and_then(|value| value.strip_prefix('/'))
            .ok_or_else(|| {
                lance::Error::internal(format!(
                    "listed Lance data object '{location}' is outside '{data_prefix}'"
                ))
            })?;
        if relative.starts_with(&prefix) && !referenced.contains(relative) {
            store.delete(&object.location).await?;
        }
    }
    Ok(())
}

async fn copy_distributed_task_data(
    store: &ObjectStore,
    base: &Path,
    operation_id: &str,
    task_attempt_id: &str,
) -> Result<(), lance::Error> {
    let operation_hex = hex_identity(operation_id);
    let task_hex = hex_identity(task_attempt_id);
    let source = path_join(
        base,
        &format!("_vane_staging/{operation_hex}/{task_hex}/data"),
    )?;
    let destination = path_join(base, "data")?;
    let destination_prefix = distributed_destination_task_prefix(operation_id, task_attempt_id);
    let objects = store
        .list(Some(source.clone()))
        .try_collect::<Vec<_>>()
        .await?;
    for object in objects {
        let source_prefix = source.as_ref();
        let location = object.location.as_ref();
        let relative = location
            .strip_prefix(source_prefix)
            .and_then(|value| value.strip_prefix('/'))
            .ok_or_else(|| {
                lance::Error::internal(format!(
                    "listed staging object '{location}' is outside '{source_prefix}'"
                ))
            })?;
        if relative.is_empty() {
            return Err(lance::Error::internal(format!(
                "staging object '{location}' has no relative path"
            )));
        }
        let destination_file = path_join(&destination, &format!("{destination_prefix}{relative}"))?;
        store.copy(&object.location, &destination_file).await?;
    }
    Ok(())
}

struct DistributedStagingTask {
    task_id: String,
    fragments: Vec<Fragment>,
    schema: LanceSchema,
    null_vector_fields: HashSet<String>,
}

fn distributed_null_vector_fields(
    transaction: &Transaction,
) -> Result<HashSet<String>, lance::Error> {
    let Some(encoded) = transaction
        .transaction_properties
        .as_deref()
        .and_then(|properties| properties.get(VANE_NULL_VECTOR_FIELDS_PROPERTY))
    else {
        return Ok(HashSet::new());
    };
    let fields: Vec<String> = serde_json::from_str(encoded).map_err(|err| {
        lance::Error::invalid_input(format!(
            "invalid distributed Lance all-NULL vector field metadata: {err}"
        ))
    })?;
    if fields.iter().any(String::is_empty) {
        return Err(lance::Error::invalid_input(
            "distributed Lance all-NULL vector field name cannot be empty".to_string(),
        ));
    }
    Ok(fields.into_iter().collect())
}

fn fixed_size_vector_type(field: &LanceField) -> Option<(VectorElementType, i32)> {
    let DataType::FixedSizeList(child, dimension) = field.data_type() else {
        return None;
    };
    let element_type = match child.data_type() {
        DataType::Float32 => VectorElementType::Float32,
        DataType::Float64 => VectorElementType::Float64,
        _ => return None,
    };
    Some((element_type, dimension))
}

fn variable_vector_type(field: &LanceField) -> Option<VectorElementType> {
    is_variable_list_vector_type(&field.data_type()).map(|(_, element_type)| element_type)
}

fn promote_distributed_vector_schema(canonical: &mut LanceSchema, candidate: &LanceSchema) {
    if canonical.fields.len() != candidate.fields.len() {
        return;
    }
    for (canonical_field, candidate_field) in
        canonical.fields.iter_mut().zip(candidate.fields.iter())
    {
        let Some(canonical_element) = variable_vector_type(canonical_field) else {
            continue;
        };
        let Some((candidate_element, dimension)) = fixed_size_vector_type(candidate_field) else {
            continue;
        };
        if canonical_element == candidate_element && dimension > 0 {
            *canonical_field = candidate_field.clone();
        }
    }
}

fn collect_field_id_mapping(
    source: &LanceField,
    canonical: &LanceField,
    mapping: &mut HashMap<i32, i32>,
) -> Result<(), lance::Error> {
    if source.name != canonical.name || source.children.len() != canonical.children.len() {
        return Err(lance::Error::invalid_input(format!(
            "cannot map distributed Lance field '{}' to canonical field '{}'",
            source.name, canonical.name
        )));
    }
    if mapping.insert(source.id, canonical.id).is_some() {
        return Err(lance::Error::invalid_input(format!(
            "duplicate distributed Lance source field id {}",
            source.id
        )));
    }
    for (source_child, canonical_child) in source.children.iter().zip(canonical.children.iter()) {
        collect_field_id_mapping(source_child, canonical_child, mapping)?;
    }
    Ok(())
}

fn collect_field_ids(field: &LanceField, field_ids: &mut HashSet<i32>) {
    field_ids.insert(field.id);
    for child in &field.children {
        collect_field_ids(child, field_ids);
    }
}

fn canonicalize_distributed_task_fragments(
    task: &mut DistributedStagingTask,
    canonical: &LanceSchema,
    compare_metadata: bool,
) -> Result<(), lance::Error> {
    for field_name in &task.null_vector_fields {
        let Some(field) = task.schema.field(field_name) else {
            return Err(lance::Error::invalid_input(format!(
                "worker transaction for task '{}' marks unknown all-NULL vector field '{}'",
                task.task_id, field_name
            )));
        };
        if variable_vector_type(field).is_none() {
            return Err(lance::Error::invalid_input(format!(
                "worker transaction for task '{}' marks non-vector field '{}' as an all-NULL vector",
                task.task_id, field_name
            )));
        }
    }

    if task.schema.fields.len() != canonical.fields.len() {
        return Err(lance::Error::invalid_input(format!(
            "worker transaction for task '{}' has {} fields but the canonical schema has {}",
            task.task_id,
            task.schema.fields.len(),
            canonical.fields.len()
        )));
    }

    let mut normalized = task.schema.clone();
    let mut omitted_field_ids = HashSet::new();
    for ((source_field, normalized_field), canonical_field) in task
        .schema
        .fields
        .iter()
        .zip(normalized.fields.iter_mut())
        .zip(canonical.fields.iter())
    {
        let Some(source_element) = variable_vector_type(source_field) else {
            continue;
        };
        let Some((canonical_element, dimension)) = fixed_size_vector_type(canonical_field) else {
            continue;
        };
        if source_element != canonical_element || dimension <= 0 {
            continue;
        }
        if !task.null_vector_fields.contains(&source_field.name) {
            return Err(lance::Error::invalid_input(format!(
                "worker transaction for task '{}' inferred variable list field '{}' but the operation uses a fixed-size vector",
                task.task_id, source_field.name
            )));
        }
        normalized_field.logical_type = canonical_field.logical_type.clone();
        normalized_field.children = canonical_field.children.clone();
        collect_field_ids(source_field, &mut omitted_field_ids);
    }

    let compare_options = SchemaCompareOptions {
        compare_metadata,
        compare_dictionary: true,
        compare_field_ids: false,
        ..Default::default()
    };
    normalized
        .check_compatible(canonical, &compare_options)
        .map_err(|err| {
            lance::Error::invalid_input(format!(
                "worker transaction schema for task '{}' is incompatible with the canonical schema: {err}",
                task.task_id
            ))
        })?;

    let mut field_id_mapping = HashMap::new();
    for (source_field, canonical_field) in task.schema.fields.iter().zip(canonical.fields.iter()) {
        if omitted_field_ids.contains(&source_field.id) {
            continue;
        }
        collect_field_id_mapping(source_field, canonical_field, &mut field_id_mapping)?;
    }

    for fragment in &mut task.fragments {
        for file in &mut fragment.files {
            if !file.column_indices.is_empty() && file.column_indices.len() != file.fields.len() {
                return Err(lance::Error::invalid_input(format!(
                    "worker transaction for task '{}' has mismatched field and column mappings for data file '{}'",
                    task.task_id, file.path
                )));
            }
            let has_column_indices = !file.column_indices.is_empty();
            let mut fields = Vec::with_capacity(file.fields.len());
            let mut column_indices = Vec::with_capacity(file.column_indices.len());
            for (index, source_id) in file.fields.iter().copied().enumerate() {
                if omitted_field_ids.contains(&source_id) {
                    continue;
                }
                let canonical_id = field_id_mapping.get(&source_id).ok_or_else(|| {
                    lance::Error::invalid_input(format!(
                        "worker transaction for task '{}' references unknown field id {} in data file '{}'",
                        task.task_id, source_id, file.path
                    ))
                })?;
                fields.push(*canonical_id);
                if has_column_indices {
                    column_indices.push(file.column_indices[index]);
                }
            }
            file.fields = fields.into();
            if has_column_indices {
                file.column_indices = column_indices.into();
            }
        }
    }
    Ok(())
}

fn canonicalize_distributed_staging_tasks(
    tasks: &mut [DistributedStagingTask],
    target_schema: Option<&LanceSchema>,
) -> Result<LanceSchema, lance::Error> {
    let mut canonical = match target_schema {
        Some(schema) => schema.clone(),
        None => tasks
            .first()
            .ok_or_else(|| {
                lance::Error::invalid_input("distributed Lance write has no schema".to_string())
            })?
            .schema
            .clone(),
    };
    if target_schema.is_none() {
        for task in tasks.iter().skip(1) {
            promote_distributed_vector_schema(&mut canonical, &task.schema);
        }
    }
    for task in tasks {
        canonicalize_distributed_task_fragments(task, &canonical, target_schema.is_none())?;
    }
    Ok(canonical)
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_distributed_write_validate(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    operation_id: *const c_char,
) -> i32 {
    match distributed_write_validate_inner(
        path,
        mode,
        option_keys,
        option_values,
        options_len,
        session,
        operation_id,
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

fn distributed_write_validate_inner(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    operation_id: *const c_char,
) -> FfiResult<()> {
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let mode = unsafe { cstr_to_str(mode, "mode")? }.to_string();
    validate_distributed_write_mode(&mode)?;
    let operation_id = unsafe { cstr_to_str(operation_id, "operation_id")? }.to_string();
    let storage_options =
        unsafe { distributed_storage_options_from_ffi(option_keys, option_values, options_len)? };
    let session = unsafe { optional_session_handle(session)? };

    match runtime::block_on(async {
        let dataset =
            load_optional_distributed_dataset(&path, &storage_options, session.clone()).await?;
        let (store, base) =
            resolve_distributed_object_store(&path, &storage_options, session.as_ref()).await?;
        if let Some(marker) = find_committed_distributed_operation(
            dataset.as_ref(),
            store.as_ref(),
            &base,
            &operation_id,
        )
        .await?
        {
            if marker.write_mode != mode {
                return Err(lance::Error::internal(format!(
                    "distributed Lance operation '{operation_id}' was already committed in '{}' mode, not '{mode}'",
                    marker.write_mode
                )));
            }
            return Ok(());
        }
        if mode == "create" && dataset.is_some() {
            return Err(lance::Error::invalid_input(format!(
                "Lance dataset already exists: {path}"
            ))
            .into());
        }
        Ok::<(), lance::Error>(())
    }) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            format!("distributed Lance write validation: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_distributed_write_commit(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    operation_id: *const c_char,
    task_attempt_ids: *const *const c_char,
    transaction_data: *const *const u8,
    transaction_lens: *const usize,
    transaction_count: usize,
    selected_rows: u64,
    out_rows: *mut u64,
) -> i32 {
    match distributed_write_commit_inner(
        path,
        mode,
        option_keys,
        option_values,
        options_len,
        session,
        operation_id,
        task_attempt_ids,
        transaction_data,
        transaction_lens,
        transaction_count,
        selected_rows,
        out_rows,
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

#[allow(clippy::too_many_arguments)]
fn distributed_write_commit_inner(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    operation_id: *const c_char,
    task_attempt_ids: *const *const c_char,
    transaction_data: *const *const u8,
    transaction_lens: *const usize,
    transaction_count: usize,
    selected_rows: u64,
    out_rows: *mut u64,
) -> FfiResult<()> {
    if out_rows.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_rows is null",
        ));
    }
    if transaction_count == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "distributed Lance write selected no worker transactions",
        ));
    }
    if task_attempt_ids.is_null() || transaction_data.is_null() || transaction_lens.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "distributed Lance transaction arrays are null",
        ));
    }

    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let mode = unsafe { cstr_to_str(mode, "mode")? }.to_string();
    validate_distributed_write_mode(&mode)?;
    let operation_id = unsafe { cstr_to_str(operation_id, "operation_id")? }.to_string();
    let storage_options =
        unsafe { distributed_storage_options_from_ffi(option_keys, option_values, options_len)? };
    let session = unsafe { optional_session_handle(session)? };
    let task_attempt_ids =
        unsafe { slice_from_ptr(task_attempt_ids, transaction_count, "task_attempt_ids")? };
    let transaction_data =
        unsafe { slice_from_ptr(transaction_data, transaction_count, "transaction_data")? };
    let transaction_lens =
        unsafe { slice_from_ptr(transaction_lens, transaction_count, "transaction_lens")? };

    let mut task_ids = Vec::with_capacity(transaction_count);
    let mut transactions = Vec::with_capacity(transaction_count);
    for idx in 0..transaction_count {
        if task_attempt_ids[idx].is_null() || transaction_data[idx].is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("distributed Lance transaction pointer is null at index {idx}"),
            ));
        }
        let task_id = unsafe { CStr::from_ptr(task_attempt_ids[idx]) }
            .to_str()
            .map_err(|err| {
                FfiError::new(
                    ErrorCode::Utf8,
                    format!("task_attempt_ids[{idx}] utf8: {err}"),
                )
            })?
            .to_string();
        let bytes =
            unsafe { std::slice::from_raw_parts(transaction_data[idx], transaction_lens[idx]) };
        let message = pb::Transaction::decode(bytes).map_err(|err| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                format!("decode distributed Lance transaction {idx}: {err}"),
            )
        })?;
        let transaction = Transaction::try_from(message).map_err(|err| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                format!("convert distributed Lance transaction {idx}: {err}"),
            )
        })?;
        task_ids.push(task_id);
        transactions.push(transaction);
    }

    async fn execute_distributed_commit(
        path: String,
        mode: String,
        operation_id: String,
        storage_options: HashMap<String, String>,
        session: Option<Arc<Session>>,
        task_ids: Vec<String>,
        transactions: Vec<Transaction>,
        selected_rows: u64,
    ) -> Result<u64, DistributedCommitError> {
        let existing =
            load_optional_distributed_dataset(&path, &storage_options, session.clone()).await?;
        let (store, base) =
            resolve_distributed_object_store(&path, &storage_options, session.as_ref()).await?;
        if let Some(marker) = find_committed_distributed_operation(
            existing.as_ref(),
            store.as_ref(),
            &base,
            &operation_id,
        )
        .await?
        {
            let committed_rows = validate_distributed_operation_marker(
                &marker,
                &operation_id,
                selected_rows,
                &mode,
            )?;
            if let Ok(staging) = distributed_staging_operation_path(&base, &operation_id) {
                if let Err(error) = remove_distributed_dir_if_exists(store.as_ref(), &staging).await
                {
                    log::warn!(
                        "failed to remove recovered distributed Lance staging directory '{}': {error}",
                        staging.as_ref()
                    );
                }
            }
            return Ok(committed_rows);
        }
        if mode == "create" && existing.is_some() {
            return Err(lance::Error::invalid_input(format!(
                "Lance dataset already exists: {path}"
            ))
            .into());
        }

        remove_distributed_destination_files(
            store.as_ref(),
            &base,
            &operation_id,
            existing.as_ref(),
        )
        .await?;

        let mut staging_tasks = Vec::with_capacity(task_ids.len());
        let mut combined_config = None;
        for (task_index, (task_id, transaction)) in task_ids.iter().zip(transactions).enumerate() {
            let null_vector_fields = distributed_null_vector_fields(&transaction)?;
            let Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values,
                initial_bases,
            } = transaction.operation
            else {
                return Err(lance::Error::invalid_input(format!(
                    "worker transaction for task '{task_id}' is not an overwrite staging transaction"
                ))
                .into());
            };
            if initial_bases
                .as_ref()
                .is_some_and(|bases| !bases.is_empty())
            {
                return Err(lance::Error::invalid_input(format!(
                    "worker transaction for task '{task_id}' contains external base paths"
                ))
                .into());
            }
            if task_index == 0 {
                combined_config = config_upsert_values.clone();
            } else if combined_config != config_upsert_values {
                return Err(lance::Error::invalid_input(
                    "distributed Lance worker dataset configurations do not match".to_string(),
                )
                .into());
            }
            for fragment in &fragments {
                for file in &fragment.files {
                    if file.base_id.is_some() {
                        return Err(lance::Error::invalid_input(format!(
                            "worker transaction for task '{task_id}' contains an external data file"
                        ))
                        .into());
                    }
                    if file.path.is_empty() || file.path.contains('/') {
                        return Err(lance::Error::invalid_input(format!(
                            "worker transaction for task '{task_id}' has non-flat data file path '{}'",
                            file.path
                        ))
                        .into());
                    }
                }
            }
            staging_tasks.push(DistributedStagingTask {
                task_id: task_id.clone(),
                fragments,
                schema,
                null_vector_fields,
            });
        }

        let target_schema = if mode == "append" {
            existing.as_ref().map(Dataset::schema)
        } else {
            None
        };
        let schema = canonicalize_distributed_staging_tasks(&mut staging_tasks, target_schema)?;

        let mut combined_fragments = Vec::new();
        for mut task in staging_tasks {
            copy_distributed_task_data(store.as_ref(), &base, &operation_id, &task.task_id).await?;
            let file_prefix = distributed_destination_task_prefix(&operation_id, &task.task_id);
            for fragment in &mut task.fragments {
                for file in &mut fragment.files {
                    file.path = format!("{file_prefix}{}", file.path);
                }
            }
            combined_fragments.extend(task.fragments);
        }

        let operation = if existing.is_some() && mode == "append" {
            Operation::Append {
                fragments: combined_fragments,
            }
        } else {
            Operation::Overwrite {
                fragments: combined_fragments,
                schema,
                config_upsert_values: combined_config,
                initial_bases: None,
            }
        };
        let mut properties = HashMap::new();
        properties.insert(VANE_OPERATION_ID_PROPERTY.to_string(), operation_id.clone());
        properties.insert(
            VANE_ROW_COUNT_PROPERTY.to_string(),
            selected_rows.to_string(),
        );
        properties.insert(VANE_WRITE_MODE_PROPERTY.to_string(), mode.clone());
        let transaction = Transaction {
            read_version: existing.as_ref().map(Dataset::version_id).unwrap_or(0),
            uuid: format!("vane-{}", hex_identity(&operation_id)),
            operation,
            tag: None,
            transaction_properties: Some(Arc::new(properties)),
        };

        let mut builder = CommitBuilder::new(path.as_str())
            .with_store_params(distributed_store_params(&storage_options));
        if let Some(session) = session.clone() {
            builder = builder.with_session(session);
        }
        if let Err(commit_error) = builder.execute(transaction).await {
            if is_definitive_commit_failure(&commit_error) {
                return Err(DistributedCommitError::Known(commit_error));
            }

            // Object-store commit APIs may return an error after publishing the
            // new manifest. Re-open and reconcile the durable operation before
            // deciding whether the write failed. A non-definitive error with no
            // proof either way is deliberately surfaced as outcome-unknown.
            let refreshed = load_optional_distributed_dataset(
                &path,
                &storage_options,
                session.clone(),
            )
            .await
            .map_err(|reconcile_error| {
                DistributedCommitError::OutcomeUnknown(format!(
                    "manifest commit returned '{commit_error}' and reopening the dataset for reconciliation failed: {reconcile_error}"
                ))
            })?;
            let marker = find_committed_distributed_operation(
                refreshed.as_ref(),
                store.as_ref(),
                &base,
                &operation_id,
            )
            .await
            .map_err(|reconcile_error| {
                DistributedCommitError::OutcomeUnknown(format!(
                    "manifest commit returned '{commit_error}' and operation reconciliation failed: {reconcile_error}"
                ))
            })?;
            let Some(marker) = marker else {
                return Err(DistributedCommitError::OutcomeUnknown(format!(
                    "manifest commit returned a non-definitive error and operation '{operation_id}' could not be proven committed: {commit_error}"
                )));
            };
            validate_distributed_operation_marker(&marker, &operation_id, selected_rows, &mode)?;
        }
        record_commit();

        let marker = DistributedOperationMarker {
            format_version: VANE_OPERATION_MARKER_FORMAT_VERSION,
            operation_id: operation_id.clone(),
            row_count: selected_rows,
            write_mode: mode.clone(),
        };
        let mut marker_error = None;
        for _ in 0..3 {
            match write_distributed_operation_marker(store.as_ref(), &base, &marker).await {
                Ok(()) => {
                    marker_error = None;
                    break;
                }
                Err(error) => {
                    marker_error = Some(error);
                    match read_distributed_operation_marker(store.as_ref(), &base, &operation_id)
                        .await
                    {
                        Ok(Some(durable)) => {
                            validate_distributed_operation_marker(
                                &durable,
                                &operation_id,
                                selected_rows,
                                &mode,
                            )?;
                            marker_error = None;
                            break;
                        }
                        Ok(None) | Err(_) => {}
                    }
                }
            }
        }
        if let Some(error) = marker_error {
            return Err(DistributedCommitError::OutcomeUnknown(format!(
                "manifest commit succeeded for operation '{operation_id}', but its durable idempotency marker could not be written after 3 attempts: {error}"
            )));
        }

        if let Ok(staging) = distributed_staging_operation_path(&base, &operation_id) {
            if let Err(error) = remove_distributed_dir_if_exists(store.as_ref(), &staging).await {
                log::warn!(
                    "failed to remove committed distributed Lance staging directory '{}': {error}",
                    staging.as_ref()
                );
            }
        }
        Ok(selected_rows)
    }

    let commit_result = runtime::block_on(execute_distributed_commit(
        path,
        mode,
        operation_id,
        storage_options,
        session,
        task_ids,
        transactions,
        selected_rows,
    ));

    match commit_result {
        Ok(Ok(rows)) => {
            unsafe {
                *out_rows = rows;
            }
            Ok(())
        }
        Ok(Err(DistributedCommitError::Known(err))) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            format!("distributed Lance commit: {err}"),
        )),
        Ok(Err(DistributedCommitError::OutcomeUnknown(message))) => Err(FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            format!("distributed Lance commit outcome is unknown: {message}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_distributed_write_abort(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    operation_id: *const c_char,
) -> i32 {
    match distributed_write_abort_inner(
        path,
        option_keys,
        option_values,
        options_len,
        session,
        operation_id,
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

fn distributed_write_abort_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    operation_id: *const c_char,
) -> FfiResult<()> {
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let operation_id = unsafe { cstr_to_str(operation_id, "operation_id")? }.to_string();
    let storage_options =
        unsafe { distributed_storage_options_from_ffi(option_keys, option_values, options_len)? };
    let session = unsafe { optional_session_handle(session)? };

    match runtime::block_on(async {
        let dataset =
            load_optional_distributed_dataset(&path, &storage_options, session.clone()).await?;
        let (store, base) =
            resolve_distributed_object_store(&path, &storage_options, session.as_ref()).await?;
        let committed = find_committed_distributed_operation(
            dataset.as_ref(),
            store.as_ref(),
            &base,
            &operation_id,
        )
        .await?;
        let staging = distributed_staging_operation_path(&base, &operation_id)?;
        remove_distributed_dir_if_exists(store.as_ref(), &staging).await?;
        if committed.is_none() {
            remove_distributed_destination_files(
                store.as_ref(),
                &base,
                &operation_id,
                dataset.as_ref(),
            )
            .await?;
        }
        Ok::<(), lance::Error>(())
    }) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            format!("distributed Lance abort: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::CString;

    use arrow_array::{new_null_array, ArrayRef, Int64Array};
    use lance::dataset::NewColumnTransform;
    use lance_table::format::DataFile;

    fn lance_schema(fields: Vec<Field>) -> LanceSchema {
        let arrow_schema = Schema::new(fields);
        let mut schema = LanceSchema::try_from(&arrow_schema).unwrap();
        schema.set_field_id(None);
        schema
    }

    fn staging_task(
        task_id: &str,
        schema: LanceSchema,
        null_vector_fields: HashSet<String>,
    ) -> DistributedStagingTask {
        let field_ids = schema.field_ids();
        let column_indices = (0..i32::try_from(field_ids.len()).unwrap()).collect();
        let mut fragment = Fragment::new(0);
        fragment.files.push(DataFile::new(
            "part.lance",
            field_ids,
            column_indices,
            2,
            1,
            None,
            None,
        ));
        DistributedStagingTask {
            task_id: task_id.to_string(),
            fragments: vec![fragment],
            schema,
            null_vector_fields,
        }
    }

    fn all_null_list_vector_batch(id: i64) -> RecordBatch {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("vector", DataType::List(item.clone()), true),
            Field::new("id", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                new_null_array(&DataType::List(item), 1),
                Arc::new(Int64Array::from(vec![id])),
            ],
        )
        .unwrap()
    }

    fn fixed_vector_batch(id: i64, values: [f32; 3]) -> RecordBatch {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("vector", DataType::FixedSizeList(item.clone(), 3), true),
            Field::new("id", DataType::Int64, false),
        ]));
        let values: ArrayRef = Arc::new(Float32Array::from(values.to_vec()));
        let vector = FixedSizeListArray::try_new(item, 3, values, None).unwrap();
        RecordBatch::try_new(
            schema,
            vec![Arc::new(vector), Arc::new(Int64Array::from(vec![id]))],
        )
        .unwrap()
    }

    fn int64_batch(columns: &[(&str, i64)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|(name, _)| Field::new(*name, DataType::Int64, false))
                .collect::<Vec<_>>(),
        ));
        RecordBatch::try_new(
            schema,
            columns
                .iter()
                .map(|(_, value)| Arc::new(Int64Array::from(vec![*value])) as ArrayRef)
                .collect(),
        )
        .unwrap()
    }

    fn write_staging_transaction(path: &str, batch: RecordBatch) -> Transaction {
        runtime::block_on(async {
            let params = WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            };
            InsertBuilder::new(path)
                .with_params(&params)
                .execute_uncommitted(vec![batch])
                .await
        })
        .unwrap()
        .unwrap()
    }

    #[test]
    fn aborted_writer_does_not_commit_partial_overwrite() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-aborted-writer-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let original = int64_batch(&[("id", 1)]);
        runtime::block_on(async {
            let params = WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            };
            InsertBuilder::new(path_text.as_str())
                .with_params(&params)
                .execute(vec![original])
                .await
        })
        .unwrap()
        .unwrap();

        let partial = int64_batch(&[("id", 999)]);
        let params = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let aborted = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = sync_channel(2);
        let join = spawn_writer_thread(
            WriterKind::Committed,
            path_text.clone(),
            params,
            partial.schema(),
            receiver,
            aborted.clone(),
        );
        sender.send(partial).unwrap();
        aborted.store(true, Ordering::Release);
        drop(sender);

        let error = match join.join().unwrap() {
            Err(error) => error,
            Ok(_) => panic!("aborted writer unexpectedly committed"),
        };
        assert!(
            error.contains("aborted"),
            "unexpected writer error: {error}"
        );

        let batch = runtime::block_on(async {
            DatasetBuilder::from_uri(path_text.as_str())
                .load()
                .await?
                .scan()
                .try_into_batch()
                .await
        })
        .unwrap()
        .unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(
            batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[1]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn aborting_uncommitted_append_removes_only_orphan_files() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-abort-transaction-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        runtime::block_on(async {
            InsertBuilder::new(path_text.as_str())
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![int64_batch(&[("id", 1)])])
                .await
        })
        .unwrap()
        .unwrap();

        let transaction = runtime::block_on(async {
            let dataset = DatasetBuilder::from_uri(path_text.as_str()).load().await?;
            InsertBuilder::new(Arc::new(dataset))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                })
                .execute_uncommitted(vec![int64_batch(&[("id", 2)])])
                .await
        })
        .unwrap()
        .unwrap();
        let candidate_paths = collect_transaction_owned_paths(&transaction);
        assert!(!candidate_paths.is_empty());
        for relative in &candidate_paths {
            assert!(dataset_path.join(relative).exists(), "missing {relative}");
        }

        let path = CString::new(path_text.clone()).unwrap();
        let transaction = Box::into_raw(Box::new(transaction)) as *mut c_void;
        abort_transaction_with_storage_options_inner(
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            transaction,
        )
        .unwrap();

        for relative in &candidate_paths {
            assert!(
                !dataset_path.join(relative).exists(),
                "orphan survived abort: {relative}"
            );
        }
        let rows = runtime::block_on(async {
            DatasetBuilder::from_uri(path_text.as_str())
                .load()
                .await?
                .count_rows(None)
                .await
        })
        .unwrap()
        .unwrap();
        assert_eq!(rows, 1);

        std::fs::remove_dir_all(&root).unwrap();
    }

    fn commit_distributed_transactions(
        path: &CString,
        mode: &CString,
        operation_id: &CString,
        task_ids: &[CString],
        transactions: &[Vec<u8>],
        selected_rows: u64,
    ) -> u64 {
        let task_id_ptrs = task_ids
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        let transaction_ptrs = transactions
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        let transaction_lens = transactions.iter().map(Vec::len).collect::<Vec<_>>();
        let mut out_rows = 0;
        distributed_write_commit_inner(
            path.as_ptr(),
            mode.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            operation_id.as_ptr(),
            task_id_ptrs.as_ptr(),
            transaction_ptrs.as_ptr(),
            transaction_lens.as_ptr(),
            transactions.len(),
            selected_rows,
            &mut out_rows,
        )
        .unwrap();
        out_rows
    }

    #[test]
    fn distributed_append_remaps_staging_field_ids() {
        let staging_schema = lance_schema(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, true),
        ]);
        let mut target_schema = staging_schema.clone();
        target_schema.fields[0].id = 4;
        target_schema.fields[1].id = 9;

        let mut tasks = vec![staging_task("task-0", staging_schema, HashSet::new())];
        let canonical =
            canonicalize_distributed_staging_tasks(&mut tasks, Some(&target_schema)).unwrap();

        assert_eq!(canonical.field_ids(), vec![4, 9]);
        assert_eq!(tasks[0].fragments[0].files[0].fields.as_ref(), &[4, 9]);
        assert_eq!(
            tasks[0].fragments[0].files[0].column_indices.as_ref(),
            &[0, 1]
        );
    }

    #[test]
    fn distributed_create_promotes_all_null_vector_task_to_fixed_size() {
        let list_item = Arc::new(Field::new("item", DataType::Float32, true));
        let list_schema = lance_schema(vec![
            Field::new("vector", DataType::List(list_item), true),
            Field::new("id", DataType::Int64, false),
        ]);
        let fixed_item = Arc::new(Field::new("item", DataType::Float32, true));
        let fixed_schema = lance_schema(vec![
            Field::new("vector", DataType::FixedSizeList(fixed_item, 3), true),
            Field::new("id", DataType::Int64, false),
        ]);
        let mut null_vector_fields = HashSet::new();
        null_vector_fields.insert("vector".to_string());
        let null_vector_ids = list_schema
            .field("vector")
            .unwrap()
            .children
            .iter()
            .map(|field| field.id)
            .chain(std::iter::once(list_schema.field("vector").unwrap().id))
            .collect::<HashSet<_>>();

        let mut tasks = vec![
            staging_task("null-task", list_schema, null_vector_fields),
            staging_task("fixed-task", fixed_schema, HashSet::new()),
        ];
        let canonical = canonicalize_distributed_staging_tasks(&mut tasks, None).unwrap();

        assert!(matches!(
            canonical.field("vector").unwrap().data_type(),
            DataType::FixedSizeList(_, 3)
        ));
        let null_file = &tasks[0].fragments[0].files[0];
        assert!(null_file
            .fields
            .iter()
            .all(|field_id| !null_vector_ids.contains(field_id)));
        assert_eq!(
            null_file.fields.as_ref(),
            &[canonical.field("id").unwrap().id]
        );
        assert_eq!(null_file.column_indices.as_ref(), &[2]);
        assert_eq!(
            tasks[1].fragments[0].files[0].fields.as_ref(),
            canonical.field_ids()
        );
    }

    #[test]
    fn distributed_create_rejects_unmarked_variable_vector_task() {
        let list_item = Arc::new(Field::new("item", DataType::Float32, true));
        let list_schema = lance_schema(vec![Field::new("vector", DataType::List(list_item), true)]);
        let fixed_item = Arc::new(Field::new("item", DataType::Float32, true));
        let fixed_schema = lance_schema(vec![Field::new(
            "vector",
            DataType::FixedSizeList(fixed_item, 3),
            true,
        )]);
        let mut tasks = vec![
            staging_task("list-task", list_schema, HashSet::new()),
            staging_task("fixed-task", fixed_schema, HashSet::new()),
        ];

        let error = canonicalize_distributed_staging_tasks(&mut tasks, None).unwrap_err();
        assert!(error.to_string().contains("inferred variable list field"));
    }

    #[tokio::test]
    async fn distributed_operation_marker_round_trips_outside_lance_history() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-operation-marker-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let uri = root.to_string_lossy().into_owned();
        let (store, base) = resolve_distributed_object_store(&uri, &HashMap::new(), None)
            .await
            .unwrap();
        let marker = DistributedOperationMarker {
            format_version: VANE_OPERATION_MARKER_FORMAT_VERSION,
            operation_id: "operation-1".to_string(),
            row_count: 7,
            write_mode: "append".to_string(),
        };

        write_distributed_operation_marker(store.as_ref(), &base, &marker)
            .await
            .unwrap();
        let loaded = read_distributed_operation_marker(store.as_ref(), &base, &marker.operation_id)
            .await
            .unwrap();

        assert_eq!(loaded, Some(marker));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn distributed_commit_survives_vacuum_and_reads_promoted_vector() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-distributed-vacuum-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let operation_id_text = "operation-vacuum";
        let task_id_texts = ["null-task", "fixed-task"];
        let staging_paths = task_id_texts.map(|task_id| {
            format!(
                "{path_text}/_vane_staging/{}/{}",
                hex_identity(operation_id_text),
                hex_identity(task_id)
            )
        });

        let mut null_transaction =
            write_staging_transaction(&staging_paths[0], all_null_list_vector_batch(1));
        let mut properties = null_transaction
            .transaction_properties
            .as_deref()
            .cloned()
            .unwrap_or_default();
        properties.insert(
            VANE_NULL_VECTOR_FIELDS_PROPERTY.to_string(),
            serde_json::to_string(&["vector"]).unwrap(),
        );
        null_transaction.transaction_properties = Some(Arc::new(properties));
        let fixed_transaction =
            write_staging_transaction(&staging_paths[1], fixed_vector_batch(2, [1.0, 2.0, 3.0]));
        let transactions = [&null_transaction, &fixed_transaction]
            .into_iter()
            .map(|transaction| pb::Transaction::from(transaction).encode_to_vec())
            .collect::<Vec<_>>();

        let path = CString::new(path_text.clone()).unwrap();
        let mode = CString::new("create").unwrap();
        let operation_id = CString::new(operation_id_text).unwrap();
        let task_ids = task_id_texts.map(|task_id| CString::new(task_id).unwrap());
        assert_eq!(
            commit_distributed_transactions(
                &path,
                &mode,
                &operation_id,
                &task_ids,
                &transactions,
                2,
            ),
            2
        );

        let append_batch = fixed_vector_batch(3, [4.0, 5.0, 6.0]);
        let dataset = runtime::block_on(async {
            let params = WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            };
            InsertBuilder::new(path_text.as_str())
                .with_params(&params)
                .execute(vec![append_batch])
                .await
        })
        .unwrap()
        .unwrap();
        assert_eq!(dataset.version_id(), 2);
        runtime::block_on(dataset.cleanup_old_versions(
            chrono::Duration::zero(),
            Some(true),
            Some(false),
        ))
        .unwrap()
        .unwrap();
        assert!(runtime::block_on(dataset.checkout_version(1))
            .unwrap()
            .is_err());

        let (store, base) = runtime::block_on(resolve_distributed_object_store(
            &path_text,
            &HashMap::new(),
            None,
        ))
        .unwrap()
        .unwrap();
        let marker = runtime::block_on(read_distributed_operation_marker(
            store.as_ref(),
            &base,
            operation_id_text,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(marker.unwrap().row_count, 2);

        assert_eq!(
            commit_distributed_transactions(
                &path,
                &mode,
                &operation_id,
                &task_ids,
                &transactions,
                2,
            ),
            2
        );
        distributed_write_abort_inner(
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            operation_id.as_ptr(),
        )
        .unwrap();

        let (schema, batches) = runtime::block_on(async {
            let dataset = DatasetBuilder::from_uri(path_text.as_str()).load().await?;
            let schema = dataset.schema().clone();
            let batches = dataset
                .scan()
                .try_into_stream()
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            Ok::<_, lance::Error>((schema, batches))
        })
        .unwrap()
        .unwrap();
        assert!(matches!(
            schema.field("vector").unwrap().data_type(),
            DataType::FixedSizeList(_, 3)
        ));
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
        assert_eq!(
            batches
                .iter()
                .map(|batch| batch.column(0).null_count())
                .sum::<usize>(),
            1
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn distributed_append_uses_evolved_target_field_ids() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-distributed-field-ids-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();

        let target_schema = runtime::block_on(async {
            let params = WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            };
            let mut dataset = InsertBuilder::new(path_text.as_str())
                .with_params(&params)
                .execute(vec![int64_batch(&[("id", 1), ("obsolete", 9)])])
                .await?;
            dataset
                .add_columns(
                    NewColumnTransform::SqlExpressions(vec![(
                        "value".to_string(),
                        "id + 100".to_string(),
                    )]),
                    Some(vec!["id".to_string()]),
                    None,
                )
                .await?;
            dataset.drop_columns(&["obsolete"]).await?;
            Ok::<_, lance::Error>(dataset.schema().clone())
        })
        .unwrap()
        .unwrap();
        assert_eq!(target_schema.field("id").unwrap().id, 0);
        assert_eq!(target_schema.field("value").unwrap().id, 2);

        let operation_id_text = "operation-evolved-schema";
        let task_id_text = "append-task";
        let staging_path = format!(
            "{path_text}/_vane_staging/{}/{}",
            hex_identity(operation_id_text),
            hex_identity(task_id_text)
        );
        let staging_transaction =
            write_staging_transaction(&staging_path, int64_batch(&[("id", 2), ("value", 200)]));
        let Operation::Overwrite {
            schema: staging_schema,
            ..
        } = &staging_transaction.operation
        else {
            panic!("staging transaction was not an overwrite");
        };
        assert_eq!(staging_schema.field("value").unwrap().id, 1);
        let transactions = vec![pb::Transaction::from(&staging_transaction).encode_to_vec()];

        let path = CString::new(path_text.clone()).unwrap();
        let mode = CString::new("append").unwrap();
        let operation_id = CString::new(operation_id_text).unwrap();
        let task_ids = [CString::new(task_id_text).unwrap()];
        assert_eq!(
            commit_distributed_transactions(
                &path,
                &mode,
                &operation_id,
                &task_ids,
                &transactions,
                1,
            ),
            1
        );

        let (schema, batch) = runtime::block_on(async {
            let dataset = DatasetBuilder::from_uri(path_text.as_str()).load().await?;
            let schema = dataset.schema().clone();
            let batch = dataset.scan().try_into_batch().await?;
            Ok::<_, lance::Error>((schema, batch))
        })
        .unwrap()
        .unwrap();
        assert_eq!(schema.field("value").unwrap().id, 2);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(
            batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[1, 2]
        );
        assert_eq!(
            batch
                .column_by_name("value")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[101, 200]
        );

        std::fs::remove_dir_all(&root).unwrap();
    }
}
