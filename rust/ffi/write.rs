use std::collections::{BTreeSet, HashMap, HashSet};
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
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
#[cfg(feature = "vane-distributed")]
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, Dataset, InsertBuilder, WriteMode, WriteParams};
use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor};
use lance::session::Session;
use lance_core::datatypes::Field as LanceField;
#[cfg(feature = "vane-distributed")]
use lance_core::datatypes::{NullabilityComparison, Schema as LanceSchema, SchemaCompareOptions};
use lance_select::RowAddrTreeMap;
use lance_table::format::{pb, DataFile, Fragment, RowIdMeta};
use lance_table::io::deletion::relative_deletion_file_path;
use lance_table::rowids::version::RowDatasetVersionMeta;
use object_store::path::Path;
use prost::Message;
#[cfg(feature = "vane-distributed")]
use serde::{Deserialize, Serialize};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::session::record_commit;
use super::util::{
    cstr_to_str, lance_mutation_outcome_unknown, optional_session_handle, slice_from_ptr, FfiError,
    FfiResult,
};

#[cfg(feature = "vane-distributed")]
const VANE_OPERATION_ID_PROPERTY: &str = "vane.distributed.operation_id";
#[cfg(feature = "vane-distributed")]
const VANE_ROW_COUNT_PROPERTY: &str = "vane.distributed.row_count";
#[cfg(feature = "vane-distributed")]
const VANE_WRITE_MODE_PROPERTY: &str = "vane.distributed.write_mode";
const VANE_NULL_VECTOR_FIELDS_PROPERTY: &str = "vane.distributed.null_vector_fields";
#[cfg(feature = "vane-distributed")]
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

/// Opaque transaction state shared by every transaction-producing C ABI.
///
/// Lance's conflict resolution needs the row addresses touched by row-level
/// mutations in addition to the serialized transaction itself.  Keep both in
/// one Rust-owned allocation so the C++ layer cannot accidentally separate or
/// lose that commit-only metadata.
pub(crate) struct VaneTransaction {
    transaction: Transaction,
    affected_rows: Option<RowAddrTreeMap>,
}

impl VaneTransaction {
    pub(crate) fn new(transaction: Transaction) -> Self {
        Self {
            transaction,
            affected_rows: None,
        }
    }

    pub(crate) fn with_affected_rows(
        transaction: Transaction,
        affected_rows: RowAddrTreeMap,
    ) -> Self {
        Self {
            transaction,
            affected_rows: Some(affected_rows),
        }
    }
}

#[derive(Debug)]
struct WriterThreadError {
    message: String,
    outcome_unknown: bool,
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
    explicit_dim: bool,
    list_kind: VectorListKind,
    element_type: VectorElementType,
}

struct WriterState {
    kind: WriterKind,
    path: String,
    params: WriteParams,
    finished: bool,

    vector_candidates: Vec<VectorConversion>,
    buffered_batches: Vec<RecordBatch>,
    buffered_rows: usize,

    output_schema: Option<SchemaRef>,
    output_sender: Option<SyncSender<RecordBatch>>,
    output_join: Option<JoinHandle<Result<WriterResult, WriterThreadError>>>,
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
        let has_explicit_dim = explicit_dim.is_some();
        candidates.push(VectorConversion {
            col_idx,
            field_name: field.name().clone(),
            dim: explicit_dim.unwrap_or(0),
            explicit_dim: has_explicit_dim,
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

fn prepare_vector_candidates(
    schema: &SchemaRef,
    vector_dims: *const c_char,
    infer_vector_dims: bool,
    path: &str,
    write_mode: WriteMode,
    storage_options: &HashMap<String, String>,
    session: Option<Arc<Session>>,
) -> FfiResult<Vec<VectorConversion>> {
    let mut candidates = parse_vector_candidates(schema, vector_dims, infer_vector_dims)?;
    if !matches!(write_mode, WriteMode::Append) || candidates.is_empty() {
        return Ok(candidates);
    }

    let target_schema = match runtime::block_on(load_optional_distributed_dataset(
        path,
        storage_options,
        session,
    )) {
        Ok(Ok(Some(dataset))) => dataset.schema().clone(),
        Ok(Ok(None)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteOpen,
                format!("cannot append because the Lance dataset does not exist: {path}"),
            ))
        }
        Ok(Err(error)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteOpen,
                format!("open Lance append target '{path}': {error}"),
            ))
        }
        Err(error) => {
            return Err(FfiError::new(
                ErrorCode::Runtime,
                format!("runtime: {error}"),
            ))
        }
    };

    candidates.retain_mut(|candidate| {
        if candidate.explicit_dim {
            return true;
        }
        let Some(target_field) = target_schema
            .fields
            .iter()
            .find(|field| field.name == candidate.field_name)
        else {
            return true;
        };
        if is_variable_list_vector_type(&target_field.data_type())
            == Some((candidate.list_kind, candidate.element_type))
        {
            // The existing dataset deliberately stores a ragged list. Automatic
            // inference must not silently change an append batch to a vector.
            return false;
        }
        if let Some((element_type, dimension)) = fixed_size_vector_type(target_field) {
            if element_type == candidate.element_type && dimension > 0 {
                candidate.dim = dimension as usize;
            }
        }
        true
    });
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
        fields[idx] = Arc::new(
            original
                .clone()
                .with_data_type(DataType::FixedSizeList(child_field, dim_i32)),
        );
    }
    Ok(Arc::new(Schema::new_with_metadata(
        fields,
        input_schema.metadata().clone(),
    )))
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
) -> Result<JoinHandle<Result<WriterResult, WriterThreadError>>, String> {
    std::thread::Builder::new()
        .name("lance-duckdb-writer".to_string())
        .spawn(move || -> Result<WriterResult, WriterThreadError> {
            let reader = ReceiverRecordBatchReader::new(schema, receiver, aborted);
            match kind {
                WriterKind::Committed => {
                    let fut = Dataset::write(reader, &path, Some(params));
                    match runtime::block_on(fut) {
                        Ok(Ok(_)) => Ok(WriterResult::Committed),
                        Ok(Err(err)) => Err(WriterThreadError {
                            outcome_unknown: !is_definitive_commit_failure(&err),
                            message: err.to_string(),
                        }),
                        Err(err) => Err(WriterThreadError {
                            message: format!("runtime: {err}"),
                            // Failure to obtain the runtime happens before the
                            // future is polled, so no commit was attempted.
                            outcome_unknown: false,
                        }),
                    }
                }
                WriterKind::Uncommitted => {
                    let source: Box<dyn RecordBatchReader + Send> = Box::new(reader);
                    let builder = InsertBuilder::new(path.as_str()).with_params(&params);
                    let fut = builder.execute_uncommitted_stream(source);
                    match runtime::block_on(fut) {
                        Ok(Ok(txn)) => Ok(WriterResult::Uncommitted(Box::new(txn))),
                        Ok(Err(err)) => Err(WriterThreadError {
                            message: err.to_string(),
                            // execute_uncommitted_stream does not expose any
                            // paths created before a late stream/write error.
                            outcome_unknown: true,
                        }),
                        Err(err) => Err(WriterThreadError {
                            message: format!("runtime: {err}"),
                            outcome_unknown: false,
                        }),
                    }
                }
            }
        })
        .map_err(|error| format!("failed to start Lance writer thread: {error}"))
}

fn writer_thread_failure(
    kind: WriterKind,
    fallback_code: ErrorCode,
    error: WriterThreadError,
) -> FfiError {
    if error.outcome_unknown {
        let context = if kind == WriterKind::Committed {
            "Lance writer commit outcome is unknown"
        } else {
            "Lance uncommitted writer may have left orphan files"
        };
        FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            format!("{context}: {}", error.message),
        )
    } else {
        FfiError::new(fallback_code, error.message)
    }
}

fn writer_channel_failure(
    kind: WriterKind,
    join: Option<JoinHandle<Result<WriterResult, WriterThreadError>>>,
    code: ErrorCode,
) -> FfiError {
    match join.map(JoinHandle::join) {
        Some(Ok(Err(error))) => writer_thread_failure(kind, code, error),
        Some(Err(_)) if kind == WriterKind::Committed => FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            "Lance writer thread panicked; commit outcome is unknown",
        ),
        Some(Err(_)) => FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            "uncommitted writer thread panicked and may have left orphan files",
        ),
        Some(Ok(Ok(WriterResult::Committed))) => FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            "Lance writer committed before accepting all input; commit outcome is unsafe to retry",
        ),
        Some(Ok(Ok(WriterResult::Uncommitted(_)))) => FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            "writer background task exited before accepting all input and returned files that could not be cleaned up",
        ),
        None if kind == WriterKind::Committed => FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            "Lance writer background task is unavailable; commit outcome is unknown",
        ),
        None => FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            "uncommitted writer background task is unavailable; file cleanup cannot be verified",
        ),
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
    let schema = Schema::try_from(ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::DatasetWriteOpen, format!("schema import: {err}"))
    })?;
    let schema: SchemaRef = Arc::new(schema);
    let data_type = DataType::Struct(schema.fields().clone());

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
    let vector_candidates = prepare_vector_candidates(
        &schema,
        vector_dims,
        infer_vector_dims != 0,
        &path,
        write_mode,
        &storage_options,
        session.clone(),
    )?;

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
    let aborted = Arc::new(AtomicBool::new(false));
    Ok(WriterHandle {
        input_schema: schema.clone(),
        data_type,
        state: Mutex::new(WriterState {
            kind: WriterKind::Uncommitted,
            path,
            params,
            finished: false,
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
    let schema = Schema::try_from(ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::DatasetWriteOpen, format!("schema import: {err}"))
    })?;
    let schema: SchemaRef = Arc::new(schema);
    let data_type = DataType::Struct(schema.fields().clone());

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
    let vector_candidates = prepare_vector_candidates(
        &schema,
        vector_dims,
        infer_vector_dims != 0,
        &path,
        write_mode,
        &storage_options,
        session.clone(),
    )?;

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

    let aborted = Arc::new(AtomicBool::new(false));
    Ok(WriterHandle {
        input_schema: schema.clone(),
        data_type,
        state: Mutex::new(WriterState {
            kind: WriterKind::Committed,
            path,
            params,
            finished: false,
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
        if guard.finished {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteBatch,
                "writer is already finished",
            ));
        }

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
                let mut out_batches = Vec::with_capacity(guard.buffered_batches.len());
                for b in guard.buffered_batches.iter() {
                    let out = convert_record_batch(b, &output_schema, &conversions)
                        .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
                    out_batches.push(out);
                }
                // Finish every fallible in-process conversion before the
                // background writer can observe end-of-stream and commit.
                let (sender, receiver) = sync_channel::<RecordBatch>(2);
                let join = spawn_writer_thread(
                    guard.kind,
                    guard.path.clone(),
                    guard.params.clone(),
                    output_schema.clone(),
                    receiver,
                    handle.aborted.clone(),
                )
                .map_err(|error| FfiError::new(ErrorCode::DatasetWriteBatch, error))?;
                guard.buffered_batches.clear();
                guard.buffered_rows = 0;
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
                // A closed receiver may otherwise look like a clean end of
                // input and let the writer commit the batches it accepted
                // before the failure.  Make the channel failure visible to
                // the reader before dropping the last sender so partial data
                // can never be committed as a successful write.
                handle.aborted.store(true, Ordering::Release);
                drop(sender);
                let (kind, join) = {
                    let mut guard = handle
                        .state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    guard.output_sender = None;
                    guard.finished = true;
                    (guard.kind, guard.output_join.take())
                };
                return Err(writer_channel_failure(
                    kind,
                    join,
                    ErrorCode::DatasetWriteBatch,
                ));
            }
        }
    }

    handle.batches_sent.fetch_add(1, Ordering::Relaxed);

    Ok(())
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
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
    let (kind, sender, join, to_send) = {
        let mut guard = handle
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.finished {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteFinish,
                "writer is already finished",
            ));
        }
        if guard.output_sender.is_none() {
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let output_schema = build_output_schema(&handle.input_schema, &conversions)
                .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinish, e))?;
            let mut out_batches = Vec::with_capacity(guard.buffered_batches.len() + 1);
            for b in guard.buffered_batches.iter() {
                let out = convert_record_batch(b, &output_schema, &conversions)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinish, e))?;
                out_batches.push(out);
            }
            if handle.batches_sent.load(Ordering::Acquire) == 0 {
                out_batches.push(RecordBatch::new_empty(output_schema.clone()));
            }
            let (sender, receiver) = sync_channel::<RecordBatch>(2);
            let join = spawn_writer_thread(
                guard.kind,
                guard.path.clone(),
                guard.params.clone(),
                output_schema.clone(),
                receiver,
                handle.aborted.clone(),
            )
            .map_err(|error| FfiError::new(ErrorCode::DatasetWriteFinish, error))?;
            guard.output_schema = Some(output_schema);
            guard.output_sender = Some(sender.clone());
            guard.output_join = Some(join);
            guard.buffered_rows = 0;
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
        guard.finished = true;
        (guard.kind, sender, join, to_send)
    };

    for b in to_send {
        if sender.send(b).is_err() {
            handle.aborted.store(true, Ordering::Release);
            drop(sender);
            return Err(writer_channel_failure(
                kind,
                Some(join),
                ErrorCode::DatasetWriteFinish,
            ));
        }
    }
    drop(sender);

    match join.join() {
        Ok(Ok(WriterResult::Committed)) => Ok(()),
        Ok(Ok(WriterResult::Uncommitted(_))) => Err(FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            "writer returned an uncommitted transaction whose files could not be cleaned up",
        )),
        Ok(Err(error)) => Err(writer_thread_failure(
            kind,
            ErrorCode::DatasetWriteFinish,
            error,
        )),
        Err(_) => Err(FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            "Lance writer thread panicked; commit outcome is unknown",
        )),
    }
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
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
    if out_transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction is null",
        ));
    }
    unsafe {
        ptr::write_unaligned(out_transaction, ptr::null_mut());
    }
    if writer.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "writer is null"));
    }

    let handle = unsafe { &*(writer as *const WriterHandle) };
    let (kind, sender, join, to_send, null_vector_fields) = {
        let mut guard = handle
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.finished {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer is already finished",
            ));
        }
        if guard.output_sender.is_none() {
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let output_schema = build_output_schema(&handle.input_schema, &conversions)
                .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinishUncommitted, e))?;
            let mut out_batches = Vec::with_capacity(guard.buffered_batches.len() + 1);
            for b in guard.buffered_batches.iter() {
                let out = convert_record_batch(b, &output_schema, &conversions)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinishUncommitted, e))?;
                out_batches.push(out);
            }
            if handle.batches_sent.load(Ordering::Acquire) == 0 {
                out_batches.push(RecordBatch::new_empty(output_schema.clone()));
            }
            let (sender, receiver) = sync_channel::<RecordBatch>(2);
            let join = spawn_writer_thread(
                guard.kind,
                guard.path.clone(),
                guard.params.clone(),
                output_schema.clone(),
                receiver,
                handle.aborted.clone(),
            )
            .map_err(|error| FfiError::new(ErrorCode::DatasetWriteFinishUncommitted, error))?;
            guard.output_schema = Some(output_schema);
            guard.output_sender = Some(sender.clone());
            guard.output_join = Some(join);
            guard.buffered_rows = 0;
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
        guard.finished = true;
        (guard.kind, sender, join, to_send, null_vector_fields)
    };

    for b in to_send {
        if sender.send(b).is_err() {
            handle.aborted.store(true, Ordering::Release);
            drop(sender);
            return Err(writer_channel_failure(
                kind,
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
                ErrorCode::DatasetCommitOutcomeUnknown,
                "Lance writer committed instead of returning an uncommitted transaction; retry is unsafe",
            ))
        }
        Ok(Err(error)) => {
            return Err(writer_thread_failure(
                kind,
                ErrorCode::DatasetWriteFinishUncommitted,
                error,
            ))
        }
        Err(_) => {
            return Err(FfiError::new(
                ErrorCode::DatasetCommitOutcomeUnknown,
                "uncommitted writer thread panicked and may have left orphan files",
            ))
        }
    };

    if !null_vector_fields.is_empty() {
        let encoded = serde_json::to_string(&null_vector_fields).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetCommitOutcomeUnknown,
                format!("serialize all-NULL vector fields after Lance staged files were written: {err}; the uncommitted transaction may have orphan files and must not be retried"),
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

    let txn = Box::new(VaneTransaction::new(*txn));
    unsafe {
        ptr::write_unaligned(out_transaction, Box::into_raw(txn) as *mut c_void);
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

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
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
    let txn = unsafe { Box::from_raw(transaction as *mut VaneTransaction) };
    let candidate_paths = collect_transaction_owned_paths(&txn.transaction);
    let cleanup_base_version = transaction_cleanup_base_version(&txn.transaction);
    let classify_argument_error = |error: FfiError| {
        if candidate_paths.is_empty() {
            error
        } else {
            FfiError::new(
                ErrorCode::DatasetCommitOutcomeUnknown,
                format!(
                    "Lance transaction was not committed, but orphan cleanup could not run because its commit arguments were invalid: {}; cleanup is incomplete",
                    error.message
                ),
            )
        }
    };
    let path = unsafe { cstr_to_str(path, "path") }
        .map_err(&classify_argument_error)?
        .to_string();
    let storage_options =
        unsafe { distributed_storage_options_from_ffi(option_keys, option_values, options_len) }
            .map_err(&classify_argument_error)?;
    let session = unsafe { optional_session_handle(session) }.map_err(&classify_argument_error)?;

    let mut builder = CommitBuilder::new(path.as_str())
        .with_store_params(distributed_store_params(&storage_options));
    if let Some(session) = session.clone() {
        builder = builder.with_session(session);
    }
    if let Some(affected_rows) = txn.affected_rows.clone() {
        builder = builder.with_affected_rows(affected_rows);
    }
    let commit_result = runtime::block_on(async {
        match builder.execute(txn.transaction.clone()).await {
            Ok(_) => Ok(()),
            Err(error) if is_definitive_commit_failure(&error) => {
                match remove_unreferenced_transaction_paths(
                    &path,
                    &storage_options,
                    session,
                    &candidate_paths,
                    cleanup_base_version,
                )
                .await
                {
                    Ok(()) => Err(FfiError::new(
                        ErrorCode::DatasetCommitTransaction,
                        error.to_string(),
                    )),
                    Err(cleanup_error) => Err(FfiError::new(
                        ErrorCode::DatasetCommitOutcomeUnknown,
                        format!(
                            "Lance transaction was definitively rejected ({error}), but orphan cleanup failed: {cleanup_error}; cleanup is incomplete"
                        ),
                    )),
                }
            }
            Err(error) => Err(FfiError::new(
                ErrorCode::DatasetCommitOutcomeUnknown,
                format!("Lance transaction commit outcome is unknown: {error}"),
            )),
        }
    });
    match commit_result {
        Ok(Ok(_)) => {
            record_commit();
            Ok(())
        }
        Ok(Err(error)) => Err(error),
        Err(err) => Err(FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            format!(
                "Lance transaction was not committed because the runtime was unavailable, but orphan cleanup could not run: {err}; cleanup is incomplete"
            ),
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
        let _ = Box::from_raw(transaction as *mut VaneTransaction);
    }
}

#[derive(Debug, Default)]
struct TransactionOwnedPaths {
    primary_paths: HashSet<String>,
    unsupported_base_ids: BTreeSet<u32>,
}

impl TransactionOwnedPaths {
    fn is_empty(&self) -> bool {
        self.primary_paths.is_empty() && self.unsupported_base_ids.is_empty()
    }
}

fn collect_fragment_owned_paths(fragment: &Fragment, paths: &mut TransactionOwnedPaths) {
    for file in &fragment.files {
        collect_data_file_owned_path(file, paths);
    }
    for overlay in &fragment.overlays {
        collect_data_file_owned_path(&overlay.data_file, paths);
    }
    if let Some(deletion_file) = &fragment.deletion_file {
        if let Some(base_id) = deletion_file.base_id {
            paths.unsupported_base_ids.insert(base_id);
        } else {
            paths
                .primary_paths
                .insert(relative_deletion_file_path(fragment.id, deletion_file));
        }
    }
    if let Some(RowIdMeta::External(file)) = &fragment.row_id_meta {
        paths.primary_paths.insert(file.path.clone());
    }
    for version_meta in [
        fragment.last_updated_at_version_meta.as_ref(),
        fragment.created_at_version_meta.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if let RowDatasetVersionMeta::External(file) = version_meta {
            paths.primary_paths.insert(file.path.clone());
        }
    }
}

fn collect_data_file_owned_path(file: &DataFile, paths: &mut TransactionOwnedPaths) {
    if let Some(base_id) = file.base_id {
        // A base file can live outside the dataset root.  We cannot safely
        // derive its object-store path from the transaction alone, so make
        // cleanup fail closed instead of guessing and deleting user data.
        paths.unsupported_base_ids.insert(base_id);
    } else {
        paths.primary_paths.insert(format!("data/{}", file.path));
    }
}

fn collect_transaction_owned_paths(transaction: &Transaction) -> TransactionOwnedPaths {
    let mut paths = TransactionOwnedPaths::default();
    match &transaction.operation {
        Operation::Append { fragments } | Operation::Overwrite { fragments, .. } => {
            for fragment in fragments {
                collect_fragment_owned_paths(fragment, &mut paths);
            }
        }
        Operation::Delete {
            updated_fragments, ..
        } => {
            for fragment in updated_fragments {
                collect_fragment_owned_paths(fragment, &mut paths);
            }
        }
        Operation::Update {
            updated_fragments,
            new_fragments,
            ..
        } => {
            for fragment in updated_fragments.iter().chain(new_fragments) {
                collect_fragment_owned_paths(fragment, &mut paths);
            }
        }
        Operation::Rewrite { groups, .. } => {
            for fragment in groups.iter().flat_map(|group| group.new_fragments.iter()) {
                collect_fragment_owned_paths(fragment, &mut paths);
            }
        }
        Operation::DataReplacement { replacements } => {
            for replacement in replacements {
                collect_data_file_owned_path(&replacement.1, &mut paths);
            }
        }
        Operation::DataOverlay { groups } => {
            for file in groups.iter().flat_map(|group| group.overlays.iter()) {
                collect_data_file_owned_path(&file.data_file, &mut paths);
            }
        }
        Operation::Merge { fragments, .. } => {
            for fragment in fragments {
                collect_fragment_owned_paths(fragment, &mut paths);
            }
        }
        _ => {}
    }
    paths
}

fn transaction_cleanup_base_version(transaction: &Transaction) -> Option<u64> {
    match &transaction.operation {
        // These operations carry final/updated fragments that can include files
        // inherited from the transaction's read version. Cleanup must protect
        // that base manifest in addition to the latest one before deleting any
        // candidate path.
        Operation::Delete { .. }
        | Operation::Update { .. }
        | Operation::Rewrite { .. }
        | Operation::DataReplacement { .. }
        | Operation::DataOverlay { .. }
        | Operation::Merge { .. } => Some(transaction.read_version),
        // Append and overwrite carry newly written fragments only.
        // Requiring the base manifest here would prevent safe orphan cleanup
        // when the commit itself was rejected because read_version was invalid.
        _ => None,
    }
}

async fn remove_unreferenced_transaction_paths(
    path: &str,
    storage_options: &HashMap<String, String>,
    session: Option<Arc<Session>>,
    candidate_paths: &TransactionOwnedPaths,
    protected_read_version: Option<u64>,
) -> Result<(), String> {
    if candidate_paths.is_empty() {
        return Ok(());
    }
    // Do this check before loading the dataset or deleting any primary paths.
    // A transaction may contain both ordinary and external-base files; once
    // an external base is present, deleting only the ordinary subset would
    // leave cleanup partial and could make a later retry unsafe.
    if !candidate_paths.unsupported_base_ids.is_empty() {
        return Err(format!(
            "cleanup cannot resolve transaction files in external Lance base id(s): {}",
            candidate_paths
                .unsupported_base_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let dataset = load_optional_distributed_dataset(path, storage_options, session.clone())
        .await
        .map_err(|error| error.to_string())?;
    let (store, base) = resolve_distributed_object_store(path, storage_options, session.as_ref())
        .await
        .map_err(|error| error.to_string())?;
    let mut referenced_paths = TransactionOwnedPaths::default();
    if let Some(dataset) = dataset.as_ref() {
        for fragment in dataset.manifest().fragments.iter() {
            collect_fragment_owned_paths(fragment, &mut referenced_paths);
        }
    }
    if let Some(read_version) = protected_read_version {
        let dataset = dataset.as_ref().ok_or_else(|| {
            format!(
                "cannot protect Lance transaction read version {read_version}: dataset no longer exists"
            )
        })?;
        let base_dataset = if dataset.version().version == read_version {
            None
        } else {
            Some(
                dataset
                    .checkout_version(read_version)
                    .await
                    .map_err(|error| {
                        format!(
                            "cannot protect Lance transaction read version {read_version}: {error}"
                        )
                    })?,
            )
        };
        let base_manifest = base_dataset
            .as_ref()
            .map(|dataset| dataset.manifest())
            .unwrap_or_else(|| dataset.manifest());
        for fragment in base_manifest.fragments.iter() {
            collect_fragment_owned_paths(fragment, &mut referenced_paths);
        }
    }
    for relative in candidate_paths
        .primary_paths
        .difference(&referenced_paths.primary_paths)
    {
        let object = path_join(&base, relative).map_err(|error| error.to_string())?;
        match store.delete(&object).await {
            Ok(()) => {}
            Err(error) if error.is_not_found() => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(())
}

pub(crate) async fn cleanup_uncommitted_fragments(
    path: &str,
    storage_options: &HashMap<String, String>,
    session: Option<Arc<Session>>,
    fragments: &[Fragment],
    protected_read_version: Option<u64>,
) -> Result<(), String> {
    let mut candidate_paths = TransactionOwnedPaths::default();
    for fragment in fragments {
        collect_fragment_owned_paths(fragment, &mut candidate_paths);
    }
    remove_unreferenced_transaction_paths(
        path,
        storage_options,
        session,
        &candidate_paths,
        protected_read_version,
    )
    .await
}

pub(crate) async fn cleanup_uncommitted_transaction(
    path: &str,
    storage_options: &HashMap<String, String>,
    session: Option<Arc<Session>>,
    transaction: &Transaction,
) -> Result<(), String> {
    let candidate_paths = collect_transaction_owned_paths(transaction);
    remove_unreferenced_transaction_paths(
        path,
        storage_options,
        session,
        &candidate_paths,
        transaction_cleanup_base_version(transaction),
    )
    .await
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
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
    let transaction = unsafe { Box::from_raw(transaction as *mut VaneTransaction) };
    let candidate_paths = collect_transaction_owned_paths(&transaction.transaction);
    let cleanup_base_version = transaction_cleanup_base_version(&transaction.transaction);
    if candidate_paths.is_empty() {
        return Ok(());
    }
    let cleanup_unavailable = |error: FfiError| {
        FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            format!(
                "uncommitted Lance transaction was consumed, but orphan cleanup could not run because its abort arguments were invalid: {}; cleanup is incomplete",
                error.message
            ),
        )
    };
    let path = unsafe { cstr_to_str(path, "path") }
        .map_err(&cleanup_unavailable)?
        .to_string();
    let storage_options =
        unsafe { distributed_storage_options_from_ffi(option_keys, option_values, options_len) }
            .map_err(&cleanup_unavailable)?;
    let session = unsafe { optional_session_handle(session) }.map_err(&cleanup_unavailable)?;

    match runtime::block_on(remove_unreferenced_transaction_paths(
        &path,
        &storage_options,
        session,
        &candidate_paths,
        cleanup_base_version,
    )) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            format!(
                "abort uncommitted Lance transaction failed: {error}; cleanup is incomplete"
            ),
        )),
        Err(error) => Err(FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            format!(
                "abort uncommitted Lance transaction could not access the runtime: {error}; cleanup is incomplete"
            ),
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
    if out_data.is_null() || out_len.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_data/out_len is null",
        ));
    }
    if super::util::output_regions_overlap(out_data, out_len) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_data and out_len must not overlap",
        ));
    }
    unsafe {
        ptr::write_unaligned(out_data, ptr::null_mut());
        ptr::write_unaligned(out_len, 0);
    }
    if transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "transaction is null",
        ));
    }

    let transaction = unsafe { &*(transaction as *const VaneTransaction) };
    if transaction.affected_rows.is_some() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "row-level Lance mutation transactions cannot be serialized because affected-row conflict metadata is commit-local",
        ));
    }
    let message = pb::Transaction::from(&transaction.transaction);
    let bytes = message.encode_to_vec().into_boxed_slice();
    let len = bytes.len();
    let data = Box::into_raw(bytes) as *mut u8;
    unsafe {
        ptr::write_unaligned(out_data, data);
        ptr::write_unaligned(out_len, len);
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

#[cfg(feature = "vane-distributed")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DistributedOperationMarker {
    format_version: u32,
    operation_id: String,
    row_count: u64,
    write_mode: String,
}

#[cfg(feature = "vane-distributed")]
#[derive(Debug)]
enum DistributedCommitError {
    Known(lance::Error),
    OutcomeUnknown(String),
}

#[cfg(feature = "vane-distributed")]
enum DistributedValidationError {
    Known(lance::Error),
    Reconciliation(lance::Error),
}

#[cfg(feature = "vane-distributed")]
impl From<lance::Error> for DistributedCommitError {
    fn from(value: lance::Error) -> Self {
        Self::Known(value)
    }
}

#[cfg(feature = "vane-distributed")]
fn distributed_reconciliation_error(
    operation_id: &str,
    error: lance::Error,
) -> DistributedCommitError {
    DistributedCommitError::OutcomeUnknown(format!(
        "could not determine whether distributed Lance operation '{operation_id}' was already committed: {error}"
    ))
}

#[cfg(feature = "vane-distributed")]
fn distributed_abort_failure(error: impl std::fmt::Display) -> FfiError {
    FfiError::new(
        ErrorCode::DatasetCommitOutcomeUnknown,
        format!(
            "distributed Lance abort could not prove or complete orphan cleanup: {error}; cleanup is incomplete"
        ),
    )
}

fn is_definitive_commit_failure(error: &lance::Error) -> bool {
    !lance_mutation_outcome_unknown(error)
}

#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
async fn find_committed_distributed_operation_in_history(
    dataset: Option<&Dataset>,
    operation_id: &str,
    recent_transactions: usize,
) -> Result<Option<DistributedOperationMarker>, lance::Error> {
    let Some(dataset) = dataset else {
        return Ok(None);
    };
    for transaction in dataset.get_transactions(recent_transactions).await? {
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

#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
fn validate_distributed_identity(value: &str, name: &str) -> FfiResult<()> {
    if value.is_empty() {
        Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("distributed Lance {name} must not be empty"),
        ))
    } else {
        Ok(())
    }
}

#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
fn distributed_operation_marker_path(
    base: &Path,
    operation_id: &str,
) -> Result<Path, lance::Error> {
    path_join(
        base,
        &format!("_vane_operations/{}.json", hex_identity(operation_id)),
    )
}

#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
fn dataset_references_distributed_operation(dataset: &Dataset, operation_id: &str) -> bool {
    let prefix = distributed_destination_operation_prefix(operation_id);
    dataset.manifest().fragments.iter().any(|fragment| {
        fragment
            .files
            .iter()
            .any(|file| file.base_id.is_none() && file.path.starts_with(&prefix))
    })
}

#[cfg(feature = "vane-distributed")]
async fn distributed_operation_destination_files_exist(
    store: &ObjectStore,
    base: &Path,
    operation_id: &str,
) -> Result<bool, lance::Error> {
    let data = path_join(base, "data")?;
    let prefix = distributed_destination_operation_prefix(operation_id);
    let objects = match store.list(Some(data.clone())).try_collect::<Vec<_>>().await {
        Ok(objects) => objects,
        Err(err) if err.is_not_found() => return Ok(false),
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
        if relative.starts_with(&prefix) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(feature = "vane-distributed")]
async fn find_committed_distributed_operation(
    dataset: Option<&Dataset>,
    store: &ObjectStore,
    base: &Path,
    operation_id: &str,
    persist_marker: bool,
) -> Result<Option<DistributedOperationMarker>, lance::Error> {
    if let Some(marker) = read_distributed_operation_marker(store, base, operation_id).await? {
        return Ok(Some(marker));
    }
    let Some(dataset) = dataset else {
        return Ok(None);
    };
    // Always inspect the newest transaction.  This is the expected recovery
    // path while the coordinator retains the mutation lease, and it is needed
    // for an empty CREATE/OVERWRITE whose commit has no deterministic data
    // files to trigger the broader history scan below.
    if let Some(marker) =
        find_committed_distributed_operation_in_history(Some(dataset), operation_id, 1).await?
    {
        if persist_marker {
            write_distributed_operation_marker(store, base, &marker).await?;
        }
        return Ok(Some(marker));
    }
    // A new operation has neither a marker nor deterministic destination
    // files. Avoid scanning every historical transaction in that overwhelmingly
    // common case. Files that are no longer in the current manifest still
    // require a history lookup: a later overwrite must not turn a committed
    // operation into an apparently safe replay or cleanup candidate.
    if !dataset_references_distributed_operation(dataset, operation_id)
        && !distributed_operation_destination_files_exist(store, base, operation_id).await?
    {
        return Ok(None);
    }
    let transaction_count = usize::try_from(dataset.version_id()).unwrap_or(usize::MAX);
    if let Some(marker) = find_committed_distributed_operation_in_history(
        Some(dataset),
        operation_id,
        transaction_count,
    )
    .await?
    {
        // Backfill the durable marker while the vacuumable transaction is still
        // available.  A failure is fail-closed: callers must not replay or clean
        // up an operation whose commit outcome is already known.
        if persist_marker {
            write_distributed_operation_marker(store, base, &marker).await?;
        }
        return Ok(Some(marker));
    }
    Err(lance::Error::internal(format!(
        "distributed Lance operation '{operation_id}' has deterministic data files but no durable commit marker or transaction; refusing unsafe replay"
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

#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
fn distributed_staging_operation_path(
    base: &Path,
    operation_id: &str,
) -> Result<Path, lance::Error> {
    path_join(
        base,
        &format!("_vane_staging/{}", hex_identity(operation_id)),
    )
}

#[cfg(feature = "vane-distributed")]
fn distributed_destination_operation_prefix(operation_id: &str) -> String {
    format!("vane_{}_", hex_identity(operation_id))
}

#[cfg(feature = "vane-distributed")]
fn distributed_destination_task_prefix(operation_id: &str, task_attempt_id: &str) -> String {
    format!(
        "{}{}_",
        distributed_destination_operation_prefix(operation_id),
        hex_identity(task_attempt_id)
    )
}

#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
async fn copy_distributed_task_data(
    store: &ObjectStore,
    base: &Path,
    operation_id: &str,
    task_attempt_id: &str,
    expected_files: &HashSet<String>,
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
    let mut copied_files = HashSet::with_capacity(expected_files.len());
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
        if !expected_files.contains(relative) {
            continue;
        }
        let destination_file = path_join(&destination, &format!("{destination_prefix}{relative}"))?;
        store.copy(&object.location, &destination_file).await?;
        copied_files.insert(relative.to_string());
    }
    if copied_files != *expected_files {
        let mut missing = expected_files
            .difference(&copied_files)
            .cloned()
            .collect::<Vec<_>>();
        missing.sort();
        return Err(lance::Error::invalid_input(format!(
            "distributed Lance task '{task_attempt_id}' is missing staged data file(s): {}",
            missing.join(", ")
        )));
    }
    Ok(())
}

#[cfg(feature = "vane-distributed")]
struct DistributedStagingTask {
    task_id: String,
    fragments: Vec<Fragment>,
    schema: LanceSchema,
    null_vector_fields: HashSet<String>,
}

#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
fn variable_vector_type(field: &LanceField) -> Option<VectorElementType> {
    is_variable_list_vector_type(&field.data_type()).map(|(_, element_type)| element_type)
}

#[cfg(feature = "vane-distributed")]
fn promote_distributed_vector_schema(canonical: &mut LanceSchema, candidate: &LanceSchema) {
    for canonical_field in &mut canonical.fields {
        let Some(candidate_field) = candidate
            .fields
            .iter()
            .find(|field| field.name == canonical_field.name)
        else {
            continue;
        };
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

#[cfg(feature = "vane-distributed")]
fn collect_field_id_mapping_by_name(
    source_fields: &[LanceField],
    canonical_fields: &[LanceField],
    omitted_field_ids: &HashSet<i32>,
    mapping: &mut HashMap<i32, i32>,
) -> Result<(), lance::Error> {
    for source in source_fields {
        if omitted_field_ids.contains(&source.id) {
            continue;
        }
        let canonical = canonical_fields
            .iter()
            .find(|field| field.name == source.name)
            .ok_or_else(|| {
                lance::Error::invalid_input(format!(
                    "cannot map distributed Lance field '{}' to the canonical schema",
                    source.name
                ))
            })?;
        collect_field_id_mapping(source, canonical, mapping)?;
    }
    Ok(())
}

#[cfg(feature = "vane-distributed")]
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
    for source_child in &source.children {
        let canonical_child = canonical
            .children
            .iter()
            .find(|field| field.name == source_child.name)
            .ok_or_else(|| {
                lance::Error::invalid_input(format!(
                    "cannot map distributed Lance child field '{}' to canonical field '{}'",
                    source_child.name, canonical.name
                ))
            })?;
        collect_field_id_mapping(source_child, canonical_child, mapping)?;
    }
    Ok(())
}

#[cfg(feature = "vane-distributed")]
fn collect_field_ids(field: &LanceField, field_ids: &mut HashSet<i32>) {
    field_ids.insert(field.id);
    for child in &field.children {
        collect_field_ids(child, field_ids);
    }
}

#[cfg(feature = "vane-distributed")]
fn canonicalize_distributed_task_fragments(
    task: &mut DistributedStagingTask,
    canonical: &LanceSchema,
    compare_metadata: bool,
    compare_dictionary: bool,
) -> Result<(), lance::Error> {
    for field_name in &task.null_vector_fields {
        let Some(field) = task
            .schema
            .fields
            .iter()
            .find(|field| field.name == *field_name)
        else {
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

    let mut normalized = task.schema.clone();
    let mut omitted_field_ids = HashSet::new();
    for source_field in &task.schema.fields {
        let Some(canonical_field) = canonical
            .fields
            .iter()
            .find(|field| field.name == source_field.name)
        else {
            continue;
        };
        let Some(normalized_field) = normalized
            .fields
            .iter_mut()
            .find(|field| field.name == source_field.name)
        else {
            continue;
        };
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
        compare_dictionary,
        compare_field_ids: false,
        // Match Lance's InsertBuilder append validation.  DuckDB exports
        // nullable columns even when an existing Lance field is non-nullable;
        // that is safe because the actual values are still validated by the
        // writer.  Field order is likewise not part of append compatibility.
        compare_nullability: NullabilityComparison::Ignore,
        allow_missing_if_nullable: true,
        ignore_field_order: true,
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
    collect_field_id_mapping_by_name(
        &task.schema.fields,
        &canonical.fields,
        &omitted_field_ids,
        &mut field_id_mapping,
    )?;

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

#[cfg(feature = "vane-distributed")]
fn canonicalize_distributed_staging_tasks(
    tasks: &mut [DistributedStagingTask],
    target_schema: Option<&LanceSchema>,
    compare_dictionary: bool,
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
        canonicalize_distributed_task_fragments(
            task,
            &canonical,
            target_schema.is_none(),
            compare_dictionary,
        )?;
    }
    Ok(canonical)
}

// The coordinator-facing distributed write entry points are part of the Vane
// adapter ABI. The transaction and cleanup primitives remain shared by the
// ordinary DuckDB writer, but these symbols stay out of an official DuckDB
// build so the OFF mode has no Vane-only C ABI surface.
#[cfg(feature = "vane-distributed")]
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

#[cfg(feature = "vane-distributed")]
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
    validate_distributed_identity(&operation_id, "operation_id")?;
    let storage_options =
        unsafe { distributed_storage_options_from_ffi(option_keys, option_values, options_len)? };
    let session = unsafe { optional_session_handle(session)? };

    match runtime::block_on(async {
        let dataset = load_optional_distributed_dataset(&path, &storage_options, session.clone())
            .await
            .map_err(DistributedValidationError::Known)?;
        let (store, base) =
            resolve_distributed_object_store(&path, &storage_options, session.as_ref())
                .await
                .map_err(DistributedValidationError::Known)?;
        if let Some(marker) = find_committed_distributed_operation(
            dataset.as_ref(),
            store.as_ref(),
            &base,
            &operation_id,
            false,
        )
        .await
        .map_err(DistributedValidationError::Reconciliation)?
        {
            if marker.write_mode != mode {
                return Err(DistributedValidationError::Known(lance::Error::internal(
                    format!(
                        "distributed Lance operation '{operation_id}' was already committed in '{}' mode, not '{mode}'",
                        marker.write_mode
                    ),
                )));
            }
            return Ok(());
        }
        if mode == "create" && dataset.is_some() {
            return Err(DistributedValidationError::Known(
                lance::Error::invalid_input(format!("Lance dataset already exists: {path}")),
            ));
        }
        if mode == "append" && dataset.is_none() {
            return Err(DistributedValidationError::Known(
                lance::Error::invalid_input(format!(
                    "cannot append because the Lance dataset does not exist: {path}"
                )),
            ));
        }
        Ok::<(), DistributedValidationError>(())
    }) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(DistributedValidationError::Known(err))) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            format!("distributed Lance write validation: {err}"),
        )),
        Ok(Err(DistributedValidationError::Reconciliation(err))) => Err(FfiError::new(
            ErrorCode::DatasetCommitOutcomeUnknown,
            format!(
                "distributed Lance validation could not reconcile operation '{operation_id}': {err}"
            ),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[cfg(feature = "vane-distributed")]
#[ffi_guard_macro::ffi_guard(dataset_mutation)]
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
#[cfg(feature = "vane-distributed")]
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
    unsafe {
        ptr::write_unaligned(out_rows, 0);
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
    validate_distributed_identity(&operation_id, "operation_id")?;
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
    let mut unique_task_ids = HashSet::with_capacity(transaction_count);
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
        validate_distributed_identity(&task_id, &format!("task_attempt_ids[{idx}]"))?;
        if !unique_task_ids.insert(task_id.clone()) {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("duplicate distributed Lance task attempt id '{task_id}'"),
            ));
        }
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
            true,
        )
        .await
        .map_err(|error| distributed_reconciliation_error(&operation_id, error))?
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
        if mode == "append" && existing.is_none() {
            return Err(lance::Error::invalid_input(format!(
                "cannot append because the Lance dataset does not exist: {path}"
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
        let mut transaction_rows = 0_u64;
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
            let mut task_files = HashSet::new();
            for fragment in &fragments {
                if !fragment.overlays.is_empty()
                    || fragment.deletion_file.is_some()
                    || fragment.row_id_meta.is_some()
                {
                    return Err(lance::Error::invalid_input(format!(
                        "worker transaction for task '{task_id}' contains unsupported overlay, deletion, or row-id metadata"
                    ))
                    .into());
                }
                let fragment_rows = fragment.physical_rows.ok_or_else(|| {
                    lance::Error::invalid_input(format!(
                        "worker transaction for task '{task_id}' has a fragment with unknown row count"
                    ))
                })?;
                transaction_rows = transaction_rows
                    .checked_add(u64::try_from(fragment_rows).map_err(|_| {
                        lance::Error::invalid_input(format!(
                            "worker transaction for task '{task_id}' has an invalid row count"
                        ))
                    })?)
                    .ok_or_else(|| {
                        lance::Error::invalid_input(
                            "distributed Lance transaction row count overflow".to_string(),
                        )
                    })?;
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
                    if !task_files.insert(file.path.clone()) {
                        return Err(lance::Error::invalid_input(format!(
                            "worker transaction for task '{task_id}' references duplicate data file '{}'",
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

        if transaction_rows != selected_rows {
            return Err(lance::Error::invalid_input(format!(
                "distributed Lance worker transactions contain {transaction_rows} rows, not the selected row count {selected_rows}"
            ))
            .into());
        }

        let target_schema = if mode == "append" {
            existing.as_ref().map(Dataset::schema)
        } else {
            None
        };
        let compare_dictionary = existing
            .as_ref()
            .is_some_and(|dataset| dataset.manifest().should_use_legacy_format());
        let schema = canonicalize_distributed_staging_tasks(
            &mut staging_tasks,
            target_schema,
            compare_dictionary,
        )?;

        let mut combined_fragments = Vec::new();
        for mut task in staging_tasks {
            let expected_files = task
                .fragments
                .iter()
                .flat_map(|fragment| fragment.files.iter())
                .map(|file| file.path.clone())
                .collect::<HashSet<_>>();
            copy_distributed_task_data(
                store.as_ref(),
                &base,
                &operation_id,
                &task.task_id,
                &expected_files,
            )
            .await?;
            let file_prefix = distributed_destination_task_prefix(&operation_id, &task.task_id);
            for fragment in &mut task.fragments {
                for file in &mut fragment.files {
                    file.path = format!("{file_prefix}{}", file.path);
                }
            }
            combined_fragments.extend(task.fragments);
        }

        let operation = if mode == "append" {
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
                true,
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
            validate_distributed_operation_marker(&marker, &operation_id, selected_rows, &mode)
                .map_err(|validation_error| {
                    DistributedCommitError::OutcomeUnknown(format!(
                        "manifest commit returned '{commit_error}', but the recovered operation metadata conflicts with this request: {validation_error}"
                    ))
                })?;
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
                    if let Ok(Some(durable)) =
                        read_distributed_operation_marker(store.as_ref(), &base, &operation_id)
                            .await
                    {
                        if let Err(validation_error) = validate_distributed_operation_marker(
                            &durable,
                            &operation_id,
                            selected_rows,
                            &mode,
                        ) {
                            return Err(DistributedCommitError::OutcomeUnknown(format!(
                                "manifest commit succeeded for operation '{operation_id}', but the durable idempotency marker conflicts with it: {validation_error}"
                            )));
                        }
                        marker_error = None;
                        break;
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
                ptr::write_unaligned(out_rows, rows);
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

#[cfg(feature = "vane-distributed")]
#[ffi_guard_macro::ffi_guard(dataset_mutation)]
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

#[cfg(feature = "vane-distributed")]
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
    validate_distributed_identity(&operation_id, "operation_id")?;
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
            true,
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
        Ok(Err(err)) => Err(distributed_abort_failure(err)),
        Err(err) => Err(distributed_abort_failure(format!("runtime: {err}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::CString;

    #[cfg(feature = "vane-distributed")]
    use arrow_array::new_null_array;
    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::Field;
    #[cfg(feature = "vane-distributed")]
    use lance::dataset::NewColumnTransform;
    use lance_core::utils::address::RowAddress;
    #[cfg(feature = "vane-distributed")]
    use lance_table::format::DataFile;
    use roaring::RoaringTreemap;

    use super::super::update::apply_deletions;

    #[test]
    fn committed_writer_errors_preserve_outcome_unknown_classification() {
        assert!(is_definitive_commit_failure(&lance::Error::invalid_input(
            "known validation failure"
        )));
        assert!(!is_definitive_commit_failure(&lance::Error::internal(
            "object-store acknowledgement was lost"
        )));

        let known = writer_thread_failure(
            WriterKind::Committed,
            ErrorCode::DatasetWriteFinish,
            WriterThreadError {
                message: "known validation failure".to_string(),
                outcome_unknown: false,
            },
        );
        assert_eq!(known.code as i32, ErrorCode::DatasetWriteFinish as i32);

        let unknown = writer_thread_failure(
            WriterKind::Committed,
            ErrorCode::DatasetWriteFinish,
            WriterThreadError {
                message: "object-store acknowledgement was lost".to_string(),
                outcome_unknown: true,
            },
        );
        assert_eq!(
            unknown.code as i32,
            ErrorCode::DatasetCommitOutcomeUnknown as i32
        );
        assert!(unknown.message.contains("outcome is unknown"));
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
    fn distributed_reconciliation_failure_is_outcome_unknown() {
        let error = distributed_reconciliation_error(
            "operation-1",
            lance::Error::internal("marker history is unavailable"),
        );

        match error {
            DistributedCommitError::OutcomeUnknown(message) => {
                assert!(message.contains("operation-1"));
                assert!(message.contains("already committed"));
            }
            DistributedCommitError::Known(error) => {
                panic!("reconciliation failure was incorrectly retryable: {error}")
            }
        }
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
    fn distributed_abort_failure_reports_incomplete_cleanup() {
        let error = distributed_abort_failure("object-store delete failed");

        assert_eq!(
            error.code as i32,
            ErrorCode::DatasetCommitOutcomeUnknown as i32
        );
        assert!(error.message.contains("cleanup is incomplete"));
    }

    #[test]
    fn row_level_transactions_reject_lossy_serialization() {
        let transaction = Transaction::new(
            1,
            Operation::Delete {
                updated_fragments: Vec::new(),
                deleted_fragment_ids: Vec::new(),
                predicate: "false".to_string(),
            },
            None,
        );
        let mut row_addrs = RoaringTreemap::new();
        row_addrs.insert(u64::from(RowAddress::new_from_parts(0, 0)));
        let transaction = Box::into_raw(Box::new(VaneTransaction::with_affected_rows(
            transaction,
            RowAddrTreeMap::from(row_addrs),
        ))) as *mut c_void;
        let mut data = ptr::null_mut();
        let mut len = 0_usize;

        let error = serialize_transaction_inner(transaction, &mut data, &mut len).unwrap_err();
        assert_eq!(error.code as i32, ErrorCode::InvalidArgument as i32);
        assert!(error.message.contains("affected-row"));
        assert!(data.is_null());
        assert_eq!(len, 0);

        unsafe { lance_free_transaction(transaction) };
    }

    #[test]
    fn vector_conversion_preserves_field_and_schema_metadata() {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let field = Field::new("vector", DataType::List(item), true).with_metadata(HashMap::from(
            [("field-key".to_string(), "field-value".to_string())],
        ));
        let input = Arc::new(Schema::new_with_metadata(
            vec![field],
            HashMap::from([("schema-key".to_string(), "schema-value".to_string())]),
        ));
        let conversions = vec![VectorConversion {
            col_idx: 0,
            field_name: "vector".to_string(),
            dim: 3,
            explicit_dim: true,
            list_kind: VectorListKind::List,
            element_type: VectorElementType::Float32,
        }];

        let output = build_output_schema(&input, &conversions).unwrap();
        assert_eq!(output.metadata(), input.metadata());
        assert_eq!(output.field(0).metadata(), input.field(0).metadata());
        assert_eq!(
            output.field(0).data_type(),
            &DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3,)
        );
    }

    #[cfg(feature = "vane-distributed")]
    fn lance_schema(fields: Vec<Field>) -> LanceSchema {
        let arrow_schema = Schema::new(fields);
        let mut schema = LanceSchema::try_from(&arrow_schema).unwrap();
        schema.set_field_id(None);
        schema
    }

    #[cfg(feature = "vane-distributed")]
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

    #[cfg(feature = "vane-distributed")]
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

    #[cfg(feature = "vane-distributed")]
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

    fn two_element_list_batch() -> RecordBatch {
        let mut builder = arrow_array::builder::ListBuilder::new(Float32Builder::new());
        builder.values().append_value(1.0);
        builder.values().append_value(2.0);
        builder.append(true);
        let vector = builder.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            vector.data_type().clone(),
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(vector)]).unwrap()
    }

    #[cfg(feature = "vane-distributed")]
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
        )
        .unwrap();
        sender.send(partial).unwrap();
        aborted.store(true, Ordering::Release);
        drop(sender);

        let error = match join.join().unwrap() {
            Err(error) => error,
            Ok(_) => panic!("aborted writer unexpectedly committed"),
        };
        assert!(
            error.message.contains("aborted"),
            "unexpected writer error: {error:?}"
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
    fn conversion_failure_does_not_start_or_commit_writer() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-conversion-failure-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let batch = two_element_list_batch();
        let schema = batch.schema();
        let data_type = DataType::Struct(schema.fields().clone());
        let mut handle = Box::new(WriterHandle {
            input_schema: schema,
            data_type,
            state: Mutex::new(WriterState {
                kind: WriterKind::Committed,
                path: dataset_path.to_string_lossy().into_owned(),
                params: WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                },
                finished: false,
                vector_candidates: vec![VectorConversion {
                    col_idx: 0,
                    field_name: "vector".to_string(),
                    dim: 3,
                    explicit_dim: true,
                    list_kind: VectorListKind::List,
                    element_type: VectorElementType::Float32,
                }],
                buffered_batches: vec![batch],
                buffered_rows: 1,
                output_schema: None,
                output_sender: None,
                output_join: None,
            }),
            batches_sent: AtomicU64::new(1),
            aborted: Arc::new(AtomicBool::new(false)),
        });

        for _ in 0..2 {
            let error = writer_finish_inner(handle.as_mut() as *mut WriterHandle as *mut c_void)
                .expect_err("mismatched list dimensions must fail conversion");
            assert!(
                error
                    .message
                    .contains("vector dim mismatch: expected 3 got 2"),
                "{error:?}"
            );
        }

        // A writer started before conversion would observe a clean, empty
        // channel and create an empty dataset while this handle remains live.
        for _ in 0..100 {
            if dataset_path.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(!dataset_path.exists());
        drop(handle);
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
    }

    #[test]
    fn committed_writer_cannot_be_finished_twice() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-double-finish-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let batch = int64_batch(&[("id", 7)]);
        let schema = batch.schema();
        let data_type = DataType::Struct(schema.fields().clone());
        let mut handle = Box::new(WriterHandle {
            input_schema: schema,
            data_type,
            state: Mutex::new(WriterState {
                kind: WriterKind::Committed,
                path: dataset_path.to_string_lossy().into_owned(),
                params: WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                },
                finished: false,
                vector_candidates: Vec::new(),
                buffered_batches: vec![batch],
                buffered_rows: 1,
                output_schema: None,
                output_sender: None,
                output_join: None,
            }),
            batches_sent: AtomicU64::new(1),
            aborted: Arc::new(AtomicBool::new(false)),
        });

        let writer = handle.as_mut() as *mut WriterHandle as *mut c_void;
        writer_finish_inner(writer).unwrap();
        let error = writer_finish_inner(writer).unwrap_err();
        assert!(error.message.contains("already finished"));

        let rows = runtime::block_on(async {
            DatasetBuilder::from_uri(dataset_path.to_string_lossy().as_ref())
                .load()
                .await?
                .count_rows(None)
                .await
        })
        .unwrap()
        .unwrap();
        assert_eq!(rows, 1);

        drop(handle);
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
        for relative in &candidate_paths.primary_paths {
            assert!(dataset_path.join(relative).exists(), "missing {relative}");
        }

        let path = CString::new(path_text.clone()).unwrap();
        let transaction = Box::into_raw(Box::new(VaneTransaction::new(transaction))) as *mut c_void;
        abort_transaction_with_storage_options_inner(
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            transaction,
        )
        .unwrap();

        for relative in &candidate_paths.primary_paths {
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

    #[test]
    fn aborting_stale_delete_preserves_files_from_its_read_version() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-abort-stale-delete-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let initial_batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1_i64, 2]))]).unwrap();
        runtime::block_on(async {
            InsertBuilder::new(path_text.as_str())
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![initial_batch])
                .await
        })
        .unwrap()
        .unwrap();

        let (transaction, base_paths) = runtime::block_on(async {
            let dataset = DatasetBuilder::from_uri(path_text.as_str()).load().await?;
            let fragment_id = u32::try_from(dataset.get_fragments()[0].id()).unwrap();
            let mut deleted_rows = RoaringTreemap::new();
            deleted_rows.insert(u64::from(RowAddress::new_from_parts(fragment_id, 0)));
            let (updated_fragments, deleted_fragment_ids) =
                apply_deletions(&dataset, &deleted_rows)
                    .await
                    .map_err(|error| lance::Error::internal(error.message))?;
            let transaction = Transaction::new(
                dataset.version().version,
                Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids,
                    predicate: "id = 1".to_string(),
                },
                None,
            );
            let mut base_paths = TransactionOwnedPaths::default();
            for fragment in dataset.manifest().fragments.iter() {
                collect_fragment_owned_paths(fragment, &mut base_paths);
            }
            Ok::<_, lance::Error>((transaction, base_paths))
        })
        .unwrap()
        .unwrap();

        let candidate_paths = collect_transaction_owned_paths(&transaction);
        let orphan_paths = candidate_paths
            .primary_paths
            .difference(&base_paths.primary_paths)
            .cloned()
            .collect::<Vec<_>>();
        assert!(!base_paths.primary_paths.is_empty());
        assert!(!orphan_paths.is_empty());

        // Make the delete transaction stale and remove its original data file
        // from the latest manifest. Cleanup must still protect version 1.
        runtime::block_on(async {
            InsertBuilder::new(path_text.as_str())
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![int64_batch(&[("id", 2)])])
                .await
        })
        .unwrap()
        .unwrap();

        let path = CString::new(path_text.clone()).unwrap();
        let transaction = Box::into_raw(Box::new(VaneTransaction::new(transaction))) as *mut c_void;
        abort_transaction_with_storage_options_inner(
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            transaction,
        )
        .unwrap();

        for relative in &base_paths.primary_paths {
            assert!(
                dataset_path.join(relative).exists(),
                "read-version file was deleted: {relative}"
            );
        }
        for relative in &orphan_paths {
            assert!(
                !dataset_path.join(relative).exists(),
                "delete orphan survived abort: {relative}"
            );
        }
        let version_one_rows = runtime::block_on(async {
            let dataset = DatasetBuilder::from_uri(path_text.as_str()).load().await?;
            dataset.checkout_version(1).await?.count_rows(None).await
        })
        .unwrap()
        .unwrap();
        assert_eq!(version_one_rows, 2);

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn abort_argument_failure_reports_incomplete_cleanup_after_consuming_transaction() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-abort-argument-cleanup-{}-{}",
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

        let path = CString::new(path_text).unwrap();
        let transaction = Box::into_raw(Box::new(VaneTransaction::new(transaction))) as *mut c_void;
        let error = abort_transaction_with_storage_options_inner(
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            1,
            ptr::null_mut(),
            transaction,
        )
        .expect_err("invalid cleanup arguments must fail closed");

        assert_eq!(
            error.code as i32,
            ErrorCode::DatasetCommitOutcomeUnknown as i32
        );
        assert!(error.message.contains("cleanup is incomplete"));
        for relative in &candidate_paths.primary_paths {
            assert!(
                dataset_path.join(relative).exists(),
                "test precondition: cleanup must not have run for {relative}"
            );
        }

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn definitive_commit_failure_removes_uncommitted_files() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-failed-commit-cleanup-{}-{}",
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

        let mut transaction = runtime::block_on(async {
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
        for relative in &candidate_paths.primary_paths {
            assert!(dataset_path.join(relative).exists(), "missing {relative}");
        }

        // Point the transaction at a manifest version that cannot exist. Lance
        // rejects this before publishing a new manifest, so the commit wrapper
        // must remove the files that the consumed transaction exclusively owns.
        transaction.read_version = u64::MAX;
        let path = CString::new(path_text.clone()).unwrap();
        let transaction = Box::into_raw(Box::new(VaneTransaction::new(transaction))) as *mut c_void;
        let error = commit_transaction_inner(
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            transaction,
        )
        .unwrap_err();
        assert_eq!(
            error.code as i32,
            ErrorCode::DatasetCommitTransaction as i32,
            "unexpected commit error: {error:?}"
        );

        for relative in &candidate_paths.primary_paths {
            assert!(
                !dataset_path.join(relative).exists(),
                "orphan survived definitive commit failure: {relative}"
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

    #[test]
    fn disjoint_row_mutations_rebase_with_affected_rows() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-affected-rows-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1_i64, 2, 3]))])
                .unwrap();
        runtime::block_on(async {
            InsertBuilder::new(path_text.as_str())
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![batch])
                .await
        })
        .unwrap()
        .unwrap();

        let (first, first_rows, second, second_rows) = runtime::block_on(async {
            let first_dataset = DatasetBuilder::from_uri(path_text.as_str()).load().await?;
            let second_dataset = DatasetBuilder::from_uri(path_text.as_str()).load().await?;
            let fragment_id = u32::try_from(first_dataset.get_fragments()[0].id()).unwrap();

            let mut first_rows = RoaringTreemap::new();
            first_rows.insert(u64::from(RowAddress::new_from_parts(fragment_id, 0)));
            let (first_updated, first_removed) = apply_deletions(&first_dataset, &first_rows)
                .await
                .map_err(|error| lance::Error::internal(error.message))?;
            let first = Transaction::new(
                first_dataset.version().version,
                Operation::Delete {
                    updated_fragments: first_updated,
                    deleted_fragment_ids: first_removed,
                    predicate: "id = 1".to_string(),
                },
                None,
            );

            let mut second_rows = RoaringTreemap::new();
            second_rows.insert(u64::from(RowAddress::new_from_parts(fragment_id, 1)));
            let (second_updated, second_removed) = apply_deletions(&second_dataset, &second_rows)
                .await
                .map_err(|error| lance::Error::internal(error.message))?;
            let second = Transaction::new(
                second_dataset.version().version,
                Operation::Delete {
                    updated_fragments: second_updated,
                    deleted_fragment_ids: second_removed,
                    predicate: "id = 2".to_string(),
                },
                None,
            );
            Ok::<_, lance::Error>((first, first_rows, second, second_rows))
        })
        .unwrap()
        .unwrap();

        let path = CString::new(path_text.clone()).unwrap();
        for transaction in [
            VaneTransaction::with_affected_rows(first, RowAddrTreeMap::from(first_rows)),
            VaneTransaction::with_affected_rows(second, RowAddrTreeMap::from(second_rows)),
        ] {
            let transaction = Box::into_raw(Box::new(transaction)) as *mut c_void;
            commit_transaction_inner(
                path.as_ptr(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null_mut(),
                transaction,
            )
            .unwrap();
        }

        let dataset =
            runtime::block_on(async { DatasetBuilder::from_uri(path_text.as_str()).load().await })
                .unwrap()
                .unwrap();
        assert_eq!(dataset.version().version, 3);
        assert_eq!(
            runtime::block_on(dataset.count_rows(None))
                .unwrap()
                .unwrap(),
            1
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(feature = "vane-distributed")]
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
    #[cfg(feature = "vane-distributed")]
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
            canonicalize_distributed_staging_tasks(&mut tasks, Some(&target_schema), false)
                .unwrap();

        assert_eq!(canonical.field_ids(), vec![4, 9]);
        assert_eq!(tasks[0].fragments[0].files[0].fields.as_ref(), &[4, 9]);
        assert_eq!(
            tasks[0].fragments[0].files[0].column_indices.as_ref(),
            &[0, 1]
        );
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
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
        let canonical = canonicalize_distributed_staging_tasks(&mut tasks, None, false).unwrap();

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
    #[cfg(feature = "vane-distributed")]
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

        let error = canonicalize_distributed_staging_tasks(&mut tasks, None, false).unwrap_err();
        assert!(error.to_string().contains("inferred variable list field"));
    }

    #[tokio::test]
    #[cfg(feature = "vane-distributed")]
    async fn distributed_operation_marker_is_found_without_dataset_history() {
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
        let loaded = find_committed_distributed_operation(
            None,
            store.as_ref(),
            &base,
            &marker.operation_id,
            true,
        )
        .await
        .unwrap();

        assert_eq!(loaded, Some(marker));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
    fn distributed_operation_is_recovered_after_a_later_overwrite() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-operation-history-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let operation_id_text = "operation-history";
        let task_id_text = "task-history";
        let staging_path = format!(
            "{path_text}/_vane_staging/{}/{}",
            hex_identity(operation_id_text),
            hex_identity(task_id_text)
        );
        let transaction = write_staging_transaction(&staging_path, int64_batch(&[("id", 1)]));
        let transaction = pb::Transaction::from(&transaction).encode_to_vec();
        let path = CString::new(path_text.clone()).unwrap();
        let mode = CString::new("create").unwrap();
        let operation_id = CString::new(operation_id_text).unwrap();
        let task_ids = [CString::new(task_id_text).unwrap()];
        assert_eq!(
            commit_distributed_transactions(
                &path,
                &mode,
                &operation_id,
                &task_ids,
                &[transaction],
                1,
            ),
            1
        );

        let dataset = runtime::block_on(async {
            InsertBuilder::new(path_text.as_str())
                .with_params(&WriteParams {
                    mode: WriteMode::Overwrite,
                    ..Default::default()
                })
                .execute(vec![int64_batch(&[("id", 2)])])
                .await
        })
        .unwrap()
        .unwrap();
        assert!(!dataset_references_distributed_operation(
            &dataset,
            operation_id_text
        ));

        let (store, base) = runtime::block_on(resolve_distributed_object_store(
            &path_text,
            &HashMap::new(),
            None,
        ))
        .unwrap()
        .unwrap();
        let marker_path = distributed_operation_marker_path(&base, operation_id_text).unwrap();
        runtime::block_on(store.delete(&marker_path))
            .unwrap()
            .unwrap();

        let recovered = runtime::block_on(find_committed_distributed_operation(
            Some(&dataset),
            store.as_ref(),
            &base,
            operation_id_text,
            true,
        ))
        .unwrap()
        .unwrap()
        .expect("the historical commit should be recovered");
        assert_eq!(recovered.row_count, 1);
        assert_eq!(recovered.write_mode, "create");
        assert!(runtime::block_on(read_distributed_operation_marker(
            store.as_ref(),
            &base,
            operation_id_text,
        ))
        .unwrap()
        .unwrap()
        .is_some());

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
    fn empty_distributed_operation_is_recovered_from_latest_transaction() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-empty-operation-history-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let operation_id_text = "operation-empty-history";
        let task_id_text = "task-empty-history";
        let staging_path = format!(
            "{path_text}/_vane_staging/{}/{}",
            hex_identity(operation_id_text),
            hex_identity(task_id_text)
        );
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let transaction = write_staging_transaction(&staging_path, RecordBatch::new_empty(schema));
        let transaction = pb::Transaction::from(&transaction).encode_to_vec();
        let path = CString::new(path_text.clone()).unwrap();
        let mode = CString::new("create").unwrap();
        let operation_id = CString::new(operation_id_text).unwrap();
        let task_ids = [CString::new(task_id_text).unwrap()];
        assert_eq!(
            commit_distributed_transactions(
                &path,
                &mode,
                &operation_id,
                &task_ids,
                &[transaction],
                0,
            ),
            0
        );

        let dataset = runtime::block_on(DatasetBuilder::from_uri(path_text.as_str()).load())
            .unwrap()
            .unwrap();
        let (store, base) = runtime::block_on(resolve_distributed_object_store(
            &path_text,
            &HashMap::new(),
            None,
        ))
        .unwrap()
        .unwrap();
        let marker_path = distributed_operation_marker_path(&base, operation_id_text).unwrap();
        runtime::block_on(store.delete(&marker_path))
            .unwrap()
            .unwrap();

        let recovered = runtime::block_on(find_committed_distributed_operation(
            Some(&dataset),
            store.as_ref(),
            &base,
            operation_id_text,
            true,
        ))
        .unwrap()
        .unwrap()
        .expect("the latest empty commit should be recovered");
        assert_eq!(recovered.row_count, 0);
        assert_eq!(recovered.write_mode, "create");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
    fn distributed_commit_rejects_selected_row_count_mismatch() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-distributed-row-count-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let operation_id_text = "operation-row-count";
        let task_id_text = "task-row-count";
        let staging_path = format!(
            "{path_text}/_vane_staging/{}/{}",
            hex_identity(operation_id_text),
            hex_identity(task_id_text)
        );
        let transaction = write_staging_transaction(&staging_path, int64_batch(&[("id", 1)]));
        let transaction = pb::Transaction::from(&transaction).encode_to_vec();
        let path = CString::new(path_text).unwrap();
        let mode = CString::new("create").unwrap();
        let operation_id = CString::new(operation_id_text).unwrap();
        let task_id = CString::new(task_id_text).unwrap();
        let task_ids = [task_id.as_ptr()];
        let transaction_data = [transaction.as_ptr()];
        let transaction_lens = [transaction.len()];
        let mut out_rows = 0;

        let error = distributed_write_commit_inner(
            path.as_ptr(),
            mode.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            operation_id.as_ptr(),
            task_ids.as_ptr(),
            transaction_data.as_ptr(),
            transaction_lens.as_ptr(),
            1,
            2,
            &mut out_rows,
        )
        .unwrap_err();
        assert!(error.message.contains("contain 1 rows"), "{error:?}");
        assert!(!dataset_path.join("_versions").exists());

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
    fn distributed_append_does_not_create_a_missing_target() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-distributed-missing-append-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let operation_id_text = "operation-missing-append";
        let task_id_text = "task-missing-append";
        let staging_path = format!(
            "{path_text}/_vane_staging/{}/{}",
            hex_identity(operation_id_text),
            hex_identity(task_id_text)
        );
        let transaction = write_staging_transaction(&staging_path, int64_batch(&[("id", 1)]));
        let transaction = pb::Transaction::from(&transaction).encode_to_vec();
        let path = CString::new(path_text.clone()).unwrap();
        let mode = CString::new("append").unwrap();
        let operation_id = CString::new(operation_id_text).unwrap();
        let task_id = CString::new(task_id_text).unwrap();
        let task_ids = [task_id.as_ptr()];
        let transaction_data = [transaction.as_ptr()];
        let transaction_lens = [transaction.len()];
        let mut out_rows = 99;

        let validation_error = distributed_write_validate_inner(
            path.as_ptr(),
            mode.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            operation_id.as_ptr(),
        )
        .expect_err("APPEND validation must reject a missing target");
        assert!(
            validation_error.message.contains("cannot append"),
            "{validation_error:?}"
        );

        let commit_error = distributed_write_commit_inner(
            path.as_ptr(),
            mode.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            operation_id.as_ptr(),
            task_ids.as_ptr(),
            transaction_data.as_ptr(),
            transaction_lens.as_ptr(),
            1,
            1,
            &mut out_rows,
        )
        .expect_err("APPEND commit must reject a target removed after validation");
        assert!(
            commit_error.message.contains("cannot append"),
            "{commit_error:?}"
        );
        assert_eq!(out_rows, 0);
        assert!(
            !dataset_path.join("_versions").exists(),
            "APPEND must not silently create the target dataset"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
    fn distributed_commit_rejects_missing_staged_data_before_manifest_commit() {
        let root = std::env::temp_dir().join(format!(
            "lance-duckdb-distributed-missing-data-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        let dataset_path = root.join("dataset.lance");
        let path_text = dataset_path.to_string_lossy().into_owned();
        let operation_id_text = "operation-missing-data";
        let task_id_text = "task-missing-data";
        let staging_path = format!(
            "{path_text}/_vane_staging/{}/{}",
            hex_identity(operation_id_text),
            hex_identity(task_id_text)
        );
        let transaction = write_staging_transaction(&staging_path, int64_batch(&[("id", 1)]));
        let Operation::Overwrite { fragments, .. } = &transaction.operation else {
            panic!("staging transaction was not an overwrite");
        };
        let file_path = fragments[0].files[0].path.clone();
        std::fs::remove_file(
            std::path::Path::new(&staging_path)
                .join("data")
                .join(file_path),
        )
        .unwrap();

        let transaction = pb::Transaction::from(&transaction).encode_to_vec();
        let path = CString::new(path_text).unwrap();
        let mode = CString::new("create").unwrap();
        let operation_id = CString::new(operation_id_text).unwrap();
        let task_id = CString::new(task_id_text).unwrap();
        let task_ids = [task_id.as_ptr()];
        let transaction_data = [transaction.as_ptr()];
        let transaction_lens = [transaction.len()];
        let mut out_rows = 99;

        let error = distributed_write_commit_inner(
            path.as_ptr(),
            mode.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            operation_id.as_ptr(),
            task_ids.as_ptr(),
            transaction_data.as_ptr(),
            transaction_lens.as_ptr(),
            1,
            1,
            &mut out_rows,
        )
        .unwrap_err();
        assert!(
            error.message.contains("missing staged data file"),
            "{error:?}"
        );
        assert_eq!(out_rows, 0);
        assert!(!dataset_path.join("_versions").exists());

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    #[cfg(feature = "vane-distributed")]
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
        runtime::block_on(super::super::schema_evolution::cleanup_vane_staging_files(
            &dataset,
            chrono::Utc::now(),
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
    #[cfg(feature = "vane-distributed")]
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
