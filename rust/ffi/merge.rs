use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::ptr;
use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow_array::{make_array, RecordBatch, RecordBatchReader, StructArray};
use arrow_schema::ArrowError;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction, UpdateMode};
use lance::dataset::{InsertBuilder, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance_select::RowAddrTreeMap;
use lance_table::format::RowIdMeta;
use lance_table::rowids::{rechunk_sequences, write_row_ids, RowIdSequence};
use roaring::RoaringTreemap;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::update::{apply_deletions, build_row_id_index};
use super::util::{cstr_to_str, optional_session_handle, slice_from_ptr, FfiError, FfiResult};
use super::write::{
    cleanup_uncommitted_fragments, cleanup_uncommitted_transaction, VaneTransaction,
};

enum MergePreparation {
    Ready(Option<Box<VaneTransaction>>),
    CleanupIncomplete(String),
}

enum MergeStageError {
    Regular(String),
    CleanupIncomplete(String),
}

impl From<String> for MergeStageError {
    fn from(message: String) -> Self {
        Self::Regular(message)
    }
}

struct MergeHandle {
    input_schema: Arc<arrow_schema::Schema>,
    data_type: DataType,
    path: String,
    session: Option<Arc<lance::session::Session>>,
    storage_options: HashMap<String, String>,
    max_rows_per_file: usize,
    max_rows_per_group: usize,
    max_bytes_per_file: usize,
    dataset_version: u64,
    delete_row_ids: RoaringTreemap,
    update_row_ids: Vec<u64>,
    update_row_id_set: HashSet<u64>,
    update_spool: Option<BatchSpool>,
    modified_columns: HashSet<String>,
    insert_spool: Option<BatchSpool>,
}

struct BatchSpool {
    writer: Option<StreamWriter<File>>,
    reader_file: Option<File>,
    cleanup_path: Option<PathBuf>,
}

impl BatchSpool {
    fn new(schema: &arrow_schema::Schema) -> Result<Self, String> {
        for _ in 0..16 {
            let path = std::env::temp_dir().join(format!(
                "lance-duckdb-merge-{}-{:016x}.arrow",
                std::process::id(),
                rand::random::<u64>()
            ));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            let file = match options.open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("create MERGE batch spool: {error}")),
            };
            let reader_file = match file.try_clone() {
                Ok(reader_file) => reader_file,
                Err(error) => {
                    // Windows cannot unlink an open file.  Close the original
                    // handle before removing the partially-created spool.
                    drop(file);
                    let _ = std::fs::remove_file(&path);
                    return Err(format!("clone MERGE batch spool: {error}"));
                }
            };
            let writer = match StreamWriter::try_new(file, schema) {
                Ok(writer) => writer,
                Err(error) => {
                    // The cloned reader handle is still open on this branch.
                    // Drop it first so cleanup works on Windows as well as
                    // POSIX filesystems.
                    drop(reader_file);
                    let _ = std::fs::remove_file(&path);
                    return Err(format!("open MERGE Arrow spool: {error}"));
                }
            };

            // POSIX keeps the open inode alive, so unlink immediately and leave
            // no cleartext pathname behind if the process is killed.
            #[cfg(unix)]
            let cleanup_path = match std::fs::remove_file(&path) {
                Ok(()) => None,
                Err(_) => Some(path),
            };
            #[cfg(not(unix))]
            let cleanup_path = Some(path);

            return Ok(Self {
                writer: Some(writer),
                reader_file: Some(reader_file),
                cleanup_path,
            });
        }
        Err("could not allocate a unique MERGE batch spool after 16 attempts".to_string())
    }

    fn write(&mut self, batch: &RecordBatch) -> Result<(), String> {
        self.writer
            .as_mut()
            .ok_or_else(|| "MERGE batch spool is already finalized".to_string())?
            .write(batch)
            .map_err(|error| format!("write MERGE Arrow spool: {error}"))
    }

    fn finish(mut self) -> Result<BatchSpoolReader, String> {
        let mut writer = self
            .writer
            .take()
            .ok_or_else(|| "MERGE batch spool is already finalized".to_string())?;
        writer
            .finish()
            .map_err(|error| format!("finish MERGE Arrow spool: {error}"))?;
        drop(writer);
        let mut file = self
            .reader_file
            .take()
            .ok_or_else(|| "MERGE batch spool has no reader".to_string())?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind MERGE Arrow spool: {error}"))?;
        let reader = StreamReader::try_new(file, None)
            .map_err(|error| format!("read MERGE Arrow spool: {error}"))?;
        Ok(BatchSpoolReader {
            reader: Some(reader),
            cleanup_path: self.cleanup_path.take(),
        })
    }
}

impl Drop for BatchSpool {
    fn drop(&mut self) {
        // Close every file handle before unlinking the named spool.  This order
        // is required on Windows; POSIX would otherwise hide the bug because
        // it permits unlinking an open inode.
        drop(self.writer.take());
        drop(self.reader_file.take());
        if let Some(path) = self.cleanup_path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

struct BatchSpoolReader {
    reader: Option<StreamReader<File>>,
    cleanup_path: Option<PathBuf>,
}

impl Iterator for BatchSpoolReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.as_mut().and_then(Iterator::next)
    }
}

impl RecordBatchReader for BatchSpoolReader {
    fn schema(&self) -> Arc<arrow_schema::Schema> {
        self.reader
            .as_ref()
            .expect("MERGE spool reader is unavailable only while dropping")
            .schema()
    }
}

impl Drop for BatchSpoolReader {
    fn drop(&mut self) {
        drop(self.reader.take());
        if let Some(path) = self.cleanup_path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

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

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_merge_begin_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    session: *mut c_void,
    dataset_version: u64,
    out_merge_handle: *mut *mut c_void,
) -> i32 {
    match merge_begin_with_storage_options_inner(
        path,
        option_keys,
        option_values,
        options_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        session,
        dataset_version,
        out_merge_handle,
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
fn merge_begin_with_storage_options_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    session: *mut c_void,
    dataset_version: u64,
    out_merge_handle: *mut *mut c_void,
) -> FfiResult<()> {
    if out_merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_merge_handle is null",
        ));
    }
    unsafe {
        ptr::write_unaligned(out_merge_handle, ptr::null_mut());
    }

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

    let max_rows_per_file = usize::try_from(max_rows_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("invalid max_rows_per_file: {err}"),
        )
    })?;
    let max_rows_per_group = usize::try_from(max_rows_per_group).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("invalid max_rows_per_group: {err}"),
        )
    })?;
    let max_bytes_per_file = usize::try_from(max_bytes_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("invalid max_bytes_per_file: {err}"),
        )
    })?;
    let session = unsafe { optional_session_handle(session)? };

    if dataset_version == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "MERGE requires a non-zero pinned dataset version",
        ));
    }

    let input_schema: Arc<arrow_schema::Schema> = match runtime::block_on(async {
        let mut builder =
            DatasetBuilder::from_uri(path.as_str()).with_storage_options(storage_options.clone());
        if let Some(session) = session.clone() {
            builder = builder.with_session(session);
        }
        let dataset = builder.load().await.map_err(|e| e.to_string())?;
        dataset
            .checkout_version(dataset_version)
            .await
            .map_err(|e| e.to_string())
    }) {
        Ok(Ok(dataset)) => Arc::new(dataset.schema().into()),
        Ok(Err(message)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("open dataset: {message}"),
            ))
        }
        Err(err) => {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("runtime: {err}"),
            ))
        }
    };

    let data_type = DataType::Struct(input_schema.fields().clone());
    let handle = Box::new(MergeHandle {
        input_schema,
        data_type,
        path,
        session,
        storage_options,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        dataset_version,
        delete_row_ids: RoaringTreemap::new(),
        update_row_ids: Vec::new(),
        update_row_id_set: HashSet::new(),
        update_spool: None,
        modified_columns: HashSet::new(),
        insert_spool: None,
    });

    unsafe {
        ptr::write_unaligned(out_merge_handle, Box::into_raw(handle) as *mut c_void);
    }
    Ok(())
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_merge_add_delete_rowids(
    merge_handle: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
) -> i32 {
    match merge_add_delete_rowids_inner(merge_handle, row_ids, row_ids_len) {
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

fn merge_add_delete_rowids_inner(
    merge_handle: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
) -> FfiResult<()> {
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }
    if row_ids_len > 0 && row_ids.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "row_ids is null with non-zero length",
        ));
    }

    let handle = unsafe { &mut *(merge_handle as *mut MergeHandle) };
    let ids = if row_ids_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(row_ids, row_ids_len, "row_ids")? }
    };

    for row_id in ids {
        handle.delete_row_ids.insert(*row_id);
    }

    Ok(())
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_merge_add_insert_batch(
    merge_handle: *mut c_void,
    array: *mut c_void,
) -> i32 {
    match merge_add_insert_batch_inner(merge_handle, array) {
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

fn import_merge_batch(handle: &MergeHandle, array: *mut c_void) -> FfiResult<RecordBatch> {
    if array.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "array is null"));
    }

    let raw_array = unsafe { ptr::read(array as *mut RawArrowArray) };
    unsafe {
        (*(array as *mut RawArrowArray)).release = None;
    }
    let ffi_array: arrow::ffi::FFI_ArrowArray = unsafe { std::mem::transmute(raw_array) };

    let array_data =
        unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, handle.data_type.clone()) }
            .map_err(|err| {
                FfiError::new(ErrorCode::DatasetMerge, format!("array import: {err}"))
            })?;

    let array = make_array(array_data);
    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| FfiError::new(ErrorCode::DatasetMerge, "array is not a struct"))?;

    RecordBatch::try_new(handle.input_schema.clone(), struct_array.columns().to_vec())
        .map_err(|err| FfiError::new(ErrorCode::DatasetMerge, format!("record batch: {err}")))
}

fn merge_add_insert_batch_inner(merge_handle: *mut c_void, array: *mut c_void) -> FfiResult<()> {
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }
    let handle = unsafe { &mut *(merge_handle as *mut MergeHandle) };
    let batch = import_merge_batch(handle, array)?;
    if handle.insert_spool.is_none() {
        handle.insert_spool = Some(
            BatchSpool::new(handle.input_schema.as_ref())
                .map_err(|error| FfiError::new(ErrorCode::DatasetMerge, error))?,
        );
    }
    handle
        .insert_spool
        .as_mut()
        .ok_or_else(|| FfiError::new(ErrorCode::DatasetMerge, "insert spool was not initialized"))?
        .write(&batch)
        .map_err(|error| FfiError::new(ErrorCode::DatasetMerge, error))?;
    Ok(())
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_merge_add_update_batch(
    merge_handle: *mut c_void,
    array: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
    modified_columns: *const *const c_char,
    modified_columns_len: usize,
) -> i32 {
    match merge_add_update_batch_inner(
        merge_handle,
        array,
        row_ids,
        row_ids_len,
        modified_columns,
        modified_columns_len,
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

fn merge_add_update_batch_inner(
    merge_handle: *mut c_void,
    array: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
    modified_columns: *const *const c_char,
    modified_columns_len: usize,
) -> FfiResult<()> {
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }
    if row_ids_len > 0 && row_ids.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "row_ids is null with non-zero length",
        ));
    }
    if modified_columns_len > 0 && modified_columns.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "modified_columns is null with non-zero length",
        ));
    }

    let handle = unsafe { &mut *(merge_handle as *mut MergeHandle) };
    let batch = import_merge_batch(handle, array)?;
    if batch.num_rows() != row_ids_len {
        return Err(FfiError::new(
            ErrorCode::DatasetMerge,
            format!(
                "MERGE update batch has {} rows but {} row IDs",
                batch.num_rows(),
                row_ids_len
            ),
        ));
    }
    let row_ids = if row_ids_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(row_ids, row_ids_len, "row_ids")? }
    };
    let modified_columns = if modified_columns_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(modified_columns, modified_columns_len, "modified_columns")? }
    };
    if !row_ids.is_empty() && modified_columns.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "MERGE update rows require at least one modified column",
        ));
    }
    let mut batch_row_ids = HashSet::with_capacity(row_ids.len());
    for row_id in row_ids {
        if !handle.delete_row_ids.contains(*row_id) {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("MERGE update row ID {row_id} was not registered for replacement"),
            ));
        }
        if handle.update_row_id_set.contains(row_id) || !batch_row_ids.insert(*row_id) {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("MERGE update row ID {row_id} was supplied more than once"),
            ));
        }
    }
    let mut validated_modified_columns = Vec::with_capacity(modified_columns.len());
    for (index, column) in modified_columns.iter().enumerate() {
        if column.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("modified_columns[{index}] is null"),
            ));
        }
        let column = unsafe { CStr::from_ptr(*column) }.to_str().map_err(|err| {
            FfiError::new(
                ErrorCode::Utf8,
                format!("modified_columns[{index}] utf8: {err}"),
            )
        })?;
        if !handle
            .input_schema
            .fields()
            .iter()
            .any(|field| field.name() == column)
        {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("MERGE update references unknown modified column '{column}'"),
            ));
        }
        validated_modified_columns.push(column.to_string());
    }

    if handle.update_spool.is_none() {
        handle.update_spool = Some(
            BatchSpool::new(handle.input_schema.as_ref())
                .map_err(|error| FfiError::new(ErrorCode::DatasetMerge, error))?,
        );
    }
    handle
        .update_spool
        .as_mut()
        .ok_or_else(|| FfiError::new(ErrorCode::DatasetMerge, "update spool was not initialized"))?
        .write(&batch)
        .map_err(|error| FfiError::new(ErrorCode::DatasetMerge, error))?;
    handle.update_row_ids.extend_from_slice(row_ids);
    handle.update_row_id_set.extend(batch_row_ids);
    handle.modified_columns.extend(validated_modified_columns);
    Ok(())
}

#[ffi_guard_macro::ffi_guard(dataset_mutation)]
#[no_mangle]
pub unsafe extern "C" fn lance_merge_finish_uncommitted(
    merge_handle: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> i32 {
    match merge_finish_uncommitted_inner(merge_handle, out_transaction) {
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

fn merge_finish_uncommitted_inner(
    merge_handle: *mut c_void,
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
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }

    let mut handle = unsafe { Box::from_raw(merge_handle as *mut MergeHandle) };

    let maybe_txn = match runtime::block_on(async move {
        let mut builder = DatasetBuilder::from_uri(handle.path.as_str())
            .with_storage_options(handle.storage_options.clone());
        if let Some(session) = handle.session.clone() {
            builder = builder.with_session(session);
        }
        let dataset = builder.load().await.map_err(|e| e.to_string())?;
        let dataset = dataset
            .checkout_version(handle.dataset_version)
            .await
            .map_err(|e| e.to_string())?;
        let dataset = Arc::new(dataset);

        let update_reader = handle
            .update_spool
            .take()
            .map(BatchSpool::finish)
            .transpose()?;
        let insert_reader = handle
            .insert_spool
            .take()
            .map(BatchSpool::finish)
            .transpose()?;
        let mut new_fragments = Vec::new();
        let mut cleanup_fragments = Vec::new();
        let stage_result = async {
            if update_reader.is_some() || insert_reader.is_some() {
                let mut store_params = ObjectStoreParams::default();
                if !handle.storage_options.is_empty() {
                    store_params.storage_options_accessor = Some(Arc::new(
                        StorageOptionsAccessor::with_static_options(handle.storage_options.clone()),
                    ));
                }

                let write_params = WriteParams {
                    mode: WriteMode::Append,
                    max_rows_per_file: handle.max_rows_per_file,
                    max_rows_per_group: handle.max_rows_per_group,
                    max_bytes_per_file: handle.max_bytes_per_file,
                    session: handle.session.clone(),
                    store_params: Some(store_params),
                    ..Default::default()
                };

                if let Some(reader) = update_reader {
                    let append_txn = InsertBuilder::new(dataset.clone())
                        .with_params(&write_params)
                        .execute_uncommitted_stream(
                            Box::new(reader) as Box<dyn RecordBatchReader + Send>
                        )
                        .await
                        .map_err(|error| {
                            MergeStageError::CleanupIncomplete(format!(
                                "MERGE update write failed and may have left orphan files: {error}"
                            ))
                        })?;
                    let mut update_fragments = match &append_txn.operation {
                        Operation::Append { fragments } => fragments.clone(),
                        _ => {
                            let message =
                                "unexpected transaction operation for merge update write"
                                    .to_string();
                            return match cleanup_uncommitted_transaction(
                                &handle.path,
                                &handle.storage_options,
                                handle.session.clone(),
                                &append_txn,
                            )
                            .await
                            {
                                Ok(()) => Err(MergeStageError::Regular(message)),
                                Err(cleanup_error) => {
                                    Err(MergeStageError::CleanupIncomplete(format!(
                                        "{message}, and orphan cleanup failed: {cleanup_error}"
                                    )))
                                }
                            };
                        }
                    };
                    cleanup_fragments.extend(update_fragments.iter().cloned());

                    if dataset.manifest.uses_stable_row_ids() {
                        let fragment_sizes = update_fragments
                            .iter()
                            .map(|fragment| {
                                fragment.physical_rows.ok_or_else(|| {
                                    "MERGE update produced a fragment with an unknown row count"
                                        .to_string()
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let expected_rows =
                            fragment_sizes.iter().try_fold(0_usize, |total, rows| {
                                total.checked_add(*rows).ok_or_else(|| {
                                    "MERGE update fragment row count overflow".to_string()
                                })
                            })?;
                        if expected_rows != handle.update_row_ids.len() {
                            return Err(MergeStageError::Regular(format!(
                                "MERGE update wrote {expected_rows} rows but captured {} stable row IDs",
                                handle.update_row_ids.len()
                            )));
                        }
                        let sequence = RowIdSequence::from(handle.update_row_ids.as_slice());
                        let fragment_sizes = fragment_sizes
                            .into_iter()
                            .map(|rows| {
                                u64::try_from(rows).map_err(|_| {
                                    "MERGE update fragment row count does not fit in u64".to_string()
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let sequences = rechunk_sequences(vec![sequence], fragment_sizes, false)
                            .map_err(|e| e.to_string())?;
                        if sequences.len() != update_fragments.len() {
                            return Err(MergeStageError::Regular(
                                "MERGE stable row-id rechunking did not match output fragments"
                                    .to_string(),
                            ));
                        }
                        for (fragment, sequence) in update_fragments.iter_mut().zip(sequences) {
                            fragment.row_id_meta =
                                Some(RowIdMeta::Inline(write_row_ids(&sequence)));
                        }
                    }
                    new_fragments.extend(update_fragments);
                }

                if let Some(reader) = insert_reader {
                    let append_txn = InsertBuilder::new(dataset.clone())
                        .with_params(&write_params)
                        .execute_uncommitted_stream(
                            Box::new(reader) as Box<dyn RecordBatchReader + Send>
                        )
                        .await
                        .map_err(|error| {
                            MergeStageError::CleanupIncomplete(format!(
                                "MERGE insert write failed and may have left orphan files: {error}"
                            ))
                        })?;
                    let fragments = match &append_txn.operation {
                        Operation::Append { fragments } => fragments.clone(),
                        _ => {
                            let message =
                                "unexpected transaction operation for merge insert write"
                                    .to_string();
                            return match cleanup_uncommitted_transaction(
                                &handle.path,
                                &handle.storage_options,
                                handle.session.clone(),
                                &append_txn,
                            )
                            .await
                            {
                                Ok(()) => Err(MergeStageError::Regular(message)),
                                Err(cleanup_error) => {
                                    Err(MergeStageError::CleanupIncomplete(format!(
                                        "{message}, and orphan cleanup failed: {cleanup_error}"
                                    )))
                                }
                            };
                        }
                    };
                    cleanup_fragments.extend(fragments.iter().cloned());
                    new_fragments.extend(fragments);
                }
            }

            let mut row_addrs = RoaringTreemap::new();
            if !handle.delete_row_ids.is_empty() {
                if dataset.manifest.uses_stable_row_ids() {
                    let row_id_index = build_row_id_index(dataset.as_ref())
                        .await
                        .map_err(|e| e.to_string())?;
                    for row_id in handle.delete_row_ids.iter() {
                        let addr = row_id_index
                            .get(row_id)
                            .ok_or_else(|| format!("row id missing from row id index: {row_id}"))?;
                        row_addrs.insert(u64::from(addr));
                    }
                } else {
                    row_addrs = handle.delete_row_ids.clone();
                }
            }

            fn collect_modified_field_ids(
                field: &lance_core::datatypes::Field,
                column_name: &str,
                out: &mut Vec<u32>,
            ) -> Result<(), String> {
                out.push(u32::try_from(field.id).map_err(|_| {
                    format!(
                        "MERGE modified column '{column_name}' has an invalid field id {}",
                        field.id
                    )
                })?);
                for child in &field.children {
                    collect_modified_field_ids(child, column_name, out)?;
                }
                Ok(())
            }

            let mut fields_for_preserving_frag_bitmap = Vec::new();
            for column in &handle.modified_columns {
                let field = dataset
                    .schema()
                    .fields
                    .iter()
                    .find(|field| field.name == *column)
                    .ok_or_else(|| {
                        format!("MERGE modified column '{column}' is not in the dataset schema")
                    })?;
                collect_modified_field_ids(
                    field,
                    column,
                    &mut fields_for_preserving_frag_bitmap,
                )?;
            }
            fields_for_preserving_frag_bitmap.sort_unstable();
            fields_for_preserving_frag_bitmap.dedup();

            // Validate every schema-derived field id before writing deletion
            // files.  After apply_deletions succeeds, construction of the
            // returned transaction must be infallible so those files cannot be
            // stranded outside either the transaction or the cleanup path.
            let (updated_fragments, removed_fragment_ids) = if row_addrs.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                apply_deletions(dataset.as_ref(), &row_addrs)
                    .await
                    .map_err(|error| {
                        if error.cleanup_incomplete {
                            MergeStageError::CleanupIncomplete(error.message)
                        } else {
                            MergeStageError::Regular(error.message)
                        }
                    })?
            };
            // apply_deletions has already materialized deletion files.  Keep
            // their fragment metadata in the same cleanup set as the staged
            // append files in case transaction construction fails afterward.
            cleanup_fragments.extend(updated_fragments.iter().cloned());

            if new_fragments.is_empty()
                && updated_fragments.is_empty()
                && removed_fragment_ids.is_empty()
            {
                return Ok::<_, MergeStageError>(None);
            }

            let operation = Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments: std::mem::take(&mut new_fragments),
                fields_modified: vec![],
                merged_generations: Vec::new(),
                fields_for_preserving_frag_bitmap,
                update_mode: Some(UpdateMode::RewriteRows),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            };
            let transaction = Transaction::new(dataset.manifest.version, operation, None);
            let txn = if row_addrs.is_empty() {
                VaneTransaction::new(transaction)
            } else {
                VaneTransaction::with_affected_rows(transaction, RowAddrTreeMap::from(row_addrs))
            };

            Ok::<_, MergeStageError>(Some(txn))
        }
        .await;

        match stage_result {
            Ok(txn) => Ok(MergePreparation::Ready(txn.map(Box::new))),
            Err(stage_error) => {
                let (message, already_incomplete) = match stage_error {
                    MergeStageError::Regular(message) => (message, false),
                    MergeStageError::CleanupIncomplete(message) => (message, true),
                };
                let cleanup_result = cleanup_uncommitted_fragments(
                    &handle.path,
                    &handle.storage_options,
                    handle.session,
                    &cleanup_fragments,
                    Some(handle.dataset_version),
                )
                .await;
                match (already_incomplete, cleanup_result) {
                    (false, Ok(())) => Err(message),
                    (_, Ok(())) => Ok(MergePreparation::CleanupIncomplete(message)),
                    (_, Err(cleanup_error)) => Ok(MergePreparation::CleanupIncomplete(format!(
                        "{message}; additional orphan cleanup failed: {cleanup_error}"
                    ))),
                }
            }
        }
    }) {
        Ok(Ok(MergePreparation::Ready(txn))) => txn,
        Ok(Ok(MergePreparation::CleanupIncomplete(message))) => {
            return Err(FfiError::new(
                ErrorCode::DatasetCommitOutcomeUnknown,
                message,
            ))
        }
        Ok(Err(message)) => return Err(FfiError::new(ErrorCode::DatasetMerge, message)),
        Err(err) => {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("runtime: {err}"),
            ))
        }
    };

    if let Some(txn) = maybe_txn {
        unsafe {
            ptr::write_unaligned(out_transaction, Box::into_raw(txn) as *mut c_void);
        }
    }

    Ok(())
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_merge_abort(merge_handle: *mut c_void) {
    if merge_handle.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(merge_handle as *mut MergeHandle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::Field;

    fn named_spool_file() -> (PathBuf, File) {
        let path = std::env::temp_dir().join(format!(
            "lance-duckdb-merge-drop-{}-{:016x}.arrow",
            std::process::id(),
            rand::random::<u64>()
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options.open(&path).unwrap();
        (path, file)
    }

    #[test]
    fn merge_spool_closes_handles_before_removing_named_file() {
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));

        let (aborted_path, aborted_file) = named_spool_file();
        let aborted_reader = aborted_file.try_clone().unwrap();
        let aborted_writer = StreamWriter::try_new(aborted_file, schema.as_ref()).unwrap();
        let spool = BatchSpool {
            writer: Some(aborted_writer),
            reader_file: Some(aborted_reader),
            cleanup_path: Some(aborted_path.clone()),
        };
        drop(spool);
        assert!(!aborted_path.exists());

        let (reader_path, reader_file) = named_spool_file();
        let mut reader_writer = StreamWriter::try_new(reader_file, schema.as_ref()).unwrap();
        reader_writer.finish().unwrap();
        let mut reader_file = reader_writer.into_inner().unwrap();
        reader_file.seek(SeekFrom::Start(0)).unwrap();
        let reader = StreamReader::try_new(reader_file, None).unwrap();
        let spool_reader = BatchSpoolReader {
            reader: Some(reader),
            cleanup_path: Some(reader_path.clone()),
        };
        drop(spool_reader);
        assert!(!reader_path.exists());
    }
}
