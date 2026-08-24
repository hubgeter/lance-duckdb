//! Native, storage-backed leases shared by Lance extension processes.
//!
//! Lease records live below the dataset in the same object store that holds
//! Lance manifests. Snapshot records may coexist; mutation and vacuum records
//! are exclusive. A short-lived coordination record serializes lease state
//! transitions, which lets Ray workers share snapshot protection without
//! copying a Python object or relying on a process-local mutex.
//!
//! The coordination record is intentionally fail-closed. If a process dies
//! while holding it or leaves a lease record behind, this ABI does not guess
//! that the owner is dead. A future service/recovery implementation can remove
//! records after proving ownership has expired.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures::TryStreamExt;
use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor};
use object_store::path::Path;
use object_store::{ObjectStore as ObjectStoreTrait, ObjectStoreExt, PutMode, PutPayload};
use sha2::{Digest, Sha256};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::DatasetHandle;
use super::util::{
    cstr_to_str, dataset_handle, optional_session_handle, slice_from_ptr, FfiError, FfiResult,
};

const LEASE_PROTOCOL_VERSION: u32 = 2;
const DEFAULT_WAIT_MS: u64 = 30_000;
const POLL_INTERVAL: Duration = Duration::from_millis(100);
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

struct LeaseHandle {
    store: Arc<ObjectStore>,
    coordination_path: Path,
    coordination_payload: Bytes,
    record_path: Path,
    record_payload: Bytes,
    token: String,
    released: bool,
}

fn hex_digest(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push(char::from(b"0123456789abcdef"[(byte >> 4) as usize]));
        out.push(char::from(b"0123456789abcdef"[(byte & 0x0f) as usize]));
    }
    out
}

fn join_path(base: &Path, suffix: &str) -> Result<Path, String> {
    let base = base.as_ref().trim_end_matches('/');
    Path::parse(format!("{base}/{suffix}")).map_err(|error| error.to_string())
}

fn lease_token() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    format!("{}-{}-{}", std::process::id(), now, sequence)
}

fn kind_name(kind: u8) -> Option<&'static str> {
    match kind {
        1 => Some("snapshot"),
        2 => Some("mutation"),
        3 => Some("vacuum"),
        _ => None,
    }
}

unsafe fn parse_storage_options(
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> FfiResult<HashMap<String, String>> {
    if options_len == 0 {
        return Ok(HashMap::new());
    }
    if option_keys.is_null() || option_values.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        ));
    }
    let keys = unsafe { slice_from_ptr(option_keys, options_len, "option_keys")? };
    let values = unsafe { slice_from_ptr(option_values, options_len, "option_values")? };
    let mut result = HashMap::with_capacity(options_len);
    for (index, (&key, &value)) in keys.iter().zip(values.iter()).enumerate() {
        if key.is_null() || value.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("option key/value is null at index {index}"),
            ));
        }
        let key = unsafe { CStr::from_ptr(key) }.to_str().map_err(|error| {
            FfiError::new(ErrorCode::Utf8, format!("option_keys[{index}]: {error}"))
        })?;
        let value = unsafe { CStr::from_ptr(value) }.to_str().map_err(|error| {
            FfiError::new(ErrorCode::Utf8, format!("option_values[{index}]: {error}"))
        })?;
        result.insert(key.to_string(), value.to_string());
    }
    Ok(result)
}

fn deadline(wait_ms: u64) -> Instant {
    Instant::now()
        + Duration::from_millis(if wait_ms == 0 {
            DEFAULT_WAIT_MS
        } else {
            wait_ms
        })
}

async fn acquire_coordination_record(
    store: &ObjectStore,
    path: &Path,
    payload: &Bytes,
    until: Instant,
) -> FfiResult<()> {
    loop {
        match store
            .inner
            .put_opts(
                path,
                PutPayload::from(payload.clone()),
                PutMode::Create.into(),
            )
            .await
        {
            Ok(_) => return Ok(()),
            Err(object_store::Error::AlreadyExists { .. }) => {
                if Instant::now() >= until {
                    return Err(FfiError::new(
                        ErrorCode::LeaseAcquire,
                        format!(
                            "Lance lease coordination record '{}' is still held; refusing unsafe access",
                            path.as_ref()
                        ),
                    ));
                }
                tokio::time::sleep(POLL_INTERVAL).await;
            }
            Err(error) => {
                return Err(FfiError::new(
                    ErrorCode::LeaseAcquire,
                    format!(
                        "create lease coordination record '{}': {error}",
                        path.as_ref()
                    ),
                ));
            }
        }
    }
}

async fn release_coordination_record(
    store: &ObjectStore,
    path: &Path,
    payload: &Bytes,
) -> FfiResult<()> {
    let existing = match store.inner.get(path).await {
        Ok(result) => result.bytes().await.map_err(|error| {
            FfiError::new(
                ErrorCode::LeaseRelease,
                format!("read lease coordination record: {error}"),
            )
        })?,
        Err(object_store::Error::NotFound { .. }) => return Ok(()),
        Err(error) => {
            return Err(FfiError::new(
                ErrorCode::LeaseRelease,
                format!(
                    "read lease coordination record '{}': {error}",
                    path.as_ref()
                ),
            ));
        }
    };
    if existing != *payload {
        return Err(FfiError::new(
            ErrorCode::LeaseRecoveryRequired,
            format!(
                "lease coordination record '{}' belongs to another owner",
                path.as_ref()
            ),
        ));
    }
    store.inner.delete(path).await.map_err(|error| {
        FfiError::new(
            ErrorCode::LeaseRelease,
            format!(
                "delete lease coordination record '{}': {error}",
                path.as_ref()
            ),
        )
    })
}

async fn active_record_kinds(store: &ObjectStore, records_prefix: &Path) -> FfiResult<Vec<String>> {
    let objects = store
        .inner
        .list(Some(records_prefix))
        .try_collect::<Vec<_>>()
        .await
        .map_err(|error| {
            FfiError::new(
                ErrorCode::LeaseAcquire,
                format!(
                    "list Lance lease records '{}': {error}",
                    records_prefix.as_ref()
                ),
            )
        })?;
    Ok(objects
        .into_iter()
        .filter_map(|object| {
            object
                .location
                .as_ref()
                .rsplit_once('.')
                .map(|(_, kind)| kind.to_string())
        })
        .collect())
}

fn conflicts(requested_kind: u8, active: &[String]) -> bool {
    match requested_kind {
        1 => active
            .iter()
            .any(|kind| kind == "mutation" || kind == "vacuum"),
        2 | 3 => !active.is_empty(),
        _ => true,
    }
}

async fn acquire_on_store(
    store: Arc<ObjectStore>,
    base: Path,
    kind: u8,
    operation_id: &str,
    wait_ms: u64,
) -> FfiResult<LeaseHandle> {
    let kind_text = kind_name(kind).ok_or_else(|| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("unknown Lance lease kind {kind}"),
        )
    })?;
    if operation_id.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "lease operation_id is empty",
        ));
    }

    let identity = format!("{}\0{}", store.store_prefix, base.as_ref());
    let identity_hash = hex_digest(&identity);
    let coordination_path = join_path(
        &base,
        &format!("_vane_leases/{identity_hash}/coordination.lock"),
    )
    .map_err(|error| FfiError::new(ErrorCode::LeaseAcquire, error))?;
    let records_prefix = join_path(&base, &format!("_vane_leases/{identity_hash}/records"))
        .map_err(|error| FfiError::new(ErrorCode::LeaseAcquire, error))?;
    let token = lease_token();
    let coordination_payload = Bytes::from(format!(
        "version={}\ntoken={}\npid={}\n",
        LEASE_PROTOCOL_VERSION,
        token,
        std::process::id()
    ));
    // Operation IDs are useful for correlating a lease with a query, but they
    // are often assembled from a dataset URI.  Keep the durable record
    // credential-free (and newline-safe) even when a caller supplied a URI
    // containing userinfo, query parameters, or headers.
    let record_payload = Bytes::from(format!(
        "version={}\nkind={}\noperation_id_hash={}\ntoken={}\npid={}\n",
        LEASE_PROTOCOL_VERSION,
        kind_text,
        hex_digest(operation_id),
        token,
        std::process::id()
    ));
    let until = deadline(wait_ms);
    let record_path = join_path(&records_prefix, &format!("{token}.{kind_text}"))
        .map_err(|error| FfiError::new(ErrorCode::LeaseAcquire, error))?;

    loop {
        acquire_coordination_record(&store, &coordination_path, &coordination_payload, until)
            .await?;
        let active = match active_record_kinds(&store, &records_prefix).await {
            Ok(active) => active,
            Err(error) => {
                let _ =
                    release_coordination_record(&store, &coordination_path, &coordination_payload)
                        .await;
                return Err(error);
            }
        };
        if !conflicts(kind, &active) {
            let create_result = store
                .inner
                .put_opts(
                    &record_path,
                    PutPayload::from(record_payload.clone()),
                    PutMode::Create.into(),
                )
                .await;
            if let Err(error) = create_result {
                let _ =
                    release_coordination_record(&store, &coordination_path, &coordination_payload)
                        .await;
                return Err(FfiError::new(
                    ErrorCode::LeaseAcquire,
                    format!(
                        "create Lance lease record '{}': {error}",
                        record_path.as_ref()
                    ),
                ));
            }
            release_coordination_record(&store, &coordination_path, &coordination_payload).await?;
            return Ok(LeaseHandle {
                store,
                coordination_path,
                coordination_payload,
                record_path,
                record_payload,
                token,
                released: false,
            });
        }
        release_coordination_record(&store, &coordination_path, &coordination_payload).await?;
        if Instant::now() >= until {
            return Err(FfiError::new(
                ErrorCode::LeaseAcquire,
                format!(
                    "Lance {} lease conflicts with active records; refusing unsafe concurrent access",
                    kind_text
                ),
            ));
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn acquire_inner(
    path: &str,
    kind: u8,
    operation_id: &str,
    storage_options: HashMap<String, String>,
    session: Option<Arc<lance::session::Session>>,
    wait_ms: u64,
) -> FfiResult<LeaseHandle> {
    if path.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "lease path is empty",
        ));
    }
    let registry = session
        .as_ref()
        .map(|session| session.store_registry())
        .unwrap_or_else(|| Arc::new(ObjectStoreRegistry::default()));
    let mut params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options),
        ));
    }
    let (store, base) = ObjectStore::from_uri_and_params(registry, path, &params)
        .await
        .map_err(|error| {
            FfiError::new(
                ErrorCode::LeaseAcquire,
                format!("open object store for lease '{path}': {error}"),
            )
        })?;
    acquire_on_store(store, base, kind, operation_id, wait_ms).await
}

async fn release_inner(handle: &mut LeaseHandle) -> FfiResult<()> {
    if handle.released {
        return Ok(());
    }
    let until = deadline(0);
    acquire_coordination_record(
        &handle.store,
        &handle.coordination_path,
        &handle.coordination_payload,
        until,
    )
    .await?;
    let result = async {
        let existing = match handle.store.inner.get(&handle.record_path).await {
            Ok(result) => result.bytes().await.map_err(|error| {
                FfiError::new(
                    ErrorCode::LeaseRelease,
                    format!("read lease record: {error}"),
                )
            })?,
            Err(object_store::Error::NotFound { .. }) => return Ok(()),
            Err(error) => {
                return Err(FfiError::new(
                    ErrorCode::LeaseRelease,
                    format!(
                        "read lease record '{}': {error}",
                        handle.record_path.as_ref()
                    ),
                ));
            }
        };
        if existing != handle.record_payload {
            return Err(FfiError::new(
                ErrorCode::LeaseRecoveryRequired,
                format!(
                    "lease record '{}' no longer belongs to token {}; refusing to delete it",
                    handle.record_path.as_ref(),
                    handle.token
                ),
            ));
        }
        handle
            .store
            .inner
            .delete(&handle.record_path)
            .await
            .map_err(|error| {
                FfiError::new(
                    ErrorCode::LeaseRelease,
                    format!(
                        "delete lease record '{}': {error}",
                        handle.record_path.as_ref()
                    ),
                )
            })?;
        Ok(())
    }
    .await;
    let coordination_result = release_coordination_record(
        &handle.store,
        &handle.coordination_path,
        &handle.coordination_payload,
    )
    .await;
    result?;
    coordination_result?;
    handle.released = true;
    Ok(())
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_acquire_lease_with_storage_options(
    path: *const c_char,
    kind: u8,
    operation_id: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    wait_ms: u64,
) -> *mut c_void {
    let result = (|| {
        let path = unsafe { cstr_to_str(path, "path")? };
        let operation_id = unsafe { cstr_to_str(operation_id, "operation_id")? };
        let options = unsafe { parse_storage_options(option_keys, option_values, options_len)? };
        let session = unsafe { optional_session_handle(session)? };
        runtime::block_on(acquire_inner(
            path,
            kind,
            operation_id,
            options,
            session,
            wait_ms,
        ))
        .map_err(|error| FfiError::new(ErrorCode::Runtime, format!("lease runtime: {error}")))?
    })();
    match result {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(error) => {
            set_last_error(error.code, error.message);
            ptr::null_mut()
        }
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_acquire_dataset_lease(
    dataset: *mut c_void,
    kind: u8,
    operation_id: *const c_char,
    wait_ms: u64,
) -> *mut c_void {
    let result = (|| {
        let handle: &DatasetHandle = unsafe { dataset_handle(dataset)? };
        let operation_id = unsafe { cstr_to_str(operation_id, "operation_id")? };
        let dataset = handle.dataset.clone();
        runtime::block_on(async move {
            let store = dataset.object_store(None).await.map_err(|error| {
                FfiError::new(
                    ErrorCode::LeaseAcquire,
                    format!("resolve dataset object store for lease: {error}"),
                )
            })?;
            acquire_on_store(
                store,
                dataset.branch_location().path,
                kind,
                operation_id,
                wait_ms,
            )
            .await
        })
        .map_err(|error| FfiError::new(ErrorCode::Runtime, format!("lease runtime: {error}")))?
    })();
    match result {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(error) => {
            set_last_error(error.code, error.message);
            ptr::null_mut()
        }
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_release_lease(lease: *mut c_void) -> i32 {
    if lease.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "lease is null");
        return -1;
    }
    let handle = unsafe { &mut *(lease as *mut LeaseHandle) };
    match runtime::block_on(release_inner(handle)) {
        Ok(Ok(())) => {
            clear_last_error();
            0
        }
        Ok(Err(error)) => {
            set_last_error(error.code, error.message);
            -1
        }
        Err(error) => {
            set_last_error(ErrorCode::Runtime, format!("lease runtime: {error}"));
            -1
        }
    }
}

/// Free only the local lease handle. Callers use this after a failed release
/// to retain the remote record for explicit recovery.
#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_close_lease(lease: *mut c_void) {
    if !lease.is_null() {
        unsafe {
            let _ = Box::from_raw(lease as *mut LeaseHandle);
        }
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_lease_token(lease: *mut c_void) -> *const c_char {
    if lease.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "lease is null");
        return ptr::null();
    }
    let handle = unsafe { &*(lease as *const LeaseHandle) };
    match CString::new(handle.token.clone()) {
        Ok(token) => {
            clear_last_error();
            token.into_raw()
        }
        Err(error) => {
            set_last_error(
                ErrorCode::LeaseAcquire,
                format!("serialize lease token: {error}"),
            );
            ptr::null()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_identity_is_stable_and_credential_free() {
        let first = hex_digest("s3://bucket/table");
        let second = hex_digest("s3://bucket/table");
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn snapshot_leases_can_coexist() {
        assert!(!conflicts(1, &["snapshot".to_string()]));
        assert!(conflicts(2, &["snapshot".to_string()]));
        assert!(conflicts(3, &["mutation".to_string()]));
    }

    #[test]
    fn lease_record_identity_is_credential_free() {
        let operation_id = "scan:s3://user:secret@example.test/table?token=hidden";
        let payload = format!("operation_id_hash={}\n", hex_digest(operation_id));
        assert!(!payload.contains("secret"));
        assert!(!payload.contains("token=hidden"));
    }
}
