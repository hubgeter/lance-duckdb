use std::any::Any;
use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum ErrorCode {
    InvalidArgument = 1,
    Utf8 = 2,
    Runtime = 3,

    DatasetOpen = 4,
    DatasetCountRows = 5,
    FragmentScan = 6,

    StreamCreate = 7,
    StreamNext = 8,
    SchemaExport = 9,
    BatchExport = 10,

    KnnSchema = 11,
    KnnStreamCreate = 12,
    ExplainPlan = 13,
    FtsSchema = 14,
    FtsStreamCreate = 15,
    DatasetScan = 16,
    HybridStreamCreate = 17,

    DatasetWriteOpen = 18,
    DatasetWriteBatch = 19,
    DatasetWriteFinish = 20,

    NamespaceListTables = 21,
    NamespaceDescribeTable = 22,
    DirNamespaceListTables = 23,
    DatasetWriteFinishUncommitted = 24,
    DatasetCommitTransaction = 25,
    DirNamespaceDropTable = 26,

    DatasetDelete = 27,
    DatasetUpdateOverwrite = 28,

    DatasetCreateIndex = 29,
    DatasetDropIndex = 30,
    DatasetDescribeIndices = 31,
    DatasetOptimizeIndices = 32,
    IndexStreamCreate = 34,

    DatasetAddColumns = 35,
    DatasetDropColumns = 36,
    DatasetAlterColumns = 37,
    DatasetUpdateMetadata = 38,
    DatasetUpdateConfig = 39,
    DatasetUpdateSchemaMetadata = 40,
    DatasetUpdateFieldMetadata = 41,
    DatasetCompactFiles = 42,
    DatasetCleanupOldVersions = 43,
    DatasetListKeyValues = 44,
    DatasetListIndices = 45,
    DatasetCreateScalarIndex = 46,
    DatasetCalculateDataStats = 47,
    DatasetTake = 48,

    NamespaceDescribeTableInfo = 49,
    NamespaceCreateEmptyTable = 50,
    NamespaceDropTable = 51,
    Exec = 52,
    DatasetMerge = 53,
    NamespaceQueryTable = 54,
    DatasetCommitOutcomeUnknown = 55,
    NamespaceMutationOutcomeUnknown = 56,
    LeaseAcquire = 57,
    LeaseRelease = 58,
    LeaseRecoveryRequired = 59,
}

struct LastError {
    code: i32,
    message: CString,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<LastError>> = const { RefCell::new(None) };
}

fn sanitize_message(message: &str) -> CString {
    match CString::new(message) {
        Ok(v) => v,
        Err(_) => CString::new(message.replace('\0', "\\0"))
            .unwrap_or_else(|_| CString::new("invalid error message").unwrap()),
    }
}

pub fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

pub fn set_last_error(code: ErrorCode, message: impl AsRef<str>) {
    let code = code as i32;
    let message = sanitize_message(message.as_ref());
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(LastError { code, message });
    });
}

pub trait FfiPanicReturn {
    fn panic_value() -> Self;
}

impl FfiPanicReturn for () {
    fn panic_value() -> Self {}
}

impl<T> FfiPanicReturn for *const T {
    fn panic_value() -> Self {
        ptr::null()
    }
}

impl<T> FfiPanicReturn for *mut T {
    fn panic_value() -> Self {
        ptr::null_mut()
    }
}

impl FfiPanicReturn for i32 {
    fn panic_value() -> Self {
        -1
    }
}

impl FfiPanicReturn for i64 {
    fn panic_value() -> Self {
        -1
    }
}

impl FfiPanicReturn for u64 {
    fn panic_value() -> Self {
        0
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

pub fn catch_ffi_panic<T, F>(function: F) -> T
where
    T: FfiPanicReturn,
    F: FnOnce() -> T,
{
    catch_ffi_panic_with_code(ErrorCode::Runtime, function)
}

pub fn catch_ffi_panic_with_code<T, F>(panic_code: ErrorCode, function: F) -> T
where
    T: FfiPanicReturn,
    F: FnOnce() -> T,
{
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(value) => value,
        Err(payload) => {
            set_last_error(
                panic_code,
                format!(
                    "panic at Lance FFI boundary: {}",
                    panic_message(payload.as_ref())
                ),
            );
            T::panic_value()
        }
    }
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub extern "C" fn lance_last_error_code() -> i32 {
    LAST_ERROR.with(|e| e.borrow().as_ref().map(|v| v.code).unwrap_or(0))
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub extern "C" fn lance_last_error_message() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow_mut().take() {
        Some(err) => err.message.into_raw() as *const c_char,
        None => ptr::null(),
    })
}

#[ffi_guard_macro::ffi_guard]
#[no_mangle]
pub unsafe extern "C" fn lance_free_string(s: *const c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s as *mut c_char);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[ffi_guard_macro::ffi_guard]
    extern "C" fn ffi_panic_probe() -> i32 {
        panic!("ffi panic probe");
    }

    #[ffi_guard_macro::ffi_guard(dataset_mutation)]
    extern "C" fn dataset_mutation_panic_probe() -> i32 {
        panic!("dataset mutation panic probe");
    }

    #[ffi_guard_macro::ffi_guard(namespace_mutation)]
    extern "C" fn namespace_mutation_panic_probe() -> i32 {
        panic!("namespace mutation panic probe");
    }

    fn consume_last_error_message() -> String {
        let message = lance_last_error_message();
        assert!(!message.is_null());
        let message_text = unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        unsafe { lance_free_string(message) };
        message_text
    }

    #[test]
    fn ffi_guard_converts_panics_to_last_error() {
        assert_eq!(ffi_panic_probe(), -1);
        assert_eq!(lance_last_error_code(), ErrorCode::Runtime as i32);

        let message_text = consume_last_error_message();
        assert!(message_text.contains("panic at Lance FFI boundary: ffi panic probe"));
    }

    #[test]
    fn mutation_guards_preserve_outcome_unknown_codes_on_panic() {
        assert_eq!(dataset_mutation_panic_probe(), -1);
        assert_eq!(
            lance_last_error_code(),
            ErrorCode::DatasetCommitOutcomeUnknown as i32
        );
        assert!(consume_last_error_message().contains("dataset mutation panic probe"));

        assert_eq!(namespace_mutation_panic_probe(), -1);
        assert_eq!(
            lance_last_error_code(),
            ErrorCode::NamespaceMutationOutcomeUnknown as i32
        );
        assert!(consume_last_error_message().contains("namespace mutation panic probe"));
    }
}
