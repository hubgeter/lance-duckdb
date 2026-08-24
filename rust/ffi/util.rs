use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

use anyhow::Context;
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use datafusion_expr::Expr;
use lance::session::Session;
use lance_core::datatypes::Schema as LanceSchema;

use crate::error::ErrorCode;

use super::types::{DatasetHandle, SchemaHandle, SessionHandle, StreamHandle};

#[derive(Debug)]
pub(crate) struct FfiError {
    pub(crate) code: ErrorCode,
    pub(crate) message: String,
}

impl FfiError {
    pub(crate) fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Convert an error returned by a Lance operation that may have crossed its
/// manifest-commit point.  Semantic and conflict errors are definitive: the
/// requested mutation did not commit.  Transport, timeout, internal, wrapped,
/// and cleanup errors cannot prove that, so callers must retain their mutation
/// lease and reconcile the dataset before retrying.
pub(crate) fn lance_mutation_outcome_unknown(error: &lance::Error) -> bool {
    // Keep this match exhaustive.  A newly introduced Lance error variant must
    // be classified deliberately instead of silently becoming retryable.
    match error {
        lance::Error::InvalidInput { .. }
        | lance::Error::DatasetAlreadyExists { .. }
        | lance::Error::SchemaMismatch { .. }
        | lance::Error::DatasetNotFound { .. }
        | lance::Error::NotSupported { .. }
        | lance::Error::CommitConflict { .. }
        | lance::Error::IncompatibleTransaction { .. }
        | lance::Error::RetryableCommitConflict { .. }
        | lance::Error::TooMuchWriteContention { .. }
        | lance::Error::Unprocessable { .. }
        | lance::Error::Arrow { .. }
        | lance::Error::Schema { .. }
        | lance::Error::IndexNotFound { .. }
        | lance::Error::InvalidTableLocation { .. }
        | lance::Error::Stop
        | lance::Error::Execution { .. }
        | lance::Error::InvalidRef { .. }
        | lance::Error::RefConflict { .. }
        | lance::Error::RefNotFound { .. }
        | lance::Error::VersionNotFound { .. }
        | lance::Error::VersionConflict { .. }
        | lance::Error::FieldNotFound { .. }
        | lance::Error::DiskCapExceeded { .. } => false,
        lance::Error::CorruptFile { .. }
        | lance::Error::Timeout { .. }
        | lance::Error::Internal { .. }
        | lance::Error::PrerequisiteFailed { .. }
        | lance::Error::NotFound { .. }
        | lance::Error::IO { .. }
        | lance::Error::Index { .. }
        | lance::Error::Wrapped { .. }
        | lance::Error::Cloned { .. }
        | lance::Error::Cleanup { .. }
        | lance::Error::Namespace { .. }
        | lance::Error::External { .. }
        | lance::Error::Fenced { .. } => true,
    }
}

pub(crate) fn lance_mutation_error(
    definitive_code: ErrorCode,
    outcome_unknown_code: ErrorCode,
    operation: &str,
    error: lance::Error,
) -> FfiError {
    FfiError::new(
        if lance_mutation_outcome_unknown(&error) {
            outcome_unknown_code
        } else {
            definitive_code
        },
        format!("{operation}: {error}"),
    )
}

pub(crate) type FfiResult<T> = Result<T, FfiError>;

pub(crate) fn output_regions_overlap<T, U>(left: *mut T, right: *mut U) -> bool {
    if left.is_null() || right.is_null() {
        return false;
    }
    let left_start = left as usize;
    let right_start = right as usize;
    let Some(left_end) = left_start.checked_add(std::mem::size_of::<T>()) else {
        return true;
    };
    let Some(right_end) = right_start.checked_add(std::mem::size_of::<U>()) else {
        return true;
    };
    left_start < right_end && right_start < left_end
}

fn is_http_header_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

pub(crate) fn parse_headers_tsv(headers_tsv: Option<&str>) -> FfiResult<Vec<(String, String)>> {
    let Some(headers_tsv) = headers_tsv else {
        return Ok(Vec::new());
    };

    headers_tsv
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let line_number = index + 1;
            let (name, value) = line.split_once('\t').ok_or_else(|| {
                FfiError::new(
                    ErrorCode::InvalidArgument,
                    format!("headers_tsv line {line_number} must contain one tab separator"),
                )
            })?;
            if name.is_empty() || !name.bytes().all(is_http_header_name_byte) {
                return Err(FfiError::new(
                    ErrorCode::InvalidArgument,
                    format!("headers_tsv line {line_number} has an invalid HTTP header name"),
                ));
            }
            if value.bytes().any(|byte| !(b' '..=b'~').contains(&byte)) {
                return Err(FfiError::new(
                    ErrorCode::InvalidArgument,
                    format!("headers_tsv line {line_number} has an invalid HTTP header value"),
                ));
            }
            Ok((name.to_string(), value.to_string()))
        })
        .collect()
}

pub(crate) fn to_c_string(s: impl AsRef<str>) -> CString {
    match CString::new(s.as_ref()) {
        Ok(v) => v,
        Err(_) => CString::new(s.as_ref().replace('\0', "\\0"))
            .unwrap_or_else(|_| CString::new("invalid string").unwrap()),
    }
}

pub(crate) fn ffi_output_string(
    value: impl Into<Vec<u8>>,
    code: ErrorCode,
    what: &str,
) -> FfiResult<CString> {
    CString::new(value).map_err(|_| {
        FfiError::new(
            code,
            format!("{what} contains a NUL byte and cannot cross the C ABI"),
        )
    })
}

pub(crate) fn push_ffi_key_value_row(output: &mut String, key: &str, value: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (index, field) in [key, value].into_iter().enumerate() {
        for byte in field.as_bytes() {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        if index == 0 {
            output.push('\t');
        }
    }
    output.push('\n');
}

pub(crate) fn join_ffi_lines(values: &[String], code: ErrorCode, what: &str) -> FfiResult<String> {
    for value in values {
        if value.is_empty()
            || value
                .bytes()
                .any(|byte| matches!(byte, b'\0' | b'\n' | b'\r'))
        {
            return Err(FfiError::new(
                code,
                format!("{what} contains an empty or unrepresentable table name"),
            ));
        }
    }
    Ok(values.join("\n"))
}

pub(crate) fn canonicalize_lance_field_path(
    schema: &LanceSchema,
    column: &str,
    what: &'static str,
) -> FfiResult<String> {
    let column = column.trim();
    if column.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} cannot be empty"),
        ));
    }

    let field = schema.field_case_insensitive(column).ok_or_else(|| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} not found: '{column}'"),
        )
    })?;

    schema.field_path(field.id).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} path normalize: {err}"),
        )
    })
}

pub(crate) unsafe fn cstr_to_str<'a>(ptr: *const c_char, what: &'static str) -> FfiResult<&'a str> {
    if ptr.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null"),
        ));
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .context("utf8 decode")
        .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("{what} utf8: {err}")))?;
    Ok(s)
}

pub(crate) unsafe fn optional_cstr_to_string(
    ptr: *const c_char,
    what: &'static str,
) -> FfiResult<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }
    Ok(Some(unsafe { cstr_to_str(ptr, what)? }.to_string()))
}

pub(crate) unsafe fn slice_from_ptr<'a, T>(
    ptr: *const T,
    len: usize,
    what: &'static str,
) -> FfiResult<&'a [T]> {
    if ptr.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null"),
        ));
    }
    // SAFETY: Caller guarantees ptr points to at least len elements.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

pub(crate) unsafe fn optional_cstr_array(
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

    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    let mut out = Vec::with_capacity(len);
    for (idx, &item) in slice.iter().enumerate() {
        if item.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("{what}[{idx}] is null"),
            ));
        }
        let s = unsafe { CStr::from_ptr(item) }
            .to_str()
            .context("utf8 decode")
            .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("{what}[{idx}] utf8: {err}")))?;
        out.push(s.to_string());
    }
    Ok(out)
}

pub(crate) unsafe fn parse_optional_filter_ir(
    filter_ir: *const u8,
    filter_ir_len: usize,
    code: ErrorCode,
    what: &'static str,
) -> FfiResult<Option<Expr>> {
    if filter_ir_len == 0 {
        return Ok(None);
    }
    if filter_ir.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null with non-zero length"),
        ));
    }

    let bytes = unsafe { std::slice::from_raw_parts(filter_ir, filter_ir_len) };
    crate::filter_ir::parse_filter_ir(bytes)
        .map(Some)
        .map_err(|err| FfiError::new(code, format!("{what} parse: {err}")))
}

pub(crate) fn u64_to_usize(v: u64, what: &'static str) -> FfiResult<usize> {
    usize::try_from(v)
        .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, format!("invalid {what}: {err}")))
}

pub(crate) fn nonzero_u64_to_usize(v: u64, what: &'static str) -> FfiResult<usize> {
    let v = u64_to_usize(v, what)?;
    if v == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} must be > 0"),
        ));
    }
    Ok(v)
}

pub(crate) fn nonzero_u64_to_i64(v: u64, what: &'static str) -> FfiResult<i64> {
    let v = i64::try_from(v).map_err(|err| {
        FfiError::new(ErrorCode::InvalidArgument, format!("invalid {what}: {err}"))
    })?;
    if v <= 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} must be > 0"),
        ));
    }
    Ok(v)
}

pub(crate) unsafe fn dataset_handle<'a>(dataset: *mut c_void) -> FfiResult<&'a DatasetHandle> {
    if dataset.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "dataset is null"));
    }
    // SAFETY: Caller guarantees dataset points to a valid DatasetHandle.
    Ok(unsafe { &*(dataset as *const DatasetHandle) })
}

pub(crate) unsafe fn optional_session_handle(
    session: *mut c_void,
) -> FfiResult<Option<Arc<Session>>> {
    if session.is_null() {
        return Ok(None);
    }
    let handle = unsafe { &*(session as *const SessionHandle) };
    Ok(Some(handle.session.clone()))
}

pub(crate) unsafe fn stream_handle_mut<'a>(stream: *mut c_void) -> FfiResult<&'a mut StreamHandle> {
    if stream.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "stream is null"));
    }
    // SAFETY: Caller guarantees stream points to a valid StreamHandle.
    Ok(unsafe { &mut *(stream as *mut StreamHandle) })
}

pub(crate) unsafe fn schema_handle<'a>(schema: *mut c_void) -> FfiResult<&'a SchemaHandle> {
    if schema.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "schema is null"));
    }
    // SAFETY: Caller guarantees schema points to a valid SchemaHandle.
    Ok(unsafe { &*(schema as *const SchemaHandle) })
}

pub(crate) unsafe fn batch_handle<'a>(batch: *mut c_void) -> FfiResult<&'a RecordBatch> {
    if batch.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "batch is null"));
    }
    // SAFETY: Caller guarantees batch points to a valid RecordBatch.
    Ok(unsafe { &*(batch as *const RecordBatch) })
}

pub(crate) fn schema_to_ffi_arrow_schema(
    schema: &Schema,
) -> FfiResult<arrow::ffi::FFI_ArrowSchema> {
    let data_type = arrow::datatypes::DataType::Struct(schema.fields().clone());
    arrow::ffi::FFI_ArrowSchema::try_from(&data_type)
        .map_err(|err| FfiError::new(ErrorCode::SchemaExport, format!("schema export: {err}")))
}

#[cfg(test)]
mod tests {
    use super::{
        join_ffi_lines, lance_mutation_error, output_regions_overlap, parse_headers_tsv,
        push_ffi_key_value_row,
    };
    use crate::error::ErrorCode;

    #[test]
    fn parses_valid_headers_tsv() {
        assert_eq!(
            parse_headers_tsv(Some("x-api-key\tsecret\nx-empty\t")).unwrap(),
            vec![
                ("x-api-key".to_string(), "secret".to_string()),
                ("x-empty".to_string(), String::new()),
            ]
        );
        assert!(parse_headers_tsv(None).unwrap().is_empty());
    }

    #[test]
    fn rejects_malformed_or_unsafe_headers_tsv() {
        for headers in [
            "missing-separator",
            "\tvalue",
            "bad name\tvalue",
            "name\tvalue\textra",
            "name\tvalue\rwith-control",
        ] {
            assert!(
                parse_headers_tsv(Some(headers)).is_err(),
                "expected invalid header TSV to fail: {headers:?}"
            );
        }
    }

    #[test]
    fn line_transport_rejects_empty_nul_and_newline_names() {
        for name in ["", "nul\0name", "line\nbreak", "carriage\rreturn"] {
            let error = join_ffi_lines(
                &[name.to_string()],
                ErrorCode::NamespaceListTables,
                "namespace list_tables response",
            )
            .unwrap_err();
            assert_eq!(error.code as i32, ErrorCode::NamespaceListTables as i32);
            assert!(error.message.contains("empty or unrepresentable"));
        }
        assert_eq!(
            join_ffi_lines(
                &["one".to_string(), "two".to_string()],
                ErrorCode::NamespaceListTables,
                "namespace list_tables response",
            )
            .unwrap(),
            "one\ntwo"
        );
    }

    #[test]
    fn key_value_transport_encodes_all_utf8_bytes_without_delimiter_collisions() {
        let mut output = String::new();
        push_ffi_key_value_row(&mut output, "same\t\n\0", "same\t\n\0");
        push_ffi_key_value_row(&mut output, "雪", "");
        assert_eq!(output, "73616d65090a00\t73616d65090a00\ne99baa\t\n");
    }

    #[test]
    fn mutation_errors_distinguish_definitive_failures_from_unknown_outcomes() {
        let definitive = lance_mutation_error(
            ErrorCode::DatasetAddColumns,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "add columns",
            lance::Error::invalid_input("bad expression"),
        );
        assert_eq!(definitive.code as i32, ErrorCode::DatasetAddColumns as i32);

        let unknown = lance_mutation_error(
            ErrorCode::DatasetAddColumns,
            ErrorCode::DatasetCommitOutcomeUnknown,
            "add columns",
            lance::Error::timeout("commit acknowledgement"),
        );
        assert_eq!(
            unknown.code as i32,
            ErrorCode::DatasetCommitOutcomeUnknown as i32
        );
    }

    #[test]
    fn output_region_validation_rejects_exact_and_partial_aliases() {
        let mut storage = [0_u8; 32];
        let start = storage.as_mut_ptr();
        assert!(output_regions_overlap(
            start.cast::<u64>(),
            start.cast::<usize>()
        ));
        assert!(output_regions_overlap(
            start.cast::<u64>(),
            unsafe { start.add(4) }.cast::<usize>()
        ));
        assert!(!output_regions_overlap(
            start.cast::<u64>(),
            unsafe { start.add(8) }.cast::<usize>()
        ));
        assert!(!output_regions_overlap(
            std::ptr::null_mut::<u64>(),
            start.cast::<usize>()
        ));
    }
}
