use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, parse_quote, Ident, ItemFn};

/// Wrap an exported C ABI function in the caller crate's common panic guard.
///
/// The generated closure lets the caller's return type select the appropriate
/// ABI-safe failure sentinel (null, -1, or zero) after recording a structured
/// last error. Mutation entry points pass `dataset_mutation` or
/// `namespace_mutation` so a panic cannot be misreported as safely retryable.
#[proc_macro_attribute]
pub fn ffi_guard(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut function = parse_macro_input!(item as ItemFn);
    let body = function.block;
    function.block = if attr.is_empty() {
        Box::new(parse_quote!({
            crate::error::catch_ffi_panic(|| #body)
        }))
    } else {
        let guard_kind = parse_macro_input!(attr as Ident).to_string();
        let error_code = match guard_kind.as_str() {
            "dataset_mutation" => quote!(crate::error::ErrorCode::DatasetCommitOutcomeUnknown),
            "namespace_mutation" => {
                quote!(crate::error::ErrorCode::NamespaceMutationOutcomeUnknown)
            }
            _ => {
                return syn::Error::new_spanned(
                    function.sig.ident,
                    "ffi_guard accepts only dataset_mutation or namespace_mutation",
                )
                .to_compile_error()
                .into();
            }
        };
        Box::new(parse_quote!({
            crate::error::catch_ffi_panic_with_code(#error_code, || #body)
        }))
    };
    TokenStream::from(quote!(#function))
}
