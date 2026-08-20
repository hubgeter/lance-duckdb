use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, parse_quote, ItemFn};

/// Wrap an exported C ABI function in the caller crate's common panic guard.
///
/// The generated closure lets the caller's return type select the appropriate
/// ABI-safe failure sentinel (null, -1, or zero) after recording a structured
/// last error.
#[proc_macro_attribute]
pub fn ffi_guard(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut function = parse_macro_input!(item as ItemFn);
    let body = function.block;
    function.block = Box::new(parse_quote!({
        crate::error::catch_ffi_panic(|| #body)
    }));
    TokenStream::from(quote!(#function))
}
