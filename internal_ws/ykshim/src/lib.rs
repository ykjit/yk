//! API to the optimised internals of the JIT runtime.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html

/// The production API.
mod prod_api;

/// The testing API.
/// These functions are only exposed to allow testing from the external workspace.
mod test_api;
