//! Client to the ykshim crate in the internal workspace.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html
//!
//! Put anything used only in testing in the `test_api` module. Everything else should go in the
//! `prod_api` module.
//!
//! The exception to this rule is `Drop` implementations for opaque pointer wrappers. These should
//! always go in the `prod_api` module. It's hard to know all of the call sites for `drop()` since
//! they are implicit, so let's assume the production API does use them.

// FIXME handle all errors that may pass over the API boundary.

mod prod_api;
pub use prod_api::*;

mod test_api;
pub use test_api::*;
