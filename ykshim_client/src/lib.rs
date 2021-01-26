//! Client to the ykshim crate in the internal workspace.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html

// FIXME handle all errors that may pass over the API boundary.

mod prod_api;
pub use prod_api::*;

mod test_api;
pub use test_api::*;
