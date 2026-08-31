#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

//! Exact PLE storage/layout support for GB10X.

mod format;
mod layout;

pub use format::*;
pub use layout::*;
