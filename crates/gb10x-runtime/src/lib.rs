#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

//! GB10-only runtime policy and host validation.

mod validate;

pub use validate::*;
