#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

//! GB10-only runtime policy and host validation.

/// Linux CPU/cache topology and host probing.
pub mod linux_probe;
mod validate;

pub use validate::*;
