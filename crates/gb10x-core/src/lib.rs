#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

//! Core immutable contracts for GB10X.

mod platform;
mod ple_hash;
mod qwen38;

pub use platform::*;
pub use ple_hash::*;
pub use qwen38::*;
