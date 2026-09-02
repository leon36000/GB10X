#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

//! GB10-only runtime policy and host validation.

/// Host-independent CPU cache ownership policy.
pub mod cache_fabric;
/// Linux CPU/cache topology and host probing.
pub mod linux_probe;
/// Host-independent PLE-Hydra tier simulation.
pub mod ple_hydra;
mod validate;

pub use validate::*;
