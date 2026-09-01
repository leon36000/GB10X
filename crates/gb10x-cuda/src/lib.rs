#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

//! GB10-only CUDA build and native execution contracts.

mod device;
#[cfg(feature = "native-cuda")]
mod native;
mod toolchain;

pub use device::*;
#[cfg(feature = "native-cuda")]
pub use native::*;
pub use toolchain::*;
