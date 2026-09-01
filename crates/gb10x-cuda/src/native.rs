//! Feature-gated Rust wrapper over the stable GB10X CUDA C ABI.

use crate::{CudaDeviceInfo, CudaDeviceInfoError, CudaDeviceInfoRawV1};
use std::mem::MaybeUninit;
use thiserror::Error;

unsafe extern "C" {
    fn gb10x_cuda_probe_device(ordinal: i32, out: *mut CudaDeviceInfoRawV1) -> i32;
    fn gb10x_cuda_smoke_v1(elements: u64, checksum: *mut u64) -> i32;
}

/// Failure while invoking or validating a native CUDA operation.
#[derive(Debug, Error)]
pub enum CudaNativeError {
    /// The native C ABI returned a nonzero status code.
    #[error("native CUDA operation returned status {0}")]
    NativeStatus(i32),
    /// Native bytes were returned, but they violated the GB10 device contract.
    #[error(transparent)]
    DeviceInfo(#[from] CudaDeviceInfoError),
}

/// Probe one CUDA device through the native CUDA Runtime bridge and validate it as GB10.
///
/// A successful result is never synthesized: the C ABI must return success, after which the raw
/// bytes are revalidated by [`CudaDeviceInfo::try_from`].
pub fn probe_device(ordinal: i32) -> Result<CudaDeviceInfo, CudaNativeError> {
    let mut raw = MaybeUninit::<CudaDeviceInfoRawV1>::uninit();
    // SAFETY: `raw` points to writable storage of the exact `#[repr(C)]` ABI type. The native
    // function initializes the complete object only when it returns status zero; `assume_init`
    // is therefore performed exclusively on that success path.
    let status = unsafe { gb10x_cuda_probe_device(ordinal, raw.as_mut_ptr()) };
    if status != 0 {
        return Err(CudaNativeError::NativeStatus(status));
    }

    // SAFETY: status zero is the C ABI contract that every field, including the fixed name buffer,
    // has been initialized. Rust still validates all semantic invariants before exposing the data.
    let raw = unsafe { raw.assume_init() };
    Ok(CudaDeviceInfo::try_from(raw)?)
}

/// Execute the deterministic GB10 CUDA smoke kernel and return its 64-bit wrapping checksum.
///
/// The native operation allocates and fills device memory, reads it through a separate reduction
/// kernel, and copies only the compact checksum back to the host.
pub fn run_smoke(elements: u64) -> Result<u64, CudaNativeError> {
    let mut checksum = 0_u64;
    // SAFETY: `checksum` is valid writable storage for the stable C ABI output. The native function
    // writes it only on success; Rust exposes the value only after a zero status is returned.
    let status = unsafe { gb10x_cuda_smoke_v1(elements, &mut checksum) };
    if status != 0 {
        return Err(CudaNativeError::NativeStatus(status));
    }
    Ok(checksum)
}
