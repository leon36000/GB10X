//! Feature-gated Rust wrapper over the stable GB10X CUDA C ABI.

use crate::{CudaDeviceInfo, CudaDeviceInfoError, CudaDeviceInfoRawV1};
use std::mem::MaybeUninit;
use thiserror::Error;

const RMSNORM_WIDTH: usize = 2560;

unsafe extern "C" {
    fn gb10x_cuda_probe_device(ordinal: i32, out: *mut CudaDeviceInfoRawV1) -> i32;
    fn gb10x_cuda_smoke_v1(elements: u64, checksum: *mut u64) -> i32;
    fn gb10x_cuda_rmsnorm_bf16_device_v1(
        input_device: *const u16,
        weight_device: *const u16,
        output_device: *mut u16,
    ) -> i32;
    fn gb10x_cuda_rmsnorm_bf16_host_test_v1(
        input_host: *const u16,
        weight_host: *const u16,
        output_host: *mut u16,
    ) -> i32;
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
    /// A host-side RMSNorm test vector did not match the fixed Qwen width.
    #[error("RMSNorm {field} length must be {expected}, found {actual}")]
    RmsNormLength {
        /// Vector whose length was invalid.
        field: &'static str,
        /// Fixed Qwen hidden width required by this kernel.
        expected: usize,
        /// Caller-provided vector length.
        actual: usize,
    },
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

/// Execute one fixed-width BF16 RMSNorm row using caller-owned CUDA device buffers.
///
/// The input, weight and output buffers each contain exactly 2560 BF16 values represented by their
/// 16-bit storage bits. Accumulation is FP32 and epsilon is fixed to `1e-6`.
///
/// # Safety
///
/// `input_device` and `weight_device` must each point to at least 2560 readable BF16 values in CUDA
/// device memory, and `output_device` must point to at least 2560 writable BF16 values in CUDA
/// device memory. The buffers must remain valid until this synchronous call returns.
pub unsafe fn rmsnorm_bf16_device(
    input_device: *const u16,
    weight_device: *const u16,
    output_device: *mut u16,
) -> Result<(), CudaNativeError> {
    // SAFETY: the caller upholds the device-pointer validity contract documented above. The C ABI
    // performs no host dereference of those pointers and synchronizes the launched kernel.
    let status = unsafe {
        gb10x_cuda_rmsnorm_bf16_device_v1(input_device, weight_device, output_device)
    };
    if status != 0 {
        return Err(CudaNativeError::NativeStatus(status));
    }
    Ok(())
}

/// Test-only convenience path that copies one BF16 row through real CUDA device memory.
#[doc(hidden)]
pub fn rmsnorm_bf16_host_for_test(
    input: &[u16],
    weight: &[u16],
) -> Result<Vec<u16>, CudaNativeError> {
    for (field, actual) in [("input", input.len()), ("weight", weight.len())] {
        if actual != RMSNORM_WIDTH {
            return Err(CudaNativeError::RmsNormLength {
                field,
                expected: RMSNORM_WIDTH,
                actual,
            });
        }
    }

    let mut output = vec![0_u16; RMSNORM_WIDTH];
    // SAFETY: the slices above were validated to contain exactly one full row, and `output` owns
    // writable storage for the same number of BF16 storage values for the duration of the call.
    let status = unsafe {
        gb10x_cuda_rmsnorm_bf16_host_test_v1(
            input.as_ptr(),
            weight.as_ptr(),
            output.as_mut_ptr(),
        )
    };
    if status != 0 {
        return Err(CudaNativeError::NativeStatus(status));
    }
    Ok(output)
}
