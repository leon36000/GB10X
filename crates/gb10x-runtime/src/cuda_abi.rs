//! C-compatible CUDA ABI types and unlinked native entry-point declarations.

use core::ffi::c_char;
use core::mem::size_of;

/// C-compatible status value returned by the CUDA ABI.
pub type CudaStatus = u32;

/// Version of the fixed CUDA ABI contract.
pub const CUDA_ABI_VERSION: u32 = 1;
/// Successful CUDA ABI operation.
pub const CUDA_STATUS_OK: CudaStatus = 0;
/// Caller and CUDA library disagree about an ABI struct layout.
pub const CUDA_STATUS_ABI_MISMATCH: CudaStatus = 1;
/// Caller supplied an invalid argument.
pub const CUDA_STATUS_INVALID_ARGUMENT: CudaStatus = 2;
/// A CUDA runtime query failed.
pub const CUDA_STATUS_CUDA_RUNTIME_FAILURE: CudaStatus = 3;
/// The CUDA ABI encountered an internal invariant violation.
pub const CUDA_STATUS_INTERNAL_FAILURE: CudaStatus = 4;
/// Encoding used by ABI v1 for the architecture-specific `a` SM variant.
pub const CUDA_TARGET_SM_VARIANT_A: u32 = 1;

/// Build facts reported by the CUDA ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaAbiInfo {
    /// Caller-provided ABI struct size in bytes.
    pub struct_size: u32,
    /// ABI version implemented by the CUDA library.
    pub abi_version: u32,
    /// CUDA SM target major version compiled into the library.
    pub target_sm_major: u32,
    /// CUDA SM target minor version compiled into the library.
    pub target_sm_minor: u32,
    /// CUDA SM target variant compiled into the library.
    pub target_sm_variant: u32,
    /// CUDA runtime header version used to compile the library.
    pub cuda_runtime_header_version: u32,
    /// CUDA runtime version loaded by the library.
    pub cuda_runtime_loaded_version: u32,
    /// Reserved and zero on successful ABI v1 calls.
    pub reserved0: u32,
}

impl CudaAbiInfo {
    /// Construct an output buffer initialized for a CUDA ABI v1 call.
    pub const fn new() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            target_sm_major: 0,
            target_sm_minor: 0,
            target_sm_variant: 0,
            cuda_runtime_header_version: 0,
            cuda_runtime_loaded_version: 0,
            reserved0: 0,
        }
    }
}

impl Default for CudaAbiInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// CUDA device facts reported by the CUDA ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfo {
    /// Caller-provided ABI struct size in bytes.
    pub struct_size: u32,
    /// CUDA device ordinal queried by the ABI.
    pub device_ordinal: u32,
    /// Device compute-capability major version.
    pub compute_major: u32,
    /// Device compute-capability minor version.
    pub compute_minor: u32,
    /// Device global-memory capacity in bytes.
    pub total_global_memory_bytes: u64,
    /// Device L2-cache capacity in bytes.
    pub l2_cache_bytes: u64,
    /// Device maximum persisting-L2 allocation in bytes.
    pub persisting_l2_max_bytes: u64,
    /// Nul-terminated CUDA device name encoded as bytes.
    pub name: [u8; 256],
}

impl CudaDeviceInfo {
    /// Construct an output buffer initialized for a CUDA ABI v1 call.
    pub const fn new() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            device_ordinal: 0,
            compute_major: 0,
            compute_minor: 0,
            total_global_memory_bytes: 0,
            l2_cache_bytes: 0,
            persisting_l2_max_bytes: 0,
            name: [0; 256],
        }
    }
}

impl Default for CudaDeviceInfo {
    fn default() -> Self {
        Self::new()
    }
}

unsafe extern "C" {
    /// Return the ABI version implemented by the linked CUDA library.
    pub fn gb10x_cuda_abi_version() -> u32;
    /// Populate a caller-initialized [`CudaAbiInfo`] output buffer.
    pub fn gb10x_cuda_get_abi_info(out_info: *mut CudaAbiInfo) -> CudaStatus;
    /// Populate a caller-initialized [`CudaDeviceInfo`] for one CUDA device ordinal.
    pub fn gb10x_cuda_probe_device(
        device_ordinal: u32,
        out_info: *mut CudaDeviceInfo,
    ) -> CudaStatus;
    /// Return a non-null static ASCII description for one CUDA ABI status value.
    pub fn gb10x_cuda_status_string(status: CudaStatus) -> *const c_char;
}
