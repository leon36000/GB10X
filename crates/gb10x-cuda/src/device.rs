//! Stable C-ABI device facts and fail-closed GB10 validation.

use thiserror::Error;

/// Version of the raw CUDA device-info C ABI.
pub const CUDA_DEVICE_INFO_ABI_V1: u32 = 1;

/// Fixed byte capacity reserved for the NUL-terminated CUDA device name.
pub const CUDA_DEVICE_NAME_BYTES: usize = 256;

/// Raw device properties crossing the C ABI from CUDA C++ into Rust.
///
/// This is deliberately independent of `cudaDeviceProp` so CUDA toolkit struct layout never
/// becomes part of GB10X's Rust ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfoRawV1 {
    /// ABI version; must equal [`CUDA_DEVICE_INFO_ABI_V1`].
    pub abi_version: u32,
    /// CUDA device ordinal used for the probe.
    pub device_ordinal: i32,
    /// CUDA compute-capability major component.
    pub compute_major: u32,
    /// CUDA compute-capability minor component.
    pub compute_minor: u32,
    /// CUDA-reported total device-visible memory in bytes.
    pub total_memory_bytes: u64,
    /// CUDA-reported L2 cache capacity in bytes.
    pub l2_bytes: u64,
    /// Maximum persisting-L2 capacity reported by the runtime, or zero when unavailable.
    pub persisting_l2_max_bytes: u64,
    /// Number of streaming multiprocessors reported by CUDA.
    pub sm_count: u32,
    /// CUDA warp size.
    pub warp_size: u32,
    /// NUL-terminated UTF-8 CUDA device name.
    pub name: [u8; CUDA_DEVICE_NAME_BYTES],
}

/// Validated CUDA properties accepted by the GB10X native layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfo {
    /// CUDA device ordinal.
    pub device_ordinal: i32,
    /// Exact CUDA compute capability.
    pub compute_capability: (u32, u32),
    /// Runtime-reported UTF-8 device name.
    pub name: String,
    /// Runtime-reported device-visible memory capacity.
    pub total_memory_bytes: u64,
    /// Runtime-reported GPU L2 capacity.
    pub l2_bytes: u64,
    /// Runtime-reported persisting-L2 maximum; zero means unavailable/unsupported.
    pub persisting_l2_max_bytes: u64,
    /// Runtime-reported SM count.
    pub sm_count: u32,
    /// Runtime-reported warp size.
    pub warp_size: u32,
}

/// Failure while converting raw CUDA C-ABI facts into a GB10X device contract.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CudaDeviceInfoError {
    /// Raw ABI version is not supported.
    #[error("unsupported CUDA device-info ABI version {found}")]
    AbiVersion {
        /// Version received across the C ABI.
        found: u32,
    },
    /// Device ordinal is invalid.
    #[error("CUDA device ordinal must be non-negative, found {0}")]
    DeviceOrdinal(i32),
    /// Compute capability is not the GB10 target 12.1.
    #[error("GB10X requires CUDA compute capability 12.1, found {major}.{minor}")]
    ComputeCapability {
        /// Discovered major component.
        major: u32,
        /// Discovered minor component.
        minor: u32,
    },
    /// Required runtime geometry was zero.
    #[error("CUDA device-info field {0} must be nonzero")]
    ZeroGeometry(&'static str),
    /// Persisting-L2 capacity cannot exceed total L2 capacity.
    #[error(
        "persisting-L2 maximum {persisting_l2_max_bytes} exceeds L2 capacity {l2_bytes}"
    )]
    PersistingL2ExceedsL2 {
        /// Runtime-reported persisting-L2 maximum.
        persisting_l2_max_bytes: u64,
        /// Runtime-reported total L2 capacity.
        l2_bytes: u64,
    },
    /// Fixed C name buffer contained no NUL terminator.
    #[error("CUDA device name is not NUL-terminated")]
    DeviceNameNotTerminated,
    /// Device-name bytes before the NUL terminator were not valid UTF-8.
    #[error("CUDA device name is not valid UTF-8")]
    DeviceNameUtf8,
    /// Device name was empty or whitespace-only.
    #[error("CUDA device name is empty")]
    DeviceNameEmpty,
}

impl TryFrom<CudaDeviceInfoRawV1> for CudaDeviceInfo {
    type Error = CudaDeviceInfoError;

    fn try_from(raw: CudaDeviceInfoRawV1) -> Result<Self, Self::Error> {
        if raw.abi_version != CUDA_DEVICE_INFO_ABI_V1 {
            return Err(CudaDeviceInfoError::AbiVersion {
                found: raw.abi_version,
            });
        }
        if raw.device_ordinal < 0 {
            return Err(CudaDeviceInfoError::DeviceOrdinal(raw.device_ordinal));
        }
        if (raw.compute_major, raw.compute_minor) != (12, 1) {
            return Err(CudaDeviceInfoError::ComputeCapability {
                major: raw.compute_major,
                minor: raw.compute_minor,
            });
        }
        require_nonzero(raw.total_memory_bytes, "total_memory_bytes")?;
        require_nonzero(raw.l2_bytes, "l2_bytes")?;
        require_nonzero(raw.sm_count, "sm_count")?;
        require_nonzero(raw.warp_size, "warp_size")?;
        if raw.persisting_l2_max_bytes > raw.l2_bytes {
            return Err(CudaDeviceInfoError::PersistingL2ExceedsL2 {
                persisting_l2_max_bytes: raw.persisting_l2_max_bytes,
                l2_bytes: raw.l2_bytes,
            });
        }

        let nul = raw
            .name
            .iter()
            .position(|&byte| byte == 0)
            .ok_or(CudaDeviceInfoError::DeviceNameNotTerminated)?;
        let name = std::str::from_utf8(&raw.name[..nul])
            .map_err(|_| CudaDeviceInfoError::DeviceNameUtf8)?;
        if name.trim().is_empty() {
            return Err(CudaDeviceInfoError::DeviceNameEmpty);
        }

        Ok(Self {
            device_ordinal: raw.device_ordinal,
            compute_capability: (raw.compute_major, raw.compute_minor),
            name: name.to_owned(),
            total_memory_bytes: raw.total_memory_bytes,
            l2_bytes: raw.l2_bytes,
            persisting_l2_max_bytes: raw.persisting_l2_max_bytes,
            sm_count: raw.sm_count,
            warp_size: raw.warp_size,
        })
    }
}

fn require_nonzero<T>(value: T, field: &'static str) -> Result<(), CudaDeviceInfoError>
where
    T: PartialEq + From<u8>,
{
    if value == T::from(0) {
        Err(CudaDeviceInfoError::ZeroGeometry(field))
    } else {
        Ok(())
    }
}
