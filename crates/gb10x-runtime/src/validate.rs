//! Fail-closed validation for the only supported GB10X production platform.

use gb10x_core::PlatformSnapshot;
use thiserror::Error;

/// Platform-validation failure.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum PlatformError {
    /// Host architecture is not Linux AArch64 as required by GB10X.
    #[error("GB10X requires aarch64, found {0}")]
    Architecture(String),
    /// GPU product identity does not identify a GB10 device.
    #[error("GB10X requires an NVIDIA GB10 GPU, found {0}")]
    GpuIdentity(String),
    /// CUDA compute capability is not exactly 12.1.
    #[error("GB10X requires compute capability 12.1, found {major}.{minor}")]
    ComputeCapability {
        /// Discovered compute-capability major version.
        major: u32,
        /// Discovered compute-capability minor version.
        minor: u32,
    },
    /// A required discovered capacity/topology fact is missing or zero.
    #[error("GB10X platform probe is incomplete: {0}")]
    Incomplete(&'static str),
}

/// Validated hardware facts retained for later cache/memory planning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gb10Validation {
    /// Validated compute capability.
    pub compute_capability: (u32, u32),
    /// Discovered GPU L2 capacity; never replaced by a published constant.
    pub discovered_l2_bytes: u64,
    /// Discovered persisting-L2 limit.
    pub discovered_persisting_l2_max_bytes: u64,
    /// Discovered host physical-memory capacity.
    pub discovered_host_memory_bytes: u64,
    /// Number of online CPU IDs observed by Linux.
    pub online_cpu_count: usize,
    /// Number of de-duplicated cache objects observed by Linux.
    pub cache_object_count: usize,
}

/// Validate that `snapshot` describes the one production platform supported by GB10X.
///
/// The validator intentionally checks discovered facts instead of substituting expected
/// published capacities. Exact cache sizes are consumed later by the cache planner.
pub fn validate_gb10(snapshot: &PlatformSnapshot) -> Result<Gb10Validation, PlatformError> {
    if snapshot.arch != "aarch64" {
        return Err(PlatformError::Architecture(snapshot.arch.clone()));
    }

    if !snapshot.gpu.name.to_ascii_uppercase().contains("GB10") {
        return Err(PlatformError::GpuIdentity(snapshot.gpu.name.clone()));
    }

    if (snapshot.gpu.compute_major, snapshot.gpu.compute_minor) != (12, 1) {
        return Err(PlatformError::ComputeCapability {
            major: snapshot.gpu.compute_major,
            minor: snapshot.gpu.compute_minor,
        });
    }

    if snapshot.online_cpus.is_empty() {
        return Err(PlatformError::Incomplete("online CPU list is empty"));
    }
    if snapshot.caches.is_empty() {
        return Err(PlatformError::Incomplete("CPU cache topology is empty"));
    }
    if snapshot.caches.iter().any(|cache| {
        cache.size_bytes == 0 || cache.line_bytes == 0 || cache.shared_cpu_ids.is_empty()
    }) {
        return Err(PlatformError::Incomplete("CPU cache entry contains zero/empty facts"));
    }
    if snapshot.mem_total_bytes == 0 {
        return Err(PlatformError::Incomplete("host memory capacity is zero"));
    }
    if snapshot.page_size_bytes == 0 {
        return Err(PlatformError::Incomplete("base page size is zero"));
    }
    if snapshot.gpu.l2_bytes == 0 {
        return Err(PlatformError::Incomplete("GPU L2 capacity is zero"));
    }
    if snapshot.gpu.total_memory_bytes == 0 {
        return Err(PlatformError::Incomplete("GPU-visible memory capacity is zero"));
    }
    if snapshot.gpu.persisting_l2_max_bytes > snapshot.gpu.l2_bytes {
        return Err(PlatformError::Incomplete(
            "persisting-L2 limit exceeds total GPU L2 capacity",
        ));
    }

    Ok(Gb10Validation {
        compute_capability: (12, 1),
        discovered_l2_bytes: snapshot.gpu.l2_bytes,
        discovered_persisting_l2_max_bytes: snapshot.gpu.persisting_l2_max_bytes,
        discovered_host_memory_bytes: snapshot.mem_total_bytes,
        online_cpu_count: snapshot.online_cpus.len(),
        cache_object_count: snapshot.caches.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb10x_core::PlatformSnapshot;

    #[test]
    fn rejects_x86_even_if_gpu_claims_sm121() {
        let mut p = PlatformSnapshot::gb10_test_fixture();
        p.arch = "x86_64".into();
        assert!(matches!(
            validate_gb10(&p),
            Err(PlatformError::Architecture(_))
        ));
    }

    #[test]
    fn rejects_non_121_compute_capability() {
        let mut p = PlatformSnapshot::gb10_test_fixture();
        p.gpu.compute_major = 12;
        p.gpu.compute_minor = 0;
        assert!(matches!(
            validate_gb10(&p),
            Err(PlatformError::ComputeCapability { .. })
        ));
    }

    #[test]
    fn accepts_exact_gb10_fixture() {
        let p = PlatformSnapshot::gb10_test_fixture();
        let result = validate_gb10(&p).expect("GB10 fixture must validate");
        assert_eq!(result.compute_capability, (12, 1));
        assert!(result.discovered_l2_bytes > 0);
    }

    #[test]
    fn rejects_impossible_persisting_l2_capacity() {
        let mut p = PlatformSnapshot::gb10_test_fixture();
        p.gpu.persisting_l2_max_bytes = p.gpu.l2_bytes + 1;
        assert_eq!(
            validate_gb10(&p),
            Err(PlatformError::Incomplete(
                "persisting-L2 limit exceeds total GPU L2 capacity"
            ))
        );
    }
}
