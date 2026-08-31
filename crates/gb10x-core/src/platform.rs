//! Immutable platform facts shared by the GB10X runtime and tooling.

use serde::{Deserialize, Serialize};

/// CPU cache category reported by Linux sysfs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CacheType {
    /// Instruction-only cache.
    Instruction,
    /// Data-only cache.
    Data,
    /// Unified instruction/data cache.
    Unified,
}

/// One physical cache object after sysfs entries have been de-duplicated.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CpuCache {
    /// Cache level (1, 2, 3, ...).
    pub level: u8,
    /// Cache category.
    pub cache_type: CacheType,
    /// Capacity in bytes.
    pub size_bytes: u64,
    /// Coherency/cache-line size in bytes.
    pub line_bytes: u32,
    /// Linux CPU IDs sharing this cache object.
    pub shared_cpu_ids: Vec<u32>,
}

/// NVIDIA GPU properties needed by the GB10 fail-closed gate and cache planner.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuSnapshot {
    /// Runtime-reported GPU product name.
    pub name: String,
    /// CUDA compute-capability major version.
    pub compute_major: u32,
    /// CUDA compute-capability minor version.
    pub compute_minor: u32,
    /// Runtime-reported GPU L2 capacity in bytes.
    pub l2_bytes: u64,
    /// Maximum L2 capacity eligible for persisting-cache policy, when reported.
    pub persisting_l2_max_bytes: u64,
    /// Total GPU-visible unified-memory capacity in bytes.
    pub total_memory_bytes: u64,
}

/// Facts collected from the host before GB10X enables model execution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlatformSnapshot {
    /// Rust/Linux architecture string.
    pub arch: String,
    /// Kernel release string.
    pub kernel_release: String,
    /// CPU model description.
    pub cpu_model: String,
    /// Online Linux CPU IDs.
    pub online_cpus: Vec<u32>,
    /// De-duplicated CPU cache objects.
    pub caches: Vec<CpuCache>,
    /// CUDA-visible GPU facts.
    pub gpu: GpuSnapshot,
    /// Host physical-memory capacity in bytes.
    pub mem_total_bytes: u64,
    /// Base OS page size in bytes.
    pub page_size_bytes: u64,
}

#[cfg(test)]
impl PlatformSnapshot {
    /// Deterministic GB10-like fixture used only by host-independent unit tests.
    pub fn gb10_test_fixture() -> Self {
        Self {
            arch: "aarch64".to_owned(),
            kernel_release: "test-linux".to_owned(),
            cpu_model: "NVIDIA GB10".to_owned(),
            online_cpus: (0..20).collect(),
            caches: vec![
                CpuCache {
                    level: 2,
                    cache_type: CacheType::Unified,
                    size_bytes: 2 * 1024 * 1024,
                    line_bytes: 64,
                    shared_cpu_ids: vec![0],
                },
                CpuCache {
                    level: 3,
                    cache_type: CacheType::Unified,
                    size_bytes: 24 * 1024 * 1024,
                    line_bytes: 64,
                    shared_cpu_ids: (0..20).collect(),
                },
            ],
            gpu: GpuSnapshot {
                name: "NVIDIA GB10".to_owned(),
                compute_major: 12,
                compute_minor: 1,
                l2_bytes: 24 * 1024 * 1024,
                persisting_l2_max_bytes: 18 * 1024 * 1024,
                total_memory_bytes: 128 * 1024 * 1024 * 1024,
            },
            mem_total_bytes: 128 * 1024 * 1024 * 1024,
            page_size_bytes: 4096,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gb10_fixture_exposes_nonzero_cache_and_memory_facts() {
        let fixture = PlatformSnapshot::gb10_test_fixture();
        assert_eq!(fixture.arch, "aarch64");
        assert!(!fixture.caches.is_empty());
        assert!(fixture.mem_total_bytes > 0);
        assert!(fixture.gpu.l2_bytes > 0);
    }
}
