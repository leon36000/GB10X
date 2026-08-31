//! Thin host-probe CLI rendering helpers over the runtime probe types.

use gb10x_runtime::linux_probe::HostProbe;

/// Serialize one probed host snapshot as stable pretty-printed JSON.
pub fn render_host_probe_json(probe: &HostProbe) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(probe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb10x_core::{CacheType, CpuCache};

    #[test]
    fn rendered_probe_preserves_cache_topology_and_architecture() {
        let probe = HostProbe {
            arch: "aarch64".into(),
            kernel_release: "test".into(),
            cpu_model: "NVIDIA GB10".into(),
            online_cpus: vec![0, 1],
            caches: vec![CpuCache {
                level: 2,
                cache_type: CacheType::Unified,
                size_bytes: 2 * 1024 * 1024,
                line_bytes: 64,
                shared_cpu_ids: vec![0],
            }],
            mem_total_bytes: 128 * 1024 * 1024 * 1024,
            page_size_bytes: 4096,
        };
        let json = render_host_probe_json(&probe).expect("probe JSON");
        assert!(json.contains("aarch64"));
        assert!(json.contains("NVIDIA GB10"));
        assert!(json.contains("2097152"));
    }
}
