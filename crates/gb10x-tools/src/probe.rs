//! Thin host-probe CLI rendering helpers over the runtime probe types.

use gb10x_runtime::linux_probe::HostProbe;
use serde_json::{Map, Value, json};

#[cfg(feature = "native-cuda")]
use gb10x_core::GpuSnapshot;
#[cfg(feature = "native-cuda")]
use gb10x_cuda::CudaDeviceInfo;
#[cfg(feature = "native-cuda")]
use gb10x_runtime::Gb10Validation;

fn host_probe_object(probe: &HostProbe) -> Result<Map<String, Value>, serde_json::Error> {
    match serde_json::to_value(probe)? {
        Value::Object(object) => Ok(object),
        _ => Ok(Map::new()),
    }
}

/// Serialize one probed host snapshot as stable pretty-printed JSON.
///
/// Until the binary is built with a real native CUDA probe path, the JSON explicitly records that
/// CUDA-native evidence is unavailable rather than allowing host facts to be mistaken for GPU
/// verification.
pub fn render_host_probe_json(probe: &HostProbe) -> Result<String, serde_json::Error> {
    let mut object = host_probe_object(probe)?;
    object.insert(
        "cuda_native".into(),
        json!({
            "state": "unavailable",
            "reason": "binary built without native-cuda feature"
        }),
    );
    serde_json::to_string_pretty(&Value::Object(object))
}

/// Convert validated native CUDA device facts into the runtime GPU snapshot without inventing or
/// normalizing any discovered capacity.
#[cfg(feature = "native-cuda")]
pub fn gpu_snapshot_from_cuda(device: &CudaDeviceInfo) -> GpuSnapshot {
    GpuSnapshot {
        name: device.name.clone(),
        compute_major: device.compute_capability.0,
        compute_minor: device.compute_capability.1,
        l2_bytes: device.l2_bytes,
        persisting_l2_max_bytes: device.persisting_l2_max_bytes,
        total_memory_bytes: device.total_memory_bytes,
    }
}

/// Serialize host facts together with real CUDA device facts and a passed GB10 validation result.
///
/// This function only renders already-observed native facts. It does not probe CUDA itself and
/// therefore remains independently testable in a CUDA build container without a physical GPU.
#[cfg(feature = "native-cuda")]
pub fn render_native_probe_json(
    host: &HostProbe,
    device: &CudaDeviceInfo,
    validation: &Gb10Validation,
) -> Result<String, serde_json::Error> {
    let mut object = host_probe_object(host)?;
    object.insert(
        "cuda_native".into(),
        json!({
            "state": "verified",
            "device": {
                "ordinal": device.device_ordinal,
                "name": &device.name,
                "compute_major": device.compute_capability.0,
                "compute_minor": device.compute_capability.1,
                "total_memory_bytes": device.total_memory_bytes,
                "l2_bytes": device.l2_bytes,
                "persisting_l2_max_bytes": device.persisting_l2_max_bytes,
                "sm_count": device.sm_count,
                "warp_size": device.warp_size
            },
            "validation": {
                "state": "passed",
                "compute_major": validation.compute_capability.0,
                "compute_minor": validation.compute_capability.1,
                "discovered_l2_bytes": validation.discovered_l2_bytes,
                "discovered_persisting_l2_max_bytes": validation.discovered_persisting_l2_max_bytes,
                "discovered_host_memory_bytes": validation.discovered_host_memory_bytes,
                "online_cpu_count": validation.online_cpu_count,
                "cache_object_count": validation.cache_object_count
            }
        }),
    );
    serde_json::to_string_pretty(&Value::Object(object))
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
        assert!(json.contains("\"cuda_native\""));
        assert!(json.contains("\"unavailable\""));
    }
}
