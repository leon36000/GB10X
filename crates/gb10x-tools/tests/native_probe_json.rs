#![cfg(feature = "native-cuda")]

use gb10x_core::{CacheType, CpuCache};
use gb10x_cuda::CudaDeviceInfo;
use gb10x_runtime::linux_probe::HostProbe;
use gb10x_runtime::validate_gb10;
use gb10x_tools::probe::{gpu_snapshot_from_cuda, render_native_probe_json};
use serde_json::Value;

fn host_probe() -> HostProbe {
    HostProbe {
        arch: "aarch64".into(),
        kernel_release: "test-linux".into(),
        cpu_model: "NVIDIA GB10".into(),
        online_cpus: (0..20).collect(),
        caches: vec![CpuCache {
            level: 3,
            cache_type: CacheType::Unified,
            size_bytes: 24 * 1024 * 1024,
            line_bytes: 64,
            shared_cpu_ids: (0..20).collect(),
        }],
        mem_total_bytes: 128 * 1024 * 1024 * 1024,
        page_size_bytes: 4096,
    }
}

fn cuda_device() -> CudaDeviceInfo {
    CudaDeviceInfo {
        device_ordinal: 0,
        compute_capability: (12, 1),
        name: "NVIDIA GB10".into(),
        total_memory_bytes: 128 * 1024 * 1024 * 1024,
        l2_bytes: 24 * 1024 * 1024,
        persisting_l2_max_bytes: 18 * 1024 * 1024,
        sm_count: 20,
        warp_size: 32,
    }
}

#[test]
fn native_json_uses_cuda_device_facts_and_reports_validation() {
    let host = host_probe();
    let device = cuda_device();
    let gpu = gpu_snapshot_from_cuda(&device);

    assert_eq!(gpu.name, device.name);
    assert_eq!((gpu.compute_major, gpu.compute_minor), (12, 1));
    assert_eq!(gpu.total_memory_bytes, device.total_memory_bytes);
    assert_eq!(gpu.l2_bytes, device.l2_bytes);
    assert_eq!(gpu.persisting_l2_max_bytes, device.persisting_l2_max_bytes);

    let snapshot = host.clone().into_platform_snapshot(gpu);
    let validation = validate_gb10(&snapshot).expect("synthetic GB10 facts must validate");
    let rendered = render_native_probe_json(&host, &device, &validation).expect("native JSON");
    let json: Value = serde_json::from_str(&rendered).expect("valid JSON");

    assert_eq!(json["cuda_native"]["state"], "verified");
    assert_eq!(json["cuda_native"]["device"]["ordinal"], 0);
    assert_eq!(json["cuda_native"]["device"]["name"], "NVIDIA GB10");
    assert_eq!(json["cuda_native"]["device"]["compute_major"], 12);
    assert_eq!(json["cuda_native"]["device"]["compute_minor"], 1);
    assert_eq!(json["cuda_native"]["device"]["l2_bytes"], device.l2_bytes);
    assert_eq!(json["cuda_native"]["validation"]["state"], "passed");
    assert_eq!(
        json["cuda_native"]["validation"]["discovered_l2_bytes"],
        validation.discovered_l2_bytes
    );
}
