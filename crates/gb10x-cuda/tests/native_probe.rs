#![cfg(feature = "native-cuda")]

use gb10x_cuda::probe_device;

#[test]
fn native_probe_device_zero_requires_real_gb10_facts() {
    let info = probe_device(0).expect("device 0 must be a valid GB10 CUDA device");
    assert_eq!(info.device_ordinal, 0);
    assert_eq!(info.compute_capability, (12, 1));
    assert!(!info.name.trim().is_empty());
    assert!(info.total_memory_bytes > 0);
    assert!(info.l2_bytes > 0);
    assert!(info.sm_count > 0);
    assert!(info.warp_size > 0);
    assert!(info.persisting_l2_max_bytes <= info.l2_bytes);
}
