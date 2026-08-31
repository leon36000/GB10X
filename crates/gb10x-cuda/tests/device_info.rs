use gb10x_cuda::{CudaDeviceInfo, CudaDeviceInfoRawV1, CudaDeviceInfoError};

const NAME_BYTES: usize = 256;

fn valid_raw() -> CudaDeviceInfoRawV1 {
    let mut name = [0_u8; NAME_BYTES];
    let label = b"NVIDIA GB10";
    name[..label.len()].copy_from_slice(label);

    CudaDeviceInfoRawV1 {
        abi_version: 1,
        device_ordinal: 0,
        compute_major: 12,
        compute_minor: 1,
        total_memory_bytes: 128 * 1024 * 1024 * 1024,
        l2_bytes: 24 * 1024 * 1024,
        persisting_l2_max_bytes: 18 * 1024 * 1024,
        sm_count: 20,
        warp_size: 32,
        name,
    }
}

#[test]
fn valid_raw_device_info_converts_without_published_capacity_assumptions() {
    let info = CudaDeviceInfo::try_from(valid_raw()).expect("valid GB10-shaped raw info");
    assert_eq!(info.device_ordinal, 0);
    assert_eq!(info.compute_capability, (12, 1));
    assert_eq!(info.name, "NVIDIA GB10");
    assert_eq!(info.total_memory_bytes, 128 * 1024 * 1024 * 1024);
    assert_eq!(info.l2_bytes, 24 * 1024 * 1024);
    assert_eq!(info.persisting_l2_max_bytes, 18 * 1024 * 1024);
    assert_eq!(info.sm_count, 20);
    assert_eq!(info.warp_size, 32);
}

#[test]
fn abi_version_other_than_v1_is_rejected() {
    let mut raw = valid_raw();
    raw.abi_version = 2;
    assert!(matches!(
        CudaDeviceInfo::try_from(raw),
        Err(CudaDeviceInfoError::AbiVersion { found: 2 })
    ));
}

#[test]
fn compute_capability_other_than_12_1_is_rejected() {
    let mut raw = valid_raw();
    raw.compute_minor = 0;
    assert!(matches!(
        CudaDeviceInfo::try_from(raw),
        Err(CudaDeviceInfoError::ComputeCapability { major: 12, minor: 0 })
    ));
}

#[test]
fn zero_required_geometry_is_rejected() {
    for field in ["total_memory_bytes", "l2_bytes", "sm_count", "warp_size"] {
        let mut raw = valid_raw();
        match field {
            "total_memory_bytes" => raw.total_memory_bytes = 0,
            "l2_bytes" => raw.l2_bytes = 0,
            "sm_count" => raw.sm_count = 0,
            "warp_size" => raw.warp_size = 0,
            _ => unreachable!(),
        }
        assert!(
            CudaDeviceInfo::try_from(raw).is_err(),
            "{field} must be required"
        );
    }
}

#[test]
fn persisting_l2_cannot_exceed_discovered_l2() {
    let mut raw = valid_raw();
    raw.persisting_l2_max_bytes = raw.l2_bytes + 1;
    assert!(matches!(
        CudaDeviceInfo::try_from(raw),
        Err(CudaDeviceInfoError::PersistingL2ExceedsL2 { .. })
    ));
}

#[test]
fn negative_device_ordinal_is_rejected() {
    let mut raw = valid_raw();
    raw.device_ordinal = -1;
    assert!(matches!(
        CudaDeviceInfo::try_from(raw),
        Err(CudaDeviceInfoError::DeviceOrdinal(-1))
    ));
}

#[test]
fn unterminated_device_name_is_rejected() {
    let mut raw = valid_raw();
    raw.name.fill(b'X');
    assert!(matches!(
        CudaDeviceInfo::try_from(raw),
        Err(CudaDeviceInfoError::DeviceNameNotTerminated)
    ));
}

#[test]
fn invalid_utf8_device_name_is_rejected() {
    let mut raw = valid_raw();
    raw.name[0] = 0xff;
    raw.name[1] = 0;
    assert!(matches!(
        CudaDeviceInfo::try_from(raw),
        Err(CudaDeviceInfoError::DeviceNameUtf8)
    ));
}

#[test]
fn empty_device_name_is_rejected() {
    let mut raw = valid_raw();
    raw.name.fill(0);
    assert!(matches!(
        CudaDeviceInfo::try_from(raw),
        Err(CudaDeviceInfoError::DeviceNameEmpty)
    ));
}
