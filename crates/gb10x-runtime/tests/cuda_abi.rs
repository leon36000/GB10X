use gb10x_runtime::cuda_abi::{
    CudaAbiInfo, CudaDeviceInfo, CUDA_ABI_VERSION, CUDA_STATUS_ABI_MISMATCH,
    CUDA_STATUS_CUDA_RUNTIME_FAILURE, CUDA_STATUS_INTERNAL_FAILURE,
    CUDA_STATUS_INVALID_ARGUMENT, CUDA_STATUS_OK, CUDA_TARGET_SM_VARIANT_A,
};
use std::mem::{align_of, offset_of, size_of};

#[test]
fn abi_info_has_the_fixed_v1_layout() {
    assert_eq!(size_of::<CudaAbiInfo>(), 32);
    assert_eq!(align_of::<CudaAbiInfo>(), 4);
    assert_eq!(offset_of!(CudaAbiInfo, struct_size), 0);
    assert_eq!(offset_of!(CudaAbiInfo, abi_version), 4);
    assert_eq!(offset_of!(CudaAbiInfo, target_sm_major), 8);
    assert_eq!(offset_of!(CudaAbiInfo, cuda_runtime_loaded_version), 24);
}

#[test]
fn device_info_has_the_fixed_v1_layout() {
    assert_eq!(size_of::<CudaDeviceInfo>(), 296);
    assert_eq!(align_of::<CudaDeviceInfo>(), 8);
    assert_eq!(offset_of!(CudaDeviceInfo, struct_size), 0);
    assert_eq!(offset_of!(CudaDeviceInfo, total_global_memory_bytes), 16);
    assert_eq!(offset_of!(CudaDeviceInfo, l2_cache_bytes), 24);
    assert_eq!(offset_of!(CudaDeviceInfo, persisting_l2_max_bytes), 32);
    assert_eq!(offset_of!(CudaDeviceInfo, name), 40);
}

#[test]
fn constructors_encode_the_v1_output_contract() {
    let abi = CudaAbiInfo::new();
    assert_eq!(abi.struct_size as usize, size_of::<CudaAbiInfo>());
    assert_eq!(abi.abi_version, 0);
    assert_eq!(abi.target_sm_major, 0);
    assert_eq!(abi.reserved0, 0);

    let device = CudaDeviceInfo::new();
    assert_eq!(device.struct_size as usize, size_of::<CudaDeviceInfo>());
    assert_eq!(device.device_ordinal, 0);
    assert_eq!(device.name, [0; 256]);
}

#[test]
fn v1_numeric_constants_are_stable() {
    assert_eq!(CUDA_ABI_VERSION, 1);
    assert_eq!(CUDA_STATUS_OK, 0);
    assert_eq!(CUDA_STATUS_ABI_MISMATCH, 1);
    assert_eq!(CUDA_STATUS_INVALID_ARGUMENT, 2);
    assert_eq!(CUDA_STATUS_CUDA_RUNTIME_FAILURE, 3);
    assert_eq!(CUDA_STATUS_INTERNAL_FAILURE, 4);
    assert_eq!(CUDA_TARGET_SM_VARIANT_A, 1);
}
