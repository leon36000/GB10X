#ifndef GB10X_CUDA_ABI_H
#define GB10X_CUDA_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GB10X_CUDA_ABI_VERSION 1u

typedef uint32_t gb10x_cuda_status;

#define GB10X_CUDA_STATUS_OK 0u
#define GB10X_CUDA_STATUS_ABI_MISMATCH 1u
#define GB10X_CUDA_STATUS_INVALID_ARGUMENT 2u
#define GB10X_CUDA_STATUS_CUDA_RUNTIME_FAILURE 3u
#define GB10X_CUDA_STATUS_INTERNAL_FAILURE 4u

#define GB10X_CUDA_SM_VARIANT_A 1u

typedef struct gb10x_cuda_abi_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t target_sm_major;
    uint32_t target_sm_minor;
    uint32_t target_sm_variant;
    uint32_t cuda_runtime_header_version;
    uint32_t cuda_runtime_loaded_version;
    uint32_t reserved0;
} gb10x_cuda_abi_info;

typedef struct gb10x_cuda_device_info {
    uint32_t struct_size;
    uint32_t device_ordinal;
    uint32_t compute_major;
    uint32_t compute_minor;
    uint64_t total_global_memory_bytes;
    uint64_t l2_cache_bytes;
    uint64_t persisting_l2_max_bytes;
    uint8_t name[256];
} gb10x_cuda_device_info;

uint32_t gb10x_cuda_abi_version(void);
gb10x_cuda_status gb10x_cuda_get_abi_info(gb10x_cuda_abi_info *out_info);
gb10x_cuda_status gb10x_cuda_probe_device(
    uint32_t device_ordinal,
    gb10x_cuda_device_info *out_info);
const char *gb10x_cuda_status_string(uint32_t status);

#ifdef __cplusplus
}
#endif

#endif
