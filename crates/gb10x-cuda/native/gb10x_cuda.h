#ifndef GB10X_CUDA_H
#define GB10X_CUDA_H

#include <stdint.h>

#define GB10X_CUDA_DEVICE_INFO_ABI_V1 1u
#define GB10X_CUDA_DEVICE_NAME_BYTES 256u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gb10x_cuda_device_info_v1 {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint32_t compute_major;
    uint32_t compute_minor;
    uint64_t total_memory_bytes;
    uint64_t l2_bytes;
    uint64_t persisting_l2_max_bytes;
    uint32_t sm_count;
    uint32_t warp_size;
    uint8_t name[GB10X_CUDA_DEVICE_NAME_BYTES];
} gb10x_cuda_device_info_v1;

int gb10x_cuda_probe_device(int ordinal, gb10x_cuda_device_info_v1* out);
int gb10x_cuda_smoke_v1(uint64_t elements, uint64_t* checksum);

#ifdef __cplusplus
}
#endif

#endif
