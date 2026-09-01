#include "gb10x_cuda.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

static_assert(sizeof(gb10x_cuda_device_info_v1) == 304,
              "GB10X CUDA device-info ABI size drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, abi_version) == 0,
              "GB10X CUDA ABI abi_version offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, device_ordinal) == 4,
              "GB10X CUDA ABI device_ordinal offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, compute_major) == 8,
              "GB10X CUDA ABI compute_major offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, compute_minor) == 12,
              "GB10X CUDA ABI compute_minor offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, total_memory_bytes) == 16,
              "GB10X CUDA ABI total_memory_bytes offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, l2_bytes) == 24,
              "GB10X CUDA ABI l2_bytes offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, persisting_l2_max_bytes) == 32,
              "GB10X CUDA ABI persisting_l2_max_bytes offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, sm_count) == 40,
              "GB10X CUDA ABI sm_count offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, warp_size) == 44,
              "GB10X CUDA ABI warp_size offset drift");
static_assert(offsetof(gb10x_cuda_device_info_v1, name) == 48,
              "GB10X CUDA ABI name offset drift");
static_assert(sizeof(std::size_t) <= sizeof(uint64_t),
              "GB10X requires size_t to fit the stable u64 ABI");

namespace {

constexpr int kInvalidArgument = -1;
constexpr int kInvalidProperty = -2;
constexpr int kCudaErrorBase = -1000;

int encode_cuda_error(cudaError_t error) {
    return kCudaErrorBase - static_cast<int>(error);
}

}  // namespace

extern "C" int gb10x_cuda_probe_device(int ordinal,
                                        gb10x_cuda_device_info_v1* out) {
    if (out == nullptr || ordinal < 0) {
        return kInvalidArgument;
    }

    std::memset(out, 0, sizeof(*out));

    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, ordinal);
    if (status != cudaSuccess) {
        return encode_cuda_error(status);
    }

    int persisting_l2_max = 0;
    status = cudaDeviceGetAttribute(&persisting_l2_max,
                                    cudaDevAttrMaxPersistingL2CacheSize,
                                    ordinal);
    if (status != cudaSuccess) {
        return encode_cuda_error(status);
    }

    if (properties.major < 0 || properties.minor < 0 ||
        properties.l2CacheSize < 0 || properties.multiProcessorCount < 0 ||
        properties.warpSize < 0 || persisting_l2_max < 0) {
        return kInvalidProperty;
    }

    std::size_t name_length = 0;
    while (name_length < sizeof(properties.name) &&
           properties.name[name_length] != '\0') {
        ++name_length;
    }
    if (name_length == sizeof(properties.name) ||
        name_length >= GB10X_CUDA_DEVICE_NAME_BYTES) {
        return kInvalidProperty;
    }

    out->abi_version = GB10X_CUDA_DEVICE_INFO_ABI_V1;
    out->device_ordinal = ordinal;
    out->compute_major = static_cast<uint32_t>(properties.major);
    out->compute_minor = static_cast<uint32_t>(properties.minor);
    out->total_memory_bytes = static_cast<uint64_t>(properties.totalGlobalMem);
    out->l2_bytes = static_cast<uint64_t>(properties.l2CacheSize);
    out->persisting_l2_max_bytes = static_cast<uint64_t>(persisting_l2_max);
    out->sm_count = static_cast<uint32_t>(properties.multiProcessorCount);
    out->warp_size = static_cast<uint32_t>(properties.warpSize);
    std::memcpy(out->name, properties.name, name_length);
    out->name[name_length] = 0;

    return 0;
}
