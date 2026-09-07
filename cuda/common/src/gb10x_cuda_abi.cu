#include "gb10x_cuda_abi.h"

#include <cuda_runtime_api.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(sizeof(gb10x_cuda_abi_info) == 32);
static_assert(alignof(gb10x_cuda_abi_info) == 4);
static_assert(sizeof(gb10x_cuda_device_info) == 296);
static_assert(alignof(gb10x_cuda_device_info) == 8);
static_assert(std::is_standard_layout_v<gb10x_cuda_abi_info>);
static_assert(std::is_standard_layout_v<gb10x_cuda_device_info>);

namespace {

constexpr std::uint32_t kTargetSmMajor = 12;
constexpr std::uint32_t kTargetSmMinor = 1;

gb10x_cuda_status cuda_status(cudaError_t status) {
    return status == cudaSuccess ? GB10X_CUDA_STATUS_OK
                                 : GB10X_CUDA_STATUS_CUDA_RUNTIME_FAILURE;
}

bool abi_info_output_is_valid(const gb10x_cuda_abi_info *out_info) {
    return out_info != nullptr && out_info->struct_size == sizeof(*out_info);
}

bool device_info_output_is_valid(const gb10x_cuda_device_info *out_info) {
    return out_info != nullptr && out_info->struct_size == sizeof(*out_info);
}

} // namespace

extern "C" uint32_t gb10x_cuda_abi_version(void) {
    return GB10X_CUDA_ABI_VERSION;
}

extern "C" gb10x_cuda_status gb10x_cuda_get_abi_info(gb10x_cuda_abi_info *out_info) {
    if (out_info == nullptr) {
        return GB10X_CUDA_STATUS_INVALID_ARGUMENT;
    }
    if (!abi_info_output_is_valid(out_info)) {
        return GB10X_CUDA_STATUS_ABI_MISMATCH;
    }

    int runtime_version = 0;
    const cudaError_t runtime_status = cudaRuntimeGetVersion(&runtime_version);
    if (runtime_status != cudaSuccess) {
        return cuda_status(runtime_status);
    }
    if (runtime_version < 0) {
        return GB10X_CUDA_STATUS_INTERNAL_FAILURE;
    }

    gb10x_cuda_abi_info result{};
    result.struct_size = static_cast<std::uint32_t>(sizeof(result));
    result.abi_version = GB10X_CUDA_ABI_VERSION;
    result.target_sm_major = kTargetSmMajor;
    result.target_sm_minor = kTargetSmMinor;
    result.target_sm_variant = GB10X_CUDA_SM_VARIANT_A;
    result.cuda_runtime_header_version = static_cast<std::uint32_t>(CUDART_VERSION);
    result.cuda_runtime_loaded_version = static_cast<std::uint32_t>(runtime_version);
    *out_info = result;
    return GB10X_CUDA_STATUS_OK;
}

extern "C" gb10x_cuda_status gb10x_cuda_probe_device(
    uint32_t device_ordinal,
    gb10x_cuda_device_info *out_info) {
    if (out_info == nullptr) {
        return GB10X_CUDA_STATUS_INVALID_ARGUMENT;
    }
    if (!device_info_output_is_valid(out_info)) {
        return GB10X_CUDA_STATUS_ABI_MISMATCH;
    }
    if (device_ordinal > static_cast<std::uint32_t>(INT_MAX)) {
        return GB10X_CUDA_STATUS_INVALID_ARGUMENT;
    }

    const int ordinal = static_cast<int>(device_ordinal);
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, ordinal);
    if (status != cudaSuccess) {
        return cuda_status(status);
    }
    if (properties.major < 0 || properties.minor < 0) {
        return GB10X_CUDA_STATUS_INTERNAL_FAILURE;
    }

    int l2_cache_bytes = 0;
    status = cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, ordinal);
    if (status != cudaSuccess) {
        return cuda_status(status);
    }

    int persisting_l2_max_bytes = 0;
    status = cudaDeviceGetAttribute(
        &persisting_l2_max_bytes, cudaDevAttrPersistingL2CacheMaxSize, ordinal);
    if (status != cudaSuccess) {
        return cuda_status(status);
    }
    if (l2_cache_bytes < 0 || persisting_l2_max_bytes < 0) {
        return GB10X_CUDA_STATUS_INTERNAL_FAILURE;
    }

    gb10x_cuda_device_info result{};
    result.struct_size = static_cast<std::uint32_t>(sizeof(result));
    result.device_ordinal = device_ordinal;
    result.compute_major = static_cast<std::uint32_t>(properties.major);
    result.compute_minor = static_cast<std::uint32_t>(properties.minor);
    result.total_global_memory_bytes = static_cast<std::uint64_t>(properties.totalGlobalMem);
    result.l2_cache_bytes = static_cast<std::uint64_t>(l2_cache_bytes);
    result.persisting_l2_max_bytes = static_cast<std::uint64_t>(persisting_l2_max_bytes);

    std::size_t name_length = 0;
    while (name_length < sizeof(properties.name) && properties.name[name_length] != '\0') {
        ++name_length;
    }
    if (name_length >= sizeof(result.name)) {
        name_length = sizeof(result.name) - 1;
    }
    std::memcpy(result.name, properties.name, name_length);

    *out_info = result;
    return GB10X_CUDA_STATUS_OK;
}

extern "C" const char *gb10x_cuda_status_string(uint32_t status) {
    switch (status) {
    case GB10X_CUDA_STATUS_OK:
        return "ok";
    case GB10X_CUDA_STATUS_ABI_MISMATCH:
        return "ABI mismatch";
    case GB10X_CUDA_STATUS_INVALID_ARGUMENT:
        return "invalid argument";
    case GB10X_CUDA_STATUS_CUDA_RUNTIME_FAILURE:
        return "CUDA runtime failure";
    case GB10X_CUDA_STATUS_INTERNAL_FAILURE:
        return "internal failure";
    default:
        return "unknown status";
    }
}
