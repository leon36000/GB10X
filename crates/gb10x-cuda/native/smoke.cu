#include "gb10x_cuda.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

static_assert(sizeof(unsigned long long) == sizeof(uint64_t),
              "GB10X smoke checksum requires a 64-bit unsigned long long");

namespace {

constexpr int kInvalidArgument = -1;
constexpr int kSizeOverflow = -2;
constexpr int kCudaErrorBase = -1000;
constexpr unsigned int kThreads = 256;
constexpr uint64_t kMaxBlocks = 4096;
constexpr uint64_t kMultiplier = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kOffset = 0xD1B54A32D192ED03ULL;
constexpr uint64_t kXorMask = 0x94D049BB133111EBULL;

int encode_cuda_error(cudaError_t error) {
    return kCudaErrorBase - static_cast<int>(error);
}

__device__ __forceinline__ uint64_t smoke_value(uint64_t index) {
    uint64_t value = index * kMultiplier;
    value += kOffset;
    value = (value << 17) | (value >> (64 - 17));
    return value ^ kXorMask;
}

__global__ void fill_smoke_values(unsigned long long* values, uint64_t elements) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = first; index < elements; index += stride) {
        values[index] = static_cast<unsigned long long>(smoke_value(index));
    }
}

__global__ void reduce_smoke_values(const unsigned long long* values,
                                    uint64_t elements,
                                    unsigned long long* checksum) {
    __shared__ unsigned long long block_sums[kThreads];

    const unsigned int lane = threadIdx.x;
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + lane;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (uint64_t index = first; index < elements; index += stride) {
        local += values[index];
    }
    block_sums[lane] = local;
    __syncthreads();

    for (unsigned int width = kThreads / 2; width > 0; width >>= 1) {
        if (lane < width) {
            block_sums[lane] += block_sums[lane + width];
        }
        __syncthreads();
    }

    if (lane == 0) {
        atomicAdd(checksum, block_sums[0]);
    }
}

void free_if_present(void* pointer) {
    if (pointer != nullptr) {
        static_cast<void>(cudaFree(pointer));
    }
}

}  // namespace

extern "C" int gb10x_cuda_smoke_v1(uint64_t elements, uint64_t* checksum) {
    if (checksum == nullptr || elements == 0) {
        return kInvalidArgument;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(unsigned long long)) {
        return kSizeOverflow;
    }

    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(unsigned long long);
    unsigned long long* values = nullptr;
    unsigned long long* device_checksum = nullptr;

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&values), bytes);
    if (status != cudaSuccess) {
        return encode_cuda_error(status);
    }
    status = cudaMalloc(reinterpret_cast<void**>(&device_checksum), sizeof(*device_checksum));
    if (status != cudaSuccess) {
        free_if_present(values);
        return encode_cuda_error(status);
    }
    status = cudaMemset(device_checksum, 0, sizeof(*device_checksum));
    if (status != cudaSuccess) {
        free_if_present(device_checksum);
        free_if_present(values);
        return encode_cuda_error(status);
    }

    const uint64_t required_blocks = elements / kThreads + (elements % kThreads != 0 ? 1 : 0);
    const unsigned int blocks = static_cast<unsigned int>(
        required_blocks < kMaxBlocks ? required_blocks : kMaxBlocks);

    fill_smoke_values<<<blocks, kThreads>>>(values, elements);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        free_if_present(device_checksum);
        free_if_present(values);
        return encode_cuda_error(status);
    }

    reduce_smoke_values<<<blocks, kThreads>>>(values, elements, device_checksum);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        free_if_present(device_checksum);
        free_if_present(values);
        return encode_cuda_error(status);
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        free_if_present(device_checksum);
        free_if_present(values);
        return encode_cuda_error(status);
    }

    unsigned long long host_checksum = 0;
    status = cudaMemcpy(&host_checksum,
                        device_checksum,
                        sizeof(host_checksum),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        free_if_present(device_checksum);
        free_if_present(values);
        return encode_cuda_error(status);
    }

    const cudaError_t checksum_free_status = cudaFree(device_checksum);
    const cudaError_t values_free_status = cudaFree(values);
    if (checksum_free_status != cudaSuccess) {
        return encode_cuda_error(checksum_free_status);
    }
    if (values_free_status != cudaSuccess) {
        return encode_cuda_error(values_free_status);
    }

    *checksum = static_cast<uint64_t>(host_checksum);
    return 0;
}
