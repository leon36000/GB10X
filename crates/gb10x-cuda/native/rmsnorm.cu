#include "gb10x_cuda.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

static_assert(sizeof(__nv_bfloat16) == sizeof(uint16_t),
              "GB10X BF16 ABI requires 16-bit CUDA bfloat16 storage");
static_assert(alignof(__nv_bfloat16) == alignof(uint16_t),
              "GB10X BF16 ABI alignment drift");

namespace {

constexpr int kInvalidArgument = -1;
constexpr int kCudaErrorBase = -1000;
constexpr unsigned int kWidth = 2560;
constexpr unsigned int kThreads = 256;
constexpr float kEpsilon = 1.0e-6F;
constexpr std::size_t kRowBytes = kWidth * sizeof(uint16_t);

int encode_cuda_error(cudaError_t error) {
    return kCudaErrorBase - static_cast<int>(error);
}

__global__ void rmsnorm_bf16_one_row(const __nv_bfloat16* input,
                                     const __nv_bfloat16* weight,
                                     __nv_bfloat16* output) {
    __shared__ float sums[kThreads];

    const unsigned int lane = threadIdx.x;
    float local_sum = 0.0F;
    for (unsigned int index = lane; index < kWidth; index += blockDim.x) {
        const float value = __bfloat162float(input[index]);
        local_sum += value * value;
    }
    sums[lane] = local_sum;
    __syncthreads();

    for (unsigned int width = kThreads / 2; width > 0; width >>= 1) {
        if (lane < width) {
            sums[lane] += sums[lane + width];
        }
        __syncthreads();
    }

    const float inverse_rms = rsqrtf(sums[0] / static_cast<float>(kWidth) + kEpsilon);
    for (unsigned int index = lane; index < kWidth; index += blockDim.x) {
        const float value = __bfloat162float(input[index]);
        const float scale = __bfloat162float(weight[index]);
        output[index] = __float2bfloat16_rn(value * inverse_rms * scale);
    }
}

void free_if_present(void* pointer) {
    if (pointer != nullptr) {
        static_cast<void>(cudaFree(pointer));
    }
}

}  // namespace

extern "C" int gb10x_cuda_rmsnorm_bf16_device_v1(const uint16_t* input_device,
                                                   const uint16_t* weight_device,
                                                   uint16_t* output_device) {
    if (input_device == nullptr || weight_device == nullptr || output_device == nullptr) {
        return kInvalidArgument;
    }

    const auto* input = reinterpret_cast<const __nv_bfloat16*>(input_device);
    const auto* weight = reinterpret_cast<const __nv_bfloat16*>(weight_device);
    auto* output = reinterpret_cast<__nv_bfloat16*>(output_device);

    rmsnorm_bf16_one_row<<<1, kThreads>>>(input, weight, output);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return encode_cuda_error(status);
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return encode_cuda_error(status);
    }
    return 0;
}

extern "C" int gb10x_cuda_rmsnorm_bf16_host_test_v1(const uint16_t* input_host,
                                                      const uint16_t* weight_host,
                                                      uint16_t* output_host) {
    if (input_host == nullptr || weight_host == nullptr || output_host == nullptr) {
        return kInvalidArgument;
    }

    uint16_t* input_device = nullptr;
    uint16_t* weight_device = nullptr;
    uint16_t* output_device = nullptr;

    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&input_device), kRowBytes);
    if (status != cudaSuccess) {
        return encode_cuda_error(status);
    }
    status = cudaMalloc(reinterpret_cast<void**>(&weight_device), kRowBytes);
    if (status != cudaSuccess) {
        free_if_present(input_device);
        return encode_cuda_error(status);
    }
    status = cudaMalloc(reinterpret_cast<void**>(&output_device), kRowBytes);
    if (status != cudaSuccess) {
        free_if_present(weight_device);
        free_if_present(input_device);
        return encode_cuda_error(status);
    }

    status = cudaMemcpy(input_device, input_host, kRowBytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        free_if_present(output_device);
        free_if_present(weight_device);
        free_if_present(input_device);
        return encode_cuda_error(status);
    }
    status = cudaMemcpy(weight_device, weight_host, kRowBytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        free_if_present(output_device);
        free_if_present(weight_device);
        free_if_present(input_device);
        return encode_cuda_error(status);
    }

    const int kernel_status =
        gb10x_cuda_rmsnorm_bf16_device_v1(input_device, weight_device, output_device);
    if (kernel_status != 0) {
        free_if_present(output_device);
        free_if_present(weight_device);
        free_if_present(input_device);
        return kernel_status;
    }

    status = cudaMemcpy(output_host, output_device, kRowBytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        free_if_present(output_device);
        free_if_present(weight_device);
        free_if_present(input_device);
        return encode_cuda_error(status);
    }

    const cudaError_t output_free_status = cudaFree(output_device);
    const cudaError_t weight_free_status = cudaFree(weight_device);
    const cudaError_t input_free_status = cudaFree(input_device);
    if (output_free_status != cudaSuccess) {
        return encode_cuda_error(output_free_status);
    }
    if (weight_free_status != cudaSuccess) {
        return encode_cuda_error(weight_free_status);
    }
    if (input_free_status != cudaSuccess) {
        return encode_cuda_error(input_free_status);
    }
    return 0;
}
