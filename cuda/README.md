# GB10X native CUDA ABI slice

This directory is an opt-in native build for the NVIDIA GB10 / DGX Spark
target only: Linux `aarch64` with CUDA code generation fixed to `sm_121a`.
It builds the stable C ABI probe library (`gb10x_cuda`) and, when enabled, one
device-probe smoke executable. It does not build or run an inference engine.

## Prerequisites

- Linux on the native `aarch64`/`arm64` host (not cross-compilation);
- CMake 3.24 or newer;
- NVIDIA `nvcc` with both `sm_121a` in `nvcc --list-gpu-code` and
  `compute_121a` in `nvcc --list-gpu-arch`;
- a GB10 device reporting compute capability 12.1 when running the smoke test.

## Native validation

Run these commands on the DGX Spark / GB10 host:

```bash
cmake -S cuda -B build/cuda-sm121a \
  -DGB10X_CUDA_ENABLE=ON \
  -DGB10X_CUDA_NATIVE_TESTS=ON
cmake --build build/cuda-sm121a --parallel
ctest --test-dir build/cuda-sm121a --output-on-failure
```

The build owns architecture selection and emits exactly this code-generation
pair for every CUDA target:

```text
--generate-code=arch=compute_121a,code=sm_121a
```

`ctest` runs `gb10x_cuda_abi_smoke`. It validates ABI v1 metadata, the fixed
`sm_121a` library target, device compute capability 12.1, and a GB10-identifying
device name. It prints CUDA runtime, memory, and cache facts. It does not test
model loading, inference correctness, throughput, or latency.

## Fail-closed behavior

`GB10X_CUDA_ENABLE=ON` rejects non-Linux hosts.

`GB10X_CUDA_ENABLE=ON` rejects non-`aarch64`/`arm64` hosts and
cross-compilation.

`GB10X_CUDA_ENABLE=ON` rejects non-NVIDIA CUDA compilers.

`GB10X_CUDA_ENABLE=ON` rejects toolchains missing `sm_121a` or `compute_121a`.

`CMAKE_CUDA_ARCHITECTURES` may be empty or `OFF` only.

`CMAKE_CUDA_FLAGS`, `CMAKE_CUDA_FLAGS_INIT`, configuration-specific
`CMAKE_CUDA_FLAGS_*` variants, and the `CUDAFLAGS` environment variable may
not choose a CUDA architecture.

`GB10X_CUDA_NATIVE_TESTS=ON` requires `GB10X_CUDA_ENABLE=ON`.

With `GB10X_CUDA_ENABLE=OFF` (the default), CMake exits before discovering a
CUDA toolkit or creating CUDA targets.
