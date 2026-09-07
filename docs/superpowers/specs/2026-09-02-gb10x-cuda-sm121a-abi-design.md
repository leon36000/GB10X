# GB10X CUDA `sm_121a` Build Surface and ABI — Design

## Status

The architectural approach was approved in chat. This document makes that
decision precise. No CUDA implementation starts until this written
specification has been reviewed and the resulting implementation plan has
been approved.

## Goal

Establish the smallest native CUDA boundary that GB10X can safely build and
test on its one supported deployment target:

- NVIDIA GB10 / DGX Spark
- Linux `aarch64`
- NVIDIA CUDA compiler support for exactly `sm_121a`

M3 delivers a fail-closed CUDA build contract, a versioned C ABI shared with
Rust, and one device-probe smoke test. It deliberately does **not** deliver a
model forward pass, a CUDA-backed inference path, or a performance claim.

The target choice is explicit because NVIDIA lists `sm_121a` and
`compute_121a` as supported Blackwell code-generation targets. The
configuration uses an explicit code-generation pair instead of the convenient
real-architecture shorthand: NVIDIA documents that the shorthand also embeds
its closest virtual architecture, whereas this milestone must produce only
the native `sm_121a` code image. See the [NVCC code-generation
documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).

## Scope and non-goals

### In scope

- An opt-in CMake CUDA project rooted at `cuda/`.
- Strict target, host, compiler, and CUDA-code-generation checks.
- A small static library named `gb10x_cuda`.
- A public, C-compatible v1 ABI for reporting build and device facts.
- A Rust `#[repr(C)]` mirror of that ABI and host-independent layout tests.
- One native CTest smoke executable that exercises only the ABI/device probe.
- Native validation instructions and an explicit evidence boundary.

### Out of scope

- GDN, QSA, MoE, MTP, KV-cache, quantization, or fused CUDA kernels.
- A model loader, tensor allocation API, or a Rust runtime path that invokes
  CUDA.
- CUDA packaging/link discovery for the Rust workspace.
- A GitHub-hosted CUDA/GB10 runner or a change to the existing Rust-only CI.
- Generic CUDA support, x86 support, cross-compilation, CPU fallbacks, PTX
  fallback images, or multi-architecture fatbins.
- Performance measurements or claims.

The later minimal forward-path milestone may consume this ABI, but must define
its own launch and ownership contract before adding any new entry point.

## Build contract

### Project and activation

`cuda/CMakeLists.txt` is a standalone CMake project with a minimum CMake
version of 3.24. It has these cache options:

| Option | Default | Meaning |
| --- | --- | --- |
| `GB10X_CUDA_ENABLE` | `OFF` | Enables CUDA language detection and native targets. |
| `GB10X_CUDA_NATIVE_TESTS` | `OFF` | Registers the GPU-executing ABI smoke test. It is meaningful only when CUDA is enabled. |

With `GB10X_CUDA_ENABLE=OFF`, CMake configures only the inert project shell;
it must not look for a CUDA toolkit or create CUDA targets. This lets the Rust
workspace remain host-independent. With `GB10X_CUDA_ENABLE=ON`, every check
below is mandatory and a failed check is a configuration error.

### Native-only checks

An enabled build accepts only all of the following:

1. `CMAKE_SYSTEM_NAME` is `Linux`.
2. `CMAKE_SYSTEM_PROCESSOR` is `aarch64` or `arm64`; cross-compilation is not
   supported in M3.
3. `CMAKE_CUDA_COMPILER_ID` is `NVIDIA`; Clang CUDA mode and other front ends
   are rejected until they have a separately validated contract.
4. `${CMAKE_CUDA_COMPILER} --list-gpu-code` reports an exact `sm_121a` token.
5. `${CMAKE_CUDA_COMPILER} --list-gpu-arch` reports an exact `compute_121a`
   token.

The checks use token-aware matching rather than accepting a prefix such as
`sm_121`. `nvcc --list-gpu-code` and `--list-gpu-arch` are the compiler's
supported-code and supported-virtual-architecture queries, respectively.

### Exact code generation

Every CUDA target must set the CMake `CUDA_ARCHITECTURES` property to `OFF`.
The CMake documentation defines that value as full suppression of CMake's
architecture flags, which gives GB10X sole control over the code-generation
arguments. The implementation then supplies exactly:

```text
--generate-code=arch=compute_121a,code=sm_121a
```

to both CUDA compilation and any CUDA device-link phase. This creates a real
`sm_121a` image and intentionally emits no PTX code image. CMake must reject a
nonempty `CMAKE_CUDA_ARCHITECTURES` value other than `OFF`, plus any
architecture-selection option supplied through `CMAKE_CUDA_FLAGS`, its
configuration-specific variants, `CMAKE_CUDA_FLAGS_INIT`, or the `CUDAFLAGS`
environment variable before CUDA language enablement.

`native`, `all`, `all-major`, `--gpu-architecture`, `--gpu-code`, and extra
`--generate-code` selections are prohibited in project configuration. This is
not a generic build: a compiler unable to produce the exact pair fails rather
than silently selecting a nearby target. CMake's [CUDA architecture
property](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
documents why `OFF` is appropriate when a project needs complete control of
the passed flags.

The build will initially create only these CUDA surfaces:

```text
cuda/
  CMakeLists.txt
  README.md
  common/
    include/gb10x_cuda_abi.h
    src/gb10x_cuda_abi.cu
  tests/
    abi_smoke.cu
```

Future kernel directories (`gdn/`, `qsa/`, `moe/`, `mtp/`, and `fused/`) are
not created as empty placeholders in this milestone.

## C ABI v1

The canonical ABI header is `cuda/common/include/gb10x_cuda_abi.h`. It is
valid C and C++: it uses `<stdint.h>`, fixed-width scalar types, POD structs,
and `extern "C"` only when compiled as C++. It contains no C++ standard-library
types, templates, exceptions, ownership transfers, callbacks, or CUDA types.

### Stable constants

```c
#define GB10X_CUDA_ABI_VERSION 1u

typedef uint32_t gb10x_cuda_status;

#define GB10X_CUDA_STATUS_OK                   0u
#define GB10X_CUDA_STATUS_ABI_MISMATCH         1u
#define GB10X_CUDA_STATUS_INVALID_ARGUMENT     2u
#define GB10X_CUDA_STATUS_CUDA_RUNTIME_FAILURE 3u
#define GB10X_CUDA_STATUS_INTERNAL_FAILURE     4u

#define GB10X_CUDA_SM_VARIANT_A 1u
```

`gb10x_cuda_status` is a `uint32_t` typedef, not a C++ enum. The numeric values
above are part of the ABI and never change within v1. Unknown numeric values
are treated as an error by callers and stringify as `"unknown status"`.

### ABI data structures

```c
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
```

For ABI v1, their required layouts are fixed:

| Type | Size | Alignment | Required contents on success |
| --- | ---: | ---: | --- |
| `gb10x_cuda_abi_info` | 32 bytes | 4 bytes | ABI version 1, target `12.1a`, CUDA header and loaded-runtime versions, and zeroed reserved field. |
| `gb10x_cuda_device_info` | 296 bytes | 8 bytes | Requested ordinal, device compute capability, memory/cache facts, and a nul-terminated device name. |

`struct_size` is the first field in each struct. Before calling an output
function, a caller sets it to the v1 size in the table. A null output pointer
returns `GB10X_CUDA_STATUS_INVALID_ARGUMENT`; a different size returns
`GB10X_CUDA_STATUS_ABI_MISMATCH`. On any non-success result, callers must not
consume output fields other than the value they supplied for `struct_size`.
All reserved bytes/fields are zero on a successful call.

### Entry points

```c
uint32_t gb10x_cuda_abi_version(void);
gb10x_cuda_status gb10x_cuda_get_abi_info(gb10x_cuda_abi_info *out_info);
gb10x_cuda_status gb10x_cuda_probe_device(
    uint32_t device_ordinal,
    gb10x_cuda_device_info *out_info);
const char *gb10x_cuda_status_string(uint32_t status);
```

`gb10x_cuda_abi_version` always returns `GB10X_CUDA_ABI_VERSION` for the
library being loaded. `gb10x_cuda_get_abi_info` reports the build target and
CUDA runtime versions; it does not load a model or launch a kernel.
`gb10x_cuda_probe_device` validates the ordinal, queries CUDA runtime device
properties, and reports facts without allocating model data. It returns
`CUDA_RUNTIME_FAILURE` for a CUDA API failure and `INVALID_ARGUMENT` when the
ordinal cannot be represented safely by the CUDA runtime API.
`gb10x_cuda_status_string` returns a non-null static, immutable ASCII string;
the caller never frees it.

The probe reports raw device facts. The smoke executable, not the general ABI
function, applies the M3 deployment assertion: the name contains `GB10` and
the compute capability is exactly 12.1. This keeps discovery useful for
diagnostics while ensuring that native validation fails on a non-GB10 device.

Changing a v1 constant, field order, field width, layout, or documented
meaning is ABI-breaking. Such a change requires a new ABI version and new
versioned types/functions while v1 remains callable.

## Rust mirror and linkage boundary

`crates/gb10x-runtime/src/cuda_abi.rs` will mirror the constants and structures
with `#[repr(C)]` and Rust fixed-width integer types. It will expose FFI
function declarations, but no default Rust target will invoke them or link
`gb10x_cuda` in M3. That keeps `cargo test --workspace --all-targets` runnable
on non-CUDA hosts. Each Rust output type provides a constructor that sets its
`struct_size` to the v1 value and zeroes every other field.

`crates/gb10x-runtime/tests/cuda_abi.rs` will independently assert the v1
constant values, type sizes, alignments, and constructor initialization
contract.
The CUDA implementation will contain matching C++ `static_assert`s for the
canonical header. This is a layout contract, not yet a Rust-to-CUDA execution
path; the first caller that links the library must be designed and approved as
a separate integration step.

## Native smoke test

When both CMake options are enabled, CTest registers a single executable named
`gb10x_cuda_abi_smoke`. It links `gb10x_cuda`, calls the ABI version and info
functions, probes ordinal zero, and fails unless:

1. every call returns `GB10X_CUDA_STATUS_OK`;
2. the ABI version is 1 and both reported `struct_size` values match v1;
3. the ABI target reports major 12, minor 1, and variant `A`;
4. the device reports compute capability 12.1; and
5. the nul-terminated CUDA device name contains `GB10`.

The executable may print the detected name, compute capability, CUDA runtime
versions, total memory, L2 size, and persisting-L2 limit for evidence. It does
not allocate model buffers, execute an inference kernel, or benchmark anything.

## CI and validation strategy

The existing GitHub Actions workflow remains Rust-only. M3 adds no CUDA job to
it, because its runners do not establish the required GB10/aarch64 contract.
The standard host gates continue to validate the Rust ABI mirror:

```text
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
```

Native DGX validation is intentionally separate:

```text
cmake -S cuda -B build/cuda-sm121a \
  -DGB10X_CUDA_ENABLE=ON \
  -DGB10X_CUDA_NATIVE_TESTS=ON
cmake --build build/cuda-sm121a --parallel
ctest --test-dir build/cuda-sm121a --output-on-failure
```

The implementation must document the exact compiler/toolkit output and CTest
result in a dated evidence file when this validation is run on native hardware.
The current development environment is `x86_64` and has neither `cmake`,
`nvcc`, nor an NVIDIA device utility available; therefore it can validate the
specification and the Rust-only gates but cannot claim native CUDA build or
runtime success.

## Acceptance criteria

M3 is complete only when all of the following are true:

1. CUDA activation is opt-in, and enabled configuration fails closed unless
   Linux/aarch64, NVIDIA `nvcc`, `sm_121a`, and `compute_121a` are present.
2. Every CUDA target suppresses CMake architecture defaults and uses only the
   explicit `compute_121a -> sm_121a` code-generation pair.
3. The static `gb10x_cuda` library defines exactly the v1 ABI above, with C++
   layout assertions.
4. The Rust mirror compiles and its host-independent tests assert the same v1
   sizes, alignments, and numeric constants.
5. The native smoke test builds and, on a supported GB10, verifies the ABI and
   device identity without exercising inference.
6. The existing Rust CI remains green without CUDA installed.
7. No forward pass, model-loading behavior, generic fallback, or performance
   behavior is introduced.

## Implementation boundary

The next step is a reviewed implementation plan that converts these contracts
into small, testable edits. It must preserve the scope above and keep native
validation explicitly pending until it is run on a GB10/DGX Spark machine.
