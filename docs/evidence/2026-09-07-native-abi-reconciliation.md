# Native CUDA ABI reconciliation

Date: 2026-09-07
Scope: M3 CUDA ABI/probe only. This record does not add a kernel, model
loading, inference, or a Rust-to-CUDA link path.

## Decision

M3 is the only canonical public native ABI for GB10X. Its authority is:

- header: **cuda/common/include/gb10x_cuda_abi.h**;
- native build: **cuda/CMakeLists.txt**;
- public probe: **gb10x_cuda_probe_device(uint32_t,
  gb10x_cuda_device_info *)**;
- output and status contract: ABI v1, including caller-initialized
  **struct_size** fields and status values **0..4**.

**origin/feature/gb10x-native-m2** remains historical work. It must not be
merged, cherry-picked, or linked with M3 as-is. Its CUDA smoke and RMSNorm
sources are deferred to a separately approved kernel milestone after M3 native
validation.

## Measured collision

| Contract | M3 | native-M2 |
| --- | --- | --- |
| Reference | 0b0bcf8febcccf5a2b97574ac5ca674754855b3c | 27b6aee194f3028341f7b23ebae80548a8cb4e5f |
| Header | cuda/common/include/gb10x_cuda_abi.h | crates/gb10x-cuda/native/gb10x_cuda.h |
| Probe symbol | gb10x_cuda_probe_device | gb10x_cuda_probe_device |
| Return type | gb10x_cuda_status / uint32_t | int, including negative error codes |
| Ordinal argument | uint32_t | int |
| First output field | caller-supplied struct_size | callee-written abi_version |
| Output type | gb10x_cuda_device_info | gb10x_cuda_device_info_v1 |
| Output size / alignment | 296 bytes / 8 | 304 bytes / 8 |
| Device-name offset | 40 | 48 |
| Build authority | opt-in CMake gb10x_cuda | feature-gated Cargo build.rs |

The native-M2 probe starts by zeroing **sizeof(*out)**, which is 304 bytes.
An M3 caller supplies a 296-byte output object. Selecting the native-M2
definition for the shared C symbol would therefore write eight bytes past the
M3 object before either implementation can report an error. Native-M2 does not
consume M3's **struct_size**, so M3's fail-closed size check cannot protect
against the wrong callee.

This is a memory-safety and semantic ABI conflict, not a merge conflict that
can be resolved by choosing lines from both branches.

## Required boundary

1. Keep **cuda/common/include/gb10x_cuda_abi.h** as the sole public header for
   the canonical symbol names.
2. Keep M3's CMake project as the sole build authority for the canonical
   **gb10x_cuda** library.
3. Do not add **crates/gb10x-cuda**, its **build.rs**, its legacy header, or
   its legacy **gb10x_cuda_probe_device** definition to the M3 branch.
4. A later kernel milestone may port native-M2 source-level algorithms only
   after it defines a new approved ABI extension. It must consume the M3 v1
   header or introduce versioned entry points/types; it may not change M3 v1
   field order, sizes, meanings, or symbols.
5. Actual M3 CMake build and device smoke validation remain required on the
   GB10/DGX Spark. This host-side audit does not substitute for that gate.

## Reproduced checks

| Check | Result |
| --- | --- |
| C11 M3 header probe: fixed sizes, alignments, and offsets | passed |
| C11 native-M2 header probe: fixed sizes, alignments, and offsets | passed |
| M3 source scan for legacy native-M2 ABI names and crate path | passed; none present |
| cargo test -p gb10x-runtime --test cuda_abi -- --nocapture | passed: 5 tests |
| Disabled CMake configure/build | unavailable: cmake is absent on this host |

The two independent C11 probes establish the layout mismatch from the headers
themselves. The Rust tests establish M3's exported layout and constructors;
they do not link either CUDA implementation.
