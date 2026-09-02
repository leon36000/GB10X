# GB10X

GB10X is an experimental inference engine designed exclusively for NVIDIA GB10 / DGX Spark (`Linux aarch64`, target `sm_121a`) and Qwen3.8-Flash-Next.

The project optimizes for correct end-to-end tokens/second on a dedicated GB10 server. It treats CPU caches, GPU caches, unified memory, Tensor/CUDA compute, speculative decoding, storage, and Linux memory policy as one coordinated inference system.

## Status

Milestone 1 provides the host-independent correctness/storage foundation and is verified in CI on an x86_64 Ubuntu runner. This does **not** constitute GB10 or CUDA hardware validation.

Implemented and host-independently verified:
- fail-closed GB10 platform-contract logic;
- Linux CPU/cache topology probing;
- exact Qwen3.8-Flash-Next model contract;
- exact PLE hash state with speculative rollback/commit-prefix semantics;
- exact PLEPack hot-overlay planning, atomic writing and mmap reading;
- immutable mmap raw-row source with SHA-256 provenance;
- byte-for-byte verification of every stored hot-overlay row;
- strict evidence/telemetry record validation;
- `gb10x-probe --json`;
- `gb10x-plepack plan|build|verify`;
- workspace release build.

Still unverified:
- execution on a real DGX Spark / GB10;
- `aarch64` target behavior;
- CUDA Compute Capability 12.1;
- CUDA / `sm_121a` compilation or execution;
- production safetensors source binding;
- full Qwen inference or any performance claim.

## M3 added; native validation pending

M3 adds an opt-in CUDA `sm_121a` build contract, a stable C ABI v1, an
unlinked Rust layout mirror, and a GB10 device-probe smoke source. CUDA
compilation and execution remain unverified until the documented DGX Spark
commands succeed; this repository's existing CI remains host-independent and
Rust-only.

See [`cuda/README.md`](cuda/README.md) for the native prerequisites, exact
configure/build/CTest commands, and fail-closed configuration rules.

M1 PLEPack uses a prepared flat exact-row source for executable verification. The sidecar duplicates only measured hot rows; cold rows remain in the immutable source. Direct model-safetensors integration belongs to the native/model-loading milestone.

See `docs/evidence/m1-bootstrap-verification.md` for the verification boundary and `docs/superpowers/specs/2026-08-31-gb10x-cache-first-v3-design.md` for the approved architecture.

## First DGX Spark gate

Before native CUDA work is considered verified, run on the actual server:

```bash
uname -m
./target/release/gb10x-probe --json
nvidia-smi
nvcc --version
```

The native milestone remains fail-closed until the server evidence confirms Linux `aarch64`, the GB10 target, CUDA Compute Capability 12.1 through a native CUDA-capable probe, and a usable CUDA toolkit.

## Non-goals

- No x86 production support.
- No non-GB10 NVIDIA GPU support.
- No AMD/Metal backend.
- No generic multi-model abstraction in the initial engine.
- No optimization is accepted without a correctness gate and an end-to-end A/B benchmark.
