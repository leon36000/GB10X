# GB10X

GB10X is an experimental inference engine designed exclusively for NVIDIA GB10 / DGX Spark (`Linux aarch64`, target `sm_121a`) and Qwen3.8-Flash-Next.

The project optimizes for correct end-to-end tokens/second on a dedicated GB10 server. It treats CPU caches, GPU caches, unified memory, Tensor/CUDA compute, speculative decoding, storage, and Linux memory policy as one coordinated inference system.

## Status

Milestone 1 established the host-independent correctness/storage foundation. Milestone 2 now has host correctness and CUDA `sm_121a` compile/link evidence, but **real DGX Spark execution is still pending**. x86 CI and a CUDA compiler container do not constitute GB10 hardware validation.

Implemented and host-independently verified:
- fail-closed GB10 platform-contract logic;
- Linux CPU/cache topology probing;
- exact Qwen3.8-Flash-Next model contract;
- exact PLE hash state with speculative rollback/commit-prefix semantics;
- exact PLEPack hot-overlay planning, atomic writing and mmap reading;
- immutable mmap raw-row source with SHA-256 provenance;
- direct mmap-backed safetensors PLE row sourcing with bounds/overlap/truncation checks;
- pinned Qwen3.8-Flash-Next PLE manifest construction from `model.safetensors.index.json`;
- byte-for-byte verification of every stored hot-overlay row;
- strict evidence/telemetry record validation;
- `gb10x-probe --json` with an explicit native-CUDA evidence state;
- `gb10x-plepack plan|build|verify`;
- workspace release build.

Implemented and CUDA 12.9 compile/link verified for `compute_121a -> sm_121a`:
- versioned Rust/C CUDA ABI;
- native CUDA device probe;
- native `gb10x-probe --json` device-fact binding;
- deterministic device-memory smoke checksum path;
- BF16 RMSNorm for Qwen width 2560, epsilon `1e-6`, FP32 accumulation;
- native Rust test executables for probe, smoke and RMSNorm.

Still unverified on the actual appliance:
- execution on a real DGX Spark / GB10;
- real Linux `aarch64` runtime behavior;
- runtime discovery of CUDA Compute Capability 12.1;
- native probe execution and GB10 validation from real device facts;
- smoke checksum correctness on GB10;
- BF16 RMSNorm numerical correctness on GB10 against the approved 1-ULP gate;
- binding the pinned Qwen checkpoint bytes on the appliance;
- full Qwen inference or any performance claim.

No current CI result should be read as a tokens/second or latency claim. M2 native kernel work is correctness infrastructure only.

See `docs/evidence/m2-native-verification.md` for the current proof boundary and `docs/superpowers/specs/2026-08-31-gb10x-cache-first-v3-design.md` for the approved architecture.

## DGX Spark M2 gate

Run the fail-closed verifier from the exact source commit on the actual server:

```bash
bash scripts/verify-gb10-m2.sh
```

The script rejects non-Linux/non-`aarch64` hosts before doing native work. On the Spark it records `uname`, `nvcc`, `nvidia-smi`, the native JSON probe, probe/smoke/RMSNorm tests, and CUDA artifact inspection. It only emits a `gb10-device-execution` PASS summary after all device gates pass.

## Non-goals

- No x86 production support.
- No non-GB10 NVIDIA GPU support.
- No AMD/Metal backend.
- No generic multi-model abstraction in the initial engine.
- No optimization is accepted without a correctness gate and an end-to-end A/B benchmark.
