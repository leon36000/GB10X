# M2 Native Verification Boundary

## Current status

M2 has crossed its **host correctness** and **CUDA compile/link** gates, but it has **not yet crossed the real DGX Spark execution gate** in this repository evidence record.

The strongest verified code proof before the device-evidence script was added is commit `4b93f523c562029b3325c92b28ad4d1b37adcc4a`, GitHub Actions run `33455657694`:

- Rust host logic: `cargo fmt`, workspace tests, clippy with warnings denied, and workspace release build all passed.
- CUDA compile/link: the native crate and native tooling compiled and linked in the pinned CUDA 12.9.1 devel container.
- `nvcc` in that proof reported CUDA 12.9 (`V12.9.86`).
- Native objects were compiled with the GB10 architecture-specific build contract `compute_121a -> sm_121a`.
- The native C ABI/device probe, deterministic smoke path, and BF16 RMSNorm path all produced linkable Rust test executables.

This is **compile evidence**, not GPU execution evidence. GitHub's x86 runner has no GB10 device, so it cannot prove CUDA runtime behavior, Compute Capability 12.1 discovery, the smoke checksum, or the RMSNorm numerical gate.

## Implemented M2 surface

The branch currently contains:

- a pure-Rust CUDA toolchain contract with minimum CUDA 12.9 for `sm_121a`;
- a stable versioned C ABI that isolates CUDA/C++ layout from Rust;
- native CUDA device probing with fail-closed Rust validation;
- explicit `cuda_native: unavailable` JSON when the binary lacks native CUDA support;
- native `gb10x-probe --json` binding that derives GPU facts only from the real CUDA probe;
- an `sm_121a` smoke path that writes device memory, rereads/reduces it on GPU, and returns a deterministic 64-bit checksum;
- an `sm_121a` BF16 RMSNorm kernel for Qwen width 2560 and epsilon `1e-6`, FP32 accumulation, BF16 output rounding;
- an mmap-backed exact safetensors PLE source with checked bounds, overlap/truncation rejection and SHA-256 provenance;
- a pinned Qwen3.8-Flash-Next PLE manifest contract for revision `34567a4712bc9766c4449e2e98e4468bfa24d915`, 128 parts, 320,001,536 rows and 160 BF16 elements per row.

## Proof classes

### 1. Host correctness — VERIFIED

Ordinary CI verifies all logic that does not require CUDA hardware. This includes ABI conversion/validation, toolchain parsing, PLE/safetensors parsing, pinned manifest construction, unavailable-vs-native JSON semantics, and the full Rust workspace gates.

### 2. CUDA `sm_121a` compile/link — VERIFIED

The CUDA CI job uses the pinned image:

`nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c`

It runs `nvcc --version`, compiles native CUDA targets with `--features native-cuda --no-run`, executes only the synthetic native JSON mapping test that does not call CUDA, and compiles all native tooling. No CPU/mock backend can satisfy those native link gates.

### 3. GB10 device execution — PENDING

The following must execute on the actual DGX Spark before M2 can claim native runtime verification:

- Linux `aarch64` identity;
- real CUDA device discovery and GB10 identity;
- Compute Capability 12.1;
- discovered nonzero memory/L2/SM/warp properties;
- `validate_gb10` success from real device facts;
- native smoke checksum equality;
- BF16 RMSNorm output within the approved 1-BF16-ULP gate for all deterministic vectors;
- `cuobjdump` inspection of the generated native CUDA objects.

Run from the exact source commit on the Spark:

```bash
bash scripts/verify-gb10-m2.sh
```

A successful run creates a timestamped directory under `docs/evidence/native-runs/` by default. It contains raw command logs plus `summary.json` and `summary.md`. The generated `summary.json` uses proof class `gb10-device-execution` and is only written after every device gate passes.

## Qwen source boundary

The safetensors parser and pinned PLE manifest rules are host-verified. The manifest is constructed from the checkpoint's own `model.safetensors.index.json` rather than from a hard-coded file table. Binding the exact local checkpoint bytes on the appliance remains an execution/deployment action; a nearby or mismatched revision must fail closed.

## Performance boundary

M2 makes **no tokens/second, latency, bandwidth, cache-hit-rate, or speedup claim**. Smoke/RMSNorm execution is correctness evidence only. End-to-end performance A/B work belongs to a later milestone after native correctness has been proven on the Spark.

## M3 boundary

M3 has not been implemented by this evidence record. In particular, there is no claim here that full Qwen inference, production attention/MoE scheduling, speculative decoding, cache policy tuning, or end-to-end model performance is complete.
