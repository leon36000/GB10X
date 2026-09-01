# M2 Native Verification Boundary

## Current status

M2 has crossed its **host correctness** and **CUDA compile/link** gates, but it has **not yet crossed the real DGX Spark execution gate** in this repository evidence record.

GitHub's x86 runner has no GB10 device. The pinned CUDA build container proves that the native ABI/tooling compiles and links for `compute_121a -> sm_121a`; it cannot prove CUDA runtime behavior, Compute Capability 12.1 discovery, smoke execution, RMSNorm numerical correctness, or the bytes actually installed on the user's Spark.

## Implemented M2 surface

The branch contains:

- a pure-Rust CUDA toolchain contract with minimum CUDA 12.9 for `sm_121a`;
- a stable versioned C ABI that isolates CUDA/C++ layout from Rust;
- native CUDA device probing with fail-closed Rust validation;
- explicit `cuda_native: unavailable` JSON when native CUDA support is absent;
- native `gb10x-probe --json` binding that derives GPU facts only from the real CUDA probe;
- an `sm_121a` smoke path that writes device memory, rereads/reduces it on GPU, and returns a deterministic 64-bit checksum;
- an `sm_121a` BF16 RMSNorm kernel for Qwen width 2560 and epsilon `1e-6`, FP32 accumulation, BF16 output rounding;
- an mmap-backed exact safetensors PLE source with checked bounds, overlap/truncation rejection and SHA-256 provenance;
- a pinned Qwen3.8-Flash-Next PLE manifest contract for revision `34567a4712bc9766c4449e2e98e4468bfa24d915`, 128 parts, 320,001,536 rows and 160 BF16 elements per row;
- `gb10x-plepack source-verify` for validating and hashing the PLE bytes actually present in a local checkpoint directory;
- a fail-closed Spark handoff script with isolated build artifacts and machine-readable evidence.

## Proof classes

### 1. Host correctness — VERIFIED

Ordinary CI verifies logic that does not require CUDA hardware: ABI conversion/validation, toolchain parsing, PLE/safetensors parsing, pinned manifest construction, `source-verify` command behavior, unavailable-vs-native JSON semantics, verifier shell contracts, and the full Rust workspace gates.

### 2. CUDA `sm_121a` compile/link — VERIFIED

The CUDA CI job uses the pinned image:

`nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c`

It runs `nvcc --version`, compiles native CUDA targets with `--features native-cuda --no-run`, executes only the synthetic native JSON mapping test that does not call CUDA, and compiles all native tooling. No CPU/mock backend can satisfy these native link gates.

### 3. GB10 device + local model-source execution — PENDING

The following must execute on the actual DGX Spark before M2 can claim native runtime verification:

- Linux `aarch64` identity;
- local Qwen checkpoint directory supplied explicitly;
- successful construction of the pinned 128-part PLE manifest from the local `model.safetensors.index.json`;
- successful mmap validation/hash over the referenced local PLE safetensors bytes;
- exact model id and revision contract, 128 parts, 320,001,536 rows and 320 bytes per logical row;
- a valid 64-hex SHA-256 local PLE source digest;
- `remote_digest_match` remaining null unless an independent expected remote digest is added later;
- real CUDA device discovery and GB10 identity;
- Compute Capability 12.1;
- discovered nonzero memory/L2/SM/warp properties;
- `validate_gb10` success from real device facts;
- native smoke checksum equality;
- BF16 RMSNorm output within the approved 1-BF16-ULP gate for all deterministic vectors;
- `cuobjdump` inspection of the generated native CUDA objects from the same isolated build.

Run from the exact source commit on the Spark:

```bash
bash scripts/verify-gb10-m2.sh --model-dir /path/to/Qwen3.8-Flash-Next
```

or:

```bash
GB10X_QWEN_MODEL_DIR=/path/to/Qwen3.8-Flash-Next bash scripts/verify-gb10-m2.sh
```

A successful run creates a timestamped directory under `docs/evidence/native-runs/` by default. It contains raw logs plus `model-source.json`, `probe.json`, `summary.json` and `summary.md`. `summary.json` uses proof class `gb10-device-execution` and is written only after all local-model and device gates pass.

## Qwen source boundary

The local-source verifier is deliberately precise about what it proves. It derives the manifest from the local checkpoint's own index, enforces the pinned Qwen model geometry/name contract, opens the referenced safetensors directly, and hashes the exact referenced PLE bytes. The output field `revision_contract` is the pinned GB10X source contract, not an independently observed remote commit identity. The output therefore keeps `remote_digest_match: null` instead of fabricating a remote-byte match.

This is stronger than merely trusting a directory name, while remaining honest about the absence of a pre-registered expected remote digest in M2.

## Artifact provenance

The Spark verifier builds in a fresh temporary `CARGO_TARGET_DIR` and requires exactly one `probe.o`, `smoke.o`, and `rmsnorm.o` from that isolated build before `cuobjdump`. Old local build products cannot satisfy the artifact-inspection gate.

## Performance boundary

M2 makes **no tokens/second, latency, bandwidth, cache-hit-rate, or speedup claim**. Smoke/RMSNorm execution is correctness evidence only. End-to-end performance A/B work belongs to a later milestone after native correctness has been proven on the Spark.

## M3 boundary

M3 has not been implemented by this evidence record. There is no claim here that full Qwen inference, production attention/MoE scheduling, speculative decoding, cache policy tuning, or end-to-end model performance is complete.
