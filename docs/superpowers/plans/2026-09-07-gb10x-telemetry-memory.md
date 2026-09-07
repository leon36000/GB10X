# Telemetry evidence and project memory — 2026-09-07

## Authorized outcome

Continue the local GB10X candidate and implement the user's per-domain Markdown memory with
read/update/readback through the existing MCP TO PC, Neon and MongoDB Atlas stores. Keep other
projects and remote Git branches unchanged. Freshly validate the next bounded host correction.

## Telemetry contract

The current pinned source profile is BF16 without added quantization. An Exact record must use
the canonical labels `precision_mode: bf16` and `quantization_mode: none`. This is a declared
profile label, not a ban on FP32 accumulators inside BF16 kernels. Other numerical/profile labels
remain available only under explicitly labelled ExperimentalApproximate mode until independently
supported by a future exact contract. Preserve String fields and their serialized format.

Present TTFT, p50, p95 and unified-memory measurements must be greater than zero. Missing values
remain allowed; absent/sub-resolution measurements should be omitted. Optional stage timings can
be zero. Preserve representable failed-correctness records, percentile ordering and existing
finite-positive float checks. This is schema consistency, not operational instrumentation.

## Work

- [x] Read current repository refs and recover dedicated connected-memory destinations.
- [x] Create concise domain notes and a root agent entry point.
- [x] Demonstrate precision/quantization and zero-integer metric gaps with public-API regressions.
- [x] Add the smallest validation fix; run targeted and complete host gates.
- [x] Update the memory checkpoint, synchronize all three backends and verify exact readback.

Result: 111 host tests and all quality gates pass. Connected-memory revision 2 was read back
successfully in MongoDB Atlas, Neon and MCP TO PC. The latter requires query `GB10X` plus local
marker/revision filtering; that observed retrieval behavior is now in the domain memory.

Evidence: existing audit's telemetry finding plus the approved design's exact/experimental
separation. No new engine subsystem, GPU workload, external model call or runtime dependency.
