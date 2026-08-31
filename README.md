# GB10X

GB10X is an experimental inference engine designed exclusively for NVIDIA GB10 / DGX Spark (`Linux aarch64`, `sm_121a`) and Qwen3.8-Flash-Next.

The project optimizes for correct end-to-end tokens/second on a dedicated GB10 server. It treats CPU caches, GPU caches, unified memory, Tensor/CUDA compute, speculative decoding, storage, and Linux memory policy as one coordinated inference system.

## Non-goals

- No x86 support.
- No non-GB10 NVIDIA GPU support.
- No AMD/Metal backend.
- No generic multi-model abstraction in the initial engine.
- No optimization is accepted without a correctness gate and an end-to-end A/B benchmark.

See `docs/superpowers/specs/2026-08-31-gb10x-cache-first-v3-design.md` for the approved architecture.
