# GB10X Evidence Contract

Every performance record must identify the exact execution state. A record is incomplete unless it contains:

- GB10X git commit.
- Model repository/revision and content digest(s).
- Exact command and runtime configuration.
- Context length and prompt/output token counts.
- Prefix/PLE/KV cache state.
- Speculation mode and acceptance counters when enabled.
- CPU affinity/isolation policy.
- Precision/quantization mode and whether it is exact or experimental.
- Prefill tokens/s, decode tokens/s, TTFT, and serving latency statistics when applicable.
- Unified-memory footprint and relevant storage/cache counters.
- Correctness-gate result.
- Hardware/thermal/power state when available.

Performance results without a correctness result are invalid. Results from approximate modes must never be mixed with exact-mode claims.
