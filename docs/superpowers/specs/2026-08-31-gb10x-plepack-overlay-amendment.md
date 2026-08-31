# GB10X PLEPack Overlay Amendment

Status: approved-by-project-lead refinement of Cache-First v3 during implementation.

## Reason

Qwen3.8-Flash-Next PLE has 16 hash-head tables whose vocabulary scale is roughly 20M rows/head. A global physical permutation would therefore require a logical-to-physical index on the order of hundreds of millions of entries; even a minimal 32-bit ordinal index would consume roughly gigabytes and add another random lookup to every cold PLE access.

## Revised exact layout

PLEPack uses two exact regions:

1. **Cold base region** — every BF16 row remains in logical-row order. Lookup is arithmetic: `base + logical_row * row_bytes`; no full-row index exists.
2. **Hot overlay** — only rows proven hot/co-accessed by measured traces are duplicated into locality-optimized blocks. A bounded overlay index maps those hot logical rows to overlay locations.

The overlay never changes row values. Missing overlay entries fall through to the exact base row.

## Advantages

- eliminates a GB-scale global remap index;
- keeps cold lookup branch-light and arithmetic;
- lets hot bundles be rebuilt from workload traces without rewriting the full PLE base;
- permits multiple overlay policies/budgets to be A/B tested against the same immutable base;
- makes rollback simple: an overlay can be discarded atomically;
- increases NVMe storage only by the bounded hot-overlay budget rather than by a full second copy.

## Gate

An overlay is promoted only when it reduces end-to-end PLE stall time / increases correct tokens/s. Trace overfitting must be tested on held-out coding, tool-use, JSON and general-text workloads.
