# PLEPack overlay admission budget — 2026-09-07

## Scope and contract

This host-only increment bounds exact PLEPack hot-overlay admission without changing PLEPack v1
bytes, source rows or cold-row behavior. `plan_exact_layout` remains unbounded for compatibility.
`plan_exact_layout_with_budget` accepts an `OverlayAdmissionBudget` with optional limits for:

- trace-distinct hot rows;
- block-padded duplicated-row data bytes.

Rows are still admitted in the existing deterministic first-seen trace-group order. A zero cap is a
valid cold-base-only plan. Every trace row is validated even after a cap is full, so a later
out-of-range ID cannot be hidden by admission truncation.

The byte cap covers only the hot data blocks, including block padding. The fixed sidecar header and
sorted index are outside that cap and remain separately reported as `file_bytes` and `index_bytes`.
This distinction is explicit in the Rust API, CLI help and README; it is not a claim of a complete
published-file quota.

`gb10x-plepack plan` and `build` now accept optional `--max-hot-rows` and
`--max-overlay-bytes`. Omitting both retains the previous unlimited behavior.

## Regression evidence

Focused tests first failed because the budget API and CLI flags did not exist. The byte-cap test
then required block-padding accounting, and the malformed-trace regression initially returned a
plan instead of rejecting a later out-of-range row. The final tests prove deterministic row
truncation, block-cap accounting, continued input validation, malformed-layout safety, helper
propagation and both CLI paths.

`LayoutPlan::hot_overlay_storage_bytes` now measures the furthest serialized placement rather than
assuming placements are contiguous. This keeps the writer's allocation calculation safe for a
deserialized malformed plan; malformed geometry returns a validation error rather than panicking.

## Fresh host gates

Rust 1.98.0 (`88d9e12ae`) on the current Linux x86_64 sandbox:

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Exit 0 |
| `cargo test --locked --workspace --all-targets` | Exit 0; 119 passed, 0 failed, 0 ignored |
| `cargo clippy --locked --workspace --all-targets -- -D warnings` | Exit 0 |
| `cargo build --locked --workspace --release` | Exit 0 in a fresh isolated target directory |
| `git diff --check` | Exit 0 |

The checkout's old default `target/release` cache contained a current `memmap2` metadata file but
an older rlib, causing a reproducible local release-link failure. A clean target rebuilt all crates
and passed; this is a transient build-cache inconsistency, not a source or lockfile failure.

## Limits

No measured production trace, overlay A/B result, real checkpoint read, GB10 execution, CUDA build,
full model inference or performance claim is added. The cap bounds planning and sidecar construction
only; promotion still requires held-out workload and end-to-end measurements.
