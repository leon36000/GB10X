# Telemetry validation and continuity — 2026-09-07

## Verified implementation

The local candidate now rejects misleading exact-mode profile labels and zero-valued integer
performance measurements. This follows the approved design's source-representation preservation
and exact/experimental separation; it closes the corresponding host-schema finding in the audit.

`ExecutionMode::Exact` requires `precision_mode: bf16` and `quantization_mode: none` for the initial
source profile. The profile label does not specify kernel accumulator precision. Other nonempty
labels remain usable under explicitly labelled `ExperimentalApproximate`; this is not automatic
approval of an experiment's numerical quality. Existing String fields and JSON shape are retained.

Present TTFT, p50, p95 and unified-memory measurements must be positive. Unavailable/sub-resolution
measurements can be omitted. Optional stage timers may still be zero, and explained correctness
failures remain valid evidence records. No runtime instrumentation or numerical kernel is added.

## Regression evidence

Before the fix, `cargo test --locked -p gb10x-telemetry --test full_contract` exited 101:
six tests passed and three failed because FP8 precision, FP8 quantization and zero TTFT were accepted.
After the fix, the same command exits 0 with all nine tests passing.

The six added tests exercise 14 invalid exact-profile labels and four zero integer metric fields,
explicit experimental JSON roundtrips, positive integer boundaries and omitted/zero-stage cases.
The old validator could accept those misleading inputs even alongside otherwise valid metrics.

## Fresh full candidate gates

Environment: Linux x86_64 in this session; official isolated Rust 1.98.0 (`88d9e12ae`). The old
temporary toolchain was absent. A new rustup installer was downloaded and checked against its
official SHA-256, `dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71`.
The new target directory `/tmp/gb10x-validation-20260907/target` started empty for this continuation.

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Exit 0 |
| `cargo test --locked --workspace --all-targets` | Exit 0; 111 passed, 0 failed, 0 ignored |
| `cargo clippy --locked --workspace --all-targets -- -D warnings` | Exit 0 |
| `cargo build --locked --workspace --release` | Exit 0 |
| `git diff --check` | Exit 0 |

The root controller inspected the diff and ran these gates. No separate independent review of
this small telemetry increment is claimed. Prior recovery and PLE reviews remain as documented.

## Durable resume design

The user explicitly requested domain Markdown and read/update of MCP TO PC, Neon and MongoDB Atlas
on each relaunch. Eight concise domain notes now start at `docs/memory/00-resume.md`; `AGENTS.md`
directs future sessions to read that index and only relevant domains, verify live state, then update
and read back connected memory. The notes contain state, decisions, proof references and next work,
not raw conversation or tool dumps.

The existing dedicated GB10X stores were recovered instead of creating duplicate cloud projects:

- Neon `steep-paper-03426819`, branch `br-polished-flower-awene32m`, database `neondb` was empty.
  The additive schema in `docs/memory/neon-schema.sql` creates a versioned singleton state table.
- Mongo project `6a9508bee9631f2b9e685733`, cluster `gb10x-memory`, database `gb10x_memory` already
  contains 19 historical checkpoints and three design decisions. Those records are preserved.
  A distinct `gb10x:current` document is reserved for the consolidated current checkpoint.
- MCP TO PC search initially returned unrelated semantic hits and no GB10X project registration.
  Use the reserved `GB10X-CANONICAL-MEMORY` marker with explicit repository/project metadata.

Full domain Markdown is mirrored in Mongo, structured state/digests in Neon and a short pointer in
MCP TO PC. Stable keys and prior-revision guards prevent blind overwrites. The actual write/readback
receipt is `docs/memory/sync-receipt.md`; read it before claiming all services are synchronized.

## Live boundaries and next work

Git refs rechecked on 2026-09-07: main `e3ef3384`, M3 `0b0bcf8`, native-M2 `27b6aee`, unchanged.
MCP TO PC now reports `gx10-e526` offline/unreachable; this supersedes the earlier reachable-host
observation without implying why it is unreachable. No GPU validation has occurred.

Physical pinned PLE hash-buffer identity, full source provenance, coherent native ABI/build,
GB10 execution, logits, inference throughput and actual instrumentation remain open. The next
independent host improvement is bounded overlay admission. Current changes remain local and
uncommitted; connected memory writes do not constitute a Git push or merge.
