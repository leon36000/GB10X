# GB10X PLE Source Recovery Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to execute this bounded recovery. Steps use checkboxes for tracking.

**Goal:** Recover the existing host-only Safetensors PLE source from the unmerged native-M2 branch onto the current main foundation, without losing CPU Cache Fabric or mixing incompatible CUDA ABIs.

**Audit follow-up:** After storage review, close the five directly observed config-validation omissions against the existing pinned fixture. This is a host contract correction, not new architecture or a claim of upstream checkpoint provenance.

**Architecture:** Reuse the existing `ExactPleRowSource` seam and the existing native-M2 Task 5 implementation. Keep the source verifier independent of CUDA. The CUDA branches remain untouched and separately qualified.

**Tech Stack:** Rust 1.98.0 for this validation, existing serde/serde_json/memmap2/SHA-256 dependencies, deterministic miniature file fixtures.

**Spec:** `docs/superpowers/specs/2026-08-31-gb10x-cache-first-v3-design.md` and its PLEPack overlay amendment. Recovery source: `27b6aee194f3028341f7b23ebae80548a8cb4e5f`, native-M2 Task 5 in `docs/superpowers/plans/2026-08-31-gb10x-native-m2.md` at that revision.

## Global Constraints

- Production target: NVIDIA GB10 / DGX Spark only; Linux `aarch64`; CUDA `sm_121a`.
- This recovery is host-independent storage correctness, not native execution or inference.
- Keep the CPU Cache Fabric / PLE-Hydra behavior already merged at `e3ef3384a20a3af640f418fa56ca32c62f7fc190` intact.
- No CUDA files, `gb10x-cuda` crate, native probe wiring, CI workflow, hardware settings, service, or model files may be changed by this recovery.
- A locally observed digest is not proof of remote checkpoint identity. Preserve `remote_digest_match: null` and the label `revision_contract`.
- Recover existing tested code; do not reimplement it gratuitously or delete the source branch.
- Source checkpoint files must remain immutable throughout mmap lifetime, as required by the existing row-source contract.
- No automatic merge or publication in this recovery task. A local candidate and its evidence are the deliverable.

## File ownership

Recovered from the exact source revision:

- `crates/gb10x-ple/src/safetensors.rs`: mmap-backed BF16 source and checked tensor geometry.
- `crates/gb10x-ple/src/qwen38_manifest.rs`: pinned model index-to-PLE mapping.
- `crates/gb10x-ple/tests/safetensors_source.rs`: miniature file/row/provenance tests.
- `crates/gb10x-ple/tests/qwen38_manifest.rs`: pinned index mapping tests.
- `crates/gb10x-tools/src/bin/gb10x-plepack.rs`: existing `source-verify` command addition only.

Integration edits:

- `crates/gb10x-ple/src/lib.rs`: module declarations and exports.
- `crates/gb10x-ple/Cargo.toml`: move serde_json from development-only to normal dependencies; no new dependency family.
- `crates/gb10x-tools/tests/cli.rs`: only source-verifier help and missing-index tests; exclude native-probe assertions from the donor branch.
- `crates/gb10x-ple/src/writer.rs`, `reader.rs`, and `tests/roundtrip.rs`: fail-closed publication and automatic exact hot-row verification required by the audit.
- `README.md`: exact recovered scope and unverified real-model boundary.
- `docs/evidence/2026-09-05-repository-reconciliation.md`: root-owned audit and validation summary, written after tests.
- Task 2 only: `crates/gb10x-core/src/qwen38.rs` and focused tests in `crates/gb10x-core/tests/qwen38_contract.rs`; do not modify the pinned fixture.

### Task 1: Recover the existing Safetensors PLE source and verifier

**Interfaces:** Consume the current `ExactPleRowSource`, `PlePackIoError`, `PlePackWriter` and `PlePackReader`. Produce the donor's `SafetensorsPlePart`, `SafetensorsPleManifest`, `SafetensorsPleSource`, `qwen38_ple_manifest_from_index` and `source-verify` CLI without renaming or changing existing raw-source commands.

- [x] **Step 1: Confirm the baseline and inspect exact donor files.**

```bash
git rev-parse HEAD
git status --short
git show 27b6aee194f3028341f7b23ebae80548a8cb4e5f:crates/gb10x-ple/src/safetensors.rs
git show 27b6aee194f3028341f7b23ebae80548a8cb4e5f:crates/gb10x-ple/src/qwen38_manifest.rs
cargo test --workspace --all-targets
```

Expected baseline: current main at `e3ef3384`; 70 tests, without CUDA.

- [x] **Step 2: Recover the two donor integration test files first using apply_patch and confirm the absent API.**

Read each test with `git show 27b6aee194f3028341f7b23ebae80548a8cb4e5f:<path>` using the exact paths in File ownership. Add its unchanged contents with `apply_patch`, then run:

```bash
cargo test -p gb10x-ple --test safetensors_source --test qwen38_manifest
```

Expected RED: unresolved imports for the source/manifest API missing from main. Record the actual failure, not historical donor test claims.

- [x] **Step 3: Recover both production modules and wire existing dependencies/exports.**

Use the exact donor module contents with apply_patch. Add these declarations/exports to `crates/gb10x-ple/src/lib.rs` alongside its existing modules:

```rust
mod qwen38_manifest;
mod safetensors;
pub use qwen38_manifest::*;
pub use safetensors::*;
```

Move `serde_json.workspace = true` to `[dependencies]`; retain every other current manifest field (do not copy the donor's unrelated Rust-version inheritance).

```bash
cargo test -p gb10x-ple --test safetensors_source --test qwen38_manifest
```

Expected GREEN: exact miniature rows across parts, hash binding, geometry and malformed-file rejection. This does not prove full-model source identity.

- [x] **Step 4: Add the donor CLI tests first, show RED, then recover the source-verifier CLI.**

Copy only the `source-verify` help assertion and `plepack_source_verify_fails_closed_without_pinned_index` test from the donor CLI tests. Run `cargo test -p gb10x-tools --test cli`, confirm the failure, then recover the donor `gb10x-plepack.rs` implementation (its only functional addition is source verification). Do not modify `gb10x-probe` or its tests. Re-run the CLI tests to GREEN.

- [x] **Step 5: Prove the exact source works with the existing overlay seam.**

Extend the miniature Safetensors integration test through `PlePackWriter::write_overlay` and `PlePackReader::open`: hot rows must be byte-identical to source, a cold row must fall back to source, and changing a referenced tensor byte followed by reopening the source must make the old overlay fail provenance validation. Use the existing fixture helpers, a tiny source and actual file I/O, not a model-sized allocation or mocked row source. Record whether these checks pass immediately as characterization; do not manufacture a RED claim for already-existing behavior.

- [x] **Step 6: Close the two high-risk existing PLEPack defects before accepting the recovered source.**

First demonstrate with real temporary files that the existing writer replaces an existing destination, including the raw source itself, and that `PlePackReader::open` currently accepts a flipped hot-data byte. Tests must initially fail for those exact defects. Then make these minimal changes:

- Publication is create-only: an existing destination must never be replaced, including a symlink, hard link, directory, or a destination created during the build. Use an atomic no-clobber publication operation on the fully written and synced temporary file; a check followed by `rename` is insufficient. On Linux a same-directory `hard_link(temp, destination)` followed by removal of the temporary name provides this primitive without a new dependency. Fail closed on filesystems that cannot support it. Clean up only a temporary file owned by this invocation, never an existing unrelated file.
- Preserve the existing PLEPack v1 disk layout. Construct a reader privately, run its existing byte-for-byte `verify_hot_overlay` before returning it from `open`, and propagate failures. A corrupted hot row must be rejected before any public exact read is possible. Keep the source and sidecar immutability precondition explicit; this is not protection against another process mutating an active mmap.
- Add same-source, existing-destination preservation, alias, and publication-race regressions plus the corrupted-hot-data rejection. Use a deterministic test source callback to create the destination during the writer's row read; do not use timing sleeps.
- Document the compatibility/cost: output paths must now be new, and opening a sidecar reads every hot source row once. No `--force`, new on-disk version, generic model mode, or claimed performance improvement.

- [x] **Step 7: Run all host gates and self-review the complete diff.**

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
git diff --check
```

Confirm the diff excludes CUDA/native/CI changes. Update README to state recovery, source commit, synthetic-test boundary and real-model/GPU gaps. Write a report with commands, exit statuses, RED/GREEN evidence and findings. Do not commit or push; the controller retains the candidate for review and user handoff.

### Task 2: Reject the five previously ignored model-contract variations

**Interfaces:** Keep `Qwen38Config::from_json_str`, `load` and `validate_exact_contract` signatures. Extend the parsed struct with the missing typed fields; preserve every existing field/check and derived dimension method. No generic model configuration mode.

**Owned files:** `crates/gb10x-core/src/qwen38.rs` and `crates/gb10x-core/tests/qwen38_contract.rs`. The root controller updates the evidence document. No other production or fixture file is owned by this task.

The authoritative values for this bounded fix are already in `tests/fixtures/qwen38-flash-next-config.json`:

| JSON field | Required value | Example previously accepted mutation |
| --- | --- | --- |
| `text_config.hidden_act` | `"silu"` | `"relu"` |
| `text_config.attention_bias` | `false` (boolean) | `true` |
| `text_config.dtype` | `"bfloat16"` | `"float32"` |
| `text_config.rope_parameters.rope_type` | `"default"` | `"linear"` |
| `text_config.rope_parameters.partial_rotary_factor` | `0.25` | `0.5` while top-level factor remains `0.25` |

- [x] Add table-driven integration tests using an unmodified copy of the existing JSON fixture. The fixture parses and validates; each independent value mutation must fail the exact contract. Run only this test target and observe the current implementation accepting the mutations (RED).
- [x] Add the missing typed parsed fields and require their JSON types. Validate them with the same exact-value semantics as existing checks, including the nested factor independently of the top-level factor. Keep errors explicit about the offending field where practical. Re-run the regression to GREEN.
- [x] Add removal and wrong-type cases for all five fields, and verify that parse/validate fails closed. Exercise the public API, not private helper implementation. Preserve existing architecture and PLE-hash tests.
- [x] Run `cargo fmt --all -- --check`, the focused core test target and `cargo test -p gb10x-core`; run `cargo clippy -p gb10x-core --all-targets -- -D warnings`. Root runs the whole-workspace gates on the reviewed final tree.
- [x] Self-review the task diff and record exact RED/GREEN command/results, changed files and remaining limits in the assigned report. Do not commit, push, or spawn subagents.

This closes five concrete omissions; it does not certify that the fixture is an authentic released checkpoint or that every future graph option has been modeled. Do not weaken unknown-field behavior, invent reference hash outputs, or change PLE storage/native code.

## Review policy

Review existing recovered parsers as real code, not trusted merely because donor documentation says verified. Any material finding must first get a deterministic failing test before a minimal correction. Preserve format compatibility for valid inputs and report any architectural decision rather than silently importing CUDA or broadening model support.
