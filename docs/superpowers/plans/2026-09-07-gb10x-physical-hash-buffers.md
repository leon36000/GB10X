# Qwen3.8 PLE Physical Hash Buffers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the three persistent Qwen3.8 PLE hash buffers in a locally available pinned checkpoint while reading only its safetensors index, bounded headers and 280 bytes of tensor payload.

**Architecture:** Add a small bounded-file reader beside the existing safetensors mmap source. It parses only the selected `weight_map` entries, opens only their referenced `.safetensors` files, rejects malformed or escaping metadata, and compares exact little-endian `I64` payloads with the values that generated the existing upstream PLE oracle. A `gb10x-plepack hash-verify` command exposes the result without claiming full checkpoint provenance, inference, GPU execution or remote identity.

**Tech Stack:** Rust 2024 workspace, `std::fs::File::read_exact`, `serde_json`, existing duplicate-member JSON parser, Clap, tempfile test fixtures.

**Spec:** `docs/memory/03-model-ple.md` and `docs/evidence/2026-09-05-ple-reference.md`.

## Global Constraints

- Target only `Qwen/Qwen3.8-Flash-Next` revision `34567a4712bc9766c4449e2e98e4468bfa24d915`.
- Require the exact state-dict paths under `model.language_model.layers.1.ple.ple_embedding` and fail closed on any mismatch.
- Read at most a 16 MiB index, 16 MiB safetensors header per referenced file, and the three expected payloads: `3 + 16 + 16` little-endian `I64` values = 280 bytes.
- Do not map a shard, enumerate all payload bytes, instantiate Transformers, download a model, or infer that a supplied revision is independently proven.
- Use lexical path safety equivalent to the existing source verifier: reject empty, absolute, parent-directory and non-`.safetensors` targets; symlink containment remains outside this increment.
- Keep `source-verify` behavior intact. This command is a smaller physical-buffer check, not a replacement for whole-PLE source hashing.
- Preserve all pre-existing uncommitted work and do not touch GLM, Qwen27B, AgentOS or other projects.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `crates/gb10x-ple/src/safetensors.rs` | Parse a safetensors header from a bounded `File` read without mmap; preserve the existing mmap parser. |
| `crates/gb10x-ple/src/qwen38_hash_buffers.rs` | Pin names/values, resolve them through the index and compare only selected payload ranges. |
| `crates/gb10x-ple/src/lib.rs` | Export the verifier result and entry point. |
| `crates/gb10x-ple/tests/qwen38_hash_buffers.rs` | Build tiny real safetensors/index fixtures and prove acceptance/rejection behavior. |
| `crates/gb10x-tools/src/bin/gb10x-plepack.rs` | Add `hash-verify --model-dir --observed-revision` and stable JSON evidence output. |
| `crates/gb10x-tools/tests/cli.rs` | Prove the command is listed and fails closed for an absent index or revision mismatch. |
| `README.md` and `docs/evidence/2026-09-07-physical-hash-buffers.md` | State capability, byte boundary and validation limit precisely. |
| `docs/memory/{00-resume,03-model-ple,04-storage-cache,06-validation-telemetry,07-memory-sync}.md` | Record only the verified result, pending physical-checkpoint input and memory readback. |

### Task 1: Bounded safetensors header reader

**Files:**
- Modify: `crates/gb10x-ple/src/safetensors.rs`
- Test: `crates/gb10x-ple/tests/qwen38_hash_buffers.rs`

**Interfaces:**
- Consumes: an open `std::fs::File`, its metadata length and an explicit maximum header length.
- Produces: `pub(crate) fn read_safetensors_header_bounded(file: &mut File, max_header_bytes: u64) -> Result<ParsedHeader, PlePackIoError>` and parsed tensor offsets relative to the data section.

- [ ] **Step 1: Write the failing header-cap and exact-payload test**

```rust
#[test]
fn verifier_rejects_a_safetensors_header_larger_than_the_bound() {
    let root = tempdir().unwrap();
    write_index_and_hash_buffers(root.path(), HashFixture::oversized_header());

    assert!(verify_qwen38_ple_hash_buffers(
        root.path(),
        QWEN38_FLASH_NEXT_REVISION,
    )
    .is_err());
}
```

- [ ] **Step 2: Run the test to verify it fails because the verifier/reader does not exist**

Run:

```bash
cargo test --locked -p gb10x-ple --test qwen38_hash_buffers verifier_rejects_a_safetensors_header_larger_than_the_bound
```

Expected: compilation failure naming the missing `verify_qwen38_ple_hash_buffers` symbol.

- [ ] **Step 3: Implement only the bounded header read primitive**

```rust
pub(crate) fn read_safetensors_header_bounded(
    file: &mut File,
    max_header_bytes: u64,
) -> Result<ParsedHeader, PlePackIoError> {
    // Seek to byte 0, read the 8-byte little-endian header length,
    // reject zero, overflow and `header_len > max_header_bytes`, then
    // read exactly `header_len` JSON bytes and parse them against file length.
}
```

Refactor the common JSON/header validation so the existing mmap parser and this reader apply the same dtype, shape, range and overlap checks.

- [ ] **Step 4: Run the targeted test to verify the bound rejects before payload access**

Run the command from Step 2.

Expected: PASS after the hash verifier is added in Task 2; until then, retain this test as the first red test for Task 2.

- [ ] **Step 5: Commit**

```bash
git add crates/gb10x-ple/src/safetensors.rs crates/gb10x-ple/tests/qwen38_hash_buffers.rs
git commit -m "feat: bound safetensors metadata reads"
```

### Task 2: Exact physical hash-buffer verifier

**Files:**
- Create: `crates/gb10x-ple/src/qwen38_hash_buffers.rs`
- Modify: `crates/gb10x-ple/src/lib.rs`
- Test: `crates/gb10x-ple/tests/qwen38_hash_buffers.rs`

**Interfaces:**
- Consumes: `model_dir: impl AsRef<Path>` and `observed_revision: &str`.
- Produces:

```rust
pub const QWEN38_PLE_HASH_BUFFER_PAYLOAD_BYTES: u64 = 280;
pub fn verify_qwen38_ple_hash_buffers(
    model_dir: impl AsRef<Path>,
    observed_revision: &str,
) -> Result<Qwen38PleHashBufferVerification, PlePackIoError>;
```

- [ ] **Step 1: Write the accepting fixture test**

```rust
#[test]
fn physical_i64_hash_buffers_match_pinned_upstream_defaults() {
    let root = tempdir().unwrap();
    write_index_and_hash_buffers(root.path(), HashFixture::valid());

    let verification = verify_qwen38_ple_hash_buffers(
        root.path(),
        QWEN38_FLASH_NEXT_REVISION,
    )
    .expect("matching physical hash buffers");

    assert_eq!(verification.payload_bytes_read, 280);
    assert_eq!(verification.buffers_verified, 3);
}
```

- [ ] **Step 2: Run it to verify it fails for the missing verifier**

Run:

```bash
cargo test --locked -p gb10x-ple --test qwen38_hash_buffers physical_i64_hash_buffers_match_pinned_upstream_defaults
```

Expected: compilation failure naming the missing verifier.

- [ ] **Step 3: Implement the minimal fail-closed verifier**

Define exact tensor names:

```text
model.language_model.layers.1.ple.ple_embedding.layer_multipliers
model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes
model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets
```

For each name, read the bounded index, require a safe `.safetensors` target, find the matching `I64` header entry with shape `[3]`, `[16]` or `[16]`, seek to its data-section offset, read only its expected byte count, decode `i64::from_le_bytes`, and compare against the pinned upstream defaults. Return counts for unique files, header bytes and payload bytes actually read.

- [ ] **Step 4: Run the accepting test and the full new test file**

Run:

```bash
cargo test --locked -p gb10x-ple --test qwen38_hash_buffers
```

Expected: PASS.

- [ ] **Step 5: Add and prove three rejection tests**

```rust
#[test]
fn changed_multiplier_is_rejected() { /* mutate one I64 payload byte */ }

#[test]
fn duplicate_index_member_is_rejected() { /* raw duplicate weight_map JSON */ }

#[test]
fn escaping_or_wrongly_typed_hash_buffer_is_rejected() { /* ../ and F32 cases */ }
```

Run the new test file after each test; each new test must first fail for the missing validation and then pass after the smallest implementation change.

- [ ] **Step 6: Commit**

```bash
git add crates/gb10x-ple/src/qwen38_hash_buffers.rs crates/gb10x-ple/src/lib.rs crates/gb10x-ple/src/safetensors.rs crates/gb10x-ple/tests/qwen38_hash_buffers.rs
git commit -m "feat: verify physical Qwen PLE hash buffers"
```

### Task 3: CLI evidence boundary

**Files:**
- Modify: `crates/gb10x-tools/src/bin/gb10x-plepack.rs`
- Modify: `crates/gb10x-tools/tests/cli.rs`
- Modify: `README.md`

**Interfaces:**
- Consumes: `gb10x-plepack hash-verify --model-dir <dir> --observed-revision <sha>`.
- Produces JSON with `state: "verified-local-hash-buffers"`, model ID, revision contract, observed revision, `buffers_verified`, `files_read`, `header_bytes_read`, `payload_bytes_read`, and the fixed `payload_bytes_expected: 280`.

- [ ] **Step 1: Write the failing CLI-help test**

```rust
#[test]
fn plepack_cli_lists_hash_verify_and_requires_a_pinned_revision() {
    let help = plepack().arg("--help").output().unwrap();
    assert!(stdout(&help).contains("hash-verify"));

    let missing = plepack().args(["hash-verify", "--model-dir", "."]).output().unwrap();
    assert!(!missing.status.success());
}
```

- [ ] **Step 2: Run the test and verify it fails because the subcommand is absent**

Run:

```bash
cargo test --locked -p gb10x-tools --test cli plepack_cli_lists_hash_verify_and_requires_a_pinned_revision
```

Expected: assertion failure because help does not include `hash-verify`.

- [ ] **Step 3: Add the exact subcommand and output contract**

```rust
HashVerify(HashVerifyArgs),

#[derive(Debug, Args)]
struct HashVerifyArgs {
    #[arg(long)] model_dir: PathBuf,
    #[arg(long)] observed_revision: String,
}
```

Call `verify_qwen38_ple_hash_buffers`; emit only local observation fields. Do not emit a remote digest match or describe the input revision as authenticated provenance.

- [ ] **Step 4: Run focused CLI tests**

Run:

```bash
cargo test --locked -p gb10x-tools --test cli
```

Expected: PASS.

- [ ] **Step 5: Document precise limits**

Add `hash-verify` to the README command list and state that it reads a bounded index/headers plus 280 payload bytes, verifies physical local buffer values only, and cannot validate a missing/remote/full checkpoint.

- [ ] **Step 6: Commit**

```bash
git add crates/gb10x-tools/src/bin/gb10x-plepack.rs crates/gb10x-tools/tests/cli.rs README.md
git commit -m "feat: expose bounded PLE hash verification"
```

### Task 4: Evidence, verification and continuity

**Files:**
- Create: `docs/evidence/2026-09-07-physical-hash-buffers.md`
- Modify: `docs/memory/00-resume.md`
- Modify: `docs/memory/03-model-ple.md`
- Modify: `docs/memory/04-storage-cache.md`
- Modify: `docs/memory/06-validation-telemetry.md`
- Modify: `docs/memory/07-memory-sync.md`
- Modify: `docs/memory/checkpoint.json`
- Modify: `docs/memory/sync-receipt.md`

**Interfaces:**
- Consumes: fresh test/build outputs and the live connected-memory readback.
- Produces: a revision incremented from 3 only after every claimed host gate succeeds.

- [ ] **Step 1: Run the fresh full workspace gates**

```bash
cargo fmt --all -- --check
CARGO_TARGET_DIR=/tmp/gb10x-hash-tests cargo test --locked --workspace --all-targets
CARGO_TARGET_DIR=/tmp/gb10x-hash-clippy cargo clippy --locked --workspace --all-targets -- -D warnings
CARGO_TARGET_DIR=/tmp/gb10x-hash-release cargo build --locked --workspace --release
git diff --check
```

Record exact pass/fail evidence; if a reused default cache fails, rerun in a fresh target and name the cache issue rather than changing source to mask it.

- [ ] **Step 2: Perform a diff review**

Check the code against the global constraints: no mmap in the verifier, no uncapped index/header read, payload counter equals 280 only on a valid fixture, safe paths, exact shapes/dtype/names and no full-model claim.

- [ ] **Step 3: Write concise evidence and update only affected memory domains**

State the exact contract and tests. Distinguish fixture proof from a physical check: until a real pinned model directory is supplied, the command proves the verifier rather than buffer agreement with a shipped checkpoint.

- [ ] **Step 4: Build revision 4 and synchronize it**

Generate the canonical nine-document bundle and code-tree manifest; update MongoDB Atlas and Neon with optimistic revision-3/hash guards; write a short pointer to MCP TO PC. Read all three back exactly, record an unavailable/stale index truthfully, and retain the local receipt.

- [ ] **Step 5: Commit**

```bash
git add README.md crates docs tests Cargo.lock
git commit -m "docs: record bounded PLE hash verification evidence"
```

## Self-Review

- Coverage: Tasks 1–2 implement bounded reads and exact comparison; Task 3 exposes it without provenance overclaim; Task 4 captures full gates and continuity.
- Placeholder scan: no `TODO`, `TBD`, or undefined later interfaces remain.
- Type consistency: `verify_qwen38_ple_hash_buffers` and `Qwen38PleHashBufferVerification` are defined in Task 2 and consumed unchanged in Task 3.

## Execution

The user authorized continued autonomous work, so execute this plan inline in the current branch using test-first increments. Do not publish or merge the existing uncommitted candidate solely because this plan is complete.
