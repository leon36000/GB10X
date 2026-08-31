# GB10X Bootstrap Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a correctness-first, testable GB10X core that validates the dedicated GB10 platform contract, validates the pinned Qwen3.8-Flash-Next architecture, implements exact PLE hashing, and builds/verifies an exact PLEPack container before CUDA inference work begins.

**Architecture:** A small Rust workspace separates immutable model/hardware contracts from Linux probing and PLE storage. Hardware-dependent facts are collected into plain structs so the policy can be unit-tested off-device; the final runtime refuses unsupported hardware. PLEPack is a deterministic, checksummed exact-value re-layout whose reader is validated against source-row fixtures.

**Tech Stack:** Rust stable, serde/serde_json, sha2, memmap2, libc, clap for tools, tempfile/proptest for tests. Linux/aarch64 is the production target; portable unit tests validate pure logic only.

**Spec:** `docs/superpowers/specs/2026-08-31-gb10x-cache-first-v3-design.md`

## Global Constraints

- Production target is NVIDIA GB10 / DGX Spark only.
- Production OS/CPU target is Linux `aarch64` only.
- Production CUDA target is native `sm_121a` only.
- Model target is Qwen3.8-Flash-Next only.
- Dedicated-server operation is assumed.
- `correctness -> measured A/B win -> production` is mandatory.
- No generic multi-model abstractions are added.
- No production RT-core path is added in this milestone.
- No reuse of the separate `nextgen memory` project or its data.

---

## File Structure

```text
Cargo.toml                         workspace definition
rust-toolchain.toml               Rust channel/components
.gitignore                        build/generated PLEPack artifacts
crates/gb10x-core/
  Cargo.toml
  src/lib.rs                      public core API
  src/platform.rs                 immutable GB10 platform contract structs
  src/qwen38.rs                   Qwen3.8-Flash-Next config contract
  src/ple_hash.rs                 exact PLE hash/window implementation
crates/gb10x-runtime/
  Cargo.toml
  src/lib.rs
  src/linux_probe.rs              sysfs/uname/NVIDIA fact collection
  src/validate.rs                 fail-closed GB10 validation
crates/gb10x-ple/
  Cargo.toml
  src/lib.rs
  src/format.rs                   PLEPack header/index definitions
  src/writer.rs                   deterministic exact pack writer
  src/reader.rs                   exact logical-row reader
  src/layout.rs                   block-placement policy
  tests/roundtrip.rs              source-vs-pack exactness
crates/gb10x-telemetry/
  Cargo.toml
  src/lib.rs                      stage/counter data types
  src/record.rs                   JSON benchmark/evidence record
crates/gb10x-tools/
  Cargo.toml
  src/bin/gb10x-probe.rs          platform probe CLI
  src/bin/gb10x-plepack.rs        PLEPack build/verify CLI
tests/fixtures/
  qwen38-flash-next-config.json   minimal released-shape fixture
  cache-topology/                 deterministic sysfs-like cache fixtures
docs/evidence/README.md           evidence naming contract
```

---

### Task 1: Initialize the Rust workspace and evidence contract

**Files:**
- Create: `Cargo.toml`
- Create: `rust-toolchain.toml`
- Create: `.gitignore`
- Create: `docs/evidence/README.md`
- Create: crate manifests and minimal `src/lib.rs` files for `gb10x-core`, `gb10x-runtime`, `gb10x-ple`, `gb10x-telemetry`, `gb10x-tools`

**Interfaces:**
- Produces a workspace where `cargo test --workspace` succeeds.
- No runtime functionality is exposed yet.

- [ ] **Step 1: Add the workspace manifest**

```toml
[workspace]
resolver = "2"
members = [
  "crates/gb10x-core",
  "crates/gb10x-runtime",
  "crates/gb10x-ple",
  "crates/gb10x-telemetry",
  "crates/gb10x-tools",
]

[workspace.package]
edition = "2024"
license = "Apache-2.0"
repository = "https://github.com/leon36000/GB10X"

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
thiserror = "2"
libc = "0.2"
memmap2 = "0.9"
clap = { version = "4", features = ["derive"] }
tempfile = "3"
proptest = "1"
hex = "0.4"
```

- [ ] **Step 2: Pin the toolchain channel and formatting components**

```toml
[toolchain]
channel = "stable"
profile = "minimal"
components = ["clippy", "rustfmt"]
```

- [ ] **Step 3: Add evidence rules**

`docs/evidence/README.md` must require every performance record to contain commit, model digest/revision, command/config, context, prompt/output tokens, cache state, speculation settings, affinity, prefill/decode/TTFT, memory, and correctness result.

- [ ] **Step 4: Verify formatting/build/test baseline**

Run:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml rust-toolchain.toml .gitignore crates docs/evidence
 git commit -m "chore: bootstrap GB10X Rust workspace"
```

---

### Task 2: Define the fail-closed GB10 platform contract

**Files:**
- Create: `crates/gb10x-core/src/platform.rs`
- Modify: `crates/gb10x-core/src/lib.rs`
- Create: `crates/gb10x-runtime/src/validate.rs`
- Test: unit tests colocated in both modules

**Interfaces:**
- Produces `PlatformSnapshot`, `CpuCache`, `GpuSnapshot`, `Gb10Validation`, `validate_gb10(&PlatformSnapshot) -> Result<Gb10Validation, PlatformError>`.

- [ ] **Step 1: Write failing validation tests**

```rust
#[test]
fn rejects_x86_even_if_gpu_claims_sm121() {
    let mut p = PlatformSnapshot::gb10_test_fixture();
    p.arch = "x86_64".into();
    assert!(matches!(validate_gb10(&p), Err(PlatformError::Architecture(_))));
}

#[test]
fn rejects_non_121_compute_capability() {
    let mut p = PlatformSnapshot::gb10_test_fixture();
    p.gpu.compute_major = 12;
    p.gpu.compute_minor = 0;
    assert!(matches!(validate_gb10(&p), Err(PlatformError::ComputeCapability { .. })));
}
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cargo test -p gb10x-runtime validate_gb10 -- --nocapture
```

Expected: FAIL because types/functions do not exist.

- [ ] **Step 3: Implement minimal contract types**

Core public shapes:

```rust
pub struct PlatformSnapshot {
    pub arch: String,
    pub kernel_release: String,
    pub cpu_model: String,
    pub online_cpus: Vec<u32>,
    pub caches: Vec<CpuCache>,
    pub gpu: GpuSnapshot,
    pub mem_total_bytes: u64,
    pub page_size_bytes: u64,
}

pub struct CpuCache {
    pub level: u8,
    pub cache_type: CacheType,
    pub size_bytes: u64,
    pub line_bytes: u32,
    pub shared_cpu_ids: Vec<u32>,
}

pub struct GpuSnapshot {
    pub name: String,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub l2_bytes: u64,
    pub persisting_l2_max_bytes: u64,
    pub total_memory_bytes: u64,
}
```

Validation must require `aarch64`, compute capability `12.1`, nonzero cache/memory facts, and a GPU name containing the GB10 identity expected from the probe. It must return discovered values rather than replacing them with published constants.

- [ ] **Step 4: Run tests**

```bash
cargo test -p gb10x-runtime
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/gb10x-core crates/gb10x-runtime
 git commit -m "feat: add fail-closed GB10 platform contract"
```

---

### Task 3: Implement Linux CPU/cache topology probing

**Files:**
- Create: `crates/gb10x-runtime/src/linux_probe.rs`
- Modify: `crates/gb10x-runtime/src/lib.rs`
- Create: `tests/fixtures/cache-topology/cpu0/index0/*`
- Create: additional fixture directories for private L2 and shared L3 groups

**Interfaces:**
- Produces `read_cpu_cache_tree(root: &Path) -> Result<Vec<CpuCache>, ProbeError>`.
- Produces `probe_host() -> Result<HostProbe, ProbeError>` for production Linux.

- [ ] **Step 1: Write failing parser tests using fixture files**

```rust
#[test]
fn parses_shared_cpu_list_ranges() {
    assert_eq!(parse_cpu_list("0-3,8,10-11").unwrap(), vec![0,1,2,3,8,10,11]);
}

#[test]
fn cache_fixture_preserves_private_and_shared_groups() {
    let caches = read_cpu_cache_tree(fixture_root()).unwrap();
    assert!(caches.iter().any(|c| c.level == 2 && c.shared_cpu_ids == vec![0]));
    assert!(caches.iter().any(|c| c.level == 3 && c.shared_cpu_ids.len() > 1));
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p gb10x-runtime linux_probe -- --nocapture
```

- [ ] **Step 3: Implement parsing without shelling out for cache topology**

Read Linux sysfs `level`, `type`, `size`, `coherency_line_size`, and `shared_cpu_list`. Support `K/M/G` suffixes with checked arithmetic. Deduplicate identical shared cache objects seen from multiple CPUs.

- [ ] **Step 4: Add production host facts**

Use `std::env::consts::ARCH`, `/proc/cpuinfo`, `/proc/meminfo`, `uname(2)`, and `sysconf(_SC_PAGESIZE)`. Keep NVIDIA GPU probing behind a trait in this task so CUDA integration can replace it in Milestone 2.

- [ ] **Step 5: Test**

```bash
cargo test -p gb10x-runtime
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/gb10x-runtime tests/fixtures/cache-topology
 git commit -m "feat: probe Linux CPU cache topology"
```

---

### Task 4: Pin and validate Qwen3.8-Flash-Next configuration

**Files:**
- Create: `crates/gb10x-core/src/qwen38.rs`
- Modify: `crates/gb10x-core/src/lib.rs`
- Create: `tests/fixtures/qwen38-flash-next-config.json`

**Interfaces:**
- Produces `Qwen38Config::load(path) -> Result<Qwen38Config, Qwen38ConfigError>`.
- Produces `Qwen38Config::validate_exact_contract() -> Result<(), Qwen38ConfigError>`.

- [ ] **Step 1: Add released-shape fixture and failing contract tests**

Tests must assert the exact immutable dimensions used by GB10X, including hidden 2560, 48 layers, repeating GDN/GDN/GDN/QSA schedule, 24 Q heads, 2 KV heads, 256 head dimension, 512 experts, top-k 10, intermediate 640, n-gram size 3, 8 heads/order, 2048 QSA budget, and 262144 max positions.

Also test one mutation:

```rust
#[test]
fn rejects_changed_expert_count() {
    let mut cfg = fixture_config();
    cfg.num_experts = 256;
    assert!(cfg.validate_exact_contract().is_err());
}
```

- [ ] **Step 2: Verify tests fail**

```bash
cargo test -p gb10x-core qwen38 -- --nocapture
```

- [ ] **Step 3: Implement parser and exact validator**

Do not add a model-family enum. This is Qwen3.8-Flash-Next-only code.

- [ ] **Step 4: Test**

```bash
cargo test -p gb10x-core
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/gb10x-core tests/fixtures/qwen38-flash-next-config.json
 git commit -m "feat: pin Qwen3.8 Flash Next architecture"
```

---

### Task 5: Implement exact PLE hash state and rollback semantics

**Files:**
- Create: `crates/gb10x-core/src/ple_hash.rs`
- Modify: `crates/gb10x-core/src/lib.rs`

**Interfaces:**
- Produces `PleHashPlan` and `PleTokenWindow`.
- `PleHashPlan::rows_for_token(&self, window: &mut PleTokenWindow, token: u32, out: &mut [u32; 16])`.
- Window supports begin/commit/abort and commit-prefix semantics for speculation.

- [ ] **Step 1: Write failing property/unit tests**

Required tests:

```rust
#[test]
fn emits_exactly_sixteen_rows_per_token() { /* fixture plan -> len 16 */ }

#[test]
fn abort_restores_previous_ngram_window() { /* begin, append, abort, compare snapshot */ }

#[test]
fn commit_prefix_matches_replaying_only_accepted_tokens() { /* speculation rollback */ }
```

Use proptest to assert emitted rows remain within each head’s `[offset, offset + vocab_size)` interval across arbitrary token IDs.

- [ ] **Step 2: Verify failure**

```bash
cargo test -p gb10x-core ple_hash -- --nocapture
```

- [ ] **Step 3: Implement with checked integer arithmetic**

Use the checkpoint-supplied multipliers, per-head vocab sizes and offsets. Missing left context is EOS; EOS resets the window.

- [ ] **Step 4: Run tests**

```bash
cargo test -p gb10x-core
```

Expected: PASS including property tests.

- [ ] **Step 5: Commit**

```bash
git add crates/gb10x-core
 git commit -m "feat: implement exact Qwen PLE hashing"
```

---

### Task 6: Define PLEPack exact format and deterministic layout planner

**Files:**
- Create: `crates/gb10x-ple/src/format.rs`
- Create: `crates/gb10x-ple/src/layout.rs`
- Modify: `crates/gb10x-ple/src/lib.rs`

**Interfaces:**
- `PlePackHeader` includes magic/version, source digest, row count, row width, block size, index digest.
- `LogicalRowPlacement { logical_row, block_id, offset_in_block }`.
- `plan_exact_layout(row_count, row_bytes, block_bytes, trace) -> Result<LayoutPlan, PlePackError>`.

- [ ] **Step 1: Write failing deterministic-layout tests**

```rust
#[test]
fn same_trace_produces_byte_identical_plan() { /* serialize two plans and compare */ }

#[test]
fn every_logical_row_is_mapped_once() { /* set equality */ }

#[test]
fn no_row_crosses_block_boundary() { /* placement offset + row_bytes <= block_bytes */ }
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p gb10x-ple layout -- --nocapture
```

- [ ] **Step 3: Implement a simple exact baseline layout**

Milestone 1 uses deterministic row packing with optional trace-derived co-access ordering. It does not quantize or transform row bytes.

- [ ] **Step 4: Test**

```bash
cargo test -p gb10x-ple
```

- [ ] **Step 5: Commit**

```bash
git add crates/gb10x-ple
 git commit -m "feat: define exact PLEPack layout"
```

---

### Task 7: Implement PLEPack writer/reader with exact round-trip verification

**Files:**
- Create: `crates/gb10x-ple/src/writer.rs`
- Create: `crates/gb10x-ple/src/reader.rs`
- Create: `crates/gb10x-ple/tests/roundtrip.rs`
- Modify: `crates/gb10x-ple/src/lib.rs`

**Interfaces:**
- `PlePackWriter::write_rows<I: Iterator<Item = Result<Vec<u8>, E>>>(...)` writes immutable pack/index files.
- `PlePackReader::open(path) -> Result<Self, PlePackError>`.
- `PlePackReader::read_exact_row(logical_row, dst) -> Result<(), PlePackError>`.
- `verify_pack_against_source(...) -> Result<VerificationReport, PlePackError>`.

- [ ] **Step 1: Write failing round-trip tests**

Generate synthetic 320-byte rows with deterministic content, pack them in a nontrivial remapped order, read all logical rows back, and compare every byte.

Corruption test:

```rust
#[test]
fn corrupted_index_digest_is_rejected() { /* flip one index byte, open must fail */ }
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p gb10x-ple --test roundtrip -- --nocapture
```

- [ ] **Step 3: Implement writer and mmap-backed exact reader**

All offsets use checked `u64`/`usize` conversions. Header/index/source digests use SHA-256. Writer uses temp files followed by atomic rename so interrupted preparation never publishes a valid-looking partial pack.

- [ ] **Step 4: Test**

```bash
cargo test -p gb10x-ple
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/gb10x-ple
 git commit -m "feat: add exact PLEPack writer and reader"
```

---

### Task 8: Add structured telemetry/evidence records

**Files:**
- Create: `crates/gb10x-telemetry/src/record.rs`
- Modify: `crates/gb10x-telemetry/src/lib.rs`

**Interfaces:**
- Produces serializable `RunIdentity`, `StageTimings`, `CacheCounters`, `PerformanceMetrics`, `CorrectnessGate`, `EvidenceRecord`.
- Produces `EvidenceRecord::validate() -> Result<(), EvidenceError>`.

- [ ] **Step 1: Write failing evidence-completeness tests**

```rust
#[test]
fn performance_record_without_commit_is_rejected() { /* empty commit -> error */ }

#[test]
fn record_requires_correctness_result() { /* None -> error */ }
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p gb10x-telemetry -- --nocapture
```

- [ ] **Step 3: Implement strict record types**

Use integers/durations for machine metrics. Store experimental mode labels explicitly so approximate and exact runs cannot be silently mixed.

- [ ] **Step 4: Test**

```bash
cargo test -p gb10x-telemetry
```

- [ ] **Step 5: Commit**

```bash
git add crates/gb10x-telemetry
 git commit -m "feat: add strict benchmark evidence records"
```

---

### Task 9: Add `gb10x-probe` and `gb10x-plepack` CLIs

**Files:**
- Create: `crates/gb10x-tools/src/bin/gb10x-probe.rs`
- Create: `crates/gb10x-tools/src/bin/gb10x-plepack.rs`
- Modify: `crates/gb10x-tools/Cargo.toml`

**Interfaces:**
- `gb10x-probe --json` prints probed host/cache facts and validation state.
- `gb10x-plepack build --source ... --out ...` builds an exact pack from a prepared row source.
- `gb10x-plepack verify --source ... --pack ...` verifies every logical row or a deterministic sample mode explicitly requested by the operator.

- [ ] **Step 1: Write CLI integration tests for argument validation**

Use `Command` tests to prove missing source/output arguments fail and `--help` succeeds.

- [ ] **Step 2: Implement thin CLIs over library APIs**

Do not duplicate platform or PLE logic in binaries.

- [ ] **Step 3: Run tests and clippy**

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
git add crates/gb10x-tools
 git commit -m "feat: add GB10 probe and PLEPack tools"
```

---

### Task 10: Milestone verification and next-phase handoff

**Files:**
- Create: `docs/evidence/m1-bootstrap-verification.md`
- Modify: `README.md`

**Interfaces:**
- Produces a verified Milestone 1 evidence document and exact commands for on-GB10 probing.

- [ ] **Step 1: Run full host-independent verification**

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: all PASS.

- [ ] **Step 2: Build release binaries in the current environment**

```bash
cargo build --workspace --release
```

Record actual host architecture and explicitly state that this does **not** prove GB10/aarch64/CUDA execution when run off the Spark.

- [ ] **Step 3: Document the first on-GB10 gate**

Required commands on DGX Spark:

```bash
uname -m
./target/release/gb10x-probe --json
nvidia-smi
nvcc --version
```

Milestone 2 may begin native CUDA work only after the probe reports `aarch64`, GB10, and CC 12.1.

- [ ] **Step 4: Update README status without overclaiming**

State exactly which components are implemented and which hardware claims remain unverified.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/evidence/m1-bootstrap-verification.md
 git commit -m "docs: record GB10X bootstrap verification"
```

## Plan self-review

- Spec coverage for this milestone: platform contract, cache topology, model contract, exact PLE hashing, exact PLEPack, evidence discipline are covered.
- Deferred deliberately to later plans: CUDA kernel ABI, NVFP4/GDN/QSA/MoE forward path, CUDA Graphs, GPU L2 controls, MTP execution, governors, ARM SVE2 kernels, L3Draft, ExpertScout, RT experiment.
- No performance claim is permitted from Milestone 1.
- The milestone is independently useful because it produces the exact platform/model/storage contracts required to prevent later CUDA work from building on incorrect assumptions.
