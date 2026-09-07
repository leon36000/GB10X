# GB10X — repository reconciliation and continuation, 2026-09-05

## Bottom line

There is real implemented work, including merged host logic and two unmerged native efforts. GB10X is not yet an operational or performance-validated inference engine. Passing host tests do not establish checkpoint identity, exact upstream model semantics, or execution on GB10.

This audit covers the repository's branch history, architecture/specifications, Rust crates and host tests, both native build/ABI surfaces, GitHub CI evidence, and available execution paths. Two independent read-only code reviews covered the host foundation and the previously overlooked native-M2 branch. This is an engineering audit, not a proof that every program behavior is correct.

Continuation as of 2026-09-07: the candidate now passes 111 host tests and includes the telemetry
validation correction plus domain memory. Read [the current resume index](../memory/00-resume.md)
and [new evidence](2026-09-07-telemetry-memory.md). Dated observations below retain their historical
scope; in particular, the DGX is now reported unreachable rather than reachable.

## Reconciled Git state

| Surface | Observed revision | Actual state |
| --- | --- | --- |
| `main` | `e3ef3384a20a3af640f418fa56ca32c62f7fc190` | M1 foundation plus M2 CPU Cache Fabric / PLE-Hydra simulation. PR #2 is merged. |
| `feature/cpu-cache-fabric-ple-hydra` | `29d7c896` | Published M2 work already integrated into main through PR #2. |
| `feature/cuda-sm121a-abi-spec`, PR #3 | `0b0bcf8febcccf5a2b97574ac5ca674754855b3c` | Open M3 branch: opt-in native CMake, C ABI v1, device probe and smoke contract, Rust layout mirror. No inference. |
| `feature/gb10x-native-m2` | `27b6aee194f3028341f7b23ebae80548a8cb4e5f` | 71 commits ahead / one behind main: separate CUDA crate/build.rs, native checksum smoke and BF16 RMSNorm, Safetensors PLE source, verifier, compile-only CI and handoff scripts. Not merged. |
| `feature/gb10x-bootstrap` | `71e7b8d56326229883ea61e87e957cec046428ff` | Squash-history divergence; its tree is the main foundation without the five M2 CPU files, not another missing implementation. |
| Current recovery candidate | `feature/ple-source-recovery-20260905`, based on `e3ef3384` | Isolated host-only PLE recovery and storage safety work. Local only; no new commit, push, PR or merge. |

The original local checkout remains on `c498fd1` with two pre-existing edits. Its tracked source content is identical to published `0b0bcf8`; these edits were preserved, not reset. Validation generated local dependency/build artifacts. The recovery uses a separate worktree.

Links: [merged PR #2](https://github.com/leon36000/GB10X/pull/2), [open PR #3](https://github.com/leon36000/GB10X/pull/3), [native-M2 branch](https://github.com/leon36000/GB10X/tree/feature/gb10x-native-m2).

## What exists by subsystem

| Subsystem | Implemented | Not established |
| --- | --- | --- |
| `gb10x-core` | Pinned-shape config parser, platform contract, PLE hashing / transactions checked against upstream vectors; config fixture confirmed as an official-config projection | Actual checkpoint hash-buffer identity; complete fail-closed graph validation |
| `gb10x-ple` on main | Exact-row abstraction, raw mmap source, deterministic overlay layout, on-disk PLEPack v1 writer/reader | Production byte budget, complete manifest provenance, immutable-file enforcement |
| `gb10x-runtime` on main | Linux host discovery, GB10 host validation, private-L2 ownership/budgets, greedy cache admission, four-tier deterministic PLE-Hydra trace simulation | Real cache placement, I/O scheduling, GPU execution, runtime throughput |
| `gb10x-telemetry` | Serializable evidence schema and validators | Operational instrumentation, full stage breakdown and strict exact/approximate representation semantics |
| `gb10x-tools` on main | Host probe and raw PLEPack build/verify commands | End-to-end model execution |
| M3 native branch | Strict opt-in aarch64 / `compute_121a -> sm_121a` CMake surface, versioned C ABI/device probe, host layout tests | Native GB10 build/run; linked Rust runtime; real kernels or model forward |
| Native-M2 branch | Opt-in Rust CUDA binding/build, device probe, device checksum, one-row BF16 RMSNorm2560, CPU oracles, PLE Safetensors loader and index mapper | Current green host CI; GB10 execution; actual checkpoint verification; full forward pass |

## Why the native branches cannot simply be merged

Both export `gb10x_cuda_probe_device` with incompatible layouts and status conventions:

| Contract | Native-M2 | M3 |
| --- | --- | --- |
| Output bytes | 304 | 296 |
| First field | Callee-written `abi_version` | Caller-initialized `struct_size` |
| Offsets 40 / 44 | `sm_count` / `warp_size` | Start of `device_name` |
| Name offset | 48 | 40 |
| Status | Negative signed integer conventions | Stable unsigned ABI statuses 0–4 |
| Native authority | `gb10x-cuda`, `build.rs`, its own header | `cuda/` CMake and M3 header; Rust mirror in runtime |

An M3 caller linked to M2's implementation can receive a 304-byte write into a 296-byte object. The reverse pairing also violates initialization/layout rules. This is a semantic conflict even though Git's textual merge reports only a README conflict. No native merge or symbol adaptation was attempted.

Before native integration, choose one ABI/build authority, settle size negotiation and device fields, and adapt kernel bodies/oracles to that authority. Do not copy both headers, probe functions or FFI declarations unchanged.

## Evidence reproduced or inspected

### Fresh local M3 host checks

Environment: Linux x86_64, isolated official Rust 1.98.0 (`88d9e12ae`), CMake 3.31.10, GCC 13.3.0. Tool installations are confined to this scratch execution environment, not PC1 or the DGX. No CUDA toolkit/GPU is exposed here.

On the original checkout whose source matches `0b0bcf8`:

- `cargo test --workspace --all-targets`: exit 0; 75 tests passed, none ignored.
- `cargo fmt --all -- --check`: exit 0.
- `cargo clippy --workspace --all-targets -- -D warnings`: exit 0.
- `cargo build --workspace --release`: exit 0.
- CMake CUDA-disabled configure and build: both exit 0.
- CMake CUDA-enabled configure on x86_64: exit 1 as required, with `GB10X CUDA requires native aarch64, found x86_64`.
- C11 header syntax/layout probe with `-Wall -Wextra -Werror`: exit 0; ABI-info size/alignment 32/4, device-info size/alignment 296/8, `struct_size` at 0 and `device_name` at 40.

The negative architecture check proves rejection on this host, not successful CUDA compilation. The C11 probe establishes header layout, not native implementation correctness.

### GitHub CI

- M3 `host-logic-ci` run #252, ID 33658687862 on `0b0bcf8`: successful format/tests/Clippy/release evidence. [Run](https://github.com/leon36000/GB10X/actions/runs/33658687862)
- Native-M2 HEAD run ID 33459606672 on `27b6aee`: CUDA 12.9 compile-only job passed; host job failed at `Workspace crates inherit Rust version`, before format/tests/Clippy/release. `gb10x-cuda/Cargo.toml` lacks `rust-version.workspace = true`. [Run](https://github.com/leon36000/GB10X/actions/runs/33459606672)
- Native-M2 last fully green run ID 33458739454 at `8116325`: useful historical host/compiler evidence. Changes from that revision to HEAD are CI/toolchain/package metadata, not native implementation code. It is nevertheless incorrect to call HEAD's complete CI green. [Run](https://github.com/leon36000/GB10X/actions/runs/33458739454)

### Actual hardware boundary

The controller reports `gx10-e526` online and reachable. A strictly read-only remote worker preflight was refused before execution with `remote worker launch requires a ForgeAI node agent; host health remains available`. The conversation does not expose the normal host terminal tool.

No remote build, SSH command, CUDA kernel, model-byte scan or service change was executed. No GLM-5.3 Flash repository, model, service or process was touched. The DGX validation gap therefore remains an execution-path blocker, not a failed GPU test.

## Current continuation

The bounded implementation plan is `docs/superpowers/plans/2026-09-05-gb10x-ple-source-recovery.md`.

- Recover the native-M2 Safetensors exact-row source, pinned index mapper and `source-verify` command without its CUDA stack or CI metadata.
- Preserve main's CPU Cache Fabric / PLE-Hydra files unchanged.
- Add real miniature Safetensors-to-PLEPack overlay coverage (hot/cold rows and source-digest invalidation).
- Prevent PLEPack publication from replacing any existing output, especially its own immutable source or an alias.
- Reject corrupt hot bytes while opening a reader, using existing exact comparison rather than silently serving them.
- Reject the five known graph/representation variations ignored by the existing config parser, using the already pinned fixture without changing it.
- Keep local digest evidence separate from external checkpoint identity: `revision_contract` is a declared contract, and `remote_digest_match: null` is not a successful remote match.

Storage implementation reported 94 passing workspace tests, then one additional CLI regression was added for basename-relative output. Its scoped independent review is approved after that fix. The parser self-review additionally rejected duplicate JSON object members in both Safetensors headers and index files. Task 2 added four table-driven public-API tests covering the accepted fixture and all 15 independent invalid cases (five value mutations, five omissions, five wrong types). Its scoped review is approved with no Critical or Important issue. No full-model or hardware claim may be inferred from fixture tests.

### Fresh combined candidate gates

The controller rebuilt from an initially empty `CARGO_TARGET_DIR=/tmp/gb10x-final-validation.BVUGwd`, using the isolated official Rust 1.98.0 toolchain. This deliberately excludes stale artifacts from earlier implementation runs.

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Exit 0 |
| `cargo test --workspace --all-targets` | Exit 0; 99 passed, 0 failed, 0 ignored (main baseline: 70) |
| `cargo clippy --workspace --all-targets -- -D warnings` | Exit 0; no warnings |
| `cargo build --workspace --release` | Exit 0 |
| `git diff --check` | Exit 0 |

The combined test run includes source preservation under direct/alias/racing destinations, ordinary basename-relative CLI output, rejection of corrupted hot bytes on open, source provenance invalidation, duplicate-member rejection, and all five newly enforced model fields. Existing CPU Cache Fabric / PLE-Hydra tests also pass and their source files remain unchanged.

One early implementation run inconsistently accepted the corruption case after the reader fix. Instrumentation forced recompilation and subsequent tests passed; stale artifacts were only an inference, not a diagnosed root cause. The independent empty-target build above reproduces correct rejection without the prior compiled artifacts.

Final whole-branch review: specification and scoped engineering quality passed. No Critical or Important findings; no required fixes. One non-blocking coverage improvement remains: add nested duplicate-member JSON cases inside tensor descriptors and `weight_map`. The current recursive implementation was judged correct; existing regression cases cover root-level duplicates. This scoped approval does not establish full-engine or production readiness.

### Handoff state

The candidate is in `/workspace/scratch/8bb39fb50649/GB10X-ple-source` on `feature/ple-source-recovery-20260905`. Changes are local and uncommitted. No new commit, push, PR or merge was performed. The original checkout and all remote branches were preserved; the latest read-only ref check still reports main `e3ef3384`, M3 `0b0bcf8`, and native-M2 `27b6aee`.

The next native step is deliberate ABI/build reconciliation followed by actual GB10 execution when a usable terminal/node-agent path is available. Do not deploy this candidate as a completed inference engine.

### Subsequent PLE reference continuation

The next host increment is complete in the same local candidate. It obtained immutable official
Transformers code and model config, generated 736 row-ID expectations by executing the upstream
PyTorch hash, and established that the existing reduced config fixture matches every field it
retains from the official config. A signed-remainder mismatch was reproduced in two boundary tests
and corrected. Ordinary released-vocabulary cases already agreed before the correction.

The candidate now passes 105 tests, format, strict Clippy and release build. Independent scoped
review passes with no findings. The generated parameters are upstream constructor defaults, not
values read from checkpoint buffers. See [PLE reference evidence](2026-09-05-ple-reference.md)
for immutable inputs, the observed failures, reproduction and precise limits. The earlier 99-test
results above describe the completed storage-recovery checkpoint; they have not been substituted
for this increment's fresh results. Changes are still local and uncommitted.

## Remaining findings and production prerequisites

| Priority | Finding | Required next proof/work |
| --- | --- | --- |
| Closed in local candidate | Main's config parser ignored activation, attention bias, dtype, RoPE type and nested factor | All five are now parsed, required, typed and pinned; 15 independent invalid cases are rejected. This is not upstream authenticity proof or certification of every future graph option. |
| Closed in local candidate | PLE hash lacked independent outputs and used unsigned remainder after wrapping | Pinned upstream PyTorch vectors now cover all 16 heads, EOS, context restoration, transactions and overflow; the demonstrated signed-remainder bug is fixed. |
| High | Physical checkpoint hash buffers and complete model-byte identity remain unverified | Compare actual pinned tensor buffers with the reference-derived parameters and verify the complete source; numeric vectors do not establish checkpoint bytes. |
| High | No coherent integrated native ABI and no GB10 runtime result | Reconcile the two native implementations, then run native build, probe, smoke and numerical oracle checks on GB10. |
| Closed in local candidate, 2026-09-07 | Telemetry accepted incompatible exact-profile labels and zero integer performance metrics | Canonical BF16/none exact labels and positive supplied metrics are enforced; six new tests pass. Actual instrumentation remains open. |
| Medium | Overlay planner has no explicit byte/row cap | Deterministic admission under an explicit budget; do not claim a bounded production overlay today. |
| Medium | Read-only mmap does not prevent another process overwriting or truncating backing files | Explicit immutable source/sidecar lifecycle and deployment enforcement; current checks are not concurrent-mutation protection. |
| Medium | PLEPack v1 lacks full versioned model/tensor/dtype metadata and a complete manifest checksum | A separately specified format evolution; do not claim production provenance from a local digest alone. |
| Low | Safetensors filenames are checked lexically, but symlinks are followed | Document trusted local input boundary; decide containment versus legitimate shared checkpoint-blob links before claiming a filesystem sandbox. |
| Low | Native-M2's direct CUDA wrapper validates CC/geometry/name presence but does not itself require a GB10 name | The production CLI applies a later target check; settle a single fail-closed target authority when reconciling native code. |

Full forward execution, exact logits/tokens, attention/MoE kernels, model loading, CUDA graphs, end-to-end telemetry, performance baselines and optimizations remain future work. They are not unlocked merely by turning on the DGX.

## Decisions taken

1. Recover host-only PLE work from main rather than merge the divergent native branch. This preserves the merged CPU work and avoids the ABI collision; the cost is a later deliberate native reconciliation.
2. Include create-only publication and mandatory hot-row comparison. Correctness takes priority over overwriting old paths or fast opens; the cost is new output names and an up-front scan of hot source rows.
3. Keep this candidate local and preserve its uncommitted review record. No shared branch is modified; a later reviewed commit/PR is still needed.
4. Close the five fixture-defined validation omissions after storage review. The cost is stricter rejection of incomplete/noncanonical configs; this still does not establish upstream checkpoint authenticity.
