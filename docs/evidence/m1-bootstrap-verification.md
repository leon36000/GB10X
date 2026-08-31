# GB10X Milestone 1 Verification

Status: host-independent bootstrap verified; real GB10/CUDA execution not yet verified.

## Verification anchor

Code commit verified by GitHub Actions: `2b8e1403b3e986c9819ad8953b346f5e6e777256`.

Verification environment:
- GitHub-hosted Ubuntu 24.04.x runner;
- host target `x86_64-unknown-linux-gnu`;
- Rust `1.98.0`;
- no DGX Spark / GB10 hardware was used by this gate.

Fresh CI gates on the verification anchor:
- `cargo fmt --all -- --check` — PASS;
- `cargo test --workspace --all-targets` — PASS, zero failures;
- `cargo clippy --workspace --all-targets -- -D warnings` — PASS;
- `cargo build --workspace --release` — PASS.

The CLI integration suite also executed the produced `gb10x-probe` and `gb10x-plepack` binaries. It verified help/argument handling, exact PLEPack build/verify, source-digest rejection after source mutation, and JSON output contracts.

## Host-independent behavior verified

- fail-closed GB10 platform validation logic;
- Linux CPU/cache topology parsing without hard-coded cache sizes;
- exact Qwen3.8-Flash-Next configuration contract used by GB10X;
- exact PLE hashing, EOS reset and speculative rollback/commit-prefix semantics;
- exact PLEPack locality planning with no global cold-row remap index;
- mmap `RawFileRowSource` with SHA-256 source provenance;
- atomic PLEPack hot-overlay writer;
- mmap PLEPack reader with source/index provenance validation;
- byte-for-byte verification of every physically stored hot-overlay row against the immutable source;
- strict telemetry/evidence record validation;
- `gb10x-probe --json` CLI contract;
- `gb10x-plepack plan`, `build` and `verify` CLI contracts.

## PLEPack exactness boundary

PLEPack is a hot-overlay sidecar, not a second copy of the full PLE. Cold logical rows remain exclusively in the immutable source and therefore are not duplicated for sidecar verification. `verify` validates the exact source digest/geometry, the complete hot index, and every row physically duplicated into the hot overlay byte-for-byte against that source.

M1 uses a prepared flat exact-row source for executable CLI verification. Direct production binding to the model's safetensors shards is deliberately deferred to the native/model-loading phase.

## Explicitly not proven by M1

The following claims remain unverified and MUST NOT be inferred from this milestone:
- execution on DGX Spark;
- `aarch64` runtime behavior on GB10;
- real GPU identity or CUDA Compute Capability 12.1;
- CUDA toolchain compatibility or `sm_121a` compilation/execution;
- CUDA kernels, Tensor Core paths, CUDA Graphs or GPU L2 controls;
- production safetensors PLE source binding;
- complete Qwen inference;
- tokens/second, latency, bandwidth or any performance improvement.

## First real DGX Spark gate

Build the release workspace on the DGX Spark and run:

```bash
uname -m
./target/release/gb10x-probe --json
nvidia-smi
nvcc --version
```

Native CUDA work is unblocked only after the evidence confirms:
1. Linux `aarch64`;
2. NVIDIA GB10 / DGX Spark target identity;
3. CUDA Compute Capability 12.1 from a native CUDA-capable probe added in the native milestone;
4. a usable CUDA toolkit on the server.

Until that hardware gate exists, all CUDA/GB10-native claims remain `UNVERIFIED`.

## Performance rule

Milestone 1 authorizes no performance claim. Every later optimization must preserve correctness first and be promoted only by end-to-end evidence on the actual GB10 server.
