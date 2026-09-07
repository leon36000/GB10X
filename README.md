# GB10X

GB10X is an experimental inference engine designed exclusively for NVIDIA GB10 / DGX Spark (`Linux aarch64`, target `sm_121a`) and Qwen3.8-Flash-Next.

The project optimizes for correct end-to-end tokens/second on a dedicated GB10 server. It treats CPU caches, GPU caches, unified memory, Tensor/CUDA compute, speculative decoding, storage, and Linux memory policy as one coordinated inference system.

## Status

Agent/session continuation starts at [the GB10X memory index](docs/memory/00-resume.md).

This local host candidate builds on main's correctness and storage foundation. Its Rust tests exercise
contracts, schemas, parsers, synthetic fixtures and pinned upstream PLE hash vectors on an x86_64 host; those tests do
**not** constitute GB10, CUDA, full-checkpoint or inference validation.

Implemented and covered by host tests:

- fail-closed GB10 platform-contract and Qwen model-contract schemas;
- Linux CPU/cache topology probing;
- PLE hash and transaction behavior checked against pinned upstream PyTorch row-ID vectors,
  including EOS, restored context and signed-overflow arithmetic;
- exact PLEPack hot-overlay planning with explicit row/data-byte admission caps, and mmap reading;
- create-only PLEPack publication: output paths must be new and are never replaced;
- mandatory byte-for-byte verification of every hot row during `PlePackReader::open`;
- mmap raw-row and Safetensors PLE sources with SHA-256 provenance;
- telemetry/evidence schema checks, including canonical exact-mode BF16/none labels and positive
  supplied performance measurements; operational telemetry is not yet validated;
- `gb10x-probe --json`;
- `gb10x-plepack plan|build|verify|source-verify|hash-verify`;
- workspace release builds.

The host-only Safetensors source and verifier were recovered from commit
`27b6aee194f3028341f7b23ebae80548a8cb4e5f`. The recovered path builds the exact pinned
Qwen3.8-Flash-Next PLE index mapping, enforces BF16 `[rows, 160]` geometry, reads rows across
physical tensor parts, and hashes the manifest contract plus referenced PLE bytes.
`source-verify` reports this as a local-byte observation with `revision_contract` and
`remote_digest_match: null`; it does not prove remote checkpoint identity.

`hash-verify --model-dir <dir> --observed-revision <sha>` is a smaller, fail-closed physical
check for the three persistent buffers that determine the Qwen PLE hash. It accepts only the
pinned revision, reads at most 16 MiB of the safetensors index, 16 MiB of header per referenced
file, and exactly 280 tensor payload bytes on a successful verification. It does not mmap a shard
or validate the remaining checkpoint. A success is local agreement of those selected buffer bytes,
not proof of remote provenance, a complete checkpoint, inference, GPU execution or logits.

Storage preconditions and costs are explicit:

- raw sources, Safetensors shards and published sidecars must remain immutable throughout their mmap
  lifetimes;
- model-file paths are checked for lexical non-escape (no absolute paths or `..`), while symlinks
  are followed so shared checkpoint layouts remain usable; this is not filesystem containment;
- opening a sidecar reads every hot source row once to reject corrupted hot data before a public
  exact read;
- cold rows remain in the immutable source and hot rows are duplicated into the sidecar.
- `gb10x-plepack plan` and `build` accept optional `--max-hot-rows` and
  `--max-overlay-bytes`; the latter caps block-padded duplicated-row data while header/index bytes
  remain separately reported.

Still unverified:

- the real pinned Qwen checkpoint bytes and independently established checkpoint provenance;
- physical checkpoint hash-buffer agreement with the upstream constructor defaults used by the
  reference vectors; `hash-verify` is available for a supplied local checkpoint, but this workspace
  contains only the pinned config/source reference and no safetensors index or shard to check;
- execution on a real DGX Spark / GB10 and `aarch64` target behavior;
- CUDA Compute Capability 12.1 or `sm_121a` compilation/execution;
- full Qwen inference, end-to-end correctness or any performance claim.

See [repository reconciliation](docs/evidence/2026-09-05-repository-reconciliation.md) for the
current audit boundary, [PLE reference evidence](docs/evidence/2026-09-05-ple-reference.md) for
the signed-remainder correction, independent vectors and their limits,
[M1 bootstrap evidence](docs/evidence/m1-bootstrap-verification.md) for the
original foundation, and the
[approved architecture](docs/superpowers/specs/2026-08-31-gb10x-cache-first-v3-design.md).

## First DGX Spark gate

Before native CUDA work is considered verified, run on the actual server:

```bash
uname -m
./target/release/gb10x-probe --json
nvidia-smi
nvcc --version
```

The native milestone remains fail-closed until the server evidence confirms Linux `aarch64`, the GB10 target, CUDA Compute Capability 12.1 through a native CUDA-capable probe, and a usable CUDA toolkit.

## Non-goals

- No x86 production support.
- No non-GB10 NVIDIA GPU support.
- No AMD/Metal backend.
- No generic multi-model abstraction in the initial engine.
- No optimization is accepted without a correctness gate and an end-to-end A/B benchmark.
