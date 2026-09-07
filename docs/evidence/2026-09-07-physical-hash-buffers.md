# Bounded physical PLE hash-buffer verification

Date: 2026-09-07. Scope: the three persistent PLE hash buffers for pinned
`Qwen/Qwen3.8-Flash-Next` revision
`34567a4712bc9766c4449e2e98e4468bfa24d915`.

## Contract implemented

`gb10x-ple` now exposes `verify_qwen38_ple_hash_buffers`. It accepts a caller-supplied observed
revision only when it equals the GB10X pin, parses `model.safetensors.index.json` with duplicate
JSON-member rejection, and resolves exactly these state-dict entries:

- `model.language_model.layers.1.ple.ple_embedding.layer_multipliers` — `I64 [3]`;
- `model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes` — `I64 [16]`;
- `model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets` — `I64 [16]`.

Their expected values are the upstream-constructor defaults already recorded in
`tests/fixtures/ple-reference.json`. Values are decoded as little-endian signed 64-bit integers and
must match exactly. A mismatch, missing entry, wrong type/shape/length, duplicate JSON member,
unsafe path or oversized metadata fails closed.

The reader uses `File::read_exact` and does not mmap the selected files. It reads no more than
16 MiB for the index, 16 MiB of header plus its 8-byte prefix for each referenced Safetensors file,
and exactly `3 * 8 + 16 * 8 + 16 * 8 = 280` payload bytes after a successful check. File paths are
lexically constrained to non-empty relative `.safetensors` paths; symlinks remain intentionally
followed, matching the existing source verifier.

`gb10x-plepack hash-verify --model-dir <dir> --observed-revision <sha>` prints those local read
counts and labels its state `verified-local-hash-buffers`. It deliberately does not print a remote
digest match or claim full checkpoint identity, model inference, GPU execution, logits or
performance.

## Test evidence

The first library test was run before implementation and failed with the expected unresolved
`verify_qwen38_ple_hash_buffers` import. The first CLI test was run before adding the command and
failed because `hash-verify` was absent from help.

Focused checks after implementation:

```text
cargo test --locked -p gb10x-ple --test qwen38_hash_buffers
7 passed, 0 failed

cargo test --locked -p gb10x-tools --test cli
9 passed, 0 failed

cargo clippy --locked -p gb10x-ple -p gb10x-tools --all-targets -- -D warnings
passed
```

The hash-buffer fixture accepts the exact values and reports three buffers, one file and 280 payload
bytes. It rejects a changed multiplier, duplicate index member, escaping shard target and a declared
header above the 16 MiB bound. A separate manual CLI fixture reported 372 index bytes, 416
prefix-plus-header bytes and 280 payload bytes.

## Current boundary

`/workspace/scratch/8bb39fb50649/gb10x-reference` contains the pinned config and upstream
Transformers source, but no `model.safetensors.index.json` or checkpoint shard. The new code is
therefore verified against a real-format tiny fixture only. No physical published Qwen checkpoint
has been compared yet; running `hash-verify` on a supplied local pinned checkpoint is the next
small proof.
