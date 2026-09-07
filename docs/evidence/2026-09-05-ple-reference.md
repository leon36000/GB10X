# PLE hash reference evidence — 2026-09-05

## Outcome and scope

The Rust PLE hash now matches independently generated upstream row IDs for the recorded ordinary,
EOS, restored-context and speculative-history cases, and for signed-overflow stress cases. This
continues the local host candidate on `feature/ple-source-recovery-20260905`, based on main
`e3ef3384a20a3af640f418fa56ca32c62f7fc190`. No checkpoint embeddings or native code were changed.

The reference is the official Transformers `Qwen4ExpTextNGramEmbedding` implementation, pinned to
commit `f62dc9bf2c90353b442a56e74391fbb8c689b55e`. The official model config is pinned to
`34567a4712bc9766c4449e2e98e4468bfa24d915`. These are separate source revisions, not interchangeable
checkpoint identities.

## Immutable inputs

| Input | Bytes | SHA-256 |
| --- | --- | --- |
| [modeling_qwen4_exp.py](https://github.com/huggingface/transformers/blob/f62dc9bf2c90353b442a56e74391fbb8c689b55e/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) | 129822 | `2e36ee6a1bc4f43fa0434ae197c2de9eba27a71aec84ee1ff111dc3a812e8283` |
| [configuration_qwen4_exp.py](https://github.com/huggingface/transformers/blob/f62dc9bf2c90353b442a56e74391fbb8c689b55e/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py) | 16592 | `b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca` |
| [Official model config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/34567a4712bc9766c4449e2e98e4468bfa24d915/config.json) | 4745 | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |

The Git blob IDs of the two Python files are respectively
`99f6eca1656ffebbcb4ac2a9922ca41ccb47086a` and `16617898d2c3b8c7d143c7f87178b11ed2821e90`.
The existing reduced config fixture has SHA-256
`25d73369d5b2c87b1cdb0700d9dd4cc735414feb057d7079a218a99b33cb8d84`.
Every field present in that fixture matches the official JSON, including value and JSON type.
The fixture is a projection, not a byte-identical or complete copy of the official config.

## Oracle construction

`scripts/generate_ple_reference.py` checks all three upstream file digests before use. It extracts
the upstream constants, multiplier/prime builders and entire `Qwen4ExpTextNGramEmbedding` class
through Python's AST. It does not translate or rewrite the hashing methods.

The sole execution substitution is the embedding boundary: a tiny `nn.Module` returns its input
row IDs with a trailing dimension, rather than allocating or reading the model's embedding table.
Upstream EOS shifting, two-/three-token products, XOR, signed remainder, offsets and final row
selection execute unchanged. The upstream `forward` runs with `past_key_values=None`; a full
sequence is recomputed for each expected trace. The tests therefore establish the Rust window's
agreement with upstream full-history rows, not a test of Transformers' cache implementation.

Environment: Python 3.12.13 and the official CPU wheel `torch==2.8.0+cpu`, one PyTorch thread, Linux
x86_64. No NumPy operation is used. The installed CPU wheel emits a harmless initialization warning
when optional NumPy is absent; reference execution and deterministic regeneration both exit 0.
No GPU or huge embedding allocation is involved.

The default seed `1234` comes from the pinned upstream config class because the model config omits
it. The upstream constructor produces multipliers
`[23703573157769, 20109073645365, 8052911324071]` and 16 prime-sized tables starting at `20000003`.
These are **config-derived upstream defaults, not values read from checkpoint tensor buffers**.
Synthetic overflow metadata and tokens outside the released vocabulary are explicitly labelled
in the generator and fixture.

`tests/fixtures/ple-reference.json` records 46 token positions × 16 heads = 736 expected row IDs
across six traces. Its SHA-256 is
`d8d1c5e2f995482096468949a7dfa0126efe730e87934bf0db737935599f9574`.
Rust consumes these numeric expectations through the public `PleHashPlan` / `PleTokenWindow` API;
normal Rust tests need neither network access, Python nor PyTorch.

## Demonstrated defect and correction

The old code preserved multiplication/XOR bits using `u64` and then used unsigned `%`. Upstream
operates on `torch.int64` and applies `torch.remainder`, giving a nonnegative remainder for a
positive table size even when the mixed 64-bit integer is negative.

Before the correction, the new six-test integration suite reported **four passes and two failures**:

| Case, first token / first head | Old Rust row | Upstream row |
| --- | --- | --- |
| Synthetic signed overflow, token `0` | 9921939 | 11566328 |
| Default multipliers with token `4294967295` | 2464139 | 4108528 |

The fix retains wrapping product/XOR bits and computes `(mixed as i64).rem_euclid(table_size)`.
The constructor already bounds every positive table size by the u32 row space, so conversion to
i64 is safe. Public signatures, table layout, EOS handling and transactions are unchanged.

Ordinary tokens in the released vocabulary with the generated default multipliers already matched
before the fix; this evidence does not claim they suffered an observed inference failure. The
counterexamples exercise broader inputs accepted by the current Rust API.

After the fix, all six tests pass:

- ordinary default rows and EOS boundaries;
- signed i64 overflow;
- large u32 token boundaries;
- restored context at all 15 splits of the EOS trace;
- aborted draft compared with upstream recomputation;
- partial draft commit including EOS compared with upstream recomputation.

## Reproduction

Place the three files above in a directory, preserving their bytes and basenames. The exact raw
download URLs are also recorded under `provenance.sources` in the JSON fixture. Prepare an isolated
Python 3.12 environment with `torch==2.8.0+cpu` from `https://download.pytorch.org/whl/cpu`, then run:

```bash
python scripts/generate_ple_reference.py --reference-dir /path/to/pinned/reference
cargo test -p gb10x-core --test ple_reference
```

The generator defaults to comparing the existing fixture and fails on any digest, config
projection, dependency-version or output difference. `--write` deliberately regenerates the JSON
for review. It performs no downloads itself.

## Validation and boundaries

Fresh targeted Rust tests and a second PyTorch generation/check both passed. The root controller
also ran the complete candidate gates with the isolated Rust 1.98.0 toolchain and
`CARGO_TARGET_DIR=/tmp/gb10x-final-validation.BVUGwd`:

| Command | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Exit 0 |
| `cargo test --workspace --all-targets` | Exit 0; 105 passed, 0 failed, 0 ignored |
| `cargo clippy --workspace --all-targets -- -D warnings` | Exit 0 |
| `cargo build --workspace --release` | Exit 0 |
| `git diff --check` | Exit 0 |

The candidate previously had 99 passing tests; this increment adds six integration tests. No Rust
dependency was added for the reference vectors. All pre-existing storage, config and CPU-runtime
tests remain green.

Independent scoped review: Pass; no Critical, Important or Minor findings. The reviewer ran the
generator in check mode and all six Rust reference tests with `--locked --offline` in a separate
compilation target. The reviewer also compared the former unsigned calculation against the stored
vectors: 40 head-output mismatches on the u32-boundary trace and 104 on the synthetic-overflow
trace; none on the ordinary traces. These counts are reviewer evidence, separate from the root's
observed two failing tests before the correction and passing full suite after it.

This establishes upstream implementation conformance for the recorded vectors and an exact
projection relationship for the config fixture. It does not establish physical checkpoint buffer
identity, exact embedding bytes, the completeness of all model graph validation, linked native ABI
correctness, GPU execution, logits, generated tokens, or performance. Those remain separate gates.

All changes remain local and uncommitted. No push, PR update, merge, DGX command or GLM-5.3 Flash
change was performed in this continuation.
