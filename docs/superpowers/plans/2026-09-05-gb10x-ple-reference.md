# PLE reference conformance — 2026-09-05

## Scope

Continue the reviewed local host candidate by resolving the audit's missing independent PLE hash
evidence. Preserve the existing recovery changes and the original M3 checkout. No native code,
remote publication, model serving, or GLM-5.3 Flash work is included.

## Reference and acceptance contract

- Pin the official model config to `34567a4712bc9766c4449e2e98e4468bfa24d915` and the official
  Transformers implementation to `f62dc9bf2c90353b442a56e74391fbb8c689b55e`.
- Check exact reference-file digests before executing the upstream hash methods. Establish that
  the existing reduced config fixture is a projection of the official config.
- Execute the upstream NGramEmbedding hash on CPU with PyTorch. Replace only its large embedding
  lookup with a row-ID recorder; preserve its shifting, multiplication, XOR, remainder and offsets.
- Generate deterministic all-16-head vectors for ordinary tokens, EOS boundaries, and signed
  64-bit overflow. Distinguish config-derived default metadata from bytes loaded from a checkpoint.
- Consume the committed vectors through the Rust public API with no Python/network test dependency.
  Cover restored contexts, speculative abort and partial commit against upstream full-sequence rows.
- Correct only demonstrated mismatches; show a failing regression before the fix if one occurs.
- Run focused and complete host gates and record evidence and limitations.

## Decisions

The reference source is downloaded separately and digest checked, not copied into production code.
Tests retain generated numeric vectors and provenance. This keeps the Rust suite offline and small;
regeneration needs the three pinned reference files and the documented CPU PyTorch environment.

The metadata generator in upstream Transformers is an independent algorithm oracle. It does not
establish that the physical checkpoint buffers agree with those defaults. No full checkpoint,
embedding weights, GPU, logits, or performance evidence is claimed.

## Work

- [x] Obtain pinned upstream code/config and compare the existing config projection.
- [x] Add reproducible upstream vector generator and offline public-API tests.
- [x] Diagnose and fix any concrete mismatch.
- [x] Run host validation and independent review; update the audit handoff.

Result: two failing signed-overflow regressions were reproduced and corrected; all six new tests
pass. Complete candidate: 105 tests, format, strict Clippy and release build pass. Independent
review passes with no findings. See `docs/evidence/2026-09-05-ple-reference.md` for exact inputs,
regeneration commands and remaining checkpoint/native boundaries.
