# Validation and telemetry

Updated: 2026-09-07.

## Candidate host checkpoint

The PLE recovery candidate passed 127 tests with zero failures/ignores, format,
strict Clippy, a fresh-target release build and diff check. Rust was 1.98.0.
Use the isolated toolchain rooted at `/tmp/gb10x-validation-20260907` when it
still exists; verify it before reuse. These tests do not establish hardware,
checkpoint identity, forward correctness or throughput.

The candidate covers exact-profile/positive telemetry validation, PLE overlay
row and padded-byte caps, and physical hash-buffer fixture validation. Evidence:

- [telemetry and continuity](../evidence/2026-09-07-telemetry-memory.md)
- [overlay budget](../evidence/2026-09-07-overlay-budget.md)
- [physical hash buffers](../evidence/2026-09-07-physical-hash-buffers.md)

## M3 reconciliation checks

The M3/native-M2 ABI audit passed two independent C11 header probes:
M3 `gb10x_cuda_device_info` is 296 bytes with name offset 40; native-M2
`gb10x_cuda_device_info_v1` is 304 bytes with name offset 48. The M3 source
contains no legacy native-M2 ABI identifiers or crate. Its focused Rust ABI
contract passed 5 tests, and GitHub Actions `host-logic-ci` run #254 passed on
published commit `ae403f64`.

CMake is unavailable on this host, so disabled-CMake and GB10-native CMake
build/CTest remain unrun. No CUDA compiler, GPU, inference path, model load,
logits or benchmark is claimed.
