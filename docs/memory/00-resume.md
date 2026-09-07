# GB10X — read first

Updated: 2026-09-07. Memory scope: **gb10x**. Repository:
**leon36000/GB10X**.

## Current state

- Active checkout: `/workspace/scratch/8bb39fb50649/GB10X-ple-source`.
  Branch: `feature/ple-source-recovery-20260905`; base HEAD: `e3ef3384`.
- Local uncommitted candidate: Safetensors source recovery, PLEPack safety,
  stricter config validation, signed PLE remainder fix, upstream vectors,
  telemetry validation, bounded overlay admission and bounded physical PLE
  hash-buffer verification.
- Last complete candidate gate: 127 tests, format, strict Clippy, fresh-target
  release build and diff check; 2026-09-07.
- M3 ABI reconciliation is published in PR #3 at `ae403f64`; GitHub Actions
  `host-logic-ci` run #254 passed. M3 is still unmerged and has no GB10-native
  compilation or device-smoke evidence.
- M3 is the sole public native ABI/build authority. Do not merge or link
  `feature/gb10x-native-m2`: its same-named probe uses a 304-byte output where
  M3 uses 296 bytes, which could overrun an M3 caller.
- DGX `gx10-e526` is currently reported offline, with SSH and Tailscale
  unreachable by MCP TO PC (2026-09-07). No GPU/native execution occurred.
- No full engine, checkpoint-byte, GPU, logits or performance proof exists.

## Next

1. Run `hash-verify` on an actual mounted pinned Qwen checkpoint without
   loading the full model.
2. When the DGX is reachable, run the M3 CMake configure/build/CTest gate on
   GB10 only and record the observed evidence.
3. Define the held-out trace and end-to-end gate needed before any overlay
   promotion. Preserve all uncommitted recovery work.

## Read only the needed domain

| Need | Note |
| --- | --- |
| Scope and binding choices | [01-scope-decisions.md](01-scope-decisions.md) |
| Branches and preservation | [02-repository.md](02-repository.md) |
| Model contract and PLE oracle | [03-model-ple.md](03-model-ple.md) |
| Storage and CPU cache | [04-storage-cache.md](04-storage-cache.md) |
| Native ABI and DGX gates | [05-native-hardware.md](05-native-hardware.md) |
| Tests, toolchain and telemetry | [06-validation-telemetry.md](06-validation-telemetry.md) |
| Connected memory and readback | [07-memory-sync.md](07-memory-sync.md) |

On each relaunch: read this page and the connected current checkpoint; compare
revision and content digest, verify live Git/host state, do the smallest useful
next increment, update affected notes, then synchronize and read back all three
connected memories. Older memory is evidence of its own date/branch, never
authority to overwrite newer work.
