# Repository and worktrees

Live refs checked: 2026-09-07. GitHub PR #3 was read after publication; no
merge or force update occurred.

| Surface | Revision | Meaning |
| --- | --- | --- |
| main | `e3ef3384a20a3af640f418fa56ca32c62f7fc190` | M1 plus merged CPU M2 / PR #2 |
| M3, PR #3 | `ae403f64e3dab0321c54dd67c87b01354b1b21ce` | M3 ABI/probe plus documented native-M2 isolation |
| native-M2 | `27b6aee194f3028341f7b23ebae80548a8cb4e5f` | Separate legacy CUDA crate/kernel effort; incompatible ABI |
| Current local candidate | main base plus uncommitted edits | PLE recovery, safety, telemetry and domain memory |

## Worktree rules

- Active candidate: `/workspace/scratch/8bb39fb50649/GB10X-ple-source` on
  `feature/ple-source-recovery-20260905`. Preserve its uncommitted/new files.
- Original M3 checkout: `/workspace/scratch/8bb39fb50649/GB10X-local`, branch
  `feature/cuda-sm121a-abi-spec`, local HEAD `c498fd1`; preserve its pre-existing
  CMake and plan edits.
- Reconciliation checkout: `/workspace/scratch/8bb39fb50649/GB10X-m3-abi-reconcile`,
  local commit `4b0e912`; its tree exactly matches published M3 commit
  `ae403f64`. Its generated untracked Cargo.lock is not part of M3.
- Read `git status` before every write. Do not reset, clean, force-push or merge
  either native branch as a shortcut.

## Native branch boundary

M3 is canonical. Native-M2 exports the same probe symbol with a 304-byte
callee-sized output, while M3 exposes a 296-byte caller-sized output. The
source-layout proof and migration rules are in M3:
`docs/evidence/2026-09-07-native-abi-reconciliation.md` at `ae403f64`.

GitHub Actions `host-logic-ci` run #254 passed on M3 `ae403f64`. This is host
validation only. The PR remains open and unmerged pending actual GB10 evidence.
