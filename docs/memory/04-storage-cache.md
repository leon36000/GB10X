# Storage and CPU cache

Reviewed: 2026-09-07. Owners: `gb10x-ple` and `gb10x-runtime`.

- Main CPU M2 implements deterministic private-L2 ownership, discovered-capacity budgets, greedy
  saved-latency/byte admission, four-tier PLE-Hydra simulation and exact/experimental state separation.
  No measured CPU/GPU cache residency or real I/O scheduler is implemented by those simulations.
- Recovered host code adds BF16 Safetensors rows, a pinned 128-part model-index mapper and CLI
  `source-verify`. Logical row width: 160 BF16 elements / 320 bytes; total rows: 320001536.
- PLEPack v1 retains logical-row cold storage plus an exact hot overlay. Disk format is unchanged.
- `OverlayAdmissionBudget` now bounds deterministic hot admission by optional unique-row and
  block-padded data-byte caps. `plan_exact_layout` remains unbounded; the budgeted planner and
  `gb10x-plepack plan|build` flags are opt-in. Caps never change source values or cold fallback.
  All trace IDs remain fail-closed validated after a cap is reached. The byte cap excludes sidecar
  header/index bytes, which are separately reported; it is not a full-file quota.
- Publication is atomic create-only via hard link from an owned synced temp file. Existing files,
  sources, symlinks, hard links and race winners are never replaced. No force-overwrite option.
- Reader opening compares every hot row byte-for-byte against the source before exposing it.
  Cost: an upfront hot-source scan; source/sidecar files must remain immutable during mmap lifetime.
- Safetensors/header/index duplicate JSON members are rejected recursively. Tiny real integration
  fixtures prove hot/cold reads and invalidation when source bytes/provenance change.
- The hash-buffer verifier adds a non-mmap Safetensors path for three selected persistent PLE
  tensors: index and header reads are capped at 16 MiB each, only referenced shards are opened,
  and a successful check reads 280 payload bytes. It validates lexical relative `.safetensors`
  paths but intentionally retains the source verifier's symlink-following boundary.
- `revision_contract` is declared input, and `remote_digest_match: null` is not a remote match.
  Source validation is a local-byte observation, not full checkpoint authenticity proof.

Remaining: held-out trace and end-to-end promotion gate for any budget; deployment enforcement of
backing-file immutability; versioned full manifest metadata/checksum; decision on symlink containment
versus shared blob paths. Paths are checked for lexical escape but symlinks are deliberately followed.
A non-blocking test gap remains for nested duplicate JSON members inside tensor descriptors and
weight_map.

Evidence and design costs: [reconciliation](../evidence/2026-09-05-repository-reconciliation.md)
and [overlay budget](../evidence/2026-09-07-overlay-budget.md).
