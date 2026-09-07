# Scope and decisions

Reviewed: 2026-09-07. Owner: project-wide contract.

- GB10X is a dedicated Qwen3.8-Flash-Next inference-engine project for NVIDIA GB10 / DGX Spark,
  Linux aarch64, strict `sm_121a`. Correct end-to-end tokens/s is the objective.
- The approved design is [Cache-First v3](../superpowers/specs/2026-08-31-gb10x-cache-first-v3-design.md)
  with the [PLEPack overlay amendment](../superpowers/specs/2026-08-31-gb10x-plepack-overlay-amendment.md).
- Exact mode preserves the pinned source representation. Non-source numerical changes belong to
  an explicitly labelled experimental mode with independent correctness evidence.
- Host-only simulation and schema tests are useful but do not establish hardware behavior.
- The GLM-5.3 Flash engine, Qwen27B training/datasets, PC4-Fusion and AgentOS are separate projects.
  Do not touch their repositories, models, processes or stored memories.
- Local recovery preserves CPU M2 and avoids blindly merging the incompatible native branches.
- The current candidate remains uncommitted and unpublished. Memory synchronization is explicitly
  authorized by the user; do not turn a memory update into a software deployment or Git merge.
- Keep dependencies isolated in this execution environment. Do not install tools onto PC1/DGX
  hosts or modify their services for routine local validation.
- User instruction, 2026-09-07: concise domain Markdown memory plus read/update of MCP TO PC,
  Neon and MongoDB Atlas at every relaunch; omit noise, duplication and unsupported claims.
