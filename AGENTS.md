# GB10X continuity

Start every resumed session with `docs/memory/00-resume.md`, then read only the domain notes
needed for the current task. Check live Git/worktree/tool state before trusting a saved checkpoint.

At the user's request, read and update the GB10X memory through MCP TO PC, Neon and MongoDB Atlas
on each relaunch. Exact destinations and the read/write/readback procedure are in
`docs/memory/07-memory-sync.md`. Reuse existing GB10X stores. Report an unavailable connector
truthfully; keep a pending sync record and continue useful local work. Never claim a successful
sync from an attempted write alone.

Update the domain notes and current checkpoint after a meaningful verified change. Keep only
decisions, state, proof references, blockers and next actions. Keep source evidence in
`docs/evidence/`; do not duplicate transcripts or raw tool inventories into active memory.

Scope: `leon36000/GB10X`, Qwen3.8-Flash-Next and GB10 only. The separate GLM-5.3 Flash engine,
Qwen27B training projects and AgentOS are not this project. Preserve the existing local edits.
Do not merge the two native branches without reconciling their incompatible probe ABI.
