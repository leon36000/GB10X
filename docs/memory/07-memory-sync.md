# Connected memory — every relaunch

Scope key: `gb10x`; updated: 2026-09-07. Never use another project's store.

## Current verified state

Revision 5 with bundle SHA-256
`8c78b1bea7602b6648be494b504ae5a2b2539667c0db186a24b72f3eccba84d1`
was written and read back exactly on 2026-09-07:

- Mongo current checkpoint and all nine fragmented source-archive records match.
- Neon current checkpoint matches on `project_key = 'gb10x'`.
- MCP TO PC `memory_search(query=GB10X, limit=10)` independently exposes the
  revision-5 marker and bundle hash.

Earlier revisions 1–4 may still appear in MCP search. They are historical
entries, never authority over the highest exact revision.

## Exact destinations

- MCP TO PC: `memory_search` and `memory_write`, reserved marker
  `GB10X-CANONICAL-MEMORY`. Use query `GB10X`, limit 10, then filter exact
  marker, repository, revision and bundle digest. Ignore NAOS, GLM and
  unrelated semantic hits.
- Neon: project `steep-paper-03426819`, branch
  `br-polished-flower-awene32m`, database `neondb`, table
  `public.gb10x_memory_state`, singleton `project_key = 'gb10x'`. See
  `neon-schema.sql`.
- MongoDB Atlas: project `6a9508bee9631f2b9e685733`, cluster
  `gb10x-memory`, database `gb10x_memory`, collection `checkpoints`, current
  key `_id: 'gb10x:current'`. A connection id is temporary and never part of
  memory.

## Procedure

1. Read Neon, Mongo and MCP entries, then read `00-resume.md`. Compare revision
   and digest with live Git/worktree/host state.
2. If stores disagree, do not overwrite a newer record. Update only affected
   domain Markdown with decisions, proof, blockers and next actions.
3. Build a canonical bundle containing exactly AGENTS.md and the eight domain
   notes. Hash canonical compact JSON with sorted keys and unescaped Unicode.
   `checkpoint.json` and `sync-receipt.md` remain outside that hash.
4. Store detailed bundle/checkpoint in Mongo, structured state in Neon and a
   concise canonical pointer in MCP TO PC. Use monotonic revision/hash guards.
5. Read back exact Mongo and Neon records and independently search MCP. Record
   actual results in `sync-receipt.md`; do not claim a write succeeded without
   readback.

The code-tree digest covers Cargo.toml/Cargo.lock and files under crates/, tests/
and scripts/. It identifies tested source rather than a published commit. The
receipt is excluded from the hashed bundle to prevent self-reference. This is a
relaunch procedure, not a background scheduler.
