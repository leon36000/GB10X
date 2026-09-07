# GB10X memory synchronization receipt

Date: 2026-09-07. Current revision: **6**.

Bundle SHA-256: `b3fa49001cc4df5bb9cd672d41a1659a6c123bdc7a95038678e9f8d8b12dc3af`.
Tested code-tree SHA-256: `caffcdf84e29566e6bb802022cd05dbed925d122b042cb8190222b3a6ae693ab`.
The local bundle and 76-file code manifest were recomputed before writing.

| Backend | Write and readback state |
| --- | --- |
| MongoDB Atlas | `gb10x_memory.checkpoints`, `_id:gb10x:current`; revision/hash, full checkpoint and all nine domain Markdown documents compare exactly. The revision-6 archive manifest and each of its nine fragments also compare exactly. |
| Neon | `public.gb10x_memory_state`, `project_key='gb10x'`; revision/hash and full structured checkpoint compare exactly. Readback timestamp: `2026-09-07T10:46:36.267Z`. |
| MCP TO PC | Write created `memory_5b877c8ef955454a9f3516a26c57189c` and event `mevent_6a037ac89f4f4b16b595c896aa19059c`; both MCP and RAG responses report `synced`. The immediate `memory_search(query=GB10X, limit=10)` did not independently expose revision 6, so search visibility remains pending. Revision 5 had become visible earlier in this session; this is treated as index lag. |

## Recoverable source backup

The full uncommitted candidate, excluding this self-referential receipt, is
preserved in Mongo manifest
`gb10x:source-overlay:d90b4c96d601b6e08c8493f67285f1a1e4f9d0477fe11d9f69cb3cef5a6799bc`
and nine ordered base64 fragments.

- Base commit: `e3ef3384a20a3af640f418fa56ca32c62f7fc190`.
- Compressed bytes: 75344; decoded patch bytes: 286392.
- Archive SHA-256: `d90b4c96d601b6e08c8493f67285f1a1e4f9d0477fe11d9f69cb3cef5a6799bc`.
- Patch SHA-256: `f62c0c9e311dc3ef4a0699c7d7c9a024a6f796031e8ab8c107c3feb8e4f7b366`.
- An isolated-base `git apply --cached --check --whitespace=error-all` passed
  before archive creation. Mongo readback compared the manifest and all nine
  fragments byte-for-byte.

This receipt is outside the hashed memory bundle and excluded from the source
overlay to avoid self-reference. It records synchronization state, not a Git
publication or hardware proof.
