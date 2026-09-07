# Native ABI and hardware

Latest health observation: 2026-09-07, MCP TO PC `host_status(gx10-e526)`
reports `online:false`, SSH unreachable and Tailscale unreachable. This is
controller evidence, not proof of power-off. The conversation exposes no
`host_terminal_run`; do not alter another project's services to reach GB10.

On 2026-09-05 the host was reachable but worker launch stopped before
execution because a ForgeAI node agent was absent. No GB10X native run resulted.

## Native ABI reconciliation

M3 is the canonical native ABI/build surface and is published in PR #3 at
`ae403f64` with CI run #254 green. Its header is
`cuda/common/include/gb10x_cuda_abi.h` and its build authority is `cuda/CMakeLists.txt`.

| Item | native-M2 | M3 |
| --- | --- | --- |
| Output size | 304 bytes | 296 bytes |
| Leading field | callee `abi_version` | caller `struct_size` |
| Name offset | 48 | 40 |
| Status | signed negative codes | stable `u32` values 0–4 |
| Build | Cargo `build.rs` | opt-in CMake |

Both branches define `gb10x_cuda_probe_device`. Linking M3's caller to the
native-M2 callee can write eight bytes past the output object. Do not merge,
cherry-pick or link `crates/gb10x-cuda` into M3. The audit is recorded in M3
at `docs/evidence/2026-09-07-native-abi-reconciliation.md`.

Native-M2's checksum smoke and BF16 RMSNorm are deferred source-level inputs
for a later approved kernel milestone; they are not M3 proof and have not been
integrated.

## Remaining hardware gates

Actual Linux aarch64 GB10 execution is still required for M3 CMake configure,
static-library/smoke build and CTest. Then come any separately designed
Rust-link, kernel, model-forward, logits/tokens and benchmark gates. The
GLM-5.3 Flash engine remains outside this project.
