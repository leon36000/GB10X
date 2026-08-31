# GB10X Milestone 1 — Host Logic Evidence

Status: **HOST-LOGIC GREEN**

Authoritative marker: `.ci/results/latest-green.txt`

The marker is published only when all three host-independent gates pass for the source commit:

- `cargo fmt --all -- --check`
- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`

A red run removes the green marker.

## Verified in M1

- fail-closed GB10 platform contract: Linux `aarch64`, NVIDIA GB10 identity, compute capability 12.1;
- Linux CPU/cache topology parsing with physical-cache de-duplication;
- Linux host probe for kernel, CPU identity, online CPUs, RAM and base page size;
- exact pinned Qwen3.8-Flash-Next architecture contract (`qwen4_exp` / `qwen4_exp_text`);
- cache-critical Qwen shape derivation: 16 PLE rows/token, 5,120 useful BF16 PLE bytes/token, 4 MiB selected BF16 QSA K+V;
- exact Qwen PLE hash with wrapping arithmetic, EOS reset, speculative begin/abort/commit-prefix semantics and property-based row-bound testing;
- scalable PLEPack layout: official source remains cold truth, metadata scales only with hot rows;
- atomic exact PLEPack hot sidecar with source digest, index SHA-256, strict structural checks, mmap hot reads and exact cold fallback;
- strict benchmark evidence schema separating exact and experimental-approximate runs;
- `gb10x-probe` host-fact CLI;
- `gb10x-plepack plan` deterministic hot-overlay planning CLI.

## Explicitly not proved by M1

M1 runs on GitHub's host-independent CI and therefore makes **no** claim that any CUDA kernel or model inference path has executed on GB10 hardware.

Still pending native GB10 proof:

- CUDA runtime/device probe and `sm_121a` binary/cubin gate;
- discovered GPU L2 / persisting-L2 limits on the actual appliance;
- NVFP4/FP8 Tensor-Core kernels;
- QSA/GDN/MoE/MTP execution;
- CUDA Graphs, TMA, Copy Engines and L2 access-policy tuning;
- ARM worker/cache residency tuning on the real X925/A725 topology;
- MPAM/resctrl availability on the installed firmware/kernel;
- safetensors-backed production PLE cold source;
- full Qwen3.8-Flash-Next inference correctness;
- tokens/s, TTFT, bandwidth, power or tokens/joule performance claims.

## Production rule

No optimization enters the exact path without:

`correctness gate -> controlled A/B -> end-to-end win -> evidence record`
