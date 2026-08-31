# GB10X Cache-First v3 — Design

## Status

Approved architecture. This document is the source of truth for the initial GB10X implementation.

## Mission

Build a local LLM inference engine for exactly one deployment target:

- NVIDIA GB10 / DGX Spark
- Linux `aarch64`
- CUDA `sm_121a`
- dedicated-server operation
- Qwen3.8-Flash-Next only

The objective is **maximum correct end-to-end tokens/second**, not maximum nominal utilization. Every useful GB10 resource is eligible for exploitation, but an optimization stays in the production path only when it improves a controlled end-to-end benchmark without violating correctness.

## Primary invariant

`correctness -> measured A/B win -> production`

No optimization enters the default path because it is theoretically elegant, raises utilization, or wins an isolated microbenchmark while making end-to-end inference slower.

## Dedicated-server assumption

The DGX Spark is treated as an appliance:

- GB10X and model assets are the only substantial workloads.
- CPU cores may be isolated for GB10X.
- IRQ placement, RCU offload, huge pages, page-cache policy, scheduler affinity, and kernel tuning may be optimized aggressively.
- No design effort is spent preserving interactive desktop responsiveness.
- Linux services that are not required for safe operation are considered noise and should be moved away from compute cores or disabled by deployment policy.

## Non-goals

Initial GB10X does not support:

- x86-64
- other NVIDIA GPUs
- AMD GPUs
- Metal
- generic model loading
- arbitrary Hugging Face architectures
- generic quantization formats
- distributed inference
- production use of RT cores unless the experiment passes exactness and speed gates

These exclusions are deliberate. Specialization is the optimization strategy.

## Model contract

The initial target is Qwen3.8-Flash-Next text inference.

The engine treats the released model structure as compile-time knowledge wherever possible. The validated configuration includes:

- hidden size: 2560
- 48 transformer layers
- repeating four-layer schedule: `GDN, GDN, GDN, QSA`, repeated 12 times
- 24 query heads
- 2 KV heads
- head dimension: 256
- 512 routed experts
- top-k experts per token: 10
- expert intermediate dimension: 640
- shared expert intermediate dimension: 640
- PLE n-gram size: 3
- PLE heads per n-gram order: 8
- 16 selected PLE rows per token
- PLE embedding width: 2560 total / 160 per selected row
- sparse-attention budget: 2048 tokens
- one MTP hidden layer in the released configuration
- maximum model position envelope: 262144 tokens

The loader must reject a checkpoint whose immutable architecture fields do not match this contract.

## Hardware contract

GB10X probes and records the real machine at startup, then fails closed if the platform is not compatible.

Required signals include:

- `uname -m == aarch64`
- NVIDIA GB10 device identity
- compute capability 12.1
- CUDA toolkit/runtime versions compatible with native `sm_121a`
- unified-memory capacity
- GPU L2 size and persistence limits
- CPU topology and cache topology discovered from sysfs
- available ARM ISA features
- storage devices and direct-I/O support

Published hardware values are defaults for expectation and diagnostics, never substitutes for runtime probing.

## Architecture

```text
Qwen3.8-Flash-Next
        |
        v
GB10X scheduler
        |
        +-----------------------+------------------------+
        |                       |                        |
CPU Cache Fabric          GPU Cache Fabric          Memory Fabric
        |                       |                        |
X925 ownership shards     registers/shared          coherent UMA
L2/L3 hot objects         L1/shared carveout        huge pages
L3Draft                   L2 persistence            Linux page cache
ExpertScout               constant memory           io_uring/O_DIRECT
PLE-Hydra                 QSA-L2                    NVMe cold PLE
        |                       |                        |
        +-----------------------+------------------------+
                                |
                         Compute Fabric
                                |
                    CUDA + Tensor Cores
                         `sm_121a` only
                                |
                   four-layer supergraph x12
```

## Runtime language split

### Rust

Rust owns:

- server/runtime state
- model manifest validation
- scheduler
- CPU topology discovery
- cache policy
- memory planner
- PLE-Hydra metadata
- io_uring orchestration
- telemetry
- autotuning decisions
- correctness harness
- OpenAI-compatible serving surface after the inference core is correct

### CUDA C++ / CuTe

CUDA owns:

- NVFP4/FP8 matrix operations
- GDN kernels
- QSA kernels
- MoE routing/packing/expert math where GPU-resident
- fused normalization/activation/residual paths
- KV operations
- MTP target verification
- on-device sampling
- cache-policy-sensitive kernels

CuTe/CUTLASS is used selectively. A generic library call is not preferred over a custom kernel merely because it is mature; whichever path wins the correctness-gated benchmark stays.

### ARM CPU kernels

ARM-specific code may use:

- SVE/SVE2
- NEON
- DotProd/I8MM where useful
- KleidiAI
- NVIDIA NVPL
- hand-written intrinsics where benchmarked

The CPU is not restricted to control-plane work.

## Cache Fabric

Cache residency is a first-class scheduling concern.

Every important object is tagged with an access class:

- `UltraHot`: tiny, repeatedly consumed within a token/superblock
- `Hot`: high near-term reuse probability
- `Warm`: reusable across turns or likely future tokens
- `Streaming`: large one-pass data such as most expert-weight traffic
- `Cold`: NVMe-backed data with low near-term reuse

A cache object also carries:

- byte size
- owner/consumer
- predicted reuse distance
- miss cost
- eviction cost
- measured hit rate
- exactness class

The cache governor maximizes saved latency per byte rather than cache occupancy.

## CPU Cache Fabric

### Topology discovery

Never hard-code cache sharing. At startup, inspect Linux sysfs cache descriptors and derive:

- private L1/L2 relationships
- shared-cache groups
- cache sizes
- cache line size
- CPU IDs sharing each cache

### Core roles

On a dedicated server, GB10X starts with an aggressive role policy:

- Cortex-X925-class cores: compute/cache owners
- Cortex-A725-class cores: I/O, HTTP, telemetry, IRQ-facing and background service work

Actual core IDs are derived from topology/benchmark probes, not assumed numbering.

### Ownership sharding

Hot CPU structures use single-owner sharding wherever possible. A key is mapped to one compute worker so repeated accesses return to the same private cache.

Targets include:

- PLE result cache
- L3Draft state
- ExpertScout state
- small speculative metadata

Cross-core mutation is minimized. Shared counters and queues use cache-line separation to avoid false sharing.

### Linux controls

GB10X probes and uses, when available and beneficial:

- cpusets
- explicit thread affinity
- IRQ affinity
- `nohz_full` deployment profile
- RCU callback offload
- MPAM/resctrl monitoring or partitioning
- perf/PMU counters

MPAM/resctrl is opportunistic. Missing firmware exposure must not break the engine.

## GPU Cache Fabric

### L2 persistence

GB10X queries actual L2 and persisting-cache limits at runtime.

The initial L2 priority order is:

1. QSA selected K/V working set
2. recurrent GDN state
3. small quantization/scaling metadata
4. router/indexer metadata
5. experimentally selected hot expert tiles

Large weight streams default to streaming/evict-first behavior and must not destroy high-reuse state.

### QSA-L2 Residency

The sparse-attention selected K/V set is a primary residency candidate. For the fixed Qwen dimensions, the BF16 selected K+V payload for 2048 tokens is approximately 4 MiB, small enough to fit repeatedly inside the GB10 GPU L2 envelope.

The QSA path should therefore:

- select candidate positions
- stage/coalesce exact selected K/V
- assign persistence-friendly cache policy
- reuse across the 24 query heads
- benchmark alternate selected-set tile sizes

### L1/shared carveout

Kernel families receive independent carveout preferences.

Examples:

- QSA tile kernels favor shared memory when the tile fits productively.
- irregular router/index kernels may favor L1.
- custom GEMM kernels use measured carveouts.

No global carveout policy is assumed optimal.

### Constant memory

Small immutable broadcast metadata should use constant memory where practical, including fixed model dimensions, layer schedule, hash multipliers, small quantization metadata, and dispatch constants.

## Four-Layer Supergraph

The repeating architecture unit is the primary scheduling unit:

`GDN -> GDN -> GDN -> QSA`

GB10X creates one highly tuned superblock execution schedule and reuses it 12 times with layer-specific pointers/metadata.

The design objective is to keep the 2560-wide current hidden state on-chip across as many adjacent elementwise/control operations as register/shared-memory pressure allows, while streaming only the large weights from unified memory.

CUDA Graph capture is applied to stable decode/verify paths after correctness is established.

## PLE-Hydra

The PLE/n-gram subsystem is a major model-specific optimization target.

### Exact arithmetic

For the released shape:

- 16 PLE rows are selected per token.
- each row is 160 BF16 values = 320 bytes.
- useful PLE row payload is therefore about 5120 bytes per token before any storage/alignment overhead.

This makes naive page-aligned random reads vulnerable to amplification.

### Hierarchy

PLE-Hydra uses four logical tiers:

1. per-owner CPU hot cache
2. shared CPU hot/warm cache
3. Linux page cache / mmap warm set
4. NVMe cold backing via asynchronous direct I/O

The exact split is autotuned from hit rates and bandwidth contention.

### Result caching

GB10X benchmarks two cache granularities:

- individual logical PLE rows
- fully combined 2560-wide PLE vectors keyed by the active n-gram state

The second option consumes more bytes per entry but can remove hashing/gather/I/O work on repeated code/tool/JSON patterns.

### PLEPack

GB10X defines a model-preparation format that preserves exact PLE values while physically reorganizing storage to reduce I/O amplification.

PLEPack requirements:

- deterministic conversion from pinned source tensors
- reversible logical-row mapping
- exact BF16 mode
- block layout chosen for real query locality
- checksummed manifest
- source revision and digest provenance
- validation that PLEPack output produces byte-identical logical rows to source

Approximate PLE encodings are separate non-default experiments and must never silently replace exact mode.

### I/O policy

GB10X compares:

- buffered/mmap reads for reusable warm data
- `madvise`/`posix_fadvise` prefetch hints
- `O_DIRECT` for cold data
- io_uring with registered files/buffers

The winner is selected from end-to-end PLE stall time and total tok/s, not raw NVMe bandwidth.

## MTP Memory Oracle

Speculative decoding is also a memory-prediction source.

When MTP proposes future tokens, GB10X derives likely future:

- PLE rows/vectors
- n-gram storage blocks
- selected metadata
- optional expert hints

and prefetches them only when expected saved latency exceeds expected wasted traffic.

A rejected draft must not corrupt any cache/state. Prefetch side effects are performance-only.

## L3Draft

GB10X includes an exactness-preserving CPU-side speculative proposal engine with a deliberately cache-bounded working set.

It may use:

- prompt lookup
- recent token sequences
- compact token trie
- code/JSON/tool grammar
- adaptive n-gram statistics

It never authorizes output. Target verification remains authoritative.

The initial working-set target is small enough to remain predominantly in shared CPU cache under dedicated-server operation. Its size is tuned empirically.

## ExpertScout and Expert-L2

ExpertScout predicts only residency/prefetch, never routing semantics.

Inputs may include:

- previous routed experts
- current layer/superblock
- token class/history
- compact hidden-state summary if inexpensive

Output is a ranked prefetch hint set.

The actual Qwen router remains authoritative. Wrong predictions may waste traffic but may not alter output.

Only small/high-value expert tiles are candidates for GPU L2 persistence; complete experts remain streaming unless measurements prove otherwise.

## Memory Fabric

### Unified-memory budget

The CPU and GPU share physical memory, so GB10X maintains one global memory budget for:

- weights
- KV/state
- PLE cache
- Linux page cache
- model-preparation buffers
- CUDA workspaces
- CPU runtime

Separate fake “RAM” and “VRAM” budgets are forbidden for capacity accounting.

### Page policy

Use larger pages for large stable arenas when verified to reduce translation overhead without harming random-access subsystems.

Do not force huge pages onto cold PLE random I/O if it increases amplification or reclaim cost.

### Stable arenas

Decode-critical GPU and CPU allocations should converge toward stable arenas after initialization so CUDA Graphs, address locality, and page residency remain stable.

## Bandwidth Governor

CPU and GPU contend for the same memory fabric. CPU utilization is therefore not a target metric.

The governor varies:

- number of active compute workers
- PLE cache size
- prefetch depth
- io_uring queue depth
- speculative width
- expert prefetch aggressiveness

and observes end-to-end tok/s plus GPU/CPU bandwidth and stall signals.

If additional CPU work lowers total tok/s, the governor backs off even if CPU utilization falls.

## Power Governor

GB10 power is shared at the SoC level. Excess CPU work may reduce GPU headroom.

The governor correlates:

- CPU worker intensity
- GPU clocks/utilization
- package power/thermal signals when exposed
- tokens/s
- tokens/joule

and keeps only configurations that improve the primary objective.

## RT Core experiment

RT cores are not general ALUs and are not part of the required production path.

An isolated experiment may represent QSA/index-search pruning as a conservative BVH query through OptiX/Vulkan RT. It may enter production only if:

1. candidate recall is exact for the production selection contract, and
2. `RT prune + exact CUDA rescore` beats the best CUDA-only path end-to-end.

Otherwise the experiment remains disabled and does not complicate production code.

## Texture/read-only paths

Read-only/texture cache paths may be benchmarked for lookup-heavy immutable data such as compact codebooks or metadata. They are optional and remain only when they improve the measured workload.

## Correctness model

The initial engine has two operating precision classes:

### Exact architecture mode

- preserves the pinned model checkpoint’s supported numerical representation
- deterministic greedy regression fixtures
- PLEPack is logically exact
- speculative proposals are verified by the target
- prefetch/routing predictors cannot alter model semantics

### Experimental approximate modes

Quantized PLE/KV or other non-source numerical changes must be explicitly enabled and independently quality-gated. Their metrics are never mixed with exact-mode claims.

## Validation gates

Every optimization requires:

1. unit/structural test
2. numerical comparison against a trusted reference path
3. repeatable end-to-end benchmark
4. telemetry proving where the gain came from
5. regression guard

Performance claims must record:

- git commit
- model revision/digests
- exact command/config
- context length
- prompt/output lengths
- cache state
- speculation settings
- CPU affinity
- power/thermal state when available
- prefill tok/s
- decode tok/s
- TTFT
- p50/p95 when serving
- memory footprint
- correctness result

## Baselines

At minimum, benchmark against viable GB10 configurations of:

- upstream llama.cpp
- vLLM/GB10-specialized stack when the model is supported
- Eider when directly comparable
- Atlas/other GB10-native engines when the same model/checkpoint path exists

Comparisons must state when quantization or model artifacts differ.

## Telemetry

The engine must expose enough internal timing to answer why a token is slow.

Initial stage timers:

- PLE hash/cache/I/O/gather
- embedding
- GDN stages
- QSA index/select/attention
- router
- expert gather/GEMM
- shared expert
- residual/norm
- MTP draft/verify
- sampling
- CPU wait
- GPU wait
- storage wait

Hardware telemetry integrations are layered so the engine remains runnable if a specific PMU/Nsight/resctrl signal is unavailable.

## Repository boundaries

Initial repository structure:

```text
crates/
  gb10x-core/        model-independent-but-GB10-only policies and types
  gb10x-runtime/     scheduler, topology, governors, serving runtime
  gb10x-qwen38/      Qwen3.8-Flash-Next manifest and forward scheduling
  gb10x-ple/         PLE-Hydra and PLEPack
  gb10x-telemetry/   metrics and benchmark records
cuda/
  common/
  gdn/
  qsa/
  moe/
  mtp/
  fused/
arm/
  cache/
  ple/
  draft/
experiments/
  rt-qsa/
bench/
tests/
tools/
docs/
```

Files remain responsibility-focused. Hardware experiments do not leak into the production path until promoted by a measured gate.

## Development strategy

Implementation is incremental and benchmark-led:

1. establish repository/toolchain contracts and hardware probe
2. pin/validate the Qwen3.8-Flash-Next manifest
3. implement deterministic PLE hash + exact PLEPack preparation
4. implement CPU cache ownership and PLE-Hydra simulation/tests
5. establish CUDA `sm_121a` build surface and kernel ABI
6. build a correctness-first minimal forward path
7. profile before fusion
8. add QSA-L2 and cache controls
9. add MTP speculation and Memory Oracle
10. add L3Draft/ExpertScout experiments
11. add governors
12. only then evaluate RT-core/texture experiments

The first performance milestone is not “all units utilized”; it is a correct Qwen3.8-Flash-Next token path with enough telemetry to identify the dominant GB10 bottleneck.

## Research references used for the design

The design is informed by the official NVIDIA GB10/DGX Spark documentation, CUDA programming/tuning documentation, Linux ARM64 cache/MPAM/resctrl and memory-management documentation, the released Qwen3.8-Flash-Next configuration, and direct code study of current GB10-oriented engines including Eider, Atlas, veloGB10, Fucina, llama.cpp GB10 work, and Spark-specific vLLM/CuTe stacks.

Upstream code may only be incorporated when its license is compatible and attribution requirements are satisfied. Ideas and benchmark methods may be reproduced independently.
