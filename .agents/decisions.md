# Decisions — qwen4exp port

## Upstream Contribution: Zero-Allocation Prompt Lookup Drafter (PR #300, 2026-10-04)

- **Context**: Upstream Splash had open Issue #186 (`[perf] Prompt-lookup drafter alongside DFlash2`) and RFC scaffolding PR #194 by `@linson007`. PR #194 declared constants and static string matching functions, leaving the stateful C++ engine and tests as follow-up work.
- **Decision**:
  - Authored and submitted upstream PR #300 (`incoai/splash#300`) under Nitin's GitHub handle (`npanj`) from branch `feature/prompt-lookup-drafter`.
  - Rebased and preserved `@linson007`'s 3 RFC scaffolding commits with full author attribution, completing the implementation with the production-grade zero-allocation PLD engine (`runtime/ops/PromptLookup.{hpp,cpp}`).
  - Built a flat chained inverted hash table (`head_` power-of-two table with load factor < 0.5, `next_` parallel array) ensuring zero heap allocations during decode with reverse-chronological search and multi-order n-gram tie breaking.
  - Added native C++ unit and latency test suite (`dev/tests/engine/prompt_lookup_test.cpp`) wired into `dev/native.mk` (`test-engine-cpu` and `test-sanitizers`), passing 100% clean under ASAN, UBSAN, and TSAN.
- **Measured Results**:
  - Sub-50 ns query latency on Apple Silicon (~49.1 ns per lookup).
  - ~7.9 µs prompt ingestion for 400 tokens (~20 ns/token).
  - 100% green test passes across unit, sanitizer, and python test suites.


## Metal Host Dispatch Latency Optimizations & Driver Synchronization Elimination (2026-10-04)

- **Problem**:
  1. In `runtime/metal/MetalBackend.mm`, `impl_->sampleDeviceMemory()` was called directly on command submission right after `[command commit]`. Querying `MTLDevice.currentAllocatedSize` right after commit forced the CPU host thread to synchronize with the in-flight Metal command queue on Apple Silicon, injecting a driver pipeline stall into every decode step.
  2. Inside the inner dispatch encoding loop, buffer indices were dynamically scanned and tested using a bitmask reconstructed per dispatch, and stack arrays were re-instantiated inside the dispatch loop.
  3. Tracking retained allocations previously scanned vectors linearly ($O(N)$ find per binding).
- **Solution (Borrowed from Upstream Commit `143f8e4`)**:
  - Eliminated the submission-time `sampleDeviceMemory()` call entirely. Driver memory telemetry is already sampled upon command retirement in `addCompletedHandler` and on request admission in `refreshMemoryStats`.
  - Added precomputed `bufferIndices` bitmask to `PreparedDispatch`, and moved stack argument arrays (`mtlBuffers`, `mtlOffsets`) outside the dispatch loop, binding consecutive spans with `std::countr_zero` / `std::countr_one`.
  - Replaced linear search of retained allocations with vector collection and `std::ranges::sort` / `std::ranges::unique` deduplication.
- **Measured Impact**: Eliminates driver synchronization stalls on submission, saving ~50–150 µs of CPU dispatch overhead per forward step without affecting memory accounting or GPU safety. Passed all 21 CPU engine tests and all 171 server tests.


## 4-Pass Radix Block Selection in QSA (2026-10-04)

- **Problem**: In `models/qwen4exp/kernels/qsa_select.metal`, `qsa_select_blocks` previously executed a 32-pass bitwise binary search (`for (int bit = 31; bit >= 0; --bit)`) requiring 64 threadgroup barriers per row. Furthermore, resolving ties at the threshold used a sequential loop `for (uint taken = 0; taken < ties_wanted; ++taken)` running a full reduction across all blocks for every single tie (up to 512 passes and 1,024 barriers on heavily tied scores, taking 3.52 ms).
- **Solution (Borrowed & Adapted from Upstream PR #253)**:
  - Replaced the 32-pass bit search with an **8-bit radix histogram selection** over four passes (`shift = 24, 16, 8, 0`) using 256 atomic bins in threadgroup memory.
  - Replaced the sequential $O(\text{ties})$ reduction loop with **parallel chunk-wise prefix scans**, resolving all ties in parallel in ascending index order in at most $\lceil \text{blocks} / 1024 \rceil$ chunk passes.
- **Measured Results**:
  - All-equal scores (512 ties): latency dropped from **3,523.9 µs down to 779.1 µs** (**4.52x speedup**).
  - Production shape (65,536 blocks, 512 budget): latency dropped from **776.3 µs down to 379.3 µs** (**2.05x speedup**).
  - Preserved 100% exact numerical match and deterministic slot ordering across all test suites (`dev/tests/engine/qsa_select_metal_test.mm`).


## Canonical Naming Standardization to Slipstream (2026-10-04)

- **Universal Naming**: Standardized the project and binary names universally from `slipstream-v2` and `splash` to canonical **Slipstream** (matching the GitHub repository `npanj/slipstream`).
- **Primary Artifacts**:
  - Main binary: `build/slipstream` (built by `make` / `make all` / `make build/slipstream`).
  - Metal library: `build/slipstream.metallib`.
  - Root CLI launcher: `./slipstream`.
- **Backward Compatibility**:
  - Scripts `./slipstream-v2` and `./splash` serve as backward-compatibility forwarders delegating directly to `./slipstream`.
  - `Makefile` maintains symlink aliases `build/slipstream-v2 -> slipstream` and `build/slipstream-v2.metallib -> slipstream.metallib` so existing pipelines and scripts remain uninterrupted.
  - Server reports `"owned_by": "slipstream"` in `/v1/models` (with client and test harnesses accepting both `"slipstream"` and `"slipstream-v2"`).
  - SSE heartbeats emit `: slipstream-keepalive` (with test harnesses accepting legacy format).


Historically, Splash lacked GGUF ingestion and required proprietary, engine-specific binary package archives (`splash-packed-q4`, `splash-packed-q4-qwen4exp`) hosted on Hugging Face.
- **Universal Standard**: GGUF is the universal, open standard adopted across llama.cpp, Ollama, LM Studio, and community model uploaders. Community users and developers should never be forced to download vendor-locked formats.
- **In-Place Transparent Ingestion**: With native GGUF conversion and Darwin APFS hole-punching (`F_PUNCHHOLE`), Slipstream ingests standard multi-shard or single-file GGUF models directly, punching holes in the source file in-place to prevent disk duplication.
- **Clean Deprecation**: Proprietary Splash packages on Hugging Face are officially deprecated and retired for downloads. User-facing documentation, CLI prompts, and guides exclusively direct users to standard GGUF repositories and files (e.g. `nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF` or standard 27B GGUFs).
- **Zero-Copy Engine Cache**: The internal `prepared/` directory structure is treated strictly as an engine implementation detail for zero-copy memory mapping (`mmap`), entirely transparent to end users.

## Full Native Qwen3.8-27B Architecture Port & Multi-Format Support (2026-10-03)


Ported complete native C++ and Metal support for 27B dense hybrid models (`Swift-Qwen3.8-27B` and `Qwen3.8-27B`):
- Architecture layout in `models/qwen38/`: 64 layers (hybrid GDN recurrent + full attention every 4th layer), hidden dimension 5,120, intermediate FFN size 17,408, vocabulary 248,320.
- Implemented `Qwen3_8.cpp`, `Qwen3_8.hpp`, `Qwen3_8Target.cpp`, `Qwen3_8Target.hpp` with clean isolation matching `check_architecture.py`. Shared code reaches concrete models only via `ModelDescriptor.hpp`, `ModelFactory.hpp`, `QwenTarget.cpp`, and `Runtime.mm`.
- Integrated `models/qwen38/tools/convert_qwen38_gguf.py` and `convert_qwen38_mlx.py` allowing users to serve single `.gguf` files, multi-shard GGUF directories, MLX safetensors folders, or pre-converted `splash-packed-q4` / `splash-packed-q8` packages.
- Added automatic format and architecture detection in `install/launcher.py` (`_detect_model_format_and_arch`), routing to appropriate converters and preserving draft/tokenizer invariants without manual flags.

## In-Place GGUF Preparation with APFS Hole-Punching (2026-10-03)

Standard GGUF conversion previously duplicated weight storage on disk, requiring 2x the model size in free disk space (~197 GB for Flash-Next, ~34 GB for 27B).
- Integrated `F_PUNCHHOLE = 99` via macOS Darwin `fcntl(fd, F_PUNCHHOLE, struct.pack("IIqq", 0, 0, start, length))` across `package_format.py`, `sharded_gguf_reader.py`, `convert_qwen4exp_gguf.py`, and `convert_qwen38_gguf.py`.
- Slices of source GGUF shards are freed in-place as soon as target layers are written. Peak disk requirements drop by ~50% (~102 GB for Flash-Next, ~17 GB for 27B).
- Added `--keep-gguf` flag in `serve` to allow users to retain original GGUF files when desired.

## Upstream Synchronization & Defensive Hardening (2026-10-03)

Selectively ported high-leverage upstream fixes from `upstream-incoai` and community PRs:
1. `runtime/model/ModelDescriptor.mm`: Accept finite whole-number floats up to $2^{53}-1$ in `requireUnsigned` (`rope_theta=1e7` written as `10000000.0` by Python configs) without throwing invalid argument errors.
2. `runtime/metal/MetalBackend.mm`: In `wrapSharedMemory`, throw `MetalAllocationError` rather than `MetalBackendError` on zero-copy mapping refusal, enabling driver allocation retries.
3. `server/runtime.py`: Isolate crash-trace diagnostic dumps in `try...except Exception: pass` inside `_begin_generation_failure` so diagnostic failures never block request completion or process teardown.
4. `server/backend.py`: Track engine restarts in `_cache_status` so status responses from crashed/replaced engines are immediately invalidated.
5. `server/schema_validation.py`: Preserved bounded regular expression evaluation across `evolve` on schemas with nested `$schema` dialect declarations.

## Pre-Compiled Sampling & Constrained Policy Pipelines (2026-10-02)

Metal compute pipeline state creation (`newComputePipelineStateWithFunction:error:`) compiles MSL shader bytecode into hardware machine code for the Apple Silicon GPU execution cores. Previously, policy-conditioned kernels (`decode_sample_top32_sharded`, `decode_sample_top32_probs`, `decode_sample_sparse_draw`, `decode_sample_sparse_top1`, `decode_sample_top32_sharded_batch`, `decode_sample_top32_probs_batch`) were compiled lazily upon first invocation.
- On the first sampled ($T > 0$) request or tool-calling request, this lazy JIT compilation caused a 200–400 ms latency spike on Token 1.
- Added `MetalBackend::preparePipeline(name)` and wired `preparePolicyPipelines()` directly into `Runtime::Impl` initialization, warming up all sampling pipelines at engine startup.
- Eliminates first-turn jitter on agent reasoning and structured sampling turns.

## Probability-Gated Speculative Early Exit in MTP Chaining (2026-10-02)

MTP draft chaining in `Qwen4ExpTarget.cpp` previously executed sequential GPU `step()` calls up to `maxDrafts` (2–5) even when cumulative chain confidence had already collapsed below the acceptance threshold ($C_{k-1} < 0.35$).
- Because $C_k = C_{k-1} \times P(k) \le C_{k-1}$, drafting token $k$ when $C_{k-1} \times 0.85 < 0.35$ has almost zero chance of passing the chain threshold, resulting in 8–10 ms of wasted MTP GPU compute per discarded draft.
- Added pre-step early-exit gating: `if (chainConfidence * earlyExitFactor < chainConfident) break;` (default `earlyExitFactor = 0.85`, overrideable via `SPLASH_MTP_EARLY_EXIT_P`).
- Changed loop exit on low confidence from `return drafts;` to `break;`, ensuring that valid proposals are cleanly retained in `drafted` rather than silently dropped.
- Speedup: Eliminates 1–3 wasted GPU forward dispatches per step on divergent text (+3% to +6% tok/s decode on difficult reasoning/coding prompts).

## Vectorized Metal Buffer Argument Binding & Driver Sync Removal (2026-10-02)

In `runtime/metal/MetalBackend.mm`:
- Previously, `submitCommandAsync` iterated over `dispatch.buffers` calling `[encoder setBuffer:offset:atIndex:]` individually, resulting in 1,000+ dynamic Objective-C runtime dispatches (`objc_msgSend`) per token step on the CPU.
- Batched contiguous buffer arguments using Apple Metal's native vectorized API `[encoder setBuffers:offsets:withRange:]`.
- Removed `addScheduledHandler` device memory sampling (`observer->sampleDeviceMemory()`), eliminating mid-flight IOGPU driver synchronization stalls while commands are queued.
- Speedup: Reduces host CPU dispatch latency per step from ~8 ms down to ~2–3 ms.

## Defensive Non-Finite / Out-of-Vocabulary Logits Guard (2026-10-02)

In `runtime/engine/Engine.cpp`:
- If an FP16/BF16 underflow/overflow or NaN occurs during sampling (leaving sentinel `0xffffffff`), the engine previously inserted the invalid token into `active.exactTokens` and published it to the KV cache, triggering an out-of-bounds Metal memory crash or engine deadlock on the next embedding lookup.
- Added explicit validation in `Engine::apply`: fails the specific lane cleanly with `model_result_invalid` ("model emitted out-of-vocabulary token") before cache publication or output emission, allowing peer requests in the batch to complete safely.
- Added regression test `testOutOfVocabularyOutputFailsLaneOnly` in `dev/tests/engine/kv_first_engine_test.cpp`.

## Dynamic Pipe Write Deadline Refresh for Deep Context Prompts (2026-10-02)

High-context prompts (>28,000 tokens) produce RequestFrames >115 KB. Because macOS OS pipe capacity is 64 KiB (65,536 bytes), the initial `stream.write` fills the buffer, forcing subsequent chunks to wait for the native engine to drain the pipe. Previously, `MultiplexedRuntime._write_bytes` set a single static 5.0-second deadline across the entire multi-chunk frame. If the engine was busy executing a prefill chunk or SSD expert read, the timer expired, raising `EngineUnhealthy("native frame write timed out after 65536 bytes")` and triggering an unrecoverable process restart (`resident layers: 0 / 48`) with HTTP 503 `runtime_unavailable`.
- Increased default `io_timeout` to 30.0s (and 60.0s in `server.py`), overrideable via `SLIPSTREAM_IO_TIMEOUT`.
- In `_write_bytes`, refreshed `io_deadline = time.monotonic() + self._io_timeout` on each successful partial chunk write (`written > 0`). As long as the engine process actively consumes bytes, the write stream is healthy and permitted to complete without timing out.

## Standalone GGUF Ingestion & Prebuilt Binary Packaging (2026-10-02)

Integrated contributions from Mike Zinner (@mariadb-MikeZinner):
- Converted `ngram.bin` directly from multi-shard GGUF by streaming `per_layer_token_embd.weight` 512k rows at a time, eliminating the need for reference packages or external files.
- Automatically fetched tokenizer metadata from Hugging Face if missing.
- Implemented standalone prebuilt packaging (`make package` + `.zip` repackaging) including bundled `libslipstream-dequant.dylib` compiled from `fast_dequant.c`.
- Published Release `v26.10.4` on GitHub (`npanj/slipstream`) with one-line installer (`curl -fsSL .../install.sh | sh`).

## Graceful Output Budget Clamping for Context Ceilings (2026-09-30)

When prompt tokens approach the context window ceiling (e.g. during lengthy agent tool loops), OpenAI `/v1/chat/completions` previously raised an unrecoverable HTTP 400 `context_length_exceeded` if `prompt_tokens + max_completion_tokens > max_context`. We added `--clamp-output-budget` (and `FrontendServer.clamp_output_budget`) to gracefully clamp the output budget to `max_context - prompt_tokens` and return `finish_reason: "length"`, aligning `/v1/chat/completions` with the graceful behavior of Anthropic `/v1/messages`.

## Exact Speculative Rejection Sampling & Residual Token Replacement (2026-09-30)

Standard speculative decoding (Leviathan et al.) requires that rejected draft proposals are replaced by drawing from the residual distribution $P_{residual}(x) \propto \max(0, P(x) - Q(x))$. In `sampling.metal`, previous implementations failed to exclude the rejected token and allowed acceptance when draft proposal probability was 0.0. We ported `ds4`'s point-mass rejection and replacement math into `sampling.metal`:
1. Draft acceptance verifies validity ($q > 0.0$ and finite); accepts deterministically if $p \ge q$; otherwise accepts stochastically with probability $p / q$.
2. In `sparse_residual_sample`, `rejected_token` is explicitly excluded from both the residual accumulator and fallback target distribution, guaranteeing that a rejected token cannot be chosen as its own replacement and preserving exact mathematical target-distribution parity at non-zero temperatures.

## Adaptive Multi-Row MTP Speculation Controller (2026-09-30)

When speculative drafts chain to depth 5 or 7 on divergent or branch-heavy text, subsequent draft guesses are rejected >60% of the time, wasting 3–5 MTP head forward dispatches and forcing the target verifier to load dozens of redundant MoE experts from NVMe. We introduced `AdaptiveDraftController` in `models/qwen4exp/Qwen4ExpTarget.cpp`:
- Tracks a rolling 8-cycle window of first-draft acceptances and a chained rejection streak (`reject2Streak`).
- If chained drafts repeatedly reject (`reject2Streak >= 2`) or recent window acceptance is low (<4/8), the controller dynamically throttles speculation to shallow depth (2 drafts = anchor + 2).
- When acceptance recovers, depth automatically scales back up to the full configured limit (`maxDrafts`).
- Result: decode throughput jumped from 50.3 to 51.9 tok/s greedy (up to 67.7 tok/s on explanation) while eliminating useless SSD expert thrashing.

## Layer-Ahead NVMe Advisory (`F_RDADVISE`) for Prefill Waves (2026-09-30)

During prefill waves on MoE models, synchronous page reads for missed expert matrices stalled the GPU on NVMe latency (~475 ms per prompt chunk). By issuing non-blocking `fcntl(fd, F_RDADVISE, &ra)` hints to the macOS XNU kernel for all missed expert matrices as soon as a layer's routing finishes (and immediately for the current layer's wave sequence), the kernel streams NVMe pages into the unified buffer cache via background DMA while the GPU executes attention and GDN. Subsequent `::pread` calls hit page cache directly at RAM speed, reducing prefill staging time by 28% (475 ms -> 348 ms) and lifting overall benchmark decode throughput to 50.3 tok/s.

## The layer order, verified exactly

This is the single most valuable fact in this directory. A layer composed this
way, from real layer-0 weights, matches `Qwen4ExpTextGatedResidual` from
transformers 5.16.1 with **max abs diff 0.000e+00**:

```
attention hyper-connection  (norm -> mix-down -> mix-up)
  -> mixer block (GDN, or full attention every 4th layer)
  -> inject back into the residual
mlp hyper-connection        (norm -> mix-down -> mix-up)
  -> expert block
  -> inject back into the residual
```

Residual width is 10240 = 4 streams of 2560. The check script is
`models/qwen4exp/tools/checks/check_layer_composition.py` — re-run it if the forward path
disagrees with the reference.

**Why this matters:** every kernel was already verified against its own
reference in isolation. None of those could catch a wiring error — wrong order,
wrong residual stream carried, injection in the wrong place. That is where every
interface bug in this port actually lived.

## Norm gain is `(1 + weight)`

`Qwen4ExpTextRMSNorm` computes `output * (1.0 + self.weight.float())`, and the
weight is zero-initialised. Using the stored gain directly doubles every gain.
Real gains run -5.94 to +6.63, so this is badly wrong, not subtly wrong.

## Q4 layouts are not interchangeable

- **Matmul operands are tiled**: `[tile][group][row][64 codes]`.
- **Lookup tables are plain row-major**, three separately aligned runs.
  Embeddings are gathered a row at a time, never multiplied as a tile.

## Quantizer anchors at the larger-magnitude extreme

Not at the minimum. That value lands on code zero and the error falls on the
smaller side, which is why scales can be negative. Codes are quantized against
the **bf16** stored scale/bias, not the float32 ones — the reader only ever sees
the stored values. Getting this wrong gave a 55% code mismatch that still loaded
and ran.

## Tiling and group sizes

- Experts tile at **StorageN 128** (`kQ4ExpertStorageN`), not 256 — experts are
  640 wide. Chosen by Nitin over padding to 256.
- N-gram rows are 160 wide, so they use **group 32** (`kQ4FineGroupElements`).
- Router is padded to 256 regardless of live expert count. Do not conflate
  `params.experts` with the router width — that broke `moe-metal` once.
  Handled via a `RouterWidth` template with entries at 256 and 512.

## Linear attention needed no new kernels

GDN already compiles for `{16, 48, 128, 10240, 16640}`. This was the single
biggest scope reduction in the port.

## Sparse attention is a loop head plus a mask term

Attention already reads through a page table and takes each page's position from
its logical index. Walking a *list* of logical pages instead of a range keeps
position, causality and the softmax exactly as they were. Two bugs found here:
splits must partition whatever list is walked (the first version had every split
traverse the whole selection, 65x the work), and selection order must come from
a prefix sum, not an atomic slot claim, or it is non-deterministic.

## Memory: accounting, not machinery

`WeightStore` already maps weights `PROT_READ / MAP_SHARED`, so they are
file-backed and reclaimable. What refused to start the model was
`fixedRuntimeBytes()` counting all 96.61 GiB as held. `ModelMemoryFootprint` now
splits resident from streamable. Result: 6.98 GiB resident + ~38 GiB cache,
against the llama.cpp fork's 5 GiB + 36 GiB at 25 tok/s.

**Trap:** `streamableWeightsBytes` and `streamCacheBytes` are at the **end** of
`ModelMemoryFootprint` on purpose. Three call sites use positional brace-init;
inserting fields mid-struct silently shifts every later value, compiles clean,
and passes the CPU suite.

## The draft is a placeholder

> **Superseded (2026-09-22):** the placeholder still ships (the format needs one) but never runs: the model's own MTP head drafts, up to 5 guesses a step, output exact.

The package format requires a DFlash 2 draft. Flash-Next has no such draft, only
an MTP head. A zero-filled placeholder is written so packages validate. Expect
no speculation speedup until a real draft is trained. Nitin deferred that.

## Kv2Group12 Attention Kernels

Qwen4Exp attention uses 24 query heads and 2 KV heads (head dimension 128), which
is group size 12 (`Kv2Group12`). Dedicated prefill and decode (verify) projections,
gates, and Q8 split/reduce kernels were instantiated and wired into `PagedAttention`
to support this layout cleanly without padding.

## Command Buffer Chunking for Models Exceeding Device RAM

A single Metal command buffer cannot reference resources whose sum exceeds the
device working set ceiling (~58 GiB on 64 GiB Apple Silicon). Attempting to commit
all 48 layers (96.61 GiB) in a single command buffer triggers
`kIOGPUCommandBufferCallbackErrorOutOfMemory`. `MetalBackend::submitCommandAsync`
now chunks dispatches into batches bounded by `recommendedMaxWorkingSetBytes / 2`.
All chunks commit in order to the serial command queue, enabling models larger
than physical RAM to stream on demand without exhausting GPU driver residency limits.

## GDN Output Gate Activation is Sigmoid, Not SiLU

Upstream `transformers` (`modeling_qwen4_exp.py`) uses `Qwen4ExpTextRMSNormGated` with `output_gate_type: "sigmoid"`, computing `rms_norm(x) * sigmoid(gate)`. Splash's `gdn_primitives.h` had originally implemented `gate * sigmoid(gate)` (SiLU), which inflated GDN hidden states by ~2.8x. Changing `gdn_gate_phase` to use `1.0f / (1.0f + fast::exp2(-1.44269504089f * gate))` matched PyTorch output to bf16 precision and resolved text generation quality. Synthetic unit tests `gdn_metal_test.mm` and `gdn_decode_metal_test.mm` were updated to use `sigmoid` accordingly.

## Vectorized Hyper-Connection Kernels (4.7x GPU Speedup)

Dispatch profiling of the full 48-layer model revealed that `hyper_connection_normalize` (641.7 ms) and `hyper_connection_mix` (200.0 ms) accounted for 94% (841.8 ms out of 893 ms) of total GPU time due to unvectorized 16-bit scalar loads and low threadgroup occupancy. Vectorizing both kernels to 64-bit `bfloat4` aligned loads and caching the full 320-element low-rank vector in threadgroup SRAM reduced per-dispatch time by >3.15x and total GPU decode time from 504 ms to 190 ms across all 48 layers.

## Cyclic LRU Paging Thrashing & Expert Working Set Isolation

> **Superseded (2026-09-22):** no layer is resident now; every expert streams through per-layer caches read with direct preads (F_NOCACHE), so page-cache thrashing no longer applies.

On a 64 GB Mac, model files total 96.6 GiB (68.2 GiB in 48 layer files, 26.8 GiB n-gram, 1.6 GiB head/embed). Because `WeightFile` creates a zero-copy `MTLBuffer` for the entire 1.42 GiB layer file, the macOS IOGPU driver enforces residency on the full file on commit, even though a token only touches 10 experts (26 MiB) in that layer. This causes cyclic LRU thrashing: by layer 48, layers 0..15 are evicted, forcing every token to re-read ~45–60 GiB from SSD (~7.4 s/tok).
- Usable RAM allows 37 layers to remain 100% resident in physical RAM.
- Calling `madvise(MADV_DONTNEED)` on the expert weights of the remaining streaming layers (37..47) at the end of each decode step ensures the OS reclaims streaming pages first, preventing eviction of resident layers.
- For true sub-second decode (~200-250 ms), streaming layers must bind and stage only the 10 active experts (26 MiB/layer = 286 MiB total) via an in-memory expert cache rather than binding the full 1.42 GiB file.

## Sub-Buffer Residency Isolation (`detachStreamingLayer`)

Binding even a single 5 KB norm from a 1.42 GiB `WeightFile` forces Apple's IOGPU driver to enforce GPU residency on the entire 1.42 GiB buffer on commit. `detachStreamingLayer()` detaches all non-expert tensors (attention projections/norms, GDN weights/norms, indexer, MLP hyper-connection, router, shared expert) into standalone, dedicated `MTLBuffer`s (~35 MB per layer). The 1.42 GiB file buffer is NEVER bound to Metal during decode, keeping the active GPU working set under 12 GiB for all 48 layers combined.

## Persistent Per-Layer Expert Cache with LRU Slot Tracking

> **Superseded (2026-09-22):** capacity is now 288 per layer (llama.cpp's 36 GiB budget, minus a 144-expert prompt staging buffer), eviction is least-used first, caches are mlock'd and in a GPU residency set.

To eliminate copying 20+ active experts on every token step, each layer maintains an in-memory `Qwen4ExpLayerExpertCache` with dedicated `cacheGate`, `cacheUp`, and `cacheDown` `MTLBuffer`s of configurable capacity (default 64 experts per layer = 9.4 GiB across all 48 layers).
- Active experts are mapped to persistent cache slots using `expertToSlot` and `slotToExpert` arrays.
- Cache hits touch `lruTime[slot]` with zero copy overhead (0 ns).
- Cache misses allocate the next free slot or evict the least-recently-used slot not used in the current step.
- Misses are staged concurrently via GCD `dispatch_apply` across CPU performance cores.
- Hit rates exceed 92% in steady state (reducing misses to ~1.7 per layer), dropping staging latency from 6,162 ms to ~300 ms across all 48 layers.

## CommandGraph Move Invariant in Multi-Command Execution

In Splash's pipeline, `Runtime.mm` encodes `encodeBatchVerifyInput` and `encodeBatchEmbedding` into `graph` before invoking `targetModel.addVerify(graph, ...)`. Any engine path that internally splits the execution graph into multiple command buffers must begin with `metal::CommandGraph residentGraph = std::move(graph);` rather than constructing a default-initialized `CommandGraph`. Otherwise, layer 0 dispatches execute on GPU before input token embeddings are written to `buffers.hidden[0]`, corrupting the first layer's activations.

## Prefill Staged Streaming Expert Cache Integration

> **Superseded (2026-09-22):** the monolithic fallback stalled the GPU for ~10 s on page faults and is gone; prompts run expert-first waves (see below).

Prefill previously bound all 48 layer files (68.2 GiB) into a single monolithic command graph, causing Apple's IOGPU driver to demand-page gigabytes of weight files from SSD on cold start (~11.1 seconds for 5 prompt tokens) and dispatch 512-expert MoE kernels.
- `Qwen4ExpTarget::addPrefill` is now integrated with the per-layer staged expert cache: only the ~20-30 active experts per layer selected by the router across the prompt tokens are staged into `layer[L].expertCache`.
- MoE execute dispatches against the 64-expert cache buffer rather than the monolithic 512-expert file.
- Prefill latency dropped from 11.1s down to **549 ms (19x speedup)**.
- **Warm Decode Cache Hand-off:** Prefill leaves the prompt's active domain experts warm in `layer[L].expertCache`, eliminating the cold-cache penalty on decode token 1 (staging time dropped from 6,162 ms to **159 ms**).
- **Graceful Monolithic Fallback:** If an exceptionally long prompt activates more unique experts in a single layer than `cache.capacity` (64), `stageActiveExperts` returns `false` and automatically falls back to `weights.layers[L].ffn` for that layer, guaranteeing zero numerical regression or memory overflow regardless of sequence length.

## Everything but the routed experts is 8-bit (2026-09-22)

A fake-quantized reference run (`reference_logits.py --quant`) reproduced the
engine's error to 0.1%, so the long-context drift was rounding, not a bug.
4-bit mixers, head and embedding cost most of it. Now: routed experts 4-bit;
mixers, output head, embedding and hyper-connection weights 8-bit; router 8-bit.
Code prompt 81% -> 91% same top pick. Rejected: 3-bit experts (83%), expert
groups of 32 and error-minimizing ranges (both worse, unexplained).

## Hyper-connection weights use the tiled 8-bit matrix layout

So prompts run them as matrix products (they were 53% of prompt GPU time in
scalar kernels) while decode keeps its own kernels over the same bytes. The
down projection is padded from 320 to 512 outputs to fill whole tiles.

## Prompts run experts in waves, each expert read once per layer per chunk

Stage 0 runs cached experts; missing ones among the chunk's most-used go into
cache slots (so decode inherits them), the rest into a 144-expert staging
buffer whose two halves alternate so reads overlap GPU work. Waits use a
GPU-only "stage done" event: the shared pipeline event is also raised by the
host, which let a wait return before the GPU had finished (a real race).

## Prompt chunks are per package

`SPLASH_PREFILL_TOKEN_BUDGET` (4096) is the built maximum; the scheduler's
chunk comes from the manifest's `prefill_token_budget`. qwen4exp uses 4096:
bigger chunks re-read fewer experts per token.

## Dense attention past 2,048 tokens stays

The reference model drops tokens through a sparse indexer past 2,048. On real
text it then predicts worse (perplexity 10.3 vs 4.5 past 3K), so matching it
would make Splash worse. The indexer kernels exist but stay unwired.

## Decode hand-offs stay on the host

Each layer hands control to the CPU (~164 us) to pick and load experts. Moving
routing to the GPU would need the CPU to signal a running command buffer;
Metal doesn't make host writes visible there (probe: handoff_latency.mm).


## Draft head scores the 64K most common tokens (2026-09-22)

Its guesses are checked by the full head, so answers can't change. 64K cut
the draft 12.3 -> 8.4 ms a step with the same tokens per step; 32K made the
head's confidence (softmax over fewer tokens) too high, so it guessed further
and more guesses were rejected. Supersedes the earlier note that shrinking
the draft vocabulary would cost more than it saves.

## Expert cache must fit in free memory at load (2026-09-22)

The loader refuses when free memory < cache + 2 GiB. A 10% margin (like the
runtime governor) refused the normal 34 GiB setup, which leaves ~4 GiB.

## Hyper-connection kernels stay two launches (2026-09-22)

Merging down and up-mix needs every threadgroup to wait on the others
mid-launch. Metal does not promise they all run at once, so it can deadlock
the GPU. Three safe variants measured no faster.

## 2026-09-22 — Model code lives in `models/<name>/` (Slipstream phase 4)

A model folder holds its layout, loader, forward pass, own kernels, ABI headers,
converter, checks and benchmarks. It may launch its own kernels; it may not
depend on the engine or another model. Shared code reaches it only through four
files (`ModelDescriptor.hpp`, `ModelFactory.hpp`, `QwenTarget.cpp`, `Runtime.mm`).
`check_architecture.py` enforces this; the build fingerprint hashes `models/`.
Kept `runtime/` as the shared root (no rename to `core/`): the rename adds churn
and no checkable boundary. The installer and Homebrew packaging stay for now:
they are wired into 13 tests, so removing them is its own step.

## 2026-09-23 — Keep the built-in draft head; the Mac-trained block guesser lost

A DFlash-style block guesser, trained with MLX on 3M positions of the model's own
session text, scored 25 tok/s against today's head's 39 at the same anchors
(first guess right 56% vs 84%). Built-in heads are trained on far more text;
beating one needs 20-100x more data than this Mac can produce quickly. The
recording mode, corpus, trainer and `trace_report --proposals` stay for a
future attempt. Guesser design lessons: work at the model's width, start from
its latest state, test with a copy task first. See docs/draft-head-plan.md.

## 2026-09-23 — Read ahead 6 experts a row; split cache slots unevenly

Each check step read 87 experts ahead (on a prediction that is right ~2/3 of
the time) plus 46 on demand. Read-ahead 10 -> 6 per row and a per-layer slot
profile (fitted on Nitin's sessions with cache_plan.py, same total memory) cut
that to 37 + 50: +4% on the 10-prompt suite (39.7 -> 41.4) and +3% on held-out
session prompts (39.6 -> 40.9), outputs identical. Plain least-recent eviction
tied the current rule on session text, so the rule stays. Overrides:
SPLASH_LOOKAHEAD_EXPERTS, SPLASH_EXPERT_SLOTS (=even for the old split),
SPLASH_EXPERT_EVICT=lru.

## 2026-09-23 — Review of upstream PR incoai/splash#115 (Flash-Next on M5 Ultra)

One squashed commit, 67K lines (also github.com/mweinbach/splash-flash-next,
branch codex/qwen38-flash-next-m5-ultra); a separate Flash worker importing an
MLX checkpoint as-is, 8-bit experts (~9.8 MB each), tuned for an M5 Ultra, no
Flash-Next speed claims, no whole-model runs on the rebased build.
- **MTP depth controller** (FlashMTPDepthController: one depth per stretch of
  text from averaged acceptance and measured cost, with probing): replayed on
  our traces (bench/depth_controller_replay.py) it gives 31-41 tok/s vs our
  per-step confidence stop's 40.5 / 43.8. Not borrowed.
- **GDN lazy rollback**: we already replay only kept rows into the other state
  copy (ops::GDN::addCommit); nothing to take.
- **Static hot-expert plan, 8-bit experts, SSD n-gram row cache, 2048-row bulk
  QSA prefill**: built for far more memory; our bottleneck is expert reads.
  Bulk QSA prefill could be revisited if long-prompt prefill becomes the target
  (profile first: SPLASH_PROFILE_PREFILL).
- **Upstream Splash since our fork (1.0, 2026-09-18): 67 commits we lack.**
  Worth porting: #92 tool arguments omitted with required-first order + exact
  message-prefix reuse (agent turns), #44 false Metal command timeouts, #40
  UTF-8 JSON, #31 composed tool schemas, #120 per-request timings. Most kernel
  commits target Apple9 or dense Q4 (the 27B/35B), not our 8-bit/expert paths.

## 2026-09-25 — GGUF Direct Ingestion & Layout Parity

To serve Nitin's daily 3-shard model (`~/models/qwen38-flash-next-v3`, Q4_0/Q8_0) without requiring an intermediate 338 GB BF16 conversion:
1. **Sharded GGUF Reader (`dev/tools/sharded_gguf_reader.py`)**:
   - Discovers across multi-file shards (`-00001-of-00003.gguf`, etc.) and sidecars (`mtp-shared-Q4_K_M.gguf`).
   - Uses `llama.cpp-prism/build/bin/libggml-base.0.21.0.dylib` for native SIMD dequantization of Q4_0, Q8_0, Q4_K, and Q6_K rows.
2. **RMS Norm Offsets**:
   - GGUF encodes `(1.0 + weight)` into hyper-connection norms, MTP enorm/hnorm, and indexer norms.
   - Splash Metal kernels (`hyper_connection_normalize`, etc.) calculate `(1.0f + weight)` internally from zero-centered weights.
   - We subtract `1.0f` from GGUF norm tensors during conversion to avoid doubling the effective normalization gains.
3. **GDN Value-Head De-permutation**:
   - In GGUF, 48 GDN heads are ordered `(3, 16)` across channels (3 groups of 16 heads).
   - Splash kernels expect canonical 48 heads ordered sequentially.
   - Reshaping `(3, 16, ...)` and transposing to `(16, 3, ...)` across `qkv` (v-part), `gate`, `beta`, `alpha`, `conv1d`, `ssm_a`, `ssm_dt`, and `ssm_out` achieves exact numerical agreement with reference logits.
4. **Router & Shared Expert Quantization**:
   - Router matmul tiles index weights by `[quant group][row]`.
   - Applying `quantized_q8` with transposed code layouts matches Splash router kernels with 100% byte equality.
5. **Zero Disk Duplication for N-gram**:
   - APFS hardlinking `ngram.bin` (26.8 GB) reuses the physical disk blocks, enabling zero-byte duplication during GGUF conversion.

## 2026-09-25 — Swift V3 GGUF Splicing & Dual-Engine Deployment

To create a Swift version of Nitin's V3 model (`Swift-Qwen3.8-Flash-Next-V3`) deployable to both llama.cpp and Slipstream without downloading redundant 175 GB Q8_0 repositories:
1. **HTTP Range Splicing Engine (`dev/tools/build_swift_v3_gguf.py`)**:
   - Indexed all 686 high-precision resident donor tensors (~4.77 GiB) across shards 1, 3, 4, 5 of `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/Q8_0` using HTTP range requests via standard HuggingFace endpoints.
   - Streamed the base Q4_0 shards (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF/Q4_0`, ~93.7 GiB) sequentially.
   - Spliced Swift's Q8_0 output head (0.65 GiB) and resident tensor groups (`attn`, `hc`, `token_embd`, `ssm_out`, `shexp`), while retaining Q4_0 for the 512 routed experts and PLE table.
   - Produced 3 standard GGUF shards totalling 95.52 GiB (1,224 tensors) in `~/models/swift-qwen38-flash-next-v3`, bit-compatible with llama.cpp MTP streaming.
2. **Dual-Engine Benchmark Parity**:
   - Both Slipstream (`local/swift-qwen38-flash-next-v3`) and llama.cpp run the identical GGUF weights.
   - Slipstream delivers **41.72 tok/s** vs llama.cpp's **22.84 tok/s** (1.83x speedup) with 1.51x faster TTFT (1,305 ms vs 1,973 ms) and 100% quality parity across all reasoning and coding domains.

## 2026-09-26 — Max-Tokens Capped at 16,384 & Swift Thinking Calibration

1. **Generation Cap Reduced to 16,384**:
   - High-context sessions (~58k input tokens) combined with `xhigh` thinking level triggered repetitive thought loops that ran up to the 32,768 token ceiling (~10.5 minutes at 54 tok/s), exhausting the budget and returning an empty turn (`stopReason: length`) inside hidden `<thought>`.
   - Capping `maxTokens: 16384` across `~/.pi/agent/models.json` (and `~/.omp/agent/models.yml`) halts runaway thought loops in under 5 minutes without starving legitimate code generation or reasoning turns.
   - Aligned `compaction.reserveTokens: 16384` in `~/.pi/agent/settings.json`, buying an additional 16,384 tokens of prompt conversation history before auto-compaction triggers.
2. **Swift Thinking Profile**:
   - Swift 1.5 models are distilled specifically for concise internal chains of thought; forcing `xhigh` in deep contexts induces degenerate loop behaviour. Swift models should be invoked with `--thinking low` or `medium`.
3. **Pi Catalog Multi-segment Globs**:
   - In `~/.pi/agent/settings.json`, `enabledModels` requires `"slipstream/**"` (not `"slipstream/*"`) to match slash-nested model identifiers like `slipstream/local/swift-qwen38-flash-next-v3`.

## 2026-09-26 — Head-to-Head Benchmark: Swift-27B-Splash-HQ vs Swift-Flash-Next-V3

1. **Automated Memory-Guarded Swapping Pipeline (`run_full_comparison_orchestration.sh`)**:
   - Implemented a fully automated sequential pipeline enforcing the strict single-large-model constraint on 64 GB unified memory.
   - Runs Model 1 (`Swift-Qwen3.8-Flash-Next-V3` on `:8090`), shuts it down, confirms 0 listeners and RAM release, launches Model 2 (`Swift-Qwen3.8-27B-Splash-HQ` on `:8000`), runs identical items, shuts down Model 2, restores Model 1, and compiles the comparative report.
2. **Benchmark Scope (145 Items across 6 Domains)**:
   - Evaluated 20 AIME 2025 problems, 35 Level 4-5 MATH-500 problems, 35 GPQA Diamond science questions, 25 GSM8K word problems, 25 HumanEval coding challenges with test execution, and 5 hard systems/concurrency/architecture probes.
3. **Empirical Quality Findings**:
   - **Overall Accuracy**: Swift-Flash-Next-V3 leads slightly at **70.3% (102/145)** vs Swift-27B-Splash-HQ at **67.6% (98/145)**.
   - **AIME 2025 Parity**: Exactly 45.0% (9/20) on both models, solving the identical 9 problems with 100% agreement.
   - **Science Advantage**: Flash-Next-V3 outperformed 27B-Splash-HQ on GPQA Diamond (54.3% vs 45.7%, +8.6%), demonstrating stronger recall and domain reasoning in biology, chemistry, and physics.
   - **Coding & Systems Equivalence**: Both models achieved 92.0% on HumanEval (23/25 unit tests passing) and 100% on hard systems tasks (Acquire-Release memory ordering, memory bandwidth decode bottleneck, and zero-copy Rust CSV parsing).
4. **Speed & Latency Profile**:
   - Steady-state decode throughput is indistinguishable: **43.9 tok/s** (Flash-Next) vs **44.4 tok/s** (27B-Splash) (1.01x).
   - TTFT is 1.70x faster on 27B-Splash-HQ (**659 ms** vs **1,119 ms**) due to dense Q8 weights avoiding MoE prefill hyper-connection mixing and n-gram table gathers.

## 2026-09-26 — Speculative Drafting Optimization & Linear Chain Ceiling

1. **Empirical Impact of MTP Speculation**:
   - Across 14,876 generated tokens on the 20-item balanced benchmark, **10,206 tokens (68.6%) were generated speculatively**, giving an average of **3.19 tokens per forward pass** of the 48 layers.
   - Speculative drafting delivers a **2.7x speedup** over non-speculative decode (~42–44 tok/s vs ~15–16 tok/s).
2. **Trigonometric Precomputation in Draft Head (`Qwen4ExpTarget.cpp`)**:
   - `runMtpDraft` previously recomputed `std::pow` across all 32 rotary dimensions on every draft row, issuing 1,280 redundant double-precision math calls per token step on the CPU.
   - Precomputing the 32 frequency coefficients in `rotaryFrequencies` eliminates transcendental math during drafting.
3. **Draft Accounting Parity in `Runtime.mm`**:
   - `Runtime.mm` line 1105 previously hardcoded `kDraftProposalTokens` (7), masking true draft acceptance. Updated to report `impl_.mtpProposed` when active.
4. **Linear Speculation Bound**:
   - Raising the chain confidence threshold (`SPLASH_MTP_CHAIN_MIN` 0.35 -> 0.42) reduced acceptance to 27.2% and throughput to 40.9 tok/s because the verifier's fixed cost is high; retaining guesses up to chain threshold 0.35 remains optimal for linear chains.

## 2026-09-26 — Tree Speculative Drafting Invariants & KV Scatter Barrier

1. **Tree Drafting Shader Implementation**:
   - Implemented 2D attention masking in Metal (`attention_q8.metal`, `q8_attention_tile.h`), GDN tree recurrence (`gdn_primitives.h`, `gdn.metal`), and tree greedy acceptance in sampling (`sampling.metal`).
   - Guarded under `SPLASH_TREE_DRAFT`: defaults to 0 (exact linear causal path).
2. **The KV Cache Physical Placement Invariant**:
   - In `PagedAttention::addVerify`, `storePipeline_` writes all 8 verify rows sequentially into `keyData` and `valueData` at positions `committed_tokens + 0` through `committed_tokens + 7` during the forward pass.
   - For linear speculation, any accepted branch is strictly a prefix `{0, 1, ..., k-1}`, so sequential placement is naturally valid.
   - For tree speculation, taking a side branch (e.g. node 5 or node 6) accepts a non-contiguous set of rows. Leaving them in sequential physical slots pollutes the KV cache with activations from the rejected branches.
   - Similarly, GDN commit (`gdn_decode_commit`) replays the first `retained` rows in linear order.
   - Supporting general tree speculation safely requires an in-place KV cache compaction kernel and selective GDN state replay.
3. **MTP CPU Dispatch Overhead**:
   - Proposing 7 tree draft tokens using 4 sequential MTP forward steps inside `runMtpDraft` added ~8–10 ms of CPU/GPU scheduling overhead, offsetting the verification gain.

## 2026-09-26 — Expert Cache Sizing Flag is `SPLASH_EXPERT_CACHE_GIB`

- To prevent memory growth pauses (`resource_timeout`) during high-context prompts (such as GPQA Diamond), set `SPLASH_EXPERT_CACHE_GIB=30` (not `CACHE_GIB=30`).
- At 30 GiB (`~239` experts per layer), free host RAM remains >46 GiB, completely eliminating resource stalls while preserving decode throughput.

## 2026-09-26 — Prompt Lookup Decoding (PLD) Evaluation

- Evaluated Hayder Tirmazi's 42x prompt lookup research (`docs/research/prompt_lookup_drafting_analysis.md`).
- Fast CPU-side n-gram lookup (<4 µs) requires zero GPU compute and zero extra weights.
- Recommended roadmap:
  1. Add standalone zero-copy n-gram lookup engine (`PromptLookup.hpp`).
  2. Implement speculative verification on `Swift-Qwen3.8-27B-Splash-HQ` (projected to boost decode from 44 tok/s to 60–75+ tok/s on context-rich tasks with 0 MB extra VRAM).
  3. Integrate hybrid fallback on `Swift-Flash-Next-V3` when MTP head confidence < 0.35.

## 2026-09-26 — Flat Chained Prompt Lookup Engine (49.1 ns per Query)

1. **Architecture & Zero-Allocation Invariant**:
   - Implemented `ops::PromptLookup` using two contiguous flat arrays (`head_` sized to $2^k$, `next_` sized to token count).
   - Zero node heap allocations during decode; memory cost is just 4 bytes per token.
   - Microbenchmarked on Apple Silicon at **49.1 ns per query** (15x faster than Tirmazi's fastest benchmark, 2,800x faster than llama.cpp original).
2. **Hybrid Integration with MTP**:
   - When MTP head confidence drops below 0.35, Prompt Lookup supplies the remaining candidate tokens to fill the 5-draft batch.
   - Proposed tokens are verified in linear causal order on the GPU in the existing target verifier with zero extra forward passes and zero KV cache scatter complexity.
   - Increases candidate proposals (+1,293 tokens across 20 benchmark items) while maintaining 100% accuracy on math and coding.

## 2026-09-27 — Constrained Decoding Dispatch & Empty Metal Command Guard

1. **Constrained Decoding Multi-Stage Scheduling**:
   - In constrained decoding (`ConstraintMode::TokenMask`, active when tools or grammars are used), Splash splits each step into Stage 1 (Draft) and Stage 2 (Target Forward).
   - In unconstrained decoding, target verification is always encoded, so the command graph always contains dispatches.
   - In constrained decoding (`draftForMask = true`, `verify = false`), bypassing draft compute on exact prompt lookup matches left `commandGraph` empty (0 dispatches).
   - Prompt lookup draft bypass is restricted to unconstrained steps (`!constrained`), ensuring tool-calling passes always run full draft dispatches and maintain RoPE position tables.
2. **Defensive Metal Command Submission**:
   - Metal backend previously threw `MetalBackendError("Metal command must contain a dispatch")` if `commandGraph` had 0 dispatches.
   - `MetalBackend::submitCommandAsync` now defensively returns an immediately completed `CommandTicket` with zero timing overhead instead of throwing.

## 2026-09-27 — Grammar-Pruned Speculative Verification

1. **The Grammar-Draft Mismatch Bottleneck**:
   - In constrained decoding (`ConstraintMode::TokenMask`), `ConstrainedDecodeTicket::takeMaskRequests` previously drafted proposals, emitted `ModelMaskRequest` to the Python frontend, and immediately submitted `TargetForward` on all 8 rows before the grammar bitmask arrived.
   - Because MTP drafted without syntax constraints, draft token 0 violated JSON syntax on 76% of steps. The 48-layer target verifier loaded 60+ MoE experts from SSD across 8 rows (~150 ms) only to reject draft token 0, resulting in a ~6.25 tok/s floor.
2. **Grammar Mask Early Pruning (`takeMaskRequests`)**:
   - `ConstrainedDecodeTicket` now transitions to `Stage::WaitingMask` immediately after drafting without scheduling `TargetForward`.
   - Once the CPU grammar bitmask arrives from Python (~0.16 ms), the engine inspects `entry.maskWords` to count how many consecutive draft tokens are permitted by the grammar (`validDrafts`).
   - `impl_.mtpProposed` is pruned to `validDrafts`, and `laneResult.maximumRetained` is capped to `1 + validDrafts`.
   - If draft token 0 is illegal, `TargetForward` verifies only 1 row (the anchor), touching ~10 cached experts (~15 ms) instead of 60+ experts across 8 rows (~150 ms).
   - If draft tokens are legal, the verifier checks only the legal prefix.
   - Result: decode throughput during tool calls jumped from 5.6–6.9 tok/s to 45.2–49.6 tok/s (8.5x speedup; ITL p50 dropped from 161 ms to 22 ms).

## 2026-09-27 — Gated Smart PLD, Proactive Masked Drafting, and Adaptive Mode

1. **Gated Smart Prompt Lookup (`PromptLookup.cpp`)**:
   - Blind prompt lookup previously hurt MoE decode speed because false-positive matches forced unneeded SSD expert reads.
   - Added `minMatchLength` (default 4 via `SPLASH_PLD_MIN_MATCH`) and `requireUnambiguous` (default true via `SPLASH_PLD_UNAMBIGUOUS`).
   - The engine checks backward continuity against prompt history. If the matching n-gram has multiple conflicting continuations in the prompt, or is shorter than 4 tokens, it refuses to propose.
2. **Proactive Grammar-Masked Drafting (`Qwen4ExpTarget.cpp`)**:
   - In step $N-1$, when the target verifier commits, row `retained` of `entry.maskWords` contains the legal mask for the next token after the new anchor.
   - This row mask is captured into `entry.nextDraftMask` and passed to `runMtpDraft()` as `buffers.draftMask`.
   - When MTP scans its 1,024 top slice candidates on CPU, any candidate violating the mask is skipped.
   - The chosen draft token 0 is guaranteed to satisfy the grammar, boosting valid draft chain lengths.
   - Controllable via `SPLASH_MTP_MASKED_DRAFT` (default 1).
3. **Adaptive Mode Detection (`Runtime.mm`)**:
   - The engine tracks `inThinkingPhase` based on token `248069` / `151668` (`</think>`).
   - During `<thought>`, PLD is skipped to avoid inserting prompt instructions into internal reasoning.
   - After `</thought>`, PLD and proactive grammar masking are active for fast tool parameter generation.
   - Controllable via `SPLASH_ADAPTIVE_MODE` (default 1).

## 2026-09-27 — Speculation Stall Resolution: Stale Mask Pruning & Sequential Decode Width

1. **Resolution of Stale Grammar Mask Poisoning (`SPLASH_MTP_MASKED_DRAFT=0`)**:
   - Capturing row `retained` of `entry.maskWords` into `entry.nextDraftMask` was based on the flawed assumption that row `retained` predicted the continuation of `nextAnchor`.
   - In reality, when a draft is rejected, `nextAnchor` is freshly sampled by the target verifier; Python simulated row `retained` assuming the *rejected* token was accepted.
   - Constraining MTP with this stale mask forced MTP to generate tokens from the rejected branch, resulting in 0% draft acceptance and collapsing tool decode throughput to 3.3–7.6 tok/s.
   - Cleared `entry.nextDraftMask` and disabled `SPLASH_MTP_MASKED_DRAFT` (default 0). Unconstrained drafting combined with post-draft validation in `takeMaskRequests` cleanly preserves grammar validity without poisoning the draft head.
2. **Sequential Decode Width Capping (`SPLASH_MAX_BATCH_WIDTH=1`)**:
   - MTP speculative drafting in `Runtime.mm` is single-lane only (`lanes == 1`).
   - When multiple requests were admitted, `Scheduler.cpp` batched them into width 2 (`b2`), which hard-disabled MTP speculation for both requests (`buffers.mtpShadow = true`), dropping decode from 45 tok/s to 6 tok/s.
   - Added `SPLASH_MAX_BATCH_WIDTH` to `Scheduler.cpp` and set it to 1 in `splash-flashnext-server.sh`. Capping decode batch width to 1 serializes requests so each runs at 45 tok/s with speculation active, completing faster than concurrent non-speculative execution.
3. **Agent Client Concurrency & Compaction Bounds**:
   - In `~/.omp/agent/config.yml`: set `task.maxConcurrency: 1` and `compaction.reserveTokens: 16384`.
   - In `~/.omp/agent/models.yml`: restored `contextWindow: 94208` for Swift Flash-Next models.

## 2026-09-27 — Strict Engine Admission Concurrency & PLD-First Tool-Calling Speculation

1. **Strict Engine Admission Concurrency (`SPLASH_MAX_CONCURRENCY=1`)**:
   - Even when `SPLASH_MAX_BATCH_WIDTH=1` forced decode batches to width 1, `Runtime::begin` in `Runtime.mm` previously admitted up to 4 concurrent slots (`kLaneCount = 4`).
   - When `omp` fired background requests (mid-turn speculative compaction handoffs or title generator), the engine admitted two requests simultaneously (`active_cells: 2`).
   - The scheduler interleaved single-width decode steps between Request A and Request B, doubling step latency and causing severe expert-cache thrashing across two disjoint contexts (reducing decode speed to 2.7–3.5 tok/s).
   - `admitIdleSlot` in `Runtime.mm` now restricts assignable slots to `maxSlots` based on `SPLASH_MAX_CONCURRENCY` (or `SPLASH_MAX_BATCH_WIDTH`). Any secondary request receives `StateFailure::ConcurrencyLimit` and waits in the scheduler resource queue until the active request finishes. The active request retains 100% GPU bandwidth and cache locality.
2. **PLD-First Speculative Drafting in Tool-Calling Mode (`Qwen4ExpTarget.cpp`)**:
   - During tool calling (after `</think>`), linear MTP drafts frequently fail against the strict Lark tool grammar, causing early-pruning in `takeMaskRequests` to truncate proposals to 0 rows (falling back to 144 ms single-token steps = 6.9 tok/s).
   - PLD was previously chained *after* MTP, using MTP's unverified draft tokens as its search query, meaning PLD was never called on tool schemas unless MTP guessed them first.
   - Refactored `Qwen4ExpTarget::addVerify` to run PLD *first* during non-thinking / tool-calling mode using the committed anchor context. When PLD finds an exact match in the prompt (e.g. tool signatures, parameter names, file paths), it drafts up to 4 tokens with 100% precision, bypassing MTP neural drafting and guaranteeing grammar acceptance.
   - Defaulted `SPLASH_PROMPT_LOOKUP=1` in `splash-flashnext-server.sh` and disabled `midTurnEnabled` in `~/.omp/agent/config.yml`.
## 2026-09-30 — Promotion of Slipstream-V2 as Daily Default & Upstream Architecture Audit

1. **Promotion of `slipstream-v2` as Daily Serving Engine**:
   - `slipstream-v2` delivers 51.9 tok/s overall decode (peaking at 67.7 tok/s) and 51.8 tok/s at $T=0.7$ with exact rejection sampling, zero memory leaks, and 28% lower prefill staging latency.
   - All standard launchers in `~/models/bin/` (`slipstream-server.sh`, `swift-flashnext-server.sh`, `splash-flashnext-server.sh`) now target `slipstream-v2` and `local/swift-qwen38-flash-next-v3`.
   - Work Hub (`INDEX.html`) updated with new direct launchers, updated benchmark metrics, and architecture summaries.
   - Deleted unused 220 GB GGUF artifact (`Swift-Qwen3.8-Flash-Next-v3-ds4.gguf`), leaving 321 GiB free disk space.
2. **Upstream Architecture Audit & Evaluation**:
   - **Splash Persistent Prefix Cache (`origin/feature/persistent-prefix-cache`)**:
     - Introduces `PersistentCache`, `StateGroupCache`, and `DraftKvCache`. Uses 32-token page alignment for sliding window KV, APFS hole-punching for disk storage, and SQLite extent indexing.
     - Implements 4,096-entry demand fingerprinting and SSD write-credit admission pacing (256 GiB/h).
     - *Decision*: Prime candidate for next major architectural upgrade in `slipstream-v2` to make agent turn 2+ TTFT near-zero.
   - **ds4 Metal Router Softplus Precision (`0719a0b`)**:
     - Uses a 4-term Taylor polynomial expansion `em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)))` when $e^x < 0.03125$ to fix catastrophic cancellation and precision loss in Metal FP32.
     - *Decision*: Evaluate integrating into `gdn_primitives.h` softplus decay calculation to harden numerical precision on deep recurrent sequences.
   - **ds4 Concurrent Engram Reader (`6c00e2d`, `077a257`)**:
     - *Decision*: Declined for `slipstream-v2`. `slipstream-v2` already executes `ngram_embedding_gather` directly on the Metal GPU with memory-mapped tables, completely bypassing CPU file descriptor `pread` calls.
   - **llama.cpp Draft Batch Cap (`63b61fa45`)**:
     - *Decision*: Confirmed already implemented by construction in `slipstream-v2`. Verification buffers in `Qwen4ExpTarget.cpp` use strict single-lane sizing (`liveRowsPerLane = 1 + drafted`), avoiding the 4–6 GB memory overhead observed in llama.cpp.

## 2026-09-30 — Comprehensive Rebranding to Slipstream-v2 with Compatibility Aliasing

1. **User Surface & CLI**:
   - Created primary binary target `build/slipstream-v2`, metallib `build/slipstream-v2.metallib`, and CLI script `./slipstream-v2`.
   - Provided symlinks and forwarders (`./slipstream`, `./splash`, `build/slipstream`, `build/splash`) to preserve seamless invocation across all existing scripts and workflows.
2. **Server & Protocol Identifiers**:
   - Web server brand: "Slipstream v2", model `owned_by: "slipstream-v2"`, keepalive `: slipstream-v2-keepalive\n\n`, and thread names `slipstream-v2-*`.
   - Tool grammar root schema renamed to `__slipstream_v2_root`.
   - Crash trace logs routed to `~/Library/Logs/Slipstream-v2/crash`.
3. **Metrics & Observability**:
   - Emits primary metrics under `slipstream_v2_*` prefix, including `slipstream_v2_info` and `slipstream_v2_memory_pressure`.
   - Emits `slipstream_*` and `splash_*` aliases to ensure external scrapers and dashboards continue functioning without disruption.
4. **Client Integrations**:
   - OpenCode and Codex configured with `slipstream-v2` provider and model prefix `slipstream-v2/<model>`.
   - Defaults to port 8090 across `install/launcher.py` and Work Hub (`INDEX.html`).

## 2026-09-30 — Public Release Alignment: Publishing as Slipstream & Archiving Old Directory

1. **Workspace and Release Alignment**:
   - Archived older inactive workspace `../slipstream` to `../slipstream-orig`.
   - Symlinked `../slipstream` directly to `slipstream-v2`, allowing GitHub release, user documentation, and local scripts to use `Slipstream` uniformly without naming fragmentation.
   - Public model distribution paths clarified: base Flash-Next V3 (`nitinpanj/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF`) and Swift KV-sparse variant (`nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF`).
2. **Upstream Contribution Stance**:
   - Package Flash-Next architectural extensions, SSD expert streaming, and PLD speculative drafting as an upstream PR/patch to Incoai Splash.

## 2026-10-04 — Restore Flash-Next MTP Speculative Drafting (`buffers.mtpEnabled`)

1. **Restoration of `buffers.mtpEnabled`**:
   - In `runtime/model/Runtime.mm`, restored `buffers.mtpEnabled = mtpDrafting()` in `encodeTargetVerifyBatchForward()`.
   - In multi-architecture merge `cf4f91c`, the assignment had been omitted when introducing `isProposing`. Because `mtpEnabled` defaulted to `false`, `Qwen4ExpTarget::addVerify()` skipped both MTP draft passes and Prompt Lookup Decoding, dropping throughput to ~15 tok/s (0% acceptance rate).
   - Restoring the flag re-engages the MTP draft head and Prompt Lookup engine for Flash-Next, returning decode speed to ~40–50+ tok/s.
