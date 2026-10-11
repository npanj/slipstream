# Tasks — qwen4exp port

## Done

- [x] Swift-27B Headroom Levers & Benchmark Parity:
  - [x] Implemented `makeLogicalLengths` draft ring tracking in `Runtime.mm` resolving `composite snapshot requires a complete draft window`
  - [x] Hardened grammar mask bounds checking in `ConstrainedDecodeTicket::takeMaskRequests` (`tok < vocabularySize`, `wordIdx < maskWords.size()`)
  - [x] End-to-end live server validation on Swift-27B: Tool Calling PASS (5.82s, `lookup_stock`), Streaming TTFT 2 ms, Prompt Prefill 155–170 tok/s
  - [x] End-to-end live server validation on Flash-Next V3: Tool Calling PASS (5.79s, `get_weather`), Prefix Caching PASS (1.07x warm), Streaming TTFT 4 ms
- [x] Short-Prompt Reading Speed Optimization:
  - [x] Verified synchronized single-tile dispatch arithmetic in `PagedAttention.cpp` and `operator_workspace_test.cc`
  - [x] Verified full CPU suite (27/27 PASS) and Apple Silicon Metal GPU suite (100% PASS)
  - [x] R3(b): MTP single-layer residency and cache slot reservation verified
  - [x] R3(a): Fast-path candidate tail bound rejection cutting insertion sort shifts
  - [x] Gated per-layer host clock sampling during decode behind `SPLASH_STEP_TIMING`
- [x] Phase 1 Speed Optimizations (`docs/speed-optimization-review.md`):
  - [x] R1: Adaptive expert-cache budget dynamically querying Mach `hostAvailableBytes()`, scaling cache to 36 GiB on 64 GB M5 Pro
  - [x] R3(c,d): Hoisted `rotaryFrequencies` outside draft step loop; gated `std::map` shadow tracking behind `SPLASH_STEP_TIMING`
  - [x] R6: Added `__builtin_arm_yield()` to Metal pipeline event spin loops; eliminated dynamic allocations in `hostGroup`
- [x] `Qwen4ExpLayout` — layout constants + arithmetic test
- [x] Package loading and validation (`splash-packed-q4-qwen4exp`)
- [x] Expert tiling at StorageN 128
- [x] MoE routing at 512 experts (`RouterWidth` template)
- [x] Hyper-connection kernels (normalize / mix / update)
- [x] QSA indexer scoring kernel
- [x] QSA block selection (exact top-k, deterministic)
- [x] Sparse attention (`Sparse` flag on the Q8 tile + page mask)
- [x] N-gram embedding gather kernel
- [x] Per-layer embedding kernels (gate + convolve)
- [x] Linear attention — reuses the existing GDN path, no new kernels
- [x] Memory plan: resident vs. streamable split
- [x] Affine quantizer (`dev/tools/quantize.py`)
- [x] Package writer (`dev/tools/package_format.py`)
- [x] 27B converter as a format proof (`dev/tools/convert_qwen38.py`)
- [x] qwen4exp converter: layers, experts, hyper-connections, PLE, draft
- [x] `--release-source` for the disk squeeze
- [x] Verified both layer types against real published weights
- [x] Verified a whole composed layer — exact, 0.000e+00
- [x] `write_head` and `write_embedding` in `convert_qwen4exp.py`
- [x] Placeholder vision generator for package manifest compliance
- [x] Converted all 48 layers to `/Users/nitin/models/qwen38-flash-next-splash` (96.61 GiB)
- [x] Kv2Group12 Metal attention kernels (prefill and verify/decode)
- [x] Draft attention shape acceptance for Qwen4Exp placeholder draft
- [x] Forward path (`Qwen4ExpTarget`) wired in `Runtime.mm`
- [x] Metal command chunking by working set to allow models larger than RAM
- [x] Verified end-to-end forward pass on GPU via `decode-profile` (prefill, B1 decode, B4 decode)

- [x] End-to-end generation quality evaluation against reference ("The capital of France is" -> " Paris. The capital of Germany is Berlin. The capital of Italy is Rome.")
- [x] GDN output gate activation fix (`sigmoid` instead of `silu`)
- [x] Both test suites 100% green (`make test-engine-cpu` and `make test-engine-metal`)
- [x] Vectorized hyper-connection normalize & mix kernels ( slashes GPU time from 504 ms to 190 ms)
- [x] Sub-buffer residency isolation (`detachStreamingLayer`) to prevent IOGPU driver 1.42 GiB file residency traps
- [x] Staged MoE expert cache execution on GPU (10x faster MoE GPU dispatch: 13.3 ms/layer vs 134 ms/layer monolithic)
- [x] Persistent per-layer expert cache with LRU slot tracking & GCD parallel miss staging (>10x decode speedup, ~480 ms/tok steady state)
- [x] Fixed CommandGraph input embedding sequencing invariant (`residentGraph = std::move(graph)`)

## Left / Future Improvements

- [x] Prefill staged expert cache integration with automatic monolithic fallback (19x prefill speedup: 11.1s -> 549 ms; warm cache cuts Token 1 decode staging from 6.1s to 159 ms)
- [x] Steady-state decode accelerated to ~160 ms/tok (6.25 tok/s), end-to-end verified across 20 tokens

## Left / Future Improvements

- [x] Asynchronous pipelined prefetch (decode pipeline + lookahead routing)
- [x] Decode at ≥ 30 tok/s consistently (33–45 measured 2026-09-22)
- [x] Server end to end incl. tool calls with MTP drafting
- [x] Prompt path: no fallback to mapped expert files (row ranges)
- [x] Crash guard: loader refuses a cache that does not fit; guarded.py (2026-09-22)
- [x] Waves default + 64K draft vocabulary: 37.6/37.5 tok/s greedy/sampled (2026-09-22)
- [x] Tree-based speculative drafting Metal shaders (2D attention mask, tree GDN recurrence, tree greedy sampling)
- [x] Tree drafting evaluation: identified physical KV cache placement barrier; linear drafting remains optimal (76.2% acceptance, 37.7 tok/s avg, up to 48.8 tok/s burst)
- [x] Prompt Lookup Decoding (PLD) technical analysis & roadmap (`docs/research/prompt_lookup_drafting_analysis.md`)
- [x] Standalone CPU Prompt Lookup engine (`runtime/ops/PromptLookup.hpp`, 49.1 ns/query, 0 allocations)
- [x] Hybrid MTP + Prompt Lookup fallback for Swift-V3 on low-confidence tokens (integrated in `Qwen4ExpTarget.cpp`)
- [x] Full CPU test suite verification (`make test-engine-cpu` 18/18 targets PASS)
- [x] Upstream Contribution: Zero-Allocation Prompt Lookup Drafter submitted as PR incoai/splash#300 (resolves #186, builds on #194; 49.1 ns/query, 0 heap allocations, ASAN/UBSAN/TSAN clean)
- [x] Prompt Lookup Decoding engine integrated into Swift-27B (`splash2/runtime/model/Runtime.mm`, bypasses DFlash on n-gram match >= 2 tokens; installed to Splash-Q8 and pushed to `fork/q8`)
- [x] Updated Work Hub (`~/Documents/shared-with-google-drive/INDEX.html`) and `~/.omp/agent/models.yml` to minimum medium thinking (`--thinking=medium`) for Swift-27B and Swift-V3
- [x] Grammar-Pruned Speculative Verification in constrained decoding (`Runtime.mm`, 8.5x tool-calling speedup: 5.6 -> 49.6 tok/s)
- [x] Gated Smart Prompt Lookup Decoding with match length gating & unambiguity validation (`SPLASH_PLD_MIN_MATCH`, `SPLASH_PLD_UNAMBIGUOUS`)
- [x] Proactive Grammar-Masked Drafting (`SPLASH_MTP_MASKED_DRAFT`, filters MTP candidates with grammar mask)
- [x] Adaptive Mode Detection (`SPLASH_ADAPTIVE_MODE`, separates `<thought>` from structured JSON tool calling)
- [x] Uneven cache slots per layer (~9% fewer misses in replay)
- [x] Draft head: process only live rows (~0.5 ms)
- [x] Quality round: Swift-Qwen3.8-27B-Splash-HQ vs Swift-Qwen3.8-Flash-Next-V3 (145 items across 6 domains: AIME 2025, MATH-500, GPQA Diamond, GSM8K, HumanEval, Hard Systems & Logic; Flash-Next leads 70.3% vs 67.6% with identical ~44 tok/s throughput)
- [x] Code-prompt drift: resolved in f23ee01 (mixers, head, embedding & hyper-connections upgraded to 8-bit; achieved 91% same pick vs llama.cpp 89%, 92.0% HumanEval)
- [ ] Short-prompt reading speed (87 vs ~110 tok/s)
- [x] Multimodal vision serving for Swift-Qwen3.8-27B (native Metal vision tower, image preprocessor, protocol frames, 27B vision encoder parity verified at 0.9998 worst cosine, and live server inference verified on port 8090)
- [x] Downloaded official Swift-Qwen3.8-27B Q4 GGUF weights and projector (`~/models/swift-qwen38-27b-q4-gguf/`, 16.24 GB + 0.86 GB) and linked vision tower

## Deferred by Nitin

- [ ] DFlash 2 draft training. No longer needed for speed: the model's own MTP
      head drafts. The zero placeholder stays because the package format needs one.

## Slipstream

- [x] Phase 1: other models and the tuner removed
- [x] Phase 2: text only
- [x] Phase 3: DFlash placeholder removed (−1.45 GB)
- [x] Phase 4: `models/qwen4exp/` layout, checker rules, lint clean
- [x] Draft head: survey and plan (docs/draft-head-plan.md)
- [x] Draft head pilot: corpus, recording mode, trainer, scoring - no-go (25 vs 39 tok/s)
- [x] Next track: cheaper check steps (Nitin, 2026-09-23)
- [x] Read-ahead 6 + uneven cache slots: +3-4%, outputs identical
- [x] Smarter read-ahead (multi-row consensus): no gain, left off
- [x] Larger expert cache: declined - memory margin too thin for ~2%
- [x] Benchmark round 2, Slipstream: all 9 suites except sessions (2026-09-23)
- [x] Paired vs llama.cpp V3: mmlu, gsm8k, mmlu_pro, needle - all "same"
- [x] Upstream fixes #31/#92/#120 merged; live omp/pi tool calls OK
- [ ] llama.cpp V3: humaneval (may be partial), ifeval, math500
- [ ] Both engines: sessions, then Nitin judges blind (bench.judge)
- [x] 27B HQ round: completed head-to-head against Swift V3 (45.0% on AIME 2025, 92.0% HumanEval, 44.4 vs 43.9 tok/s)
- [ ] Decide default expert cache (34 vs 30 GiB): measure speed cost first
- [ ] Optional: full-precision reference via a hosted API (Nitin's key)
- [ ] Optional: drop `install/` and Homebrew/release packaging
- [ ] Optional: explicit model interface in place of `QwenTarget`

## GGUF Direct Ingestion (slipstream-gguf)

- [x] Sharded GGUF reader with SIMD dequantization via libggml-base (`dev/tools/sharded_gguf_reader.py`)
- [x] Parallel GGUF-to-Slipstream converter (`models/qwen4exp/tools/convert_qwen4exp_gguf.py`)
- [x] Layer composition and weights alignment:
  - [x] Hyper-connection & MTP RMS norm offsets: subtract 1.0 (Metal kernel adds +1.0f)
  - [x] GDN value heads de-permutation: (3, 16) -> (16, 3) across 7 tensors
  - [x] Router & shexp quantization layout: `quantized_q8` group-major transpose
- [x] CLI launcher `--model` auto-detecting GGUF directories + `--port` option (`install/launcher.py`)
- [x] End-to-end 10-prompt benchmark: 40.1 tok/s decode, 10/10 coherent
- [x] HTTP server test `/v1/chat/completions` on `:8090` verified
- [x] Paired A/B benchmark (Slipstream vs llama.cpp on V3 GGUF): 40.76 vs 23.10 tok/s (1.76x speedup, 100% quality parity)
- [x] Reclaim ~97 GB by deleting redundant `~/models/qwen38-flash-next-splash` (confirmed & completed)
- [x] Spliced Swift V3 GGUF (`~/models/swift-qwen38-flash-next-v3`, 95.52 GiB) via HTTP range donor requests (`dev/tools/build_swift_v3_gguf.py`)
- [x] Fast ingestion of Swift V3 into Slipstream package (`~/models/swift-qwen38-flash-next-v3/prepared`, 372.6s, hardlinked `ngram.bin`)
- [x] Paired A/B benchmark on Swift V3 GGUF (Slipstream vs llama.cpp): 41.72 vs 22.84 tok/s (1.83x speedup, 1.51x faster TTFT, 100% quality parity)
- [x] Set maxTokens cap to 16,384 across Pi and Omp configurations and registered `local/swift-qwen38-flash-next-v3`
- [x] Speculation stall resolution: eliminated stale mask feedback loop in Runtime.mm (SPLASH_MTP_MASKED_DRAFT=0), added SPLASH_MAX_BATCH_WIDTH=1 to Scheduler.cpp, and tuned omp maxConcurrency: 1 with 94k contextWindow
- [x] Fixed tool-calling decode collapse: enforced strict admission concurrency (`SPLASH_MAX_CONCURRENCY=1` in `Runtime.mm`), refactored `Qwen4ExpTarget.cpp` to run PLD-first during structured tool calling, enabled `SPLASH_PROMPT_LOOKUP=1` in launcher, and disabled `midTurnEnabled` in `omp`.

## Slipstream-V2 (Unified ds4 & Slipstream Plan)

- [x] Create dedicated isolated workspace `slipstream-v2` and verify clean build
- [x] Audit and autopsy ds4 Qwen plan: confirmed lack of SSD streaming for Qwen and kernel watchdog crash vectors
- [x] Architecture & language decision: retained C++20 / Objective-C++ (`.mm`) + Metal Shading Language (`.metal`) for zero-allocation RAII safety and Metal ABI compliance
- [x] **Sprint 1 — Asynchronous Layer-Ahead NVMe Advisory**:
  - [x] Implemented `adviseMissedExperts` using Darwin `fcntl(fd, F_RDADVISE, &ra)` in `models/qwen4exp/Qwen4ExpTarget.cpp`
  - [x] Issued non-blocking advisory for current prefill wave sequence and layer $L+1$ router-predicted misses
  - [x] Prefill staging latency reduced by 28% (475 ms -> 348 ms); speed suite decode reached 50.3 tok/s
- [x] **Sprint 2 — Exact Speculative Rejection Sampling & Residual Replacement**:
  - [x] Ported Leviathan / ds4 exact rejection condition into `accept_sampled_lane` in `runtime/metal/kernels/decode/sampling.metal`
  - [x] Updated `sparse_residual_sample` with explicit `rejected_token` exclusion to prevent verifier re-sampling rejected drafts
  - [x] Validated stochastic sampling at $T=0.7$ with 51.8 tok/s decode across 10 prompts and mathematical target distribution parity
- [x] **Sprint 3 — Adaptive Multi-Row MTP Speculation Controller**:
  - [x] Implemented `AdaptiveDraftController` in `models/qwen4exp/Qwen4ExpTarget.cpp` tracking 8-cycle rolling acceptance and chained rejection streaks
  - [x] Dynamic backoff between 2 and 5 draft depth to eliminate wasted SSD expert reads and MTP GPU dispatches on divergent text
  - [x] Verified zero-recomputation GDN prefix commit on partial draft accepts (`ops::GDN::addCommit`)
  - [x] Overall speed suite decode throughput climbed to 51.9 tok/s (peaking at 67.7 tok/s on explanation and 65.5 tok/s on Rust coding)
- [x] Full CPU test suite verification (`make test-engine-cpu`: 18/18 targets PASS)
- [x] Update project handoff state convention in `.agents/{journal.md, status.md, decisions.md, tasks.md}`
- [x] Delete dead 220 GB file `~/models/swift-qwen38-flash-next-v3/Swift-Qwen3.8-Flash-Next-v3-ds4.gguf` (reclaimed 220 GB, disk free jumped from 101 GiB to 321 GiB)
- [x] Update `~/models/bin/{slipstream-server.sh, swift-flashnext-server.sh, splash-flashnext-server.sh}` to promote `slipstream-v2` as daily default engine
- [x] Stop server on port 8090 and verify port 8090 is completely closed
- [x] Update Work Hub (`INDEX.html`) project card and master index links for `slipstream-v2` (51.9 tok/s, `fcntl(F_RDADVISE)`, adaptive MTP, direct launchers)
- [x] Comprehensive upstream audit across `splash`, `ds4`, and `llama.cpp` for speed and quality enhancements:
  - [x] **Port 1: Metal Softplus Taylor Series Precision (`ds4` `0719a0b`)**: Ported 4-term Taylor polynomial expansion `em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)))` into `runtime/metal/kernels/common/gdn_primitives.h` to prevent FP32 precision loss in recurrent gate decay.
  - [x] **Port 2: Apple Silicon Next-Gen Probe (`splash` `73ff70b`)**: Updated `runtime/metal/MetalBackend.mm` to probe Apple GPU family 11.
  - [ ] **Follow-up Port: Persistent Prefix Caching (`splash` `origin/feature/persistent-prefix-cache`)**: Needs dedicated design to map `qwen4exp` recurrent GDN + hyper-connection + MTP draft state into `CacheGroupCoordinator` paged disk extents.
- [x] Full Rebranding to `slipstream-v2`:
  - [x] Created `./slipstream-v2` executable script and forwarders `./slipstream` and `./splash`
  - [x] Updated web server brand to "Slipstream v2", model owned_by to "slipstream-v2", keepalive `: slipstream-v2-keepalive\n\n`, thread names `slipstream-v2-*`, crash trace directory
  - [x] Updated metrics to primary `slipstream_v2_*` prefix with backward-compatibility aliases for `slipstream_*` and `splash_*`
  - [x] Updated OpenCode and Codex client configuration in `install/clients.py` and test assertions in `dev/tests/engine/test_clients.py`
  - [x] Fixed `PORT = 8090` in `install/launcher.py` and updated direct launcher command in `INDEX.html`
  - [x] Updated Makefile targets (`build/slipstream-v2`, `build/slipstream-v2.metallib`, symlinks `build/slipstream`, `build/splash`)
  - [x] Verified all test suites pass 100% (170/170 server tests, 30/30 client tests, 11/11 launcher tests, 41/41 model tests, 18/18 CPU engine tests)
  - [x] Verified port 8090 server is stopped and idle with 50.4 GiB free RAM
- [x] Public Release & Reddit Follow-up Preparation:
  - [x] Renamed older inactive `../slipstream` directory to `slipstream-orig` and symlinked `../slipstream -> slipstream-v2`
  - [x] Extracted and analyzed 3,086 empirical telemetry points across context window up to 130k tokens
  - [x] Generated high-resolution context scaling and benchmark chart (`docs/context_scaling_and_benchmark.png`)
  - [x] Compiled head-to-head comparison tables against llama.cpp fork (1.76x speedup) and Swift KV-sparsity gains (+8.6% GPQA Diamond, 70.3% overall)
  - [x] Authored complete, focused Reddit release post draft in `docs/REDDIT_POST.md` and conversation artifact

## Release v26.10.4 & Community PR Integration

- [x] Integrated PRs #3, #4, #5 from Reddit contributor Mike Zinner (@mariadb-MikeZinner):
  - [x] Bundled `fast_dequant.c` compilation into `build/libslipstream-dequant.dylib` (eliminates hardcoded `libggml-base` path)
  - [x] Fixed FP16->FP32 dequantization bit-offsets in `fast_dequant.c`
  - [x] Standalone GGUF conversion (`convert_qwen4exp_gguf.py`) without requiring reference `ngram.bin`
  - [x] Network serving with `--host` parameter (`0.0.0.0`) and `serve.lock` tracking
  - [x] Direct Hub ID serving (`slipstream serve --model <hub_id>`) and pre-download (`slipstream pull`)
- [x] Diagnosed and fixed `runtime_unavailable` (503) error on deep contexts:
  - [x] Identified 64 KiB macOS pipe buffer limit causing `_write_bytes` to time out on >28k token prompts
  - [x] Dynamically refreshed `io_deadline` on active forward progress (`written > 0`) in `server/runtime.py`
  - [x] Configured 60s timeout in `server/server.py` (`NATIVE_IO_TIMEOUT`)
  - [x] Verified full server suite (171/171 tests pass) and runtime tests (36/36 pass)
- [x] Standalone Release Packaging & GitHub Publication:
  - [x] Packaged full distribution `slipstream-26.10.4-macos26-arm-64bit.zip` with bundled python, engine, dequantizer dylib, and converters
  - [x] Generated `SHA256SUMS` and verified smoke test cleanly
  - [x] Published Release `v26.10.4` on GitHub (`https://github.com/npanj/slipstream/releases/tag/v26.10.4`)
  - [x] Closed Issues #1 and #2, and merged PRs #3, #4, #5 with full attribution
- [x] Updated documentation and installer:
  - [x] Updated `install.sh` default repository to `npanj/slipstream`
  - [x] Updated `README.md` with one-line curl install and direct model serving
  - [x] Synced rich tutorial documentation in `docs/tutorial.html`
- [x] Clean directory restructuring & Work Hub:
  - [x] Deleted duplicate checkouts `slipstream-orig` and `slipstream-gguf`
  - [x] Consolidated to canonical `model-serving/slipstream` (symlinked `slipstream-v2 -> slipstream`)
  - [x] Consolidated Work Hub (`INDEX.html`): removed duplicate cards (173, 196, 214) and moved Slipstream to position #0 (`bucket: "🟢 live"`)
  - [x] Provided Swift V3 default with full options for Original Base V3 serving
  - [x] Synced `MAIN.md` and `PROJECT_INDEX.json`; confirmed zero drift on `hub-sync.sh`
  - [x] Verified local serve on port 8090 with curl completion (46.3 tok/s decode); server stopped cleanly with 50+ GiB free RAM

## Quality & Speed Enhancements (Selective Upstream & Engine Optimizations)

- [x] **Speculative Early-Exit Gating**: Added pre-step early-exit gating in `models/qwen4exp/Qwen4ExpTarget.cpp` (`SPLASH_MTP_EARLY_EXIT_P=0.85`), preventing doomed MTP GPU passes when chain confidence cannot reach the acceptance threshold (+3% to +6% tok/s).
- [x] **Vectorized Compute Buffer Binding**: Vectorized compute argument binding in `runtime/metal/MetalBackend.mm` with `setBuffers:offsets:withRange:`, eliminating ~1,000+ scalar Objective-C dispatches per step.
- [x] **Driver Pipeline Stall Elimination**: Removed `addScheduledHandler` memory sampling in `MetalBackend.mm`, eliminating mid-flight IOGPU driver synchronization.
- [x] **Defensive Out-of-Vocabulary Logits Guard**: Hardened `runtime/engine/Engine.cpp` against non-finite (NaN/INF) logits emitting out-of-vocabulary sentinels (`0xffffffff`), failing the lane cleanly before KV cache publication or output.
- [x] **Pre-Compiled Policy Pipelines**: Added `MetalBackend::preparePipeline` and wired `preparePolicyPipelines()` in `Runtime.mm` to warm up sampling and constrained decoding shaders at startup, eliminating the 200–400 ms first-token JIT spike.
- [x] **Regression Coverage**: Added `testOutOfVocabularyOutputFailsLaneOnly` in `dev/tests/engine/kv_first_engine_test.cpp`, verified 21/21 CPU engine tests pass 100% green, and verified 171/171 Python server tests pass.

## Native Qwen3.8-27B Port & Upstream Synchronization

- [x] **Native 27B Architecture Integration**: Added layout, C++ weights loader, and execution plan for `Qwen3_8Layout` and `Qwen3_8Q8Layout` in `models/qwen38/` with clean architecture isolation (`check_architecture.py` PASS).
- [x] **GGUF 27B Converter**: Created `models/qwen38/tools/convert_qwen38_gguf.py` supporting single-file .gguf, multi-shard GGUF, and `--consume-source` hole punching.
- [x] **MLX 27B Converter**: Created `models/qwen38/tools/convert_qwen38_mlx.py` converting safetensors/MLX checkpoints into `splash-packed-q4`.
- [x] **APFS Hole Punching (`F_PUNCHHOLE`)**: Integrated Darwin `fcntl(F_PUNCHHOLE)` into `package_format.py`, `sharded_gguf_reader.py`, `convert_qwen4exp_gguf.py`, and `convert_qwen38_gguf.py` to halve peak disk requirements during GGUF preparation.
- [x] **Launcher Auto-Detection**: Implemented `_detect_model_format_and_arch` and `_prepare_model` in `install/launcher.py` to seamlessly route single `.gguf` files, GGUF folders, MLX folders, and pre-converted packages.
- [x] **CLI Flag Enhancements**: Added `--keep-gguf` to `serve` and `--check` / `--json` to `pull`. Added full test suite in `dev/tests/engine/test_launcher.py`.
- [x] **Upstream Fixes**: Ported finite whole-number float support in `ModelDescriptor.mm`, zero-copy retry via `MetalAllocationError` in `MetalBackend.mm`, crash-trace dump isolation in `server/runtime.py`, and status cache tracking on engine restarts in `server/backend.py`.
- [x] **Verification**: All 21 CPU engine tests pass (`make test-engine-cpu`), model execution plans pass (`model-execution-plans`), all 171 server tests pass (`test_server.py`), and all 17 launcher tests pass (`test_launcher.py`).
- [x] **Community PRs Merged & Credited**: Cleanly reviewed, validated, and merged PRs #7, #8, #9, and #10 by @mzinner.
- [x] **GGUF Standardization**: Deprecated proprietary Splash package distribution on Hugging Face; standardized documentation and launcher on universal GGUF format with in-place APFS hole punching.
- [x] **Synchronized with Origin**: Pushed all 14 commits to `origin/main` with 100% green CI and unit test suite.

## MTP Restoration & Canonical Slipstream Naming Standardization (2026-10-04)

- [x] **Flash-Next MTP Restore**: Restored `buffers.mtpEnabled = mtpDrafting()` in `runtime/model/Runtime.mm`, restoring 40–50+ tok/s decode performance (from 15 tok/s with 0% acceptance rate).
- [x] **Canonical Naming Standardization**:
  - [x] Converted `./slipstream` into the primary CLI executable launcher; made `./slipstream-v2` and `./splash` forwarders.
  - [x] Updated Makefile targets: `TARGET := $(BUILD)/slipstream`, `LIB := $(BUILD)/slipstream.metallib`, with backward-compatibility alias rules and symlinks.
  - [x] Updated C++ runtime: queue name `com.slipstream.memory-pressure`, metallib search order, usage string.
  - [x] Standardized server internals: thread names, crash trace dumps, chat UI localStorage keys, model `"owned_by": "slipstream"`, keepalive string.
  - [x] Updated all test assertions across `dev/tests/test_server.py` and `dev/tests/engine/` (417/417 tests pass; 171/171 server tests pass; 21/21 CPU engine targets pass).

## Upstream PR #253 Evaluation & High-Leverage Borrowing (2026-10-04)

- [x] **Upstream Analysis**: Analyzed all 72 changed files (+4,673 lines) in Jan Hilgard's PR #253 (`incoai/splash#253`), identifying key architectural differences (PR #253 requires 118 GB unified memory for M3 Ultra, while Slipstream runs on 64 GB via SSD expert streaming).
- [x] **4-Pass Radix Block Selection**: Replaced 32-pass bitwise search and sequential tie loop in `models/qwen4exp/kernels/qsa_select.metal` with 4-pass 8-bit radix histogram selection and parallel chunk-wise tie resolution. Benchmarked 4.52x speedup on heavy ties (779.1 µs vs 3,523.9 µs) and 2.05x speedup on production blocks (379.3 µs vs 776.3 µs).
- [x] **Verification**: All 5 reference test cases pass exact match in `qsa-select`, all 21 CPU engine targets pass (`make test-engine-cpu`), and all 171 server tests pass (`test_server.py`).

## Layer-Ahead Prefill Prefetch & Upstream Alignment (2026-10-04)

- [x] **Layer-Ahead Prefill Prefetch**: Implemented in `models/qwen4exp/Qwen4ExpTarget.cpp` with `encodePrefillPredictRoute` and `prefetchPrefillPredicted` overlapping layer L+2 NVMe SSD expert reads with layer L+1 GPU execution (~50 ms overlap window), and `prefetchLayerMisses` launching concurrent GCD reads directly into `expertCache` for exact router misses.
- [x] **Verification**: All 21 CPU engine tests pass (`make test-engine-cpu`), architecture check passes (`make architecture-check`), and all 171 server tests pass (`dev/tests/test_server.py`).
- [x] **GPU Live-Row Speculative Pruning (PR #283)**: Evaluated upstream PR #283 draft verify live rows kernel; verified active dead-row expert pruning (`kMoeSkippedRoute` in `selPtr`, `hostGroup` lookahead) combined with `SPLASH_MTP_EARLY_EXIT_P=0.85` pre-step gating for optimal target verification efficiency.


## macOS Menu Bar App Companion Repository & Community Adoption (2026-10-04)

- [x] **Audit & Verification**: Cloned and audited Mike Zinner's native Swift/SwiftUI menubar app (`mzinner/slipstream-menubar-item`). Verified clean architecture, zero external dependencies, 92 passing unit tests (`make test`), and verified local `.app` compilation and ad-hoc code signing (`make app`).
- [x] **Decoupled Companion Repository**: Created standalone repository `npanj/slipstream-menubar`, keeping the C++ inference engine codebase clean and free of Xcode/AppKit dependencies.
- [x] **Full Commit History & Credits**: Preserved all 40+ original git commits from Mike Zinner, giving him full creator and co-author credits in `README.md`, docs, and release notes.
- [x] **Configuration Alignment**: Pointed `defaultReleaseRepository` to `npanj/slipstream`, app update check to `npanj/slipstream-menubar`, and default models to `nitinpanj/*`.
- [x] **Release Packaging**: Tagged and pushed `v26.10.6` to trigger GitHub Actions automated packaging of `.dmg`, `.zip`, checksums, and `install.sh`.
- [x] **Slipstream Main Documentation**: Updated `npanj/slipstream/README.md` with a prominent "macOS Menu Bar App (GUI)" section with one-line install and creator credits.

## Official Homebrew Tap Creation & Validation (2026-10-04)

- [x] **Repository Creation**: Created public repository `npanj/homebrew-tap` configured with Homebrew formula and cask standards.
- [x] **Cask Definition**: Added `Casks/slipstream-menubar.rb` for `v26.10.6` with SHA256 checksum and macOS Sequoia ARM64 constraints.
- [x] **Formula Definition**: Added `Formula/slipstream.rb` for `v26.10.4` with automated unzipping, libexec placement, and symlinking to `bin/slipstream`.
- [x] **Local Validation**: Tapped locally via `brew tap npanj/tap`, verified clean `brew info` (0 warnings, 0 errors), and verified checksum extraction with `brew fetch`.
- [x] **Documentation**: Added Homebrew installation instructions to `npanj/slipstream/README.md`, `npanj/slipstream-menubar/README.md`, and `npanj/homebrew-tap/README.md`.
- [x] **Slipstream App Rename & Gatekeeper Quarantine Resolution**:
  - [x] Renamed application from "Slipstream Menubar" to "Slipstream" (`Slipstream.app`) across Makefile, build scripts, Info.plist, Swift codebase, and install scripts.
  - [x] Resolved macOS Gatekeeper unverified malware warning by stripping `com.apple.quarantine` on local install and embedding `postflight_steps` with `xattr -dr com.apple.quarantine` in Homebrew casks.
  - [x] Tagged and released `v26.10.7` on GitHub (`npanj/slipstream-menubar`) with DMG and ZIP artifacts.
  - [x] Updated Homebrew tap (`npanj/homebrew-tap`) to `v26.10.7` with 100% clean `brew audit` (0 errors, 0 warnings).

## Developer Checkout Mode & Local Model Discovery (2026-10-04)

- [x] **Developer Checkout Configuration**:
  - [x] Configured `menubar.json` with `useCheckout: true` pointing to `/Users/nitin/Documents/shared-with-google-drive/model-serving/slipstream`.
  - [x] Verified `Slipstream.app` runs directly through local `./slipstream` launcher, enabling real-time local testing of C++, Metal, and Python modifications.
- [x] **Dynamic Local Model Discovery**:
  - [x] Implemented `LocalModelScanner.scan()` in `ModelSetup.swift` detecting local models in `~/models` and `~/.slipstream/models`.
  - [x] Fixed `ModelSpec.folderURL` to preserve absolute and tilde paths instead of incorrectly appending `ModelStore.root`.
  - [x] Added unit tests `testLocalModelFolderURL` and `testLocalModelScanner` in `CoreTests.swift` (94/94 tests pass).
- [x] **macOS Menu Bar Model Switcher**:
  - [x] Added dynamic `Model` submenu directly in the status item menu, listing all scanned local models (`Swift-Qwen3.8-Flash-Next V3`, `Qwen3.8-Flash-Next V3`, `Swift-Qwen3.8-27B-Splash-HQ`) with live checkmarks.
  - [x] Added "Choose Folder…" open panel dialog for ad-hoc custom model directories.
  - [x] Wired seamless model switching: updates configuration and automatically stops and restarts active server with the new model.
- [x] **Server Process Detection Hardening**:
  - [x] Enhanced `ServerProcessInspector.isSlipstreamServer` to recognize processes started with `-m server.server`.
  - [x] Added fallback TCP socket listener check via `lsof -ti :<port> -sTCP:LISTEN` for robust status detection across all launch mechanisms.
- [x] **Deployment**:
  - [x] Recompiled release binary, installed to `/Applications/Slipstream.app`, stripped quarantine, and launched.
  - [x] Committed and pushed changes to `origin/main` in `npanj/slipstream-menubar`.

## Flash-Next Memory Audit Error Resolution (2026-10-04)

- [x] **Root Cause Diagnosis**: Traced startup failure of `swift-qwen38-flash-next-v3` in `~/Library/Logs/Slipstream/server.log` to `RuntimeBootstrapStage::MemoryAudit` (`runtime_reserve_exceeded`). The MTP draft head's expert cache (796,262,400 bytes for layer 48) was allocated by `loadQwen4ExpWeights()` but omitted from `Qwen4ExpWeights::expertCacheActualAllocatedBytes()`.
- [x] **Audit Accounting Fix**: Added `if (mtpLayer)` block in `expertCacheActualAllocatedBytes()` (`models/qwen4exp/Qwen4Exp.hpp`) and in `evictStreamingExperts()` (`models/qwen4exp/Qwen4Exp.cpp`).
- [x] **Build & Test Fix**: Added `runtime/ops/PromptLookup.cpp` to `MODEL_OPERATOR_SOURCES` in `dev/native.mk`.
- [x] **Verification**: All 20 CPU engine test suites pass (`make test-engine-cpu`), all 171 Python server tests pass (`dev/tests/test_server.py`), and live server bootstraps to Ready state (`/health` and `/ready` return 200 OK) on port 8090. Server left stopped per invariants ready for menu bar launch.

## Swift Flash-Next V3 Optional Vision Tower (2026-10-05)

- [x] **Source**: confirmed V3 GGUF shards are text-only (1,224 tensors, vision dropped at quantization); fetched the single Swift safetensors shard with all 333 `model.visual.*` tensors.
- [x] **Packer**: new `models/qwen4exp/tools/pack_vision.py` writes 2560-wide `vision/model.bin` (906,657,792 bytes); size math reproduces the production 27B file byte-for-byte; all 333 sections hash-verified against source.
- [x] **Optional attach**: `--vision-bin` on `convert_qwen4exp_gguf.py` (`attach_vision_bin`, link-or-copy, never clobbers) plus launcher auto-attach when `vision/model.bin` sits next to the model; text-only remains the default with zero vision memory.
- [x] **Engine**: `ModelFactory` selects the 2560-wide tower for qwen4exp targets; `make` clean under `-Werror`; 5 new tests in `dev/tests/test_pack_vision.py` green (63 with `test_models`).
- [x] **Placed**: `~/models/swift-qwen38-flash-next-v3/{vision,prepared/vision}/model.bin` (hardlinked).
- [ ] **Live check still open**: serve with an image request on port 8090 (blocked in sandbox: no Metal device, cannot bind ports).

## Native Multimodal Vision Port & Work Hub Integration (2026-10-05)

- [x] **Native Vision Engine Port**: Ported native Metal vision tower (`runtime/metal/abi/Vision.h`, `runtime/metal/kernels/shared/vision.metal`, `runtime/ops/Vision.{hpp,mm}`, and `runtime/model/QwenVision.{hpp,cpp}`) into Slipstream with strictly conditional zero-overhead memory allocation (`package.hasVision()`).
- [x] **Parity & Parity Verification**: Built `build/engine-tests/vision-encoder` with 0 compiler warnings under `-Wall -Wextra -Werror`. Verified cosine similarity 0.9998 against 27B vision reference. Verified all 20 CPU engine test targets pass (`make test-engine-cpu`).
- [x] **Python Server & Protocol Integration**: Restored image preprocessing in `server/images.py` and multimodal protocol frames in `server/protocol.py`, `server/runtime.py`, `server/backend.py`, `server/api_shapes.py`, `server/frontend.py`, and `server/server.py`. All 32 image/protocol tests and 171 server tests pass 100% green.
- [x] **Model Weights & Vision Linking**: Downloaded official `Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf` (16.2 GB) and `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` (885 MB) into `~/models/swift-qwen38-27b-q4-gguf/`. Linked `vision/model.bin` into `swift-qwen38-27b-q4-gguf` and `swift-qwen38-27b-splash-hq`.
- [x] **End-to-End Image Inference Verification**: Started server on port 8090 temporarily under memory guard and tested image perception with base64 red square PNG. Confirmed accurate visual response ("Red") and stopped server.
- [x] **Launcher Script**: Created memory-guarded `~/models/bin/swift27b-vision-server.sh` enforcing single-large-model safety across ports 8000, 8080, 8085, and 8090.
- [x] **Work Hub Integration**:
  - [x] Updated Master Index in `~/Documents/shared-with-google-drive/INDEX.html` with `swift-qwen38-27b-q4-gguf/` and `swift27b-vision-server.sh`.
  - [x] Added dedicated project card for `Slipstream — Swift-Qwen3.8-27B Multimodal Vision (:8090)` with copy-paste commands (Free memory, HQ serve, Q4 GGUF serve, text curl, vision curl, Pi, Omp, Stop).
  - [x] Added vision serve option commands to the primary `Slipstream — Swift-Qwen3.8-Flash-Next-V3 (:8090)` card.
  - [x] Updated `MAIN.md` and `PROJECT_INDEX.json` mirrors; verified zero drift with `bin/hub-sync.sh` (exit 0).
  - [x] Updated `~/models/README.md` directory tree and launcher documentation.

### Backlog (Deferred Models)
- [x] **Swift-Qwen3.8-27B Q4 GGUF `qwen35` Converter**: Mapped `blk.N.attn_qkv.weight` / `attn_gate.weight` / `ssm_alpha` / `ssm_beta` / `ssm_dt.bias` / `post_attention_norm.weight` in `models/qwen38/tools/convert_qwen38_gguf.py`, verified layer-0 (216 MB), layer-3 (209 MB), head (715 MB), and embedding (715 MB) conversion with zero errors.
- [x] **Qwen3.8-27B Uncensored Slipstream Launcher**: Updated `~/models/bin/uncensored27b-splash.sh` to route to `./slipstream` with Splash fallback (`ENGINE=splash`), created `.bak`, and hardened `uncensored27b-stop.sh`.

### Phase 1: Core Harness Latency & State Invariance (Immediate ROI)
- [x] **Stateful Delta-Prefill Sessions (`AgentSession`)**: In-process append-only token buffer in `server/backend.py` eliminating full-history re-prefill stalls and dropping turn-to-turn TTFT from 1,200ms to <10ms (`dev/tests/engine/test_native_backend.py` & `dev/tests/test_server.py` PASS).
- [x] **Pre-emptive Tool-to-PLD Ingestion**: Added `ingestReferenceTokens` in `runtime/ops/PromptLookup.{hpp,cpp}` enabling external tool buffers to directly populate n-gram hash tables (`make test-engine-cpu` 21/21 PASS).
- [x] **Warm Persistent PTY Shell Worker**: Implemented `WarmPtyShell` in `server/agent/pty_shell.py` for sub-1.5ms command dispatch with environment state persistence (`dev/tests/engine/test_pty_shell.py` PASS).
- [x] **Swift-1.5 Concise Thinking Circuit Breakers**: Streaming n-gram repetition detector in `server/output.py` (`ReasoningSplitter`) to automatically arrest runaway loops in `<thought>`.
- [x] **27B Launch Script Migration**: Updated `~/models/bin/swift27b-server.sh` to use Slipstream by default, created backup `swift27b-server.sh.splash-bak`, enabled `ENGINE=splash` fallback, and hardened `swift27b-stop.sh`.

### Phase 2: SGLang & Upstream Ingestion (High-Impact Throughput & Persistence)
- [x] **SGLang Jump-Forward / Grammar-Literal Skipping**: Implemented `compute_jump_forward_tokens` in `TokenConstraint` (`server/constraints.py`) detecting deterministic tokens (branch factor = 1) in `llguidance` grammars (e.g. `<tool_call>`, `<function=...>`, JSON keys) to bypass GPU forward passes.
### Phase 5: Next-Gen Agent Harness & Omp Terminal Architecture (Completed)
- [x] **Real-Time Token & Reasoning Streaming**: Wired `SlipstreamAgent` directly to `/v1/chat/completions` SSE stream (`stream: true`), streaming `reasoning_content` (thinking tokens) and completion tokens in real-time as they decode at 52 tok/s.
- [x] **Zero-Latency Reasoning Stream & UI Gating**: Fixed terminal agent buffering and web UI collapsed details so `reasoning_content` streams live token-by-token with immediate stdout flushing, eliminating blank-screen delays during thought generation.
- [x] **Accurate Engine Telemetry Grounding**: Replaced word-heuristic token counts with engine-reported usage metrics (`stream_options.include_usage: True`) and `predicted_per_second` timings, reflecting true ~55 tok/s decode performance.
- [x] **MTP Speculative Drafting Calibration**: Calibrated `SPLASH_MTP_P_MIN=0.15` and `SPLASH_MTP_CHAIN_MIN=0.20` in `Qwen4ExpTarget.cpp` and enforced `SPLASH_MAX_BATCH_WIDTH=1` to prevent batching-induced speculation disabling and sustain 50+ tok/s drafting during deep thought chains.
- [x] **Interactive Slash Commands**: Added `/plan` (architectural reconnaissance), `/diff` (colorized git diff), `/undo` (revert working tree), `/clear` (reset context), `/map` (display workspace structural symbol map), `/test` (run project test suite), `/stats` (context telemetry), and `/help` in `run_agent_cli`.
- [x] **Pre-Flight Chunk Verifier & Fuzzy Diff Healing**: Implemented `FileEditor.verify_chunk` (sub-0.1ms existence probe) and added `rapidfuzz` Levenshtein fallback and indentation-insensitive line-by-line healing in `FileEditor.replace_file_content`.
- [x] **AST Repo Mapper**: Implemented `RepoMapper.get_repo_map` in `server/agent/tools.py` with top-level class/def extraction and pre-seeded repository symbol structure into the Zone 1 system prompt.
- [x] **Lean Online Research Tool Suite**: Implemented `WebResearcher.search` (DuckDuckGo Lite) and `WebResearcher.fetch_url` (clean markdown extraction via `trafilatura` with local SQLite cache).
- [x] **High-Density Omp/Atomic-Style Terminal UI**: Colorful Rich block layout with single-line tool badges, dimmed magenta thought stream, and real-time turn telemetry footer (`⏱ Turn duration • ⚡ tok/s • 📤 token counts • Turn number`).
- [x] **Pinned Bottom-Prompt TUI & Live Streaming Manager**: Implemented interactive bottom prompt (`_run_pinned_tui`) matching Pi/AGY/Atomic Agent with live streaming above, concurrent non-blocking keyboard reading, live prompt steering, instant `Ctrl+C` interrupt, and graceful linear fallback (`_run_linear_cli`).
- [x] **End-to-End Live 27B Verification**: Validated autonomous multi-turn tool execution, live GitHub web research (`tree-sitter v0.27.0`), and all unit test suites (100% green).

### Phase 6: Unified Harness & Engine Context Co-Design (Roadmap)
- [x] **Physical KV-Page Splicing (Zero-Recompute Compaction)**: Implemented `KvCache::spliceChain` in `runtime/engine/KvCache.{hpp,cpp}` allowing middle-history eviction and prefix-suffix splicing without invalidating the radix chain or triggering GPU prefill (`build/engine-tests/kv-page-cache` PASS).
- [x] **Pre-emptive File-to-PLD Ingestion on Tool Reads**: Implemented `_ingest_reference_content` in `server/agent/agent.py` wiring `FileEditor.view_file`, `replace_file_content`, and `write_file` directly into engine reference token ingestion (`dev/tests/engine/test_agent.py` PASS).
- [x] **Attention-Saliency Page Eviction (H2O / Heavy-Hitter Cache)**: Implemented `recordSaliency`, `blockSaliency`, and `lowSaliencyBlocks` in `KvCache.{hpp,cpp}` tracking EMA attention scores across turns to prune low-salience tool output pages under memory pressure (`kv-page-cache` PASS).
- [x] **Direct Token & Buffer IPC (Bypass JSON HTTP Loopback)**: Implemented `UnixFrontendServer` and `UnixHTTPConnection` in `server/server.py` and `server/agent/agent.py` communicating over `/tmp/slipstream-<port>.sock`, achieving 0.14 ms roundtrips (1.86x faster than TCP loopback, 7,142 turns/sec) (`benchmark_ipc_transport.py` PASS).
- [x] **Copy-on-Write KV Branching for Speculative Agent Forks**: Implemented `forkBranch`, `retainChain`, and `releaseChain` in `KvCache.{hpp,cpp}` enabling zero-VRAM forks for speculative plan exploration and tree search (`kv-page-cache` PASS).
- [x] **Shared AST Symbol Table & Prefix Pre-baking**: Pre-tokenize and warm-prefill repository symbol indexes and system prompts via `.slipstream/repo_tokens.idx`, maintaining permanent resident KV roots across all agent sessions.

### Phase 7: Single-Tenant Unified Memory & Fast-Disk Architecture (Roadmap)
- [x] **APFS Memory-Mapped KV Vault (`kv_vault.bin`)**: Persistent disk-backed KV cache that swaps cold 32-token pages to NVMe at ~5 GB/s, enabling instant session resumption with zero re-prefill latency (100k tokens in 2.57 ms, 1,217x faster than cold prefill) (`benchmark_kv_vault.py` PASS).
- [x] **CRAM-Inspired Direct-Read In-RAM Compressed KV Attention**: In-register direct dot-product engine (`runtime/engine/CompressedKv.hpp`, `dev/tests/engine/compressed_kv_test.cpp`) reading 3-bit PolarQuant and 4-bit nibbles directly in registers without temporary scratch buffers, reducing 262k context memory footprint from 32.25 GiB down to 13.00 GiB (2.48x density boost) without swap thrashing.
- [x] **Shared Compressed Repo Token Trie (`.slipstream/repo_tokens.idx`)**: A single binary `mmap` data structure shared between Python and C++ (`RepoTokenIndex.hpp`, `repo_indexer.py`) containing tokenized workspace files and n-gram hash tables, giving the engine instant speculative PLD drafts in 0.082 ms without runtime file tokenization (`repo-token-index` PASS).
- [x] **Zero-Overhead Local Socket / Token Bus IPC**: Single-tenant Unix domain socket transport between `SlipstreamAgent` and `FrontendServer`, bypassing TCP loopback, IP packet routing, and kernel network overhead.
- [x] **Permanent Root Prefix Pinning**: `KvCache::pinRootChain` freezes core system instructions, tool definitions, and workspace symbols into a warm root KV block at server boot, guaranteeing zero prefill overhead and 100% prefix cache hits for every future conversation (`kv-page-cache` PASS).
- [x] **Track 3: Metal GPU Kernel Integration for Direct-Read In-RAM Compressed Attention**: Implemented `verify_attention_q4_split` and `verify_attention_q3_split` in `runtime/metal/kernels/decode/attention_compressed.metal`, compiled into `build/slipstream.metallib`, verified exact parity (`cosine = 1.0000000`), achieving a 1.33x Metal GPU decode speedup and 2.48x density boost (`compressed-kv-metal` PASS).
- [x] **Track 4: Comprehensive Systems Architecture Blueprints & Publication Packaging**: Formulated 3 end-to-end ASCII architecture blueprints (Decoupled vs Unified Stack, In-Register SIMD Attention Pipeline, Co-Designed Turn Lifecycle) and publication-ready empirical analysis in `docs/research/slipstream_unified_codesign_research.md`.

### Phase 8: Advanced Literature & Systems Innovations (Post-v26.10 Research Roadmap)

#### Track 1: Extreme Attention & KV Sparsity (Sub-10 GiB 256k Context)
- [x] **Quest Query-Aware Page Sparsity (ICML 2024)**:
  - [x] Defined `QuestPageBounds` (64 bytes per page) and `SplashQuestEstimateParams` in `runtime/metal/abi/PagedAttention.h`.
  - [x] Implemented `quest_estimate_page_bounds` and `quest_update_page_bounds` Metal kernels in `runtime/metal/kernels/decode/quest_sparsity.metal` compiled into `build/slipstream.metallib`.
  - [x] Added `questSparsity` and `questTopKRatio` configuration to `VerifyAttentionConfig` in `runtime/ops/PagedAttention.hpp`.
  - [x] Verified bounding score estimation and recency priority window in `build/engine-tests/compressed-kv-metal` (PASS, scores: `[0.640625, 1.28125, 1.92188, 1e+09]`).
- [x] **DuoAttention Head Splitting (ICLR 2025)**:
  - [x] Profile and classify attention heads into Retrieval Heads (keep full KV chain) vs Streaming Heads (local 256-token circular buffer + 4 sinks).
  - [x] Add dual-page-table allocation and constant-memory circular ring buffers in `runtime/ops/PagedAttention.hpp` (`splash::ops::duo`).
  - [x] Validated zero retrieval degradation on needle-in-a-haystack test suite in `compressed_kv_metal_test.mm` (PASS: retrieval needle prob=0.9905, 910x memory reduction at 256k).
- [x] **MiniCache Depth Merging (NeurIPS 2024)**:
  - [x] Implement SLERP directional vector sharing for adjacent layers $L$ and $L+1$ in mid-to-deep layers (`runtime/engine/CompressedKv.hpp`, `runtime/ops/PagedAttention.hpp`).
  - [x] Verified 0.9917-0.9952 reconstruction fidelity and 2.67x memory reduction (9.06 GiB at 256k tokens; 4.53 GiB when combined with DuoAttention) across CPU and Metal GPU (`compressed-kv` & `compressed-kv-metal` PASS).

#### Track 2: Speculative Decoding & Heterogeneous Execution
- [x] **Hybrid DFlash + Prompt Lookup Speculative Drafting**:
  - [x] In `Runtime.mm`, probe `PromptLookup` first; when n-gram match $\ge 2$, set proposed tokens and skip DFlash 5-layer GPU pass.
  - [x] Fall back to DFlash when $pld < 2$; preserve draft proposals without clamping maximumRetained to 1.
- [x] **EAGLE-2 Context-Aware Dynamic Draft Trees (EMNLP 2024)**:
  - [x] Implemented entropy-driven dynamic tree switching in `runMtpDraft()` (`Qwen4ExpTarget.cpp`): deep linear chaining when top-1 confidence is high ($\ge 0.75$), broad multi-branch tree exploration when top-1/top-2 margin is small ($<0.15$).
  - [x] Verified tree traversal with 2D causal mask and tree ancestor masking in `sampling.metal` and `q8_attention_tile.h`.
  - [x] Verified 100% green compilation and pass rate across all 24 CPU engine tests (`make test-engine-cpu`).
- [x] **Apple Neural Engine (ANE) Heterogeneous Draft Offloading**:
  - [x] Implemented `AneDraftOffloader` in `runtime/ops/AneDraftOffload.hpp` enabling concurrent target verification on Metal GPU while ANE generates draft sequences.
  - [x] Verified heterogeneous asynchronous execution in `build/engine-tests/ane-draft` (PASS, 6.28 ms concurrent execution).
- [x] **Multi-Branch Lookahead Speculation for Code**:
  - [x] Extended `ops::PromptLookup` (`runtime/ops/PromptLookup.{hpp,cpp}`) with `proposeBranches()`, proposing multi-ngram candidate continuation paths ranked by match length and reverse chronological recency.
  - [x] Added unit tests in `dev/tests/engine/repo_token_index_test.cpp` verifying multi-branch discovery on branching symbol prefixes.
  - [x] Verified full suite `build/engine-tests/repo-token-index` PASS and `make test-engine-cpu` 100% green.

#### Track 3: MoE Working-Set Locality & Memory Hierarchy
- [x] **MoE-Infinity Temporal Sticky Affinity (ArXiv 2024)**:
  - [x] Add session-level token activation frequency tracking in `Qwen4ExpTarget.cpp` (`countExpertUse`).
  - [x] Implement configurable `SPLASH_EXPERT_HALF_LIFE` (1024) and `SPLASH_EXPERT_STICKY_FLOOR` (4) to protect foundational coding/reasoning experts from SSD eviction.
- [x] **Multi-Layer Router Horizon Lookahead ($L+2 \to L+3$)**:
  - [x] Integrated Darwin `fcntl(F_RDADVISE)` non-blocking kernel advisory in `prefetchPredicted()` for active predicted misses.
  - [x] Implemented `adviseHorizonPredicted()` in `Qwen4ExpTarget.cpp` issuing horizon advisories 2 layers ahead ($L+2$) during pipelined decode loop, pre-faulting NVMe pages into unified page cache via background DMA.
  - [x] Recompiled `build/slipstream` and verified 100% pass across all 24 CPU engine tests (`make test-engine-cpu`).
  - [x] **2026-10-10 REGRESSION FIX — advisory defaults off**: measured +17–24 ms/step cost at pinned 32–34 GiB caches (wasted readahead dominates SSD traffic); `horizonAdvisoryEnabled()` in `Qwen4Exp.hpp` defaults off, opt-in `SPLASH_HORIZON_ADVISORY=1`. Same-prompt 09-26 tasks back to 41.7 tok/s (parity with published). EAGLE-2 also defaulted off (+9% sampled). Guard stays on. Covered by `decodeFeatureSwitchDefaults()` in `qwen4exp_layout_test.cpp`.

#### Track 4: Single-Tenant Workstation Co-Design
- [x] **Mooncake-Style Predictive Background Promotion**:
  - [x] Added background worker thread `slipstream-mooncake-promoter` in server (`backend.py`) monitoring candidate session activity.
  - [x] Speculatively promotes candidate session tokens/pages from APFS fast vault into unified RAM before prompt completion.
  - [x] Exposed `/v1/session/promote`, `/v1/session/prewarm`, and `/v1/session/stats` HTTP endpoints in `server.py`.
  - [x] Benchmarked interactive turn latency: <0.1 ms in-RAM, <5 ms from APFS vault (verified in `dev/tests/engine/test_native_backend.py`).
- [x] **Speculative Tool Execution in Agent Loops**:
  - [x] Stream tool invocation parser in `server/agent/agent.py` detecting valid JSON arguments in real-time.
  - [x] Speculatively trigger read-only tool executions (`view_file`, `search`, `web_search`, `fetch_url`) concurrently in background thread pool.
  - [x] Harvest pre-computed results in 0.0 ms upon generation completion; verified with unit test in `test_agent.py`.
- [x] **Virtual Memory Aliasing (`vm_remap`) Zero-Copy**:
  - [x] Implemented `MachVmRemapBuffer` in `runtime/engine/MachVmRemap.hpp` using Darwin Mach kernel `mach_vm_remap()` for zero-copy page table sharing.
  - [x] Benchmarked in `dev/benchmarks/benchmark_mach_vm_remap.py` (64 MB shared in 7.71 µs, 1,242x faster than `memmove`).
  - [x] Verified via `build/engine-tests/mach-vm-remap` (PASS, 11 µs remapping).
- [x] **Persistent GDN Recurrent State Extents**:
  - [x] Implemented `PersistentGdnExtent` in `runtime/engine/PersistentGdnState.hpp` serializing recurrent GDN linear attention states and convolution buffers to APFS extents.
  - [x] Verified via `build/engine-tests/persistent-gdn` (PASS, 6.75 MB saved in 1.44 ms / 4.90 GB/s, loaded in 1.64 ms / 4.31 GB/s with bit-exact float fidelity).

### Phase 9: Speed Review Roadmap (R1–R6) Execution (Completed)
- [x] **R1 — Adaptive Expert-Cache Budgeting (`models/qwen4exp/Qwen4Exp.cpp`)**: Dynamically samples `hostAvailableBytes()`, auto-scaling expert cache budget up to 33.0 GiB when available RAM $>52$ GiB, eliminating on-demand NVMe misses while preserving working-set safety.
- [x] **R2 — Cold-Page Tiered Compressed KV Attention (`runtime/ops/PagedAttention.{hpp,cpp}`)**: Integrated `compressed_tier` namespace with `SPLASH_COMPRESSED_KV` and `SPLASH_KV_COMPACT` environment flags, mapping cold history pages beyond the 4,096 hot window to direct-read Q4/Q3 attention for 2.48x density and 1.33x GPU decode acceleration.
- [x] **R3(a) — MTP Draft-Head GPU Fusion (`models/qwen4exp/kernels/mtp_pick.metal`)**: Implemented `mtp_reduce_top16` GPU reduction kernel, fusing 64 slices $\times$ 16 candidates ($1,024$) down to top 16 directly on Metal GPU before host readback, cutting ~1.5–2.0 ms per decode cycle.
- [x] **R6 — Contiguous NVMe Advisory Batching & Host Spin Yield (`Qwen4ExpTarget.cpp`, `MetalBackend.mm`)**: Batched contiguous sequential expert matrix `radvisory` runs into consolidated `fcntl(F_RDADVISE)` system calls, cutting syscall overhead during prefill and decode. Verified with `__builtin_arm_yield()` pipeline spin loops.

### Phase 10: Engine & Transport Micro-Optimizations (Completed)
- [x] **IPC Transport Notification Coalescing & 4KB Drain (`FdTransport.cpp`)**: Added atomic `wakePending` flag to `CompletionWake` skipping redundant pipe writes; enlarged drain buffer to 4KB.
- [x] **Adaptive Spin-Then-Block Decode Handoff (`FdTransport.cpp`)**: Added configurable bounded spin loop (`SPLASH_SPIN_WINDOW_NS=2500`) with `yield` before `poll()`, catching GPU completion without thread sleep.
- [x] **Speculative Multi-Token Streaming Batching (`backend.py`)**: Batched multi-token speculative steps in `Streamer.put_tokens()` into a single `_send()` chunk when no stop sequence is present.
- [x] **Gated Telemetry Clock Traps (`Qwen4ExpTarget.cpp`)**: Gated ~200+ `AwakeClock::now()` calls behind `collectTiming` during decode steps, eliminating clock syscalls.
- [x] **Zero-Allocation Direct Wire Serialization for `TokensEvent` (`Protocol.cpp`)**: Serialized `TokensEvent` directly into the wire buffer, eliminating intermediate `Frame` heap allocation and vector copy.
- [x] **Vectorized Memory Primitives in `Protocol.cpp`**: Upgraded `Writer::u16`, `u32`, `u64`, `words` and `Reader::words` with unaligned `std::memcpy` on little endian.

### Phase 10: Agent Streaming Modes & Delta Suppression (Completed)
- [x] **Option A: Smart Hybrid Streaming (`stream_mode = "hybrid"`, Default)**: Routes tool-calling turns to non-streaming execution (`stream: false`), eliminating invisible JSON tool-call SSE deltas and Python socket parsing overhead; streams live text on direct conversational turns.
- [x] **Option B: Complete Suppression / Full Batch Mode (`stream_mode = "never"` / `--no-stream`)**: Suppresses streaming deltas across all turns, rendering output exclusively via formatted Markdown Panels upon turn completion.
- [x] **CLI Flags & Slash Commands**: Added `--stream [hybrid|always|never]` and `--no-stream` to `./slipstream code` (`install/launcher.py`); added interactive `/stream` and `/no-stream` slash commands with live footer status indicator in `server/agent/agent.py`.
- [x] **Double-Printing Fix**: Guarded assistant response panel rendering behind `not agent.last_turn_streamed`, eliminating duplicate response rendering in TUI scrollback.
- [x] **Telemetry & Reasoning Parity**: Captured `reasoning_content`, `predicted_per_second`, and `completion_tokens` during non-streaming turns for full telemetry parity.
- [x] **Unit Testing**: Added comprehensive test suites in `dev/tests/engine/test_launcher.py` (25/25 PASS) and `dev/tests/engine/test_agent.py` (27/27 PASS). Verified `test_server.py` (172/172 PASS) and `make test-engine-cpu` (24/24 PASS).

### Phase 11: Agent Live Context Tracking & UI Polish (Completed)
- [x] **Live Context Length in Bottom Panel (`build_footer()`)**: Added dynamic `📚 ctx: X,XXX / Yk (Z.Z%)` rendered in both idle and active generator states (`ACTION`, `THINKING`, `STREAMING`).
- [x] **Context Metrics in Inline Tool Execution Panels**: Added context token metrics to inline execution panels on every tool completion.
- [x] **Status Badging**: Standardized tool execution badge with `⚡ Action`.

### Phase 12: Agent Timeout Hardening & Session Vault (Completed)
- [x] **Timeout Resilience (`server/agent/agent.py`)**: Bumped default request/stream timeout from 120s to 600s (`DEFAULT_TIMEOUT`), added `SLIPSTREAM_AGENT_TIMEOUT` and CLI `--timeout` parameter. Explicitly caught socket `TimeoutError` and wrapped with clear recovery prompts.
- [x] **APFS Session Vault (`~/.cache/slipstream/agent_vaults/`)**: Automatic atomic checkpointing on turn completion, interruption, and errors to `<session_id>.json` + `latest.json`.
- [x] **In-Session Resume & Continuation**: Added `/continue` (or plain `continue` / `c`), `/retry`, `/resume [id]`, `/sessions`, `/save` slash commands.
- [x] **CLI Flags**: Added `-c`, `--continue`, `--resume [id]`, `--sessions`, `--timeout` to `./slipstream code`.
- [x] **Work Hub & Documentation**: Updated `INDEX.html` CLI Arguments table, in-session commands, and agent card. Verified with `bin/hub-sync.sh`.
- [x] **Unit Testing**: Added tests in `test_agent.py` and verified `test_launcher.py` (all 54 tests PASS).

### Phase 13: Unbounded Timeout & Tier-1 IPC Clarity (Completed)
- [x] **Unbounded Generation Wait Time (`server/agent/agent.py`)**: Defaulted `SlipstreamAgent.timeout = None`, executing pure blocking socket/HTTP I/O (`sock.settimeout(None)`). Added `--no-timeout` flag in `install/launcher.py`.
- [x] **Two-Tier Architecture Clarification**: Tier 1 (C++ Native Engine ↔ Server Daemon) uses binary IPC via `runtime/engine/FdTransport.cpp`, while Tier 2 (Agent CLI ↔ Server Daemon) uses Unix domain sockets (`/tmp/slipstream-8090.sock`) to keep the ~30–40 GB model resident in unified GPU RAM across CLI sessions.
- [x] **Unit Testing**: Added unit tests in `test_launcher.py` and `test_agent.py` (57/57 PASS).

### Phase 14: Qwen Optimization Roadmap Execution (Completed)
- [x] **P0 — Python Boundary & Fast-Path SSE (`server/server.py`, `server/metrics.py`)**:
  - [x] Cached selector in `_client_disconnected` on `FrontendHandler`, eliminating per-token `DefaultSelector()` allocations and kqueue descriptors.
  - [x] Fast-path text delta SSE string construction in `put_text()`, bypassing dictionary allocations.
  - [x] Instrumented `sse_decode_ms` vs `engine_decode_ms` (`decode_gap_ms`) in `/timings` and `/metrics`.
  - [x] Verified full server suite (`test_server.py`: 172/172 PASS).
- [x] **P1 — Agent-Loop Audit, Prefix Cache Hit-Rate & Tool Caps (`server/agent/agent.py`, `server/agent/tools.py`)**:
  - [x] Tracked cumulative `total_prompt_tokens` and `cached_prompt_tokens` in `SlipstreamAgent.stats` across streaming and non-streaming turns.
  - [x] Added `Prefix Cache Hit Rate` row to `/stats` table (`hit_rate% (cached / prompt tokens)`).
  - [x] Implemented incremental tool-parse cursor tracking (`tool_parse_cursors`) in streaming speculative execution to avoid redundant failing `json.loads` calls.
  - [x] Set tool output cap to 30,000 characters across `cap_tool_output` and `compress_tool_output` with 30-line tail preservation.
  - [x] Verified agent suite (`test_agent.py`: 59/59 PASS).
- [x] **P2 — Kernel Verification Pass**:
  - [x] Confirmed sampler readback in `Runtime.mm:1657-1691`: reads only `laneResult.retained` tokens (1 token steady state), never unaccepted rows or full logits.
  - [x] Confirmed Split-KV + GQA packing in `attention_q8.metal`: threadgroups partition along KV head and history split.
  - [x] Confirmed 4-pass 8-bit radix histogram selection in `qsa_select.metal` with 256 atomic bins and parallel chunk-wise tie scans.
- [x] **P3 — Metal PSO Pointer Caching & Makefile Audit (`MetalBackend.{hpp,mm}`, `Makefile`)**:
  - [x] Added `-DNDEBUG` to `ENGINE_CXXFLAGS` in `Makefile`.
  - [x] Added `pipelineMap` (`std::unordered_map`) and `cachedPipeline` direct pointer caching in `MetalBackend.{hpp,mm}` to eliminate `NSString` allocations and dictionary lookups on hot token decode steps.
  - [x] Verified C++ engine test targets (`make test-engine-cpu`: 24/24 PASS).

### Phase 15: Agent TUI Revamp, Plan Auto-Advancement & Loop Circuit Breaker (Completed)
- [x] **Robust Execution Plan Auto-Advancement (`server/agent/agent.py`, `server/agent/tools.py`)**:
  - [x] Hardened `manage_todo` argument extraction to support `task_id`, `step`, `id`, `task`, string integers, and hallucinated keys.
  - [x] Added automatic fallback to the next pending or in-progress step via `TodoManager.get_next_pending_or_in_progress()`, ensuring plans advance even if the model omits an ID.
  - [x] Printed updated execution plan directly into terminal scrollback on turn completion in both streaming and non-streaming modes.
- [x] **Consecutive-Only Tool Loop Circuit Breaker (`server/agent/agent.py`)**:
  - [x] Replaced global turn tool call count (`count >= 3`) with consecutive identical tool call detection (`self.recent_tool_signatures[-1] == self.recent_tool_signatures[-2] == self.recent_tool_signatures[-3]`).
  - [x] Preserved normal multi-turn inspections of different files/commands without premature termination.
- [x] **Prompt Enforced Reasoning (`server/agent/agent.py`)**:
  - [x] Updated agent `SYSTEM_PROMPT` to explicitly require rationale and analysis before tool calls.
- [x] **Agent Screen TUI Modernization ("Less Ugly, More Useful")**:
  - [x] Redesigned startup banner and bottom prompt box with clean high-contrast badge pills (`SLIPSTREAM`, `AGENT`, model, port, thinking level, stream mode).
  - [x] Added 6-segment visual context usage meter (`[■■□□□□]`) and real-time token headroom display.
  - [x] Styled execution plan checklist with high-contrast glyphs (`✔`, `▶ [IN PROGRESS]`, `○`, `✗`).
- [x] **Unit Testing & Verification**:
  - [x] Added unit tests in `dev/tests/engine/test_agent.py` for plan auto-advancing and consecutive circuit breaker (all 34 tests PASS).
  - [x] Verified 100% green across all 24 engine CPU tests (`make test-engine-cpu`) and server test suite.

## Architecture Improvement Ideas (from 2026-10-09 review)

Unprioritized ideas. Full rationale, evidence, and phasing in
`docs/architecture-roadmap.md`; performance-only items stay in
`docs/speed-optimization-review.md` (R1–R7). Harness-specific findings are in
`docs/agent-harness-review.md` (F1–F8, S1–S8). Web-search/retrieval provider
research and sequencing are in `docs/websearch-retrieval-sources.md` (R1–R6).

- [ ] **Single-source the wire protocol** — generate or drift-check `server/protocol.py` and `runtime/engine/Protocol.cpp` from one schema; removes the highest risk-per-line debt.
- [ ] **Replace spin waits with `os_sync_wait_on_address`** — `MetalBackend::waitPipelineEvent`/`waitPipelineStageDone` (20 ms `__builtin_arm_yield`) and the `while (slotReads...)` busy loop; stop stealing CPU from the SSD read threads.
- [ ] **Make expert-cache degradation loud and self-healing** — treat `mlock` failure as first-class: shrink capacity to what is actually resident, log once, expose pinned GiB / miss rate in `/status`.
- [ ] **Deeper async I/O pipeline** — dedicated prefetch pool with explicit queue depth (`preadv`/`dispatch_io`), L+2/L+3 reads on a timer; keep the measured 1-read-per-matrix rule.
- [ ] **Concurrency: bounded prefill interleave or width>1 with per-lane speculation** — reduce head-of-line blocking without regressing single-stream tok/s (design doc first; MTP/PagedAttention geometry constraints).
- [ ] **Declarative memory budget** — replace `physicalGiB − 34` and scattered reserves with fixed/elastic/reserved classes validated in `MemoryPlan`/`MemoryGovernor` so bootstrap cannot be broken by constants.
- [ ] **Split `Qwen4ExpTarget.cpp` by phase** (prefill / decode / draft) behind the `submit` seam and prune no-go/dead flags; make the decode state machine reviewable.
- [ ] **Measure approximations in isolation** — Q3/Q4 compressed KV, MiniCache SLERP, DuoAttention streaming heads vs. the uncompressed reference on the 145-item scorecard.
- [ ] **Reproducible benchmark harness** — one checksum-pinned command regenerating the headline comparisons, wired as a CI regression gate.
- [ ] **Generalize the runtime model interface** — replace the `QwenTarget` type-switch front class and abstract KV/state for non-Qwen-hybrid targets (prerequisite for any CUDA/ROCm port).
- [ ] **Unblock the research path** — register `fetch_url` in `TOOL_DEFINITIONS` (F2) and pin `duckduckgo_search`/`trafilatura`/`rapidfuzz` in `install/requirements.txt` (F6); see `docs/websearch-retrieval-sources.md`.
- [ ] **Provider-agnostic search adapter** — one `SLIPSTREAM_SEARCH_PROVIDER` switch behind `WebResearcher` (DDG default; Brave fast/cheap; Exa for technical; Parallel for deep research) plus a research loop over the SQLite cache and a thin MCP client.

## Qwen3.8 GGUF converter fixes (2026-10-11, commit 8616d43)
- [x] GDN value-head interleave in `models/qwen38/tools/convert_qwen38_gguf.py` (split GGUFs group 48 heads 3x16; engine reads interleaved; scoped to attn_qkv/attn_gate branch) — Q8_0 package 16/91 -> 75/91 vs HQ, serves clean ~50 tok/s
- [x] Q5_K (type 13) decode in `dev/tools/fast_dequant.c` + reader dispatch (bit-exact vs libggml-base) — unblocks Q4_K_M files; Swift Q4_K_M reconverted + probed clean
- [x] Regression tests: `dev/tests/test_qwen38_gdn_order.py`, `dev/tests/test_sharded_gguf_reader.py` (both fail pre-fix, pass post-fix)
- [ ] IQ4_XS/IQ4_NL/IQ3_S/Q3_K kernels still missing — UD-Q4_K_M (131 Q5_K + 117 IQ4_XS + others) not yet convertible
- [ ] `test_server.py` full + `test_agent.py` unsandboxed reruns before push
