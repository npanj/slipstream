# Tasks — qwen4exp port

## Done

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
- [x] Prompt Lookup Decoding engine integrated into Swift-27B (`splash2/runtime/model/Runtime.mm`, bypasses DFlash on n-gram match >= 2 tokens; installed to Splash-Q8 and pushed to `fork/q8`)
- [x] Updated Work Hub (`~/Documents/shared-with-google-drive/INDEX.html`) and `~/.omp/agent/models.yml` to minimum medium thinking (`--thinking=medium`) for Swift-27B and Swift-V3
- [x] Grammar-Pruned Speculative Verification in constrained decoding (`Runtime.mm`, 8.5x tool-calling speedup: 5.6 -> 49.6 tok/s)
- [x] Gated Smart Prompt Lookup Decoding with match length gating & unambiguity validation (`SPLASH_PLD_MIN_MATCH`, `SPLASH_PLD_UNAMBIGUOUS`)
- [x] Proactive Grammar-Masked Drafting (`SPLASH_MTP_MASKED_DRAFT`, filters MTP candidates with grammar mask)
- [x] Adaptive Mode Detection (`SPLASH_ADAPTIVE_MODE`, separates `<thought>` from structured JSON tool calling)
- [ ] Uneven cache slots per layer (~9% fewer misses in replay)
- [ ] Draft head: process only live rows (~0.5 ms)
- [x] Quality round: Swift-Qwen3.8-27B-Splash-HQ vs Swift-Qwen3.8-Flash-Next-V3 (145 items across 6 domains: AIME 2025, MATH-500, GPQA Diamond, GSM8K, HumanEval, Hard Systems & Logic; Flash-Next leads 70.3% vs 67.6% with identical ~44 tok/s throughput)
- [ ] Code-prompt drift: 79% same pick vs llama.cpp 89%
- [ ] Short-prompt reading speed (87 vs ~110 tok/s)
- [x] 27B 4-bit package prepared: Swift-Qwen3.8-27B converted to Splash format (local/Swift-Qwen3.8-27B, 16.9 GB)

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
