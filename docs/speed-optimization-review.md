# Slipstream Speed Review — Findings, Recommendations, Execution Plan

**Date:** 2026-10-09 · **Scope:** decode/prefill hot path, MoE streaming, speculation,
IPC, host overhead · **Method:** full source audit of `runtime/`, `models/qwen4exp/`,
`server/`, plus `docs/profiling.md` baseline and `.agents/journal.md` history.
All claims carry file:line evidence. Note: internals use the `splash::` namespace and
`SPLASH_*` env names (inherited from the Splash fork); `build/slipstream.metallib`
is the current build artifact (`Makefile:52`).

## 1. Baseline (docs/profiling.md, qwen4exp, 10-prompt suite)

| Part | ms/step | share |
|---|---:|---:|
| GPU busy (verify, 47 layers) | ~41 | 55% |
| Waiting for SSD expert reads | ~15 | 20% |
| MTP draft head | ~7 | 9% |
| Host work between stages | ~12 | 16% |
| **Total** | **~75** (2.83 tok/step → 38–39 tok/s) | |

**Cost model:** each extra verify row ≈ 7.35 ms, mostly ~16 extra experts read from
SSD. Acceptance rate and miss count are therefore the two dominant levers.
Ceiling with current draft head + perfect stopping: ~45 tok/s.

## 2. Verified findings (code evidence)

### F1. Expert cache default is conservative vs. its own measurements
`models/qwen4exp/Qwen4Exp.cpp:269–278`: budget = `physicalGiB − 34` (30 GiB on 64 GB).
The comment itself records llama.cpp data: 16 GiB → 10–13 tok/s, 36 GiB → 14–19 tok/s,
swap past ~38 GiB. Journal (2026-10-08 15:55) raised 28→34 GiB reserve to fix boot
deficits; `.agents/next-session.md` lists "34 GiB (fastest) vs 30 GiB" as an open
decision. The budget is static at boot; the MemoryGovernor's live pressure is not
used. Every GiB of cache ≈ fewer on-demand misses out of ~50/step.

### F2. MTP draft head: 2 synchronous GPU round-trips + host top-k merge per draft token
`models/qwen4exp/Qwen4ExpTarget.cpp:2835, 2872`: each draft step submits graph A
(`submitCommand` = submit+wait), then host-stages the MTP layer's experts
(`hostSelect`/`stageActiveExperts`/`hostGroup`), then graph B (`submitCommand` =
submit+wait) ending in `mtp_pick_slices` (64 slices × 16 candidates). The host then
merges 1024 candidates down to 16 in a C++ insertion loop (`:2894–2925`) and computes
softmax mass on CPU. Up to 5 sequential steps ⇒ ~10 host↔GPU handoffs + ~10 host
merges per decode step, inside the ~7 ms draft budget.
Also per draft step: RoPE cos/sin recomputed on host with `std::cos/std::sin`
(`:2757–2762`), `rotaryFrequencies` recomputed with `std::pow` per call (`:2743`),
and a **shadow-acceptance `static std::map` mutated every step** (`:3262`) whose
output is only printed under `SPLASH_STEP_TIMING`.

### F3. Compressed KV (Q4/Q3) is built, tested, and not wired in
`runtime/engine/CompressedKv.hpp` and `runtime/metal/kernels/decode/
attention_compressed.metal` (`verify_attention_q4_split`, `verify_attention_q3_split`)
exist with passing CPU+Metal parity tests (`dev/tests/engine/compressed_kv*_test.*`,
journal 2026-10-08 12:05: cosine 1.0000000, 1.33× GPU decode speedup, 2.48× density:
262k context 32.25 GiB → 13.00 GiB). **No production file references them**
(grep over `runtime/ops`, `runtime/model`, `MetalBackend.mm`: zero hits).
At 130k context decode drops to 32–43 tok/s; attention is a growing share of the
~41 ms GPU time. This is the largest ready-made win.

### F4. Decode pipeline is well-pipelined; residual host costs are small allocations
`Qwen4ExpTarget.cpp:3340+`: all layers encoded up front, one `submitPipelineAsync`,
host↔GPU handoff via `MTLSharedEvent` with spin-then-block waits
(`MetalBackend.mm:1289–1307`, 20 ms spin — comment: blocking wake cost ~9 ms/token
at 47 handoffs). The spin has no `pause`/yield hint (burns a full core; minor power/
SSD-contention cost). Per layer per step, `hostSelect`/`stageActiveExperts` allocate
`std::vector<Miss>`, `std::bitset<512>`, `routesOf` vectors (47×/step). Prefill's
`encodeLayerExpertsByExpert` allocates `std::vector<std::vector<uint32_t>> routesOf(512)`
per layer per chunk (`:1436–1441`).

### F5. Speculation: tree drafting is implemented and plumbed, not evaluated
`SPLASH_TREE_DRAFT` (7-draft tree, structure A) exists in the draft loop
(`Qwen4ExpTarget.cpp:3033+`), verify attention (`tree_parents=0x60132100`,
`Runtime.mm:1447`), and acceptance (`encodeBatchAcceptance`, `Runtime.mm:1520–1545`).
Chain-confidence stop tuned to P_MIN=0.15 / CHAIN_MIN=0.20 / EARLY_EXIT=0.85
(journal 2026-10-08 17:03). Acceptance: 69.7% Flash-Next, 45.7% 27B.
`trace_report.py` strategy replay predicts real tok/s within ~1% (39.5 vs 39.1).
PLD+MTP hybrid exists (PLD bypasses the GPU draft when ≥2-token n-gram match).

### F6. SSD read mechanics are near-optimal; miss count is the lever
`readMissedExperts` (`Qwen4ExpTarget.cpp:426–459`): parallel `pread` via
`dispatch_apply`, 1 read/matrix measured fastest (4-piece split was slower:
30→39 ms/step). `F_RDADVISE` issued ahead of reads (`adviseMissedExperts`).
Lookahead predicts next-layer top-6 with consensus; sticky affinity
(`SPLASH_EXPERT_STICKY_FLOOR=4`, half-life 1024) landed 2026-10-08.
Reads: 37 ahead + 50 on-demand/step (docs). Remaining gain is in **prediction
quality and cache size**, not I/O mechanics.

### F7. IPC/server boundary is not the bottleneck
`FdTransport.cpp`: coalesced wake pipe, non-blocking drain, poll loop.
`server/backend.py`: batched multi-token SSE, cached selector, fast-path text
deltas (P0, journal 2026-10-08 21:35). `decode_gap_ms` telemetry exists in
`/metrics`. At 40 tok/s (25 ms/token) the Python path is ≪1 ms/token.

### F8. Build/measurement hygiene is good
`-O3 -DNDEBUG` (`Makefile:49`), `-O3` metallib (`:47`), PSO pointer cache
(`MetalBackend` `pipelineMap`). Trace tooling (`SPLASH_TRACE`, `SPLASH_STEP_TIMING`,
`trace_report.py`, `cache_plan.py`) reports exactly the numbers needed:
`miss_reads`, `host_stage`, `mtp_gpu_a/b/pick`, `predicted/misses`.

## 3. Updated recommendations (ranked)

**R1 — Adaptive expert-cache budget (target +5–15% decode, low risk).**
Set budget from *free* memory at boot (vm API + MemoryGovernor pressure), not a
fixed −34 GiB; keep `SPLASH_EXPERT_CACHE_GIB` override. A/B:
`identical/check.sh SPLASH_EXPERT_CACHE_GIB=30 vs =34` per `next-session.md`.

**R2 — Wire compressed KV for long context (target: +10–25% at 32k+, medium risk).**
Integrate `attention_compressed.metal` as a cold-page tier in
`PagedAttention::addVerify` (e.g., Q8 for the last N pages, Q4 beyond). Kernels and
parity tests exist; work is engine integration + page-format migration policy.
Gate behind a flag; validate with `compressed-kv-metal` test + `compare_logits.py`.

**R3 — Cut MTP draft host overhead (target +3–7%, low risk).**
(a) Full top-16 merge on GPU in `mtp_pick_slices` (one 16-id readback, delete host
1024→16 loop); (b) keep the single MTP layer's experts resident (it is one layer;
`SPLASH_RESIDENT_LAYERS` exists — measure residency for the MTP head only);
(c) hoist `rotaryFrequencies`/cos-sin out of `step()` into a precomputed table;
(d) gate the `static std::map` shadow stats behind `SPLASH_STEP_TIMING`.

**R4 — Evaluate tree drafting with trace replay (target: +5–10% if acceptance
improves, low cost).** `trace_report.py` already replays strategies on a recorded
run. Compare chain vs tree (structure A) acceptance×tokens/ms on the speed suite;
enable/tune only if replay predicts >2% (noise floor).

**R5 — Router-lookahead predictor v2 (target: −10–20 misses/step, medium).**
Current lookahead = next-layer top-6 from current router scores. Roadmap Track 3
(multi-layer lookahead) + offline-fitted linear predictor (Mixture-of-Depends
style) trained from `SPLASH_ROUTE_LOG` logs; ship coefficients in the package.
Rank plans offline with `cache_plan.py` first.

**R6 — Host micro-trims (target: +1–3%, low risk).**
Reuse scratch vectors in `hostSelect`/`stageActiveExperts`/`routesOf` (arena, no
per-step allocation); add a `pause`-style spin hint to
`waitPipelineEvent`/`waitPipelineStageDone`; batch `F_RDADVISE` into one `fcntl`
per contiguous run where offsets allow.

**R7 — Measurement gate for everything (free).**
Every change lands with: `make test-engine-cpu`, `test_server.py`,
`check_decode_consistency.py` (bit-exact), `identical/check.sh` 10/10, and
alternating A/B `speed_suite.py` runs (±2% noise rule from docs/profiling.md).

## 4. Execution plan

**Phase 0 — Baseline (½ day).** One traced run (`SPLASH_TRACE` + `trace_report.py`)
on the current build; record ms/step split, misses/step, acceptance, tok/s.
This numbers every phase below against today, not against 2026-09-22.

**Phase 1 — Safe host-side wins (1–2 days): R1, R3(c,d), R6.**
No output changes; bit-exactness guaranteed by construction.
Exit: identical outputs; speed_suite ≥ +2%; all test suites green.

**Phase 2 — Draft-path changes (2–3 days): R3(a,b).**
GPU top-16 merge (new kernel variant + host readback of 16 ids/probs), MTP-layer
residency. Verify: `check_decode_consistency.py`, acceptance-rate parity in
`/status`, speed_suite A/B.

**Phase 3 — Speculation policy (2–3 days): R4.**
Replay study first (no code), then enable tree behind default-on flag if replay
predicts >2%; tune P_MIN/CHAIN_MIN jointly with tree depth.

**Phase 4 — Compressed KV integration (1–2 weeks): R2.**
Design doc first (page tiering, migration, KV-cache eviction interaction with
`KvCache::pinRootChain`/CoW forks), then kernels→ops→engine wiring behind
`SPLASH_KV_COMPACT=1`, parity via `compare_logits.py`, speed at 4k/32k/130k.

**Phase 5 — Predictor v2 (1–2 weeks): R5.**
Offline: fit predictor on route logs, rank with `cache_plan.py`. Online: ship
coefficients, A/B misses/step + tok/s. Only after R1 (cache size changes the
miss baseline the predictor is trained against).

**Explicitly not doing now:** multi-lane batching (single-tenant product;
MTP disabled at width>1 by design), IPC work (F7: not the bottleneck),
I/O read-splitting (F6: measured slower), EAGLE-2 trained draft head
(new model artifact; out of scope for a code change).

## 5. Status

| Phase | Status |
|---|---|
| 0 Baseline trace | ☐ Not started |
| 1 Host-side wins | ☐ Not started |
| 2 Draft-path | ☐ Not started |
| 3 Speculation policy | ☐ Not started |
| 4 Compressed KV | ☐ Not started |
| 5 Predictor v2 | ☐ Not started |

Safety (non-negotiable, from `.agents/next-session.md`): one engine at a time via
`dev/benchmarks/guarded.py`; never two servers; restore GPU wired limit after reboot.
