# Magnitude Review — How It Works, How It Compares to Slipstream, What to Borrow

**Date:** 2026-10-09 · **Scope:** `magnitudedev/magnitude` (Apache-2.0) as it exists
today, compared against Slipstream's native engine (`runtime/`) and its colocated
agent harness (`server/agent/`). **Method:** read the Magnitude README, site, and
its checked-in design corpus (`design/inference/**`, `inference-v2/design/**`) —
roughly 3.5k lines of first-party design docs — against Slipstream's own docs and
source. No Slipstream source was modified.

> **See also:** [`magnitude-file-review.md`](magnitude-file-review.md) — the
> source-level pass over a sparse clone of the repo (5,625 files), with benchmark
> runs, a coverage ledger, and 26 prioritized ideas. Two conclusions here were
> revised after that pass: the batching risk (B1) and the declared-cache-point
> value (B3).

**Sources (retrieved 2026-10-09):** `raw.githubusercontent.com/magnitudedev/magnitude/main`
(`README.md`, `design/inference/seismic-compilation.md`,
`design/inference/engine/{scheduler,prefix-cache,speculative-generation,residency,memory}.md`,
`design/model-management/harness-connections.md`, `design/models/speculative-methods.md`,
`inference-v2/design/{performance,development/kernel-optimization}.md`,
`design/inference/session-bench.md`); `magnitude.dev`; `docs.magnitude.dev`.

## 0. Verdict

**Magnitude stopped being a browser-automation framework and became a direct
competitor to Slipstream.** The repo under `magnitudedev/magnitude` today is an
"inference engine for agents that optimizes itself for your exact hardware" — a
desktop app + CLI + native engine that runs GGUF models locally and plugs them
into *other* people's agents. The old vision-first browser agent lives on
separately (`magnitudedev/browser-agent`).

So the interesting comparison is engine-to-engine, not agent-to-agent — and on
that axis Magnitude is ahead on **multi-session serving and on-device kernel
tuning**, and behind on **running models that don't fit in RAM**. Slipstream's
moat (SSD-expert streaming of a 95 GiB model on a 64 GB Mac) is not something
Magnitude does at all; Magnitude's FAQ says "more memory lets you run larger
models."

The most valuable transfer is not any single kernel: it is Magnitude's
**engineering discipline** — a scheduler contract written as invariants and
acceptance criteria, a per-model speculation catalog, and a kernel-tuning loop
that measures on the actual device and keeps a replayable record. Several
concrete items below are pure wins for Slipstream; two are A/B-gated by the
"quality cannot drop" constraint.

## 1. What Magnitude is today

- One-click desktop app (macOS/Windows/Linux, Apple Silicon/NVIDIA/AMD/CPU).
- Advertised: **up to 2× faster than llama.cpp** — Metal M4 Pro 48 GB: prefill
  466→507 tok/s (+9%), decode 30→57 tok/s (+92%); CUDA DGX Spark: 2,033→2,507
  (+23%), 49→58 (+19%). Model: Qwen3.6 35B-A3B 4-bit, 64k context, **no
  speculative decoding**.
- Sells three properties Slipstream does not currently have: on-device kernel
  tuning, per-agent memory that is released when the agent stops, and concurrent
  sessions that share prefix caches.
- Connects the agent you already use (Pi, OpenCode, Hermes, OpenClaw, Codex,
  Claude Code, Oh My Pi, Cline) by rewriting each harness's native config; any
  other client uses the OpenAI-compatible API.

**Honest caveat on the benchmark.** The 92% Metal decode win is on a 35B-A3B
4-bit model whose weights fit a 48 GB machine, with speculation off, versus
llama.cpp with precompiled kernels. Slipstream's own 1.76–1.83× vs llama.cpp
(`.agents/tasks.md`) is a fundamentally different regime: a ~95 GiB model whose
experts stream from NVMe, with MTP/PLD drafting on. The two numbers are not
apples-to-apples, and Magnitude's advantage would shrink or vanish on
Slipstream's workload.

## 2. Architecture, part by part

### 2.1 Engine generations and language

Rust/TypeScript monorepo, three engine generations in-tree (`inference/`,
`inference-v2/`, plus `old-inference/`), a Bun/TS CLI, Electron desktop app, and a
`packages/` tree (providers, roles, storage, tracing, VCS, skills, shell
classifier). The current engine is Rust (`inference/engine/**`), with Metal
kernels and a Python "Seismic" kernel-generation project. This is a *much* larger
surface than Slipstream, and the fact that it ships four parallel engine
generations is a warning, not a model to copy.

### 2.2 Scheduler — continuous batching with time-shared prefill/decode

`design/inference/engine/scheduler.md` specifies the whole serving contract:

- One execution thread owns everything; there is no periodic re-drive. Requests
  are admitted *logically and immediately* and become resident between flights.
- Decode rounds and prompt chunks alternate using **measured execution time**:
  "while both phases are ready, prefill consumes a time budget that decode must
  replenish." For decode share `s`, a chunk of length `p` incurs `p·s/(1−s)` of
  decode debt; overshoot carries forward, capped at one decode round.
- Per-request state machines (non-resident, ready, awaiting credit, parked,
  memory wait, awaiting a peer's prefix), plus an **availability epoch** that
  advances only when capacity is actually freed.
- Speculation is not a special mode: "requests own speculative progress;
  execution batches own shared computation." Each row is accepted independently
  — no batch-wide acceptance length, no fixed cohort, no slowest-drafter barrier.
- Correctness teeth that matter for quality: **position-addressed RNG**
  ("prevents batch membership or rejected work from changing a seeded request's
  random draws"), rejected tokens never enter committed history, grammar state or
  retained prefixes, and target-sampled prefix matching.

This is the single biggest gap versus Slipstream. Slipstream is deliberately
single-tenant: `total_slots: 1` (`server/server.py:347`), `SPLASH_MAX_BATCH_WIDTH=1`,
admission concurrency 1, and "multi-lane batching as the default" is listed as
explicitly *not* being done (`docs/architecture-roadmap.md`, `docs/speed-optimization-review.md`).
Roadmap item **W5 (concurrency: batching or prefill interleave)** is exactly the
territory Magnitude has specified in full. Slipstream doesn't need Magnitude's
full machinery, but W5 should borrow the *time-share debt* policy and the
"independent per-row acceptance" rule.

### 2.3 Prefix cache — "a request resumes from the deepest cached state on its path"

`design/inference/engine/prefix-cache.md` is the most directly transferable
document. Highlights:

- A **path** is a token sequence *plus the media identities conditioned on it*,
  ending at an exact layout boundary; a **hit** exists only when the entry's path
  equals the query up to a position at an exact boundary.
- **Declared cache points.** Chat preparation anticipates "a request that changes
  only its last message." It re-renders the request with that content replaced by
  a probe, finds the first differing byte, maps the last added-token boundary
  (turn markup) to an input row, and declares it at admission. The scheduler
  plans it as a branch point, prefill stops there and retains state — so every
  later request whose last message differs resumes there **whatever its content
  begins with**. A declared point is only planned when it hits the same
  minimum-branch-gain gate as any branch point.
- Resume state includes encoded features of media spans that **straddle** the
  resume position, with placements of one image sharing its features.
- Eviction releases numerical state and features but keeps the request's input,
  so replay is conditioned identically to the first pass.

Slipstream has the ingredients — radix-block cache (`runtime/engine/Cache.cpp`),
`KvCache::pinRootChain` permanent root pinning, pre-baked repo-token index — but
the *declared-cache-point-at-the-last-message* trick is new and cheap, and it
generalizes the Zone 1 pinning story from "system prompt is identical" to "the
conversation prefix is identical even when the tool result changes."

### 2.4 Speculation — a per-model catalog and separate published drafters

`design/models/speculative-methods.md` is a maintained table of
model → best-known method → current method → whether the draft is *embedded* or a
*separate file*. Two facts stand out:

- The drafters are **DFlash / DFlash2 / DSpark**, published as standalone GGUFs on
  Hugging Face (`magnitudedev/Qwen3.8-27B-DFlash2-GGUF`, `Qwen3.6-35B-A3B-DFlash-GGUF`, …).
  Slipstream's own DFlash draft (`runtime/model/DFlashDraft.hpp`) is the same
  family — 5 layers, hidden 5120, 8 KV heads, target hidden 25600 — but it is
  baked into the model package, and `.agents/status.md` still records
  "DFlash neural drafter restoration as the key 4× accelerator to hit 55–65 tok/s"
  on Swift-27B.
- Policy encoded in the catalog: **MTP is disabled for every vision-capable
  model**; if a compatible DFlash/DFlash2/DSpark exists Magnitude activates it,
  otherwise the model runs with no speculation.

`design/inference/engine/residency.md` adds the mechanical piece: a **progressive
head** replaces the stored Q8_0 output head with five planes plus per-row radii
(`[V, 2]`, so leading rows are a draft head's draft-vocabulary rows) whose bytes
are unchanged; "a draft head binds the target's resident planes, imported once."
That is a concrete, low-cost way to hang a draft head off the target's own
embedding/output weights instead of duplicating them — relevant to Slipstream's
64 GB budget and its existing 64K draft vocabulary.

### 2.5 Memory — residency keyed by representation and layout

- `ResidencyStore` is the sole importer/cache; its key is (artifact identity,
  tensor name, **resident representation + layout**), so tied weights share
  physical storage.
- Layout is chosen per backend for that backend's kernels: Metal `rows32`, Vulkan
  `rows16`, CUDA `mma16`, CPU `rows8`. Every import is **exact** — "each resident
  value equals the source format's reference dequantization bit for bit" — with a
  registered conversion into a wider representation when no exact one exists,
  otherwise the model is `Unsupported` at plan time.
- Each import validates a `WeightPlan` before any device work; on Metal,
  consecutive whole tensors share one page-rounded mapped window and are wired
  like every engine holding; failed imports leave no cache entry.
- Charging is per physical allocation ("charge each complete physical image until
  its last row or execution pin ends") and is the basis for the advertised
  "27% less memory per agent, freed when agents stop."

Slipstream has `MemoryPlan`/`MemoryGovernor`/`MemoryAudit` and its own repacked
layouts, plus an expert cache with an adaptive budget. The borrowable part is
formalizing the (artifact, tensor, representation, layout) residency key and the
"exact or refuse at plan time" rule — Slipstream's converters already verify
exactness (`dev/tools/convert_qwen38.py`, "whole composed layer — exact,
0.000e+00"), so this would mostly be moving an existing discipline to the load
path.

### 2.6 Seismic — compiling and tuning kernels on the device

`design/inference/seismic-compilation.md` (~1,018 lines) describes a compiler
whose search is *time-bounded and per-device*:

- Native top-level kernels take a separate route: checked entry contract +
  embedded native source + authored launch, with "the only search is tuning, fast
  enough to run at program preparation." Tuning does a time-bounded local search
  over the author's declared parameter domain, minimizing one weighted
  device-measured cost (`Σ weight × median`), cheapest-form-first, restarting
  from the farthest unvisited point.
- Every candidate must pass **numerical validation at the timed points** before
  repeated timing; the default is validated like every candidate; a candidate
  that fails at untimed points gives way to the defaults, which must also pass
  there.
- The search is **deterministic and replayable** from a recorded evaluator.
- Measurement hygiene worth stealing outright: a kernel's first pass is never a
  sample; a device idle >20 ms runs at a lowered clock and can read up to 4× too
  slow, so it is kept busy until speed stabilizes; configurations the device
  cannot form cost +∞; the K cheapest within 10% of the leader are re-measured
  with more samples before ranking.

This is the mechanism behind "kernels are tuned on your hardware before a model
runs." Slipstream's Metal kernels are hand-written with a growing PSO pointer
cache (`pipelineMap`/`cachedPipeline`), so the win here is narrower — tile /
threadgroup / split-K parameters for the GEMM and paged-attention paths — but the
validation-and-replay discipline maps cleanly onto Slipstream's "identical greedy
output" gate.

### 2.7 Session-level benchmarking

`design/inference/session-bench.md` measures *serving* with simulated agent
sessions, not single prompts: pinned hash-verified BFCL tool interactions, a
prose-copy workload, RULER-derived retrieval recipes with independent sizing, and
explicit session / concurrency / fork / memory-pressure sections. It insists that
"no retention claim follows from session shape alone" and that history sharing be
distinguished from actual retained-prefix reuse. Fixed completion allowances
(32,768 for tools, 256 prose, 1,024 retrieval); context targets are approximate
and terminal engine counts are authoritative.

Slipstream's equivalent is a 10-prompt suite, a 145-item quality scorecard, and
per-mechanism microbenchmarks under `dev/benchmarks/`. A session-shaped harness is
the missing rung between "microbenchmark" and "Nitin judges blind."

### 2.8 Harness connections

`design/model-management/harness-connections.md` is a transactional
configuration manager for other agents' native configs:

- One connector per harness owns that harness's provider projection, native
  active-model representation + restoration, reasoning-control projection,
  protocol/endpoint, skill target, and read-only inspection. Protocols:
  Chat Completions (Pi, OMP, OpenCode, OpenClaw, Hermes, Cline), Responses
  (Codex), Anthropic Messages (Claude Code).
- **Connected is an observation, not a receipt** — required fields, skill
  contents and plugin activation must still match. "Unreadable or malformed
  configuration is Unavailable and is never overwritten to manufacture a
  successful connection."
- One cross-process lock; compensations registered before mutations, run in
  reverse, all attempted; "manifest commit is the uninterruptible durability
  point"; apply is an idempotent replacement of Magnitude-owned state; disconnect
  restores the prior model only while the current selection is still
  Magnitude-owned.
- **Reasoning-effort correctness** is model-relative: each model owns an exact
  normalized effort domain and default, each connector projects it as faithfully
  as its harness allows, and every locally admitted request applies
  `RoundUpOrClamp` after model resolution — "correctness cannot depend on
  selecting a harness-specific admission mode."

Slipstream is already protocol-complete here: `server/api_shapes.py` +
`server/server.py` serve `/v1/chat/completions`, `/v1/responses`, `/v1/messages`
and `/v1/messages/count_tokens`, and `install/clients.py` configures OpenCode and
Codex. What's missing is the *connection discipline* (observation, compensation,
restoration) and the *admission clamp* invariant.

## 3. Side-by-side

| Axis | Magnitude | Slipstream |
|---|---|---|
| Hardware | Metal, CUDA, AMD, CPU | Apple Silicon Metal only |
| Model scale | Fits in RAM | **Streams MoE experts from NVMe** (95 GiB on 64 GB) |
| Concurrency | Continuous batching, time-shared prefill/decode, multi-session | Single tenant, `total_slots: 1`, batching off by design (roadmap W5 open) |
| Prefix cache | Path + media, declared cache points, media straddle features | Radix 32-token blocks, `pinRootChain`, pre-baked repo index, KV vault |
| Speculation | MTP / DFlash / DFlash2 / DSpark, per-model catalog, separate published drafts | MTP + DFlash + PLD + hybrid + EAGLE trees + ANE offload; draft baked into package |
| KV footprint | Q4/Q5/Q8-style GGUF representations, exact import | **Compressed KV** (3-bit PolarQuant, MiniCache, DuoAttention), APFS KV vault |
| Kernel tuning | **On-device bounded autotune, numerical validation, replayable** | Hand-written kernels + PSO cache |
| Agent | Connects external harnesses (Pi/Codex/Claude Code/Cline/…) | Own harness (`server/agent/`), protocol-complete, in-process over Unix socket |
| Quality gate | Numerical validation per candidate; position-addressed RNG | Identical greedy output vs reference; exact rejection sampling; 145-item scorecard |
| Benchmarking | Session-shaped (BFCL/prose/RULER/fork/memory) | 10-prompt suite, 145-item scorecard, per-mechanism microbenchmarks |

## 4. Where Slipstream is already ahead

Worth stating plainly so the borrow list doesn't read as a deficit:

1. **SSD expert streaming.** Magnitude's entire FAQ assumes weights fit. Slipstream
   serves a ~95 GiB model on a 64 GB Mac at 40–52 tok/s with `F_RDADVISE`
   layer-ahead prefetch and an adaptive expert cache.
2. **KV compression research.** PolarQuant 3-bit direct-read attention, MiniCache
   depth merging, DuoAttention head splitting, 262k-context footprint of 13 GiB
   (`runtime/engine/CompressedKv.hpp`, `.agents/tasks.md`).
3. **Drafting breadth.** MTP + PLD (49.1 ns, zero-alloc, upstream PR #300) +
   hybrid routing + EAGLE-2 trees + ANE/GPU heterogeneous drafting. Magnitude
   ships *one* method per model.
4. **In-process harness.** Delta-prefill `AgentSession`, Unix-socket token bus
   (0.14 ms roundtrip), speculative read-tool dispatch, grammar-pruned
   speculative verification.
5. **Zero external dependencies at runtime** in the engine — Magnitude's engine
   ships four generations of its own plus a Python kernel compiler.

## 5. Borrow list

Each item names the Magnitude source, the Slipstream target, and the regression
risk against the non-negotiable "no quality or speed regression" constraint.
"Pure win" = cannot change model output.

| # | Borrow | Target | Risk |
|---|---|---|---|
| B1a | Prefill interleave only: time-share debt policy alternating bounded prompt chunks with decode rounds, single-lane path preserved | `runtime/engine/Scheduler.cpp` (roadmap W5 option b) | **Low** — changes scheduling order, not numerics or batch geometry (revised down from A/B after the source-level pass) |
| B1b | Width>1 batching with per-request speculative progress and independent per-row acceptance | `runtime/engine/Scheduler.cpp` (roadmap W5 option a) | **A/B required** (batching changes numerics → re-run identical-greedy + 145-item scorecard) |
| B2 | Position-addressed RNG so batch membership or rejected work can't change a seeded request's draws | `runtime/ops/Sampling.*` | **A/B required**, but this is the enabling correctness rule for B1 |
| B3 | Declared cache point below a divergence (probe re-render → first differing byte → added-token boundary → declared at admission) | `server/frontend.py` + `runtime/engine/Cache.cpp` | **Speed only** — but *narrowed*: a live probe (1,819-token prompt) showed Slipstream already resumes a same-prefix/different-last-message request at the cold-repeat latency (267 ms vs 4,450 ms cold), so value remains only for mid-prompt divergences, recurrent GDN state, and spliced/compacted ranges |
| B4 | Resume state must carry encoded features for media spans that straddle the resume position | `runtime/engine/KvCache.{hpp,cpp}` (vision is live) | Speed only, but requires a vision-correctness test |
| B5 | Per-model speculation catalog + separate published draft artifacts + "no MTP on vision models" policy | `install/launcher.py`, model manifests, `.agents/tasks.md` | Speed only; quality is protected by target verification |
| B6 | Progressive head planes so a draft head binds the target's resident output planes instead of duplicating them | `runtime/model/WeightStore.*`, `MemoryPlan` | **Pure win** gains; needs bit-exactness check on the planes |
| B7 | Residency key = (artifact, tensor, representation, layout), with exact-or-`Unsupported` at plan time and tied weights sharing storage | `runtime/model/WeightStore.*`, `runtime/engine/MemoryPlan.cpp` | **Pure win** (load-time) |
| B8 | Per-session memory attribution and release when an agent stops | `runtime/engine/Memory{Plan,Governor,Audit}` | **Pure win** if accounting only; eviction behavior needs a soak test |
| B9 | Bounded, deterministic, replayable on-device autotuner: validate numerically before timing, default validated like every candidate, skip first pass, warm a cold device, re-measure K cheapest within 10% | `runtime/metal/`, PSO cache, `Makefile` | **Speed only**, gated by existing identical-output check |
| B10 | Session-shaped benchmark: pinned tool-call fixtures, prose-copy, retrieval with distractors, explicit fork/concurrency/memory sections | `dev/benchmarks/` | **Pure win** (measurement) |
| B11 | Connection manager: observation (Connected ≠ receipt), transactional apply with reverse-order compensation, idempotent replace, conditional restoration | `install/clients.py` | **Pure win** |
| B12 | Model-relative reasoning-effort domain + global `RoundUpOrClamp` at admission, with per-harness projection completeness | `server/thinking.py`, `server/constraints.py` | Clamp is a **pure win**; changing the *default* effort is **A/B required** |
| B13 | Adopt "optimization is an explanation of unnecessary cost" review discipline: state cost removed / cost introduced / why correctness holds / which conditions benefit; verify the mechanism, not just the timing | `docs/` review template | **Pure win** |

### Priority

- **Do first (pure win, small):** B10 (session bench answers the open
  "sessions" benchmark gap directly), B11, B13, B6, B7, B12-clamp.
- **Do next (speed only, gated by existing gates):** B1a (prefill interleave), B5,
  B9, B3 (narrowed as above).
- **Plan carefully (A/B, real design):** B1b + B2. These are the only items that can
  change output, and B2 must land *with* B1b, never after.
- **Study before committing:** B4, B8 — both touch the vision path and the memory
  accounting that currently keeps Slipstream inside a 64 GB budget.

## 6. What not to copy

- **Four engine generations in one repo.** Slipstream's single C++20/Obj-C++
  engine with an architecture checker is a strength.
- **Pure-RAM model residency.** Slipstream's SSD expert streaming is the moat;
  Magnitude's "more memory lets you run larger ones" is the thing Slipstream
  already solved.
- **Full-machinery concurrency.** Magnitude's scheduler doc is 246 terse lines
  describing a system that coordinates availability epochs, memory waits, peer
  prefix waits and parked rounds. Slipstream is single-tenant by product choice;
  take the time-share policy and the per-row independence rule, not the state
  machine.

## 7. Bottom line

Magnitude's headline numbers are real but earned in a regime Slipstream is not
in (small model, all-resident, no speculation, tuned kernels). Its value to
Slipstream is as a **specification of two things Slipstream has deliberately
deferred** — concurrent sessions and on-device kernel tuning — written with
invariants, acceptance criteria and a quality gate that matches Slipstream's own
"output cannot change" discipline. The cheapest high-value transfers are the
session-shaped benchmark, the connection manager, and the declared-cache-point
trick; the largest eventual payoff is the batching scheduler, and it is the one
that needs the most care because it is the one that can change what the model
says.
