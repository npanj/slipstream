# Slipstream roadmap: what to do, in what order, and why

**Status:** proposal, 2026-10-22. Companion to [`docs/kapitsa-review.md`](kapitsa-review.md) (the evidence: what
Slipstream can borrow from Kapitsa and what it can't).

**How to read it.** Every item below has an **impact class**, an **effort** in focused dev-days for one experienced
engineer (agent-assisted), a **risk**, a **dependency** and an **acceptance test** — a number that has to move, measured
by a tool that already exists or that phase 0 adds. Estimates are labelled: `measured` (a number is in this repo or in
Kapitsa's docs), `derived` (arithmetic on measured numbers), `guess` (judgement, to be replaced by measurement in
phase 0).

**Impact classes.** **A** = changes a headline number by more than 25%, or removes a whole class of failure.
**B** = 5–25%. **C** = under 5%, or it buys maintainability rather than speed.

---

## 1. The one number this project should optimise

Slipstream's README leads with **decode tok/s** (41–52). That number is real and it is the hardest thing to improve
from here: Slipstream already beats llama.cpp on it by 1.76×, and the remaining headroom is SSD bandwidth, not
scheduling.

But decode is not what a user feels. What a user feels is **the wall clock of a 20-turn coding session at 16–32K
context**, and in that number:

| Term | Today | Share of a 20-turn session (derived) |
|---|---|---|
| Prefill on a cold turn (`TTFT 13.6 s` avg at 16–32K, `README` telemetry) | 13.6 s | dominant whenever the prefix is not cached |
| Prefill on a cached turn | `guess` 0.2–1 s | small, **if** the prefix survives; unknown today |
| Decode (say 500 new tokens at 35 tok/s) | 14.3 s | large, and hard to move |
| Server/template/tool-mask overhead | not measured | `guess` 100–300 ms a turn |

So the ordering below is **latency-first, then quality assurance, then decode**. That is also the order in which
Kapitsa's own work pays off: their headline is decode, but their *engineering* investment (SSD prefix cache, slot reuse,
block drafters) is mostly about not re-reading the conversation.

**North-star metrics** (all but the last are obtainable today; the last needs phase 0):

1. **Agent-session wall clock** — replay one 20-turn session at 16–32K: total time, TTFT p50/p90, decode rate.
   Kapitsa's harness already has this phase (`bench.py --agentic`, `agent_replay.py`), driven over HTTP against
   Slipstream's port. This should be the number in the README.
2. **Follow-up TTFT after a cache hit, and the hit rate** — in-memory today, across restarts/evictions after item L1.
3. **Cold prefill tokens/s at 4K and 16K** — `/metrics` already exposes `slipstream_prefill_tokens_per_second`;
   the open backlog calls this "short-prompt reading speed (87 vs ~110 tok/s)".
4. **Decode tok/s** (the current headline) and **tokens per verify step**.
5. **Quality**: top-1 agreement and KL against the bf16 reference (`models/qwen4exp/bench/compare_logits.py`), and a
   self-noise floor (this phase adds it).

---

## 2. What is already good (do not re-litigate)

Being explicit here is what keeps the roadmap honest; several "obvious" improvements are already done and measured:

| Already in place | Evidence |
|---|---|
| Uneven expert-cache slots per layer | `Qwen4Exp.cpp:313-336` `kSlotProfile`, fitted by `cache_plan.py`: SSD reads 95 → 87 a step, 39.7 → 41.5 tok/s at the same memory. (Still listed as open in `.agents/tasks.md` — that is doc drift, see O4.) |
| Read-ahead + waves, pipelined staging | `docs/profiling.md`; tasks.md "Read-ahead 6 + uneven cache slots: +3–4%" |
| Prompt Lookup Decoding (PLD), PLD-first in tool mode | `runtime/ops/PromptLookup.*` (49 ns/query), `Qwen4ExpTarget.cpp:3183-3196`; 8.5× tool-calling speedup recorded |
| Grammar-pruned speculative verification | llguidance masks over the pipe, `Protocol.cpp` `simulateTokens × wordsPerMask` |
| Metal 4 `matmul2d` wide prefill tiles | `q4_mpp_tiles.h`, `gguf_staged_tile.h`, `prefill/moe.metal` |
| Tree-mask plumbing in kernels | ancestor masks, depth RoPE, GDN parent-chain conv/scan, tree sampling (`kapitsa-review.md` §3.3) |
| Request metrics | `/metrics`: TTFT/ITL p50/p95, prefill/decode rates, draft acceptance, admission, memory governor; `cached_tokens` per response |
| Test surface | 87 engine test files; CPU suite + sanitizers + Python matrix in CI on macOS runners |

Corollary: **do not spend time on** kernel micro-optimisation, a new drafter, or a native grammar engine before the
items below. And note what is *missing* from the gates rather than from the code:

- **Metal tests are not in CI** (`.github/workflows/ci.yml` runs `check-native-cpu` and sanitizers; `test-engine-metal`
  runs only locally, and the real-model matrix is a manual `workflow_dispatch` on self-hosted runners). Any kernel change
  is therefore ungated until someone runs it on the Mac.
- **No numerical regression gate.** `compare_logits.py` compares to a bf16 reference, not to *the previous build* on the
  same Mac, so "did this change the answers?" is answered by reading outputs.

---

## 3. Workstreams

### W0 — Measurement and gates (do this first; everything else depends on it)

| ID | Item | Impact | Effort | Risk | Acceptance |
|---|---|---|---|---|---|
| M0 | **Baseline the north-star set** on the target Mac with Kapitsa's harness + the config in `dev/benchmarks/kapitsa-harness/`: agent-session replay, cached TTFT, cold prefill 4K/16K, decode (T=0.6 and greedy), flags and provenance recorded | — | 2–3 d | low | One results JSON per configuration, regenerable by one command; `compare_table.py` table committed |
| M1 | **Logit dump + self-noise floor**: a `logit_dump` mode (paths: `decode`, `verify`, `batch(4)`, `prefill`) and `logit_floor.py` writing a per-Mac JSON, as Kapitsa does (`CONTRIBUTING.md#output-check`) | A (quality) | 3 d | low | Floor JSON for this Mac; a deliberate one-kernel perturbation is caught by `logit_compare` and a benign refactor is not |
| M2 | **Engine-internal counters as metrics**: SSD reads/step, expert hit/late/miss/prefetch-used, GPU stall ms waiting for reads, row cost ms, per-depth acceptance | B (makes W3/W4 steerable) | 2 d | low | `curl /metrics` shows them; they agree with `SPLASH_ROUTE_LOG` replay within a few % |
| M3 | **Metal gate**: run `make test-engine-metal` plus a 50-token generation smoke on the self-hosted M5 Pro on every PR (or a nightly), with the logit floor as the check | A (risk) | 2–3 d | med (runner availability) | A kernel PR that changes outputs shows red without a human reading text |

**Why first:** every later item claims a percentage. Without M0 there is no denominator; without M1 any "output
unchanged" claim is an anecdote; without M3 a kernel change can land broken.

### W1 — Turn-to-turn latency: make the conversation survive

| ID | Item | Impact | Effort | Risk | Acceptance |
|---|---|---|---|---|---|
| L1 | **SSD prefix cache**: KV blocks + composite states (incl. draft ring) as content-keyed, CRC-checked files, with namespace digests, quota, `flock`, temp-file+rename, `F_NOCACHE` (port of Kapitsa `prefix_cache.*` / `engine_disk.cpp`, see review §3.1) | **A** | 8–10 d + 2 d tests | med | On the harness's cache phases: follow-up TTFT on a 16K conversation drops ≥5× *when it currently misses*; a restart keeps the prefix; a corrupted file is detected and ignored, never loaded |
| L2 | **Measure and fix the in-memory hit rate first**: instrument `CacheLookupSnapshot` (already exists: `lookups`, `kvHitTokens`, `stateHitTokens`, `lazyJunctions`) into `/metrics`, then find why 16–32K turns still average 13.6 s TTFT | A | 2 d | low | A report: for 20 real turns, how many tokens were prefilled vs reused, and where the miss came from |
| L3 | **Cache-aware admission**: prefer resuming a request whose prefix is cached over first-come-first-served when the GPU is busy; cap the damage of eviction by pinning the active conversation | B | 3 d | med | Under two parallel agent loops, both keep their prefix (no thrash); p90 TTFT improves |
| L4 | **Derive the prefix from the KV page keys we already compute**, not from a second hash — one source of truth for "same tokens, same state" | C (correctness) | 1–2 d | low | Property test: any two prompts sharing a page chain hit the cache; changing one token in the middle misses from that page on |
| L5 | **Long context**: measure prefill rate and decode retention from 8K to 130K with a *cold* prefix (the telemetry in the README mixes cached and cold turns), and fix what degrades (page-table growth, admission, split-K attention, KV pool extents) | B | 3–5 d | med | A table of prefill tok/s and decode tok/s vs context, cold, on one Mac; no cliff before 130K |

**Privacy constraint on L1** (decide before writing the code, not after): a prefix cache on disk stores the
conversation's tokens and its KV state in plain files — Kapitsa's entries carry the tokens themselves. For a local
engine that is defensible, but it must be (a) off or opt-in by default, (b) in one documented location, (c) removable
with one command, and (d) excluded from anything that syncs to a cloud folder. State this in the server's `--help` and
in `docs/configuration.md` (O2).

**Sequencing note:** do L2 *before* L1. If the in-memory cache already covers normal turns, L1's value is restarts and
evictions, which changes the effort it deserves. This is the single most important measurement in the roadmap.

### W2 — Reading speed (prefill / TTFT on cold turns)

| ID | Item | Impact | Effort | Risk | Acceptance |
|---|---|---|---|---|---|
| R1 | **Chunk-size experiment**: package manifest `prefill_token_budget` 2048 → 4096 (compiled ceiling is `SPLASH_PREFILL_TOKEN_BUDGET`, 4096), and staging/lookahead sizes to match (`kPromptStagingExperts`, `SPLASH_LOOKAHEAD_EXPERTS`, `SPLASH_CHUNK_WORKING_SET_GIB`) | B (target +5–10%; Kapitsa measured **1.05×** cold prefill at 4K and 16K from the same change on their 35B, with a different staging design) | 2–3 d | low (measure, revert if worse) | Cold prefill tok/s at 4K and 16K, 3 runs each, medians; outputs identical |
| R2 | **Count where prompt time goes** with the existing trace (`[Prefill Timing]` resident/staging/GPU split) on 4K and 16K cold prompts; publish the split | — | 1 d | low | A table like `docs/profiling.md`'s, for prefill |
| R3 | **Expert reads per prompt token**: hypothesis — bigger chunks read fewer distinct experts per token; if R1 shows the opposite (staging-pool thrash), size the staging pool from the routing log (`SPLASH_ROUTE_LOG` + `cache_plan.py`) | B | 3 d | med | Reads/token down at equal or better prefill tok/s |
| R4 | **Short-prompt path**: the open "87 vs ~110 tok/s" item; check fixed overheads (warm-up, page-table sync, first-chunk staging) before touching kernels | B | 2 d | low | 1K-token prompt prefill tok/s within 10% of the 4K number |

### W3 — Speculation: spend the verify rows where they pay

| ID | Item | Impact | Effort | Risk | Acceptance |
|---|---|---|---|---|---|
| D1 | **Measured row budget** (port of Kapitsa `kflash_budget.*`): calibrated acceptance per depth + measured cost table per (model, chip), cache file, `E(n) − rate·cost(n)` per cycle; replaces `SPLASH_MTP_P_MIN`, `SPLASH_MTP_CHAIN_MIN`, `AdaptiveDraftController`, `SPLASH_MTP_ADAPTIVE_DEPTH` | B (target +2–5%) | 4–6 d | med | 10-prompt suite decode ≥ +2%, logit gate passes, the four env knobs are gone |
| D2 | **Draft only live rows** (open backlog item, ~0.5 ms/step) | C | 1–2 d | low | Step time −0.5 ms; identical tokens |
| D3 | **Batch budget** (`KfBatch` equivalent): with drafting off in batched lanes today (`Runtime.mm:859-868`), decide rows per lane by worth; re-test the "one request at a time" rule (`decisions.md` 2026-09-27) with numbers | B (only for multi-request use) | 4–5 d | med | Two concurrent agent loops: aggregate tok/s vs today's serialised default, plus per-request p50 ITL |
| D4 | **Tree verifier** — *only after D1 and E1*: KV store at parent depth + post-pass compaction of the accepted path (`q8_store_row.h:56` writes `committed + row` today), best-first builder over the MTP head's top-16 candidates, budget decides n | B (+2–5%, `derived`) | 8–12 d | **high** (numerics) | Same 10-prompt outputs under the M1 gate; tokens/step up at equal rows; `SPLASH_TREE_DRAFT` no longer a footgun (see Q1) |

**D4 is deliberately late.** The kernels already know about trees, which makes it tempting; but each extra row costs
~8.9 ms of SSD traffic here, so a branch must clear a ~0.34-token bar to pay (`kapitsa-review.md` §1). Kapitsa gets +6%
on a *dense, resident* model where a row is nearly free, and +3% on an MoE. The same mechanism is worth less here, and
it is the riskiest change in the list.

### W4 — Row cost and the expert store (the biggest decode bet)

| ID | Item | Impact | Effort | Risk | Acceptance |
|---|---|---|---|---|---|
| E1 | **Expert store v1**: one GPU arena + per-layer slot table read by the kernels, GPU→CPU event handshake per layer, reader-thread pool with `F_NOCACHE` preads and queue-jumping, **next-layer read-ahead from the current router** | **A** (row cost) | 10–15 d | **high** | `stall_ms`/`read_wait_ms` reported; row cost down ≥10% on the suite; generated tokens identical |
| E2 | **ARC replacement** over the shared arena, scan-resistant (prompt sweep must not evict the decode working set) | B | 3 d | med | Replay of a real session (`cache_plan.py`) shows fewer misses than LRU at the same bytes |
| E3 | **Route-aware drafting / row choice** (EcoSpec-style): prefer drafts whose experts are already resident; a term in D1's objective | C–B | 3–4 d | med | Measured on tool-call turns, where routing is repetitive |
| E4 | **Keep the mmap path as a fallback** behind one switch until E1 has run a week | — (risk) | included | — | Removing the flag changes no tokens |

**Why after W3:** E1's payoff depends on how rows are chosen (D1) and shows up as `stall_ms` (M2). Doing E1 first means
optimising a cost the policy is not yet steering.

### W5 — Quality assurance (the other half of "quality")

| ID | Item | Impact | Effort | Risk | Acceptance |
|---|---|---|---|---|---|
| Q1 | **Retire or finish `SPLASH_TREE_DRAFT`** — today a flag that turns on a hand-written tree whose rows are *verified* with parent masks but *stored* linearly; either delete flag+code (≤1 d) or gate it off until D4 | A (quality risk) | 0.5–1 d | low | `grep SPLASH_TREE_DRAFT` finds nothing, or the flag refuses to enable without compaction |
| Q2 | **Regression table**: one command that runs the 145-item quality suite *and* the logit dump on a fixed commit set, writing a row per build (accuracy, same-pick, KL, noise-floor deltas) | A (decision quality) | 3–4 d | low | Two builds can be compared in one table; the 79%-vs-89% code-prompt outlier is either reproduced or retired |
| Q3 | **Thermal / power-state honesty**: record `pmset -g therm` and AC/battery in every benchmark result; flag battery runs (Kapitsa flags them)`guess` | B (credibility) | 0.5 d | low | Every committed result carries the flag |
| Q4 | **Sampling correctness test suite**: T=0 identity with plain decoding, T>0 exact speculative sampling, greedy ties — as property tests over a fixed seed | B (quality) | 3 d | med | A deliberately broken sampler fails; the current one passes over 1k steps |

### W6 — Operations and developer experience

| ID | Item | Impact | Effort | Risk | Acceptance |
|---|---|---|---|---|---|
| O1 | **`slipstream doctor`**: report chip, memory, macOS, wired limit, which packages fit (using `MemoryPlan.cpp`, the same plan the engine uses), and the expert-cache default it would pick | B (support load) | 2 d | low | A 64 GB Mac answers "does my model fit, and with what cache" before anything loads |
| O2 | **One documented configuration surface**: 79 `SPLASH_*` knobs exist, 24 are documented. Classify them (supported / diagnostic / internal), document the supported set, delete dead ones | B (maintainability) | 2–3 d | low | A single `docs/configuration.md`; no supported behaviour requires an undocumented env var |
| O3 | **Fail loudly, never freeze**: keep `guarded.py`, but make the engine refuse an expert cache that cannot fit (it partly does) and surface the reason in the HTTP error instead of a hang | B (reliability) | 2 d | low | A too-large cache is rejected with a message naming the numbers |
| O4 | **Reconcile the docs with the code**: `.agents/tasks.md` (items done in code but unchecked), `DEVELOPMENT.md` ("inherited from Splash, out of date"), `docs/architecture.md` ("cleanup still to do") | C | 1 d | low | A reader can trust the checkbox that says "done" |
| O5 | **Commit the bench reports**: `bench/results`-style records under `dev/benchmarks/results/`, one per run, generated table (Kapitsa's model: 629 records, CI rejects a stale table) | B (credibility) | 2 d | low | The README table is generated, not hand-typed |

### W7 — Product (only after the above)

| ID | Item | Impact | Effort | Risk | Acceptance |
|---|---|---|---|---|---|
| P1 | **Second package (27B)** through the same paths, so "model-agnostic" is true rather than asserted | B | 5–8 d | med | Both packages pass M1's gate and M0's bench |
| P2 | **CLI parity** with the small conveniences: `list`, `rm`, `import`, `update` | C | 2–3 d | low | — |
| P3 | **Packaging cleanup** (`install/`, Homebrew, `QwenTarget` → explicit model interface) — already logged as optional in `tasks.md` | C | 3–5 d | low | — |
| P4 | **Do not** port an ANE split, a native grammar engine, or a third backend. Kapitsa's ANE work is ~5% of prefill on resident models and uses private APIs; neither moves an SSD-bound engine | — | — | — | — |

---

## 4. The recommended order

### Re-plan triggers (decide these from data, not from preference)

| If measurement shows | Then |
|---|---|
| L2: real turns are already ≥80% cache hits in memory | L1 drops to P3 (restart-only value); spend the days on R1 and E1 instead |
| D1: the budget's chosen row count beats today's fixed chain by <1% | do **not** start D4 (the tree); keep the budget for the batched case and move to W4 |
| M2: GPU stall waiting for expert reads is already <10% of step time | skip E1/E2; the remaining decode headroom is elsewhere (draft cost, kernel time) |
| R1: 4096-row chunks are neutral or worse | keep 2048, and spend W2's budget on the read-count axis (R3) only |
| M1: the noise floor is already as large as the difference between two candidate builds | treat speed work as free of quality risk and stop requiring dumps for policy-only changes |

### Budget

Each phase reserves roughly 30% of its days for measurement and review. The Mac runs **one engine at a time** (the
project's own rule, `.agents/status.md`), so benchmark blocks are scheduled on the calendar rather than squeezed in
between experiments; a phase that tries to share the machine with development will produce numbers nobody should trust.

### Priority table

| Priority | Items | Why this order |
|---|---|---|
| **P0 — now (weeks 1–2)** | M0, M1, M2, L2, Q1 | Cheap, irreversible-in-the-good-sense, and they create the yardstick. Q1 removes a correctness footgun the same week it is found. |
| **P1 — next (weeks 2–5)** | L1, R1, R2, R4, D1, D2, M3 | The two biggest user-visible wins (prefix survives; cold prefill faster), plus the decode policy that deletes four env knobs. All measurable against P0's baseline. |
| **P2 — after (weeks 5–9)** | L5, E1, E2, Q2, Q3, O1, O2, O3 | The row-cost work is the biggest decode bet and needs P1's numbers; quality and ops items make the project self-sustaining. |
| **P3 — later / conditional** | D3, D4, E3, R3, L3, O4, O5, P1–P3 | Conditional on measurements: tree only if D1 says branches pay; batching only if multi-agent use is real; second model only after the first is airtight. |

### Sequencing graph

```
M0 ─┬─> L2 ─> L1 ─> L3
    ├─> L5 (long context, cold-prefix sweep)
    ├─> R2 ─> R1 ─> R3
    ├─> D1 ─┬─> D2
    │       └─> D4 (only with Q1 done and E1 measured)
    ├─> M2 ─> E1 ─> E2 ─> E3
    └─> M1 ─> Q2 ─> (gates every default change)
M3 (Metal gate) ── independent, do early because it protects everything
```

### Phase 0 — the first ten days (concrete)

| Day | Work | Output |
|---|---|---|
| 1 | M0: run Kapitsa's harness with `dev/benchmarks/kapitsa-harness/slipstream.json` (four phases + `--agentic`); commit the result JSON | Baseline table: session wall clock, cached TTFT, cold prefill 4K/16K, decode T=0.6/greedy |
| 2 | L2: wire `CacheLookupSnapshot` into `/metrics`; replay one real 20-turn agent session and count prefilled vs reused tokens | A one-page answer to "does the prefix cache actually hit?" |
| 2–3 | Q1: remove the `SPLASH_TREE_DRAFT` code path and its constant parent table, or make it refuse to run | Flag gone; `grep` clean; token stream unchanged |
| 3–5 | M1: `logit_dump` + `logit_floor.py` for this Mac; run it on `main` twice to see the spread | Floor JSON; a pass/fail gate to reuse for the rest of the roadmap |
| 5–6 | M2: engine counters into `/metrics` | Reads/step, hit/miss/stall, row cost visible |
| 7–8 | R2 + R1: prefill split, then the 4096-chunk experiment | Prefill tok/s at 4K/16K before/after; a decision on the package's chunk size |
| 8–9 | D1 start: budget skeleton + cost table from M2, calibration from the existing trace records | Decision logged: expected tokens/step and chosen n per cycle in the log |
| 10 | Review: does L1 still look like the biggest win? Does D1 already show +2%? | Updated priorities, one table |

---

## 5. Acceptance gates that apply to every item

1. **Numerics:** `logit_compare` against the previous build's dump passes at 2× this Mac's floor (M1). No exceptions
   for "it's only the draft path".
2. **Output identity:** greedy output on the 10-prompt suite is byte-identical, unless the item is explicitly allowed to
   change it (then the noise-floor gate is the argument, and the PR says so).
3. **Speed:** three interleaved runs, medians, same power state, one server at a time, under the GPU lock; report the
   spread, not one number.
4. **No new env knob** unless it is a temporary diagnostic with a removal plan in the same PR.
5. **Docs:** the item's row in this table is updated with what was measured, including when it was worse than expected.

---

## 6. Risk register

| Risk | Where it bites | Mitigation |
|---|---|---|
| **Disk cache loads a corrupt or foreign state** | L1: silently wrong answers that look fast | CRC on payload *and* header, namespace digest over model/drafter/state layout, format version in the header, refuse-and-miss on any mismatch, CPU-only tests including a deliberately corrupted file |
| **The disk cache writes conversations to disk** | L1: a local-only privacy promise turns into a file someone syncs | opt-in/off by default, one documented directory, one-command delete, never inside a cloud-synced folder; say it in `--help` |
| **Serialized state drifts from the engine's layout** | L1 after a refactor | Put a layout hash in the namespace digest; add a test that fails when the state structs change without bumping it |
| **Expert store concurrency (readers vs GPU handshake)** | E1: rare hangs, hard to reproduce | Keep the mmap path behind a flag for a week; add a stress test that runs 10k handshakes with injected read delays; assert on `stall_ms` bounds |
| **Tree path produces plausible-but-wrong logits** | D4 (and today, behind the flag) | Q1 first; parent/depth invariants as unit tests; M1's gate on the actual verify path (its dump already covers `verify`) |
| **Kernel change lands unvalidated** | anywhere in W3/W4 | M3 (Metal gate) before starting kernel work, not after |
| **Thermals / battery flatter a result** | all benchmarks | Q3: record `pmset` state and AC/battery in every result; rerun on battery-flip |
| **One server at a time freezes the Mac** | benchmarking | Keep `guarded.py`; the harness takes `/tmp/kapitsa-gpu.lock`; never run two |
| **Scope creep into "support every model"** | P1: splits focus, breaks assumptions in `Runtime.mm` | Second model only after the first passes M0+M1+Q2; treat it as a separate release |

---

## 7. Scorecard (fill after each phase; this is the project's report card)

| Metric | Baseline (day 1) | Phase 1 target | Phase 2 target | Phase 3 target |
|---|---|---|---|---|
| Agent session, 20 turns @16–32K: wall clock | TBD by M0 | −15% | −30% | −40% |
| Follow-up TTFT (cache hit) | TBD | same | same | same |
| Follow-up TTFT after restart (16K) | full prefill | full prefill | ≤1 s | ≤1 s |
| Cold prefill tok/s @4K / @16K | TBD | +5% | +10% | +10% |
| Decode tok/s (T=0.6, 512-token answer) | 41–52 (measured) | +2% | +5% | +10% |
| Row cost (ms/verify row) | ~7.4–8.9 (measured) | — | −10% | −20% |
| GPU stall per step (ms) | not visible | visible | −25% | −50% |
| Quality: same-pick / KL vs bf16 reference | 91% / 0.12 (measured, playbook) | unchanged | unchanged | unchanged |
| Quality: flips vs previous build within noise floor | not measured | measured | 0 regressions | 0 regressions |
| Config knobs (supported / total) | 24 / 79 | — | 20 / 40 | 20 / 25 |
| Time from `git clone` to a benchmark number | TBD | 1 command | 1 command | 1 command |

---

## 8. Why not the things people usually suggest

- **A better drafter (DFlash-style).** Already evaluated and rejected with numbers in `docs/draft-head-plan.md`: a
  guesser trained on 3M local tokens scored 1.55 tokens/step against the shipped head's 3.28, i.e. 25 vs 39 tok/s.
  Nothing has changed since except Kapitsa's reminder that the *policy* around the drafter (calibrated acceptance, row
  budget) is where the easy remaining points are — that is D1.
- **Speculation depth.** More drafts lose money here: row 4 buys 0.24 tokens against a 0.34 break-even
  (`kapitsa-review.md` §1).
- **Batching by default.** Their own 2026-09-27 decision (one request at a time) was correct for a 95 GB model whose
  experts come off the SSD; revisit only with D3's budget and numbers.
- **Neural Engine / ANE.** Kapitsa's split helps prefill on resident dense models by single digits and uses private
  APIs. An SSD-bound engine gains nothing.
- **A second GPU backend (CUDA/ROCm).** It would dominate the roadmap and cannot be tested by this project. Keep the
  README's call for porting partners; do not start it.
- **Rewriting the server in C++.** The Python layer is not on the hot path (masks go over the pipe in blocks, and
  `/metrics` already shows decode/prefill rates that match the engine's); O2 and M2 are cheaper ways to remove the
  remaining friction.

---

## Appendix: where each number in this document comes from

| Number | Source |
|---|---|
| 41–52 tok/s decode, 1.76× llama.cpp | `README.md`; `dev/benchmarks/swift_benchmark_results/` |
| TTFT 0.84–2.34 s (six-prompt suite), 13.6 s avg at 16–32K | `README.md` benchmark and context-scaling tables |
| 2.82 tokens/step, 70.8 ms/step, acceptances 0.82/0.72/0.67/0.60/0.61 | `docs/draft-head-plan.md` |
| Step ≈ 32.2 + 8.94·rows + 2.23·guesses ms | `models/qwen4exp/bench/depth_controller_replay.py` (`C0, C1, M1`) |
| ~7.35 ms per extra row; 87 read-ahead + 46 on-demand reads/step | `docs/profiling.md` |
| Uneven slots: 95 → 87 reads, 39.7 → 41.5 tok/s | `models/qwen4exp/Qwen4Exp.cpp:313-336` comment, from `cache_plan.py` |
| 91% same-pick / KL 0.12 vs llama.cpp 89% / 0.18 | `docs/new-model-playbook.md` §7 |
| 79% same-pick on a code prompt (open item) | `.agents/tasks.md` |
| `prefill_token_budget` 2048 in the package, 4096 compiled ceiling | `models/qwen4exp/tools/convert_qwen4exp.py:832`; `.agents/decisions.md:202` |
| 79 `SPLASH_*` knobs, 24 documented | `grep -rhoE "SPLASH_[A-Z_]+" runtime models server install \| sort -u \| wc -l`, and the same over `README.md docs DEVELOPMENT.md` |
| Metal tests absent from CI | `.github/workflows/ci.yml` (jobs: `engine-check` CPU+sanitizers, `python-compatibility`, manual `release-hardware`) |
| Kapitsa comparisons (prefix cache 0.16 s vs 17 s, KFlash +6%/+3%, +23% batched, 629 runs) | `kapitsa-review.md` §3, sourced to their docs and bench records |
