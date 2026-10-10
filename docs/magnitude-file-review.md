# Magnitude — File-by-File Review, Benchmarks, and Improvement Ideas

**Date:** 2026-10-09 · **Companion:** [`magnitude-review.md`](magnitude-review.md) (the
architecture comparison). This document is the *file-level* pass: what each
decision-relevant file actually contains, what Slipstream can lift from it, and
the benchmark runs that were performed here.

**Method.** Sparse clone of `magnitudedev/magnitude@main` into `/tmp/magnitude`
(`git clone --depth 1 --filter=blob:none --sparse`, then
`sparse-checkout set inference inference-v2 cli packages docs design web specs old-inference`):
**5,625 files, 121 MB**. File inventory and line counts were produced for every
`*.rs`/`*.metal`/`*.md`/`*.ts` in scope (478,262 lines under `inference/`
alone). Then the decision-relevant files were read directly — not just their
design docs. Everything below cites a real path and, where it matters, a symbol.

## 0. Coverage ledger (honest accounting)

| Scope | Lines | Treatment |
|---|---:|---|
| `inference/engine/{scheduler,generation,chat,grammar,batching}` | 24,154 | **Read**: `policy.rs`, `prefix_cache.rs`, `round.rs`, `mtp.rs`, `dflash.rs`, `acceptance.rs`, `preparation.rs` in full; other files enumerated only |
| `inference/engine/{state,executor,kernels,serving,artifacts,families}` | 155,829 | Headers, structure and key symbols only; `native/tuning/mod.rs` symbol census |
| `inference/seismic/**` + `solver/**` | 237,237 | Design docs read earlier; one file (`softfloat.metal`, 3,707 lines) noted only |
| `inference/service/**` | 34,531 | File inventory + `residency/mod.rs` (2,469) located; not read line-by-line |
| `cli/**` + `packages/**` (34 packages) | 224,319 | `harness-connections`, `shell-classifier`, `utils/patch`, `tracing`, `roles`, `skills` enumerated; `shell-classifier/classifier.ts` and `roles/prompts/leader.txt` read |
| `web/**`, `desktop/**`, `old-inference/**`, `inference-v3|v4/**` | large | Inventory only — deliberately deprioritized (superseded or UI) |

**Not read, and flagged as such:** the Seismic compiler internals, the CUDA/Vulkan
backends, `state/src/lib.rs` (5,883 lines — read as headers only), and everything
under `web/`. Claims about those areas below are design-doc-level, not
source-level, and are marked accordingly.

## 1. What the engine files actually contain

### 1.1 `inference/engine/scheduler/src/policy.rs` (211 lines)

The whole concurrency policy in one small file:

- `ServiceLimits { prefill_tokens, decode_tokens, decode_share, locality_seconds }`
  with `selection_bound()` = decode allowance and `launch_rows()` =
  `max(prefill, decode)`; `validate()` rejects `decode_tokens > 256` and a share
  outside `(0, 1)`.
- `Scheduler::select` chooses `Phase::Decode` when decoding and
  (prefill absent **or** first contended selection **or** `decode_debt_ns > 0`),
  else prefill. Priority is
  `waiting_ns + locality_seconds·1e9·(active + resident + preemption_debt)`.
- `Scheduler::completed` is where the debt lives:
  `decode_debt -= elapsed` on decode, `+= elapsed·share/(1-share)` on prefill —
  the exact formula from the design doc, in ~6 lines.
- `order_victims` ranks preemption candidates by `output_blocked`, then
  `preemption_debt`, then the ratio `exclusive_bytes × replay_tokens`
  (cross-multiplied in `u128` to avoid overflow), then `service_ns`, then id.
- `AvailabilityEpoch` — a counter advanced **only** when capacity is freed.

**Why this matters to Slipstream:** this is the smallest possible implementation
of roadmap item **W5 option (b)** ("interleave bounded prefill chunks between
decode steps"). Slipstream is single-tenant with `"total_slots": 1`
(`server/server.py:347`) and `SPLASH_MAX_BATCH_WIDTH=1`, so it does not need
`order_victims`, `AvailabilityEpoch`, or the per-request state machine. It needs
the ~40 lines of `select`/`completed`: a debt counter and a phase choice.

### 1.2 `inference/engine/scheduler/src/prefix_cache.rs` (766 lines)

A complete, tested prefix cache in one file. Constants and mechanisms:

- `MIN_PREFIX_HIT = 64`, `MIN_BRANCH_GAIN = 128` — "a branch point costs a split
  prefill chunk and a retained recurrent bank; below this the recomputed prefill
  is cheaper."
- `PrefixRetention::{Transient, Retain { cache_points }}` — the request opts in
  and *declares* the prompt positions where it expects divergence.
- `PrefixPath` = prompt + accepted output + `InputLayout`; `exact_boundary()`
  refuses positions inside an indivisible media span; `conditioning_at()` yields
  the media identities before a boundary; `shared_boundary()` returns the deepest
  exact boundary where tokens **and** conditioning identities agree — "equal
  placeholder tokens with different media diverge at the image."
- `PrefixCache::best` requires `position >= MIN_PREFIX_HIT`,
  `position < bound` (a resumed request must still compute the row it samples
  from), and an exact shared boundary; ties break by recency.
- Eviction: entry-count bound only, LRU, **never an entry in submitted use**
  (`begin_submitted`/`end_submitted` reference counting); the cache deliberately
  owns no byte budget — "what an eviction releases is Seismic's charge, observed
  by the memory heap after the entry's state drops."
- The 12-test module is the specification: terminal turn seeds the next turn,
  divergent questions share the deepest common prefix, a state at the last
  message serves *every* changed message, media identity mismatch diverges before
  the image, submitted entries block capacity without overcommitting.

### 1.3 `inference/engine/chat/src/preparation.rs` (313 lines)

The "declared cache point" idea, implemented two ways:

- `CONTENT_PROBE = "\u{E000}"`; `last_message_boundary()` clones the request with
  the last message's content replaced by the probe, re-renders the template, finds
  the first differing byte, and maps it back through `added_token_ends` to a
  prompt token position. `next_turn_boundary()` appends probe assistant + user
  turns to find where the *conversation* leaves this prompt (a template may render
  the generation prompt unlike the same turn in history).
- `shared_boundary()` returns "the position after the last added token within the
  bytes this prompt shares" and filters out positions at/after the prompt end.
- `prepare()` has a documented certification fallback: if the template's
  parallel-tool-call grammar cannot be certified because "certification renders
  lexemes one byte at a time and the first mask exceeds the matcher's item limit",
  the request is prepared for one tool call per turn and whichever plan has fewer
  character lexemes wins.

### 1.4 `inference/engine/generation/src/{round,mtp,dflash,acceptance}.rs` (~1.0k lines)

- `round.rs`: `RoundKind::{Progress, Forced, Verification}`; `proposal_limit` =
  `min(allowance, remaining, output_credit) - 1` (0 for a plain method);
  `reconcile` accepts the matching proposal prefix then emits the
  target-sampled bonus token; `publication_bound` accounts for forced tokens and
  prefill emission, not just selection rows. `RoundForward.prime` carries the
  drafter entry that is drafted **behind the prompt chunk on the device**.
- `acceptance.rs` (47 lines): `accept_prefix` — "a stop token is never accepted as
  a draft match; it may only be emitted as the target-sampled bonus token." Tabular
  test at the bottom.
- `mtp.rs` / `dflash.rs`: one `Drafter` enum, two forms (`Chained` for MTP, `Block`
  for DFlash), separate checkpoints so "another method's checkpoint cannot restore
  it". The load-bearing rule: *committed drafter rows are target-conditioned only;
  rows the drafter conditioned on its own features while drafting are never kept.*

### 1.5 Grammar, tuning, kernels (enumerated + symbol census)

- `inference/engine/grammar/src/` (4,309 lines): `lexeme.rs` (876), `certify.rs`
  (611, greedy-exactness certification of the token mask), `terminal.rs` (525),
  `network.rs` (354), `gbnf.rs` (323), `vocabulary.rs`, `elimination.rs`,
  `cache.rs` — i.e. a grammar→mask engine with a *certification* pass that proves
  the mask is exact rather than approximating it.
- `inference/engine/executor/src/native/tuning/mod.rs` (2,645 lines):
  `TuningLimits`, `PointShape`, `row_points`, `served_row_points`,
  `with_contexts`, `attention_points`, `search_settings_for(share, backend)`,
  `TuningPlan`, `TunedEntry`, `TuningEvent`, `SearchProgress`,
  `UnreportedTuning`, plus failure catalogue hooks — the implementation behind the
  Seismic design doc.
- `inference/engine/kernels/tests/` (36,331 lines across 34 files):
  `projection.rs` (5,486), `routed.rs` (4,230), `attention_k8v4.rs` (3,208),
  `golden.rs` (2,995), `recurrent_stages.rs` (2,135), `cuda_projection.rs`,
  `routed_general.rs`, `attention_family.rs`. A kernel conformance corpus larger
  than most of Slipstream's engine.
- `inference/docs/kernels/lane-order.md` (170 lines) and
  `drafting-and-verification.md` (110) document the lane order each backend's
  cooperative multiply assumes.

### 1.6 Integrations and harness packages

- `packages/harness-connections/src/`: `service.ts` (445), `shared.ts` (251),
  `contract.ts` (150), `inspection.ts` (98), `connectors/{pi-package 219,
  codex 188, cline 124, openclaw 114, opencode 102}`, plus `service.test.ts` (2,175)
  and `transaction.test.ts` (128). The transaction/observation discipline is
  backed by tests an order of magnitude larger than the code.
- `cli/src/agent-docs/topics/*.md` (374 lines): `cli`, `custom-endpoints`,
  `recommendations`, `remote-access`, `speculative-methods` — the CLI ships
  agent-readable topic docs so an agent can self-serve setup answers.
- `packages/shell-classifier/src/` (4,770 lines): `classifier.ts` exposes
  `classifyShellCommand`, `isGitAllowed`, `writesStayWithin`, `isPathWithin`;
  `parser.ts` (405) plus per-domain tools `container.ts` (238), `sysadmin.ts`
  (230), `cloud-cli.ts` (212), `iac.ts` (162), and git/database/k8s/package-manager
  modules, with 10 test suites (classifier, parser, git, container, iac, kubernetes, database, package-managers, cloud-cli, sysadmin).
- `packages/utils/src/patch/` (3,010 lines): `apply.ts` (316), `diff.ts` (163),
  `compiled-map.ts` (351), `types.ts` (128) plus `exhaustive.test.ts` (1,718).
- `packages/tracing/src/` (157) and `packages/tracing/dashboard/` — Svelte
  components `ForkTree`, `LogprobTooltip`, `RawTokenStrip`, `TokenRenderer`,
  `SessionDetail`.
- `packages/roles/src/prompts/` — 11 role prompts (`leader`, `architect`,
  `engineer`, `critic`, `scout`, `scientist`, `artisan`, `advisor`, `observer`,
  `autopilot`, `autopilot-task`) and `shared/{agent-common,thinking-shared,worker-base}.txt`.
  `leader.txt` states the safety traits plainly: "Always ask before mutating the
  user's files", "No high stakes or irreversible operations without explicit
  confirmation from the user."

## 2. Benchmarks and checks actually run here

All commands run in the Slipstream checkout on this machine (**Apple M5 Pro,
Metal 4**). Exit status captured for each.

| # | Command | Result | RC |
|---|---|---|---|
| 1 | `make test-engine-cpu` | all engine test targets pass | **0** |
| 2 | `./build/engine-tests/attention-sweep build/slipstream.metallib --histories 8192,32768 --shapes 27b --lanes 1,4 --repeat 20` | prefill fused 23.5 ms @8k, 89.2 ms @32k; verify 0.613 ms @8k/1 lane, 0.616 ms @8k/4 lanes, 2.06 ms @32k/4 lanes; split kernel dominates (22.9 of 23.5 ms) | 0 |
| 3 | `.venv/bin/python dev/benchmarks/benchmark_mach_vm_remap.py` | 64 MB remap in **6.25 µs**, shared-page mutation visible in 0 ns, **898.4×** vs memcpy (11.95 GB/s) | 0 |
| 4 | `.venv/bin/python dev/benchmarks/benchmark_ipc_transport.py` | Unix socket vs TCP: small 0.17 vs 0.25 ms (**1.50×**), medium 1.27×, large (100 KB) **0.86×**, deep (400 KB) **0.47×** | 0 |
| 5 | `.venv/bin/python dev/benchmarks/benchmark_kv_vault.py` | vault load 0.16–2.67 ms vs estimated cold prefill 31.2–3,125 ms → **108–1,182×** (best case 1,181.8× at 50k tokens); 100k tokens in 2.67 ms vs 3,125 ms | 0 |
| 6 | live `/v1/chat/completions` against the already-running server (`local/swift-qwen38-27b-splash-hq`, PID 93556, pre-existing — not started or stopped by this review) | **45.3 / 45.3 / 45.4 tok/s** decode (234 tokens × 3 runs, greedy, TTFT 262–416 ms) | — |
| 7 | live prefix-cache probe: 1,819-token prompt, three ways | cold TTFT **4,450 ms** (~409 tok/s prefill) → identical repeat **282 ms** (15.8×) → *same prefix, different final question* **267 ms** | — |

**Findings from the benchmark runs:**

1. **The live decode number matches the recorded one.** 45.3 tok/s on the 27B HQ
   model is consistent with `.agents/tasks.md` (44.4 vs 43.9 tok/s; 40–52 tok/s
   band), so the current build has not drifted.
2. **Slipstream already covers Magnitude's headline cache case.** With only the
   last message changed, TTFT was 267 ms — indistinguishable from an identical
   repeat. Slipstream's radix-block cache shares the whole prefix up to the
   divergence at the *end* of the prompt. This narrows borrow item **B3** (declared
   cache points) far more than the design doc suggested; see §3.1.
3. **The Unix-socket IPC win is payload-dependent.** 1.50× on small prompts, but
   *slower* than TCP at 100 KB (0.86×) and 400 KB (0.47×). Any claim that the
   token bus helps deep-context turns is not supported by Slipstream's own
   benchmark at the sizes where it matters most.
4. **`attention-sweep` is a manual autotuner.** It prints one hand-chosen
   (history, lanes) point per run and a per-kernel breakdown that shows the split
   kernel at 97% of prefill time. That is exactly the input a Magnitude-style
   tuner consumes — and exactly the work Slipstream is currently doing by hand.

**Blocked / not run (reported as limitations, not passes):**

- **Magnitude itself was not built or run.** `cargo`/`rustc` are absent on this
  machine and `bun` is not installed. Its 2×-vs-llama.cpp figures therefore remain
  vendor claims in this review.
- **No cold model load was attempted.** A server was already serving on 8090;
  switching models would pin ~40 GiB and disturb a live process, so the existing
  one was left untouched and only client requests were issued.
- `dev/benchmarks/test_27b_profile.py` / `test_features_profile.py` were not run
  (they need a fresh serve cycle); `benchmark_kv_vault.py` needed the repo
  virtualenv (`.venv/bin/python`) because `tokenizers` is not in the system
  interpreter.

## 3. Improvement ideas (file-backed, prioritized)

Regression risk is against the non-negotiable "no quality or speed regression"
constraint. Where a risk changed after the live probe, it is marked *revised*.

### 3.1 Prefix cache and multi-turn serving

| # | Idea | Magnitude file | Slipstream target | Risk | Cost |
|---|---|---|---|---|---|
| P1 | **Split the W5 decision.** Take prefill interleave (single-lane path preserved) now; keep width>1 batching as a separate, later design. | `scheduler/src/policy.rs` (`decode_debt_ns`) | `runtime/engine/Scheduler.cpp` | **Low** — interleave changes scheduling order, not numerics or geometry. *Revised down from the earlier "A/B required", which assumed width>1.* | ~1 day |
| P2 | **Declared cache points, narrow form.** Retain a state below the divergence when the divergence is *not* a 32-token block boundary and the state cannot be cut back to a shorter prefix. | `scheduler/src/prefix_cache.rs` (`Retain { cache_points }`, `MIN_BRANCH_GAIN`) + `chat/src/preparation.rs` (`last_message_boundary`) | `runtime/engine/Cache.cpp`, `KvCache.cpp` | **Low** — cache hit only; output is always verified against a cold run. *Revised: the end-of-prompt case already works (267 ms probe), so only the mid-prompt / recurrent / spliced cases need this.* | ~2–3 days |
| P3 | **Path-conditioned resume for vision.** A cached prefix's identity must include media identities, so equal placeholder tokens with different images diverge. | `prefix_cache.rs` (`conditioning_at`, `shared_boundary`) | `runtime/engine/KvCache.*` (vision is live) | Low, but needs a vision correctness test | ~1–2 days |
| P4 | **Bound the cache by entry count, not bytes, and let the memory owner price eviction.** | `prefix_cache.rs` (header comment; `evict_one`) | `Cache.cpp`, `MemoryGovernor.cpp` | Low | ~1 day |
| P5 | **Reference-count in-flight uses** so an entry in submitted use is never evicted. | `prefix_cache.rs` (`begin_submitted`/`end_submitted`) | `KvPool.cpp` | Low | ~1 day |

### 3.2 Speculation

| # | Idea | Magnitude file | Slipstream target | Risk | Cost |
|---|---|---|---|---|---|
| P6 | **Never keep drafter rows conditioned on the drafter's own features** — *already satisfied for the MTP path*: `Qwen4ExpTarget.cpp:3067` refreshes accepted rows with the target's own residuals, and `:3218` documents that the next step's guesses "overwrite these positions with the accepted tokens' real rows". Remaining scope is an **audit of the DFlash path** (`runtime/model/DFlashDraft.cpp` ring commit) against the same rule. | `generation/src/mtp.rs` (module doc + `observe`) | `runtime/model/DFlashDraft.*` | Low | hours (audit) |
| P7 | **Stop tokens are never accepted as draft matches** — only as the target-sampled bonus token. | `generation/src/acceptance.rs` | `runtime/metal/kernels/decode/sampling.metal` | Low | hours |
| P8 | **Per-method checkpoints** so one drafter cannot restore another's state (MTP vs DFlash vs PLD in one session). | `mtp.rs` (`Drafter::restored`) | `Runtime.mm` draft paths | Low | ~1 day |
| P9 | **Per-model speculation catalog with separate draft artifacts** + "no MTP on vision models" policy. | `design/models/speculative-methods.md` | model manifests, `install/launcher.py` | Low — target verifies every draft | ~2 days |

### 3.3 Kernel tuning and conformance

| # | Idea | Magnitude file | Slipstream target | Risk | Cost |
|---|---|---|---|---|---|
| P10 | **Autotune the parameters `attention-sweep` already exposes** (history × lanes × split), with numerical validation per candidate and a replayable recorded result. | `executor/src/native/tuning/mod.rs`, `design/inference/seismic-compilation.md` | `dev/benchmarks/attention_sweep.mm`, `runtime/metal/MetalBackend.mm` | Low — identical-output gate already exists | ~1 week |
| P11 | **Adopt the measurement hygiene wholesale:** first pass is never a sample; keep a cold device busy until clock stabilizes (>20 ms idle can read 4× slow); re-measure the K cheapest within 10%; unformable configurations cost +∞. | `seismic-compilation.md` | `attention_sweep.mm`, `backend_benchmark.mm` | Low | ~1–2 days |
| P12 | **Grow the kernel conformance corpus.** `inference/engine/kernels/tests/` is 36,331 lines for 15,401 lines of kernel source (2.4:1); Slipstream's `build/engine-tests/*` are behavioural, not per-kernel golden. | `kernels/tests/{golden,projection,routed,attention_k8v4}.rs` | `dev/tests/engine/` | Low | ongoing |
| P13 | **Document per-backend lane/repack order** (Slipstream has repacked layouts and a quantizer but no `lane-order.md` equivalent). | `inference/docs/kernels/lane-order.md` | `docs/` | Low | hours |
| P14 | **Certify grammar masks are *exact*, not merely plausible**, and degrade deterministically when certification cannot be reached. | `grammar/src/certify.rs`, `chat/src/preparation.rs` (character-lexeme fallback) | `server/constraints.py`, `Runtime.mm` mask path | Low | ~2–3 days |

### 3.4 Agent harness: safety and editing *(highest value per line of code)*

| # | Idea | Magnitude file | Slipstream target | Risk | Cost |
|---|---|---|---|---|---|
| P15 | **Classify shell commands before running them.** Block (`isGitAllowed`), confine writes (`writesStayWithin`, `isPathWithin`), and domain-classify (`container`, `sysadmin`, `cloud-cli`, `iac`, k8s, db). | `packages/shell-classifier/src/` (4,770 lines, 10 test suites) | `server/agent/tools.py` `run_command` | **Low** — a *guard* narrows what the agent does; it changes policy, not model output | ~3–5 days |
| P16 | **Write files atomically through a patch layer** instead of in-place string replacement, with an exhaustive apply/diff test suite. | `packages/utils/src/patch/` (3,010 lines) | `server/agent/tools.py` (`replace_file_content`, `write_file`) | Low | ~2–3 days |
| P17 | **Put "ask before mutating" and "no irreversible operations without confirmation" into the system prompt**, and make `/undo`, `/clear` obey it. | `packages/roles/src/prompts/leader.txt` | `server/agent/agent.py` `SYSTEM_PROMPT` + slash commands | Prompt text changes the request → **A/B required** | ~1 day |
| P18 | **Ship agent-readable docs as CLI topics** so the harness can self-answer setup/serving questions. | `cli/src/agent-docs/topics/*.md` | `docs/`, `./slipstream --help` | Low | ~1 day |

> P15/P16/S-safety is the single most defensible borrow: `docs/agent-harness-review.md`
> already flags `/undo` running `git checkout .` with no confirmation (F4), and
> `run_command` is unguarded. Magnitude spent 4,770 tested lines on exactly this
> problem.

### 3.5 Memory, artifacts, distribution

| # | Idea | Magnitude file | Slipstream target | Risk | Cost |
|---|---|---|---|---|---|
| P19 | **Residency key = (artifact, tensor, representation, layout)** with tied weights sharing storage, per-backend fastest layout, and exact-or-`Unsupported` at plan time. | `service/server/src/residency/mod.rs` | `runtime/model/WeightStore.*`, `MemoryPlan.cpp` | Low | ~1 week |
| P20 | **Charge per physical allocation and release when a session stops** (the mechanism behind "27% less memory per agent"). | `seismic`/memory docs, `residency/mod.rs` | `MemoryAudit.cpp`, `MemoryGovernor.cpp` | Low if accounting-only; eviction needs a soak test | ~1 week |
| P21 | **Connection manager with observation and compensation** (Connected ≠ receipt; Unavailable config is never overwritten; reverse-order compensations; idempotent apply; conditional restoration). | `packages/harness-connections/src/` (`service.ts`, `inspection.ts`, connectors) | `install/clients.py` | Low | ~1 week |
| P22 | **Model catalog / Discover experience** as a first-class surface. | `service/models/src/{catalog,inventory,download,preview}.rs` | `install/catalog.py`, `install/models.py` | Low | ~1 week |
| P23 | **Decode trace viewer** with per-token logprob and fork tree. | `packages/tracing/dashboard/` | `server/metrics.py`, `docs/` | Low | ~1 week |

### 3.6 Benchmarking and observability

| # | Idea | Magnitude file | Slipstream target | Risk | Cost |
|---|---|---|---|---|---|
| P24 | **Session-shaped benchmark**: pinned tool-call fixtures, prose-copy, retrieval with distractors, explicit fork/concurrency/memory sections; "no retention claim follows from session shape alone". | `design/inference/session-bench.md`, `inference/benchmarks/**` | `dev/benchmarks/` | Low | ~1 week |
| P25 | **Checked-in cross-engine experiment corpus** (their `packages/inference-benchmark/experiments/*.ts` pins model + engine + sweep and records results). | `packages/inference-benchmark/` | `dev/benchmarks/comparison_*_results.json` | Low | ~2–3 days |
| P26 | **Fix the IPC claim.** The benchmark shows the Unix socket is *slower* above ~100 KB; either confine the fast path to small turns or fix the large-payload path, and stop citing a flat speedup. | — (Slipstream's own #4 result) | `server/agent/agent.py` (`UnixHTTPConnection`) | Low | ~1 day |

## 4. Sequencing

1. **This week, near-zero risk:** P16, P26, P13, P7, P11, P24. (Editing safety,
   a corrected claim, documentation, one acceptance rule, measurement honesty,
   the missing benchmark rung.)
2. **Next, low risk, real payoff:** P15 (shell classifier) and P1 (prefill
   interleave). P1 is the only concurrency work that does not touch numerics, and
   P15 closes a live destructive-command gap.
3. **Then:** P3, P5, P6, P8, P9, P10, P14, P21.
4. **Design-first, do not rush:** P2 (declared cache points — only the mid-prompt
   and recurrent cases remain), P19/P20 (residency and memory charging), and any
   width>1 batching.
5. **Prompt-text changes (P17) must be A/B'd** on the 145-item scorecard before
   becoming defaults — they change what is sent to the model.

## 5. Limitations of this review

- Magnitude could not be executed here (no Rust toolchain, no Bun), so its
  performance claims are unverified; only its *code and design* were reviewed.
- `state/src/lib.rs` (5,883 lines), the Seismic compiler, the CUDA/Vulkan
  backends and all of `web/` were not read line-by-line; §0 lists the treatment.
- The live numbers come from a single pre-existing server on one machine
  (M5 Pro, 27B Q4 HQ). They validate the current build's baseline, not any
  proposed change; every borrow item above still needs its own before/after run.
- Slipstream source was read (not modified). Any implementation of P1–P26 must be
  validated with `make test-engine-cpu`, the identical-greedy-output check, and —
  for P17 — the 145-item scorecard.
