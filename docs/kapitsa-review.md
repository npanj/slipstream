# What Slipstream can take from Kapitsa (and what it can't)

**Reviewed:** [nace-ai/kapitsa](https://github.com/nace-ai/kapitsa) at `e6e1f46` (2026-10-22), against this tree
(`main` @ `2c7ac4d`). Kapitsa is Apache-2.0, like Slipstream, so code can be ported with attribution — and Kapitsa
itself credits Splash 1.4.0's streamed experts for its expert store, i.e. it is already downstream of the codebase
Slipstream forked.

**Short answer.** Kapitsa is not faster at the thing Slipstream does: it runs models that **fit in RAM** and never
streams 95 GB from the SSD. Its numbers (Flash-Next 1165/106 tok/s) are on a 128 GB M5 Max with 105.9 GB resident,
against Slipstream's 41–52 tok/s on a 64 GB M5 Pro streaming 95.5 GiB. Those are different problems.

What is genuinely borrowable is **policy and machinery**, in this order:

| # | Borrow | Where it pays | Effort | Expected |
|---|---|---|---|---|
| 1 | **SSD prefix cache** (`src/engine/prefix_cache.*`, `engine_disk.cpp`) | TTFT on agent turn 2+, and after a restart | M | 10–100× better follow-up TTFT (Kapitsa: 16K follow-up 0.16 s vs 17 s) |
| 2 | **Measured verify-row budget** (`kflash_budget.*`) replacing `SPLASH_MTP_CHAIN_MIN`/`SPLASH_MTP_P_MIN` | decode tok/s, every step | S–M | +2–5%, and removes hand-tuned env knobs |
| 3 | **Finish the tree verifier** (KFlash's builder + the missing KV compaction) | decode tok/s | L | +3–8% (Kapitsa measured +6% on the 27B, +5% on the 4B, +3% on the 35B-A3B); the plumbing is ~80% there already |
| 4 | **Expert store design** (`expert_store.h`: one arena, ARC, per-layer slot table, GPU↔CPU handshake, reader pool) over today's `mmap`+`MADV_WILLNEED` | row cost (8.9 ms/row) and GPU stall time | L | each 1 ms off a row is ~1.3% tok/s |
| 5 | **Noise-floor logit gate** (`CONTRIBUTING.md#output-check`) | output quality guarantees | S–M | makes numerics claims testable instead of anecdotal |
| 6 | **Cross-engine bench protocol** (`bench/harness`, 629 recorded runs, flags, provenance) | the README comparison itself | S | credibility; a config file is provided in `dev/benchmarks/kapitsa-harness/` |

Numbers 1, 2, 5 and 6 are cheap and low-risk. 3 and 4 are real engineering; do them after 1/2 with the existing
trace tooling.

---

## 1. Where Slipstream's time actually goes

Slipstream already measures this better than most projects, so the leverage ranking is arithmetic, not taste
(`docs/draft-head-plan.md`, `docs/profiling.md`):

- step ≈ **32.2 ms + 8.94 ms × rows verified + 2.23 ms × draft guesses**
  (`models/qwen4exp/bench/depth_controller_replay.py`, constants `C0, C1, M1`; `docs/profiling.md` quotes 7.35 ms/row
  on a different run — SSD reads dominate either way, ~16–20 extra experts per row)
- today: **2.82 tokens/step at 70.8 ms**, acceptances by depth **0.82 / 0.72 / 0.67 / 0.60 / 0.61**
- TTFT: 0.84–2.34 s on the six-prompt suite; every agent turn pays it again unless the prefix is cached

Applying their own acceptance and cost numbers to a plain chain (computed here, not measured):

| guesses k | rows | step ms | E[tokens] | tok/s |
|---:|---:|---:|---:|---:|
| 0 | 1 | 41.1 | 1.00 | 24.3 |
| 1 | 2 | 52.3 | 1.82 | 34.8 |
| **2** | **3** | **63.5** | **2.41** | **38.0** |
| 3 | 4 | 74.7 | 2.81 | 37.6 |
| 4 | 5 | 85.8 | 3.05 | 35.5 |
| 5 | 6 | 97.0 | 3.19 | 32.9 |

Two conclusions that should drive the roadmap:

1. **The chain should be short (2–3 guesses) and the marginal row must buy ≥ 0.34 tokens** (break-even = rate × row
   cost = 0.038 tok/ms × 8.94 ms). Row 4 buys 0.238 — it loses money. That is exactly what the hand-tuned
   `SPLASH_MTP_CHAIN_MIN=0.35` is approximating, and why `trace_report.py` found 39.1 tok/s at 0.35 vs 38.0 at 0.45.
2. **A tree only wins where an alternate branch has path probability above that same 0.34 threshold.** In Kapitsa's
   dense case (8 rows ≈ 1.1 × a 1-row step) almost any branch pays, hence +6%. On Slipstream's SSD-streamed MoE each
   row is a full 8.94 ms, so only near-tie branches near the root pay — which is why Kapitsa's own MoE result is
   +3–5%, not +6%, and why the honest expectation here is **+2–5%, not +15%**.

Then the other levers, in order of size:

| Lever | Why it matters here |
|---|---|
| **TTFT / prefix reuse** | 0.8–2.3 s per turn against 41–52 tok/s decode: a 32K agent turn can spend more time prefilling than decoding. This is the biggest real-world win and the least numerically risky. |
| **Row cost (8.94 ms)** | 1 ms off a row ≈ +1.3% tok/s at chain 3; it also raises the tree's break-even odds. |
| **GPU stall on on-demand expert reads** | "87 ahead + 46 on demand" per step (`docs/profiling.md`): the on-demand ones are pure GPU idle. |
| **Draft cost (2.23 ms/guess)** | 7% of a 3-row step; only worth attention after the above. |

---

## 2. The two engines, side by side

| | Slipstream | Kapitsa |
|---|---|---|
| Target | One 125.7B-A7.3B hybrid, 95.5 GiB **streamed from SSD** on 64 GB | Qwen 4B/9B/27B/35B, Nemotron, Flint — **resident** in RAM |
| Expert memory | mmap + `MADV_WILLNEED` on last step's routing, per-layer fixed-capacity slots, LRU (`models/qwen4exp/Qwen4Exp.cpp:676-784`) | one GPU arena, per-layer slot table, **ARC**, 16 reader threads, `F_NOCACHE` preads, GPU↔CPU event handshake, next-layer read-ahead from the current router (`expert_store.h`), with hit/late/miss/stall counters |
| Drafters | model's MTP head (chain, confidence stop) + prompt lookup | DFlash/DFlash2, DSpark, LiLiCorr block drafters (**16 candidates per position**), MTP fallback |
| Verify | 8 rows/lane; drafting is single-lane only, so batching gives 1 row per lane (`Runtime.mm:859-868`, `ExecutionLimits` in `Model.hpp:164-177`) | 8-row tiles (16/32 opt-in) for dense; **1–8 rows/cycle chosen per cycle** for MoE; batched decode shares one pass |
| Speculative policy | fixed thresholds from env, 7 hard-coded tree shapes behind `SPLASH_TREE_DRAFT` | online **calibrated** acceptance (logistic fit per depth) + **measured cost table** per chip/model, cached to disk (`kflash_budget.h`) |
| Prefix reuse | in-memory KV blocks + composite state checkpoints (`Cache.cpp`, `StateCache.cpp`) — **no disk** | the same, plus **SSD prefix cache**: KV blocks and states as CRC-checked files, keyed by content (`prefix_cache.h`, `engine_disk.cpp`) |
| Server | Python/llguidance for grammar and masks over a binary pipe | native C++ grammar/mask (`grammar.cpp`, `logit_mask.cpp`) |
| Accelerators | Metal 4 `matmul2d` for 4-bit prefill + MoE; no ANE | Metal 4 tensor ops (QMT) incl. MoE prefill; **Neural Engine split** for selected prefill work |
| Quality gate | `compare_logits.py` vs a bf16 reference | **noise-floor** gate: `logit_dump` → `logit_compare`, differences allowed only within 2× what the same build moves between legitimate paths |
| Benchmarks | 6 prompts, one run, no flags | 629 recorded runs, frozen corpus + nonce, GPU lock, contention/invalid flags, cached-TTFT and agent-replay phases, provenance JSON, generated table |

---

## 3. What to borrow, concretely

### 3.1 SSD prefix cache — the biggest real-world win

Kapitsa: `src/engine/prefix_cache.{h,cpp}` (941 lines, CPU only, no model needed to test), `engine_disk.cpp` (760),
`tools/prefix_cache_test`. Their reported result: a 16K-token follow-up starts in **0.16 s instead of 17 s** (~100×),
and a new question on a shared system prompt in **0.06 s instead of 1.9 s**; the cache also survives a restart.

The design worth copying verbatim:

- **Namespace digest** over the model's files, the drafter and the state layouts, so no model reads another's cache.
- **Three file kinds**: `kv/<key>` (one KV block's rows), `st/<key>` (a recurrent/draft-ring state snapshot),
  `e/<key>` (an entry: a conversation's tokens, its blocks, its resume points).
- **Content keys**: tokens up to the block's end + the requests that computed them, so a conversation's turns share
  earlier blocks and a write only adds the new ones.
- **4 KiB header** (magic, kind, ns, key, sizes, CRC-32C of payload and header) + payload, written to a temp file then
  renamed, all I/O `F_NOCACHE`.
- **Quota**: `min(quota, max(0, avail + held − floor) / 4)`, so the cache's own files don't shrink its limit, and the
  rest of the volume filling up does.
- **Per-namespace `flock`** so two servers can't fight, and an `about.txt` for humans.

Where it plugs into Slipstream (no new concepts needed — every hook exists):

| Hook | File | What to add |
|---|---|---|
| Block publish | `runtime/engine/Cache.cpp` `publishCommittedBlocks` | hand the (key, rows) pair to the store instead of only to the pool |
| State boundaries | `runtime/engine/StateCache.cpp` | write composite states tagged by boundary; they are already keyed by KV block |
| Admission | `Engine::admitKv` / resume boundary (`Engine.cpp`) | look up the longest cached prefix, restore KV pages + state, set `promptProcessed` |
| Eviction | `runtime/engine/MemoryGovernor.cpp` | spill pinned prefixes to disk instead of dropping them |
| Reporting | `server/protocol.py` usage | report `cached_tokens` (the harness and users need it) |
| Test | new CPU-only test next to `dev/tests/engine/kv_page_cache_test.cpp` | write/read/CRC-corrupt/namespace-isolation, no GPU |

Two details that are easy to get wrong. Expert residency does **not** belong in the key or the entry: which experts are
cached is a performance detail, not a correctness one, so a restored prefix is valid whatever the arena holds. The
**draft ring / MTP state** does belong in it (Kapitsa stores `dctx/dvalid` per restore point); otherwise the first turn
after a restore drafts from a stale ring and the first guesses are garbage.

### 3.2 A measured verify-row budget instead of env-var thresholds

Kapitsa's `kflash_budget.*` is ~950 lines that replace every constant in Slipstream's draft path with two online
estimates:

1. **Calibration.** A candidate with drafter probability `q` at depth `d` is accepted with
   `p = σ(α_d + β_d·logit q)`, refit by logistic regression every 16 cycles over the last 2048 records — where a record
   is every verified node whose **parent was accepted**, and whether the target took it. Before any data, `α=0, β=1`.
   This is exactly what Slipstream prints as its "confidence calibration" table in `trace_report.py` but never feeds
   back into the decision.
2. **Cost.** Per row count `n`: a base that moves with context length plus `n`'s own offset, measured on this Mac, with
   a small exploration phase and a **cache file per (model, drafter, chip)**; before 16 cycles it uses `E(n)/cost(n)`,
   after that `E(n) − rate·cost(n)` where `rate` is the recent tokens/ms (a row must beat what the machine is already
   producing).

Then each cycle picks `n` (1–8) that maximises the objective, and `KfBatch` does the same over a whole batch's rows.

For Slipstream this is the cheapest real speed-up available: it deletes `SPLASH_MTP_P_MIN` (0.3), `SPLASH_MTP_CHAIN_MIN`
(0.35), the `AdaptiveDraftController` heuristic (`Qwen4ExpTarget.cpp:2562`), the fixed 7-draft tree and the
`SPLASH_TREE_DRAFT` flag semantics, and replaces them with one policy that adapts to prompt type, temperature, cache
state and Mac. The measurement inputs already exist:

- `[Verify Timing]` (`mtpParts`, `hostParts`) and `SPLASH_STEP_TIMING=1` give per-step GPU/host time splits;
- `SPLASH_TRACE` already records per-depth candidates, confidences and what was accepted;
- `trace_report.py` already fits `step = a + b·rows` and replays policies offline — it just never ships the winner.

Suggested landing spot that respects `dev/tools/check_architecture.py`: put the budget in `runtime/ops/DraftBudget.{hpp,cpp}`
(no model names, no engine dependency), have `Runtime.mm` ask it for a row/depth choice, and persist the table under
`~/Library/Application Support/Slipstream/draft-budget/<model>-<chip>.json`.

Also borrow the **sizing rule for the batch**: today `lanes > 1` gives every lane exactly one row, i.e. batching
silently disables speculation (`Runtime.mm:859-868` `mtpShadow`; `Qwen4ExpTarget.cpp:1908` is the workaround that then skips
hyper-connection work for rows past the first). Kapitsa runs the same weight read once for all
members and then spends the 8-row budget where it buys most (`KfBatch`; their reported gain is up to **+23% total
throughput** for three requests at once on the 27B). With SSD-streamed experts the win is smaller (experts are per-row),
but the budget decides that automatically: on a 64 GB Mac it will simply choose one row per lane, which is what the code
does by hand today, and the recorded table will say so instead of a comment. On a resident-model build (a 27B, say) it
will choose more. Either way this is also the safest place to re-open the "one request at a time" decision recorded in
`.agents/decisions.md` (2026-09-27): measure it, don't assume it.

### 3.3 The tree verifier: less work than `trace_report.py` claims

`models/qwen4exp/bench/trace_report.py` says "Upper bound rows need a tree verifier (not built)". Actually most of it
exists behind `SPLASH_TREE_DRAFT=1`:

| Piece | Status |
|---|---|
| Ancestor mask in page softmax | ✅ `q8_attention_tile.h:84` `tree_ancestors`, nibble parent table |
| Depth-based RoPE per row | ✅ `Runtime.mm:760-775` `kTreeDepths{0,1,2,3,4,2,1,2}` |
| GDN conv taps along the parent chain | ✅ `gdn_primitives.h:7-30` |
| DeltaNet scan from the parent's state | ✅ `gdn.metal:140` `p = tree_parent(...)` |
| Tree-aware acceptance/sampling walk | ✅ `sampling.metal:566-639` |
| Fixed tree shape | ⚠️ hard-coded `0x60132100` in `Runtime.mm:901` + hand-written Structure A draft path (`Qwen4ExpTarget.cpp:2983-3067`) |
| **KV store at depth + compaction of the accepted path** | ❌ `attention_q8_store.metal`/`q8_store_row.h:56` write `logical_token = committed + row`, so two branches at the same depth collide and no post-pass compaction exists |
| **Tree builder + budget** | ❌ (this is what `kflash_budget.*` + `src/kernels/kflash.h` give you) |

So the remaining work is: (a) store verify rows per node and move the accepted path into consecutive positions after
the pass (Kapitsa: "the accepted path's K/V rows move into consecutive positions … the other rows are dropped"), (b) a
best-first builder over the MTP head's top-16 candidates (`draftCandidates[7][16]` / `draftProbabilities` are already
collected), (c) the budget from §3.2 to choose how many rows.

Kapitsa's builder is a good model for it: node score = product along the path of calibrated probabilities, pop the best
frontier node, push its children, stop at n rows; because a child never scores above its parent, the first n rows are
the best n-row tree for every n — **so the size can be decided after the tree is built**. It runs as one 32-thread GPU
kernel at the end of the drafting pass, so building never waits on the CPU (worth copying: Slipstream's draft pass is
already GPU-resident).

Expected: +2–5% at equal row cost (see §1), and it makes the row budget more valuable because branches compete for rows
on merit. **Do §3.2 first** — it stands alone, is cheaper, and gives the numbers that tell you whether the tree earns
its keep on your prompts.

### 3.4 Expert store: one arena, ARC, handshake, read-ahead from the router

This is the part where Kapitsa is genuinely better engineered, and it is where both engines' numbers come from —
Kapitsa explicitly credits Splash 1.4.0 (`ExpertStore`, `SlotRecycler`, `SsdReaders`), i.e. **Slipstream's ancestor had
this and Slipstream replaced it with `mmap` + `madvise`**.

What `src/model/expert_store.h` does that `models/qwen4exp/Qwen4Exp.cpp:753-784` does not:

| | Slipstream today | Kapitsa / Splash 1.4 |
|---|---|---|
| Placement | weights are `mmap`ed; "cache" is the page cache plus per-layer fixed slot tables | explicit GPU arena of slots; per-layer slot table (`E` words: chunk \| page) read by the expert kernels, so resident and streamed experts run the **same kernel** |
| Prefetch source | `lastSelectedExperts[l]` — **this step's** routing for the **same** layer, i.e. a 1-step-lagged guess via `MADV_WILLNEED` | the **next layer's** experts predicted from the current layer's router output, read during the current layer's GPU work |
| Replacement | LRU per layer (`lruTime`, `slotToExpert`) | **ARC** over one shared arena, scan-resistant; a prompt's one-time sweep neither promotes nor reorders, so the decode working set survives |
| Reads | page-cache dependent (`madvise`), no control over pollution | `F_NOCACHE` `pread` on 16 reader threads; 16 KiB-aligned pieces read **straight into the slot**; a read the turn waits for jumps the queue |
| Pathological case | page cache thrash with a 95 GB model on 64 GB | explicit, bounded, and measured |
| Observability | `predictIssued` / `predictUseful` counters | `handshakes, lookups, hits, late, misses, prefetches, prefetch_used, evictions, bytes, stall_ms, read_wait_ms` |

`stall_ms` and `read_wait_ms` are the numbers that tell you whether to spend a week here; Slipstream's
`SPLASH_ROUTE_LOG` + `cache_plan.py` replay can estimate the same from routing logs without touching the engine — do
that first (their `cache_plan.py` already replays eviction rules and an uneven split; `simulate_cache.py` compares
policies). Then port the mechanism, not the numbers.

Two sub-ideas with their own quotes:

- **Uneven slots per layer** (~9% fewer SSD reads in Kapitsa's own replay note). `models/qwen4exp/Qwen4Exp.cpp:325-330`
  sizes per-layer capacity statically today; `cache_plan.py` can compute the uneven split and there is already a
  `SPLASH_EXPERT_SLOTS` escape hatch.
- **Route-aware row selection** (EcoSpec, already listed in `docs/draft-head-plan.md`): prefer drafts whose experts are
  resident. With the budget from §3.2 this becomes a term in the objective rather than a separate heuristic.

### 3.5 Quality: the noise-floor gate

Kapitsa's rule (`CONTRIBUTING.md#output-check`): a default changes only after an **interleaved A/B** and a logit
comparison against the old build, allowing only differences within **twice this Mac's noise floor** — where the floor
is measured as how far the logits move between legitimate paths of the *same* build (alone vs batched, plain decode vs
verify, chain vs tree, prefill chunk 512 vs 2048, ANE split vs GPU only). It fails a change on mean/p99 KL, on the flip
count, and on any flip that was not a near-tie.

Slipstream has `models/qwen4exp/bench/compare_logits.py` (KL vs a bf16 **reference**) and
`check_decode_consistency.py`, but nothing that measures its own path-to-path noise. Consequences worth fixing:

- You cannot tell a real regression from the 4-bit model's own jitter on a new Mac or a new kernel.
- Turning on trees, batches, or a different prefill chunk cannot be gated numerically — only by "does the answer look
  the same".
- `runtime/ops/PromptLookup` + MTP + tree paths change logits by construction; a floor makes that a measurement.

Cheapest version for Slipstream: a `logit_dump` mode that writes (prefix tokens, per-row fp32 logits) for the paths
`decode`, `verify`, `batch(4)`, `prefill` at fixed prefixes — Slipstream can reuse the layout
`compare_logits.py` already reads — plus `logit_floor.py` that measures the spread between those paths and writes a
per-Mac JSON. Their dump writes ~190 MB; that is fine on a 1 TB Mac and it is the difference between "40.8 tok/s,
output unchanged" and "40.8 tok/s, we think output is unchanged".

### 3.6 The benchmark protocol — this is what "quality of the comparison" means

Slipstream's README table is 6 hand-written prompts, **one run each with no warm-up**, averaged across 6 prompts,
on one macOS version and power state, with no repetition-loop check, no contention control, no nonce (so an engine's own
prefix cache can help a rerun), no cached-TTFT phase and no provenance. (`dev/tools/compare_slipstream_vs_llamacpp.py`
does exclude TTFT from the decode rate and records TTFT separately, which is right.) Kapitsa's harness (`bench/harness/`, Apache-2.0,
stdlib-only Python, runs on macOS) is engine-agnostic — it drives **any** OpenAI-compatible `/v1/chat/completions` —
and it fixes every one of those weaknesses:

- **One engine at a time under a GPU lock** (`flock /tmp/kapitsa-gpu.lock`), + a quiet-GPU wait, + a 2 s contention
  monitor that flags a run `CONTENTION` with the culprit.
- **Cold prompts** cut from a frozen corpus with a random nonce after `<|im_start|>system`, so no cached prefix helps,
  and a ±1% token-count check against a recorded reference (catches truncation and chat-template drift).
- **Decode rate over the middle of the stream** (first 16 and last 4 tokens dropped) — TTFT cannot flatter it; 5 runs,
  median; must end on `length` at exactly 512 tokens; **distinct-4-gram ratio < 0.5 flags a repetition loop**, which is
  the failure mode that inflates speculative decoding.
- **Greedy and T = 0.6** with every sampler parameter sent explicitly (llama.cpp defaults `min_p=0.05`, Ollama's tag
  sets `presence_penalty=1.5`), thinking off and *checked* (`<think>` in the answer ⇒ `INVALID`).
- **Cached TTFT** phases: a follow-up on a grown conversation, three new questions on a shared prefix, then a second
  conversation and back — which is exactly the axis Slipstream wins or loses real agent sessions on, and which
  Slipstream currently does not measure at all.
- **Provenance**: exact command, env (secrets redacted), weight files and sizes, macOS version, power source, swap, GPU
  memory, every contention sample.
- A generated results table (`bench/gen_readme.py`) with CI refusing a stale one.

**A ready config is in `dev/benchmarks/kapitsa-harness/`** — copy it into `kapitsa/bench/harness/configs/`, then:

```bash
cd kapitsa/bench/harness
cp local.env.example local.env          # set BENCH_MODELS, SLIPSTREAM_ROOT
python3 compare_engines.py configs/slipstream.json --dry-run
python3 compare_engines.py configs/slipstream.json configs/flash-next.json configs/llama.cpp.json --only 'flash-*'
python3 compare_table.py                # markdown table of the newest clean run per config
```

**But be honest about the regime.** The only fair head-to-head is the same Mac, the same weights, one slot, thinking
off:

- On a **64 GB** Mac, Kapitsa must stream too (`qwen4xp`'s Q2_0 routed experts through `expert_store.h`), so this is a
  genuine apples-to-apples test of the *streaming* designs — and the most interesting number the two projects could
  produce. If Kapitsa wins it, the expert store (§3.4) is the reason; if Slipstream wins it, the MTP head plus PLD is.
- On a **128 GB** Mac nothing is streamed, and Kapitsa's 106 tok/s is not evidence about Slipstream's design at all. The
  current README contrast (1.76× over llama.cpp) should keep saying what it measured — the same Mac, same file, one
  prompt set — and must not be read next to Kapitsa's M5 Max row.

Also worth borrowing as-is: **`kapitsa doctor`** reports which models fit this Mac using the same memory plan the engine
uses. Slipstream has `MemoryPlan.cpp` and a governor; exposing it as `slipstream doctor` would remove the most common
support question (which quant fits a 64 GB Mac with 96 GB of weights on disk).

---

## 4. What *not* to borrow

| Item | Why not |
|---|---|
| **8-row dense verify economics** | Kapitsa's "8 rows cost ~1.1× a 1-row step, so a tree is nearly free" holds because every weight is resident. Slipstream pays 8.94 ms/row in SSD reads; the tree must clear the 0.34-token break-even per row, which is a much higher bar. |
| **DFlash/DFlash2/LiLiCorr drafters** | They are separate trained models for other checkpoints. Nothing exists for Flash-Next; `docs/draft-head-plan.md` already evaluated and rejected training one (3M tokens ≠ enough). |
| **Neural Engine prefill split** (`src/ane/`) | Kapitsa's split is ~5% of prompt time on resident models and uses private framework interfaces; for a model whose bottleneck is SSD expert reads it moves nothing. Do not take on private API risk for this. |
| **Metal 4 QMT prefill** | Slipstream already runs `matmul2d` 4-bit prefill tiles (`q4_mpp_tiles.h`, `prefill/moe.metal`, `gguf_staged_tile.h`). Difference is tuning, not architecture — measure yours before porting theirs. |
| **Native grammar engine** | Slipstream's llguidance-based masks are computed for a whole draft block per round trip (`simulateTokens` × `wordsPerMask` in `Protocol.cpp`), and drafts are already pruned by the mask. Kapitsa's `grammar.cpp` is 2000 lines of work for parity, not advantage. |
| **Their absolute numbers** | 105.9 GB resident on an M5 Max is a different machine regime; quoting it beside 41–52 tok/s on 64 GB would be the kind of claim their own harness flags as `INVALID`. |

---

## 5. Suggested order of work

Each step ends in a number, from tools that already exist.

| Step | Change | Ends when |
|---|---|---|
| 1 | `slipstream doctor`: expose the existing memory plan as a CLI report | A 64 GB Mac says which quants fit, with the plan's numbers |
| 2 | Port the SSD prefix cache (§3.1) with a CPU-only test | Trace shows `cached_tokens`; a 16K follow-up's TTFT drops ≥ 5× on turn 2 and survives a restart |
| 3 | Row budget (§3.2) behind `SPLASH_DRAFT_BUDGET=1`, fed by existing trace/timing hooks | A/B on the 10-prompt suite: equal or better tok/s, `SPLASH_MTP_*` env knobs deleted, table cached per chip |
| 4 | Noise floor (§3.5) | `logit_floor.py` writes a JSON for this Mac; step 3's A/B is gated by the dump, not by eyeballing |
| 5 | KV compaction + tree builder (§3.3) behind the budget | Identical greedy output on the suite (logit gate), tokens/step up at the same rows |
| 6 | Expert store (§3.4): arena + slot table + handshake + router-driven read-ahead, instrumented | `waiting for SSD reads` per step down; `stall_ms`/`read_miss` visible; identical tokens |
| 7 | Bench harness config (§3.6) in CI or a release checklist | One table, one Mac, flags and provenance, regenerated by script |

Steps 2, 3, 4 and 7 are each a few days and none of them can change the model's output if done right — the acceptance
rule stays the target's own. Steps 5 and 6 are the ones to schedule deliberately, and 6 is where the remaining 1.76× →
more is hiding.

---

## Appendix: sources

- Kapitsa: `docs/kflash.md` (tree design, calibration, sizing), `docs/architecture.md` (slots, turns, batching,
  expert store, prefix cache, ANE), `src/model/kflash_budget.h` (policy + cost table + cache file),
  `src/engine/prefix_cache.h` (file format, keys, quota), `src/model/expert_store.h` (ARC, handshake, read-ahead,
  stats), `bench/harness/README.md` (protocol, fairness rules), `CONTRIBUTING.md#output-check` (noise floor).
- Slipstream: `docs/architecture.md`, `docs/profiling.md` (cost per row, SSD read counts),
  `docs/draft-head-plan.md` (acceptance by depth, row cost, the "tree verifier not built" note),
  `models/qwen4exp/bench/{trace_report,depth_controller_replay,cache_plan,compare_logits}.py`,
  `models/qwen4exp/Qwen4ExpTarget.cpp` (draft/tree/PLD paths), `models/qwen4exp/Qwen4Exp.cpp:676-784`
  (expert slot tables, prefetch, evict), `runtime/model/Runtime.mm:760-903` (tree positions and parent table),
  `runtime/metal/kernels/common/{q8_attention_tile.h,gdn_primitives.h}`, `runtime/metal/kernels/decode/{gdn,sampling,
  attention_q8_store}.metal`, `runtime/engine/{Cache,StateCache,KvCache,MemoryGovernor}.cpp`.
- Numbers in §1 are computed from Slipstream's own published acceptance rates (0.82/0.72/0.67/0.60/0.61) and its
  measured cost model (`C0=32.2, C1=8.94, M1=2.23` ms); they are arithmetic, not measurements.
