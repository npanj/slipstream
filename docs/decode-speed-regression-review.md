# Decode speed regression review — 2026-10-09

Measured on `main` at `19c4414` — the 2026-10-05…10-09 work — on the same
machine and model packages used for the published baselines.

## Conclusion

Decode has regressed on both served models, measured against the numbers in the
project's own release posts. On the identical 6 prompts behind
`dev/benchmarks/comparison_swift-qwen38-flash-next-v3_results.json`, Flash-Next
V3 decodes at 31.2 tok/s against a published 41.7 (−25%). On the identical
5-domain suite in `npanj/splash-plus`, the 27B decodes at 30.8 against a
published 36.9 (−17%). Both are same-prompt comparisons, so the gap is not a
workload artifact. See *Published vs measured* below.

The regression is real, and it is entirely per-step latency — not draft yield.

The published `origin/main` state was rebuilt on this machine (commit `6c0ba57`,
2026-10-04 21:15, which contains no `adviseHorizonPredicted` at all) and run on
the identical prompts, package and harness — see *The published build, rebuilt on
this machine* below:

| build | tok/s | tok/step | ms/step |
|---|---|---|---|
| published `origin/main` (10-04), default cache | 40.7 (40.6–40.8) | 2.87 | 70.6 |
| published `origin/main` (10-04), 34 GiB | 40.6 | 2.87 | 70.8 |
| current build, default cache | 35.2 (34.8–35.5) | 3.06 | 87.0 |
| current build, 34 GiB | 32.7 (31.9–33.6) | 3.06 | 93.4 |

Decode rate is `1000 × tokens-per-step ÷ ms-per-step`, so at the default cache the
move from 40.7 to 35.2 tok/s (−13.5%) decomposes as ×1.066 from tokens per step
*against* ×0.811 from per-step latency: **per-step cost grew 16.4 ms/step (+23%)
and tokens per step moved the other way, up 6.6%.** The draft-yield axis that
looked dominant earlier in this review was an artifact of comparing against the
journal's 51.9 tok/s / 4.38 tok/step record: that record is not the published
build, and the published build has *lower* yield than today's. Yield is not the
regression.

The one identified per-step cost, `adviseHorizonPredicted(layer + 2)` (added
2026-10-09 12:54), accounts for 16.9 ms/step of the 22.6 measured at 34 GiB, and
**nothing at the default cache — where the full +16.4 ms/step regression stands
unattributed.** (*Confirmed by three-way isolation* below.)

Speculation is **not** broken on either model. Flash-Next runs 3.06 tokens per
step and the 27B runs 3.61 (greedy) / 3.85 (T=0.2). A dead drafter reads ~1.0
tokens per step; that is not what is happening.

## Published vs measured

Both comparisons below run the **same prompts** as the published benchmark, on
the same machine class (M5 Pro, 64 GB), greedy, against the current build.

Flash-Next V3 — the 6 tasks of `dev/tools/compare_slipstream_vs_llamacpp.py`:

| task | published | current build | delta |
|---|---|---|---|
| gsm8k_math | 44.5 | 33.5 | −25% |
| math500_series | 43.7 | 32.0 | −27% |
| constraint_logic | 46.2 | 30.5 | −34% |
| python_intervals | 36.4 | 26.9 | −26% |
| rust_csv | 39.9 | 29.9 | −25% |
| tech_explanation | 39.6 | 34.1 | −14% |
| **AVERAGE** | **41.7** | **31.2** | **−25%** |

The published column is `comparison_swift-qwen38-flash-next-v3_results.json`;
the release post's own table reports 40.8 for the same six tasks. `gsm8k_math`
(196 tokens generated here against 192 published) and `rust_csv` (1200 against
1199, both at the cap) are the clean controls: generation length is not what
produces the gap. The 145-item scorecard figure of 43.9 for the same model
agrees within a few percent of this 6-task average.

27B Splash-HQ native 8-bit — the 5 domains of `npanj/splash-plus`
(`scripts/run_all_models_standard_bench.py`, temperature 0.0, max_tokens 250):

| domain | published | current build | delta |
|---|---|---|---|
| Math & Logic | 54.8 | 39.3 | −28% |
| Coding & Algorithms | 34.7 | 31.9 | −8% |
| Constraint Reasoning | 39.3 | 29.9 | −24% |
| Domain Knowledge | 21.9 | 21.9 | 0% |
| Nuanced Writing | 33.7 | 30.8 | −9% |
| **AVERAGE** | **36.9** | **30.8** | **−17%** |

A correction worth recording: matching these tasks to the local suite by *task
name* suggested the two builds were within ~5%, because a local "GSM8K" or
"Rust CSV" prompt is not the published prompt. Only running the published
prompts shows the real, larger gap. Task-name similarity is not a comparison.

## How this was measured

Two harnesses, both memory-guarded:

```zsh
# Flash-Next V3, 10 prompts x 256 tokens, same suite the journal's numbers came from
dev/benchmarks/guarded.py --max-seconds 900 -- \
  .venv/bin/python models/qwen4exp/bench/speed_suite.py --temperature 0.0 --tokens 256

# 27B Splash-HQ, 4 prompts x 128 tokens
SPLASH_MAX_BATCH_WIDTH=1 dev/benchmarks/guarded.py --max-seconds 900 -- \
  .venv/bin/python dev/benchmarks/bench27b_speed.py "$PWD" ~/models/swift-qwen38-27b-splash-hq 128
```

`build/engine-tests/generate-sample` and `decode-profile` had not compiled since
the 2026-10-05 vision port added `RuntimeContext::maximumImagePatches` to
`ModelFactory.hpp`; the dev harnesses still initialised the struct positionally,
so every field after `operators` shifted into the wrong member. One argument
added to each fixes it. That break is why no offline decode baseline existed to
compare against.

## Flash-Next V3, greedy (2,498 tokens per run)

Two samples are shown for the configurations that carry the argument; the rest
are single runs.

| build / configuration | tok/s | tok/step | ms/step |
|---|---|---|---|
| working tree, as-is | 34.6, 32.6 | 3.06 | 88.4, 93.9 |
| as-is + `SPLASH_DISABLE_RDADVISE=1` | 38.4, 35.0 | 3.06 | 79.6, 87.3 |
| `Qwen4ExpTarget.cpp` reverted to 10-04 HEAD | 38.2 | 2.87 | 75.2 |
| HEAD file + new MTP thresholds (0.15/0.20) | 38.7 | 3.06 | 79.0 |
| as-is + old expert frequency (half-life 512, sticky floor 0) | 29.9 | 3.06 | 102.4 |
| *recorded 2026-10-08 (20-item suite)* | *~43* | *3.19* | *~74* |
| *recorded, same harness, best* | *51.9* | *4.38* | *84.4* |

Run-to-run noise on timing is about ±8% on this machine (125.5 against 115.7
ms/step for identical work), well above the ±2% the project notes assume. That
is why the pivotal comparison was repeated. Paired repeats of the advisory pair
gave -8.8 and -6.6 ms/step; means 91.2 ms/step with the advisory against 83.5
without, so the direction and rough size hold up. Tokens per step is exactly
3.06 in every one of these runs — greedy drafting is deterministic, so only the
timing varies between them.

## The published build, rebuilt on this machine (2026-10-09)

`origin/main` is commit `6c0ba57` (2026-10-04 21:15), the code state behind the
README's benchmark tables, and it contains no `adviseHorizonPredicted` at all
(`grep -c` = 0), confirming the call is a 10-09 addition. It was checked out into
a scratch worktree, built from scratch (41 objects, no errors), and driven on the
same prompts file, the same package (`~/models/swift-qwen38-flash-next-v3/prepared`)
and the same harness wrapper as the current build, interleaved across two cache
settings and two reps each. It honours `SPLASH_EXPERT_CACHE_GIB` and defaults to
the same `physicalGiB − 28 = 36 GiB`, so the arms are like for like.

Two observations beyond the headline:

- **The published build is insensitive to the expert cache budget** — 70.7 against
  70.8 ms/step, 40.6 tok/s either way. The current build is not: 87.0 at the
  default against 93.4 at 34 GiB. Whatever landed made decode more sensitive to
  how much of the expert set is resident, which is a lead in its own right.
- The 10-04 engine loads today's prepared package without complaint, so this is
  two builds of code against one set of weights, not two packages. What it does
  not exclude is the weights having changed under both builds at some earlier
  date.

## Where the extra time goes: expert misses (2026-10-09)

Both builds were run on one prompt with `SPLASH_STEP_TIMING=1`, which emits one
`[Verify Timing]` line per decode step, and the lines were averaged:

| build | misses/step | staging | miss reads | pure GPU | GPU wall |
|---|---|---|---|---|---|
| current | **128.1** | 41.2 ms | 34.0 ms | 50.8 ms | 82.0 ms |
| published 10-04 | **73.2** | 21.2 ms | 21.9 ms | 44.5 ms | 64.7 ms |

**The current build misses 75% more experts per step.** That is the regression:
staging and the host reads that serve it grow by about 20 ms/step, with a much
smaller +6 ms on pure GPU compute. It also explains the cache sensitivity noted
above — a build missing 128 experts a step is far more exposed to how many slots
the budget buys than one missing 73.

This pointed at expert cache policy, and the victim-eviction safeguard added in
`Qwen4ExpTarget.cpp` — which refuses to admit a new expert when the incumbent's
frequency is higher and the candidate has fewer than two votes — was the prime
suspect. It is only a small part of it. Giving it its own switch
(`SPLASH_NO_EVICTION_GUARD`, default unchanged) and re-running the same prompt:

| build | misses/step | staging | miss reads | pure GPU | GPU wall |
|---|---|---|---|---|---|
| current, guard on | 128.1 | 41.2 ms | 34.0 ms | 50.8 ms | 82.0 ms |
| current, `SPLASH_NO_EVICTION_GUARD=1` | 113.1 | 39.2 ms | 31.0 ms | 50.8 ms | 79.8 ms |
| published 10-04 | 73.2 | 21.2 ms | 21.9 ms | 44.5 ms | 64.7 ms |

The guard costs ~15 misses/step and ~2 ms/step — real, but about a quarter of the
55-miss gap. The code bounds the rest: this guard appears exactly once here and
**zero times in the 10-04 baseline**, so the other ~40 extra missing experts per
step come from somewhere else in the residency path. The admission order is the
obvious next place to look, since the predictor-v2 change touches how candidates
are ordered before they are offered to the cache.

Single prompt, 18–21 decoded steps per build: directional, not yet a constant.

### Confirmed by three-way isolation — conditional on the expert cache (2026-10-09)

Neither call in the lookahead block could be measured through the switches that
already existed, because `prefetchPredicted(layer + 1)` and
`adviseHorizonPredicted(layer + 2)` share a single condition. Giving the horizon advisory its own switch and running
all three arms interleaved — three reps each, alternating order, one session, one
environment — removes that ambiguity and the session drift with it. At
`SPLASH_EXPERT_CACHE_GIB=34`, which is what the published 6-task harness sets
(`dev/tools/compare_slipstream_vs_llamacpp.py:337`):

| arm | tok/s (3 runs) | ms/step (3 runs) | tok/step |
|---|---|---|---|
| as-is: prefetch + horizon advisory | 32.7 (31.9–33.6) | 93.4 (91.1–95.9) | 3.06 |
| `SPLASH_NO_HORIZON_ADVISORY=1`: prefetch kept | 40.0 (39.5–40.5) | 76.5 (75.6–77.3) | 3.06 |
| `SPLASH_NO_LOOKAHEAD=1`: both off | 40.0 (39.9–40.1) | 76.4 (76.2–76.6) | 3.06 |

The horizon advisory is worth **16.9 ms/step, +22% tok/s**. Keeping the prefetch
while dropping it costs nothing — 76.5 against 76.4 ms/step — so
`prefetchPredicted` is free and should stay. No as-is run is as fast as any run
of either other arm, and tokens per step is 3.06 throughout.

**The same three arms at the default dynamic expert cache show no effect at all**
(2 reps each, same session and harness, variable unset):

| arm | tok/s (2 runs) | ms/step (2 runs) | tok/step |
|---|---|---|---|
| as-is | 35.2 (34.8–35.5) | 87.0 (86.2–87.8) | 3.06 |
| `SPLASH_NO_HORIZON_ADVISORY=1` | 34.5 (33.6–35.4) | 88.7 (86.3–91.0) | 3.06 |
| `SPLASH_NO_LOOKAHEAD=1` | 36.1 (35.4–36.8) | 84.7 (83.1–86.3) | 3.06 |

Every arm sits within ±4% of the others and the ordering reverses between reps.
So the advisory is not free-standingly expensive: **its cost is a function of how
tight the expert cache is.** At 34 GiB the run is already miss-bound and its
speculative reads contend with the misses that matter; sized dynamically at 36 GiB
here, there is headroom and it costs nothing measurable. That is consistent with
the bandwidth-contention story above, but it is inference — this establishes the
conditional, not the mechanism.

The cache budget also matters on its own account, with no advisory switch
involved: the same as-is build decodes at 32.7 tok/s at 34 GiB against 35.2 at the
default size, 8.6 ms/step of pure cache budget. The earlier rows in the table
above match the 34 GiB arm here (32.6 against 32.7 tok/s, 93.9 against 93.4
ms/step), so those were taken with the variable pinned too. The server script uses
32 GiB (`~/models/bin/splash-flashnext-server.sh`, retuned down from 34), tighter
than either case measured here.

Sampled decode, temperature 0.7 (2,430 tokens):

| configuration | tok/s | tok/step | ms/step |
|---|---|---|---|
| default (EAGLE-2 tree drafting on) | 28.4 | 2.92 | 102.9 |
| `SPLASH_EAGLE2_DRAFT=0` | 30.4 | 3.00 | 98.7 |
| *recorded, same harness* | *51.8* | *4.57* | *88.2* |

## Root cause

`adviseMissedExperts()` is now called synchronously from the host routing path
whenever misses are staged, and `adviseHorizonPredicted(layer + 2)` runs once per
pipelined stage. The latter scans up to 512 experts of a future layer, takes the
eight with frequency ≥ 4 that are not resident, and issues three
`fcntl(F_RDADVISE)` calls each. That is on the order of a thousand syscalls per
token, on the host thread that must stage experts between GPU pipeline stages.

Measured cost, three-way isolated at `SPLASH_EXPERT_CACHE_GIB=34`: **16.9
ms/step (+22% tok/s) for the horizon advisory and 0.1 ms/step for
`prefetchPredicted`.** Only the horizon call is worth acting on, and only at that
cache budget — the same isolation with the variable unset shows no effect, so read
the numbers below before acting. The synchronous miss-path `adviseMissedExperts`
is not separately measurable — leaving it enabled and switching off only the
horizon call reaches the same 40.0 tok/s as switching the whole block off.

The advice is also speculative readahead for *guessed* experts, so it competes
for SSD bandwidth with the real misses it is meant to hide.

## Ruled out by measurement, not by reading

- **Lowered MTP thresholds** (`SPLASH_MTP_P_MIN` 0.3 → 0.15, `SPLASH_MTP_CHAIN_MIN`
  0.35 → 0.20). Near-neutral on their own: 38.2 → 38.7 tok/s, and they raise
  tokens per step 2.87 → 3.06 at the cost of ~4 ms/step. Not the regression.
- **MoE-Infinity sticky affinity** is a win, not a cost. Reverting it (half-life
  512, sticky floor 0) drops the run to 29.9 tok/s / 102.4 ms/step.
- **EAGLE-2 tree drafting** cannot explain the greedy regression — it requires
  `mtp.temperature > 0` and greedy never enters it. It does cost 7% in sampled
  mode, which is the mode the agent uses, and it is on by default.
- **The new DFlash draft-context code in `Runtime.mm`** is inert for Flash-Next.
  `loadPackage` only loads draft weights when the target is `isQwen38`, and both
  `package.hasDraft()` and `geometry.hasDraft` are false for the MoE target, so
  every `hasDraft` branch, the prefill capture plan and the decode draft-commit
  batch are skipped.

## Still open

Per-step latency is back to the recorded level when the file is reverted
(75.2 vs the recorded 72.4 ms/step), but tokens per step stays below the historic
band (3.06 against 3.19–4.38). The residual ~4 ms/step inside that
file is unexplained and untested, and the suspects remain the predictor-v2 `std::sort` that runs per layer,
and the victim-eviction safeguard that refuses to admit a new expert when the
incumbent's frequency is higher and the candidate has fewer than two votes.

The dominant open question is now narrowed to: **why the current build misses 75%
more experts per step (128.1 against 73.2), which is the +16.4 ms/step.** No
switch measured here touches it — the horizon advisory, the only identified
per-step cost, is worth nothing at the default cache. At 34 GiB it explains 16.9
of the 22.6 ms/step, leaving ~6 that behaves as if it were cache-independent, so
the default-cache figure is not advisory plus a constant. The victim-eviction guard was tested and is only ~15 of the 55 extra misses per
step (~2 ms/step of 16.4); it is real but not the cause. The prime suspect for
the remaining ~40 misses is the admission order — how candidates are ranked before
being offered to the cache — since the predictor-v2 change touches exactly that,
and the predictor-v2 `std::sort` is the only other in-file candidate.

Draft yield is **not** the open axis this review once said it was. Against the
rebuilt published build it went up, 2.87 → 3.06 tok/step. The journal's 51.9 tok/s
/ 4.38 tok/step figure still cannot be attributed to any revision, and until it
can, decompositions that lean on it should be discarded.

Inside `Qwen4ExpTarget.cpp` the predictor-v2 `std::sort` that runs per layer is the
only untested in-file suspect left; the victim-eviction guard is measured and
accounts for ~2 of the 4.4 ms/step. Reverting the file to its 10-04 state reaches
75.2 ms/step against the baseline build's 70.8, so about 4.4 ms/step of the file's
delta is still unexplained — a small slice of the whole.

The strongest lead, and the one that has not been pursued: **the published build
is insensitive to the expert cache budget (70.7 against 70.8 ms/step) and the
current build is not (87.0 against 93.4).** That points at expert residency rather
than at the advisory itself.

## 27B Splash-HQ

| mode | tok/s | tok/step | ms/step |
|---|---|---|---|
| greedy | 31.2, 28.7 | 3.61 | 115.7, 125.5 |
| T=0.2 (the agent's setting) | 33.3 | 3.85 | 115.5 |

Per prompt, greedy: palindrome 39.1 tok/s (4.57 tok/step), ducks 34.6 (4.00),
lru 30.8 (3.56), explain 24.1 (2.78).

Speculation is engaging in both cohorts, so the "speculation is not working"
hypothesis does not hold up: this is ~2.6–2.9 accepted drafts beyond the anchor
per step, and tokens per step reproduced exactly (3.61 in both greedy runs). It
is nonetheless slower than the claim of 46.8 tok/s with 45.7% acceptance, which
at a similar yield implies ~78 ms/step against the 115.7–125.5 measured. The
same-suite comparison in *Published vs measured* is the firm version of this:
30.8 against a published 36.9, and the two domains that fall hardest are Math &
Logic (−28%) and Constraint Reasoning (−24%) — the structured-derivation text
where the drafter has to guess most.

What was *not* reproduced is the reported "very slow" speed; nothing offline went
below 24 tok/s. A slow run that this harness cannot see would have to come from
long context (per-step latency grows with KV size; these prompts were 122–173
tokens), constrained tool-calling decode, or a lost prefix cache — the cold first
prefill took 6,102 ms against 468 ms warm, so a prefix-cache miss would read as
"very slow" while decode is fine.

**Fragility worth knowing:** this model's draft weights are symlinks out of the
model directory into
`.../model-serving/splash2/install/models/incoai/Qwen3.8-27B-Splash-HQ/`. They
are intact today (328,794,112 bytes plus five 187,449,344-byte layers). If that
`splash2` tree is ever moved or deleted, `is_regular_file` fails, the drafter is
not loaded, and decode collapses to ~11 tok/s with no error message.

## Recommended changes, in order

1. **Implemented as a switch; whether it pays depends on the cache budget.**
   The horizon advisory now has its own condition, `SPLASH_NO_HORIZON_ADVISORY`
   (`Qwen4ExpTarget.cpp:3549-3558`), which leaves `prefetchPredicted(layer + 1)`
   and the prefill advisories the README credits with a 28% win untouched, and
   preserves today's default. Turning it off recovers the measured 16.9 ms/step
   (+22% tok/s, 32.7 → 40.0) when the expert cache is pinned to 34 GiB — the
   setting the published harness uses, so it is relevant to that comparison — and
   nothing measurable at the default size. Settle the cache budget first: at the
   default the same build is 8.6 ms/step faster than at 34 GiB with no code
   change at all.
2. Default `SPLASH_EAGLE2_DRAFT=0` until it is measured as a win on this
   MoE-plus-SSD-streaming path; it costs 7% sampled today and was landed without
   a decode measurement.
3. Measure acceptance with the instruments that already exist rather than adding
   capture: `/status` emits `draft_acceptance_rate` and `accepted_draft_tokens`
   (`runtime/engine/Status.cpp:276`), `dev/benchmarks/benchmark_drafting.py`
   reports the pre/post delta of `splash_drafted_tokens_total` against
   `splash_accepted_draft_tokens_total`, and `SPLASH_TRACE` plus
   `models/qwen4exp/bench/trace_report.py` gives acceptance by depth and why
   guessing stopped. The recorded figure to compare against is 68.7% cumulative
   on Flash-Next (`.agents/decisions.md`). Speculation is engaging but the yield
   is low on structured-derivation text — the 27B loses most on Math & Logic and
   Constraint Reasoning — so the next question is acceptance, not raw speed.
4. Keep the two published suites runnable from the repository — the 6-task
   Flash-Next prompts and the 5-domain 27B suite — so the next regression is
   measured against a like-for-like baseline instead of a cross-suite average.

## Limitations

- Timing noise is about ±8% run to run on this hardware. The pivotal advisory
  comparison was repeated and holds in both pairs; the remaining single-sample
  numbers (the file revert, the threshold pair, the expert-frequency revert)
  should be read with that band in mind. The expert-frequency result (102.4
  ms/step) and the file revert (75.2) sit outside it; the threshold result does
  not.
- The advisory's cost is conditional on the expert cache, and every measurement
  of it above (5.8 ms/step for one pair, 6.6–8.8 across paired repeats, 16.9 in
  the interleaved run) was taken with `SPLASH_EXPERT_CACHE_GIB` pinned. Run
  without the variable — the shipped default — the effect disappears. So neither
  "16.9 ms/step" nor "~7 ms/step" is a property of the build; both are
  properties of the build under a cache budget. The cache setting behind the
  recorded 51.9 tok/s run is not documented, which is the one thing that keeps
  the decomposition above from being assignable to a single budget with certainty.
- Sample counts are small: three reps for the pinned-cache arms, two for the
  default-cache arms. Within that, the pinned difference (16.9 ms/step) is far
  outside the spread while the default-cache differences are inside it, and the
  arm ordering reverses between reps when there is no real effect.
- The "few days back" reference points (43 tok/s / 3.19 tok/step from 10-08, and
  51.9 / 4.38 earlier in the same suite) come from the project journal, not from a
  pristine checkout re-measured here. The within-run comparisons above — 34.6 vs
  38.4 vs 38.2 vs 29.9 — are the hard evidence. For the baseline itself that gap
  is now closed — `origin/main` was rebuilt and measured — but the 51.9 / 4.38
  record still cannot be attributed to any revision, and the decomposition above
  deliberately does not use it.
- Both published columns were produced by earlier builds of this engine (the
  27B figures by the `npanj/splash` q8 fork), so the comparison is
  current-build versus published-build, not two builds of identical code. The
  two engines compute `stream_tokens_per_second` in the same spirit — decode
  only, prefill excluded — but their measurement windows could differ in detail.
- The published 27B suite tokenises ~50 tokens shorter per prompt than the
  current server does, a chat-template difference; user prompts and generation
  length are identical.
- The 27B live path was not measured: no server acceptance telemetry, and no
  long-context or constrained-decoding run.
- The lower tokens per step (3.06 against 4.38 on this suite) is not attributed
  to a code change. The Flash-Next package has been re-prepared since that
  measurement, so a weights difference could account for some of it.
- Nothing here is fixed. This is diagnosis only.
