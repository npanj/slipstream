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

On Flash-Next the regression is almost entirely in one file, `models/qwen4exp/Qwen4ExpTarget.cpp`, and inside it the dominant
cost is the NVMe advisory work added on 2026-10-09 12:54: `adviseHorizonPredicted`
called on every pipelined stage, plus a synchronous `adviseMissedExperts` call in
the host miss-staging path. Together they cost ~9 of the ~13 ms/step that
per-step latency has grown by (`SPLASH_DISABLE_RDADVISE=1` recovers 34.6 → 38.4
tok/s with no other change).

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
Measured cost: about 8 ms/step, roughly 8% of decode throughput, reproduced on
two paired runs.

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
band (3.06 against 3.19–4.38). The residual ~4 ms/step and the lower yield are
both still unexplained. The remaining working-tree changes in
`Qwen4ExpTarget.cpp` are the suspects: the predictor-v2 `std::sort` that runs per
layer, and the victim-eviction safeguard that refuses to admit a new expert when
the incumbent's frequency is higher and the candidate has fewer than two votes.

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

1. Gate the horizon advisory and the synchronous miss-path advisory behind an
   environment variable that defaults off, or drop the horizon advisory entirely.
   Expected: back to ~38 tok/s greedy.
2. Default `SPLASH_EAGLE2_DRAFT=0` until it is measured as a win on this
   MoE-plus-SSD-streaming path; it costs 7% sampled today and was landed without
   a decode measurement.
3. Capture `draft_acceptance_rate` from `/slots` during a live run. Speculation
   is engaging but the yield is low on structured-derivation text — the 27B loses
   most on Math & Logic and Constraint Reasoning — so the next question is
   acceptance, not raw speed.
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
- The "few days back" reference points (43 tok/s / 3.19 tok/step from 10-08, and
  51.9 / 4.38 earlier in the same suite) come from the project journal, not from a
  pristine checkout re-measured here. The within-run comparisons above — 34.6 vs
  38.4 vs 38.2 vs 29.9 — are the hard evidence.
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
