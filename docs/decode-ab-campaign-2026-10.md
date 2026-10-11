# Decode A/B campaign, Flash-Next + 27B — 2026-10-10/11

**Bottom line.** Two weeks of changes left decode flat because two new
features each cost more than everything else saved. Both are now off by
default (committed). Of 14+ follow-up experiments, exactly one won: 7 MTP
drafts instead of 6 (+1%, committed). Everything else was neutral or worse.
Current speed: ~38 tok/s default cache, ~42 at 34 GiB. Reaching the 55 tok/s
target needs a trained guesser (weeks, cloud GPUs) — no remaining patch-level
option gets there.

Companion doc: `decode-speed-regression-review.md` (the regression itself).
Runners: `dev/benchmarks/decode_ab/`. Prompt set: `build/qwen4exp-speed-prompts.json`.

## Words we can't avoid

- **MTP drafts**: the model's built-in guesser. Each decode step it proposes
  extra tokens; the big model checks them in one pass. Accepted guesses =
  free tokens.
- **Yield (tok/step)**: average tokens accepted per decode step. Higher yield
  = fewer steps = faster. This is the number every experiment moves.
- **Greedy**: always pick the most likely token. Deterministic, so two runs
  must produce identical tokens — our quality check.

## How every run was measured (so future runs compare fairly)

1. Same-day controls: speed drifts ~1 tok/s day to day. Every claim pairs a
   control arm run the same day.
2. Token-identity check: greedy runs must emit byte-identical tokens
   (e.g. 7-draft arm reproduced all 2498 control tokens). If tokens differ,
   the switch changed quality, not just timing.
3. Machine guard: refuse the run below 28 GiB free (Flash) / 18 GiB (27B);
   kill below a 10 GiB floor; watch swap (steady ~230–310 MB is fine).
   Chrome closed. One engine at a time.
4. `SPLASH_STEP_TIMING` milliseconds are untrustworthy; miss counts and
   tok/s are trustworthy.

## What shipped (7 commits, all separate)

- Host trims: gate decode stats/clocks on `collectTiming`, trim per-step
  allocations (`a78976d`).
- **7 MTP drafts by default** (`aed657a`): fills the 8-row verify window
  (1 anchor + 7). Yield 3.06 → 3.21 over 4 runs, identical 2498 tokens, +1%
  tok/s. Override with `SPLASH_MTP_DRAFTS`.
- Horizon advisory defaults **off** (was +17–24 ms/step of wasted SSD
  readahead under contention).
- EAGLE-2 tree drafting defaults **off** (inert under greedy).
- Eviction guard stays **on** (turning it off: −0.8%, +15 misses).
- `countExpertUse` deliberately NOT gated (feeds eviction + prefetch).
- Agent fixes: per-turn loop-detector reset, resolved plan-step display,
  line-buffered streaming (`3b0c283`).
- Tree-draft warning for 27B (`b8d02f7`).
- PLD matcher fix + sampler `step_accepted[]` telemetry
  (`e724eab`, `9063f73`) — built and measured, NOT shipped (below).

## Experiments that lost (do not retry without a new idea)

- **No-MTP ablation: 19.3 tok/s.** Proves MTP earns its ~2x; the guesser is
  not overhead.
- **PLD (prompt-lookup drafting).** Root cause found and fixed: the backward
  extension was misaligned with the query end, so it hit 0/154 matches
  (proven by offline replay + unit tests `testPromptLookupSingleAnchorVsTail`,
  `testPromptLookupDeepChain`, `testPromptLookupPendingAnchorExtends`).
  After the fix it still verifies worse than MTP (3.0–3.2 vs 3.32 tok/step).
  Stays opt-in: `SPLASH_PROMPT_LOOKUP=1`, master kill `SPLASH_NO_PLD=1`,
  knobs `SPLASH_PLD_SCAN_DEPTH`, `SPLASH_PLD_MIN_MATCH`, `SPLASH_ADAPTIVE_MODE`.
- **N-gram table idea: closed.** The 51 GB table is the PLE weight table
  (~30 GB), fully wired and CPU-gathered by design — not a guess source.
- **Attention vs length: closed.** 8K context costs ~85 ms/step vs ~80 short;
  cost tracks verify rows (yield), not length. Untested past 8K.
- **34 GiB cache re-run: SUSPENDED until Nitin asks explicitly.** Earlier
  parity run hit 41.7; re-runs kept tripping the memory guard. 36 GiB
  freezes the box.

## 27B findings

- Control ~36 tok/s at 4.10 tok/step. `SPLASH_TREE_DRAFT=1` costs −17%
  (29.9 tok/s) — documented, stays off.
- Accept rate by genre: code 5.40, math 3.41, prose 1.89. Zeros cluster at
  novel turns, arithmetic, and `</think>` transitions — a training-scale fix,
  not a patch.
- Shared-prompt comparison (2026-10-11, same 3 texts, per-model
  tokenization, single runs, calm machine, 24 GiB cache):

| prompt | Flash-Next accept/step | Flash tok/s | 27B accept/step | 27B tok/s |
|---|---|---|---|---|
| code echo | 2.05 | 36.9 | 2.49 | 30.4 |
| math | 3.51 | 37.8 | 3.97 | 43.4 |
| prose | 2.20 | 40.0 | 1.74 | 24.0 |

  27B's guesser wins math decisively (3.97, 43.4 tok/s) and loses prose
  (1.74). Neither guesser dominates everywhere; per-genre drafting policy is
  open future work. Raw token dumps: `/tmp/prof/S2flash.toks.json`,
  `/tmp/prof/S2qwen27.toks.json` (scratch — re-run via
  `dev/benchmarks/decode_ab/run_shared.sh`).

## Path to 55 tok/s average (peaks higher)

- **Option 1 — train a 5-layer DFlash guesser.** The `draft/` folder has a
  manifest with `"placeholder": true`: no trained weights exist. Needs
  training data (prompts + target states + next tokens), cloud GPUs, weeks.
  Only option plausibly reaching 55 average. Decision is Nitin's.
- **Option 2 — merge PLD guesses with MTP guesses in the 8-row verify
  budget.** Plain version: today the 8 rows hold 1 anchor + 7 MTP guesses.
  Option 2 spends some rows on prompt-lookup guesses (text copied from the
  prompt — helps code tasks most, where answers repeat the prompt) and the
  rest on MTP, keeping whichever verifies. It is experimental-gated so the
  operator can switch it off. Honest caveat: fixed PLD alone verifies worse
  than MTP (3.0–3.2 vs 3.32), so the merge must prefer MTP on short matches —
  that preference rule is the remaining work, estimated in days.
- Headroom from 7 drafts + 34 GiB cache reaches ~42–45, not 55.

## Open / queued (machine-gated, not blocked on ideas)

- Attention behavior past 8K context: untested.
- PLD↔MTP merge preference rule (the rest of Option 2).
- Per-genre drafting policy from the shared-prompt data (needs repeats;
  today's numbers are single runs).
- Full `test_server.py` + unsandboxed `test_agent.py` reruns before push
  (2 pre-existing tty failures on base, unrelated).
