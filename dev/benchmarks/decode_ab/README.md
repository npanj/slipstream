# Decode A/B runners (2026-10-10/11 campaign)

One benchmark arm per script: preflight the machine, run `generate-sample`
under `guarded.py`, print tok/s + yield, dump tokens for identity checks.

- `run_arm.sh <prefix>` — Flash-Next, 6-task suite
  (`build/qwen4exp-speed-prompts.json`), default cache. Needs 28 GiB free.
- `run_arm27b.sh <prefix>` — 27B, 4 prompts, 128 tokens. Needs 18 GiB free.
- `run_pld_probe.sh <prefix>` — Flash-Next repetition probe (code echo +
  extend), 24 GiB cache. Needs 28 GiB free.
- `run_shared.sh <model-dir> <prefix>` — shared 3-text guesser comparison
  (code echo, math, prose), 24 GiB cache, per-step `step_accepted[]`
  histogram. Needs 28 GiB free.

All write `$OUT/<prefix>.raw.json|.raw.log|.toks.json` with `OUT=/tmp/prof`
(create it first). All refuse when another engine runs or memory is short.

Full method, results, and env-knob reference:
`docs/decode-ab-campaign-2026-10.md`.
