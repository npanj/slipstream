# Native agent vs pi — benchmark 2026-10-11

**Bottom line.** The native agent earns its keep: 6/6 tasks solved on both
harnesses, native 1.2–2.9x faster wall-clock, 3x lighter process (40 vs
121 MB). The speed comes from denser tool use per model call, not from
faster decoding. One real overhead found and removed (dead /tokenize
POST per file tool, committed `382f201`). One prompt idea tested and
reverted (no batching effect, 0/3).

## Setup (fair fight)

- Same server, same model: Flash-Next V3 on :8090, `CTX=65536`,
  `--thinking medium` both sides, default modes (hybrid vs pi defaults).
- Same tasks, fresh dirs, alternating order (N,P,P,N,N,P).
- 3 tasks with scripted graders (`check.sh` verified both directions):
  T1 fix off-by-one + pytest, T2 add `--json` flag, T3 rename symbol +
  pytest.
- Server notes: default CTX refused at load (KV budget fits ~98K with
  32 GiB cache — use `CTX=65536`); needs `sudo sysctl
  iogpu.wired_limit_mb=59392` after reboot (no terminal for it this
  session, ran with the limit as-is).

## Scoreboard (official: first runs, interleaved)

| task | native wall | native calls | pi wall | pi turns/calls |
|---|---|---|---|---|
| T1 fix | 25 s | 5 tools | 30 s | 4 turns, 3 tools |
| T2 flag | 23 s | 3 tools | 59 s | 6 turns, 6 tools |
| T3 rename | 23 s | 8 tools | 67 s | 7 turns, 10 tools |

All PASS. Native prompt tokens 7.9–13.1K at 71–92% prefix-cache hits.
pi input 4.0–5.5K + output up to 1.6K per task.

## Resources (T1/T2 arms, `ps` sampling incl. children)

- Native: peak **40 MB** RSS (python + Rich + warm PTY zsh).
- pi: peak **~121 MB** RSS (node runtime + per-call bash spawns).
- CPU: ~0% both at 2 s sampling — both harnesses wait on the network;
  neither computes.
- GPU: neither touches it. All GPU work is server-side (~50 GB wired,
  identical for both). Harness delta (~80 MB) is 0.16% of the server.

## Why native wins (evidence, not slogans)

1. **Denser model calls.** Similar round-trip counts (5v4, 4v6, 6v7) but
   native packs view→edit→verify with no exploratory detours; pi spent
   turns on `ls`/`cat` before reading, and re-verified more.
2. **Sub-millisecond local tools.** view/edit/search run 0.2–3 ms
   in-process (unix-socket IPC confirmed live:
   `/tmp/slipstream-8090.sock`); pytest in the warm PTY runs in
   2–190 ms with zero spawn cost.
3. **Prefix cache compounds.** 71–92% hit rates held across each
   single-task session.

## Caveats (read before quoting numbers)

- Single runs per task per harness (repeats: native T1 26 s, pi T1 55/53 s,
  pi T2 62/54 s — pi varies more).
- Server warmth matters: 3 identical native-T3 repeats went 24 → 11 →
  10 s with identical tool sequences. Interleaving balances this, but
  ±2x run-to-run swings exist.
- Tasks are small (3–10 tool calls). Long-horizon behavior (compaction,
  70% threshold) untested.

## Opportunities

- [x] Removed dead `/tokenize` POST per view/edit (nothing consumed the
  tokens; `ingestReferenceTokens` has no production caller).
- [ ] Tested batching guideline ("view all files at once") — 0/3 runs
  batched, reverted. The executor supports parallel blocks; the model
  doesn't use them. Open: structural nudge (e.g. multi-view tool) rather
  than prompt words.
- [ ] `/stats` shows prompt/cached but no completion totals — add them so
  the next benchmark can split generation length vs latency.
- [ ] pi's higher variance + longer outputs suggest its system prompt
  elicits more reasoning tokens; a token-normalized rematch is the honest
  next round, plus a long-horizon task for the compactor.
