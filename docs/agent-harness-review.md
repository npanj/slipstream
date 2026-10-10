# Slipstream Agent Harness Review — Functionality, Performance, and UX

**Date:** 2026-10-09 · **Scope:** the coding-agent harness in `server/agent/`
(`agent.py`, `tools.py`, `pty_shell.py`, `repo_indexer.py`, ~2,946 lines) and its
tests. **Method:** full source read; a live end-to-end turn against a running
engine on 8090; and the harness unit suite run with the exit status preserved.
All claims carry `file:line` evidence. No harness source was modified.

**Relationship to existing docs:** this is a *review of what was built*, not a
design proposal. The design intent lives in
[`agent-intelligence-and-harness-survey.md`](agent-intelligence-and-harness-survey.md),
[`fast-agent-harness-design.md`](fast-agent-harness-design.md), and
[`harness-and-inference-optimizations.md`](harness-and-inference-optimizations.md).
The engine-level backlog is in [`architecture-roadmap.md`](architecture-roadmap.md);
the research/search-provider follow-up to F2 and F6 is in
[`websearch-retrieval-sources.md`](websearch-retrieval-sources.md);
actionable items distilled from this review are tracked in
[`.agents/tasks.md`](../.agents/tasks.md).

## 0. Verdict

**The harness works.** A live tool turn drove the real engine, called `view_file`
once, and returned the correct answer. The defects below are mostly *edges of the
harness itself* (timeouts, an unreachable tool, a no-op "ingestion" claim, test
hermeticity, packaging) rather than a broken core loop. Ordered by user impact.

## 1. Live verification

- Engine already serving on 8090 (`local/swift-qwen38-flash-next-v3`).
- Non-stream completion: `content='OK'`, 0.66 s.
- Streaming: **TTFT 0.326 s**, 15 content chunks.
- Full `step()` tool turn: `view_file:{"path":"/tmp/agent_live_probe.txt"}` fired
  once, correct "3 lines" answer, 12.59 s wall.
- Probe patched `save_checkpoint` to a no-op so the user's vault was untouched;
  `PY_EXIT=0`.

The interactive-tty TUI path could not be verified headlessly (see F5).

## 2. Findings (ranked)

### F1 — `run_command` hard-caps every command at 30 s *(High)*
`server/agent/pty_shell.py:79` — `run_command(self, cmd, timeout: float = 30.0)`.
Every call site takes the default: the tool path at `agent.py:612`, and the `/test`
command at `agent.py:1471`, which runs the project's own `make test-engine-cpu`.
The tool schema exposes **only** `command` (`tools.py:304–318`) — no `timeout`
parameter. Long builds and test runs are truncated and return `rc=-1`. This
directly breaks the harness's own guideline *"Test your changes using run_command
before concluding"* (`agent.py:53`).

### F2 — `fetch_url` is implemented but unreachable *(High)*
`execute_tool` handles it (`agent.py:649–652`) and `SYSTEM_PROMPT` advertises it
(`agent.py:45`), but it is **absent from `TOOL_DEFINITIONS`** (`tools.py:304–440`,
7 tools: `run_command`, `view_file`, `replace_file_content`, `write_file`,
`search`, `web_search`, `manage_todo`). The model therefore never emits it, so
links returned by `web_search` can never be opened.

### F3 — `_ingest_reference_content` is a no-op *(High — false capability)*
`agent.py:548–588` POSTs `/tokenize`, counts the returned tokens into
`stats["ingested_reference_tokens"]`, and returns. It never feeds any engine
index: grepping `server/*.py` for ingest / reference / prompt_lookup wiring finds
nothing, and `PromptLookup` exposes no session-ingest call this touches. The
advertised speculative (PLD) acceptance boost is **not happening**.

### F4 — `/undo` runs `git checkout .` with no confirmation *(High — destructive)*
`agent.py:1463`: `rc, out = agent.pty.run_command("git checkout .")`, reachable
from the `/undo` slash command (`agent.py:1288`) with no confirmation prompt. In
this very tree that would discard thousands of uncommitted lines.

### F5 — Tests are non-hermetic *(Medium)*
Running the harness suite yields **50 tests, 2 failures + 2 errors** (`PY_EXIT=1`):
- `_resolve_model` (`agent.py:172`) and `_ingest_reference_content`
  (`agent.py:548`) prefer the live UNIX socket when one exists, bypassing the
  `urllib` mocks. Result: `test_agent_preemptive_reference_token_ingestion`
  (`AssertionError: 43 != 4`) and `test_agent_resolve_model_fallback_and_discovery`
  (got `local/swift-qwen38-flash-next-v3`, expected `local/discovered-27b`).
- `_run_pinned_tui` calls `termios.tcgetattr(sys.stdin)` at `agent.py:1784` with
  no tty guard *inside* the function (the `isatty` check lives in
  `run_agent_cli`), so both TUI tests raise
  `termios.error: (19, 'Operation not supported by device')`.
The failures are environment-caused, which means the suite cannot currently
certify the harness in CI.

### F6 — Packaging drift *(Medium)*
`duckduckgo_search` (`tools.py:164`), `trafilatura` (`tools.py:224`), and
`rapidfuzz` (`tools.py:94`) are imported at runtime but are **not** listed in
`install/requirements.txt` (41 lines, none of the three). They happen to be
installed in this `.venv`, so they are present but unpinned — a fresh install
would lose `web_search` / markdown extraction / fuzzy edit healing.

### F7 — Dead / incorrect code *(Low)*
- `FileEditor.verify_chunk` (`tools.py:30`) is used only by its own test
  (`dev/tests/engine/test_agent.py:220`), never by the edit path.
- `RepoMapper._extract_symbols` precedence bug (`tools.py:270`):
  `if (A or B) and "{" in stripped or ";" in stripped:` parses as
  `((A or B) and "{" in stripped) or (";" in stripped)`, so **any** C++ line
  containing a `;` is captured as a "symbol".
- `_post_stream` contains a duplicated `except json.JSONDecodeError`.

### F8 — `replace_file_content` is non-atomic *(Low)*
No temp-file-and-rename, no `.bak`, and the normalization path can rewrite
whole-file line endings — a failed or fuzzy edit can leave a file in an
unintended state.

## 3. Recommendations

Each is labeled by **regression risk** against the "no quality or speed
regression" constraint. "Pure win" = cannot change model output.

| # | Fix | Fixes | Regression risk |
|---|---|---|---|
| S1 | Add a `timeout` parameter to the `run_command` schema and forward it; raise the PTY default; emit a heartbeat so long commands are not read as a hang | F1 | **Pure win** (new optional arg; default behavior unchanged) |
| S2 | Register `fetch_url` in `TOOL_DEFINITIONS` | F2 | Changes the request payload sent to the engine → **A/B required** on the 145-item scorecard |
| S3 | Implement a real session-ingest endpoint and wire it to `PromptLookup::ingestReferenceTokens`; or remove the claim | F3 | **Cannot regress quality** — speculative tokens are always verified by the target. Speed upside only |
| S4 | Add a confirmation prompt to `/undo` (and `/clear`); refuse when the tree is dirty without an explicit `--force` | F4 | **Pure win** |
| S5 | Guard `_run_pinned_tui` on `isatty`; force tests off the live socket (inject a mock URL or disable socket discovery under an env flag) | F5 | **Pure win** |
| S6 | Pin `duckduckgo_search`, `trafilatura`, `rapidfuzz` in `install/requirements.txt` | F6 | **Pure win** |
| S7 | Fix the `_extract_symbols` precedence bug; either expose `verify_chunk` on the edit path or delete it and its test; drop the duplicate `except` | F7 | **Pure win** |
| S8 | Write `replace_file_content` atomically (temp + rename) with an optional `.bak` | F8 | **Pure win** |

### Performance (no output change)
- Long-command timeout override (S1) removes the most common wasted turn.
- Use real `/tokenize` counts for the 70 % compaction threshold — currently
  `int(total_chars / 3.8)` (`agent.py:227`), an approximation.
- Hoist the per-turn `ThreadPoolExecutor` (`agent.py:774`, and `:1070`) and build
  the repo map lazily instead of on every turn.

### UX (must be A/B'd before shipping)
- `reasoning_effort` defaults to `medium`, which makes every tool turn slow.
  **Lowering per-turn effort changes model output**, so it must be validated on
  the 145-item scorecard before becoming a default — do not ship it as a "free"
  speedup.
- Detect and refuse a stale UNIX socket rather than silently preferring it
  (this is also the root cause quantified in F5).

## 4. Harness facts worth recording

- `stream_mode` defaults to `hybrid`: non-streaming whenever tools are present.
- `SlipstreamAgent.timeout` defaults to `None` (unbounded generation).
- Vault at `~/.cache/slipstream/agent_vaults/`; checkpoints auto-save on turn
  end / interrupt / error.
- 12 slash commands; `SYSTEM_PROMPT` mandates a `manage_todo` plan and
  rationale-before-tools.
- Consecutive-identical-tool circuit breaker trips at 3×.
- Speculative read-tool dispatch during streaming covers
  `{view_file, search, web_search, fetch_url}` (`agent.py:934`) — i.e. it already
  anticipates `fetch_url` reaching the model, which F2 prevents.

## 5. Status

| # | Finding | Severity | Status |
|---|---|---|---|
| F1 | `run_command` 30 s cap | High | ☐ Open |
| F2 | `fetch_url` not registered | High | ☐ Open |
| F3 | `_ingest_reference_content` no-op | High | ☐ Open |
| F4 | `/undo` destructive, unconfirmed | High | ☐ Open |
| F5 | Non-hermetic tests / missing tty guard | Medium | ☐ Open |
| F6 | Runtime imports not pinned | Medium | ☐ Open |
| F7 | Dead / incorrect code | Low | ☐ Open |
| F8 | Non-atomic file replace | Low | ☐ Open |

**Constraint (non-negotiable):** S2 and the `reasoning_effort` default both change
what is sent to or sampled from the model and therefore require an A/B on the
145-item scorecard before shipping. Everything else in §3 is behavior-preserving
and cannot trade quality or speed for the fix.
