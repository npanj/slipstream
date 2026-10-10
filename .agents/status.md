# Status — Slipstream (handoff)

**Updated:** 2026-10-10 08:30 PDT by muse-code (decode regression fixed: horizon advisory + EAGLE-2 defaulted off; Flash-Next back to published 41.7 tok/s).
**Branch:** `main` (canonical workspace `model-serving/slipstream`, release `v26.10.4`, app `npanj/slipstream-menubar` `v26.10.7`, tap `npanj/homebrew-tap`).

## In one line

Fixed the Flash-Next decode regression (32.4 → 41.7 tok/s on the published 09-26 prompts, exact parity): the L+2 horizon advisory now defaults off and EAGLE-2 tree drafting defaults off, both as opt-ins pinned by a new regression test. Greedy outputs are bit-identical pre/post fix. 27B has no in-window regression (10-04 build can't draft on today's package at all: 11.5 vs current 30.4 tok/s). Working tree holds 7 uncommitted files; nothing pushed.

## Regression fix (2026-10-10) — current state

- **Files changed (uncommitted):** `models/qwen4exp/Qwen4Exp.{hpp,cpp}`, `Qwen4ExpTarget.cpp` (3 switch call-sites, horizon top-by-frequency ranking, spin-wait yield), `server/server.py` (unix_server init-order fix), `dev/tests/engine/qwen4exp_layout_test.cpp` (+`decodeFeatureSwitchDefaults`), `dev/tests/test_server.py` (+bind-failure test), `docs/decode-speed-regression-review.md` (Resolution section + 2 corrected claims).
- **Verification observed:** `make` clean; `make test-engine-cpu` green; `qwen4exp-layout` PASS; same-prompt 41.7/41.7 @34 GiB, 38.5 @default, 36.6 sampled, 27B 31.6 unchanged; `test_server.py` full suite + `test_agent.py` environment-blocked in this session (socket bind / pty denied), server/agent untouched except the init fix.
- **Do not run:** 36 GiB expert cache on the 64 GB box (near-freeze observed); `test_server.py`/`test_agent.py` need an unsandboxed terminal.

---

## Executed Qwen Optimization Roadmap (P0–P3)

| Priority | Track | Implementation | Verification |
|---|---|---|---|
| **P0** | **Python Boundary & SSE Optimization** | • Cached `_disconnect_selector` on `FrontendHandler`, eliminating per-token `DefaultSelector()` allocations and kqueue descriptors.<br>• Fast-path string formatting for text deltas in `put_text()`, bypassing dictionary serialization.<br>• Instrumented `sse_decode_ms` vs `engine_decode_ms` (`decode_gap_ms`) in `/metrics` and `/timings`. | `test_server.py` (172/172 PASS in 128s) |
| **P1** | **Agent-Loop & Telemetry Audit** | • Added cumulative prefix cache hit-rate telemetry (`total_prompt_tokens`, `cached_prompt_tokens`) in `SlipstreamAgent` and displayed in `/stats`.<br>• Incremental tool-parse cursor tracking in streaming speculative execution to prevent redundant `json.loads` calls.<br>• Tool output cap set to 30,000 characters across `cap_tool_output` and `compress_tool_output`. | `test_agent.py` & `test_launcher.py` (59/59 PASS in 4.9s) |
| **P2** | **Kernel Verification Pass** | • Sampler readback: confirmed `Runtime.mm:1657-1691` reads exactly `laneResult.retained` tokens (1 token steady state, up to $K$ accepted drafts), never full logits.<br>• Split-KV + GQA packing: confirmed in `attention_q8.metal` (`QueryHeadsPerKVHead`, threadgroup partitioning across KV heads and history splits).<br>• Radix top-k: confirmed 4-pass 8-bit radix histogram in `qsa_select.metal` with 256 atomic bins and parallel chunk-wise tie scans. | Code & kernel audit confirmed; bit-exact outputs |
| **P3** | **Metal PSO Pointer Caching & Makefile Audit** | • Added `-DNDEBUG` to `ENGINE_CXXFLAGS` in `Makefile`.<br>• Added `pipelineMap` (`std::unordered_map`) and `cachedPipeline` pointer in `MetalBackend.{hpp,mm}`, bypassing `NSString` allocations and Obj-C dictionary lookups in hot decode loops. | `make test-engine-cpu` (24/24 PASS) |

---

## Agent Timeout & Generation Wait Reference

1. **Unbounded Generation by Default**:
   - `SlipstreamAgent` defaults to `timeout = None`, executing pure blocking socket/HTTP I/O (`sock.settimeout(None)`).
   - The model can take as long as it needs on multi-thousand token reasoning chains (`thinking: high/max`) or long prompt prefills (>32k tokens) without timing out.
   - Steer, interrupt, or cancel anytime with `Ctrl+C` or `Enter`.
2. **CLI Flags**:
   - `--no-timeout`: Explicitly forces unbounded generation time (default).
   - `--timeout <seconds>`: Optional watchdog timer (pass e.g. `--timeout 300` or `0` for none).
   - `SLIPSTREAM_AGENT_TIMEOUT`: Environment variable control (`0`, `none`, or seconds).

---

## Agent Session Continuity & Vault Reference

### 1. Automatic Checkpointing
- Every turn completion, manual interruption (`Ctrl+C`), and generation error automatically saves an atomic JSON snapshot to `~/.cache/slipstream/agent_vaults/<session_id>.json` and updates `latest.json`.
- State captured includes full conversation history, active execution checklist (`todos`), model configuration, token consumption telemetry, working directory, and the last prompt.

### 2. Resuming Previous Work
- **CLI Commands**:
  - `./slipstream code -c` or `./slipstream code --continue`: Restores and resumes the most recent session (`latest.json`).
  - `./slipstream code --resume [id]`: Restores a specific session ID (or short prefix).
  - `./slipstream code --sessions`: Displays a formatted table of recent vault sessions and exits.
- **In-Session Slash Commands**:
  - `/continue` (or plain `continue` / `c`): Continues interrupted work or executes the next pending plan task.
  - `/retry`: Re-dispatches the last failed or timed-out user prompt.
  - `/resume [id]`: In-place restore of another session without restarting the agent process.
  - `/sessions`: Displays saved sessions table in-session.
  - `/save`: Explicitly snapshots the active session to disk.

---

## Architecture: Engine IPC vs Agent Sockets

1. **Tier 1 — C++ Native Engine ↔ Server (`runtime/engine/FdTransport.cpp` ↔ `server/runtime.py`)**:
   - **Pure binary IPC** using Unix pipes/file descriptors and shared-memory doorbells.
   - Zero TCP, zero HTTP, zero socket overhead. Uses `CompletionWake` with atomic `wakePending` flag coalescing and adaptive `yield` spin handoff directly to Apple Silicon Metal GPU scheduler.
2. **Tier 2 — Agent Harness ↔ Server (`server/agent/agent.py` ↔ `server/server.py`)**:
   - Decoupled process allowing the ~30–40 GB model to remain hot in unified RAM while the agent CLI exits and re-enters.
   - Uses local UNIX domain sockets (`/tmp/slipstream-8090.sock`) to bypass the TCP loopback stack, with `settimeout(None)` for unconstrained local generation.

---

## Agent Bottom Panel & Status Display

The interactive CLI harness (`./slipstream code`) displays full contextual awareness at all times in the bottom panel:

1. **Active Turn / Action State (`is_generating.is_set()` is True)**:
   - Badges: `[spin] ACTION` (when running tool commands/functions), `[spin] THINKING` (during reasoning/planning), or `[spin] STREAMING` (streaming answer).
   - Detail: `• ⚡ Action: <tool> <args> (<elapsed>s)`
   - Live Context: `• 📚 ctx:<cur_ctx:,>/<max_c // 1024>k (<c_pct:.1f>%)` color-coded dynamically (cyan `< 50%`, yellow `< 80%`, red `≥ 80%`).
   - Steering: `• 🛑 Press Ctrl+C or Enter to steer`

2. **Idle / Ready State (`is_generating.is_set()` is False)**:
   - Badge: `⚡ READY`
   - Config: `• 💭 think:<effort> • 🌊 stream:<mode>`
   - Live Context: `• 📚 ctx:<cur_ctx:,>/<max_c // 1024>k (<c_pct:.1f>%)`
   - Turn Stats: `• ⏱ <dt>s • ⚡ <tok/s> • 📤 ~<tok> • 🎯 Turn #<N>`

3. **Inline Tool Action Panels (`out_console.print(Panel(...))`):**
   - Displays: `❯ <tool> <arg> · <dt>ms · ctx:<cur_ctx:,>/<max_c // 1024>k ✔`

---

## Agent Streaming & Suppression Modes

| Mode | CLI Flag | Slash Command | Behavior | Best For |
|---|---|---|---|---|
| **Option A: Smart Hybrid** *(Default)* | `--stream hybrid` | `/stream hybrid` | Uses `stream: false` during tool-calling exploration turns (zero JSON delta SSE churn, instant tool execution); streams live text during direct conversational answers. | Maximum coding speed + responsive conversational feel. |
| **Option B: Complete Suppression** | `--no-stream` or `--stream never` | `/no-stream` or `/stream batch` | Suppresses streaming deltas across all turns. Assistant renders exclusively via formatted Markdown Panels upon turn completion. Zero flicker, no raw stdout collision. | Highest throughput, batch workflows, zero UI flicker. |
| **Always Streaming** | `--stream always` | `/stream always` | Streams all deltas in real-time, including thinking thoughts and tool call JSON fragments. | Debugging raw model token generation and speculative tool parsing. |

---

## How to Launch for Personal Testing

Remember: **One model server at a time!** (Port 8090 is currently idle and free).

### Launch Daily Recommended Model (Flash-Next V3)
```zsh
./slipstream serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090
```

### Launch Interactive Agent Harness (Default: No Timeout + Smart Hybrid)
```zsh
./slipstream code --port 8090
```

### Launch Interactive Agent Harness Resuming Previous Session
```zsh
./slipstream code --port 8090 -c
```

---

## Invariants & Rules

- **One model server at a time.** Two pin more memory than the Mac has and freeze it. Run engine experiments through `dev/benchmarks/guarded.py -- <cmd>`; check `lsof -i :8090` before starting.
- **Do not launch daemon without user request.** Server on port 8090 must remain stopped until explicitly launched.
- **After any engine change:** `make`, `make test-engine-cpu`, `.venv/bin/python -m unittest dev/tests/test_server.py`.
