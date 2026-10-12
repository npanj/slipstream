# Memory — qwen4exp port

## Splash's geometry is compile-time

Model dimensions live in C++ layout structs, not a config file. A new
architecture therefore needs engine changes, not just a converter. This is the
fact that set the whole shape of this project.

## Where things are

| | |
|---|---|
| Source checkpoint | `~/models/qwen38-flash-next-bf16` — 338 GB, 131 shards, complete |
| Tensor name prefix | `model.language_model.` (the 27B uses `language_model.model.`) |
| Converter | `models/qwen4exp/tools/convert_qwen4exp.py` |
| 27B converter (format proof) | `dev/tools/convert_qwen38.py` |
| Verification scripts | `models/qwen4exp/tools/checks/` |
| Nitin's working V3 model | `~/models/qwen38-flash-next-v3` (GGUF, llama.cpp) |
| Swift V3 GGUF model | `~/models/swift-qwen38-flash-next-v3` (GGUF + prepared Slipstream) |
| Flash-Next parameters | 125.7B neural network weights (120.8B in 512 MoE experts + 5.0B dense backbone); 7.3B active/token; +51.2B n-gram lookup table (176.9B total on disk) |

## The file format is unforgiving, which is good

16-byte header, sections on 16384-byte boundaries, and the file must be consumed
**exactly** — `WeightFile::finish()` throws on a leftover byte. A missing or
mis-sized section fails loudly at load rather than quietly at inference.

Use `--dry-run` freely: it writes headers and leaves bodies as holes, so a
96.61 GiB package costs 920 KiB and the engine will still load, validate and
plan it. This caught a manifest constant guessed at 256 MiB where the engine
wanted 128 MiB, long before any real weights existed.

## Traps that cost real time

- **A synthetic test that agrees with itself proves nothing.** I reserved a
  section for `block_inject_weight` on the final hyper-connection mixer, which
  is built `use_combine=False` and has only three tensors. Every real package
  would have failed. My test wrote the same wrong layout and passed.
- **Confirm the build succeeded before interpreting a test result.** I read a
  stale binary's output twice and re-reasoned about a fix that was already
  correct.
- **N-gram table shape**: 320,001,536 x 160 (16 heads), not 20M x 2560. Same
  element count, so the 26.82 GiB total matched and hid the error.
- **`make` does not relink `generate-sample`.** Run
  `make build/engine-tests/generate-sample` after every engine change.
- **`/tmp` is not storage.** The bf16 reference logits took ~40 min to make;
  they now live in `~/models/qwen38-flash-next-reference/`.
- **Changing a shared constant breaks tests quietly in many places.** Raising
  the prompt chunk to 4096 needed ~12 test edits where 2048/2049 were
  hardcoded; tests should use the constants.
- **Two engines at once crash the Mac** (kernel watchdog panic, forced reboot;
  twice on 2026-09-22). Each engine pins ~40 GiB that macOS cannot page out.
  Run every experiment through `dev/benchmarks/guarded.py -- <cmd>`: it
  refuses to start beside another engine and kills a run under 4 GiB free.
  Never start a new background run until the previous one has provably
  exited (`pgrep -fl generate-sample`); `pkill -f` patterns can silently miss.
  Diagnose a crash from `/Library/Logs/DiagnosticReports/panic-full-*.panic`
  (JSON; `processByPid` shows who held the memory).
- **The GPU memory limit (`iogpu.wired_limit_mb`) resets on every reboot.**
  At the default the Splash server refuses to start (short ~1 GB). The
  launchers set 59392 with sudo; agents cannot enter the password.
- **Run-to-run noise is about +-2% tok/s** (SSD timing). To compare kernels,
  repeat the kernel under test (`SPLASH_HC_REPEAT=8`) so its cost dwarfs the noise.
- **PLE gate offset**: normalized rows go at `row + (taps-1)*dilation`, because
  the convolution reads history first.
- **GGUF RMS norm offsets**: GGUF weights store `(1.0 + weight)` for hyper-connection and MTP RMS norms, but Splash Metal kernels compute `(1.0f + weight)` internally from zero-centered weights. Subtract 1.0f or gains double.
- **GGUF GDN head interleaving**: GGUF groups 48 GDN heads as `(3, 16)`. Reshape and transpose `(3, 16) -> (16, 3)` before feeding Splash kernels.
- **APFS hardlinks for n-gram tables**: Using `os.link` reuses physical SSD blocks (0 extra bytes) during conversion.
- **Swift models loop under `xhigh` thinking**: Swift 1.5 is distilled for concise thinking; forcing `xhigh` in deep contexts (>50k tokens) causes degenerate repetitive thinking loops that consume the full generation token budget and emit empty output. Always default Swift to `--thinking low` or `medium`.
- **Expert cache sizing flag is `SPLASH_EXPERT_CACHE_GIB`**: Pass `SPLASH_EXPERT_CACHE_GIB=30` (not `CACHE_GIB=30`) to control expert cache allocation. At 30 GiB, free host RAM remains >46 GiB, preventing `resource_timeout` on high-context prompts.
- **Tree drafting KV physical placement invariant**: `PagedAttention::addVerify` stores the 8 verify rows linearly into `keyData`/`valueData` during the forward pass. Taking non-linear tree branches leaves unaccepted branch activations in physical slots unless followed by a compaction pass.
- **Prompt Lookup Engine latency**: `ops::PromptLookup` delivers 49.1 ns per query on Apple Silicon using a flat chained hash table with zero heap allocations during decode.
- **Grammar-draft pruning in constrained decode**: In `Runtime.mm`, `ConstrainedDecodeTicket` prunes speculative draft rows against `entry.maskWords` after the CPU mask arrives and before `TargetForward`. If draft 0 violates JSON grammar, only 1 row is verified (~15 ms, ~10 cached experts), saving ~135 ms of SSD expert thrashing on doomed speculation chains.
- **Stale mask poisoning in constrained speculation**: Never seed MTP draft candidate selection with `entry.maskWords[laneResult.retained]`. When the verifier rejects a speculative draft, `nextAnchor` is freshly sampled and Python has never seen it; row `retained` was computed assuming the *rejected* token was accepted. Passing this stale mask forces MTP to guess along a phantom branch, collapsing draft acceptance to 0%. Keep MTP drafting unconstrained by prior masks and prune proposals post-generation in `takeMaskRequests`.
- **MTP decode batch width invariant**: MTP speculative drafting in `Runtime.mm` is single-lane only (`lanes == 1`). If `Scheduler.cpp` batches multiple decode requests into `b2`, MTP speculation is completely disabled (`buffers.mtpShadow = true`), dropping throughput from 45 tok/s to 6 tok/s. Set `SPLASH_MAX_BATCH_WIDTH=1` to serialize decode requests and preserve 45 tok/s speculation.
- **macOS Pipe Buffer & High-Context Write Deadline**: macOS pipe buffer capacity is exactly 64 KiB (65,536 bytes). Prompts >28k tokens serialize into RequestFrames >115 KB, filling the buffer on the first write. `_write_bytes` must refresh its I/O deadline on each partial chunk progress (`written > 0`); otherwise a busy native engine causes a 5.0-second timeout, raising `EngineUnhealthy` and killing the engine (503 `runtime_unavailable`, `[Qwen4Exp] resident layers: 0 / 48`).
- **Canonical Workspace Directory**: The active workspace is `model-serving/slipstream` (symlink `slipstream-v2 -> slipstream`). All previous clones (`slipstream-orig`, `slipstream-gguf`) are deleted. Public releases and GitHub repository are under `npanj/slipstream`.

## About Nitin

- Runs this on an Apple-silicon laptop; disk is a recurring constraint.
- Prefers taking the smaller/simpler path first and deferring optimisation
  (chose a smaller model before expert streaming; deferred draft training).
- Wants plain language, conclusion first. See `/Users/nitin/.agents/writing-style.md`.
- Works across Claude Code, Codex and Gemini — keep this directory current.
