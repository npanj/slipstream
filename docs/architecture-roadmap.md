# Slipstream Architecture Roadmap — Bottlenecks, Debt, and Sequencing

**Date:** 2026-10-09 · **Scope:** system architecture, bottleneck removal, and
structural debt across `runtime/`, `models/`, `server/`, `install/`.
**Method:** source audit of the decode loop, scheduler, IPC transport, MoE
streaming path, and Python frontend.
**Relationship to existing docs:** performance-only work is already specified in
[`speed-optimization-review.md`](speed-optimization-review.md) (R1–R7) and is kept
here as **Track A** so the two do not diverge. This document adds the
architecture-level tracks (W1–W10) that the speed review deliberately excluded.
All claims carry `file:line`-style evidence; internals keep the inherited
`splash::`/`SPLASH_*` names.

## 0. The one-paragraph thesis

Slipstream is a well-built single-tenant engine whose throughput is dominated by
one structural fact — the MoE model is ~3× DRAM, so it streams experts from NVMe
and hides the latency with a per-layer LRU cache plus speculation. Everything
else (batching off, prefill/decode serialized, Python on the wire, hand-written
protocol twice) is a tax paid to keep that trick simple. The roadmap attacks the
taxes in increasing order of risk, and attacks the dominant cost (misses/step and
acceptance) only with measurement gates.

## 1. Where the time and the risk actually are

| # | Bottleneck / Risk | Evidence | Class |
|---|---|---|---|
| B1 | SSD expert misses (~4.5–6.8 ms/layer; ~50 ms/step) | `models/qwen4exp/Qwen4ExpTarget.cpp` `readMissedExperts`, `.agents/decisions.md` | Dominant cost |
| B2 | Speculation acceptance is the throughput multiplier | `runtime/model/Runtime.mm` draft/verify; 69.7% FX / 45.7% 27B | Dominant cost |
| B3 | One command in flight; prefill blocks decode; width pinned to 1 | `runtime/engine/Engine.hpp` `commandInFlight`, `runtime/engine/Scheduler.cpp` `next()`, `server/server.py:25`, `install/launcher.py:367` | Structural |
| B4 | Spin waits burn a core; blocking `dispatch_apply` on the read path | `runtime/metal/MetalBackend.mm:1288,1302` (20 ms spin); `models/qwen4exp/Qwen4ExpTarget.cpp` `while (slotReads...)` | Latency/energy |
| B5 | Silent expert-cache degradation on `mlock` failure | `models/qwen4exp/Qwen4Exp.cpp` warns only on `mlock` failure | Reliability |
| B6 | Ad-hoc memory arithmetic, history of bootstrap failures | `models/qwen4exp/Qwen4Exp.cpp` `physicalGiB − 34`; `kv_pool_does_not_fit` records | Reliability |
| B7 | Wire protocol implemented twice, no schema | `server/protocol.py` (1,736) vs `runtime/engine/Protocol.cpp` (1,648) | Debt |
| B8 | Quality traded for memory by stacked approximations | Q3/Q4 KV, MiniCache SLERP, DuoAttention streaming heads | Correctness risk |
| B9 | Monolithic units and feature-flag sprawl | `models/qwen4exp/Qwen4ExpTarget.cpp` 3,685; `runtime/model/Runtime.mm` 2,850; `server/server.py` 1,937; `dev/tests/test_server.py` 7,061 | Debt |
| B10 | Benchmarks are single-machine and non-reproducible | README figures; no independent harness | Verification |
| B11 | Shared runtime assumes the Qwen hybrid family | `docs/architecture.md` "Honest limit" | Strategic ceiling |

## 2. Workstreams (W1–W10), ranked by leverage per unit risk

Each item states change, target, risk, and the check that closes it.

### W1 — Single-source the wire protocol *(B7)*
- **Change:** define frames once (a small IDL or a shared spec) and generate/verify
  both `server/protocol.py` and `runtime/engine/Protocol.cpp`. At minimum, add a
  constants+layout conformance test that fails on drift.
- **Target:** eliminate the highest risk-per-line class of defect; no runtime cost.
- **Risk:** medium (touches both sides); do it before other protocol-adjacent work.
- **Close:**  round-trip conformance test over every frame type; existing
  `dev/tests/engine/test_protocol_python.py` and
  `dev/tests/engine/protocol_test.cpp` stay green.

### W2 — Futex-equivalent synchronization *(B4)*
- **Change:** replace the 20 ms `__builtin_arm_yield()` spins in
  `MetalBackend::waitPipelineEvent`/`waitPipelineStageDone` and the bare
  `while (slotReads[slot].load()) {}` with `os_sync_wait_on_address` (or a
  completion dispatch queue), keeping the fast-path spin window.
- **Target:** stop stealing CPU from the GCD threads doing the SSD reads; lower
  power. Throughput-neutral or slightly positive.
- **Risk:** low; behavior-preserving, bit-exact.
- **Close:** `make test-engine-cpu`, `check_decode_consistency.py` bit-exact,
  speed suite within noise.

### W3 — Loud, self-healing expert-cache degradation *(B5)*
- **Change:** treat `mlock` failure as a first-class state — reduce cache capacity
  to what is actually resident and pin-able, log once at a clear severity, and
  expose miss rate / pinned GiB in `/status`.
- **Target:** convert a silent "minutes-long stall" failure mode into a bounded,
  observable one.
- **Risk:** low.
- **Close:** unit test forcing `mlock` failure path; `/status` fields present; no
  output change on the healthy path.

### W4 — Deeper asynchronous I/O pipeline *(B1, complements Track A R5/R6)*
- **Change:** a dedicated prefetch pool with explicit queue depth and
  `preadv`/`dispatch_io`; issue L+2/L+3 reads on a timer rather than piggybacking
  on the router; keep the measured "1 read per matrix" rule
  (`readMissedExperts` comment: 4-piece split was slower).
- **Target:** overlap more of the ~15 ms/step wait behind GPU work; reduce
  misses/step at unchanged prediction quality.
- **Risk:** medium (I/O scheduling interacts with QoS classes).
- **Close:** traced A/B with `SPLASH_TRACE` + `models/qwen4exp/bench/trace_report.py`;
  ±2% noise rule.

### W5 — Concurrency: batching or prefill interleave *(B3)*
- **Change:** either (a) enable width > 1 with per-lane speculation, or (b)
  interleave bounded prefill chunks between decode steps so a long prompt does not
  starve decoders. Keep single-lane speculation as the default product path.
- **Target:** reduce head-of-line blocking for concurrent requests without
  regressing single-request tok/s.
- **Risk:** high (scheduler + MTP geometry; `PagedAttention` verify requires
  `chunk_tokens == kQ8VerifyMaximumRows`). Needs a design doc first.
- **Close:** scheduler/ragged tests, `test_server.py`, A/B on concurrent-request
  latency and single-stream tok/s.

### W6 — Declarative memory budget *(B6)*
- **Change:** replace `physicalGiB − 34` and scattered reserves with an explicit
  budget model (fixed / elastic / reserved classes) validated at plan time in
  `runtime/engine/MemoryPlan.cpp` / `runtime/engine/MemoryGovernor.cpp`, with a
  single override env.
- **Target:** bootstrap cannot be broken by a new model folder or machine size.
- **Risk:** medium; must preserve the existing working-set ceiling behavior.
- **Close:** `dev/tests/engine/runtime_bootstrap_test.mm`, boot on 64 GB fixture,
  no `kv_pool_does_not_fit`.

### W7 — Complexity reduction *(B9)*
- **  Change:** split `models/qwen4exp/Qwen4ExpTarget.cpp` by phase (prefill / decode
  / draft) behind the existing `submit` seam; prune no-go and dead paths (tree
  drafting if W/Track A R4 says no, draft-head training) behind a compile-time
  profile; delete flags that no longer change behavior.
- **Target:** make the decode state machine reviewable; reduce flag surface.
- **Risk:** low if mechanical (no behavior change).
- **Close:** identical outputs
  (`models/qwen4exp/bench/check_decode_consistency.py`), all suites green,
  `dev/tools/check_architecture.py` passes.

### W8 — Validate the approximations in isolation *(B8)*
- **Change:** measure Q3/Q4 compressed KV, MiniCache depth merging, and
  DuoAttention streaming heads against the uncompressed reference on the 145-item
  scorecard at matched context — not bundled into a single "parity" number.
- **Target:** know the real quality cost before these become default; decide
  prompts/tasks where they are disallowed.
- **Risk:** low (measurement only).
- **Close:** per-feature scorecard table in `docs/`, with enable flags documented.

### W9 — Reproducible verification harness *(B10)*
- **Change:** a scripted, checksum-pinned benchmark that regenerates the
  headline comparisons from fixed prompts/seeds and emits machine-readable output;
  wire it as a CI gate for regressions (not for absolute claims).
- **Target:** any external claim is re-runnable by a third party.
- **Risk:** low.
- **Close:** one command reproduces the table within stated tolerance; run
  captures exit status rather than being filtered.

### W10 — Generalize the runtime contract *(B11)*
- **Change:** formalize the model interface (replace the `QwenTarget` type-switch
  front class) and abstract KV/state so a non-Qwen-hybrid model can plug in. This
  is the prerequisite for any CUDA/ROCm port.
- **Target:** remove the "shared runtime assumes the Qwen family" ceiling.
- **Risk:** high; sequence last and behind W7.
- **Close:** `dev/tools/check_architecture.py` passes with a second, structurally
  different target compiling against the interface.

## 3. Phase plan

Effort is rough engineer-days. Every phase ends with the same measurement gate
unless noted.

**Phase 0 — Baseline & protocol safety net (2–3 days): W1, W9.**
Establish the reproducible baseline and stop protocol drift before touching
either side. Exit: conformance test green; benchmark harness reproduces today's
numbers.

**Phase 1 — Safe, behavior-preserving wins (3–5 days): W2, W3, W7.**
No output changes. Exit: bit-exact outputs; speed suite within noise or better;
`mlock`-failure path tested; `models/qwen4exp/Qwen4ExpTarget.cpp` split with no
result diff.

**Phase 2 — Memory robustness (1 week): W6, W8.**
Memory planner can no longer be broken by constants; quality costs are known and
documented. Exit: boot on fixture; per-feature scorecard published.

**Phase 3 — Performance tracks (2–4 weeks): W4 + Track A R1/R2/R3/R5.**
Deepen the I/O pipeline and land the ready-made speed wins (compressed KV, MTP
host trims, adaptive cache, predictor v2) with traced A/B. Exit: each R measured
independently against Phase 0.

**Phase 4 — Concurrency (1–2 weeks, design first): W5.**
Only after Phase 0's harness exists to prove no single-stream regression.

**Phase 5 — Strategic (multi-week): W10.**
Interface generalization and the portability prerequisite; do after W7 reduces the
surface it must abstract.

## 4. Explicitly not doing now (and why)

- **Multi-lane batching as the default** — the product is single-tenant and MTP is
  disabled at width > 1 by design; W5 explores interleave, not default batching.
- **IPC/latency micro-work** — `speed-optimization-review.md` F7 shows the Python
  path is ≪1 ms/token at 40 tok/s.
- **I/O read-splitting** — measured slower (30 → 39 ms/step); W4 changes
  *scheduling*, not read granularity.
- **EAGLE-2 / trained draft head** — new model artifact, out of scope for code.

## 5. Status

| Track | Item | Status |
|---|---|---|
| W1 | Single-source protocol | ☐ Not started |
| W2 | Futex-equivalent sync | ☐ Not started |
| W3 | Loud cache degradation | ☐ Not started |
| W4 | Deeper async I/O pipeline | ☐ Not started |
| W5 | Concurrency / prefill interleave | ☐ Not started |
| W6 | Declarative memory budget | ☐ Not started |
| W7 | Complexity reduction | ☐ Not started |
| W8 | Approximation quality isolation | ☐ Not started |
| W9 | Reproducible verification harness | ☐ Not started |
| W10 | Runtime contract generalization | ☐ Not started |
| A | `speed-optimization-review.md` R1–R7 | See that doc |

**Safety (non-negotiable):** one engine at a time through
`dev/benchmarks/guarded.py`; never two servers on 8090/8000; after any engine
change run `make`, `make test-engine-cpu`, and
`.venv/bin/python -m unittest dev/tests/test_server.py`; restore the GPU wired
limit after reboot.
