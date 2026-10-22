# What can be executed where: the cloud answer to the roadmap

**Question this answers:** can the plan in [`docs/roadmap.md`](roadmap.md) be executed in a cloud coding environment?

**Short answer: no, not the plan. About a third of it, yes** — the parts that are pure CPU and pure logic. The rest is
gated on Apple silicon, because this repository *is* a Metal engine and every number in the roadmap is measured on an
M5 Pro.

---

## 1. The environment, measured

The development environment used to write this branch:

| Check | Result |
|---|---|
| `uname -a` | `Linux … 6.1.158 x86_64 GNU/Linux` |
| `sw_vers` | not macOS |
| `/System/Library/Frameworks` | absent (no Foundation, no Metal, no IOKit) |
| `xcrun` | not found |
| Toolchain | `g++ (Debian 12.2.0)`; no `clang`, no `metal` compiler |
| `make -n` (dry run) | only *prints* `xcrun -sdk macosx clang++ …` and `xcrun -sdk macosx metal …` commands that cannot run here |
| GPU | none; nothing to measure |

Consequences, stated plainly:

- The engine cannot be built here at all (it needs the macOS SDK and Metal), so **no engine change can be
  compile-verified** in this environment. CI on `macos-26` can compile plain C++ and the macOS targets; only a Mac can
  run Metal kernels.
- No benchmark, trace, logit dump, `pmset` reading, SSD measurement or thermal check is possible. Every speed and
  quality number in the roadmap stays unverified until it runs on the Mac.
- Anything touching `runtime/metal/`, `models/*/kernels/*.metal`, `Runtime.mm`, `Qwen4ExpTarget.cpp` or the expert
  store is **out of reach** for confident work here: it would be unverified code in the hottest, most numerical path.

## 2. The roadmap by executability

| Priority | Item | Here? | Why |
|---|---|---|---|
| P0 | M0 baseline (harness run) | ✗ | needs the Mac and the model |
| P0 | M1 logit dump + noise floor | ~ | the *tools* (`logit_dump` mode, `logit_floor.py`) are C++/Python that touch the engine and need the model to produce a floor: written here, run there |
| P0 | M2 engine counters as metrics | ~ | plumbing through `Status.cpp`/`metrics.py` can be drafted; it must be built and checked on the Mac |
| P0 | L2 measure the in-memory hit rate | ✗ | the counters exist; only a real session measures them |
| P0 | Q1 retire `SPLASH_TREE_DRAFT` | ~ | mechanical removal, but it must compile: do it on the Mac or let CI prove it |
| P1 | **L1 SSD prefix cache** | **✓ core / ~ integration** | the store is CPU-only, POSIX and model-agnostic: **implemented and tested here** (this branch). Wiring it into `Engine`/`Cache`/`StateCache` needs a build on the Mac |
| P1 | R1/R2/R4 prefill experiments | ✗ | measurements |
| P1 | **D1 measured row budget** | **✓ core** | calibration, cost model and chooser are pure arithmetic: portable and unit-testable; the hook into `Qwen4ExpTarget.cpp` is not |
| P1 | D2 live-row draft head | ✗ | GPU-side |
| P1 | M3 Metal gate in CI | ✓ | workflow YAML + runner labels: writable and reviewable here, exercised by the self-hosted runners |
| P2 | E1/E2 expert store | ✗ | Metal + I/O + handshake; must be developed on the Mac |
| P2 | L5 long-context sweep | ✗ | measurements |
| P2 | Q2 regression table | ✓ | a Python runner over existing scripts; runnable on the Mac |
| P2 | Q3 thermal flags | ✓ | harness/reporting code |
| P2 | O1 `doctor` | ~ | the memory-plan math is C++ that links the engine; draftable, build-gated |
| P2 | O2 configuration inventory | ✓ | a script over the sources; runs anywhere |
| P2 | O3 fail loudly | ~ | engine-side; build-gated |
| P3 | D3 batch budget, D4 tree verifier, E3 | ✗ | the two riskiest items, both GPU |
| P3 | P1 second package, P2 CLI, P3 packaging | ~ | Python-side mostly draftable; verification is on the Mac |

`✓` = can be done and verified in a cloud container. `~` = partly. `✗` = needs the Mac.

## 3. What was actually executed here

**L1's core: `runtime/engine/PrefixCache.{hpp,cpp}` (899 lines) and
`dev/tests/engine/prefix_cache_test.cpp` (546 lines, 20 test cases).**

The store is the half of the SSD prefix cache that has no GPU in it: content keys, the file format, atomic writes,
checksums, the namespace lock, the index and the space quota. Kapitsa's equivalent is CPU-only for the same reason, and
it is the piece that most often goes wrong in ways that look like "the model got dumber".

What it does:

| Property | How | Test that proves it |
|---|---|---|
| Atomic publish | write to `.tmp-XXXXXX`, `fsync`, `rename` | a killed process's leftover temp file is never listed, counted or served |
| Corruption is a miss, never data | CRC-32C over the payload **and** the header | flipped payload byte, flipped header byte, truncated file: all misses, counted in `stats().corrupt` |
| No cross-model contamination | 128-bit namespace digest in every header, checked on read | a file written by another namespace is refused |
| No format surprises | magic + version in the header | a foreign file is refused |
| Durability | directory scan rebuilds the index after a restart | reopen finds the entry and the right byte count |
| One writer per namespace | `flock` on `<ns>/lock` | a second `Store` on the same namespace comes up **disabled**, not competing |
| Bounded disk | `min(quota, (free + held − floor) / 4)`, evict least-recently-used | quota eviction order; a read refreshes recency; shrinking the quota shrinks the cache |
| Thread safety | one mutex; const readers update only bookkeeping | 4 threads × 32 put/get round trips, clean under TSan |
| No page-cache pollution | `F_NOCACHE` on macOS | (compile-time on the Mac) |

How it was verified here — the same flags the repo's own test target uses (`-std=c++20 -O2 -Wall -Wextra -Werror
-Iruntime -I. -Idev`), with `xcrun -sdk macosx clang++` replaced by `g++` and the macOS-only flag dropped:

```text
$ make -n build/engine-tests/prefix-cache
… xcrun -sdk macosx clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Iruntime -I. -Idev \
      -mmacosx-version-min=26.4 runtime/engine/PrefixCache.cpp \
      dev/tests/engine/prefix_cache_test.cpp -o build/engine-tests/prefix-cache

$ g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iruntime -I. -Idev \
      runtime/engine/PrefixCache.cpp dev/tests/engine/prefix_cache_test.cpp -o /tmp/prefix_cache_test
$ /tmp/prefix_cache_test
Prefix cache tests passed

# sanitizers, as `make test-sanitizers` runs them
$ g++ -std=c++20 -O1 -g -fno-omit-frame-pointer -Wall -Wextra -Werror -fsanitize=address,undefined …
Prefix cache tests passed
$ g++ -std=c++20 -O1 -g -Wall -Wextra -fsanitize=thread …
Prefix cache tests passed
```

Also wired: `Makefile` (production source list) and `dev/native.mk` (the `prefix-cache` test target, in both test lists),
so `make test-engine-cpu` runs it on the Mac and CI compiles it on every pull request.

**Honest limit of that evidence:** the ARM hardware-CRC branch of `crc32c()` (`__crc32cd`) is compiled only on Apple
silicon, so it is *not* exercised here; the portable path that the tests do exercise is byte-identical in output, and
the test vectors would catch a wrong hardware path on the first Mac run. Nothing in this change was compiled by the
project's own toolchain.

## 4. What remains for L1 to be worth having

The store is necessary and not sufficient. The rest is engine work that must be developed with a build:

| # | Work | Where | Verify with |
|---|---|---|---|
| 1 | Publish committed blocks | `Cache::publishCommittedBlocks` (`runtime/engine/Cache.cpp:139`) hands (key, rows) to the store after a prefill/decode commit | a trace shows files appearing as tokens are committed |
| 2 | Save states at boundaries | `StateCache` already keys states by KV block; write them as `Kind::State`, **including the draft ring** | a restore resumes drafting with a coherent ring |
| 3 | Restore on admission | `Engine::admitKv` / resume boundary (`Engine.cpp:676`): longest cached prefix, restore pages + state, set `promptProcessed` | follow-up TTFT after a restart, vs the M0 baseline |
| 4 | Namespace digest inputs | model files' SHA/size, drafter, `StateLayout`, the store's format version | changing a weight file changes the digest; a test fails if the state layout changes without a bump |
| 5 | CLI + reporting | `--disk-cache auto\|off\|SIZE`; `cached_tokens` must count restored prefixes; `/metrics` for bytes/hits/evictions | the harness's `cached TTFT` phase |
| 6 | Privacy contract | off or opt-in by default, one documented directory, a one-command delete, never in a cloud-synced folder | `--help` text + `docs/configuration.md` |
| 7 | Eviction interaction | `MemoryGovernor` should spill pinned prefixes instead of dropping them | p90 TTFT under memory pressure |

Items 1–3 are where the risk is, and they cannot be validated here. That is the handoff in §6.

## 5. What I will not do from a container

- **Touch the Metal path.** Kernels, `Runtime.mm`, `Qwen4ExpTarget.cpp`, the expert store: unverified edits there are
  worse than no edits.
- **Claim a speedup.** Every "B" impact class in the roadmap is a hypothesis until measured on the Mac.
- **Change a default.** The repo's own rule is that a default changes only after an interleaved A/B and a logit check;
  neither can happen here.

## 6. Handoff: finishing L1 on the Mac

```bash
# 1. Build and run the new test through the project's own target (this is the gate
#    that has not been run yet).
make -j4 build/engine-tests/prefix-cache && build/engine-tests/prefix-cache
make test-engine-cpu            # the target is in the list now
make test-sanitizers            # crc, atomic writes and eviction under ASan/UBSan

# 2. Then integrate, in this order, checking after each step that outputs are
#    unchanged (greedy, 10-prompt suite) and that the M0 numbers do not regress.
#    (1) publish blocks  (2) save states  (3) restore on admission

# 3. Measure what the whole item is for (needs the harness config in
#    dev/benchmarks/kapitsa-harness/):
python3 compare_engines.py configs/slipstream.json --only 'flash-slipstream'

# 4. Before a default change: the logit gate (roadmap M1).
```

**Recommendation on how to split the work.** Keep the integration PR small and machine-verified: one commit per hook
above, each with `make test-engine-cpu` and a token-identity check. A container can draft those commits, but a Mac must
compile them, so treat anything I write beyond this point as a *draft to be compiled*, not as tested code.

## 7. Queue for the next container session

In the same order as the roadmap, limited to what a container can verify:

| Next | Item | Deliverable | Verifiable here |
|---|---|---|---|
| 1 | **D1 core** | `runtime/ops/DraftBudget.{hpp,cpp}`: per-depth logistic calibration, per-row-count cost table, `E(n) − rate·cost(n)` chooser, exploration, cache file | unit tests with a synthetic drafter of known acceptance; replay of a fixed record set |
| 2 | **O2** | `dev/tools/config_inventory.py`: every `SPLASH_*` / `SLIPSTREAM_*` read with file:line, classified | runs on any machine |
| 3 | **Q2 skeleton** | `dev/benchmarks/regression.py`: one command that runs the quality suite + logit comparison and appends a row per build | dry-run against the existing scripts |
| 4 | **M3** | CI workflow for the self-hosted Metal runners | YAML review, then a real run on the runners |
| 5 | **O5** | `dev/benchmarks/results/` layout + generated README table + a CI check that the table is current | generated from committed JSON |
| 6 | **L1 integration drafts** | the six hooks in §4 as reviewable patches, marked "must be compiled" | **not verifiable here** |

Everything else — the measurements, the kernels, the expert store, the tree verifier — needs the M5 Pro.
