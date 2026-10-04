# How Slipstream fits together

**The short version:** a Python server turns chat requests into token ids and
talks to one native engine process over a pipe. The engine schedules requests.
The shared model runtime runs the decode loop. **Everything specific to one model
lives in its own folder, `models/<name>/`.** `dev/tools/check_architecture.py`
enforces who may depend on whom.

```
 server/              Python: HTTP APIs, chat template, tools/grammar, PDF text
   │  binary protocol over a pipe (server/protocol.py <-> runtime/engine/Protocol.cpp)
 runtime/engine/      requests, scheduling, KV prefix cache, memory plan and governor
 runtime/model/       shared model runtime: decode loop, drafting, per-request state
   │  reaches a concrete model only through 4 plug-in files (below)
 models/qwen4exp/     Flash-Next (125.7B MoE): layout, loader, forward pass, own kernels, tools
 models/qwen38/       Qwen3.8-27B (Dense): layout, loader, forward pass, own kernels, tools
 runtime/ops/         one C++ entry point per shared kernel family
 runtime/metal/       GPU backend: buffers, command graphs, submission; shared kernels
```

---

## What each part owns

| Part | Owns | Key files |
|---|---|---|
| **server** | OpenAI and Anthropic request shapes, chat template, thinking levels, tool-call grammar, streaming | `api_shapes.py`, `frontend.py`, `backend.py`, `runtime.py`, `protocol.py` |
| **engine** | Admitting requests within memory, batching, prompt chunks, reusing cached prompt prefixes, status | `Engine.cpp`, `Scheduler.cpp`, `Cache.cpp`, `KvCache.cpp`, `MemoryPlan.cpp`, `MemoryGovernor.cpp` |
| **shared model runtime** | The decode loop, guess checking, per-request recurrent state, buffers | `Runtime.mm`, `QwenState.*`, `QwenTarget.*`, `ModelDescriptor.*` |
| **model folder** | Everything only this model has | `models/qwen4exp/`, `models/qwen38/` |
| **ops** | Launching shared kernels with the right sizes and variant | `Linear.*`, `MoE.*`, `PagedAttention.*`, `GDN.*`, `Sampling.*` |
| **metal** | Talking to the GPU: allocations, pipelines, events, pipelined submission | `MetalBackend.mm`, `CommandGraph.hpp`, `kernels/`, `abi/` |

**Startup** is assembled in one place: `runtime/engine/RuntimeResources.mm`
(load package → plan memory → allocate) and `Bootstrap.mm`.

## Inside a model folder

### Flash-Next 125.7B MoE (`models/qwen4exp/`)
- `Qwen4Exp.hpp/.cpp`: 125.7B layout (512 routed experts, 7.3B active/token, 48 layers), package loader, expert cache plan.
- `Qwen4ExpTarget.hpp/.cpp`: Forward pass: prompt processing, checking guesses, output layer, MTP draft head.
- `kernels/*.metal`: Hyper-connection mixing, n-gram embedding, sparse attention picker, MTP pick.
- `tools/`: GGUF converter (`convert_qwen4exp_gguf.py`) with APFS hole punching.

### Qwen3.8-27B Dense (`models/qwen38/`)
- `Qwen3_8.hpp/.cpp`: 27B layout (64 dense hybrid layers, 5,120 hidden dim, 17,408 intermediate FFN).
- `Qwen3_8Target.hpp/.cpp`: Forward pass: prompt processing, dense attention + GDN verification, output layer.
- `kernels/*.metal`: `capture.metal`, `draft.metal`, `draft_context.metal`.
- `tools/`: GGUF converter (`convert_qwen38_gguf.py`) and MLX converter (`convert_qwen38_mlx.py`).

The build picks up `models/*/kernels/*.metal` and `models/*/*.cpp` automatically.

---

## Shared versus model-specific

| Shared (`runtime/`) | Specific to qwen4exp (`models/qwen4exp/`) |
|---|---|
| Metal backend, command graphs, pipelined submission | Layout and package format (magic `MDFN0031`) |
| 4-bit and 8-bit matrix kernels, paged 8-bit KV attention | Hyper-connection kernels (4 residual streams) |
| Linear-attention (GDN) kernels and recurrent state | Per-layer n-gram embedding kernels |
| MoE routing, grouped expert tiles | Expert streaming: per-layer cache, SSD reads, "waves" |
| Sampling and checking of guesses | Draft head (MTP), draft vocabulary, `mtp_pick` kernel |
| Engine, memory plan, prefix cache, server | Converter and verification scripts |

## Where a model plugs in

A new model adds its own folder, then registers itself in **four shared files**
(the checker allows no others):

| File | What the model adds |
|---|---|
| `runtime/model/ModelDescriptor.hpp` | Its layout, as one alternative in `TargetLayout` |
| `runtime/model/ModelFactory.hpp` | Its weights, as one alternative in `TargetWeights` |
| `runtime/model/QwenTarget.cpp` | Calls to its `addPrefill / addVerify / addHead / addEmbedding` |
| `runtime/model/Runtime.mm` | Whether it has a draft head (`mtpDrafting()`) |

## Rules the checker enforces

| Rule | Why |
|---|---|
| **A model folder may launch its own kernels**; shared model code may not | Model-specific GPU work stays in one place |
| **A model folder never depends on the engine** or on another model | Models stay swappable |
| **Shared GPU code and ops never depend on a model folder** | Shared code stays shared |
| **The engine never names a concrete model**, except its startup files | Scheduling and memory policy work for any model |

**Honest limit:** the shared runtime (`Runtime.mm`, state, arenas) assumes the Qwen
hybrid family: linear-attention and full-attention layers, an 8-bit paged KV
cache, and a recurrent state per request. A model from this family plugs in as
above. A model outside it (for example pure attention with a different cache)
also needs the runtime's buffers and state generalized. That work is not done.

---

## One decode step, end to end

```
draft head guesses up to 5 tokens (stops early when unsure)       ~7 ms
  └ each guess: its own attention layer + experts + 64K-token output layer
full model checks anchor + guesses in one pass                    ~65 ms
  └ per layer: router → pick experts → cached ones run on the GPU
               while missing ones are read from the SSD ("waves")
keep the guesses the model agrees with, plus one token of its own
```

Tokens per step ≈ 2.8–3.1, so ~38–40 tok/s. See `docs/profiling.md` for how each
number is measured.

---

## Cleanup still to do

| What | Why |
|---|---|
| Drop Homebrew/release packaging and the model installer (`install/`, `dev/tools/package*.py`) | Not used here; wired into 13 tests, so it is its own step |
| Rename `QwenTarget` into an explicit model interface | Today it is a front class that switches on the model's type |
