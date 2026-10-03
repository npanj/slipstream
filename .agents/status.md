# Status — Slipstream (handoff)

**Updated:** 2026-10-03 11:32 PDT by antigravity.
**Branch:** `main` (canonical workspace `model-serving/slipstream`, release `v26.10.4`, synced with `origin/main`).

## In one line

All community pull requests (#7, #8, #9, #10) cleanly reviewed, merged, and pushed to `origin/main`. Native 27B model architecture (`models/qwen38/`, `Qwen3_8Layout`, `Qwen3_8Q8Layout`) and Flash-Next operational with standard GGUF and MLX ingestion, in-place GGUF preparation via APFS hole-punching (`F_PUNCHHOLE`), and repository pre-flight inspection (`pull --check`). Proprietary Splash packages deprecated in favor of standard GGUF. All 21 CPU engine tests, model execution plans, 171/171 server tests, 58/58 model tests, and 21/21 launcher tests pass 100% green.

---

## How to use it

```zsh
# One-line install of prebuilt binary:
curl -fsSL https://raw.githubusercontent.com/npanj/slipstream/main/install.sh | sh

# Serve Swift-Qwen3.8-Flash-Next-V3 (local directory or Hugging Face repo):
./slipstream serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090
./slipstream serve --model nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF --port 8090

# Serve 27B models (GGUF folder, single .gguf file, MLX safetensors, or prepared package):
./slipstream serve --model ~/models/swift-qwen38-27b-splash-hq --port 8090
./slipstream serve --model /path/to/qwen38-27b.gguf --port 8090
./slipstream serve --model /path/to/mlx-27b-directory --port 8090

# Inspect a remote Hugging Face repository before downloading:
./slipstream pull nitinpanj/qwen38-flash-next-v3 --check
./slipstream pull nitinpanj/qwen38-flash-next-v3 --check --json

# Backward compatibility forwarders work identically:
./slipstream-v2 serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090
./splash serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090
```

---

## Head-to-Head Benchmarks: llama.cpp Fork vs. Slipstream (Flash-Next V3)

Across 6 standard reasoning and coding tasks (temperature 0.0, M5 Pro 64 GB):

| Task | Category | llama.cpp Fork | Slipstream | Speedup | llama.cpp TTFT | Slipstream TTFT |
|---|---|---:|---:|---:|---:|---:|
| `gsm8k_math` | Math Reasoning | 24.0 tok/s | **43.6 tok/s** | **1.82x** | 4,024 ms | **2,337 ms** |
| `math500_series` | Math Derivation ($p - q$) | 24.3 tok/s | **43.1 tok/s** | **1.77x** | 1,655 ms | **1,587 ms** |
| `constraint_logic`| Constraint Logic | 25.4 tok/s | **46.0 tok/s** | **1.81x** | 1,469 ms | **1,042 ms** |
| `python_intervals`| Python Coding | 19.7 tok/s | **35.0 tok/s** | **1.77x** | 1,507 ms | **1,070 ms** |
| `rust_csv` | Systems Coding | 22.7 tok/s | **37.5 tok/s** | **1.65x** | 1,257 ms | **859 ms** |
| `tech_explanation`| Technical Writing | 22.5 tok/s | **39.4 tok/s** | **1.75x** | 1,267 ms | **843 ms** |
| **OVERALL AVG** | Across all 6 tasks | **23.1 tok/s** | **40.8 tok/s** | **1.76x** | **1,863 ms** | **1,290 ms** |

---

## Head-to-Head Quality: Plain Flash-Next V3 vs. Swift-Flash-Next V3

Completed across 145 paired items (seed 1234, temperature 0.0), run sequentially under memory guard:

| Domain / Benchmark | Items | Swift-Flash-Next-V3 | Plain Flash-Next V3 | Accuracy Delta | Swift Decode | Plain Decode |
|---|---:|---:|---:|---:|---:|---:|
| **AIME 2025** | 20 | **45.0% (9/20)** | **45.0% (9/20)** | 0.0% | 44.3 tok/s | 44.3 tok/s |
| **MATH-500 (L4-5)** | 35 | **62.9% (22/35)** | 60.0% (21/35) | **+2.9%** | 44.8 tok/s | 44.8 tok/s |
| **GPQA Diamond** | 35 | **54.3% (19/35)** | 45.7% (16/35) | **+8.6%** | 44.8 tok/s | 44.8 tok/s |
| **GSM8K** | 25 | **96.0% (24/25)** | **96.0% (24/25)** | 0.0% | 45.6 tok/s | 45.6 tok/s |
| **HumanEval** | 25 | **92.0% (23/25)** | **92.0% (23/25)** | 0.0% | 40.6 tok/s | 40.6 tok/s |
| **Hard Systems & Logic**| 5 | **100.0% (5/5)** | **100.0% (5/5)** | 0.0% | 39.2 tok/s | 39.2 tok/s |
| **TOTAL / OVERALL** | **145** | **70.3% (102/145)** | **67.6% (98/145)** | **+2.8%** | **44.4 tok/s** | **43.9 tok/s** |

---

## Invariants & Rules

- **One model server at a time.** Two pin more memory than the Mac has and freeze it. Run engine experiments through `dev/benchmarks/guarded.py -- <cmd>`; check `lsof -i :8090` before starting.
- **Do not launch daemon without user request.** Server on port 8090 must remain stopped until explicitly launched.
- **After any engine change:** `make`, `make test-engine-cpu`, `.venv/bin/python -m unittest dev/tests/test_server.py`.
