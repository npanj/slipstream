# Slipstream

**High-performance Apple Silicon inference for Qwen3.8-Flash-Next (125.7B) and Qwen3.8-27B.**

Slipstream is a lean C++ and Metal inference engine built specifically for Apple Silicon. It pairs **SSD expert streaming** and **predictive read-ahead** with **Prompt Lookup + MTP speculative drafting** to serve frontier-scale models on Mac hardware.

It natively supports two model families:
1. **Qwen3.8-Flash-Next V3** (125.7B parameters, 512 routed experts, 7.3B active per token) and its **Swift KV-sparse variant** at **41–52 tok/s** on a 64 GB Mac, with context scaling tested to **130,000 tokens without decode collapse**.
2. **Qwen3.8-27B & Swift-Qwen3.8-27B** (27B dense hybrid, 64 layers) at **42–46 tok/s** with ultra-fast Time-to-First-Token (**659 ms**, 1.70x faster than MoE) and a compact RAM-resident footprint (~16.9–27 GB).

Everything is open source under Apache-2.0.

---

## Supported Models

| Architecture | Model Variant | Checkpoint / Format | Active / Total | Reasoning Scorecard | TTFT | Decode Speed | Memory Footprint |
|---|---|---|---:|---:|---:|---:|---:|
| **125.7B MoE** | **Swift-Flash-Next V3** *(KV-Sparse)* | [nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF](https://huggingface.co/nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF) | 7.3B / 125.7B | **70.3%** (GPQA 54.3%, MATH 62.9%) | 1,119 ms | **41–52 tok/s** | 64 GB Mac (SSD Streamed) |
| **125.7B MoE** | **Qwen3.8-Flash-Next V3** *(Base)* | [nitinpanj/qwen38-flash-next-v3](https://huggingface.co/nitinpanj/qwen38-flash-next-v3) | 7.3B / 125.7B | **67.6%** (GPQA 45.7%, MATH 60.0%) | 1,290 ms | **40–48 tok/s** | 64 GB Mac (SSD Streamed) |
| **27B Dense** | **Swift-Qwen3.8-27B-Splash-HQ** | Prepared Package / GGUF / MLX | 27B / 27B | **67.6%** (AIME 45.0%, GSM8K 96.0%) | **659 ms** *(1.7x faster)* | **42–46 tok/s** | 32–64 GB Mac (RAM Resident) |
| **27B Dense** | **Qwen3.8-27B** *(Base)* | Standard GGUF / MLX safetensors | 27B / 27B | Frontier 27B baseline | **670 ms** | **40–45 tok/s** | 32–64 GB Mac (RAM Resident) |

### Key Highlights
- **Frontier Reasoning on a Laptop:** Flash-Next V3 reaches 70.3% composite accuracy across 145 benchmark items (54.3% GPQA Diamond, 62.9% MATH-500, 92.0% HumanEval, 96.0% GSM8K).
- **Fast Generation:** 41–52 tok/s sustained decode on an M5 Pro for both architectures.
- **Ultra-Fast TTFT on 27B:** Dense 27B achieves sub-second prompt evaluation (570–659 ms), 1.7x faster than large MoE.
- **No Context Collapse:** Flash-Next maintains 32–43 tok/s out to 130,000 tokens through 48 recurrent linear DeltaNet layers ($O(1)$ state growth) and only 16 full-attention layers.
- **In-Place GGUF Preparation:** Automatic Darwin APFS hole-punching (`F_PUNCHHOLE`) converts standard multi-shard GGUFs without doubling disk consumption.

---

## Quickstart (Step-by-Step)

### macOS Menu Bar App (GUI)

Prefer a native Mac app instead of Terminal commands?
Use [**Slipstream Menubar**](https://github.com/npanj/slipstream-menubar), created and co-authored by **Mike Zinner** ([@mzinner](https://github.com/mzinner)):

- **One-click server control:** Start, stop, and switch models directly from the macOS menu bar.
- **Glanceable speed:** Displays real-time prompt (`↓`) and generation (`↑`) tokens/second in the menu bar.
- **Live HUD Stats Panel:** Floating dashboard graphing token speeds, TTFT, speculative draft acceptance rate, KV cache pressure, and Mac GPU/memory utilization.
- **Zero configuration friction:** Automatically configures `--port`, raises macOS GPU wired memory limits on 64 GB Macs, and prepares models in place.

Install the menu bar app via Homebrew:
```zsh
brew install --cask npanj/tap/slipstream-menubar
```
*(Or install via script: `curl -fsSL https://github.com/npanj/slipstream-menubar/raw/main/install.sh | sh`, or download the `.dmg` from [Slipstream Menubar Releases](https://github.com/npanj/slipstream-menubar/releases/latest).)*

---

### Prerequisites
- **Hardware:**
  - **Flash-Next V3 (125.7B MoE):** Apple Silicon Mac with **64 GB Unified Memory** (M2/M3/M4/M5 Pro/Max).
  - **Qwen3.8-27B (Dense):** Apple Silicon Mac with **32 GB or 64 GB Unified Memory**.
- **Disk:**
  - Flash-Next V3: ~100 GB. Slices of GGUFs are freed in place via APFS hole punching (`F_PUNCHHOLE`) during preparation.
  - 27B Dense: ~17–27 GB depending on quantization (Q4 / Q8).
- **macOS:** macOS 15.0+ (macOS 26.4+ SDK).

---

### Step 1: Install Slipstream

You can install Slipstream directly without compiling, or build from source:

#### Option A: Homebrew or One-Line Install (Recommended — No compilation required)

Install via Homebrew:
```zsh
brew install npanj/tap/slipstream
```

Or install via one-line curl script:
```zsh
curl -fsSL https://raw.githubusercontent.com/npanj/slipstream/main/install.sh | sh
```

*This automatically detects your Apple Silicon Mac, downloads the latest prebuilt release binary, verifies its SHA256 checksum, unpacks into `~/.local/share/slipstream/`, and links `slipstream` into `~/.local/bin`.*

You can also download `slipstream-<version>-macos26-arm-64bit.zip` directly from [GitHub Releases](https://github.com/npanj/slipstream/releases/latest).

#### Option B: Build from source

```zsh
git clone https://github.com/npanj/slipstream.git
cd slipstream
make -j4
```

#### Where Slipstream keeps things

| What | Where | Override |
| :--- | :--- | :--- |
| Installed versions (the two newest are kept) | `~/.local/share/slipstream/<version>/` | `SLIPSTREAM_PREFIX` |
| The `slipstream` command (a link to the newest version) | `~/.local/bin/slipstream` | `SLIPSTREAM_BINDIR` |
| Models | `~/.slipstream/models/<owner>/<repo>/` | `SLIPSTREAM_MODELS` |
| Downloaded packages (models link into it) | the Hugging Face cache, `~/.cache/huggingface/hub/` | `HF_HUB_CACHE` |
| Runtime data, logs and caches of a release | `~/Library/Application Support/Slipstream/` | |

**Why `~/.local`:** the installer puts Slipstream in your home folder, so it needs no administrator
rights, no `sudo` and no package manager, and removing it is deleting two paths. `~/.local/bin` and
`~/.local/share` are the per-user locations of the XDG convention that command-line tools on macOS
increasingly use: Claude Code's installer links `claude` into `~/.local/bin` the same way, the
installer is modelled on MariaDB Shell's, and `pipx` and `uv` install tools there too. Versions sit
side by side, so going back is relinking `~/.local/bin/slipstream`. If `~/.local/bin` is not on your
`PATH`, the installer says how to add it.

**Why `~/.slipstream/models`:** a model outlives any one installation. Every
installation (a release, a source checkout, or a frontend that runs `slipstream pull`) uses the
same store, so a model is downloaded once and survives upgrades, reinstalls and `git clean`. It is
the convention of other local model servers: Ollama keeps models in `~/.ollama/models`, oMLX in
`~/.omlx/models`. The store is excluded from Time Machine when it is created (`tmutil
addexclusion`), since everything in it can be downloaded again.

**Before this, models lived inside each installation:** a release kept them in
`~/Library/Application Support/Slipstream/models`, a source checkout in `install/models`. Nothing
is moved automatically. Either move a model into the store, keeping its `<owner>/<repo>` folder:

```zsh
mkdir -p ~/.slipstream/models/nitinpanj
mv ~/Library/Application\ Support/Slipstream-v2/models/nitinpanj/<repo> ~/.slipstream/models/nitinpanj/
```

or point `SLIPSTREAM_MODELS` at the old folder. A folder you downloaded yourself (e.g. with
`huggingface-cli` into `~/models/…`) is served from where it is, by its path, and never changed.

---

### Step 2: Download or Pull the Model

#### Model 1: Qwen3.8-Flash-Next V3 (125.7B MoE — Frontier Reasoning)

Serve directly by Hugging Face repo ID (Slipstream downloads and sets up weights and MTP sidecars automatically):

```zsh
# Download without serving:
slipstream pull nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF

# Only check whether a repository can be served, and its size (downloads nothing):
slipstream pull nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF --check
```

Or download manually using the Hugging Face CLI:

```zsh
# Recommended: Swift-Qwen3.8-Flash-Next V3 (95.5 GiB, KV-sparse)
huggingface-cli download nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF \
    --local-dir ~/models/swift-qwen38-flash-next-v3

# Or download the plain base model:
huggingface-cli download nitinpanj/qwen38-flash-next-v3 \
    --local-dir ~/models/qwen38-flash-next-v3
```

#### Model 2: Qwen3.8-27B & Swift-Qwen3.8-27B (Dense Hybrid — Ultra-Fast TTFT)

Slipstream natively ingests standard GGUF files, MLX safetensors checkpoints, or pre-converted packages:

```zsh
# Download any standard 27B GGUF:
huggingface-cli download Qwen/Qwen3.8-27B-GGUF qwen3.8-27b-q4_0.gguf \
    --local-dir ~/models/qwen38-27b

# Or download an MLX 4-bit checkpoint:
huggingface-cli download mlx-community/Qwen3.8-27B-4bit \
    --local-dir ~/models/mlx-qwen38-27b
```

---

### Step 3: Raise GPU Memory Limit (One-Time per Boot)

On 64 GB Macs, macOS defaults the wired GPU limit to ~48 GiB. Raise it to 58 GiB so the SSD expert cache and KV pool have ample headroom:

```zsh
sudo sysctl iogpu.wired_limit_mb=59392
```

---

### Step 4: Serve the Model

Run `slipstream serve` pointing at your model path or Hugging Face repo ID:

#### Serving Flash-Next V3 (125.7B MoE)

```zsh
# Serve local model directory:
slipstream serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090

# Or serve directly by Hugging Face repo ID (downloads & sets up automatically):
slipstream serve --model nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF --port 8090

# Or serve on your local network:
slipstream serve --model ~/models/swift-qwen38-flash-next-v3 --host 0.0.0.0 --port 8090 --api-key YOUR_KEY
```

#### Serving Qwen3.8-27B (Dense Hybrid)

```zsh
# Serve a prepared 27B package:
slipstream serve --model ~/models/swift-qwen38-27b-splash-hq --port 8090

# Or serve a single .gguf file directly:
slipstream serve --model ~/models/qwen38-27b/qwen3.8-27b-q4_0.gguf --port 8090

# Or serve an MLX directory:
slipstream serve --model ~/models/mlx-qwen38-27b --port 8090
```

> **Memory Rule — One Model at a Time:** Only run one inference engine at a time on Apple Silicon. Stop any active server (`Ctrl+C`) before switching models.
>
> **Automatic Format & Architecture Detection:** Slipstream detects whether a model is 27B (`qwen38`) or Flash-Next (`qwen4exp`), and whether it is in GGUF, MLX, or prepared package format. On first launch, multi-shard GGUFs are prepared into `<model-dir>/prepared/` using APFS hole-punching (`F_PUNCHHOLE`) to prevent disk duplication. Subsequent loads take **~10–15 seconds**. Add `--keep-gguf` if you wish to retain original GGUF files for other tools.

---

### Step 5: Connect Your Tools & Clients

The server provides a standard OpenAI-compatible API on `http://127.0.0.1:8090`:

#### curl
```zsh
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local/swift-qwen38-flash-next-v3",
    "messages": [
      {"role": "user", "content": "Write a clean, optimal Python function for interval merging."}
    ],
    "temperature": 0.0
  }'
```

#### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="local/swift-qwen38-flash-next-v3",
    messages=[{"role": "user", "content": "Explain multi-head self-attention with linear algebra."}],
    temperature=0.0,
)
print(response.choices[0].message.content)
```

#### Oh My Pi (omp) & Coding Agents
```zsh
# Launch omp connected directly to Slipstream
omp --model splash-flashnext/local/swift-qwen38-flash-next-v3 \
    --tools=read,write,edit,bash,grep,glob,todo \
    --thinking=low --approval-mode=yolo
```

---

## Choosing Between Flash-Next V3 and Qwen3.8-27B

| Dimension | Qwen3.8-Flash-Next V3 (125.7B MoE) | Qwen3.8-27B & Swift-27B (Dense Hybrid) |
|---|---|---|
| **Architecture** | 512 routed experts (7.3B active/token) + GDN recurrence | 64-layer dense hybrid (full attention every 4th layer + GDN) |
| **Reasoning Scorecard** | **70.3% composite** (leads GPQA Diamond by +8.6%, MATH-500 by +2.9%) | **67.6% composite** (AIME 45.0%, GSM8K 96.0%, HumanEval 92.0%) |
| **Decode Throughput** | 41–52 tok/s | 42–46 tok/s |
| **Time-to-First-Token** | ~1,100–1,300 ms | **570–659 ms (1.70x faster)** |
| **Memory Footprint** | ~34 GB expert cache + SSD streaming (requires 64 GB Mac) | **16.9–27 GB completely RAM-resident** (runs on 32 GB or 64 GB Macs) |
| **Context Horizon** | Scaled to **130,000 tokens** without decode collapse | Standard 32k–131k context window |
| **When to Use** | Deep analytical reasoning, coding agents, math proofs, massive documents | Fast conversational turns, latency-critical tasks, lighter memory environments |

---

## Benchmarks & Performance

### 1. Head-to-Head: llama.cpp Fork vs. Slipstream
*Evaluated on Apple MacBook M5 Pro (64 GB Unified Memory, Temperature 0.0):*

| Task Domain | Benchmark / Prompt | llama.cpp Fork | Slipstream | Speedup | llama.cpp TTFT | Slipstream TTFT |
|---|---|---:|---:|---:|---:|---:|
| **Math Reasoning** | GSM8K (eggs derivation) | 24.0 tok/s | **43.6 tok/s** | **1.82x** | 4,024 ms | **2,337 ms** |
| **Math Derivation** | MATH-500 series ($p - q$) | 24.3 tok/s | **43.1 tok/s** | **1.77x** | 1,655 ms | **1,587 ms** |
| **Constraint Logic** | 3-chair deduction | 25.4 tok/s | **46.0 tok/s** | **1.81x** | 1,469 ms | **1,042 ms** |
| **Python Coding** | `merge_intervals` ($O(N \log N)$) | 19.7 tok/s | **35.0 tok/s** | **1.77x** | 1,507 ms | **1,070 ms** |
| **Systems Coding** | Rust CSV parser | 22.7 tok/s | **37.5 tok/s** | **1.65x** | 1,257 ms | **859 ms** |
| **Tech Communication** | Multi-head attention | 22.5 tok/s | **39.4 tok/s** | **1.75x** | 1,267 ms | **843 ms** |
| **OVERALL AVERAGE** | Across all 6 domains | **23.1 tok/s** | **40.8 tok/s** | **1.76x** | **1,863 ms** | **1,290 ms** |

![Throughput Comparison](docs/images/1_slipstream_vs_llamacpp_throughput.png)

---

### 2. Reasoning Accuracy: Swift V3 vs. Plain Base V3
*Evaluated across 145 standardized items under memory guard (Seed 1234, T=0.0):*

| Benchmark | Items | Plain Flash-Next V3 | Swift-Flash-Next V3 | Accuracy Delta |
|---|---:|---:|---:|---:|
| **AIME 2025** | 20 | 45.0% (9/20) | **45.0% (9/20)** | 0.0% |
| **MATH-500 (L4–5)** | 35 | 60.0% (21/35) | **62.9% (22/35)** | **+2.9%** |
| **GPQA Diamond** | 35 | 45.7% (16/35) | **54.3% (19/35)** | **+8.6%** |
| **GSM8K** | 25 | 96.0% (24/25) | **96.0% (24/25)** | 0.0% |
| **HumanEval** | 25 | 92.0% (23/25) | **92.0% (23/25)** | 0.0% |
| **Hard Logic** | 5 | 100.0% (5/5) | **100.0% (5/5)** | 0.0% |
| **OVERALL SCORECARD** | **145** | **67.6% (98/145)** | **70.3% (102/145)** | **+2.8%** |

![Reasoning Benchmarks](docs/images/3_swift_v3_quality_benchmarks.png)

---

### 3. Context Scaling: Live Telemetry to 130,000 Tokens
*Measured across 3,086 live agent requests on Apple Silicon (M5 Pro 64 GB):*

| Context Range (Tokens) | Live Runs | Average Decode | Median Decode (p50) | Peak Decode | Avg TTFT |
|---|---:|---:|---:|---:|---:|
| **< 1,000** | 314 | **41.5 tok/s** | 41.9 tok/s | 59.8 tok/s | 2.16 s |
| **1k – 4,000** | 21 | **41.0 tok/s** | 42.5 tok/s | 64.5 tok/s | 5.26 s |
| **4k – 8,000** | 58 | **43.6 tok/s** | 43.2 tok/s | 67.2 tok/s | 7.36 s |
| **8k – 16,000** | 117 | **43.6 tok/s** | 44.6 tok/s | 58.2 tok/s | 7.91 s |
| **16k – 32,000** | 562 | **38.2 tok/s** | 40.9 tok/s | 58.0 tok/s | 13.59 s |
| **32k – 64,000** | 1,029 | **35.0 tok/s** | 37.5 tok/s | 55.6 tok/s | 13.24 s |
| **64k – 96,000** | 650 | **32.4 tok/s** | 34.7 tok/s | 53.9 tok/s | 12.81 s |
| **96k – 130,000** | 364 | **32.9 tok/s** | 33.3 tok/s | 43.8 tok/s | 7.95 s |

![Context Scaling](docs/images/2_context_scaling_130k_telemetry.png)

---

## Foundation for Qwen4

The core primitives implemented in Slipstream:
- 512-route sparse MoE streaming with predictive read-ahead
- Quasi-Sparse Attention (QSA) indexer and selection kernels
- Hyper-connection mixing and per-layer embedding gathers
- Metal GPU-mapped n-gram tables
- Single-lane speculative verification with Prompt Lookup Decoding (PLD) + Multi-Token Prediction (MTP)

...were engineered to match the upcoming model generation. Assuming **Qwen4** follows Flash-Next's architectural blueprint (hybrid linear recurrence + sparse attention + routed MoE experts), Slipstream can serve as a direct template to run Qwen4 locally on Apple Silicon on day one.

---

## Call for Porting Partners: NVIDIA (CUDA) & AMD (ROCm)

- **Current Status:** Tested exclusively on an **Apple MacBook Pro (M5 Pro, 64 GB Unified Memory)**.
- **Porting:** Because I do not have access to modern NVIDIA (CUDA) or AMD (ROCm) GPU hardware, I cannot build and test those backends myself.
- **Collaboration Offer:** If anyone in the community has hardware available and wants to bring these expert-streaming and speculative decoding gains to CUDA or ROCm, **I am happy to collaborate and help with the port**. Open an issue or reach out!

---

## Credits & Acknowledgments

- **Mike Zinner** ([@mzinner](https://github.com/mzinner)): Creator and co-author of [Slipstream Menubar](https://github.com/npanj/slipstream-menubar), the official native macOS menu bar app for running, configuring, and monitoring Slipstream.
- **Splash Team (Incoai)**: Full credit to the creators of Splash ([github.com/incoai/splash](https://github.com/incoai/splash)). Their C++ Metal speculative decoding design and memory architecture provided the foundation for this work. We will prepare a clean PR/patch proposing these Flash-Next and SSD streaming extensions to the Splash upstream repo.
- **ds4 Team**: For their valuable insights on Metal router numerical precision (Taylor polynomial softplus expansion) and streaming scheduling designs.
- **Qwen Team**: For training Qwen3.8-Flash-Next and open-sourcing the hybrid linear MTP architecture.
- **ukisai**: For the Swift-1.5 distillation work enabling KV-sparse reasoning.
- **bartowski & unsloth**: For donor quants and quantization tooling.
- **mihailescu2m**: For initial expert streaming concepts in llama.cpp.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
