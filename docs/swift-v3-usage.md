# Running Swift-Qwen3.8-Flash-Next V3 on Slipstream

Step by step: download, serve, and use the Swift V3 model on one Apple Silicon Mac —
**with Swift** (the KV-sparse variant, what `pi` uses today) and **without Swift**
(the plain Qwen3.8-Flash-Next base). Everything below is generated from this
repository's actual CLI (`install/launcher.py`, `server/server.py`) and the actual
Hugging Face file listings, not from memory.

Every flag in this document is grep-verified in this repo. Env-var names carry the
historical `SPLASH_` prefix.

---

## 0. What "with Swift" and "without Swift" mean

| | **With Swift** | **Without Swift** |
|---|---|---|
| Model | Swift-Qwen3.8-Flash-Next V3 — Qwen3.8-Flash-Next with **KV-sparse (Swift) layers** | Qwen3.8-Flash-Next V3 — plain hybrid, dense KV |
| Base GGUF | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` (Q4_0 + Q8_0 donor splice) | `nitinpanj/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF` |
| Local dir | `~/models/swift-qwen38-flash-next-v3/` | `~/models/qwen38-flash-next-v3/` |
| Model id served | `local/swift-qwen38-flash-next-v3` | `local/qwen38-flash-next-v3` |
| Measured (this Mac, M5 Pro 64 GB) | 41.72 tok/s decode, TTFT 1,305 ms | 43.9 tok/s decode, comparable quality (70.3 % vs 67.6 % for Swift, 145 items) |
| Extra disk for the prepared package | ~100 GB (`prepared/`) | ~100 GB (`prepared/`) |

Both use the **same engine and the same command** — only `--model <dir>` changes.
`slipstream-gguf` contains no Swift-specific code path: Swift is a property of the
weights, not of the engine. That is why the two columns below are one command each.

**Requirements:** Apple Silicon with 64 GB unified memory, ~350 GB free disk (GGUF +
prepared package + headroom), macOS GPU memory limit raised to 58 GiB (the launcher
asks for your password once per boot), and one engine at a time — two big models
pin more memory than the Mac has and freeze it until the watchdog reboots it.

---

## 1. Download from Hugging Face

Repos are public under `nitinpanj/`. Pick **one** path.

### 1a. Fastest start — download the prepared package (no conversion)

The `*-Splash*` repos hold the engine's native packed format (`manifest.json`,
`target/`, `draft/`, `tokenizer/`), which the server loads directly.

```zsh
hf download nitinpanj/Swift-Qwen3.8-Flash-Next-Splash \
    --repo-type model --local-dir ~/models/swift-qwen38-flash-next-v3/prepared
```

> **Current state of that repo (check before relying on it):** it is the **Sept-24
> build** and holds only `target/layer-0..19`. The model this Mac serves is the V3
> build with **48** layers plus `ngram.bin`, `mtp-layer.bin` and `tokenizer/`
> (44 files, 87.5 GB, are not on the hub as of this writing). Until that upload
> lands, use **1b** — it needs no conversion and is complete. `manifest.json` and
> `draft/*` are byte-identical between the two builds (SHA-256 verified).

### 1b. Download the GGUF and let the launcher convert it (recommended, works today)

```zsh
# With Swift
hf download nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF \
    --repo-type model --local-dir ~/models/swift-qwen38-flash-next-v3

# Without Swift
hf download nitinpanj/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF \
    --repo-type model --local-dir ~/models/qwen38-flash-next-v3
```

The GGUFs are multi-shard (`*-00001-of-00003.gguf` …). The launcher auto-converts
on first `serve` (`install/launcher.py`: a directory containing `*.gguf` is prepared
into `<dir>/prepared/` by `models.qwen4exp.tools.convert_qwen4exp_gguf`). Expect a
one-time ~6–7 minute conversion plus ~100 GB of disk.

### 1c. Reproduce the Swift V3 weights yourself (not a download of the final file)

```zsh
python3 dev/tools/build_swift_v3_gguf.py
```

Streams the Q8_0 donor tensors by HTTP range request from
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` and splices 686 resident tensors
(`output`, `token_embd`, `attn_*`, `hc_*`, `ssm_out`, `shexp`) into the Q4_0 base.
Peak temp disk 42 GiB, output ~95.5 GiB. The `Swift-*-v3-ds4.gguf` file on this
machine (236 GB, single file) is a **denser local quant**: it is not derived from
the Q4_0-Q8out shards and is not on the hub.

---

## 2. Serve it

One command; `--model` takes a **directory** or single **`.gguf` file**, or HF repo id.

### Flash-Next V3 (125.7B MoE)

```zsh
cd ~/Documents/shared-with-google-drive/model-serving/slipstream

# WITH Swift (KV-sparse)
./slipstream serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090

# WITHOUT Swift (plain base)
./slipstream serve --model ~/models/qwen38-flash-next-v3 --port 8090
```

### Swift-Qwen3.8-27B (Dense Hybrid)

```zsh
# Prepared 27B package (resident in RAM, ultra-fast 570–660 ms TTFT):
./slipstream serve --model ~/models/swift-qwen38-27b-splash-hq --port 8090

# Or serve a single 27B .gguf file directly:
./slipstream serve --model ~/models/qwen38-27b/qwen3.8-27b-q4_0.gguf --port 8090

# Or serve an MLX directory:
./slipstream serve --model ~/models/mlx-qwen38-27b --port 8090
```

Ways to point at a model (`install/launcher.py` `serve()`):

| `--model` value | What the launcher does |
|---|---|
| single `.gguf` file or dir with `*.gguf` | converts to `<dir>/prepared/` with APFS hole punching if missing, then serves it |
| dir containing `manifest.json` | serves it directly (pre-converted package) |
| dir containing `config.json` + `*.safetensors` | converts MLX checkpoint to package format and serves it |
| HF repo id (e.g. `nitinpanj/...`) | downloads via `install/models.py`, prepares in-place, serves |

Model id the server advertises is `local/<dir-name>` — e.g. `local/swift-qwen38-flash-next-v3`
or `local/swift-qwen38-27b-splash-hq`. The server speaks OpenAI **and** Anthropic APIs on that port.

`--max-memory`, `--max-context`, `--api-key`, `--no-webui`, `--allowed-host`,
`--max-image-pixels` are the other real `serve` flags. Env knobs that matter
(`SPLASH_` prefix, all optional, output-identical):

| Setting | Default | Effect |
|---|---|---|
| `SPLASH_NO_MTP=1` | off | Disable the MTP draft head: one token per step, much slower. Use this to measure what Swift's sparsity alone buys |
| `SPLASH_EXPERT_CACHE_GIB` | 32 | Experts kept in RAM; the rest stream from SSD. Set `30` if you skip the sudo step |
| `SPLASH_MTP_CHAIN_MIN` | `0.35` | Stop drafting when proposal confidence drops below this |
| `SPLASH_LOOKAHEAD_EXPERTS` | `6` | Experts per row read ahead from a prediction (5–8 tie) |
| `SPLASH_TREE_DRAFT=1` | on in launcher scripts | Tree speculation |

Watch and stop:

```zsh
~/models/bin/slipstream-log.sh          # one line per request with tok/s
curl -s localhost:8090/status | python3 -c "import json,sys;d=json.load(sys.stdin);print('ready',d['ready'],'| memory',d['memory_pressure'])"
~/models/bin/slipstream-stop.sh
```

---

## 3. Use it from pi / omp / curl

`~/.pi/agent/models.json` needs `enabledModels: ["slipstream/**"]` — the `**`, not
`*`, or slash-nested ids like `local/swift-qwen38-flash-next-v3` never match.

```zsh
# pi, with Swift
pi --model slipstream/local/swift-qwen38-flash-next-v3 --thinking xhigh

# pi, without Swift
pi --model slipstream/local/qwen38-flash-next-v3 --thinking xhigh

# quick connectivity check (expect exactly: OK)
pi --model slipstream/local/swift-qwen38-flash-next-v3 -p "Reply with exactly: OK" </dev/null

# curl, the model id must match the served directory name
curl -s localhost:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "local/swift-qwen38-flash-next-v3",
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 200, "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

Thinking levels for this template: `off`, `low`, `medium`, `xhigh`. Context:
126,976 tokens (4K under the server's 131,072) — lower `contextWindow` in
`models.json` if you start with a smaller context.

---

## 4. Verify a download is the model you think it is

`manifest.json` fingerprints the package; the layer bins are the weights. Compare
local against the hub's LFS SHA-256 (the hub exposes them via `?blobs=true`):

```zsh
python3 - <<'PY'
# verified: prints match=1 differ=20 not-on-hub=44 against the Sept-24 archive repo
import json,os,hashlib,urllib.request
repo="nitinpanj/Swift-Qwen3.8-Flash-Next-Splash"   # repo to compare against
base=os.path.expanduser("~/models/swift-qwen38-flash-next-v3/prepared")
tok=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
api=lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u,headers={"Authorization":"Bearer "+tok})))
lfs={f["rfilename"]:(f.get("lfs") or {}).get("sha256") for f in api(f"https://huggingface.co/api/models/{repo}?blobs=true")["siblings"] if f.get("lfs")}
def sh(p,h=hashlib.sha256()):
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<26),b""): h.update(b)
    return h.hexdigest()
ok=bad=miss=0
for r,_,fs in os.walk(base):
    for fn in sorted(fs):
        p=os.path.join(r,fn); k=os.path.relpath(p,base)
        if k not in lfs: miss+=1; continue
        good=sh(p)==lfs[k]; ok+=good; bad+=not good
        print(("OK   " if good else "DIFF ")+k)
print(f"match={ok} differ={bad} not-on-hub={miss}")
PY
```

`OK` on every line means your local weights are the hub's weights, byte for byte.
**Matching file sizes are not proof** — the Sept-24 hub build matches all 21 file
sizes and differs on 14 SHA-256s. Always compare hashes.

---

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `127.0.0.1:8090 is in use` | Another engine is running. `lsof -i :8090`, `pgrep -fl generate-sample`, stop it. One big model at a time |
| Server refuses to start at `CACHE_GIB=32` | You skipped the sudo GPU-limit step; start with `SPLASH_EXPERT_CACHE_GIB=30` |
| `Directory ... does not contain GGUF files or a manifest.json` | You pointed `--model` at a parent dir. Point it at the dir holding `*.gguf` **or** the dir holding `manifest.json` |
| pi does not list the model | `enabledModels` needs `"slipstream/**"`; id must be `local/<dir-name>` |
| Downloaded package won't load | The repo is the 20-layer Sept-24 archive. Use §1b, or check §4's hash report |
| `guarded: killed ...` in the log | Memory guard stopped the server below 4 GiB free. Close apps, lower `CTX` |
| Slow decode after an upload/copy ran | SSD saturation, not the model. Check `decode ... tok/s` in `slipstream-log.sh` after the other job exits |

**Benchmarks in-repo** (both take `--model-dir`, so they do Swift-vs-base comparisons):

```zsh
python3 dev/benchmarks/compare_slipstream_vs_llamacpp.py --model-dir ~/models/swift-qwen38-flash-next-v3
python3 dev/benchmarks/benchmark_diverse_workloads.py --model local/swift-qwen38-flash-next-v3
```
