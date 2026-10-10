#!/usr/bin/env python3
"""Drive the 27B Splash-HQ package through generate-sample and report
decode throughput and tokens per step (the proxy for speculation working).
The 10-prompt suite in models/qwen4exp/bench/speed_suite.py is Flash-Next
specific; this is the same measurement for the dense 27B and its DFlash drafter.

usage: bench27b_speed.py REPO_ROOT MODEL_ROOT [MAX_TOKENS]
"""

import json
import os
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
model = Path(sys.argv[2])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128
assert (repo / "build/engine-tests/generate-sample").exists(), "build the harness first"

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(model / "tokenizer"), local_files_only=True)

texts = {
    "palindrome": "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.",
    "ducks": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Think it through.",
    "explain": "Explain how a transformer's attention mechanism works to a software engineer who knows linear algebra but not machine learning.",
    "lru": "Implement an LRU cache class in Python with get and put in O(1), then write unit tests for it.",
}

prompts = {}
for name, text in texts.items():
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts[name] = tok.encode(rendered, add_special_tokens=False)

names = list(prompts)
joined = ";".join(",".join(map(str, prompts[n])) for n in names)

env = dict(os.environ)
for key in ("SPLASH_MAX_BATCH_WIDTH", "SPLASH_MTP_P_MIN", "SPLASH_MTP_CHAIN_MIN"):
    print(f"[env] {key}={env.get(key, '<unset>')}")

run = subprocess.run(
    [
        str(repo / "build/engine-tests/generate-sample"),
        str(repo / "build/splash.metallib"),
        str(model),
        str(max_tokens),
        joined,
    ],
    env=env,
    capture_output=True,
    text=True,
)
if run.returncode:
    print("ENGINE FAILED:", run.stderr[-3000:], file=sys.stderr)
    raise SystemExit(run.returncode)

rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
print(f"{max_tokens} tokens per prompt · {model.name}")
tot_tok = tot_ms = tot_steps = 0
for n, r in zip(names, rows):
    toks, ms, steps = len(r["generated_tokens"]), sum(r["step_ms"]), r["decode_steps"]
    tot_tok += toks
    tot_ms += ms
    tot_steps += steps
    print(
        f"  {n:11s} prompt {r['prompt_tokens']:5d} prefill {r['prefill_ms']:7.0f} ms  "
        f"{1000 * toks / ms:5.1f} tok/s  {toks / steps:4.2f} tok/step"
    )
print(
    f"TOTAL {1000 * tot_tok / tot_ms:.1f} tok/s  ({tot_tok} tokens, "
    f"{tot_tok / tot_steps:.2f} tok/step, {tot_ms / tot_steps:.1f} ms/step)"
)
