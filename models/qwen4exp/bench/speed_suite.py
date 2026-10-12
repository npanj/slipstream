#!/usr/bin/env python3
"""Decode speed over ten varied prompts on one warm engine.

One prompt is too noisy to judge a speed change: a small numeric difference
changes the generated text, and different text drafts differently (tokens per
step moved 3.66 <-> 4.17 between runs that were equal in quality). This sums
tokens and time over ten prompts of different kinds and lengths.

    .venv/bin/python models/qwen4exp/bench/speed_suite.py [--temperature 0.7] [--tokens 256]

Prints total tok/s (all tokens / all decode time), tokens per step, and each
prompt's rate. Prompt ids are built once into build/qwen4exp-speed-prompts.json.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROMPTS = ROOT / "build/qwen4exp-speed-prompts.json"


def model_path():
    env_dir = os.environ.get("SPLASH_MODEL_DIR", "").strip()
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    p = Path.home() / "models/swift-qwen38-flash-next-v3/prepared"
    if p.exists():
        return p
    return Path.home() / "models/qwen38-flash-next-splash"


TOKENIZER = model_path() / "tokenizer"


def texts():
    def src(*names):
        return "\n\n".join(
            f"// ===== {x} =====\n" + (ROOT / x).read_text() for x in names
        )

    return {
        "palindrome": "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.",
        "gsm8k": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Think it through.",
        "lru": "Implement an LRU cache class in Python with get and put in O(1), then write unit tests for it.",
        "explain": "Explain how a transformer's attention mechanism works to a software engineer who knows linear algebra but not machine learning.",
        "plan": "I'm migrating a Django monolith to services. Give me a step-by-step migration plan, with risks and how to test each step.",
        "sql": "Write a SQL query that finds, for each customer, their three most recent orders and the running total of spend, using window functions. Explain it briefly.",
        "rust": "Write a Rust function that parses a CSV line with quoted fields (commas inside quotes allowed) into a Vec<String>, with tests.",
        "review": src("runtime/ops/GDN.cpp")
        + "\n\nReview this code: explain what it does, then list any bugs you see.",
        "longcode": src(
            "runtime/ops/GDN.cpp",
            "runtime/ops/MoE.cpp",
            "runtime/model/StateLayout.hpp",
        )
        + "\n\nSummarise what each file does and how they fit together.",
        "summary": (ROOT / "README.md").read_text()
        + "\n\nSummarise this README in five bullet points for a new contributor.",
    }


def build():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    out = {}
    for name, text in texts().items():
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out[name] = tok.encode(rendered, add_special_tokens=False)
    PROMPTS.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--temperature", default="")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    prompts = (
        build()
        if a.rebuild or not PROMPTS.exists()
        else json.loads(PROMPTS.read_text())
    )
    names = list(prompts)
    joined = ";".join(",".join(map(str, prompts[n])) for n in names)
    env = dict(os.environ)
    if a.temperature:
        env["SPLASH_TEMPERATURE"] = a.temperature
    run = subprocess.run(
        [
            str(ROOT / "build/engine-tests/generate-sample"),
            str(ROOT / "build/splash.metallib"),
            str(model_path()),
            str(a.tokens),
            joined,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if run.returncode:
        raise SystemExit(run.stderr[-2000:])
    rows = [
        json.loads(line)
        for line in run.stdout.strip().splitlines()
        if line.startswith("{")
    ]
    tot_tok = tot_ms = tot_steps = 0
    print(f"T={a.temperature or 'greedy'}  {a.tokens} tokens per prompt")
    for n, r in zip(names, rows):
        toks, ms, steps = (
            len(r["generated_tokens"]),
            sum(r["step_ms"]),
            r["decode_steps"],
        )
        tot_tok, tot_ms, tot_steps = tot_tok + toks, tot_ms + ms, tot_steps + steps
        print(
            f"  {n:11s} prompt {r['prompt_tokens']:5d}  {1000 * toks / ms:5.1f} tok/s  {toks / steps:4.2f} tok/step"
        )
    print(
        f"TOTAL {1000 * tot_tok / tot_ms:.1f} tok/s  ({tot_tok} tokens, {tot_tok / tot_steps:.2f} tok/step, "
        f"{tot_ms / tot_steps:.1f} ms/step)"
    )


if __name__ == "__main__":
    main()
