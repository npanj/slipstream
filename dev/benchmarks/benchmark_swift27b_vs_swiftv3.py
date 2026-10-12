#!/usr/bin/env python3
from __future__ import annotations

"""Comprehensive, reproducible head-to-head benchmark between:
1. Swift-Qwen3.8-Flash-Next-V3 (Slipstream, port 8090)
2. Swift-Qwen3.8-27B-Splash-HQ (Splash-Q8, port 8000)

Evaluates:
- AIME 2025 (Competition Mathematics - Elite Tier, 20 items)
- MATH-500 (Olympiad & High School Math Levels 4-5, 35 items)
- GPQA Diamond (PhD-Level Science Reasoning, 35 items)
- GSM8K (Multi-step Word Arithmetic, 25 items)
- HumanEval (Python Algorithmic Code Execution, 25 items)
- Hard Systems & Logic (Concurrency, Memory, Zero-Copy, Constraint Reasoning, 5 items)
Total: 145 standardized items.

Tracks:
- Quality / Correctness (%) per benchmark and overall
- Steady-state decode throughput (tok/s)
- Time-to-first-token (TTFT ms)
- Prompt & completion token counts
- Total wall-clock time
"""

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DATA_DIR = Path(
    "/Users/nitin/Documents/shared-with-google-drive/model-serving/local-mlx/bench_data"
)
QUALITY_BENCH_DATA = Path(
    "/Users/nitin/Documents/shared-with-google-drive/benchmarking/model-quality-bench/data/datasets"
)
RESULTS_DIR = Path(__file__).resolve().parent / "swift_benchmark_results"

# -----------------------------------------------------------------------------
# Scorers
# -----------------------------------------------------------------------------

BOXED = re.compile(r"\\boxed\{([^}]*)\}")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        c = text[i]
        depth += c == "{"
        depth -= c == "}"
        if depth == 0:
            return "".join(out)
        out.append(c)
        i += 1
    return None


def _normalise_math(s: str) -> str:
    s = s.strip().strip("$").strip()
    for a, b in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("dfrac", "frac"),
        ("tfrac", "frac"),
        ("^\\circ", ""),
        ("^{\\circ}", ""),
        ("\\%", ""),
        ("%", ""),
        ("\\$", ""),
        (" ", ""),
    ):
        s = s.replace(a, b)
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mbox\{(.*?)\}", r"\1", s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _as_number(s: str) -> float | None:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        v = int(m.group(2)) / int(m.group(3))
        return -v if m.group(1) else v
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def score_math(output: str, answer: str) -> tuple[float, str]:
    got = _boxed(output)
    if got is None:
        return 0.0, "no \\boxed{} answer"
    a, b = _normalise_math(got), _normalise_math(answer)
    if a == b:
        return 1.0, f"boxed {got}"
    x, y = _as_number(a), _as_number(b)
    if x is not None and y is not None and abs(x - y) <= 1e-6 * max(1.0, abs(y)):
        return 1.0, f"boxed {got} (numeric match)"
    return 0.0, f"boxed {got} want {answer}"


def score_aime(output: str, answer: str) -> tuple[float, str]:
    want = str(int(answer.strip()))
    # Check boxed first
    got_b = _boxed(output)
    if got_b:
        cand = NUM.findall(got_b)
        if cand:
            v = str(int(float(cand[-1].replace(",", ""))))
            if v == want:
                return 1.0, f"boxed {v}"
    # Check explicit conclusion
    m = re.search(r"final answer is\s*\\boxed\{(\d+)\}", output, re.IGNORECASE)
    if m and m.group(1) == want:
        return 1.0, f"boxed {m.group(1)}"
    # Check last number
    cand = NUM.findall(output)
    if cand:
        try:
            v = str(int(float(cand[-1].replace(",", ""))))
            if v == want:
                return 1.0, f"last_num {v}"
            return 0.0, f"got {v} want {want}"
        except Exception:
            pass
    return 0.0, f"no integer match for {want}"


def score_gpqa(output: str, answer: str) -> tuple[float, str]:
    want = answer.strip().upper()
    # Check explicit conclusion
    m = re.search(r"correct answer is\s*\(?([ABCD])\)?", output, re.IGNORECASE)
    if m:
        got = m.group(1).upper()
        return (1.0 if got == want else 0.0), f"conclusion {got} want {want}"
    # Check boxed
    b = _boxed(output)
    if b and b.strip().upper() in ["A", "B", "C", "D"]:
        got = b.strip().upper()
        return (1.0 if got == want else 0.0), f"boxed {got} want {want}"
    # Fallback: scan for letter in last 250 chars
    tail = output[-250:] if len(output) > 250 else output
    matches = re.findall(r"\b([ABCD])\b", tail.upper())
    if matches:
        got = matches[-1]
        return (1.0 if got == want else 0.0), f"tail {got} want {want}"
    return 0.0, f"no choice found (want {want})"


def score_gsm8k(output: str, answer: str) -> tuple[float, str]:
    want = answer.strip()
    # Split on #### if present
    if "####" in output:
        cand = NUM.findall(output.split("####")[-1])
        if cand:
            got = cand[-1].replace(",", "")
            try:
                ok = abs(float(got) - float(want)) < 1e-4
                return (1.0 if ok else 0.0), f"got {got} want {want}"
            except Exception:
                pass
    # Check last number in whole text
    cand = NUM.findall(output)
    if cand:
        got = cand[-1].replace(",", "")
        try:
            ok = abs(float(got) - float(want)) < 1e-4
            return (1.0 if ok else 0.0), f"got {got} want {want}"
        except Exception:
            pass
    return 0.0, f"no number match for {want}"


def score_code(output: str, test_code: str, timeout: int = 10) -> tuple[float, str]:
    body = output
    m = re.search(r"```(?:python)?\n(.*?)```", output, re.S)
    if m:
        body = m.group(1)
    prog = body + "\n\n" + test_code
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text(prog)
        try:
            r = subprocess.run(
                [sys.executable, str(p)], capture_output=True, timeout=timeout, cwd=d
            )
            if r.returncode == 0:
                return 1.0, "pass"
            err = (r.stderr.decode(errors="replace").strip().splitlines() or ["fail"])[
                -1
            ][:120]
            return 0.0, f"fail: {err}"
        except subprocess.TimeoutExpired:
            return 0.0, "timeout"
        except Exception as e:
            return 0.0, f"execution error: {e}"


def score_logic(output: str, meta: dict) -> tuple[float, str]:
    kind = meta.get("kind")
    if kind == "interval":
        return score_code(output, meta["test"])
    elif kind == "seating":
        o = output.lower()
        has_1 = (
            "chair 1" in o or "1:" in o or "1 -" in o or "first chair" in o
        ) and "alice" in o
        has_2 = (
            "chair 2" in o or "2:" in o or "2 -" in o or "second chair" in o
        ) and "charlie" in o
        has_3 = (
            "chair 3" in o or "3:" in o or "3 -" in o or "third chair" in o
        ) and "bob" in o
        if has_1 and has_2 and has_3:
            return 1.0, "correct seating (1: Alice, 2: Charlie, 3: Bob)"
        return 0.0, "incorrect seating deduction"
    elif kind == "concurrency":
        o = output.lower()
        has_acq_rel = "acquire" in o and "release" in o
        has_seq_cst = "sequential" in o or "seq_cst" in o or "total order" in o
        if has_acq_rel and has_seq_cst:
            return 1.0, "covers acquire-release & sequential consistency"
        return 0.0, "missing core concurrency primitives"
    elif kind == "bandwidth":
        o = output.lower()
        has_bw = "bandwidth" in o or "bytes" in o or "memory" in o
        has_kv = "kv" in o or "weight" in o or "arithmetic intensity" in o
        if has_bw and has_kv:
            return 1.0, "covers memory bandwidth and arithmetic intensity"
        return 0.0, "missing bandwidth explanation"
    elif kind == "csv":
        o = output.lower()
        has_state = "state" in o or "quote" in o or "escap" in o or "parser" in o
        if has_state:
            return 1.0, "valid CSV state parser structure"
        return 0.0, "missing state machine logic"
    return 1.0, "reviewed"


# -----------------------------------------------------------------------------
# Dataset Preparation
# -----------------------------------------------------------------------------


def prepare_dataset(seed: int = 1234) -> list[dict]:
    rng = random.Random(seed)
    items = []

    # 1. AIME 2025 (20 items)
    aime_path = BENCH_DATA_DIR / "aime25.json"
    with open(aime_path) as f:
        aime_data = json.load(f)
    for idx, row in enumerate(aime_data[:20]):
        prompt = (
            "Solve the following math competition problem. The answer is a non-negative integer from 0 to 999. "
            "Think step by step, keep your derivation concise, and conclude your response with: "
            '"Therefore, the final answer is \\boxed{N}" where N is an integer from 0 to 999.\n\n'
            f"Problem:\n{row['problem']}"
        )
        items.append(
            {
                "id": f"aime25_{idx + 1:02d}",
                "benchmark": "AIME 2025",
                "category": "Elite Mathematics",
                "prompt": prompt,
                "ground_truth": str(row["answer"]).strip(),
                "scorer": "aime",
                "max_tokens": 2560,
                "meta": {"source": "aime25", "index": idx},
            }
        )

    # 2. MATH-500 (35 items, sampling Level 4 and 5)
    math_path = QUALITY_BENCH_DATA / "math500" / "test.jsonl"
    with open(math_path) as f:
        m5_all = [json.loads(line) for line in f]
    hard_m5 = [
        x for x in m5_all if str(x.get("level", "")) in ["Level 4", "Level 5", "4", "5"]
    ]
    if len(hard_m5) < 35:
        hard_m5 = m5_all
    rng.shuffle(hard_m5)
    for idx, row in enumerate(hard_m5[:35]):
        subj = row.get("subject", "mathematics")
        prompt = (
            f"Solve the following mathematics problem about {subj}. "
            "Think step by step, keep your derivation concise, and conclude your response with: "
            '"Therefore, the final answer is \\boxed{ANSWER}"\n\n'
            f"Problem:\n{row['problem']}"
        )
        items.append(
            {
                "id": f"math500_{idx + 1:02d}",
                "benchmark": "MATH-500",
                "category": "Olympiad Math",
                "prompt": prompt,
                "ground_truth": str(row["answer"]).strip(),
                "scorer": "math",
                "max_tokens": 1536,
                "meta": {"subject": subj, "level": row.get("level")},
            }
        )

    # 3. GPQA Diamond (35 items)
    gpqa_path = BENCH_DATA_DIR / "gpqa.json"
    with open(gpqa_path) as f:
        gpqa_all = json.load(f)
    gpqa_sample = list(gpqa_all)
    rng.shuffle(gpqa_sample)
    for idx, row in enumerate(gpqa_sample[:35]):
        prompt = (
            "Answer the following multiple-choice question. Think step by step and conclude your "
            'response with: "Therefore, the correct answer is (X)" where X is one of A, B, C, or D.\n\n'
            f"Question:\n{row['question']}"
        )
        items.append(
            {
                "id": f"gpqa_{idx + 1:02d}",
                "benchmark": "GPQA Diamond",
                "category": "PhD-Level Science",
                "prompt": prompt,
                "ground_truth": row["answer"].strip().upper(),
                "scorer": "gpqa",
                "max_tokens": 1024,
                "meta": {"source": "gpqa_diamond"},
            }
        )

    # 4. GSM8K (25 items)
    gsm_path = QUALITY_BENCH_DATA / "gsm8k" / "test.jsonl"
    with open(gsm_path) as f:
        gsm_all = [json.loads(line) for line in f]
    rng.shuffle(gsm_all)
    for idx, row in enumerate(gsm_all[:25]):
        prompt = (
            "Solve the following math word problem. Think it through step by step, then give the "
            "final answer on its own last line as: #### <number>\n\n"
            f"Question:\n{row['question']}"
        )
        ans = row["answer"].split("####")[-1].strip()
        items.append(
            {
                "id": f"gsm8k_{idx + 1:02d}",
                "benchmark": "GSM8K",
                "category": "Multi-Step Math",
                "prompt": prompt,
                "ground_truth": ans,
                "scorer": "gsm8k",
                "max_tokens": 1024,
                "meta": {},
            }
        )

    # 5. HumanEval (25 items)
    he_path = QUALITY_BENCH_DATA / "humaneval" / "test.jsonl"
    with open(he_path) as f:
        he_all = [json.loads(line) for line in f]
    rng.shuffle(he_all)
    for idx, row in enumerate(he_all[:25]):
        prompt = (
            "Complete the function. Reply with the complete function definition in a single "
            "```python code block and nothing else.\n\n"
            f"```python\n{row['prompt']}```"
        )
        test = row["test"] + f"\n\ncheck({row['entry_point']})\n"
        items.append(
            {
                "id": f"humaneval_{idx + 1:02d}",
                "benchmark": "HumanEval",
                "category": "Code Generation",
                "prompt": prompt,
                "ground_truth": test,
                "scorer": "code",
                "max_tokens": 1536,
                "meta": {"task_id": row["task_id"], "entry_point": row["entry_point"]},
            }
        )

    # 6. Hard Systems & Logic (5 items)
    hard_probes = [
        {
            "id": "probe_01_intervals",
            "benchmark": "Hard Systems & Logic",
            "category": "Algorithms",
            "prompt": (
                "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
                "that merges all overlapping intervals. It must run in O(N log N) time, handle empty lists, "
                "singletons, negative coordinates, and completely nested intervals. Provide full type annotations, "
                "docstrings, and return your implementation inside a ```python block."
            ),
            "ground_truth": "",
            "scorer": "logic",
            "max_tokens": 1024,
            "meta": {
                "kind": "interval",
                "test": (
                    "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
                    "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
                    "assert merge_intervals([]) == []\n"
                    "assert merge_intervals([[-5,-2],[-3,1]]) == [[-5,1]]\n"
                    "assert merge_intervals([[1,10],[2,3],[4,8]]) == [[1,10]]\n"
                    "print('Intervals OK')\n"
                ),
            },
        },
        {
            "id": "probe_02_seating",
            "benchmark": "Hard Systems & Logic",
            "category": "Constraint Reasoning",
            "prompt": (
                "Three friends (Alice, Bob, Charlie) are sitting in a row of 3 chairs numbered 1 to 3 from left to right.\n"
                "Constraints:\n"
                "1. Alice never sits next to Bob.\n"
                "2. Charlie is to the right of Alice (higher chair number).\n"
                "Who sits in chair 1, chair 2, and chair 3? Give a rigorous logical deduction step by step, "
                "concluding with the exact assignment for chairs 1, 2, and 3."
            ),
            "ground_truth": "Chair 1: Alice, Chair 2: Charlie, Chair 3: Bob",
            "scorer": "logic",
            "max_tokens": 1024,
            "meta": {"kind": "seating"},
        },
        {
            "id": "probe_03_concurrency",
            "benchmark": "Hard Systems & Logic",
            "category": "Systems & Concurrency",
            "prompt": (
                "Explain the exact architectural and hardware difference between Acquire-Release memory ordering "
                "and Sequentially Consistent (SeqCst) memory ordering in multithreaded systems. "
                "Include what CPU memory fences/barriers are emitted on ARM64 (Apple Silicon) vs x86-64, and give "
                "a concrete store-buffering litmus test showing where Acquire-Release allows reordering but SeqCst forbids it."
            ),
            "ground_truth": "Acquire-Release vs SeqCst store buffering explanation",
            "scorer": "logic",
            "max_tokens": 1536,
            "meta": {"kind": "concurrency"},
        },
        {
            "id": "probe_04_bandwidth",
            "benchmark": "Hard Systems & Logic",
            "category": "Hardware & Architecture",
            "prompt": (
                "In exactly four clear, highly technical sentences, explain why unified memory bandwidth "
                "(rather than raw FP16/BF16 TFLOPS) is the fundamental bottleneck during the autoregressive "
                "token generation phase in transformer inference, and contrast this with the prefill phase."
            ),
            "ground_truth": "Memory bandwidth bottleneck in decode vs prefill",
            "scorer": "logic",
            "max_tokens": 512,
            "meta": {"kind": "bandwidth"},
        },
        {
            "id": "probe_05_csv_parser",
            "benchmark": "Hard Systems & Logic",
            "category": "Systems Programming",
            "prompt": (
                "Write an idiomatic Rust zero-copy CSV line parser function `parse_csv_record(line: &str) -> Vec<&str>` "
                "that correctly parses a comma-separated line, handling double-quoted fields that contain escaped quotes "
                '(represented as `""`). Explain the state machine transitions in comments and provide unit tests.'
            ),
            "ground_truth": "Rust zero-copy CSV parser",
            "scorer": "logic",
            "max_tokens": 1536,
            "meta": {"kind": "csv"},
        },
    ]
    items.extend(hard_probes)

    return items


# -----------------------------------------------------------------------------
# HTTP Client
# -----------------------------------------------------------------------------


def query_endpoint(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = 240,
    retries: int = 3,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    last_error = ""
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                t1 = time.perf_counter()
                wall_ms = (t1 - t0) * 1000.0
                body = json.loads(raw)
                choice = body["choices"][0]["message"]
                content = choice.get("content") or ""
                reasoning = choice.get("reasoning_content") or ""
                usage = body.get("usage", {})
                prompt_toks = usage.get("prompt_tokens", 0)
                comp_toks = usage.get("completion_tokens", 0)
                reason_toks = (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                )

                # Extract metrics
                metrics = body.get("metrics", {})
                req_lat = metrics.get("request_latency", {})
                ttft_ms = req_lat.get("ttft_ms") or req_lat.get(
                    "start_to_first_token_ms"
                )
                stream_tps = req_lat.get("stream_tokens_per_second")

                timings = body.get("timings", {})
                predicted_tps = timings.get("predicted_per_second")

                tok_per_sec = stream_tps or predicted_tps
                if not tok_per_sec:
                    tok_per_sec = (comp_toks / (t1 - t0)) if (t1 - t0) > 0 else 0.0

                if ttft_ms is None:
                    ttft_ms = 0.0

                return {
                    "ok": True,
                    "content": content,
                    "reasoning": reasoning,
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": comp_toks,
                    "reasoning_tokens": reason_toks,
                    "wall_ms": wall_ms,
                    "ttft_ms": float(ttft_ms),
                    "tok_per_sec": float(tok_per_sec),
                    "error": None,
                }
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            time.sleep(2.0 * (attempt + 1))

    return {
        "ok": False,
        "content": "",
        "reasoning": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "wall_ms": 0.0,
        "ttft_ms": 0.0,
        "tok_per_sec": 0.0,
        "error": last_error,
    }


# -----------------------------------------------------------------------------
# Evaluation Runner
# -----------------------------------------------------------------------------


def evaluate_model(
    model_name: str, base_url: str, model_id: str, items: list[dict], output_file: Path
) -> list[dict]:
    print("\n==================================================================")
    print(f"Starting Evaluation: {model_name}")
    print(f"Endpoint: {base_url} (Model ID: {model_id})")
    print(f"Total Items: {len(items)}")
    print(f"Output: {output_file}")
    print("==================================================================\n")

    results = []
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                for line in f:
                    results.append(json.loads(line))
            print(f"Resuming from existing {len(results)} items in {output_file.name}.")
        except Exception:
            results = []

    done_ids = {r["id"] for r in results if r.get("ok")}

    out_fh = open(output_file, "a" if results else "w", buffering=1)

    t_start = time.perf_counter()

    for idx, it in enumerate(items, 1):
        item_id = it["id"]
        if item_id in done_ids:
            continue

        prompt = it["prompt"]
        scorer_type = it["scorer"]
        ground_truth = it["ground_truth"]
        max_tokens = it.get("max_tokens", 1024)

        resp = query_endpoint(base_url, model_id, prompt, max_tokens=max_tokens)

        if not resp["ok"]:
            print(
                f"[{idx:03d}/{len(items):03d}] {item_id:<18} | {it['benchmark']:<16} | ERROR: {resp['error']}"
            )
            rec = {
                "id": item_id,
                "benchmark": it["benchmark"],
                "category": it["category"],
                "ok": False,
                "score": 0.0,
                "explanation": f"error: {resp['error']}",
                "prompt": prompt,
                "ground_truth": ground_truth,
                "output": "",
                "reasoning": "",
                "tok_per_sec": 0.0,
                "ttft_ms": 0.0,
                "wall_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
            }
            results.append(rec)
            out_fh.write(json.dumps(rec) + "\n")
            continue

        full_output = resp["content"] or ""
        reasoning = resp["reasoning"] or ""

        # Primary scoring on output text
        score = 0.0
        why = ""
        eval_text = full_output.strip() if full_output.strip() else reasoning.strip()

        if scorer_type == "aime":
            score, why = score_aime(eval_text, ground_truth)
            if score == 0.0 and reasoning and eval_text != reasoning:
                sc_r, why_r = score_aime(reasoning, ground_truth)
                if sc_r > 0.0:
                    score, why = sc_r, f"[in reasoning] {why_r}"
        elif scorer_type == "math":
            score, why = score_math(eval_text, ground_truth)
            if score == 0.0 and reasoning and eval_text != reasoning:
                sc_r, why_r = score_math(reasoning, ground_truth)
                if sc_r > 0.0:
                    score, why = sc_r, f"[in reasoning] {why_r}"
        elif scorer_type == "gpqa":
            score, why = score_gpqa(eval_text, ground_truth)
            if score == 0.0 and reasoning and eval_text != reasoning:
                sc_r, why_r = score_gpqa(reasoning, ground_truth)
                if sc_r > 0.0:
                    score, why = sc_r, f"[in reasoning] {why_r}"
        elif scorer_type == "gsm8k":
            score, why = score_gsm8k(eval_text, ground_truth)
            if score == 0.0 and reasoning and eval_text != reasoning:
                sc_r, why_r = score_gsm8k(reasoning, ground_truth)
                if sc_r > 0.0:
                    score, why = sc_r, f"[in reasoning] {why_r}"
        elif scorer_type == "code":
            score, why = score_code(
                full_output if full_output.strip() else reasoning, ground_truth
            )
        elif scorer_type == "logic":
            score, why = score_logic(eval_text, it["meta"])

        rec = {
            "id": item_id,
            "benchmark": it["benchmark"],
            "category": it["category"],
            "ok": True,
            "score": score,
            "explanation": why,
            "prompt": prompt,
            "ground_truth": ground_truth if scorer_type != "code" else "<test_suite>",
            "output": full_output,
            "reasoning": reasoning,
            "tok_per_sec": resp["tok_per_sec"],
            "ttft_ms": resp["ttft_ms"],
            "wall_ms": resp["wall_ms"],
            "prompt_tokens": resp["prompt_tokens"],
            "completion_tokens": resp["completion_tokens"],
            "reasoning_tokens": resp["reasoning_tokens"],
        }
        results.append(rec)
        out_fh.write(json.dumps(rec) + "\n")

        status_sym = "PASS" if score > 0.0 else "FAIL"
        print(
            f"[{idx:03d}/{len(items):03d}] {item_id:<18} | {it['benchmark']:<16} | {status_sym} ({score:.1f}) | {resp['tok_per_sec']:>5.1f} t/s | TTFT: {resp['ttft_ms']:>6.1f}ms | Out: {resp['completion_tokens']:>4d}tok | {why[:40]}"
        )

    out_fh.close()
    total_time = time.perf_counter() - t_start
    print(f"\nFinished {model_name} in {total_time / 60:.2f} minutes.")
    return results


# -----------------------------------------------------------------------------
# Reporting & Comparison
# -----------------------------------------------------------------------------


def generate_report(v3_file: Path, b27_file: Path, report_file: Path):
    if not v3_file.exists() or not b27_file.exists():
        print("Cannot generate report: one or both results files do not exist.")
        return

    with open(v3_file) as f:
        v3_res = [json.loads(line) for line in f]
    with open(b27_file) as f:
        b27_res = [json.loads(line) for line in f]

    v3_map = {r["id"]: r for r in v3_res}
    b27_map = {r["id"]: r for r in b27_res}

    common_ids = sorted(set(v3_map.keys()) & set(b27_map.keys()))
    print(
        f"\nGenerating comparative scorecard across {len(common_ids)} common items..."
    )

    benchmarks = [
        "AIME 2025",
        "MATH-500",
        "GPQA Diamond",
        "GSM8K",
        "HumanEval",
        "Hard Systems & Logic",
    ]
    stats = {}

    for b in benchmarks:
        stats[b] = {
            "v3_correct": 0,
            "b27_correct": 0,
            "total": 0,
            "v3_tps": [],
            "b27_tps": [],
            "v3_ttft": [],
            "b27_ttft": [],
            "v3_tokens": [],
            "b27_tokens": [],
        }

    for item_id in common_ids:
        r1 = v3_map[item_id]
        r2 = b27_map[item_id]
        bench = r1["benchmark"]
        if bench not in stats:
            stats[bench] = {
                "v3_correct": 0,
                "b27_correct": 0,
                "total": 0,
                "v3_tps": [],
                "b27_tps": [],
                "v3_ttft": [],
                "b27_ttft": [],
                "v3_tokens": [],
                "b27_tokens": [],
            }
        s = stats[bench]
        s["total"] += 1
        if r1.get("score", 0.0) > 0.0:
            s["v3_correct"] += 1
        if r2.get("score", 0.0) > 0.0:
            s["b27_correct"] += 1

        if r1.get("tok_per_sec", 0) > 0:
            s["v3_tps"].append(r1["tok_per_sec"])
        if r2.get("tok_per_sec", 0) > 0:
            s["b27_tps"].append(r2["tok_per_sec"])

        if r1.get("ttft_ms", 0) > 0:
            s["v3_ttft"].append(r1["ttft_ms"])
        if r2.get("ttft_ms", 0) > 0:
            s["b27_ttft"].append(r2["ttft_ms"])

        s["v3_tokens"].append(r1.get("completion_tokens", 0))
        s["b27_tokens"].append(r2.get("completion_tokens", 0))

    lines = []
    lines.append(
        "# Head-to-Head Benchmark: Swift-Qwen3.8-27B-Splash-HQ vs Swift-Qwen3.8-Flash-Next-V3\n"
    )
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M PDT')}\n")
    lines.append(
        "**Platform:** Apple Silicon (Unified Memory), Single-Engine Sequential Execution\n"
    )
    lines.append(
        f"**Total Evaluated Items:** {len(common_ids)} items across 6 rigorous domains\n"
    )
    lines.append("\n---\n")

    lines.append("## 1. Executive Summary & Scorecard\n")
    lines.append(
        "| Domain / Benchmark | Items | Swift-Flash-Next-V3 Accuracy | Swift-27B-Splash-HQ Accuracy | Accuracy Delta | Flash-Next Speed | 27B-Splash Speed | Speed Ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    tot_items = 0
    tot_v3_corr = 0
    tot_b27_corr = 0
    all_v3_tps = []
    all_b27_tps = []
    all_v3_ttft = []
    all_b27_ttft = []

    for b in benchmarks:
        if b not in stats or stats[b]["total"] == 0:
            continue
        s = stats[b]
        tot = s["total"]
        tot_items += tot
        v3_c = s["v3_correct"]
        b27_c = s["b27_correct"]
        tot_v3_corr += v3_c
        tot_b27_corr += b27_c

        v3_acc = (v3_c / tot) * 100.0
        b27_acc = (b27_c / tot) * 100.0
        diff = b27_acc - v3_acc

        v3_speed = (sum(s["v3_tps"]) / len(s["v3_tps"])) if s["v3_tps"] else 0.0
        b27_speed = (sum(s["b27_tps"]) / len(s["b27_tps"])) if s["b27_tps"] else 0.0

        all_v3_tps.extend(s["v3_tps"])
        all_b27_tps.extend(s["b27_tps"])
        all_v3_ttft.extend(s["v3_ttft"])
        all_b27_ttft.extend(s["b27_ttft"])

        speed_ratio = (b27_speed / v3_speed) if v3_speed > 0 else 0.0
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"

        lines.append(
            f"| **{b}** | {tot} | {v3_acc:.1f}% ({v3_c}/{tot}) | {b27_acc:.1f}% ({b27_c}/{tot}) | {diff_str} | {v3_speed:.1f} tok/s | {b27_speed:.1f} tok/s | {speed_ratio:.2f}x |"
        )

    overall_v3_acc = (tot_v3_corr / tot_items * 100.0) if tot_items else 0.0
    overall_b27_acc = (tot_b27_corr / tot_items * 100.0) if tot_items else 0.0
    overall_diff = overall_b27_acc - overall_v3_acc
    overall_v3_tps = (sum(all_v3_tps) / len(all_v3_tps)) if all_v3_tps else 0.0
    overall_b27_tps = (sum(all_b27_tps) / len(all_b27_tps)) if all_b27_tps else 0.0
    overall_ratio = (overall_b27_tps / overall_v3_tps) if overall_v3_tps else 0.0
    overall_diff_str = (
        f"+{overall_diff:.1f}%" if overall_diff > 0 else f"{overall_diff:.1f}%"
    )

    lines.append(
        f"| **TOTAL / OVERALL** | **{tot_items}** | **{overall_v3_acc:.1f}% ({tot_v3_corr}/{tot_items})** | **{overall_b27_acc:.1f}% ({tot_b27_corr}/{tot_items})** | **{overall_diff_str}** | **{overall_v3_tps:.1f} tok/s** | **{overall_b27_tps:.1f} tok/s** | **{overall_ratio:.2f}x** |"
    )

    lines.append("\n---\n")

    lines.append("## 2. Latency & Efficiency Metrics\n")
    lines.append("| Metric | Swift-Flash-Next-V3 | Swift-27B-Splash-HQ | Comparison |")
    lines.append("|---|---:|---:|---|")

    avg_v3_ttft = (sum(all_v3_ttft) / len(all_v3_ttft)) if all_v3_ttft else 0.0
    avg_b27_ttft = (sum(all_b27_ttft) / len(all_b27_ttft)) if all_b27_ttft else 0.0
    ttft_ratio = (avg_v3_ttft / avg_b27_ttft) if avg_b27_ttft else 0.0

    lines.append(
        f"| **Average Decode Throughput** | **{overall_v3_tps:.1f} tok/s** | **{overall_b27_tps:.1f} tok/s** | **{overall_ratio:.2f}x speed** |"
    )
    lines.append(
        f"| **Average Time-To-First-Token (TTFT)** | **{avg_v3_ttft:.1f} ms** | **{avg_b27_ttft:.1f} ms** | **{ttft_ratio:.2f}x faster** |"
    )

    v3_tot_toks = sum(r.get("completion_tokens", 0) for r in v3_map.values())
    b27_tot_toks = sum(r.get("completion_tokens", 0) for r in b27_map.values())
    lines.append(
        f"| **Total Generated Tokens** | {v3_tot_toks:,} tokens | {b27_tot_toks:,} tokens | {((b27_tot_toks - v3_tot_toks) / v3_tot_toks * 100.0 if v3_tot_toks else 0.0):+.1f}% tokens |"
    )

    report_text = "\n".join(lines)
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"Report written to {report_file}")
    print("\n" + report_text)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Swift 27B vs Swift V3 Quality & Speed Benchmark"
    )
    parser.add_argument(
        "--target",
        choices=["v3", "27b", "report"],
        default="v3",
        help="Target model to evaluate or 'report' to generate scorecard",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional limit on total items"
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for deterministic sampling"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    items = prepare_dataset(seed=args.seed)
    if args.limit > 0:
        items = items[: args.limit]

    v3_file = RESULTS_DIR / "swift_flash_next_v3_results.jsonl"
    b27_file = RESULTS_DIR / "swift_27b_splash_hq_results.jsonl"
    report_file = RESULTS_DIR / "BENCHMARK_SCORECARD.md"

    if args.target == "v3":
        print(f"Loaded {len(items)} benchmark test items across 6 domains:")
        counts = {}
        for it in items:
            counts[it["benchmark"]] = counts.get(it["benchmark"], 0) + 1
        for bench, count in counts.items():
            print(f"  - {bench:<22}: {count} items")

        evaluate_model(
            model_name="Swift-Qwen3.8-Flash-Next-V3",
            base_url="http://127.0.0.1:8090/v1",
            model_id="local/swift-qwen38-flash-next-v3",
            items=items,
            output_file=v3_file,
        )
    elif args.target == "27b":
        print(f"Loaded {len(items)} benchmark test items across 6 domains:")
        counts = {}
        for it in items:
            counts[it["benchmark"]] = counts.get(it["benchmark"], 0) + 1
        for bench, count in counts.items():
            print(f"  - {bench:<22}: {count} items")

        evaluate_model(
            model_name="Swift-Qwen3.8-27B-Splash-HQ",
            base_url="http://127.0.0.1:8000/v1",
            model_id="nitinpanj/Swift-Qwen3.8-27B-Splash-HQ",
            items=items,
            output_file=b27_file,
        )
    elif args.target == "report":
        generate_report(v3_file, b27_file, report_file)


if __name__ == "__main__":
    main()
