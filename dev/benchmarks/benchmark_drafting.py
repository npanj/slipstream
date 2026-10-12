#!/usr/bin/env python3
"""Drafting Benchmark Harness:
Evaluates decode throughput, acceptance rate, and task quality across a balanced suite:
- 5 GSM8K word math problems
- 5 HumanEval Python programming problems
- 5 GPQA Diamond science questions
- 5 Hard Systems & Logic probes
Total: 20 standardized items.

Tracks:
- Pre/Post splash_drafted_tokens_total and splash_accepted_draft_tokens_total
- Acceptance ratio (%)
- Tokens generated per step
- Decode throughput (tok/s)
- Accuracy / Correctness (%)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dev/benchmarks"))
from benchmark_swift27b_vs_swiftv3 import (  # noqa: E402
    prepare_dataset,
    query_endpoint,
    score_aime,
    score_code,
    score_gpqa,
    score_gsm8k,
    score_logic,
    score_math,
)


def fetch_metrics(base_url: str = "http://127.0.0.1:8090/v1") -> dict:
    root_url = re.sub(r"/v1/?$", "", base_url)
    try:
        req = urllib.request.Request(f"{root_url}/metrics")
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8")
        m = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    m[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return m
    except Exception as e:
        print(
            f"Warning: failed to fetch /metrics from {root_url}: {e}", file=sys.stderr
        )
        return {}


def prepare_suite(seed: int = 1234, count_per_domain: int = 5) -> list[dict]:
    all_dataset = prepare_dataset(seed)
    domains = ["GSM8K", "HumanEval", "GPQA Diamond", "Hard Systems & Logic"]
    selected = []
    for d in domains:
        d_items = [it for it in all_dataset if it["benchmark"] == d]
        selected.extend(d_items[:count_per_domain])
    return selected


def run_benchmark(
    base_url: str = "http://127.0.0.1:8090/v1",
    model: str = "local/swift-qwen38-flash-next-v3",
    count_per_domain: int = 5,
) -> dict:
    suite = prepare_suite(count_per_domain=count_per_domain)
    print(f"=== Running Speculative Drafting Evaluation on {base_url} ({model}) ===")
    print(
        f"Items: {len(suite)} across {len(set(it['benchmark'] for it in suite))} domains"
    )

    m_start = fetch_metrics(base_url)
    t0_wall = time.perf_counter()

    results = []
    correct_count = 0
    total_tokens = 0
    decode_speeds = []

    for i, it in enumerate(suite):
        print(
            f"[{i + 1:02d}/{len(suite):02d}] {it['benchmark']:<20} ({it['id']})... ",
            end="",
            flush=True,
        )
        res = query_endpoint(
            url=base_url,
            model=model,
            prompt=it["prompt"],
            max_tokens=it["max_tokens"],
            temperature=0.0,
        )
        if not res["ok"]:
            print(f"FAILED ({res['error']})")
            results.append({"id": it["id"], "ok": False, "score": 0.0})
            continue

        full_output = res["content"] or ""
        reasoning = res["reasoning"] or ""
        eval_text = full_output.strip() if full_output.strip() else reasoning.strip()

        scorer_type = it["scorer"]
        ground_truth = it["ground_truth"]
        score = 0.0
        why = ""

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
            score, why = score_logic(eval_text, it.get("meta", {}))
        else:
            score, why = 1.0, "ok"

        correct_count += score >= 0.99
        total_tokens += res["completion_tokens"]
        decode_speeds.append(res["tok_per_sec"])
        status = "PASS" if score >= 0.99 else "FAIL"
        print(
            f"{status} | {res['tok_per_sec']:.1f} tok/s | {res['completion_tokens']} toks ({why})"
        )

    t1_wall = time.perf_counter()
    m_end = fetch_metrics(base_url)

    drafted_delta = m_end.get("splash_drafted_tokens_total", 0) - m_start.get(
        "splash_drafted_tokens_total", 0
    )
    accepted_delta = m_end.get("splash_accepted_draft_tokens_total", 0) - m_start.get(
        "splash_accepted_draft_tokens_total", 0
    )
    acc_ratio = (accepted_delta / drafted_delta * 100.0) if drafted_delta > 0 else 0.0

    avg_tps = (sum(decode_speeds) / len(decode_speeds)) if decode_speeds else 0.0
    accuracy = (correct_count / len(suite)) * 100.0
    wall_sec = t1_wall - t0_wall

    summary = {
        "items": len(suite),
        "correct": correct_count,
        "accuracy_pct": accuracy,
        "avg_decode_tok_s": avg_tps,
        "total_completion_tokens": total_tokens,
        "drafted_tokens": drafted_delta,
        "accepted_draft_tokens": accepted_delta,
        "acceptance_ratio_pct": acc_ratio,
        "wall_seconds": wall_sec,
    }

    print("\n" + "=" * 60)
    print("SPECULATIVE DRAFTING BENCHMARK REPORT")
    print("=" * 60)
    print(f"Overall Accuracy:         {accuracy:.1f}% ({correct_count}/{len(suite)})")
    print(f"Average Decode Speed:     {avg_tps:.2f} tok/s")
    print(f"Total Completion Tokens:  {total_tokens}")
    print(f"Drafted Tokens:           {drafted_delta:,.0f}")
    print(f"Accepted Draft Tokens:    {accepted_delta:,.0f}")
    print(f"Draft Acceptance Ratio:   {acc_ratio:.2f}%")
    print(f"Total Wall Clock Time:    {wall_sec:.1f} s")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", default="local/swift-qwen38-flash-next-v3")
    parser.add_argument("--count-per-domain", type=int, default=5)
    args = parser.parse_args()
    run_benchmark(
        base_url=args.url, model=args.model, count_per_domain=args.count_per_domain
    )
