#!/usr/bin/env python3
"""Diverse Workload Benchmark Suite for Slipstream-GGUF (Swift-Qwen3.8-Flash-Next-V3).

Evaluates speculative decoding, prompt lookup, and grammar-pruning across 6 diverse workloads:
1. Tool Calling / Agentic JSON (Constrained decoding, TokenMask, JSON grammar)
2. Mathematical & Step-by-Step Reasoning (Deep CoT inside <thought>)
3. Code Generation & Algorithm Synthesis (Python interval merging with test assertions)
4. Structured Table Information Extraction (Exact prompt lookup n-gram copying)
5. Systems Architecture Q&A (Technical prose explanation)
6. Structured JSON Extraction (Direct JSON object output)
7. Long Document / Log Analysis (1.5k context prefill + pinpoint extraction)

Measures:
- Decode speed (tok/s)
- Time to first token (TTFT ms)
- Draft acceptance rate (%)
- Inter-token latency (ITL p50 ms)
- Correctness & validity of responses
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def get_status(port: int = 8090) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", "/status")
        resp = conn.getresponse()
        raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    finally:
        conn.close()


def send_chat_completion(
    port: int,
    messages: list[dict],
    model: str = "local/swift-qwen38-flash-next-v3",
    max_tokens: int = 512,
    temperature: float = 0.0,
    tools: list[dict] | None = None,
    tool_choice: dict | str | None = None,
    reasoning_effort: str = "none",
) -> tuple[int, dict, float]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    body: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice

    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    t0 = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", payload, headers)
        resp = conn.getresponse()
        raw = resp.read()
        elapsed = time.perf_counter() - t0
        data = json.loads(raw.decode("utf-8"))
        return resp.status, data, elapsed
    finally:
        conn.close()


def _validate_json_profile(resp: dict) -> bool:
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        clean = re.sub(r"^```json\s*", "", content.strip())
        clean = re.sub(r"\s*```$", "", clean)
        obj = json.loads(clean)
        return (
            obj.get("name") == "John Doe"
            and obj.get("age") == 29
            and "Python" in obj.get("skills", [])
            and obj.get("location", {}).get("city") == "Seattle"
        )
    except Exception:
        return False


def _generate_log_prompt() -> str:
    lines = ["Here are the operational logs from node-cluster-east-04:"]
    for i in range(1, 45):
        timestamp = f"2026-09-27T18:{i:02d}:00.123Z"
        if i == 27:
            lines.append(
                f"[{timestamp}] [CRITICAL] [service=payment-gateway] [err_code=database_connection_pool_exhausted] "
                f"Fatal: connection pool max capacity 100 reached. 42 incoming transactions timed out."
            )
        else:
            lines.append(
                f"[{timestamp}] [INFO] [service=node-health] Heartbeat OK. Mem: {40 + (i % 15)}% CPU: {12 + (i % 20)}% "
                f"Disk: 54% Requests: {1000 + i * 17}/sec."
            )
    lines.append(
        "\nIdentify the single critical failure in these logs.\n"
        "Output the timestamp, service name, exact err_code, and root cause summary in 2 lines."
    )
    return "\n".join(lines)


WORKLOADS = [
    {
        "id": "agentic_tool_call_1",
        "category": "Agentic Tool Calling",
        "name": "Git & Repository Tool Call",
        "messages": [
            {
                "role": "user",
                "content": "Check the git repository status and commit all modified documentation files with the message 'docs: update spec'.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Commit staged changes to git.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "files": {"type": "array", "items": {"type": "string"}},
                            "amend": {"type": "boolean"},
                        },
                        "required": ["message", "files"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
        ],
        "tool_choice": "auto",
        "reasoning_effort": "none",
        "max_tokens": 128,
        "validator": lambda resp: (
            len(resp.get("choices", [{}])[0].get("message", {}).get("tool_calls", []))
            > 0
            and resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
            == "git_commit"
        ),
    },
    {
        "id": "agentic_tool_call_2",
        "category": "Agentic Tool Calling",
        "name": "Database Query Tool Call",
        "messages": [
            {
                "role": "user",
                "content": "Query the analytics database for active users created in the last 7 days with a limit of 50 rows.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "sql_query",
                    "description": "Execute a SQL query against the database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "database": {"type": "string"},
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["database", "query"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "sql_query"}},
        "reasoning_effort": "none",
        "max_tokens": 160,
        "validator": lambda resp: (
            len(resp.get("choices", [{}])[0].get("message", {}).get("tool_calls", []))
            > 0
            and "query"
            in json.loads(
                resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            )
        ),
    },
    {
        "id": "deep_reasoning_math",
        "category": "Reasoning & Math (CoT)",
        "name": "Multi-Step Revenue Arithmetic",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A boutique store sold 120 shirts. 40% of them were blue, 35% were red, "
                    "and the rest were green. Blue shirts sold for $25 each, red shirts for $30 each, "
                    "and green shirts for $20 each. What was the total revenue in dollars? "
                    "Show step-by-step reasoning and give the final number clearly."
                ),
            }
        ],
        "reasoning_effort": "medium",
        "max_tokens": 512,
        "validator": lambda resp: (
            "3060" in resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        ),
    },
    {
        "id": "code_generation",
        "category": "Coding & Algorithm",
        "name": "Interval Merging in Python",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a clean, self-contained Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
                    "that merges overlapping intervals. Include type hints, sorting, and docstrings. Then provide 3 assert test cases."
                ),
            }
        ],
        "reasoning_effort": "none",
        "max_tokens": 384,
        "validator": lambda resp: (
            "def merge_intervals"
            in resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            and "assert"
            in resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        ),
    },
    {
        "id": "table_prompt_lookup",
        "category": "Prompt Lookup & Extraction",
        "name": "Structured Markdown Table Extraction",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Here is a cluster service inventory table:\n\n"
                    "| Service Name | Port | Protocol | Memory Limit | Replicas | Status |\n"
                    "| redis-primary | 6379 | TCP | 2048Mi | 1 | healthy |\n"
                    "| auth-service | 8080 | HTTP | 512Mi | 3 | healthy |\n"
                    "| postgres-db | 5432 | TCP | 8192Mi | 1 | degraded |\n"
                    "| ingress-gateway | 443 | HTTPS | 1024Mi | 2 | healthy |\n"
                    "| metrics-agent | 9100 | HTTP | 256Mi | 5 | healthy |\n"
                    "| worker-pool | 9090 | gRPC | 4096Mi | 8 | healthy |\n\n"
                    "Extract ONLY the healthy services that use HTTP or HTTPS protocols.\n"
                    "For each matching service, output exactly one line in this format:\n"
                    "Service: <Service Name>, Port: <Port>, Memory: <Memory Limit>\n"
                    "Do not add any additional explanation."
                ),
            }
        ],
        "reasoning_effort": "none",
        "max_tokens": 160,
        "validator": lambda resp: (
            "auth-service"
            in resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            and "ingress-gateway"
            in resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            and "metrics-agent"
            in resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        ),
    },
    {
        "id": "structured_json_output",
        "category": "Direct JSON Output",
        "name": "User Profile to Schema JSON",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Convert the following description into a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "name": string,\n'
                    '  "age": integer,\n'
                    '  "skills": [string],\n'
                    '  "location": {"city": string, "state": string}\n'
                    "}\n\n"
                    "Description: John Doe is a 29-year-old software engineer living in Seattle, Washington. "
                    "He is proficient in Python, Rust, and Metal shaders.\n"
                    "Output ONLY the JSON object."
                ),
            }
        ],
        "reasoning_effort": "none",
        "max_tokens": 192,
        "validator": lambda resp: _validate_json_profile(resp),
    },
    {
        "id": "systems_architecture_qa",
        "category": "Systems Architecture Q&A",
        "name": "Write-Through vs Write-Back Caching",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the difference between write-through and write-back caching policies in computer architecture. "
                    "Discuss write latency, memory bus bandwidth consumption, and the purpose of the dirty bit. "
                    "Answer in two concise, technical paragraphs."
                ),
            }
        ],
        "reasoning_effort": "none",
        "max_tokens": 300,
        "validator": lambda resp: (
            "dirty"
            in resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .lower()
            and "write-through"
            in resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .lower()
        ),
    },
    {
        "id": "long_context_log_extraction",
        "category": "Long Context Extraction",
        "name": "Log Anomaly Root-Cause Analysis",
        "messages": [
            {
                "role": "user",
                "content": _generate_log_prompt(),
            }
        ],
        "reasoning_effort": "none",
        "max_tokens": 160,
        "validator": lambda resp: (
            "database_connection_pool_exhausted"
            in resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            or "pool"
            in resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .lower()
        ),
    },
]


def run_workloads(port: int = 8090, model: str = "local/swift-qwen38-flash-next-v3"):
    print(
        "\n=========================================================================="
    )
    print("  SLIPSTREAM DIVERSE WORKLOAD BENCHMARK SUITE")
    print(f"  Target: http://127.0.0.1:{port} ({model})")
    print(f"  Total Workloads: {len(WORKLOADS)}")
    print(
        "==========================================================================\n"
    )

    initial_status = get_status(port)
    initial_metrics = initial_status.get("metrics", {})
    init_drafted = initial_metrics.get("drafted_tokens", 0)
    init_accepted = initial_metrics.get("accepted_draft_tokens", 0)

    results = []

    for idx, wl in enumerate(WORKLOADS, 1):
        print(
            f"[{idx:02d}/{len(WORKLOADS):02d}] {wl['category']} — {wl['name']}... ",
            end="",
            flush=True,
        )

        st_before = get_status(port)
        m_before = st_before.get("metrics", {})

        status_code, resp_data, wall_time = send_chat_completion(
            port=port,
            messages=wl["messages"],
            model=model,
            max_tokens=wl.get("max_tokens", 384),
            temperature=0.0,
            tools=wl.get("tools"),
            tool_choice=wl.get("tool_choice"),
            reasoning_effort=wl.get("reasoning_effort", "none"),
        )

        st_after = get_status(port)
        m_after = st_after.get("metrics", {})

        if status_code != 200:
            print(f"FAILED (HTTP {status_code})")
            results.append(
                {
                    "id": wl["id"],
                    "category": wl["category"],
                    "name": wl["name"],
                    "ok": False,
                    "error": resp_data.get("error", {}).get("message", "unknown error"),
                }
            )
            continue

        usage = resp_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Delta metrics
        delta_drafted = m_after.get("drafted_tokens", 0) - m_before.get(
            "drafted_tokens", 0
        )
        delta_accepted = m_after.get("accepted_draft_tokens", 0) - m_before.get(
            "accepted_draft_tokens", 0
        )
        acceptance_pct = (
            (delta_accepted / max(1, delta_drafted)) * 100.0
            if delta_drafted > 0
            else 0.0
        )

        # Timing
        resp_metrics = resp_data.get("metrics", {})
        ttft_ms = resp_metrics.get("request_latency", {}).get("ttft_ms", 0.0)
        decode_ms = resp_metrics.get("native_delta", {}).get("decode_wall_ms", 0.0)
        if decode_ms <= 0:
            decode_ms = wall_time * 1000.0 - ttft_ms

        decode_tok_s = (
            (completion_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        )

        # Validator
        validator = wl.get("validator")
        valid = validator(resp_data) if validator else True

        print(
            f"OK | {completion_tokens:3d} toks | {decode_tok_s:5.1f} tok/s | Draft Acc: {acceptance_pct:5.1f}% | TTFT: {ttft_ms:6.1f} ms | {'PASS' if valid else 'WARN-VALIDATION'}"
        )

        results.append(
            {
                "id": wl["id"],
                "category": wl["category"],
                "name": wl["name"],
                "ok": True,
                "valid": valid,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "decode_tok_s": decode_tok_s,
                "ttft_ms": ttft_ms,
                "delta_drafted": delta_drafted,
                "delta_accepted": delta_accepted,
                "acceptance_pct": acceptance_pct,
                "response": resp_data,
            }
        )

    # Summary
    final_status = get_status(port)
    final_metrics = final_status.get("metrics", {})
    total_drafted = final_metrics.get("drafted_tokens", 0) - init_drafted
    total_accepted = final_metrics.get("accepted_draft_tokens", 0) - init_accepted
    overall_acc = (
        (total_accepted / max(1, total_drafted)) * 100.0 if total_drafted > 0 else 0.0
    )

    print(
        "\n=========================================================================="
    )
    print("  BENCHMARK RESULTS SUMMARY ACROSS ALL WORKLOADS")
    print("==========================================================================")
    print(
        f"{'Category / Workload':<35} | {'Tok/s':<8} | {'Draft Acc':<10} | {'Tokens':<8} | {'TTFT':<9} | {'Status'}"
    )
    print(
        "------------------------------------------------------------------------------------------------"
    )
    for r in results:
        if not r["ok"]:
            print(
                f"{r['name']:<35} | {'FAIL':<8} | {'N/A':<10} | {'N/A':<8} | {'N/A':<9} | FAIL: {r.get('error', '')}"
            )
        else:
            status_str = "PASS" if r["valid"] else "WARN"
            print(
                f"{r['name']:<35} | {r['decode_tok_s']:<6.1f}   | {r['acceptance_pct']:<5.1f}%    | {r['completion_tokens']:<8} | {r['ttft_ms']:<6.1f} ms | {status_str}"
            )

    print(
        "------------------------------------------------------------------------------------------------"
    )
    avg_speed = sum(r["decode_tok_s"] for r in results if r["ok"]) / max(
        1, len([r for r in results if r["ok"]])
    )
    total_output = sum(r["completion_tokens"] for r in results if r["ok"])
    print(
        f"Aggregate Draft Acceptance : {overall_acc:.1f}% ({total_accepted}/{total_drafted} tokens)"
    )
    print(
        f"Average Decode Throughput  : {avg_speed:.1f} tok/s across {total_output} output tokens"
    )
    print(
        f"Current Host Available RAM : {final_status.get('memory_governor', {}).get('host_available_bytes', 0) / (1024**3):.2f} GiB"
    )
    print(
        "==========================================================================\n"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", type=str, default="local/swift-qwen38-flash-next-v3")
    args = parser.parse_args()

    run_workloads(port=args.port, model=args.model)
