#!/usr/bin/env python3
"""Run V3 GGUF model sequentially on Slipstream-GGUF and llama.cpp, comparing quality & speed."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BENCH_PROMPTS = [
    {
        "id": "gsm8k_math",
        "category": "Math Reasoning",
        "prompt": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins for her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. How much in dollars does "
            "she make every day at the farmers' market? Think step by step and conclude your "
            "response with: Therefore, the final answer is $ANSWER."
        ),
        "expected": "18",
        "max_tokens": 800,
    },
    {
        "id": "math500_series",
        "category": "Advanced Math Derivation",
        "prompt": (
            "Define\n"
            "\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]\n"
            "Find a way to write\n"
            "\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]\n"
            "in terms of $p$ and $q.$\n"
            "Think step by step and conclude your response with: "
            "Therefore, the final answer is \\boxed{ANSWER}"
        ),
        "expected": "p - q",
        "max_tokens": 1200,
    },
    {
        "id": "constraint_logic",
        "category": "Constraint Reasoning",
        "prompt": (
            "Three friends (Alice, Bob, Charlie) are sitting in a row of 3 chairs numbered 1 to 3 "
            "from left to right. Constraints:\n"
            "1. Alice never sits next to Bob.\n"
            "2. Charlie is to the right of Alice (higher chair number).\n"
            "Who sits in chair 1, chair 2, and chair 3? Give a logical deduction step by step, "
            "then state the final arrangement clearly."
        ),
        "expected": "Chair 1: Alice, Chair 2: Charlie, Chair 3: Bob",
        "max_tokens": 800,
    },
    {
        "id": "python_intervals",
        "category": "Coding & Algorithms (Python)",
        "prompt": (
            "Write a clean, optimal Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
            "that merges all overlapping intervals in O(N log N) time and O(N) extra space. "
            "Handle edge cases such as empty lists and negative coordinates. "
            "Include type annotations, a concise docstring, and 3 unit test assertions using `assert`."
        ),
        "expected": None,
        "max_tokens": 1000,
    },
    {
        "id": "rust_csv",
        "category": "Systems Coding (Rust)",
        "prompt": (
            "Write a Rust function `parse_csv_line(line: &str) -> Vec<String>` that parses a single CSV line "
            "with quoted fields (commas inside double quotes are preserved) into a Vec<String>. "
            'Handle double-quote escaping ("") properly. Include two unit tests in a `#[cfg(test)]` module.'
        ),
        "expected": None,
        "max_tokens": 1200,
    },
    {
        "id": "tech_explanation",
        "category": "Technical Communication",
        "prompt": (
            "Explain how the multi-head self-attention mechanism works to a software engineer "
            "who knows linear algebra (matrices, dot products, projections) but no deep learning. "
            "Keep the explanation clear, mathematically precise, and within 3 concise paragraphs."
        ),
        "expected": None,
        "max_tokens": 800,
    },
]


def check_port_free(port: int):
    try:
        out = (
            subprocess.check_output(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return len(out) == 0
    except subprocess.CalledProcessError:
        return True


def kill_on_port(port: int):
    try:
        out = (
            subprocess.check_output(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if out:
            for pid in out.split():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except OSError:
                    pass
            time.sleep(2)
            for pid in out.split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
    except subprocess.CalledProcessError:
        pass


def wait_for_ready(url: str, timeout: int = 180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bench"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stream_chat_completion(
    base_url: str, prompt: str, max_tokens: int, model: str = None
):
    url = f"{base_url}/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if model:
        payload["model"] = model

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    first_token_time = None
    chunks_text = []
    reasoning_text = []
    token_count = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str == "data: [DONE]":
                    continue
                if line_str.startswith("data: "):
                    json_str = line_str[6:]
                    try:
                        data_obj = json.loads(json_str)
                        choices = data_obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content_piece = delta.get("content", "")
                            reasoning_piece = delta.get("reasoning_content", "")
                            if content_piece or reasoning_piece:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                token_count += 1
                                if content_piece:
                                    chunks_text.append(content_piece)
                                if reasoning_piece:
                                    reasoning_text.append(reasoning_piece)
                    except Exception:
                        pass

        t_end = time.perf_counter()
        elapsed = t_end - t0
        ttft_ms = ((first_token_time - t0) * 1000.0) if first_token_time else 0.0
        decode_duration = (t_end - first_token_time) if first_token_time else 0.0
        decode_tok_s = (
            ((token_count - 1) / decode_duration)
            if (decode_duration > 0 and token_count > 1)
            else 0.0
        )

        full_content = "".join(chunks_text)
        full_reasoning = "".join(reasoning_text)

        return {
            "token_count": token_count,
            "elapsed_s": elapsed,
            "ttft_ms": ttft_ms,
            "decode_tok_s": decode_tok_s,
            "content": full_content,
            "reasoning": full_reasoning,
        }
    except Exception as e:
        return {"error": str(e), "elapsed_s": time.perf_counter() - t0}


def run_benchmark_on_engine(engine_name: str, base_url: str, model_id: str = None):
    print(f"\n{'=' * 70}\nRunning Benchmark on: {engine_name} ({base_url})\n{'=' * 70}")
    results = []
    for item in BENCH_PROMPTS:
        print(
            f"\n--> [{item['id']}] {item['category']} (max {item['max_tokens']} tok)...",
            end="",
            flush=True,
        )
        res = stream_chat_completion(
            base_url, item["prompt"], item["max_tokens"], model=model_id
        )
        if "error" in res:
            print(f" ERROR: {res['error']}")
            results.append(
                {"id": item["id"], "category": item["category"], "error": res["error"]}
            )
        else:
            print(
                f" Done in {res['elapsed_s']:.2f}s | TTFT: {res['ttft_ms']:.1f}ms | Decode: {res['decode_tok_s']:.2f} tok/s | Tokens: {res['token_count']}"
            )
            ans_display = (
                res["content"].strip()
                if res["content"].strip()
                else res["reasoning"].strip()
            )
            snippet = (
                ans_display[-120:].replace("\n", " ")
                if len(ans_display) > 120
                else ans_display.replace("\n", " ")
            )
            print(f"    Tail snippet: {snippet}")
            results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "expected": item.get("expected"),
                    "token_count": res["token_count"],
                    "elapsed_s": res["elapsed_s"],
                    "ttft_ms": res["ttft_ms"],
                    "decode_tok_s": res["decode_tok_s"],
                    "content": res["content"],
                    "reasoning": res["reasoning"],
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="A/B Benchmark: Slipstream-GGUF vs llama.cpp on V3 GGUF"
    )
    parser.add_argument(
        "--model-dir",
        default=str(Path.home() / "models/qwen38-flash-next-v3"),
        help="Model directory for Slipstream",
    )
    parser.add_argument(
        "--llama-model", default=None, help="GGUF shard 1 file for llama.cpp"
    )
    parser.add_argument("--only-slipstream", action="store_true")
    parser.add_argument("--only-llamacpp", action="store_true")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    model_dir_path = Path(args.model_dir).expanduser().resolve()
    llama_model = args.llama_model
    if llama_model is None:
        shards = sorted(model_dir_path.glob("*00001-of-*.gguf"))
        if shards:
            llama_model = str(shards[0])
        else:
            llama_model = str(
                Path.home()
                / "models/qwen38-flash-next-v3/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00001-of-00003.gguf"
            )

    out_path_str = args.out
    if not out_path_str:
        out_path_str = str(
            ROOT / f"dev/benchmarks/comparison_{model_dir_path.name}_results.json"
        )
    out_file = Path(out_path_str)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    all_data = {}

    print(f"Target Model Dir: {model_dir_path}")
    print(f"Llama Checkpoint: {llama_model}")
    print(f"Results Output:   {out_file}")

    # Preflight memory safety check
    print("Checking active ports and existing model engines...")
    kill_on_port(8090)
    kill_on_port(8080)
    subprocess.run(["pkill", "-f", "generate-sample"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "splash serve-native"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "llama-server"], stderr=subprocess.DEVNULL)
    time.sleep(3)

    # 1. RUN SLIPSTREAM-GGUF
    if not args.only_llamacpp:
        print("\n" + "=" * 70)
        print("STEP 1: Starting Slipstream-GGUF on port 8090...")
        print("=" * 70)
        slip_env = os.environ.copy()
        slip_env["REPO"] = str(ROOT)
        slip_env["PKG"] = str(model_dir_path / "prepared")
        slip_env["PORT"] = "8090"
        slip_env["SPLASH_EXPERT_CACHE_GIB"] = "34"
        slip_env["CTX"] = "65536"

        cmd = [
            str(ROOT / "dev/benchmarks/guarded.py"),
            "--floor-gib",
            "4",
            "--max-seconds",
            "1800",
            "--",
            str(ROOT / "splash"),
            "serve",
            "--model",
            str(model_dir_path),
            "--port",
            "8090",
        ]
        log_file = open("/tmp/slipstream_bench_server.log", "w")
        proc_slip = subprocess.Popen(
            cmd, env=slip_env, stdout=log_file, stderr=subprocess.STDOUT
        )

        print("Waiting for Slipstream-GGUF on http://127.0.0.1:8090/v1/models...")
        if not wait_for_ready("http://127.0.0.1:8090/v1/models", timeout=120):
            print(
                "ERROR: Slipstream-GGUF failed to start! Check /tmp/slipstream_bench_server.log"
            )
            proc_slip.kill()
            return 1

        print("Slipstream-GGUF is live!")
        slip_results = run_benchmark_on_engine(
            "Slipstream-GGUF",
            "http://127.0.0.1:8090/v1",
            model_id=f"local/{model_dir_path.name}",
        )
        all_data["slipstream"] = slip_results

        print("\nStopping Slipstream-GGUF cleanly...")
        proc_slip.send_signal(signal.SIGINT)
        try:
            proc_slip.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc_slip.kill()
        kill_on_port(8090)
        subprocess.run(["pkill", "-f", "generate-sample"], stderr=subprocess.DEVNULL)
        log_file.close()
        time.sleep(5)
        print("Slipstream-GGUF stopped. Host memory returned to baseline.")

    # 2. RUN LLAMA.CPP
    if not args.only_slipstream:
        print("\n" + "=" * 70)
        print(
            f"STEP 2: Starting llama.cpp (llama-server) on port 8080 with {Path(llama_model).name}..."
        )
        print("=" * 70)
        llama_env = os.environ.copy()
        llama_env["MODEL"] = str(llama_model)
        llama_env["PORT"] = "8080"
        llama_env["CACHE"] = "34"
        llama_env["CTX"] = "65536"
        llama_env["WIRED_MB"] = "59392"
        llama_env["DRIVE_PAUSE"] = "0"

        cmd = [
            str(ROOT / "dev/benchmarks/guarded.py"),
            "--floor-gib",
            "4",
            "--max-seconds",
            "1800",
            "--",
            str(Path.home() / "models/bin/qwen-q40-server.sh"),
        ]
        log_file_llama = open("/tmp/llamacpp_bench_server.log", "w")
        proc_llama = subprocess.Popen(
            cmd, env=llama_env, stdout=log_file_llama, stderr=subprocess.STDOUT
        )

        print("Waiting for llama.cpp on http://127.0.0.1:8080/health...")
        if not wait_for_ready("http://127.0.0.1:8080/health", timeout=180):
            print(
                "ERROR: llama.cpp failed to start! Check /tmp/llamacpp_bench_server.log"
            )
            proc_llama.kill()
            return 1

        print("llama.cpp is live!")
        llama_results = run_benchmark_on_engine(
            "llama.cpp", "http://127.0.0.1:8080/v1", model_id="qwen3.8-flash-next-q40"
        )
        all_data["llamacpp"] = llama_results

        print("\nStopping llama.cpp cleanly...")
        proc_llama.send_signal(signal.SIGINT)
        try:
            proc_llama.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc_llama.kill()
        kill_on_port(8080)
        log_file_llama.close()
        time.sleep(5)
        print("llama.cpp stopped.")

    # 3. SAVE AND PRINT COMPARISON
    out_file.write_text(json.dumps(all_data, indent=2))
    print(f"\nSaved raw benchmark data to {out_file}")

    if "slipstream" in all_data and "llamacpp" in all_data:
        print("\n" + "=" * 80)
        print("HEAD-TO-HEAD SPEED & QUALITY COMPARISON (V3 GGUF Model)")
        print("=" * 80)
        print(
            f"{'Prompt ID':<20} | {'Slipstream (tok/s)':<18} | {'llama.cpp (tok/s)':<18} | {'Speedup':<8} | {'Quality Match'}"
        )
        print("-" * 80)

        slip_map = {r["id"]: r for r in all_data["slipstream"] if "decode_tok_s" in r}
        llama_map = {r["id"]: r for r in all_data["llamacpp"] if "decode_tok_s" in r}

        slip_speeds = []
        llama_speeds = []

        for p in BENCH_PROMPTS:
            pid = p["id"]
            if pid in slip_map and pid in llama_map:
                s_tok = slip_map[pid]["decode_tok_s"]
                l_tok = llama_map[pid]["decode_tok_s"]
                slip_speeds.append(s_tok)
                llama_speeds.append(l_tok)
                ratio = s_tok / l_tok if l_tok > 0 else 0.0

                s_content = slip_map[pid]["content"] or slip_map[pid]["reasoning"]
                l_content = llama_map[pid]["content"] or llama_map[pid]["reasoning"]

                q_status = "N/A"
                if p.get("expected"):
                    exp = p["expected"]
                    s_ok = exp.lower() in s_content.lower()
                    l_ok = exp.lower() in l_content.lower()
                    if s_ok and l_ok:
                        q_status = f"Both CORRECT ({exp})"
                    elif s_ok:
                        q_status = f"Slipstream ONLY ({exp})"
                    elif l_ok:
                        q_status = f"llama.cpp ONLY ({exp})"
                    else:
                        q_status = f"Neither matched ({exp})"
                else:
                    q_status = "Both completed"

                print(
                    f"{pid:<20} | {s_tok:>14.2f} t/s | {l_tok:>14.2f} t/s | {ratio:>6.2f}x | {q_status}"
                )

        if slip_speeds and llama_speeds:
            avg_s = sum(slip_speeds) / len(slip_speeds)
            avg_l = sum(llama_speeds) / len(llama_speeds)
            overall_ratio = avg_s / avg_l if avg_l > 0 else 0.0
            print("-" * 80)
            print(
                f"{'AVERAGE':<20} | {avg_s:>14.2f} t/s | {avg_l:>14.2f} t/s | {overall_ratio:>6.2f}x |"
            )
            print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
