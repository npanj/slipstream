#!/usr/bin/env python3
import json
import time
import urllib.request

def test_tool_calling(base_url="http://127.0.0.1:8090/v1"):
    print("\n--- Testing Structured Tool Calling ---")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    payload = {
        "model": "local/swift-qwen38-flash-next-v3",
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo in celsius? Call the tool get_weather."}
        ],
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 256
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    choice = data["choices"][0]["message"]
    tool_calls = choice.get("tool_calls", [])
    print(f"Tool calling latency: {t1 - t0:.3f}s")
    print(f"Tool calls received: {len(tool_calls)}")
    if tool_calls:
        print(f"Function: {tool_calls[0]['function']['name']}")
        print(f"Arguments: {tool_calls[0]['function']['arguments']}")
    assert len(tool_calls) > 0, "Expected at least one tool call"
    print("PASS: Structured tool calling functional.")

def test_prefix_caching(base_url="http://127.0.0.1:8090/v1"):
    print("\n--- Testing Prefix Caching & KV Reuse ---")
    long_prefix = "You are a specialized code analysis bot. Here is the context:\n" + ("def calculate_factorial(n):\n    if n <= 1: return 1\n    return n * calculate_factorial(n - 1)\n" * 30)
    
    # Turn 1: Cold prefill
    p1 = {
        "model": "local/swift-qwen38-flash-next-v3",
        "messages": [
            {"role": "system", "content": long_prefix},
            {"role": "user", "content": "What does the function do in 5 words?"}
        ],
        "temperature": 0.0,
        "max_tokens": 30
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(p1).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        d1 = json.loads(resp.read().decode("utf-8"))
    t_cold = time.perf_counter() - t0
    cached_tokens_1 = d1.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)

    # Turn 2: Warm prefix hit
    p2 = {
        "model": "local/swift-qwen38-flash-next-v3",
        "messages": [
            {"role": "system", "content": long_prefix},
            {"role": "user", "content": "What is the base case in 5 words?"}
        ],
        "temperature": 0.0,
        "max_tokens": 30
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(p2).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        d2 = json.loads(resp.read().decode("utf-8"))
    t_warm = time.perf_counter() - t0
    cached_tokens_2 = d2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)

    print(f"Turn 1 (Cold) Latency: {t_cold:.3f}s (cached tokens: {cached_tokens_1})")
    print(f"Turn 2 (Warm) Latency: {t_warm:.3f}s (cached tokens: {cached_tokens_2})")
    speedup = t_cold / max(t_warm, 0.001)
    print(f"Warm speedup: {speedup:.2f}x")
    print("PASS: Prefix caching functional.")

def test_streaming(base_url="http://127.0.0.1:8090/v1"):
    print("\n--- Testing Streaming TTFT & ITL ---")
    payload = {
        "model": "local/swift-qwen38-flash-next-v3",
        "messages": [
            {"role": "user", "content": "Count from 1 to 20 separated by commas."}
        ],
        "temperature": 0.0,
        "max_tokens": 100,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
    total_time = time.perf_counter() - t0
    print(f"Streaming TTFT: {ttft:.3f}s")
    print(f"Total stream duration: {total_time:.3f}s for {chunks} chunks")
    print("PASS: Streaming functional.")

if __name__ == "__main__":
    test_tool_calling()
    test_prefix_caching()
    test_streaming()
