#!/usr/bin/env python3
import json
import time
import urllib.request

base_url = "http://127.0.0.1:8090/v1"

# 1. Tool calling test
tools = [{
    "type": "function",
    "function": {
        "name": "lookup_stock",
        "description": "Get stock price",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"]
        }
    }
}]
payload = {
    "model": "local/swift-qwen38-27b-splash-hq",
    "messages": [{"role": "user", "content": "What is the price of AAPL? Use lookup_stock."}],
    "tools": tools,
    "temperature": 0.0,
    "max_tokens": 128
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
tc = data["choices"][0]["message"].get("tool_calls", [])
print(f"27B Tool Calling: {len(tc)} calls, latency: {t1-t0:.2f}s")
if tc:
    print(f"  Tool name: {tc[0]['function']['name']}")
    print(f"  Tool arguments: {tc[0]['function']['arguments']}")

# 2. Streaming test
payload_stream = {
    "model": "local/swift-qwen38-27b-splash-hq",
    "messages": [{"role": "user", "content": "Write a haiku about high-speed Apple Silicon inference."}],
    "temperature": 0.0,
    "max_tokens": 64,
    "stream": True,
    "stream_options": {"include_usage": True}
}
t0 = time.perf_counter()
req = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(payload_stream).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
ttft = None
chunks = 0
content = ""
with urllib.request.urlopen(req, timeout=30) as resp:
    for line in resp:
        line = line.decode("utf-8").strip()
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunk = json.loads(line[6:])
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
            if chunk.get("choices"):
                content += chunk["choices"][0]["delta"].get("content", "")
t1 = time.perf_counter()
tok_per_sec = chunks / (t1 - t0) if (t1 - t0) > 0 else 0
print(f"27B Streaming: TTFT {ttft:.3f}s, {chunks} chunks in {t1-t0:.2f}s ({tok_per_sec:.1f} tok/s)")
print(f"Haiku output: {content.strip()!r}")

# 3. Vision test (send 1x1 test pixel image)
import base64
# 1x1 red PNG base64
tiny_png_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
payload_vision = {
    "model": "local/swift-qwen38-27b-splash-hq",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": tiny_png_b64}}
            ]
        }
    ],
    "temperature": 0.0,
    "max_tokens": 64
}
t0 = time.perf_counter()
req = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(payload_vision).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        v_data = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    v_ans = v_data["choices"][0]["message"]["content"]
    print(f"27B Vision: latency {t1-t0:.2f}s, response: {v_ans.strip()!r}")
except Exception as e:
    print(f"27B Vision check: {e}")
