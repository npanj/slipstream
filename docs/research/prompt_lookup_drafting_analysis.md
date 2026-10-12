# Analysis & Plan: Fast Prompt Lookup Decoding for Slipstream

**Author:** antigravity\
**Date:** 2026-09-26\
**Subject:** Technical evaluation of Hayder Tirmazi's *"42x Faster Prompt Lookup Drafting in llama.cpp"* and its application to Slipstream.

---

## 1. Conclusion First

**Prompt lookup speculation can give Swift models free speedups with zero GPU memory and zero extra weights.**

In coding, document editing, question answering, and structured JSON output, up to 40–70% of generated words already appear in the prompt or recent context. Instead of running a neural network to guess tokens, prompt lookup finds repeating patterns in CPU RAM in under 4 microseconds.

For Slipstream, this offers two major opportunities:
1. **Accelerate `Swift-Qwen3.8-27B-Splash-HQ` (which has no draft head) from 44 tok/s to 60–75+ tok/s** on tasks with context overlap, with zero GPU memory cost.
2. **Hybrid drafting for `Swift-Flash-Next-V3`**: When the MTP neural head is uncertain (confidence < 0.35), prompt lookup can step in and provide candidate tokens rather than falling back to slow 1-token steps.

---

## 2. Words We Can't Avoid

| Term | Plain Meaning |
|---|---|
| **Prompt Lookup Decoding (PLD)** | Guessing the next tokens by finding matches to the current word in the prompt text. |
| **N-gram** | A sequence of $N$ consecutive tokens (e.g. a 3-token phrase). |
| **Branchless Binary Search** | Searching a sorted list using arithmetic instead of `if` statements, preventing CPU pipeline stalls. |
| **Segmented Hash Map** | A hash table storing items in cache-friendly memory blocks instead of scattered pointers. |
| **Binary Fuse Filter** | A tiny, static lookup table that checks if an item exists using almost no memory. |

---

## 3. What Tirmazi Achieved in llama.cpp

Original prompt lookup in llama.cpp was slow: it took **165 microseconds per token**. Spending 1–2 milliseconds on CPU search destroyed the time saved by speculative drafting.

Tirmazi cut lookup latency to **0.9–4.0 microseconds per token (42x faster)** using four computer science optimizations:

```
+----------------------------------------------------------------------------+
| 1. Stop copying inner containers: Use string views and spans               |
|    Saved: ~35 µs per query (eliminated heap allocations)                   |
+----------------------------------------------------------------------------+
                                     |
                                     v
+----------------------------------------------------------------------------+
| 2. Replace std::unordered_map with ankerl::unordered_dense::segmented_map  |
|    Saved: ~55 µs per query (eliminated pointer chasing & cache misses)     |
+----------------------------------------------------------------------------+
                                     |
                                     v
+----------------------------------------------------------------------------+
| 3. Replace candidate sets with sorted vectors + branchless binary search   |
|    Saved: ~45 µs per query (eliminated CPU branch mispredictions)          |
+----------------------------------------------------------------------------+
                                     |
                                     v
+----------------------------------------------------------------------------+
| 4. Replace dynamic map with static Lemire binary fuse filter (constmap)    |
|    Saved: ~25 µs per query (compact immutable index for the prompt text)   |
+----------------------------------------------------------------------------+
                                     |
                                     v
                 Result: 165 µs -> 0.9–4.0 µs per token (42x faster)
```

### Detailed Breakdown of the Four Techniques

### Optimization 1: Zero-Copy Candidate Windows
- **The Problem:** The original code copied `std::vector<llama_token>` instances when querying matches.
- **The Fix:** Use `std::span<const uint32_t>` or offset pairs `(start_idx, length)`. Memory allocations drop to zero during decode.

### Optimization 2: Segmented Dense Hash Map
- **The Problem:** `std::unordered_map` is a node-based linked list. Every lookup chases pointers across scattered memory addresses, causing L1/L2 CPU cache misses.
- **The Fix:** Use `ankerl::unordered_dense::segmented_map`. Buckets are contiguous in RAM, fitting directly inside modern CPU cache lines.

### Optimization 3: Branchless Binary Search on Candidate Suffixes
- **The Problem:** A typical binary search uses `if (arr[mid] < target)`. When searching unpredictable token IDs, the CPU's branch predictor guesses wrong ~50% of the time, flushing the execution pipeline (costing 15–20 CPU cycles per comparison).
- **The Fix:** A branchless binary search layout:
  ```cpp
  // Step arithmetic without branching:
  while (n > 1) {
    size_t half = n / 2;
    base = (base[half] < target) ? base + half : base;
    n -= half;
  }
  return (*base < target) ? base + 1 : base;
  ```
  The compiler converts this to conditional move instructions (`cmov` on x86, `csel` on Apple Silicon ARM64), executing with zero pipeline stalls.

### Optimization 4: Static Binary Fuse Filter (`constmap`)
- **The Problem:** The prompt text is 100% fixed once generation begins. Keeping a mutable, dynamic hash table wastes RAM and has hash collision overhead.
- **The Fix:** Build an immutable minimal perfect hash index or binary fuse filter (based on Daniel Lemire's research) during prompt ingestion. Lookup becomes a single direct mathematical probe into read-only memory.

---

## 4. Why This Matters for Slipstream

Slipstream serves models on Apple Silicon Macs where unified memory bandwidth is the primary bottleneck.

### 1. Speeding up `Swift-Qwen3.8-27B-Splash-HQ` (The Biggest Win)
- **Current State:** Swift 27B has no MTP draft head. Every token requires a full forward pass through its 64 layers (~22.5 ms per token, 44.4 tok/s).
- **With Fast Prompt Lookup:**
  - When editing code, answering questions about provided context, or producing formatted output, prompt lookup proposes 3–5 tokens.
  - Verification runs through the 27B model in one batch step (~28 ms for 4 tokens).
  - Effective decode jumps to **70–85 tok/s** on context-rich workloads, with **0 extra gigabytes of weights**.

### 2. Hybrid Speculation on `Swift-Flash-Next-V3`
- **Current State:** Swift V3's MTP draft head is accurate on common phrases, but when chain confidence drops below 0.35, it stops drafting.
- **With Hybrid Prompt Lookup:**
  - If MTP head confidence is low, the engine checks the prompt lookup table (sub-4 µs on CPU).
  - If the prompt contains the matching 3-gram, prompt lookup supplies the remaining draft tokens.
  - This bridges difficult transitions without wasting GPU compute.

---

## 5. Implementation Roadmap for Slipstream

### Phase 1: Standalone CPU Prompt Lookup Engine (Zero Engine Risk)
1. Add `runtime/ops/PromptLookup.hpp` and `PromptLookup.cpp`.
2. Implement 3-gram hashing with `ankerl::unordered_dense::segmented_map`.
3. Provide `indexPrompt(std::span<const uint32_t> promptTokens)` and `propose(std::span<const uint32_t> recentTokens, uint32_t maxDrafts)`.
4. Add unit test measuring lookup speed on Apple Silicon (target: < 3 µs per query).

### Phase 2: Speculative Verifier Integration on Swift-27B
1. Wire `PromptLookup` into `runtime/model/Runtime.mm` for models where `geometry.hasDraftHead == false`.
2. During decode, call `promptLookup.propose()` to fill `DecodeTensor::ProposedTokens`.
3. Use existing linear speculative verification kernels in Metal.
4. Benchmark speedup across coding and Q&A benchmarks.

### Phase 3: Hybrid MTP + Prompt Lookup Fallback on Flash-Next V3
1. In `models/qwen4exp/Qwen4ExpTarget.cpp`, if `lastConfidence < 0.35`, query `promptLookup` to complete the draft proposals.
2. Measure tokens/step increase and verify output identity.

---

## 6. Comparison: MTP Neural Drafting vs Prompt Lookup Drafting

| Feature | MTP Neural Drafting (Swift V3) | Prompt Lookup Drafting (Tirmazi / PLD) |
|---|---|---|
| **Weight Overhead** | ~1.5 GB dedicated MTP weights | **0 MB** (uses existing text in RAM) |
| **GPU Compute** | Requires 1 GPU draft forward pass (~8 ms) | **0 GPU compute** (runs on CPU in < 4 µs) |
| **Novel Generation** | Excellent (can predict brand new words) | None (can only predict words seen in context) |
| **Code / JSON / RAG** | Good (~3.2 tokens/step) | **Exceptional** (often 4–6 tokens/step on repetitive syntax) |
| **Model Compatibility** | Requires model-specific trained MTP head | **Works on ANY model** (27B, 35B, 70B, etc.) |
