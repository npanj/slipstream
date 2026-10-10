#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct FullPrefillParams {
  uint32_t tokens;
  uint32_t cache_stride;
  uint32_t row_stride;
};

static_assert(sizeof(FullPrefillParams) == 12,
              "Full attention prefill parameters are 12 bytes on both sides");

struct FullDecodeBatchParams {
  uint32_t tokens;
  uint32_t cache_stride;
  uint32_t row_stride;
  uint32_t lanes;
};

static_assert(sizeof(FullDecodeBatchParams) == 16,
              "Full attention verify parameters are 16 bytes on both sides");

// One full-attention layer's paged KV. Every current row is written directly
// into its final Q8 page slot before attention. Prefill and verify both read
// all visible history from the same paged representation. Decode accepts rows
// only by advancing committed_tokens; the next command overwrites rejected
// slots.
struct SplashChunkedPrefillParams {
  uint32_t committed_tokens;
  uint32_t chunk_tokens;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t physical_page_count;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

static_assert(sizeof(SplashChunkedPrefillParams) == 32,
              "Q8 chunked store parameters are 32 bytes on both sides");

// Prefill divides each query tile's visible Page32 history into balanced
// splits. The same count and partition rule are used by split and reduce.
struct SplashQ8PrefillAttentionParams {
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t physical_page_count;
  uint32_t split_count;
  uint32_t reserved0;
  uint32_t reserved1;
};

static_assert(sizeof(SplashQ8PrefillAttentionParams) == 32,
              "Q8 prefill attention parameters are 32 bytes on both sides");

struct SplashQ8VerifyAttentionParams {
  uint32_t committed_tokens;
  uint32_t active_rows;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t physical_page_count;
  // Filled from the plan: this lane's history-scaled split count and the
  // plan-wide slot stride that every lane's partials use.
  uint32_t split_count;
  uint32_t slot_splits;
  uint32_t tree_parents;
};

static_assert(sizeof(SplashQ8VerifyAttentionParams) == 32,
              "Q8 verify attention parameters are 32 bytes on both sides");

// Quest (ICML 2024): Query-Aware Page Sparsity.
// Per-page bounding box of key vectors across the 8 sub-vector groups (groupSize 32).
// Stores k_min and k_max for each group, enabling rapid estimation of the upper-bound
// attention score S_max(q, P) = sum_g max(q_g * k_min_g, q_g * k_max_g).
struct QuestPageBounds {
  float k_min[8]; // 32 bytes
  float k_max[8]; // 32 bytes
};

static_assert(sizeof(QuestPageBounds) == 64,
              "QuestPageBounds is 64 bytes per page on both sides");

struct SplashQuestEstimateParams {
  uint32_t committed_tokens;
  uint32_t active_rows;
  uint32_t total_pages;
  uint32_t top_k_pages;
  uint32_t min_recency_pages;
  uint32_t kv_heads;
  uint32_t reserved0;
  uint32_t reserved1;
};

static_assert(sizeof(SplashQuestEstimateParams) == 32,
              "SplashQuestEstimateParams is 32 bytes on both sides");

// DuoAttention (ICLR 2025): Dual-head attention splitting.
// Separates heads into Retrieval Heads (full KV history) and Streaming Heads
// (local circular ring buffer + sink tokens).
struct SplashDuoAttentionParams {
  uint32_t retrieval_head_mask; // Bitmask of retrieval heads (up to 32 query heads)
  uint32_t sink_tokens;         // Number of initial attention sinks (default 4)
  uint32_t streaming_window;    // Local circular buffer size (default 256 tokens = 8 pages)
  uint32_t reserved0;
};

static_assert(sizeof(SplashDuoAttentionParams) == 16,
              "SplashDuoAttentionParams is 16 bytes on both sides");

// MiniCache (NeurIPS 2024): KV Cache Compression via Cross-Layer State Sharing.
// In mid-to-deep layers, adjacent layers (L, L+1) share direction vectors using
// Spherical Linear Interpolation (SLERP) while retaining layer-specific scalar norms.
struct SplashMiniCacheParams {
  uint32_t merge_start_layer;  // First layer where cross-layer merging is active (e.g. layer 16)
  uint32_t total_layers;       // Total transformer layers (e.g. 48)
  float similarity_threshold;  // Minimum cosine similarity to merge layer pairs (default 0.90f)
  uint32_t reserved0;
};

static_assert(sizeof(SplashMiniCacheParams) == 16,
              "SplashMiniCacheParams is 16 bytes on both sides");

