#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/PagedAttention.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

// Quest (ICML 2024): Query-Aware Page Sparsity
// Computes upper-bound score S_max(q, P) = sum_g max(q_g * k_min_g, q_g * k_max_g)
// across each historical KV page. Pages with the highest upper-bounds are selected
// for full attention evaluation, while pages below the threshold are bypassed.

kernel void quest_estimate_page_bounds(
    device const bfloat *queries [[buffer(0)]],                  // [Batch * KVHeads, M * D]
    device const QuestPageBounds *page_bounds [[buffer(1)]],     // [TotalPages * KVHeads]
    device const uint *page_table [[buffer(2)]],                 // [Pages]
    device float *page_scores [[buffer(3)]],                     // [KVHeads, Pages]
    constant SplashQuestEstimateParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],                // x = page_idx, y = kv_head
    uint thread_index [[thread_index_in_threadgroup]]) {          // 0..31 (one SIMD group)

  uint page_idx = group.x;
  uint kv_head = group.y;

  if (page_idx >= params.total_pages || kv_head >= params.kv_heads) {
    return;
  }

  // Active recency window is always preserved with maximum priority score
  if (page_idx + params.min_recency_pages >= params.total_pages) {
    if (thread_index == 0) {
      page_scores[kv_head * params.total_pages + page_idx] = 1e9f;
    }
    return;
  }

  uint physical_page = page_table[page_idx];
  ulong bounds_idx = ulong(physical_page) * params.kv_heads + kv_head;
  QuestPageBounds bounds = page_bounds[bounds_idx];

  // Evaluate query projection across the 8 groups.
  // Thread 0..7 evaluate groups 0..7.
  float bound_dot = 0.0f;
  if (thread_index < 8) {
    uint g = thread_index;
    float k_lo = bounds.k_min[g];
    float k_hi = bounds.k_max[g];

    // Compute dot product of query elements in group g
    // Query dimension is 256, 8 groups of 32 elements.
    // For fast estimation, we sample the dominant direction of the query group
    device const bfloat *q_grp = queries + (kv_head * 256) + (g * 32);
    float q_sum = 0.0f;
    for (uint i = 0; i < 32; i += 4) {
      q_sum += float(q_grp[i]);
    }
    // Upper-bound dot product contribution
    bound_dot = max(q_sum * k_lo, q_sum * k_hi);
  }

  // Reduce across the 8 groups
  float total_score = simd_sum(bound_dot);

  if (thread_index == 0) {
    page_scores[kv_head * params.total_pages + page_idx] = total_score;
  }
}

// Update Quest page bounds from newly computed Q8 / Q4 key tiles
kernel void quest_update_page_bounds(
    device const char *key_data [[buffer(0)]],
    device const float *key_scales [[buffer(1)]],
    device QuestPageBounds *page_bounds [[buffer(2)]],
    constant uint &physical_page [[buffer(3)]],
    constant uint &kv_heads [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],     // x = kv_head, y = group (0..7)
    uint thread_index [[thread_index_in_threadgroup]]) {

  uint kv_head = group.x;
  uint g = group.y;

  if (kv_head >= kv_heads || g >= 8) {
    return;
  }

  // 32 tokens in the page, group has 32 elements.
  // Find min and max key values across all 32 tokens in group g.
  float local_min = 1e9f;
  float local_max = -1e9f;

  if (thread_index < 32) {
    uint token = thread_index;
    ulong scale_idx = (ulong(physical_page) * 32 + token) * kv_heads + kv_head;
    float scale = key_scales[scale_idx];

    ulong data_offset = ((ulong(physical_page) * 32 + token) * kv_heads + kv_head) * 256 + g * 32;
    device const char *token_keys = key_data + data_offset;

    float group_sum = 0.0f;
    for (uint i = 0; i < 32; i += 4) {
      group_sum += float(token_keys[i]) * scale;
    }
    local_min = group_sum;
    local_max = group_sum;
  }

  float global_min = simd_min(local_min);
  float global_max = simd_max(local_max);

  if (thread_index == 0) {
    ulong bounds_idx = ulong(physical_page) * kv_heads + kv_head;
    page_bounds[bounds_idx].k_min[g] = global_min;
    page_bounds[bounds_idx].k_max[g] = global_max;
  }
}
#endif
