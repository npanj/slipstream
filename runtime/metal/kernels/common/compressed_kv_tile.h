#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/PagedAttention.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define COMPRESSED_CONSTANT constant constexpr
#else
#include <algorithm>
#include <cmath>
#include <cstdint>
using uchar = uint8_t;
using uint = uint32_t;
using ulong = uint64_t;
#define COMPRESSED_CONSTANT inline constexpr
#endif

COMPRESSED_CONSTANT uint SplashCompressedPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS; // 32
COMPRESSED_CONSTANT uint SplashCompressedHeadDimension = 256;
COMPRESSED_CONSTANT uint SplashCompressedGroupSize = 32;
COMPRESSED_CONSTANT uint SplashCompressedGroupsPerHead = SplashCompressedHeadDimension / SplashCompressedGroupSize; // 8

// 4-bit Quantized 32-element group with fp32 scale and bias
struct SplashQ4Group {
  uchar nibbles[SplashCompressedGroupSize / 2]; // 16 bytes
  float scale;                                  // 4 bytes
  float bias;                                   // 4 bytes
};

// 3-bit Quantized 32-element group: 32 * 3 = 96 bits = 12 bytes
struct SplashQ3Group {
  uchar packed[12];                             // 12 bytes
  float scale;                                  // 4 bytes
  float bias;                                   // 4 bytes
};

static_assert(sizeof(SplashQ4Group) == 24, "SplashQ4Group is 24 bytes");
static_assert(sizeof(SplashQ3Group) == 20, "SplashQ3Group is 20 bytes");

template <uint KVHeads>
inline ulong splash_compressed_group_index(uint page, uint head, uint token, uint group) {
  constexpr ulong GroupsPerPage = ulong(KVHeads) * SplashCompressedPageTokens * SplashCompressedGroupsPerHead;
  return ulong(page) * GroupsPerPage +
         (ulong(head) * SplashCompressedPageTokens + token) * SplashCompressedGroupsPerHead +
         group;
}

inline float unpack_q4(uchar b, bool is_high) {
  return float(is_high ? ((b >> 4) & 0x0F) : (b & 0x0F));
}

inline float unpack_q3(uchar b0, uchar b1, uchar b2, uint i) {
  uint bits = uint(b0) | (uint(b1) << 8) | (uint(b2) << 16);
  return float((bits >> (i * 3)) & 0x07);
}

#ifdef __METAL_VERSION__
// Split partitioning helper matching splash_attention_pages_per_split
inline bool splash_compressed_verify_attention_contract_valid(
    constant SplashQ8VerifyAttentionParams &params) {
  return params.active_rows > 0 &&
         params.active_rows <= SPLASH_TARGET_VERIFY_ROWS &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.chunk_stride >= params.active_rows &&
         params.page_table_entries >=
             (params.committed_tokens + params.active_rows +
              SplashCompressedPageTokens - 1) /
                 SplashCompressedPageTokens &&
         params.physical_page_count > 0 &&
         ulong(params.committed_tokens) + params.active_rows <=
             ulong(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS) &&
         params.split_count > 0 &&
         params.split_count <= SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS &&
         params.slot_splits >= params.split_count &&
         params.slot_splits <= SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS;
}

inline uint splash_compressed_pages(uint visible_tokens) {
  return (visible_tokens + SplashCompressedPageTokens - 1) / SplashCompressedPageTokens;
}

inline uint splash_compressed_pages_per_split(uint pages, uint splits) {
  return (pages + splits - 1) / splits;
}

// Direct-Read Q4 Attention Tile
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_q4_attention_direct_tile(
    device const bfloat *tile_queries,
    device const SplashQ4Group *q4_keys,
    device const SplashQ4Group *q4_values,
    device const uint *page_table,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *tg_scratch, // At least 8 floats for group reduction + 32 floats for scores
    uint thread_index) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = SplashCompressedHeadDimension; // 256
  constexpr uint N = SplashCompressedPageTokens;    // 32

  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_compressed_pages(visible_tokens);
  uint per_split = splash_compressed_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages) return;
  uint page_end = min(pages, page_begin + per_split);

  const uint d = thread_index; // 0..255
  const uint simd_id = d / 32; // 0..7 (group index)
  const uint lane_id = d % 32; // 0..31 (offset in group)

  float running_out[M];
  float row_max[M];
  float row_sum[M];
#pragma unroll
  for (uint m = 0; m < M; ++m) {
    running_out[m] = 0.0f;
    row_max[m] = -INFINITY;
    row_sum[m] = 0.0f;
  }

  for (uint page_step = page_begin; page_step < page_end; ++page_step) {
    uint physical = page_table[page_step];
    uint token_start = page_step * N;

    for (uint m = 0; m < M; ++m) {
      float q_val = float(tile_queries[m * D + d]);
      float q_group_sum = simd_sum(q_val);

      // Evaluate QK dot products across the 32 tokens of the page
      for (uint t = 0; t < N; ++t) {
        uint token = token_start + t;
        bool visible = (token < visible_tokens);

        if (visible) {
          ulong key_grp_idx = splash_compressed_group_index<KVHeads>(physical, kv_head, t, simd_id);
          device const SplashQ4Group &grp = q4_keys[key_grp_idx];
          uchar byte_val = grp.nibbles[lane_id / 2];
          float k_int = unpack_q4(byte_val, (lane_id & 1) != 0);
          float int_term = q_val * k_int;
          float int_dot = simd_sum(int_term);
          float grp_score = int_dot * grp.scale + q_group_sum * grp.bias;

          if (lane_id == 0) {
            tg_scratch[t * 8 + simd_id] = grp_score;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (thread_index < N) {
        uint token = token_start + thread_index;
        if (token < visible_tokens) {
          float total = 0.0f;
          for (uint g = 0; g < 8; ++g) total += tg_scratch[thread_index * 8 + g];
          tg_scratch[256 + thread_index] = total * 0.0625f; // Scale by 1/sqrt(256)
        } else {
          tg_scratch[256 + thread_index] = -INFINITY;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Online Softmax step
      float local_max = -INFINITY;
      for (uint t = 0; t < N; ++t) {
        local_max = max(local_max, tg_scratch[256 + t]);
      }
      const float previous_max = row_max[m];
      const float next_max = max(previous_max, local_max);

      float local_sum = 0.0f;
      float probs[N];
      for (uint t = 0; t < N; ++t) {
        float score = tg_scratch[256 + t];
        probs[t] = (score != -INFINITY) ? fast::exp(score - next_max) : 0.0f;
        local_sum += probs[t];
      }

      const float rescale = (next_max == -INFINITY || next_max == previous_max)
                                ? 1.0f
                                : fast::exp(previous_max - next_max);
      running_out[m] = running_out[m] * rescale;
      row_sum[m] = row_sum[m] * rescale + local_sum;
      row_max[m] = next_max;

      // PV accumulation: each thread updates dimension d directly
      for (uint t = 0; t < N; ++t) {
        if (probs[t] > 0.0f) {
          ulong val_grp_idx = splash_compressed_group_index<KVHeads>(physical, kv_head, t, simd_id);
          device const SplashQ4Group &v_grp = q4_values[val_grp_idx];
          uchar v_byte = v_grp.nibbles[lane_id / 2];
          float v_int = unpack_q4(v_byte, (lane_id & 1) != 0);
          float v_val = v_int * v_grp.scale + v_grp.bias;
          running_out[m] += probs[t] * v_val;
        }
      }
    }
  }

  // Write out partials and statistics
  for (uint m = 0; m < M; ++m) {
    partials[slot * M * D + m * D + d] = running_out[m];
    if (d == 0) {
      statistics[(slot * M + m) * 2] = row_max[m];
      statistics[(slot * M + m) * 2 + 1] = row_sum[m];
    }
  }
}

// Direct-Read Q3 PolarQuant Attention Tile
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile>
inline void splash_q3_attention_direct_tile(
    device const bfloat *tile_queries,
    device const SplashQ3Group *q3_keys,
    device const SplashQ3Group *q3_values,
    device const uint *page_table,
    uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *tg_scratch, // At least 8 floats for group reduction + 32 floats for scores
    uint thread_index) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = SplashCompressedHeadDimension; // 256
  constexpr uint N = SplashCompressedPageTokens;    // 32

  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_compressed_pages(visible_tokens);
  uint per_split = splash_compressed_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages) return;
  uint page_end = min(pages, page_begin + per_split);

  const uint d = thread_index; // 0..255
  const uint simd_id = d / 32; // 0..7 (group index)
  const uint lane_id = d % 32; // 0..31 (offset in group)
  const uint chunk = lane_id / 8;
  const uint elem_in_chunk = lane_id % 8;
  const uint bIdx = chunk * 3;

  float running_out[M];
  float row_max[M];
  float row_sum[M];
#pragma unroll
  for (uint m = 0; m < M; ++m) {
    running_out[m] = 0.0f;
    row_max[m] = -INFINITY;
    row_sum[m] = 0.0f;
  }

  for (uint page_step = page_begin; page_step < page_end; ++page_step) {
    uint physical = page_table[page_step];
    uint token_start = page_step * N;

    for (uint m = 0; m < M; ++m) {
      float q_val = float(tile_queries[m * D + d]);
      float q_group_sum = simd_sum(q_val);

      // Evaluate QK dot products across the 32 tokens of the page
      for (uint t = 0; t < N; ++t) {
        uint token = token_start + t;
        bool visible = (token < visible_tokens);

        if (visible) {
          ulong key_grp_idx = splash_compressed_group_index<KVHeads>(physical, kv_head, t, simd_id);
          device const SplashQ3Group &grp = q3_keys[key_grp_idx];
#ifdef __METAL_VERSION__
          uint code_word = 0;
          if (elem_in_chunk == 0) {
            code_word = uint(grp.packed[bIdx]) |
                        (uint(grp.packed[bIdx + 1]) << 8) |
                        (uint(grp.packed[bIdx + 2]) << 16);
          }
          code_word = simd_shuffle(code_word, ushort(lane_id & ~7u));
          float k_int = float((code_word >> (elem_in_chunk * 3)) & 0x07);
#else
          uchar b0 = grp.packed[bIdx];
          uchar b1 = grp.packed[bIdx + 1];
          uchar b2 = grp.packed[bIdx + 2];
          float k_int = unpack_q3(b0, b1, b2, elem_in_chunk);
#endif
          float int_term = q_val * k_int;
          float int_dot = simd_sum(int_term);
          float grp_score = int_dot * grp.scale + q_group_sum * grp.bias;

          if (lane_id == 0) {
            tg_scratch[t * 8 + simd_id] = grp_score;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (thread_index < N) {
        uint token = token_start + thread_index;
        if (token < visible_tokens) {
          float total = 0.0f;
          for (uint g = 0; g < 8; ++g) total += tg_scratch[thread_index * 8 + g];
          tg_scratch[256 + thread_index] = total * 0.0625f; // Scale by 1/sqrt(256)
        } else {
          tg_scratch[256 + thread_index] = -INFINITY;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Online Softmax step
      float local_max = -INFINITY;
      for (uint t = 0; t < N; ++t) {
        local_max = max(local_max, tg_scratch[256 + t]);
      }
      const float previous_max = row_max[m];
      const float next_max = max(previous_max, local_max);

      float local_sum = 0.0f;
      float probs[N];
      for (uint t = 0; t < N; ++t) {
        float score = tg_scratch[256 + t];
        probs[t] = (score != -INFINITY) ? fast::exp(score - next_max) : 0.0f;
        local_sum += probs[t];
      }

      const float rescale = (next_max == -INFINITY || next_max == previous_max)
                                ? 1.0f
                                : fast::exp(previous_max - next_max);
      running_out[m] = running_out[m] * rescale;
      row_sum[m] = row_sum[m] * rescale + local_sum;
      row_max[m] = next_max;

      // PV accumulation: each thread updates dimension d directly
      for (uint t = 0; t < N; ++t) {
        if (probs[t] > 0.0f) {
          ulong val_grp_idx = splash_compressed_group_index<KVHeads>(physical, kv_head, t, simd_id);
          device const SplashQ3Group &v_grp = q3_values[val_grp_idx];
#ifdef __METAL_VERSION__
          uint v_word = 0;
          if (elem_in_chunk == 0) {
            v_word = uint(v_grp.packed[bIdx]) |
                     (uint(v_grp.packed[bIdx + 1]) << 8) |
                     (uint(v_grp.packed[bIdx + 2]) << 16);
          }
          v_word = simd_shuffle(v_word, ushort(lane_id & ~7u));
          float v_int = float((v_word >> (elem_in_chunk * 3)) & 0x07);
#else
          uchar vb0 = v_grp.packed[bIdx];
          uchar vb1 = v_grp.packed[bIdx + 1];
          uchar vb2 = v_grp.packed[bIdx + 2];
          float v_int = unpack_q3(vb0, vb1, vb2, elem_in_chunk);
#endif
          float v_val = v_int * v_grp.scale + v_grp.bias;
          running_out[m] += probs[t] * v_val;
        }
      }
    }
  }

  // Write out partials and statistics
  for (uint m = 0; m < M; ++m) {
    partials[slot * M * D + m * D + d] = running_out[m];
    if (d == 0) {
      statistics[(slot * M + m) * 2] = row_max[m];
      statistics[(slot * M + m) * 2 + 1] = row_sum[m];
    }
  }
}
#endif // __METAL_VERSION__

