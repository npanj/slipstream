#include "metal/kernels/common/compressed_kv_tile.h"

// Compressed KV Attention (Q4 Packed & Q3 PolarQuant)
// Executes direct-read dot-product and online softmax in registers without
// intermediate float scratch allocations or decompress-to-DRAM passes.

struct SplashCompressedVerifyTile {
  device const bfloat *queries;
  device const uint *page_table;
  ulong slot;
  uint kv_head;
  uint split;
  uint splits;
  uint committed_tokens;
  uint active_rows;
  bool active;
};

template <uint KVHeads, uint QueryHeadsPerKVHead>
inline SplashCompressedVerifyTile splash_compressed_verify_tile_at(
    device const bfloat *queries, device const uint *page_table0,
    device const uint *page_table1, device const uint *page_table2,
    device const uint *page_table3,
    constant SplashQ8VerifyAttentionParams *params, uint3 group) {
  constexpr uint D = SplashCompressedHeadDimension;
  SplashCompressedVerifyTile tile{};
  uint kv_head = group.x;
  uint split = group.y;
  uint batch = group.z;
  constant SplashQ8VerifyAttentionParams &lane_params = params[batch];
  if (!splash_compressed_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || split >= lane_params.split_count)
    return tile;
  ulong group_stride = ulong(lane_params.chunk_stride) * QueryHeadsPerKVHead * D;
  tile.queries = queries + (ulong(batch) * KVHeads + kv_head) * group_stride;
  tile.page_table =
      batch == 0 ? page_table0
                 : (batch == 1 ? page_table1
                               : (batch == 2 ? page_table2 : page_table3));
  tile.slot =
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits + split;
  tile.kv_head = kv_head;
  tile.split = split;
  tile.splits = lane_params.split_count;
  tile.committed_tokens = lane_params.committed_tokens;
  tile.active_rows = lane_params.active_rows;
  tile.active = true;
  return tile;
}

#define COMPRESSED_VERIFY_SIGNATURE(Name, GroupType)                            \
  kernel void Name(                                                             \
      device const bfloat *queries [[buffer(0)]],                               \
      device const GroupType *keys [[buffer(1)]],                               \
      device const GroupType *values [[buffer(2)]],                             \
      device float *partials [[buffer(3)]],                                     \
      device float *statistics [[buffer(4)]],                                   \
      device const uint *page_table0 [[buffer(5)]],                             \
      device const uint *page_table1 [[buffer(6)]],                             \
      device const uint *page_table2 [[buffer(7)]],                             \
      device const uint *page_table3 [[buffer(8)]],                             \
      constant SplashQ8VerifyAttentionParams *params [[buffer(9)]],             \
      uint3 group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]])

#define COMPRESSED_VERIFY_TILE_AT(Heads, Group)                                 \
  const SplashCompressedVerifyTile tile =                                       \
      splash_compressed_verify_tile_at<Heads, Group>(                           \
          queries, page_table0, page_table1, page_table2, page_table3, params,  \
          group);                                                               \
  if (!tile.active) return;

#define Q4_VERIFY_SPLIT(Name, Heads, Group)                                     \
  COMPRESSED_VERIFY_SIGNATURE(Name, SplashQ4Group) {                            \
    threadgroup float tg_scratch[288];                                          \
    COMPRESSED_VERIFY_TILE_AT(Heads, Group)                                     \
    splash_q4_attention_direct_tile<Heads, Group, SPLASH_TARGET_VERIFY_ROWS>(   \
        tile.queries, keys, values, tile.page_table, tile.kv_head,              \
        tile.committed_tokens, tile.active_rows, tile.splits, tile.split,       \
        partials, statistics, tile.slot, tg_scratch, thread_index);             \
  }

#define Q3_VERIFY_SPLIT(Name, Heads, Group)                                     \
  COMPRESSED_VERIFY_SIGNATURE(Name, SplashQ3Group) {                            \
    threadgroup float tg_scratch[288];                                          \
    COMPRESSED_VERIFY_TILE_AT(Heads, Group)                                     \
    splash_q3_attention_direct_tile<Heads, Group, SPLASH_TARGET_VERIFY_ROWS>(   \
        tile.queries, keys, values, tile.page_table, tile.kv_head,              \
        tile.committed_tokens, tile.active_rows, tile.splits, tile.split,       \
        partials, statistics, tile.slot, tg_scratch, thread_index);             \
  }

Q4_VERIFY_SPLIT(verify_attention_q4_split, 4, 6)
Q4_VERIFY_SPLIT(verify_attention_q4_split_kv2_g8, 2, 8)
Q4_VERIFY_SPLIT(verify_attention_q4_split_kv2_g12, 2, 12)

Q3_VERIFY_SPLIT(verify_attention_q3_split, 4, 6)
Q3_VERIFY_SPLIT(verify_attention_q3_split_kv2_g8, 2, 8)
Q3_VERIFY_SPLIT(verify_attention_q3_split_kv2_g12, 2, 12)

#undef Q3_VERIFY_SPLIT
#undef Q4_VERIFY_SPLIT
#undef COMPRESSED_VERIFY_TILE_AT
#undef COMPRESSED_VERIFY_SIGNATURE
