#include "metal/abi/KernelABI.h"

// Block selection for the qwen4exp sparse-attention indexer.
//
// The scoring kernel leaves one score per visible block; this takes the
// highest `budget / compress_ratio` of them. At the production budget that
// is 512 blocks out of as many as 65,536, which is well past the point where
// sorting is sensible.
//
// Scores are a sum of relu terms and so are never negative: for non-negative
// floats the IEEE bit pattern orders the same way the value does.
// An 8-bit radix histogram settles the exact threshold in only four passes
// (shift = 24, 16, 8, 0) using threadgroup atomic bins.
//
// Ties at the threshold are broken by block index, ascending, so a query's
// selection does not depend on which thread happened to get there first.

struct QsaSelectParams {
  uint blocks;
  uint budget;
};

constant constexpr uint kThreads = 1024;
constant constexpr uint kSimdgroups = kThreads / 32;

kernel void qsa_select_blocks(
    device const float *scores [[buffer(0)]],
    device uint *selected [[buffer(1)]],
    device atomic_uint *selected_count [[buffer(2)]],
    constant QsaSelectParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint histogram[256];
  threadgroup uint shared[4];
  threadgroup uint partial[kSimdgroups];
  threadgroup uint simd_bases[kSimdgroups];

  const uint wanted = min(params.budget, params.blocks);
  if (thread_index == 0) {
    atomic_store_explicit(selected_count, wanted, memory_order_relaxed);
  }

  // Fast path: if all blocks fit within the budget, take all of them.
  if (params.blocks <= params.budget) {
    for (uint i = thread_index; i < params.blocks; i += kThreads) {
      selected[i] = i;
    }
    return;
  }

  // Radix-select the top-wanted score in 4 passes over 8-bit shifts: 24, 16, 8, 0.
  uint prefix = 0u;
  uint prefix_mask = 0u;
  uint pass_wanted = wanted;

  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint i = thread_index; i < 256; i += kThreads) {
      atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = thread_index; i < params.blocks; i += kThreads) {
      const uint v = as_type<uint>(scores[i]);
      if ((v & prefix_mask) == prefix) {
        atomic_fetch_add_explicit(&histogram[(v >> shift) & 255u], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_index == 0) {
      uint above = 0u;
      uint bin = 255u;
      for (;; --bin) {
        const uint count = atomic_load_explicit(&histogram[bin], memory_order_relaxed);
        if (above + count >= pass_wanted || bin == 0u) {
          shared[0] = bin;
          shared[1] = pass_wanted - above;
          break;
        }
        above += count;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    prefix |= shared[0] << uint(shift);
    prefix_mask |= 255u << uint(shift);
    pass_wanted = shared[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint threshold = prefix;
  const uint ties_wanted = pass_wanted;
  const uint strictly_above_wanted = wanted - ties_wanted;

  // Emit all elements strictly above the threshold deterministically.
  uint mine_above = 0u;
  for (uint i = thread_index; i < params.blocks; i += kThreads) {
    if (as_type<uint>(scores[i]) > threshold) ++mine_above;
  }
  const uint within_simd_above = simd_prefix_exclusive_sum(mine_above);
  if (simd_lane == 31) partial[simd_group] = within_simd_above + mine_above;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (thread_index == 0) {
    uint running = 0u;
    for (uint s = 0; s < kSimdgroups; ++s) {
      simd_bases[s] = running;
      running += partial[s];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint slot_above = simd_bases[simd_group] + within_simd_above;
  for (uint i = thread_index; i < params.blocks; i += kThreads) {
    if (as_type<uint>(scores[i]) > threshold) {
      if (slot_above < strictly_above_wanted) {
        selected[slot_above] = i;
      }
      ++slot_above;
    }
  }

  // Emit ties at the threshold in ascending index order in parallel chunks.
  if (ties_wanted > 0) {
    threadgroup uint ties_taken;
    if (thread_index == 0) ties_taken = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint chunk = 0; chunk < params.blocks; chunk += kThreads) {
      if (ties_taken >= ties_wanted) break;
      const uint i = chunk + thread_index;
      const uint is_tie = (i < params.blocks && as_type<uint>(scores[i]) == threshold) ? 1u : 0u;
      const uint within_simd_tie = simd_prefix_exclusive_sum(is_tie);
      if (simd_lane == 31) partial[simd_group] = within_simd_tie + is_tie;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (thread_index == 0) {
        uint running = 0u;
        for (uint s = 0; s < kSimdgroups; ++s) {
          simd_bases[s] = running;
          running += partial[s];
        }
        shared[2] = running; // total ties in this chunk
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint rank_in_chunk = simd_bases[simd_group] + within_simd_tie;
      if (is_tie && (ties_taken + rank_in_chunk < ties_wanted)) {
        selected[strictly_above_wanted + ties_taken + rank_in_chunk] = i;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (thread_index == 0) {
        ties_taken += shared[2];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
