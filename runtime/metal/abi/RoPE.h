#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Separate from ops::RoPETableShape so host-only fields cannot change the ABI.
struct RopeTableParams {
  uint32_t target_rows;
};

static_assert(sizeof(RopeTableParams) == 4,
              "RoPE table parameters are 4 bytes on both sides");

struct DraftRopeTableParams {
  uint32_t draft_rows;
};

static_assert(sizeof(DraftRopeTableParams) == 4,
              "Draft RoPE table parameters are 4 bytes on both sides");
