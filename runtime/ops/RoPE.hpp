#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

class RoPE final {
public:
  // Cosine and sine tables for `rows` target rows of (t, h, w) positions.
  static void addTables(metal::CommandGraph &graph, metal::MetalBuffer positions,
                        metal::MetalBuffer inverseFrequencies,
                        metal::MetalBuffer cosine, metal::MetalBuffer sine,
                        uint32_t rows, uint32_t maximumRows);
  // Cosine and sine tables for `rows` draft rows of scalar positions.
  static void addDraftTables(metal::CommandGraph &graph,
                             metal::MetalBuffer positions,
                             metal::MetalBuffer inverseFrequencies,
                             metal::MetalBuffer cosine,
                             metal::MetalBuffer sine, uint32_t rows,
                             uint32_t maximumRows);
};

} // namespace splash::ops
