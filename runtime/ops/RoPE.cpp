#include "RoPE.hpp"

#include "metal/abi/RoPE.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {

void RoPE::addTables(metal::CommandGraph &graph, metal::MetalBuffer positions,
                     metal::MetalBuffer inverseFrequencies,
                     metal::MetalBuffer cosine, metal::MetalBuffer sine,
                     uint32_t rows, uint32_t maximumRows) {
  if (!rows || rows > maximumRows)
    throw std::invalid_argument("invalid RoPE table row count");
  const uint64_t elements = uint64_t{rows} * 32;
  graph.add("rope_build_tables",
            {std::move(positions), std::move(inverseFrequencies),
             std::move(cosine), std::move(sine)},
            RopeTableParams{rows}, {(elements + 255) / 256, 1, 1}, {256, 1, 1});
}

void RoPE::addDraftTables(metal::CommandGraph &graph,
                          metal::MetalBuffer positions,
                          metal::MetalBuffer inverseFrequencies,
                          metal::MetalBuffer cosine, metal::MetalBuffer sine,
                          uint32_t rows, uint32_t maximumRows) {
  if (!rows || rows > maximumRows)
    throw std::invalid_argument("invalid draft RoPE table row count");
  const uint64_t elements = uint64_t{rows} * 64;
  graph.add("rope_build_draft_tables",
            {std::move(positions), std::move(inverseFrequencies),
             std::move(cosine), std::move(sine)},
            DraftRopeTableParams{rows}, {(elements + 255) / 256, 1, 1},
            {256, 1, 1});
}

} // namespace splash::ops
