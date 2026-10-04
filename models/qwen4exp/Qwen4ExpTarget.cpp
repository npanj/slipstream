#include "models/qwen4exp/Qwen4ExpTarget.hpp"
#include "AwakeClock.hpp"

#include "models/qwen4exp/abi/HyperConnection.h"
#include "metal/abi/MoE.h"
#include "models/qwen4exp/abi/PerLayerEmbedding.h"
#include "model/WeightStore.hpp"
#include "ops/Embedding.hpp"
#include "ops/GDN.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PromptLookup.hpp"

#include <algorithm>
#include <bitset>
#include <random>
#include <cmath>
#include <map>
#include <deque>
#include <functional>
#include <cstdio>
#include <cstring>
#include <bit>
#include <cstdlib>
#include <dispatch/dispatch.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

namespace splash::model {
namespace {

// Matches kMoeSkippedRoute in the MoE kernels: a route with no expert.
constexpr uint32_t kMoeSkippedRoute = 0xFFFFFFFFu;

// qwen4exp gates its linear-attention output with a sigmoid (the checkpoint's
// config sets output_gate_type), where Qwen3.8 uses silu. Everything else about
// the GDN shape is shared, so only this flag differs.
[[nodiscard]] ops::GdnShape gdnShape(const QwenTargetGeometry &geometry,
                                     uint32_t treeParents = 0) {
  ops::GdnShape shape = geometry.gdnShape();
  shape.sigmoidGate = true;
  shape.treeParents = treeParents;
  return shape;
}

// ---------------------------------------------------------------------------
// Per-layer embedding.
//
// Added to the residual entering layer geometry.pleLayer:
//
//   ids      = hashed n-grams of each token and its two predecessors
//   emb      = one gathered row per head, concatenated        rows x 2560
//   keys     = emb @ key projection                           rows x 10240
//   values   = emb @ value projection                         rows x 2560
//   gated    = gate(keys, values, residual)                   rows x 10240
//   residual += gated + silu(dilated conv over normalized gated)
//
// The n-gram hash reads two tokens of history and the convolution nine
// normalized rows, both kept in the GDN cell's auxiliary region.
// ---------------------------------------------------------------------------

PerLayerEmbeddingStateParams pleStateParams(const QwenTargetGeometry &geometry,
                                            uint32_t rows) {
  return {rows, geometry.residualWidth(), geometry.pleHistoryRows,
          geometry.pleEndToken, rows, 0, 0, 0};
}

// One sequence's shifted token rows and gathered n-gram embedding rows.
void gatherNgramRowsOnCpu(const Qwen4ExpWeights &weights,
                          const QwenTargetGeometry &geometry,
                          const uint32_t *tokens, const uint8_t *state,
                          uint16_t *output, uint32_t rows, uint32_t groupRows,
                          uint32_t liveRows);

// `onCpu` requires every earlier submission to have completed, so the token
// ids and stored history can be read; otherwise the lookup is a GPU kernel,
// which is correct but pays for keeping the whole table resident.
void addPleGather(const Qwen4ExpWeights &weights, metal::MetalBackend &backend,
                  metal::CommandGraph &graph, const QwenTargetPleBuffers &ple,
                  const QwenTargetGeometry &geometry, metal::MetalBuffer state,
                  uint32_t rowBegin, uint32_t rows, bool onCpu,
                  uint32_t liveRows,
                  std::vector<std::function<void()>> *deferred = nullptr) {
  if (onCpu) {
    auto gather = [&weights, &geometry, tokens = ple.tokens, state,
                   embedding = ple.embedding, rowBegin, rows, liveRows] {
      gatherNgramRowsOnCpu(
          weights, geometry,
          static_cast<const uint32_t *>(tokens.contents()) + rowBegin,
          static_cast<const uint8_t *>(state.contents()),
          static_cast<uint16_t *>(embedding.contents()) +
              uint64_t{rowBegin} * geometry.pleEmbeddingSize,
          rows, rows, liveRows);
    };
    // A pipelined step encodes every layer before its inputs exist; the
    // gather then runs between stages, once they do.
    if (deferred)
      deferred->push_back(std::move(gather));
    else
      gather();
    return;
  }
  const Qwen4ExpLayout &layout = weights.layout;
  const auto &table = weights.perLayerEmbedding;
  const PerLayerEmbeddingStateParams params = pleStateParams(geometry, rows);
  metal::MetalBuffer tokens =
      backend.view(ple.tokens, uint64_t{rowBegin} * 4, uint64_t{rows} * 4);
  metal::MetalBuffer shifted = backend.view(
      ple.shifted, uint64_t{rowBegin} * 3 * 4, uint64_t{rows} * 3 * 4);
  graph.add("per_layer_embedding_shift", {tokens, std::move(state), shifted},
            params, {(rows + 63) / 64, 1, 1}, {64, 1, 1});
  const NgramEmbeddingParams gather{rows, layout.ngramHeads(),
                                    layout.ngramHeadDimension(),
                                    layout.ngramSize, layout.ngramHeadsPerOrder,
                                    kQ4FineGroupElements};
  const uint64_t embeddingRow = uint64_t{geometry.pleEmbeddingSize} * 2;
  graph.add("ngram_embedding_gather",
            {std::move(shifted), table.layerMultipliers,
             table.headVocabularySizes, table.headOffsets, table.table.weights,
             table.table.scales, table.table.biases,
             backend.view(ple.embedding, rowBegin * embeddingRow,
                          rows * embeddingRow)},
            gather, {rows, 1, 1}, {128, 1, 1});
}

// The n-gram table lookup, on the CPU. The table is 27 GB and a GPU kernel
// that binds it forces all of it resident; a row per head per token is 80
// bytes read from the mapped file. Rows at or past `liveRows` of each group
// of `groupRows` are zero-filled: their output is discarded.
//
// Must run only once the token ids and the stored history are final, which
// the staged execution guarantees at the point layer pleLayer is encoded.
void gatherNgramRowsOnCpu(const Qwen4ExpWeights &weights,
                          const QwenTargetGeometry &geometry,
                          const uint32_t *tokens, const uint8_t *state,
                          uint16_t *output, uint32_t rows, uint32_t groupRows,
                          uint32_t liveRows) {
  const Qwen4ExpLayout &layout = weights.layout;
  const auto &ple = weights.perLayerEmbedding;
  const uint32_t heads = layout.ngramHeads();
  const uint32_t dimension = layout.ngramHeadDimension();
  const uint32_t groups = dimension / kQ4FineGroupElements;
  const auto *offsets = static_cast<const int64_t *>(ple.headOffsets.contents());
  const auto *vocabulary =
      static_cast<const int64_t *>(ple.headVocabularySizes.contents());
  const auto *multipliers =
      static_cast<const int64_t *>(ple.layerMultipliers.contents());
  const auto *packed = static_cast<const uint8_t *>(ple.table.weights.contents());
  const auto *scales = static_cast<const uint16_t *>(ple.table.scales.contents());
  const auto *biases = static_cast<const uint16_t *>(ple.table.biases.contents());
  const uint32_t eos = geometry.pleEndToken;
  const auto *header = reinterpret_cast<const uint32_t *>(state);
  const bool valid = header[0] != 0;
  const uint32_t older = valid ? header[1] : eos;
  const uint32_t newer = valid ? header[2] : eos;
  auto widen = [](uint16_t bits) {
    return std::bit_cast<float>(uint32_t{bits} << 16);
  };
  auto narrow = [](float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
  };
  const uint64_t width = uint64_t{heads} * dimension;
  dispatch_apply(rows, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t index) {
    const uint32_t row = static_cast<uint32_t>(index);
    uint16_t *destination = output + row * width;
    if (row % groupRows >= liveRows) {
      std::fill_n(destination, width, uint16_t{0});
      return;
    }
    // Same window as per_layer_embedding_shift: tokens restart after an
    // end-of-sequence token, and the stored pair precedes the first row.
    const uint32_t base = row - row % groupRows;
    const uint32_t local = row - base;
    const uint32_t current = tokens[row];
    const uint32_t oneBack = local >= 1 ? tokens[row - 1] : newer;
    const uint32_t twoBackRaw =
        local >= 2 ? tokens[row - 2] : (local == 1 ? newer : older);
    const uint32_t twoBack = oneBack == eos ? eos : twoBackRaw;
    const uint32_t shifted[3] = {current, oneBack, twoBack};
    for (uint32_t head = 0; head < heads; ++head) {
      const uint32_t order = head / layout.ngramHeadsPerOrder + 2;
      // Wrapping 64-bit arithmetic, as the reference's int64 tensors wrap.
      uint64_t mixed = uint64_t{shifted[0]} * uint64_t(multipliers[0]);
      for (uint32_t position = 1; position < order; ++position)
        mixed ^= uint64_t{shifted[position]} * uint64_t(multipliers[position]);
      int64_t remainder = static_cast<int64_t>(mixed) % vocabulary[head];
      if (remainder < 0)
        remainder += vocabulary[head];
      const uint64_t entry = uint64_t(remainder + offsets[head]);
      const uint8_t *codes = packed + entry * dimension / 2;
      for (uint32_t i = 0; i < dimension; ++i) {
        const float code = float((codes[i / 2] >> ((i & 1) * 4)) & 0xF);
        const uint64_t parameter = entry * groups + i / kQ4FineGroupElements;
        destination[head * dimension + i] =
            narrow(code * widen(scales[parameter]) + widen(biases[parameter]));
      }
    }
  });
}


// SPLASH_CAPTURE_LAYERS=11,23,47 keeps the residual after those prompt layers
// (every row of the chunk, all four streams) for guesser training. Measurement
// only: it adds one copy per listed layer and changes no result.
struct PrefillCapture {
  std::vector<uint32_t> layers;
  std::vector<metal::MetalBuffer> buffers;
  uint32_t rows = 0;
};

PrefillCapture &prefillCapture() {
  static PrefillCapture capture = [] {
    PrefillCapture result;
    if (const char *list = std::getenv("SPLASH_CAPTURE_LAYERS")) {
      for (const char *cursor = list; *cursor;) {
        char *end = nullptr;
        const unsigned long layer = std::strtoul(cursor, &end, 10);
        if (end == cursor)
          throw std::invalid_argument("SPLASH_CAPTURE_LAYERS: expected a layer list");
        result.layers.push_back(static_cast<uint32_t>(layer));
        cursor = *end == ',' ? end + 1 : end;
      }
    }
    return result;
  }();
  return capture;
}

void addPrefillCapture(metal::CommandGraph &graph, metal::MetalBackend &backend,
                       const QwenTargetGeometry &geometry, uint32_t layerIndex,
                       const metal::MetalBuffer &residual, uint32_t rows) {
  PrefillCapture &capture = prefillCapture();
  const auto found =
      std::find(capture.layers.begin(), capture.layers.end(), layerIndex);
  if (found == capture.layers.end())
    return;
  if (layerIndex >= geometry.layers)
    throw std::invalid_argument("SPLASH_CAPTURE_LAYERS: layer out of range");
  const size_t tap = static_cast<size_t>(found - capture.layers.begin());
  const uint64_t rowBytes = uint64_t{geometry.residualWidth()} * sizeof(uint16_t);
  if (capture.buffers.empty())
    for (size_t i = 0; i < capture.layers.size(); ++i)
      capture.buffers.push_back(backend.allocateBuffer(
          ExecutionLimits::prefillTokenBudget * rowBytes,
          metal::BufferStorage::Shared, "prefill-capture"));
  const uint32_t count = rows * geometry.residualWidth();
  graph.add("capture_rows",
            {backend.view(residual, 0, rows * rowBytes), capture.buffers[tap]},
            count, {256, 1, 1}, {256, 1, 1});
  capture.rows = rows;
}

// SPLASH_ROUTE_LOG=path appends every routed expert choice, one line per
// (phase, layer, row): "P|D layer e0 e1 ... e9". Measurement only; it lets
// cache sizes and policies be replayed offline against real routing.
void logRoutes(char phase, uint32_t layer, const uint32_t *selected,
               uint32_t rows, uint32_t routesPerRow, uint32_t perToken,
               uint32_t liveRows, uint32_t groupRows) {
  static FILE *file = [] {
    const char *path = std::getenv("SPLASH_ROUTE_LOG");
    return path ? std::fopen(path, "a") : nullptr;
  }();
  if (!file)
    return;
  for (uint32_t row = 0; row < rows; ++row) {
    if (row % groupRows >= liveRows)
      continue;
    std::fprintf(file, "%c %u", phase, layer);
    for (uint32_t k = 0; k < perToken; ++k)
      std::fprintf(file, " %u", selected[row * routesPerRow + k]);
    std::fputc('\n', file);
  }
}

// The hyper-connection mix in front of a block: normalize the streams, reduce
// them (and compute the injection gates), then mix them into the block input.
// Weight-stationary kernels: each weight is read once for all rows.
void addHyperConnection(metal::CommandGraph &graph,
                        const QwenTargetGeometry &geometry,
                        metal::MetalBuffer input,
                        const Qwen4ExpHyperConnection &weights,
                        metal::MetalBuffer normalized,
                        metal::MetalBuffer reduced, metal::MetalBuffer mixed,
                        metal::MetalBuffer injection, uint32_t rows,
                        uint32_t rowStep = 1,
                        const ops::ExecutionPlans *operators = nullptr) {
  const bool withInject = weights.blockInject.has_value();
  const uint32_t splits = rows <= kHyperDownSplitRows ? kHyperDownSplits : 1;
  const HyperConnectionParams params{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f, withInject ? 1u : 0u, rowStep,
      splits};
  graph.add("hyper_connection_rms", {input, weights.norm, normalized},
            params, {rows, geometry.hyperConnectionCount, 1}, {256, 1, 1});
  const uint32_t outputs = geometry.hyperConnectionLowRank +
                           (withInject ? geometry.hyperConnectionCount : 0);
  // A prompt's rows run the projections as 8-bit matrix products over the
  // same tiled weights; the scalar kernels below are for decode's few rows.
  static const bool scalarPrompt = std::getenv("SPLASH_HC_SCALAR_PROMPT") != nullptr;
  if (operators && !scalarPrompt && rows > kHyperDownSplitRows && rowStep <= 1) {
    const uint32_t width = geometry.hyperConnectionCount * geometry.hiddenSize;
    const uint32_t lowRank = geometry.hyperConnectionLowRank;
    const ops::LinearMatrix down{kHyperDownPadded, width};
    const ops::LinearMatrix up{width, lowRank};
    const auto &linear = operators->linear();
    linear.addPrefillSums(graph, normalized, weights.promptSums, down, rows);
    linear.addPrefill(graph, normalized, weights.mixDown, weights.promptDown,
                      weights.promptSums, down, rows);
    static_assert(kHyperDownPadded == 512, "hyper_connection_prompt_reduce assumes 512");
    graph.add("hyper_connection_prompt_reduce", {weights.promptDown, reduced},
              params, {(rows * lowRank + 255) / 256, 1, 1}, {256, 1, 1});
    if (withInject)
      graph.add("hyper_connection_prompt_inject",
                {normalized, *weights.blockInject, std::move(injection)}, params,
                {rows, geometry.hyperConnectionCount, 1}, {256, 1, 1});
    linear.addPrefillSums(graph, reduced, weights.promptSums, up, rows);
    linear.addPrefill(graph, reduced, weights.mixUp, weights.promptLogits,
                      weights.promptSums, up, rows);
    graph.add("hyper_connection_prompt_mix",
              {std::move(normalized), weights.promptLogits, std::move(mixed)},
              params, {(rows * geometry.hiddenSize + 255) / 256, 1, 1}, {256, 1, 1});
    return;
  }
  // SPLASH_HC_REPEAT=n encodes the decode mix n times. Every pass rewrites
  // the same outputs from the same inputs, so answers are unchanged and the
  // added GPU time is what the mix costs. Measurement only.
  static const uint32_t repeat = [] {
    const char *value = std::getenv("SPLASH_HC_REPEAT");
    return value ? static_cast<uint32_t>(std::clamp(std::atoi(value), 1, 8)) : 1u;
  }();
  // SPLASH_HC_REPEAT_PARTS picks which repeat: 1 normalize, 2 down,
  // 4 down-finish, 8 up-mix (default all).
  static const uint32_t parts = [] {
    const char *value = std::getenv("SPLASH_HC_REPEAT_PARTS");
    return value ? static_cast<uint32_t>(std::atoi(value)) : 15u;
  }();
  for (uint32_t pass = 1; pass < repeat; ++pass) {
    if (parts & 1)
    graph.add("hyper_connection_rms", {input, weights.norm, normalized},
              params, {rows, geometry.hyperConnectionCount, 1}, {256, 1, 1});
    if (parts & 2)
    graph.add("hyper_connection_down",
              {normalized, weights.mixDown.weights, weights.mixDown.scales,
               weights.mixDown.biases,
               withInject ? *weights.blockInject : weights.mixDown.scales, reduced,
               withInject ? injection : reduced, weights.downPartials},
              params, {(outputs + 7) / 8, splits, 1}, {256, 1, 1});
    if (splits > 1 && (parts & 4))
      graph.add("hyper_connection_down_finish",
                {weights.downPartials, reduced, withInject ? injection : reduced},
                params, {(rows * outputs + 255) / 256, 1, 1}, {256, 1, 1});
    if (parts & 8)
    graph.add("hyper_connection_up_mix",
              {normalized, reduced, weights.mixUp.weights,
               weights.mixUp.scales, weights.mixUp.biases, mixed},
              params, {geometry.hiddenSize / 8, 1, 1}, {256, 1, 1});
  }
  graph.add("hyper_connection_down",
            {normalized, weights.mixDown.weights, weights.mixDown.scales,
             weights.mixDown.biases,
             withInject ? *weights.blockInject : weights.mixDown.scales, reduced,
             withInject ? injection : reduced, weights.downPartials},
            params, {(outputs + 7) / 8, splits, 1}, {256, 1, 1});
  if (splits > 1)
    graph.add("hyper_connection_down_finish",
              {weights.downPartials, reduced, withInject ? injection : reduced},
              params, {(rows * outputs + 255) / 256, 1, 1}, {256, 1, 1});
  graph.add("hyper_connection_up_mix",
            {std::move(normalized), std::move(reduced), weights.mixUp.weights,
             weights.mixUp.scales, weights.mixUp.biases, std::move(mixed)},
            params, {geometry.hiddenSize / 8, 1, 1}, {256, 1, 1});
}

// Reads per matrix of one expert (SPLASH_READ_PIECES, default 1). A slot's
// read counter starts at 3 * this and each finished piece takes one off.
uint32_t readPieces() noexcept {
  static const uint32_t pieces = [] {
    const char *value = std::getenv("SPLASH_READ_PIECES");
    return value ? static_cast<uint32_t>(std::clamp(std::atoi(value), 1, 16)) : 1u;
  }();
  return pieces;
}

// Asynchronously advises the macOS XNU kernel (via F_RDADVISE) to begin
// background NVMe DMA transfers for missed expert matrices into the unified
// buffer cache before pread is called.
template <class Miss>
void adviseMissedExperts(const Qwen4ExpExpertSource &source, const Miss *misses,
                         size_t count, uint64_t stride) {
#if defined(F_RDADVISE)
  static const bool enabled = [] {
    const char *v = std::getenv("SPLASH_DISABLE_RDADVISE");
    return !(v && std::atoi(v) != 0);
  }();
  if (!enabled || !count || source.fd < 0)
    return;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t expert = misses[i].expert;
    struct radvisory ra;
    ra.ra_count = static_cast<int>(std::min<uint64_t>(stride, static_cast<uint64_t>(INT_MAX)));
    ra.ra_offset = static_cast<off_t>(source.gate + expert * stride);
    (void)::fcntl(source.fd, F_RDADVISE, &ra);
    ra.ra_offset = static_cast<off_t>(source.up + expert * stride);
    (void)::fcntl(source.fd, F_RDADVISE, &ra);
    ra.ra_offset = static_cast<off_t>(source.down + expert * stride);
    (void)::fcntl(source.fd, F_RDADVISE, &ra);
  }
#else
  (void)source;
  (void)misses;
  (void)count;
  (void)stride;
#endif
}

// Reads missed experts straight from the layer file into their cache slots:
// three reads per expert (gate, up, down), all in flight together. The SSD
// reaches ~15 GB/s this way; faulting pages in one at a time reached ~1.5.
template <class Miss>
void readMissedExperts(const Qwen4ExpExpertSource &source, const Miss *misses,
                       size_t count, uint64_t stride, char *gate, char *up,
                       char *down, std::atomic<uint32_t> *slotReads = nullptr) {
  if (!count)
    return;
  // One read per matrix. Splitting each into 4 smaller reads in flight
  // together was measured slower (staging 30 -> 39 ms a step).
  const uint32_t kPieces = readPieces();
  const uint64_t piece = (stride + kPieces - 1) / kPieces;
  dispatch_apply(count * 3 * kPieces,
                 dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t index) {
    const Miss &miss = misses[index / (3 * kPieces)];
    const uint32_t matrix = static_cast<uint32_t>(index / kPieces % 3);
    const uint64_t begin = uint64_t(index % kPieces) * piece;
    const uint64_t length = std::min(piece, stride - std::min(stride, begin));
    if (!length)
      return;
    char *slot = (matrix == 0 ? gate : matrix == 1 ? up : down) +
                 uint64_t{miss.slot} * stride + begin;
    const uint64_t offset =
        (matrix == 0 ? source.gate : matrix == 1 ? source.up : source.down) +
        uint64_t{miss.expert} * stride + begin;
    uint64_t done = 0;
    while (done < length) {
      const ssize_t got = ::pread(source.fd, slot + done, length - done,
                                  static_cast<off_t>(offset + done));
      if (got <= 0)
        break; // Leaves the slot short; the file was validated at load.
      done += static_cast<uint64_t>(got);
    }
    if (slotReads)
      slotReads[miss.slot].fetch_sub(1, std::memory_order_release);
  });
}

// Top-k of one row of the router's bf16 scores, as moe_route_select picks
// them: highest first, ties to the lower expert id; weights are the softmax
// over just the k chosen scores.
void topExperts(const uint16_t *scores, uint32_t experts, uint32_t k,
                uint32_t *ids, float *weights) {
  auto widen = [](uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); };
  float chosen[16];
  // One pass, keeping the k best in order. Scanning ids upward and moving a
  // value past only strictly smaller ones keeps ties at the lower id.
  uint32_t filled = 0;
  for (uint32_t e = 0; e < experts; ++e) {
    const float v = widen(scores[e]);
    if (filled == k && !(v > chosen[k - 1])) continue;
    uint32_t at = filled < k ? filled++ : k - 1;
    while (at > 0 && chosen[at - 1] < v) {
      chosen[at] = chosen[at - 1];
      ids[at] = ids[at - 1];
      --at;
    }
    chosen[at] = v;
    ids[at] = e;
  }
  if (!weights) return;
  float total = 0.0f;
  for (uint32_t r = 0; r < k; ++r) total += std::exp(chosen[r] - chosen[0]);
  for (uint32_t r = 0; r < k; ++r) weights[r] = std::exp(chosen[r] - chosen[0]) / total;
}

// sigmoid of the shared expert's scalar gate: a Q8 projection padded to 256
// outputs, of which output 0 is used (see moe_route_select_impl).
float sharedExpertGate(const ops::Q8Projection &gate, const uint16_t *input,
                       uint32_t size) {
  constexpr uint32_t kStorageN = 256;
  const auto *w = static_cast<const uint8_t *>(gate.weights.contents());
  const auto *scales = static_cast<const uint16_t *>(gate.scales.contents());
  const auto *biases = static_cast<const uint16_t *>(gate.biases.contents());
  auto widen = [](uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); };
  float total = 0.0f;
  for (uint32_t d = 0; d < size; ++d) {
    const uint32_t group = d / 64;
    const float value = float(w[uint64_t{group} * kStorageN * 64 + d % 64]) *
                            widen(scales[uint64_t{group} * kStorageN]) +
                        widen(biases[uint64_t{group} * kStorageN]);
    total += widen(input[d]) * value;
  }
  return 1.0f / (1.0f + std::exp(-total));
}

uint16_t toBf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

constexpr uint32_t kFrequencyHalfLife = 512;

// Counts a use of `expert`, halving every count each kFrequencyHalfLife steps.
void countExpertUse(Qwen4ExpLayerExpertCache &cache, uint32_t expert) {
  if (cache.frequency.empty())
    cache.frequency.assign(cache.expertToSlot.size(), 0);
  if (cache.clock >= cache.nextHalving) {
    for (auto &count : cache.frequency) count >>= 1;
    cache.nextHalving = cache.clock + kFrequencyHalfLife;
  }
  cache.frequency[expert] += 16;
}

// The slot to evict: least used, then least recent, never one this step uses.
// SPLASH_EXPERT_EVICT=lru ignores use counts (least recent only).
int32_t pickVictim(const Qwen4ExpLayerExpertCache &cache) {
  static const bool leastRecentOnly = [] {
    const char *value = std::getenv("SPLASH_EXPERT_EVICT");
    return value && std::string_view(value) == "lru";
  }();
  int32_t best = -1;
  uint64_t bestKey = UINT64_MAX;
  for (uint32_t s = 0; s < cache.capacity; ++s) {
    if (cache.lruTime[s] == cache.clock) continue;
    if (cache.slotReads && cache.slotReads[s].load(std::memory_order_acquire)) continue;
    const int16_t expert = cache.slotToExpert[s];
    const uint64_t count = expert >= 0 && !cache.frequency.empty() && !leastRecentOnly
                               ? cache.frequency[expert] : 0;
    const uint64_t key = (count << 32) | cache.lruTime[s];
    if (key < bestKey) { bestKey = key; best = static_cast<int32_t>(s); }
  }
  return best;
}

// One sequence's gate, convolution and residual update. `normalized` holds
// this sequence's history rows followed by its current rows.
void addPleApply(const Qwen4ExpWeights &weights, metal::MetalBackend &backend,
                 metal::CommandGraph &graph, const QwenTargetPleBuffers &ple,
                 const QwenTargetGeometry &geometry, metal::MetalBuffer state,
                 metal::MetalBuffer normalized, metal::MetalBuffer residual,
                 uint32_t rowBegin, uint32_t rows) {
  const auto &table = weights.perLayerEmbedding;
  const uint64_t width = uint64_t{geometry.residualWidth()} * 2;
  const uint64_t hidden = uint64_t{geometry.hiddenSize} * 2;
  const PerLayerEmbeddingStateParams stateParams =
      pleStateParams(geometry, rows);
  const PerLayerEmbeddingParams params{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      weights.layout.pleConvolutionTaps, weights.layout.ngramSize, 1e-6f};
  metal::MetalBuffer gated =
      backend.view(ple.gated, rowBegin * width, rows * width);
  graph.add("per_layer_embedding_history", {std::move(state), normalized},
            stateParams, {64, 1, 1}, {256, 1, 1});
  graph.add("per_layer_embedding_gate",
            {backend.view(ple.keys, rowBegin * width, rows * width),
             backend.view(ple.values, rowBegin * hidden, rows * hidden),
             residual, table.keyNorm, table.queryNorm, table.convolutionNorm,
             gated, normalized},
            params, {rows, 1, 1}, {256, 1, 1});
  graph.add("per_layer_embedding_convolve",
            {normalized, table.convolutionWeights, gated}, params, {64, 1, 1},
            {256, 1, 1});
  graph.add("per_layer_embedding_add", {gated, std::move(residual)},
            stateParams, {64, 1, 1}, {256, 1, 1});
}

metal::MetalBuffer auxiliaryView(metal::MetalBackend &backend,
                                 const QwenTargetGeometry &geometry,
                                 const metal::MetalBuffer &cell) {
  return backend.view(cell, geometry.stateLayout.auxiliaryOffset(),
                      geometry.stateLayout.auxiliaryBytes);
}

} // namespace

void Qwen4ExpTarget::addPerLayerEmbeddingCommit(
    const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
    metal::CommandGraph &graph, const QwenTargetCommitBuffers &buffers,
    uint32_t lanes) {
  constexpr uint32_t laneRows = ExecutionLimits::targetVerifyRows;
  const uint64_t laneNormalized =
      (uint64_t{geometry.pleHistoryRows} + laneRows) *
      geometry.residualWidth() * 2;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    PerLayerEmbeddingStateParams params = pleStateParams(geometry, laneRows);
    params.retained_from_buffer = 1;
    params.lane = lane;
    graph.add("per_layer_embedding_commit",
              {backend.view(buffers.pleTokens, uint64_t{lane} * laneRows * 4,
                            uint64_t{laneRows} * 4),
               backend.view(buffers.pleNormalized, lane * laneNormalized,
                            laneNormalized),
               auxiliaryView(backend, geometry, buffers.currentStates[lane]),
               auxiliaryView(backend, geometry, buffers.nextStates[lane]),
               buffers.retainedCounts},
              params, {64, 1, 1}, {256, 1, 1});
  }
}

void Qwen4ExpTarget::addEmbedding(
    const Qwen4ExpWeights &weights,
    const QwenTargetGeometry &geometry,
    metal::CommandGraph &graph,
    metal::MetalBuffer tokens,
    metal::MetalBuffer hidden,
    metal::MetalBuffer scratch,
    uint32_t rows) {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget) {
    throw std::invalid_argument("invalid embedding row count");
  }
  // 1. Gather tokens into [rows, hiddenSize] scratch buffer
  ops::Embedding::add(graph, std::move(tokens), weights.tokenEmbedding, scratch,
                      rows);

  // 2. Broadcast across hyperConnectionCount streams
  const HyperConnectionParams params{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f, 1, 1, 0};
  const uint32_t total = rows * geometry.residualWidth();
  const uint32_t groups = (total + 255) / 256;
  graph.add("hyper_connection_broadcast",
            {std::move(scratch), std::move(hidden)}, params,
            {std::min(groups, 64U), 1, 1}, {256, 1, 1});
}

CapturedPrefillLayers
Qwen4ExpTarget::capturedPrefillLayers(const QwenTargetGeometry &geometry) {
  const PrefillCapture &capture = prefillCapture();
  CapturedPrefillLayers result{capture.layers, {}, capture.rows,
                               geometry.residualWidth()};
  for (const metal::MetalBuffer &buffer : capture.buffers)
    result.data.push_back(static_cast<const uint16_t *>(buffer.contents()));
  if (result.data.size() != result.layers.size())
    result.rows = 0;
  return result;
}

void Qwen4ExpTarget::addHead(
    const Qwen4ExpWeights &weights,
    const QwenTargetGeometry &geometry,
    const ops::ExecutionPlans &operators,
    metal::CommandGraph &graph,
    metal::MetalBuffer hidden,
    metal::MetalBuffer finalHidden,
    metal::MetalBuffer logits,
    metal::MetalBuffer headNormalized,
    metal::MetalBuffer headReduced,
    uint32_t normalizedRows) {
  if (!normalizedRows ||
      normalizedRows > ExecutionLimits::targetVerifyRows) {
    throw std::invalid_argument("invalid Qwen head row count");
  }
  // Collapse the streams: the final mixer has no injection gates.
  addHyperConnection(graph, geometry, std::move(hidden),
                     weights.hyperConnectionMixer, headNormalized, headReduced,
                     finalHidden, {}, normalizedRows);

  // 3. Project finalHidden -> logits
  const ops::LinearMatrix head{geometry.vocabularySize, geometry.hiddenSize};
  operators.linear().addDecode(graph, std::move(finalHidden),
                               weights.logitsProjection, std::move(logits),
                               head);
}

void Qwen4ExpTarget::addPrefill(
    const Qwen4ExpWeights &weights,
    const QwenTargetGeometry &geometry,
    metal::MetalBackend &backend,
    const ops::ExecutionPlans &operators,
    metal::CommandGraph &graph,
    QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences,
    uint32_t rows,
    std::span<const kv::Q8LayerStorage> kvLayers) {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }

  const ops::LinearMatrix gdnInput{geometry.packedGdnWidth,
                                   geometry.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry.packedAttentionWidth,
                                         geometry.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry.hiddenSize,
                                      geometry.attentionWidth};
  const ops::MoePlan moePlan = operators.moePrefill(geometry.moe, rows);

  auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend.view(buffer, uint64_t{begin} * width * sizeof(uint16_t),
                         uint64_t{count} * width * sizeof(uint16_t));
  };
  auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend.view(buffer, uint64_t{begin} * width * sizeof(float),
                         uint64_t{count} * width * sizeof(float));
  };

  const HyperConnectionParams hcParams{
      rows, geometry.hiddenSize, geometry.hyperConnectionCount,
      geometry.hyperConnectionLowRank, 1e-6f, 1, 1, 0};

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;

  auto encodePerLayerEmbedding = [&](metal::CommandGraph &g,
                                     uint32_t layerIndex, bool onCpu) {
    const uint64_t width = uint64_t{geometry.residualWidth()} * 2;
    const uint32_t history = geometry.pleHistoryRows;
    for (const QwenTargetPrefillSequence &sequence : sequences)
      addPleGather(weights, backend, g, buffers.ple, geometry,
                   sequence.auxiliaryIn, sequence.rowBegin, sequence.rows,
                   onCpu, sequence.rows);
    const ops::LinearMatrix keys{geometry.residualWidth(),
                                 geometry.pleEmbeddingSize};
    const ops::LinearMatrix values{geometry.hiddenSize,
                                   geometry.pleEmbeddingSize};
    // The sums depend only on the input rows, so one pass serves both.
    operators.linear().addPrefillSums(g, buffers.ple.embedding,
                                      buffers.projectionSums, keys, rows);
    operators.linear().addPrefill(g, buffers.ple.embedding,
                                  weights.perLayerEmbedding.keyProjection,
                                  buffers.ple.keys, buffers.projectionSums,
                                  keys, rows);
    operators.linear().addPrefill(g, buffers.ple.embedding,
                                  weights.perLayerEmbedding.valueProjection,
                                  buffers.ple.values, buffers.projectionSums,
                                  values, rows);
    for (uint32_t index = 0; index < sequences.size(); ++index) {
      const QwenTargetPrefillSequence &sequence = sequences[index];
      // Each sequence's history sits ahead of its own rows.
      metal::MetalBuffer normalized = backend.view(
          buffers.ple.normalized,
          (uint64_t{sequence.rowBegin} + uint64_t{index} * history) * width,
          (uint64_t{history} + sequence.rows) * width);
      addPleApply(weights, backend, g, buffers.ple, geometry,
                  sequence.auxiliaryIn, normalized,
                  u16(buffers.hidden[layerIndex & 1], sequence.rowBegin,
                      sequence.rows, geometry.residualWidth()),
                  sequence.rowBegin, sequence.rows);
      // A prefill keeps every row it processes.
      const PerLayerEmbeddingStateParams commit =
          pleStateParams(geometry, sequence.rows);
      metal::MetalBuffer tokens =
          backend.view(buffers.ple.tokens, uint64_t{sequence.rowBegin} * 4,
                       uint64_t{sequence.rows} * 4);
      g.add("per_layer_embedding_commit",
            {tokens, normalized, sequence.auxiliaryIn, sequence.auxiliaryOut,
             tokens},
            commit, {64, 1, 1}, {256, 1, 1});
    }
  };

  // priorWorkComplete: every earlier submission of this step has finished,
  // so host-side reads of GPU-written inputs are safe.
  auto encodeAttentionHC = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               bool priorWorkComplete = false) {
    if (geometry.hasPerLayerEmbedding() && layerIndex == geometry.pleLayer)
      encodePerLayerEmbedding(g, layerIndex, priorWorkComplete);
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    addHyperConnection(g, geometry, input, layer.attentionHyperConnection,
                       buffers.normalized, buffers.hyperReduced,
                       buffers.hyperMixed, buffers.hyperInjection, rows, 1,
                       &operators);
  };

  auto encodeMixer = [&](metal::CommandGraph &g, uint32_t layerIndex,
                         uint32_t &gdnIdx, uint32_t &attnIdx) -> metal::MetalBuffer {
    const auto &layer = weights.layers[layerIndex];
    if (std::holds_alternative<QwenGdnWeights>(layer.mixer)) {
      const auto &mixer = std::get<QwenGdnWeights>(layer.mixer);
      operators.linear().addPrefillSums(g, buffers.hyperMixed,
                                        buffers.projectionSums, gdnInput, rows);
      operators.linear().addPrefill(g, buffers.hyperMixed,
                                    mixer.inputProjectionQ8, buffers.gdnPacked,
                                    buffers.projectionSums, gdnInput, rows);
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        ops::GDN::addPrefill(
            g,
            {u16(buffers.gdnPacked, sequence.rowBegin, sequence.rows,
                 geometry.packedGdnWidth),
             mixer.convolutionWeights, sequence.convolutionIn[gdnIdx],
             sequence.convolutionOut[gdnIdx],
             u16(buffers.gdnQueries, sequence.rowBegin, sequence.rows,
                 geometry.gdnKeyWidth()),
             u16(buffers.gdnKeys, sequence.rowBegin, sequence.rows,
                 geometry.gdnKeyWidth()),
             u16(buffers.gdnValues, sequence.rowBegin, sequence.rows,
                 geometry.attentionWidth),
             mixer.decay, mixer.timeBias,
             f32(buffers.gdnDecay, sequence.rowBegin, sequence.rows,
                 geometry.gdnValueHeads),
             u16(buffers.gdnBeta, sequence.rowBegin, sequence.rows,
                 geometry.gdnValueHeads),
             sequence.recurrentIn[gdnIdx],
             sequence.recurrentOut[gdnIdx],
             u16(buffers.recurrent, sequence.rowBegin, sequence.rows,
                 geometry.attentionWidth),
             mixer.mixerNorm,
             u16(buffers.gdnHidden, sequence.rowBegin, sequence.rows,
                 geometry.attentionWidth)},
            gdnShape(geometry), sequence.rows);
      }
      operators.linear().addPrefillSums(g, buffers.gdnHidden,
                                        buffers.projectionSums, mixerOutput,
                                        rows);
      operators.linear().addPrefill(g, buffers.gdnHidden,
                                    mixer.outputProjectionQ8, buffers.gdnOutput,
                                    buffers.projectionSums, mixerOutput, rows);
      ++gdnIdx;
      return buffers.gdnOutput;
    } else {
      const auto &mixer = std::get<QwenAttentionWeights>(layer.mixer);
      operators.linear().addPrefillSums(g, buffers.hyperMixed,
                                        buffers.projectionSums, attentionInput,
                                        rows);
      operators.linear().addPrefill(g, buffers.hyperMixed,
                                    mixer.inputProjectionQ8, buffers.fullPacked,
                                    buffers.projectionSums, attentionInput,
                                    rows);
      for (const QwenTargetPrefillSequence &sequence : sequences) {
        const uint64_t queryBytes =
            uint64_t{geometry.attentionQueryHeads} *
            sequence.attentionStride * geometry.attentionHeadDimension *
            sizeof(uint16_t);
        const uint64_t kvBytes =
            uint64_t{geometry.attentionKvHeads} *
            sequence.attentionStride * geometry.attentionHeadDimension *
            sizeof(uint16_t);
        metal::MetalBuffer queries = backend.view(
            buffers.fullQueries, sequence.queryOffset, queryBytes);
        metal::MetalBuffer attentionRows = backend.view(
            buffers.fullAttention, sequence.queryOffset, queryBytes);
        metal::MetalBuffer keys = backend.view(
            buffers.chunkKeys, sequence.kvOffset, kvBytes);
        metal::MetalBuffer values = backend.view(
            buffers.chunkValues, sequence.kvOffset, kvBytes);
        ops::PagedAttention::addPrefillProjection(
            g,
            u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                geometry.packedAttentionWidth),
            mixer.queryNorm, mixer.keyNorm,
            f32(buffers.ropeCos, sequence.rowBegin, sequence.rows,
                geometry.rotaryPairs),
            f32(buffers.ropeSin, sequence.rowBegin, sequence.rows,
                geometry.rotaryPairs),
            queries, keys, values, sequence.rows,
            sequence.attentionStride, sequence.attentionStride,
            geometry.attentionQueryHeads, geometry.kvLayout);
        ops::PagedAttention::addPrefillStore(
            g, kvLayers[attnIdx], keys, values,
            sequence.pageTable, sequence.q8, geometry.kvLayout);
        ops::PagedAttention::addPrefill(
            g, kvLayers[attnIdx], queries, attentionRows,
            buffers.attentionPartials, buffers.attentionStatistics,
            sequence.pageTable, sequence.q8,
            operators.prefillAttention(
                sequence.rows, geometry.attentionQueryHeads,
                geometry.kvLayout, sequence.q8.committed_tokens));
        ops::PagedAttention::addPrefillGate(
            g,
            u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                geometry.packedAttentionWidth),
            attentionRows,
            u16(buffers.attentionHidden, sequence.rowBegin, sequence.rows,
                geometry.attentionWidth),
            sequence.rows, sequence.attentionStride, sequence.attentionStride,
            geometry.attentionQueryHeads, geometry.kvLayout);
      }
      operators.linear().addPrefillSums(g, buffers.attentionHidden,
                                        buffers.projectionSums, mixerOutput,
                                        rows);
      operators.linear().addPrefill(g, buffers.attentionHidden,
                                    mixer.outputProjectionQ8,
                                    buffers.attentionOutput,
                                    buffers.projectionSums, mixerOutput, rows);
      ++attnIdx;
      return buffers.attentionOutput;
    }
  };

  auto encodeMixerUpdate = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               metal::MetalBuffer mixerOutBuffer) {
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    g.add("hyper_connection_update",
          {input, mixerOutBuffer, buffers.hyperInjection}, hcParams,
          {64, 1, 1}, {256, 1, 1});
  };

  auto encodeMlpHC = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    addHyperConnection(g, geometry, input, layer.mlpHyperConnection,
                       buffers.normalized, buffers.hyperReduced,
                       buffers.hyperMixed, buffers.hyperInjection, rows, 1,
                       &operators);
  };

  auto encodeMoERoute = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    ops::MoE::addRoute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        layer.ffn, moePlan);
  };

  auto encodeMoEExecute = [&](metal::CommandGraph &g, const ops::MoeWeights &weightsToUse) {
    ops::MoE::addExecute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weightsToUse, moePlan, /*addResidual=*/false);
  };

  auto encodeMlpUpdate = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    g.add("hyper_connection_update_out",
          {input, output, buffers.gdnOutput, buffers.hyperInjection},
          hcParams, {64, 1, 1}, {256, 1, 1});
    addPrefillCapture(g, backend, geometry, layerIndex, output, rows);
  };


  const uint32_t R = std::min(weights.residentLayers, geometry.layers);
  const bool useStreamingCache = (R < geometry.layers && weights.streamingCacheGate);

  if (!useStreamingCache) {
    for (uint32_t layerIndex = 0; layerIndex < geometry.layers; ++layerIndex) {
      encodeAttentionHC(graph, layerIndex);
      metal::MetalBuffer mixerOut = encodeMixer(graph, layerIndex, gdnIndex, attentionIndex);
      encodeMixerUpdate(graph, layerIndex, mixerOut);
      encodeMlpHC(graph, layerIndex);
      ops::MoE::add(
          graph,
          {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
           buffers.selectedExperts, buffers.routingWeights,
           buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
           buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
           buffers.expertOutput},
          weights.layers[layerIndex].ffn, moePlan, /*addResidual=*/false);
      encodeMlpUpdate(graph, layerIndex);
    }
    if (gdnIndex != geometry.stateLayout.layers ||
        attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
      throw std::logic_error("Qwen target layer partition mismatch");
    }
    return;
  }

  // =========================================================================
  // Staged Streaming Execution Path with Active Expert Caching for Prefill
  // =========================================================================

  uint32_t totalMisses = 0;

  // Loads the experts rows [begin, end) route to into the layer's cache and
  // points those rows' routes at their slots. The range's experts must fit.
  auto stageActiveExperts = [&](uint32_t layerIndex, uint32_t begin,
                                uint32_t end) {
    auto &cache = weights.layers[layerIndex].expertCache;
    auto *selPtr = static_cast<uint32_t *>(buffers.selectedExperts.contents());
    const uint32_t routesPerRow = moePlan.shape().routesPerToken();
    const uint32_t expertsPerToken = weights.layout.expertsPerToken;
    const uint32_t totalExperts = weights.layout.experts;
    constexpr uint32_t kMaxExperts = 512;
    if (begin == 0)
      logRoutes('P', layerIndex, selPtr, rows, routesPerRow, expertsPerToken,
                rows, rows);
    int16_t stepExpertSeen[kMaxExperts];
    std::fill_n(stepExpertSeen, kMaxExperts, -1);
    std::vector<uint32_t> uniqueExperts;
    uniqueExperts.reserve(32);

    for (uint32_t r = begin; r < end; ++r) {
      for (uint32_t k = 0; k < expertsPerToken; ++k) {
        uint32_t exp = selPtr[r * routesPerRow + k];
        if (exp < totalExperts && stepExpertSeen[exp] == -1) {
          stepExpertSeen[exp] = 1;
          uniqueExperts.push_back(exp);
        }
      }
    }

    if (layerIndex < weights.lastSelectedExperts.size()) {
      weights.lastSelectedExperts[layerIndex] = uniqueExperts;
    }

    if (uniqueExperts.size() > cache.capacity)
      throw std::logic_error("prefill slice needs more experts than the cache holds");

    struct Miss {
      uint32_t expert;
      uint32_t slot;
    };
    std::vector<Miss> misses;
    misses.reserve(uniqueExperts.size());
    ++cache.clock;

    int16_t expertToAssignedSlot[kMaxExperts];
    std::fill_n(expertToAssignedSlot, kMaxExperts, -1);

    for (uint32_t exp : uniqueExperts) {
      int16_t slot = cache.expertToSlot[exp];
      if (slot != -1) {
        // Cache hit: refresh LRU timestamp
        cache.lruTime[slot] = cache.clock;
        expertToAssignedSlot[exp] = slot;
      } else {
        // Cache miss: find a slot
        uint32_t assignSlot = 0;
        if (cache.numCached < cache.capacity) {
          assignSlot = cache.numCached++;
        } else {
          // Evict LRU slot not in active step
          uint32_t oldest = UINT32_MAX;
          uint32_t best = 0;
          for (uint32_t s = 0; s < cache.capacity; ++s) {
            if (cache.lruTime[s] == cache.clock) continue;
            if (cache.lruTime[s] < oldest) {
              oldest = cache.lruTime[s];
              best = s;
            }
          }
          assignSlot = best;
          int16_t evicted = cache.slotToExpert[assignSlot];
          if (evicted != -1) {
            cache.expertToSlot[evicted] = -1;
          }
        }
        cache.slotToExpert[assignSlot] = exp;
        cache.expertToSlot[exp] = assignSlot;
        cache.lruTime[assignSlot] = cache.clock;
        expertToAssignedSlot[exp] = assignSlot;
        misses.push_back({exp, assignSlot});
      }
    }

    totalMisses += static_cast<uint32_t>(misses.size());

    if (!misses.empty()) {
      const auto &layer = weights.layers[layerIndex];
      const uint64_t stride = layer.ffn.expertGate.expertStrideBytes;
      char *cg = static_cast<char *>(cache.cacheGate.contents());
      char *cu = static_cast<char *>(cache.cacheUp.contents());
      char *cd = static_cast<char *>(cache.cacheDown.contents());

      // The expert regions are mapped for random access, so copying a missed
      // expert faults it in one 16 KB page at a time, each its own disk read.
      // Asking for every missed range first lets the reads go out together
      // and in large pieces; the copies below then find the pages in memory.
      readMissedExperts(layer.expertSource, misses.data(), misses.size(),
                        stride, cg, cu, cd);
    }

    for (uint32_t r = begin; r < end; ++r) {
      for (uint32_t k = 0; k < expertsPerToken; ++k) {
        uint32_t exp = selPtr[r * routesPerRow + k];
        if (exp < totalExperts) {
          selPtr[r * routesPerRow + k] = expertToAssignedSlot[exp];
        }
      }
      selPtr[r * routesPerRow + expertsPerToken] = 512;
    }

  };

  // Row ranges whose experts fit in the layer's cache together. A prompt
  // chunk can route to more distinct experts than a layer caches; reading
  // those from the mapped file instead stalled the GPU for seconds on page
  // faults, so the chunk's expert step runs one range at a time.
  auto planSlices = [&](uint32_t layerIndex) {
    const auto &cache = weights.layers[layerIndex].expertCache;
    const auto *selPtr = static_cast<const uint32_t *>(buffers.selectedExperts.contents());
    const uint32_t routesPerRow = moePlan.shape().routesPerToken();
    const uint32_t expertsPerToken = weights.layout.expertsPerToken;
    std::vector<uint32_t> ends;
    std::bitset<512> seen;
    // Rows go in pairs so every range starts on an even row: the routing
    // weights are 2-byte values, 11 to a row, and a view's offset must stay
    // 4-byte aligned.
    for (uint32_t r = 0; r < rows; r += 2) {
      std::bitset<512> pair;
      for (uint32_t k = 0; k < 2 * expertsPerToken; ++k) {
        const uint32_t pairRow = r + k / expertsPerToken;
        if (pairRow >= rows)
          break;
        const uint32_t exp = selPtr[pairRow * routesPerRow + k % expertsPerToken];
        if (exp < 512)
          pair[exp] = true;
      }
      if ((seen | pair).count() > cache.capacity) {
        ends.push_back(r);
        seen.reset();
      }
      seen |= pair;
    }
    ends.push_back(rows);
    return ends;
  };

  auto encodeMoESlice = [&](metal::CommandGraph &g, const ops::MoeWeights &weightsToUse,
                            uint32_t begin, uint32_t count) {
    const uint32_t routesPerRow = moePlan.shape().routesPerToken();
    auto rowsOf = [&](const metal::MetalBuffer &buffer, uint64_t bytesPerRow) {
      return backend.view(buffer, begin * bytesPerRow, count * bytesPerRow);
    };
    const uint64_t hiddenBytes = uint64_t{geometry.hiddenSize} * 2;
    const metal::MetalBuffer input = rowsOf(buffers.hyperMixed, hiddenBytes);
    ops::MoE::addExecute(
        g,
        {input, input, rowsOf(buffers.gdnOutput, hiddenBytes),
         rowsOf(buffers.selectedExperts, routesPerRow * 4ull),
         rowsOf(buffers.routingWeights, routesPerRow * 2ull),
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weightsToUse, operators.moePrefill(geometry.moe, count),
        /*addResidual=*/false);
  };

  auto makeCacheWeights = [&](uint32_t layerIndex) -> ops::MoeWeights {
    const auto &layer = weights.layers[layerIndex];
    const auto &cache = layer.expertCache;
    const uint64_t stride = layer.ffn.expertGate.expertStrideBytes;
    return ops::MoeWeights{
        .router = layer.ffn.router,
        .expertGate = {cache.cacheGate, cache.capacity,
                       weights.layout.expertIntermediateSize, weights.layout.hiddenSize, stride},
        .expertUp = {cache.cacheUp, cache.capacity,
                     weights.layout.expertIntermediateSize, weights.layout.hiddenSize, stride},
        .expertDown = {cache.cacheDown, cache.capacity,
                       weights.layout.hiddenSize, weights.layout.expertIntermediateSize, stride},
        .sharedGate = layer.ffn.sharedGate,
        .sharedUp = layer.ffn.sharedUp,
        .sharedDown = layer.ffn.sharedDown,
        .sharedExpertGate = layer.ffn.sharedExpertGate,
    };
  };

  metal::CommandGraph residentGraph = std::move(graph);
  for (uint32_t layerIndex = 0; layerIndex < R; ++layerIndex) {
    encodeAttentionHC(residentGraph, layerIndex);
    metal::MetalBuffer mixerOut = encodeMixer(residentGraph, layerIndex, gdnIndex, attentionIndex);
    encodeMixerUpdate(residentGraph, layerIndex, mixerOut);
    encodeMlpHC(residentGraph, layerIndex);
    ops::MoE::add(
        residentGraph,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weights.layers[layerIndex].ffn, moePlan, /*addResidual=*/false);
    encodeMlpUpdate(residentGraph, layerIndex);
  }

  // Layer R base up to router
  encodeAttentionHC(residentGraph, R);
  metal::MetalBuffer mixerOutR = encodeMixer(residentGraph, R, gdnIndex, attentionIndex);
  encodeMixerUpdate(residentGraph, R, mixerOutR);
  encodeMlpHC(residentGraph, R);
  encodeMoERoute(residentGraph, R);

  // Submit residentGraph and prefetch streaming experts in background
  auto t0 = AwakeClock::now();
  metal::CommandTicket residentTicket = backend.submitCommandAsync(residentGraph.dispatches());
  // No blanket read-ahead of the previous step's experts: after a long
  // prompt that is nearly every expert, ~68 GB the OS then reads in the
  // background for minutes, stalling decode. Misses are fetched on demand.
  (void)residentTicket.wait();
  auto t1 = AwakeClock::now();
  double residentMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  double totalStageMs = 0.0;
  double totalGpuMs = 0.0;

  // Loop through streaming layers R .. geometry.layers - 2
  // Every range but the last runs now, each after its experts are loaded
  // (and after the range before it finished with the slots it may evict).
  // The last range goes into `g`, ahead of the rest of the layer.
  uint32_t totalSlices = 0;
  // Expert-major, the default: each expert this chunk routes to is read at
  // most once per layer. Pass 0 runs the experts already cached (and the
  // shared expert) while the first wave of missing experts is read into
  // slots it does not use; each later wave reuses the slots of the wave two
  // before it, once that wave's GPU pass is done. Row slicing re-read experts
  // every slice (72 GB for a 1.5K-token prompt).
  static const bool rowSlices = std::getenv("SPLASH_PREFILL_ROW_SLICES") != nullptr;
  auto encodeLayerExpertsByExpert = [&](metal::CommandGraph &g, uint32_t L) {
    auto &cache = weights.layers[L].expertCache;
    if (cache.pending) {
      dispatch_group_wait(cache.inflight, DISPATCH_TIME_FOREVER);
      cache.pending = false;
    }
    const auto ts0 = AwakeClock::now();
    const uint32_t routesPerRow = moePlan.shape().routesPerToken();
    const uint32_t perToken = weights.layout.expertsPerToken;
    const uint32_t experts = weights.layout.experts;
    const uint32_t tileRows = moePlan.tileRows();
    const auto *selected = static_cast<const uint32_t *>(buffers.selectedExperts.contents());
    std::vector<std::vector<uint32_t>> routesOf(experts);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t j = 0; j < perToken; ++j) {
        const uint32_t e = selected[r * routesPerRow + j];
        if (e < experts) routesOf[e].push_back(r * routesPerRow + j);
      }
    std::vector<uint32_t> hits, misses;
    for (uint32_t e = 0; e < experts; ++e) {
      if (routesOf[e].empty()) continue;
      countExpertUse(cache, e);
      (cache.expertToSlot[e] >= 0 ? hits : misses).push_back(e);
    }
    ++cache.clock;
    // Waves. After this chunk the cache should hold the chunk's most-used
    // experts: they are what decoding the same text needs next. Stage 0 runs
    // the cached experts (and the shared expert). Missing experts among the
    // most-used are read into cache slots - first those holding experts this
    // chunk never uses (free at once), then those of rarely used cached ones
    // (free once stage 0 is done). The rest, used by few routes, are read
    // into the prompt staging buffer, whose two halves alternate so reads
    // overlap the GPU. Every expert is read at most once.
    struct Load { uint32_t expert, slot; };
    struct Wave {
      std::vector<Load> loads;
      bool staging = false;  // runs from the staging buffer
      int after = -1;        // stage that must finish before these reads
      int set = -1;          // staging half
    };
    // Recent use first: decoding continues from the chunk's last rows.
    static const uint32_t recentRows = [] {
      const char *value = std::getenv("SPLASH_PROMPT_RECENT_ROWS");
      return value ? static_cast<uint32_t>(std::atoi(value)) : 256u;
    }();
    const uint32_t recentFrom = rows > recentRows ? rows - recentRows : 0;
    std::vector<uint32_t> recent(experts, 0);
    for (uint32_t e = 0; e < experts; ++e)
      for (uint32_t route : routesOf[e])
        recent[e] += route / routesPerRow >= recentFrom;
    auto usage = [&](uint32_t e) {
      return uint64_t{recent[e]} * 65536 + routesOf[e].size();
    };
    std::vector<uint32_t> used(hits);
    used.insert(used.end(), misses.begin(), misses.end());
    // Ranked by this chunk's (recent) use; SPLASH_PROMPT_KEEP=frequency ranks
    // by the long-run use counts instead (measured worse: 54.6 decode misses
    // a step after the code prompt, against 51.8).
    static const bool byUsage = [] {
      const char *value = std::getenv("SPLASH_PROMPT_KEEP");
      return !(value && std::string_view(value) == "frequency");
    }();
    std::sort(used.begin(), used.end(), [&](uint32_t x, uint32_t y) {
      if (byUsage && usage(x) != usage(y)) return usage(x) > usage(y);
      if (cache.frequency[x] != cache.frequency[y]) return cache.frequency[x] > cache.frequency[y];
      if (usage(x) != usage(y)) return usage(x) > usage(y);
      return x < y;
    });
    std::vector<bool> keep(experts, false);
    for (size_t i = 0; i < used.size() && i < cache.capacity; ++i) keep[used[i]] = true;
    std::vector<bool> usedNow(experts, false);
    for (uint32_t e : used) usedNow[e] = true;
    std::vector<uint32_t> freeNow, freeAfterFirst;
    for (uint32_t slot = 0; slot < cache.capacity; ++slot) {
      const int16_t holder = cache.slotToExpert[slot];
      if (holder < 0 || !usedNow[holder]) freeNow.push_back(slot);
      else if (!keep[holder]) freeAfterFirst.push_back(slot);
    }
    std::vector<uint32_t> keepMisses, restMisses;
    for (uint32_t e : used)
      if (cache.expertToSlot[e] < 0) (keep[e] ? keepMisses : restMisses).push_back(e);
    struct AdviseMiss { uint32_t expert; };
    std::vector<AdviseMiss> allMisses;
    allMisses.reserve(keepMisses.size() + restMisses.size());
    for (uint32_t e : keepMisses) allMisses.push_back({e});
    for (uint32_t e : restMisses) allMisses.push_back({e});
    adviseMissedExperts(weights.layers[L].expertSource, allMisses.data(), allMisses.size(),
                        weights.layers[L].ffn.expertGate.expertStrideBytes);
    std::vector<Wave> waves(1);
    for (uint32_t e : hits)
      waves[0].loads.push_back({e, static_cast<uint32_t>(cache.expertToSlot[e])});
    size_t next = 0;
    for (int pass = 0; pass < 2; ++pass) {
      const auto &slots = pass ? freeAfterFirst : freeNow;
      Wave wave;
      wave.after = pass ? 0 : -1;
      for (size_t i = 0; i < slots.size() && next < keepMisses.size(); ++i)
        wave.loads.push_back({keepMisses[next++], slots[i]});
      if (!wave.loads.empty()) waves.push_back(std::move(wave));
    }
    auto &staging = weights.promptStaging;
    const uint32_t halfSlots = staging.capacity / 2;
    std::array<int, 2> lastUser{-1, -1};
    for (size_t i = 0, set = 0; i < restMisses.size(); set ^= 1) {
      Wave wave;
      wave.staging = true;
      wave.set = static_cast<int>(set);
      wave.after = lastUser[set];
      for (uint32_t k = 0; k < halfSlots && i < restMisses.size(); ++k)
        wave.loads.push_back({restMisses[i++], static_cast<uint32_t>(set * halfSlots + k)});
      lastUser[set] = static_cast<int>(waves.size());
      waves.push_back(std::move(wave));
    }
    for (const Wave &wave : waves)
      if (!wave.staging)
        for (const Load &load : wave.loads) {
          const int16_t old = cache.slotToExpert[load.slot];
          if (old >= 0 && old != static_cast<int16_t>(load.expert))
            cache.expertToSlot[old] = -1;
          cache.slotToExpert[load.slot] = static_cast<int16_t>(load.expert);
          cache.expertToSlot[load.expert] = static_cast<int16_t>(load.slot);
          cache.lruTime[load.slot] = cache.clock;
          cache.numCached = std::max(cache.numCached, load.slot + 1);
        }
    // Tiles in wave order; each wave's tiles are one contiguous range.
    auto *tiles = static_cast<MoeTileDescriptor *>(buffers.tileDescriptors.contents());
    auto *grouped = static_cast<uint32_t *>(buffers.groupedRoutes.contents());
    auto *routeRows = static_cast<uint32_t *>(buffers.routeRows.contents());
    std::fill_n(routeRows, uint64_t{rows} * routesPerRow, 0xFFFFFFFFu);
    uint32_t tile = 0;
    auto addTiles = [&](uint32_t descriptorExpert, const std::vector<uint32_t> &routes) {
      for (uint32_t first = 0; first < routes.size(); first += tileRows) {
        const uint32_t count = std::min<uint32_t>(tileRows, routes.size() - first);
        tiles[tile] = MoeTileDescriptor{descriptorExpert, count};
        for (uint32_t i = 0; i < tileRows; ++i) {
          const uint32_t row = tile * tileRows + i;
          grouped[row] = i < count ? routes[first + i] : 0xFFFFFFFFu;
          if (i < count) routeRows[routes[first + i]] = row;
        }
        ++tile;
      }
    };
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    for (size_t w = 0; w < waves.size(); ++w) {
      const uint32_t begin = tile;
      for (const Load &load : waves[w].loads) addTiles(load.slot, routesOf[load.expert]);
      if (w == 0) {
        std::vector<uint32_t> shared(rows);
        for (uint32_t r = 0; r < rows; ++r) shared[r] = r * routesPerRow + perToken;
        addTiles(experts, shared);
      }
      ranges.push_back({begin, tile - begin});
    }
    if (tile > moePlan.maximumTiles())
      throw std::logic_error("prompt experts need more tiles than the plan holds");
    *static_cast<uint32_t *>(buffers.tileCount.contents()) = tile;
    if (!weights.prefillRangeCounts)
      weights.prefillRangeCounts = backend.allocateBuffer(
          64 * 256, metal::BufferStorage::Shared, "qwen4exp-prefill-ranges");
    if (waves.size() > 64)
      throw std::logic_error("prompt experts need more than 64 waves");
    auto rangeCount = [&](size_t w) {
      *reinterpret_cast<uint32_t *>(static_cast<char *>(weights.prefillRangeCounts.contents()) + w * 256) =
          ranges[w].second;
      return backend.view(weights.prefillRangeCounts, w * 256, 4);
    };
    const ops::MoeBuffers moe{
        buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
        buffers.selectedExperts, buffers.routingWeights, buffers.tileDescriptors,
        buffers.tileCount, buffers.groupedRoutes, buffers.routeRows,
        buffers.groupedInput, buffers.expertIntermediate, buffers.expertOutput};
    const ops::MoeWeights cacheWeights = makeCacheWeights(L);
    ops::MoeWeights stagingWeights = cacheWeights;
    const uint64_t stride = weights.layers[L].ffn.expertGate.expertStrideBytes;
    stagingWeights.expertGate.packed = staging.cacheGate;
    stagingWeights.expertUp.packed = staging.cacheUp;
    stagingWeights.expertDown.packed = staging.cacheDown;
    stagingWeights.expertGate.experts = stagingWeights.expertUp.experts =
        stagingWeights.expertDown.experts = staging.capacity;
    if (!weights.prefillGateScratch) {
      // Sized for the largest chunk, not this one.
      const ops::MoePlan largest =
          operators.moePrefill(geometry.moe, ExecutionLimits::prefillTokenBudget);
      weights.prefillGateScratch = backend.allocateBuffer(
          uint64_t{largest.maximumTiles()} * largest.tileRows() *
              largest.shape().expertIntermediateSize * 2,
          metal::BufferStorage::Private, "qwen4exp-prefill-gate");
    }
    totalStageMs += std::chrono::duration<double, std::milli>(
                        AwakeClock::now() - ts0).count();
    const auto &source = weights.layers[L].expertSource;
    auto loadWave = [&](size_t w) {
      const auto readStart = AwakeClock::now();
      const Wave &wave = waves[w];
      struct Miss { uint32_t expert, slot; };
      std::vector<Miss> reads;
      for (const Load &load : wave.loads) reads.push_back({load.expert, load.slot});
      totalMisses += static_cast<uint32_t>(reads.size());
      const auto &into = wave.staging ? staging : cache;
      readMissedExperts(source, reads.data(), reads.size(), stride,
                        static_cast<char *>(into.cacheGate.contents()),
                        static_cast<char *>(into.cacheUp.contents()),
                        static_cast<char *>(into.cacheDown.contents()));
      totalStageMs += std::chrono::duration<double, std::milli>(
                          AwakeClock::now() - readStart).count();
    };
    for (const Load &load : waves[0].loads) cache.lruTime[load.slot] = cache.clock;
    auto addWave = [&](metal::CommandGraph &stage, size_t w) {
      if (w == 0) ops::MoE::addGather(stage, moe, moePlan);
      ops::MoE::addExpertRange(stage, moe, waves[w].staging ? stagingWeights : cacheWeights,
                               moePlan, backend, ranges[w].first, ranges[w].second,
                               rangeCount(w), weights.prefillGateScratch);
    };
    if (waves.size() == 1) {
      addWave(g, 0);
      ops::MoE::addCombine(g, moe, moePlan, /*addResidual=*/false);
      totalSlices += 1;
      return;
    }
    // The per-dispatch profiler replays commands one dispatch at a time and
    // cannot honour pipeline events: run the waves one after another.
    if (backend.dispatchProfiling()) {
      for (size_t w = 0; w < waves.size(); ++w) {
        if (w > 0) loadWave(w);
        metal::CommandGraph stage;
        addWave(stage, w);
        (void)backend.submitCommandAsync(stage.dispatches()).wait();
      }
      ops::MoE::addCombine(g, moe, moePlan, /*addResidual=*/false);
      totalSlices += static_cast<uint32_t>(waves.size());
      return;
    }
    // One pipelined submission, a stage per wave. Stage w starts once the
    // host has read wave w, which first waits for the last stage that read
    // the same staging half.
    std::deque<metal::CommandGraph> stages;
    std::vector<metal::ComputeDispatch> all;
    std::vector<size_t> starts;
    for (size_t w = 0; w < waves.size(); ++w) {
      metal::CommandGraph &stage = stages.emplace_back();
      addWave(stage, w);
      starts.push_back(all.size());
      for (const auto &d : stage.dispatches()) all.push_back(d);
    }
    const uint64_t base = backend.reservePipelineEvents(static_cast<uint32_t>(waves.size()));
    const auto gpuStart = AwakeClock::now();
    metal::CommandTicket ticket = backend.submitPipelineAsync(all, starts, base);
    for (size_t w = 1; w < waves.size(); ++w) {
      if (waves[w].after >= 0 &&
          !backend.waitPipelineStageDone(base, static_cast<uint32_t>(waves[w].after), 60000))
        throw std::runtime_error("prompt expert wave timed out");
      loadWave(w);
      backend.signalPipelineEvent(base + 2 * w);
    }
    (void)ticket.wait();
    totalGpuMs += std::chrono::duration<double, std::milli>(
                      AwakeClock::now() - gpuStart).count();
    ops::MoE::addCombine(g, moe, moePlan, /*addResidual=*/false);
    totalSlices += static_cast<uint32_t>(waves.size());
  };
  auto encodeLayerExperts = [&](metal::CommandGraph &g, uint32_t L) {
    if (!rowSlices)
      return encodeLayerExpertsByExpert(g, L);
    const std::vector<uint32_t> ends = planSlices(L);
    totalSlices += static_cast<uint32_t>(ends.size());
    uint32_t begin = 0;
    for (size_t i = 0; i + 1 < ends.size(); ++i) {
      auto ts0 = AwakeClock::now();
      stageActiveExperts(L, begin, ends[i]);
      auto ts1 = AwakeClock::now();
      totalStageMs += std::chrono::duration<double, std::milli>(ts1 - ts0).count();
      metal::CommandGraph slice;
      encodeMoESlice(slice, makeCacheWeights(L), begin, ends[i] - begin);
      (void)backend.submitCommandAsync(slice.dispatches()).wait();
      totalGpuMs += std::chrono::duration<double, std::milli>(
                        AwakeClock::now() - ts1).count();
      begin = ends[i];
    }
    auto ts0 = AwakeClock::now();
    stageActiveExperts(L, begin, rows);
    totalStageMs += std::chrono::duration<double, std::milli>(
                        AwakeClock::now() - ts0).count();
    if (begin == 0)
      encodeMoEExecute(g, makeCacheWeights(L));
    else
      encodeMoESlice(g, makeCacheWeights(L), begin, rows - begin);
  };

  for (uint32_t L = R; L < geometry.layers - 1; ++L) {
    metal::CommandGraph stepGraph;
    encodeLayerExperts(stepGraph, L);
    encodeMlpUpdate(stepGraph, L);

    // Layer L + 1 base up to router
    encodeAttentionHC(stepGraph, L + 1, /*priorWorkComplete=*/true);
    metal::MetalBuffer mixerOutNext = encodeMixer(stepGraph, L + 1, gdnIndex, attentionIndex);
    encodeMixerUpdate(stepGraph, L + 1, mixerOutNext);
    encodeMlpHC(stepGraph, L + 1);
    encodeMoERoute(stepGraph, L + 1);

    auto tg0 = AwakeClock::now();
    (void)backend.submitCommandAsync(stepGraph.dispatches()).wait();
    auto tg1 = AwakeClock::now();
    totalGpuMs += std::chrono::duration<double, std::milli>(tg1 - tg0).count();

    // Layer L + 1 router has completed on the GPU; advise kernel of its missed experts early
    if (L + 1 < geometry.layers) {
      const auto &nextSource = weights.layers[L + 1].expertSource;
      const auto &nextCache = weights.layers[L + 1].expertCache;
      const uint64_t nextStride = weights.layers[L + 1].ffn.expertGate.expertStrideBytes;
      const auto *selPtr = static_cast<const uint32_t *>(buffers.selectedExperts.contents());
      const uint32_t routesPerRow = moePlan.shape().routesPerToken();
      const uint32_t perToken = weights.layout.expertsPerToken;
      std::bitset<512> nextSeen;
      struct AdviseMiss { uint32_t expert; };
      std::vector<AdviseMiss> nextMisses;
      for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < perToken; ++k) {
          uint32_t exp = selPtr[r * routesPerRow + k];
          if (exp < weights.layout.experts && !nextSeen.test(exp)) {
            nextSeen.set(exp);
            if (nextCache.expertToSlot[exp] < 0) {
              nextMisses.push_back({exp});
            }
          }
        }
      }
      adviseMissedExperts(nextSource, nextMisses.data(), nextMisses.size(), nextStride);
    }
  }

  const uint32_t lastL = geometry.layers - 1;
  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  // Encode final layer's MoE and MLP update into the caller's graph
  encodeLayerExperts(graph, lastL);
  encodeMlpUpdate(graph, lastL);

  // MTP head: fill its attention cache for this chunk's positions. Row r
  // pairs the trunk's final residual at r with the prompt token at r + 1, so
  // every row but the chunk's last has its input (the last prompt row is
  // covered by the first decode step). Only the keys and values are needed:
  // combiner, attention mix, projection, store - no attention, experts or head.
  if (weights.mtpLayer && weights.mtpCombiner && rows > 1 &&
      !std::getenv("SPLASH_NO_MTP")) {
    const auto &head = *weights.mtpLayer;
    const auto &combiner = *weights.mtpCombiner;
    const uint32_t kvIndex = geometry.kvLayout.attentionLayers - 1;
    const uint32_t hidden = geometry.hiddenSize;
    const uint32_t count = geometry.hyperConnectionCount;
    const uint32_t m = rows - 1;
    const metal::MetalBuffer residual = buffers.hidden[geometry.layers & 1];
    const metal::MetalBuffer X = buffers.hidden[(geometry.layers & 1) ^ 1];
    if (!weights.mtpPrefillOnes) {
      weights.mtpPrefillOnes = backend.allocateBuffer(
          uint64_t{ExecutionLimits::prefillTokenBudget} * count * 2,
          metal::BufferStorage::Shared, "mtp-prefill-ones");
      std::fill_n(static_cast<uint16_t *>(weights.mtpPrefillOnes.contents()),
                  ExecutionLimits::prefillTokenBudget * count, uint16_t{0x3F80});
    }
    const ops::LinearMatrix square{hidden, hidden};
    // Prefill linear kernels read and write whole 32-row tiles, so views are
    // padded to one; the padding rows are scratch.
    auto padded = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                      uint32_t count, uint32_t width) {
      return u16(buffer, begin, (count + 31) / 32 * 32, width);
    };
    const HyperConnectionParams single{m, hidden, 1, geometry.hyperConnectionLowRank,
                                       1e-6f, 0, 1, 0};
    const HyperConnectionParams all{m, hidden, count, geometry.hyperConnectionLowRank,
                                    1e-6f, 1, 1, 0};
    // Tokens shifted by one row.
    metal::MetalBuffer next = backend.view(buffers.ple.tokens, 4, uint64_t{m} * 4);
    metal::MetalBuffer embedded = padded(buffers.recurrent, 0, m, hidden);
    metal::MetalBuffer normalizedE = padded(buffers.gdnHidden, 0, m, hidden);
    metal::MetalBuffer e = padded(buffers.attentionHidden, 0, m, hidden);
    ops::Embedding::add(graph, next, weights.tokenEmbedding, embedded, m);
    graph.add("hyper_connection_rms", {embedded, combiner.embeddingNorm, normalizedE},
              single, {m, 1, 1}, {256, 1, 1});
    operators.linear().addPrefillSums(graph, normalizedE, buffers.projectionSums, square, m);
    operators.linear().addPrefill(graph, normalizedE, combiner.fcEmbedding, e,
                                  buffers.projectionSums, square, m);
    graph.add("hyper_connection_rms", {residual, combiner.hiddenNorm, buffers.normalized},
              all, {m, count, 1}, {256, 1, 1});
    // fc_hidden on each stream: m rows of 4 streams, in prefill-sized pieces.
    const uint32_t streamRows = m * count;
    for (uint32_t first = 0; first < streamRows;
         first += ExecutionLimits::prefillTokenBudget) {
      const uint32_t piece = std::min(ExecutionLimits::prefillTokenBudget, streamRows - first);
      metal::MetalBuffer in = padded(buffers.normalized, first, piece, hidden);
      metal::MetalBuffer out = padded(X, first, piece, hidden);
      operators.linear().addPrefillSums(graph, in, buffers.projectionSums, square, piece);
      operators.linear().addPrefill(graph, in, combiner.fcHidden, out,
                                    buffers.projectionSums, square, piece);
    }
    graph.add("hyper_connection_update", {X, e, weights.mtpPrefillOnes}, all,
              {64, 1, 1}, {256, 1, 1});
    addHyperConnection(graph, geometry, X, head.attentionHyperConnection,
                       buffers.normalized, buffers.hyperReduced, buffers.hyperMixed,
                       buffers.hyperInjection, m, 1, &operators);
    const auto &mixer = std::get<QwenAttentionWeights>(head.mixer);
    const ops::LinearMatrix attentionInput{geometry.packedAttentionWidth, hidden};
    operators.linear().addPrefillSums(graph, buffers.hyperMixed, buffers.projectionSums,
                                      attentionInput, m);
    operators.linear().addPrefill(graph, buffers.hyperMixed, mixer.inputProjectionQ8,
                                  buffers.fullPacked, buffers.projectionSums,
                                  attentionInput, m);
    for (const QwenTargetPrefillSequence &sequence : sequences) {
      if (sequence.rows < 2)
        continue;
      const uint32_t kept = sequence.rows - 1;
      const uint64_t queryBytes = uint64_t{geometry.attentionQueryHeads} *
                                  sequence.attentionStride *
                                  geometry.attentionHeadDimension * 2;
      const uint64_t kvBytes = uint64_t{geometry.attentionKvHeads} *
                               sequence.attentionStride *
                               geometry.attentionHeadDimension * 2;
      metal::MetalBuffer queries = backend.view(buffers.fullQueries, sequence.queryOffset, queryBytes);
      metal::MetalBuffer keys = backend.view(buffers.chunkKeys, sequence.kvOffset, kvBytes);
      metal::MetalBuffer values = backend.view(buffers.chunkValues, sequence.kvOffset, kvBytes);
      ops::PagedAttention::addPrefillProjection(
          graph, u16(buffers.fullPacked, sequence.rowBegin, kept, geometry.packedAttentionWidth),
          mixer.queryNorm, mixer.keyNorm,
          f32(buffers.ropeCos, sequence.rowBegin, kept, geometry.rotaryPairs),
          f32(buffers.ropeSin, sequence.rowBegin, kept, geometry.rotaryPairs),
          queries, keys, values, kept, sequence.attentionStride,
          sequence.attentionStride, geometry.attentionQueryHeads, geometry.kvLayout);
      kv::Q8ChunkedPrefillParams store = sequence.q8;
      store.chunk_tokens = kept;
      ops::PagedAttention::addPrefillStore(graph, kvLayers[kvIndex], keys, values,
                                           sequence.pageTable, store, geometry.kvLayout);
    }
  }

  // Developer timing, off in normal logs: SPLASH_STEP_TIMING=1 turns it on.
  static const bool prefillLog = std::getenv("SPLASH_STEP_TIMING") != nullptr;
  if (prefillLog)
    std::cerr << "[Prefill Timing] Resident 0.." << R << ": " << residentMs << " ms | Streaming Staging: "
              << totalStageMs << " ms (misses: " << totalMisses << ", expert passes: "
              << totalSlices << ") | Streaming GPU: " << totalGpuMs << " ms\n";

  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }
}

uint32_t mtpDraftLimit() noexcept {
  static const uint32_t limit = [] {
    const char *value = std::getenv("SPLASH_MTP_DRAFTS");
    const char *tree = std::getenv("SPLASH_TREE_DRAFT");
    const bool treeDrafting = tree != nullptr && std::atoi(tree) != 0;
    const int defaultDrafts = treeDrafting ? 7 : 5;
    const int parsed = value ? std::atoi(value) : defaultDrafts;
    return static_cast<uint32_t>(std::clamp(parsed, 1, 7));
  }();
  return limit;
}

void Qwen4ExpTarget::addVerify(
    const Qwen4ExpWeights &weights,
    const QwenTargetGeometry &geometry,
    metal::MetalBackend &backend,
    const ops::ExecutionPlans &operators,
    metal::CommandGraph &graph,
    QwenTargetVerifyBuffers buffers,
    std::span<const kv::Q8LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify,
    uint32_t lanes,
    ops::Q4DispatchStats &stats) {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      q8.size() != ExecutionLimits::maximumBatchWidth ||
      verify.size() != ExecutionLimits::maximumBatchWidth ||
      kvLayers.size() != geometry.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = verify[lane].committed_tokens;
  const auto attentionPlan = operators.verifyAttention(
      lanes, geometry.attentionQueryHeads, geometry.kvLayout, histories);
  const ops::LinearMatrix gdnInput{geometry.packedGdnWidth,
                                   geometry.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry.packedAttentionWidth,
                                         geometry.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry.hiddenSize,
                                      geometry.attentionWidth};
  const ops::MoePlan moePlan = operators.moeDecode(geometry.moe, lanes);

  // With one live row per lane, hyper-connections run on those rows only:
  // lane rows 0, 8, 16, ... The other rows' results are never kept.
  // One lane: its live rows are rows 0..live-1. Several lanes with one live
  // row each: rows 0, 8, 16, ...
  // Recomputed after the MTP head drafts, since it may guess fewer rows.
  // MTP draft time per verify step: GPU part A (combiner, attention, route
  // scores), host expert staging, GPU part B (experts, mixer, head), CPU pick.
  std::array<double, 4> mtpParts{};
  // Host expert staging per verify step: waiting for predicted reads,
  // blocking reads of unpredicted misses, and the GPU's wait on the host
  // (event raised to event signalled, summed over stages).
  std::array<double, 3> hostParts{};
  // Lookahead quality: for each layer, the experts predicted (issued top-k,
  // and a wider top-16 kept only for this count).
  std::vector<std::bitset<512>> predictedNarrow(geometry.layers + 1), predictedWide(geometry.layers + 1);
  static std::array<uint64_t, 5> predictionCounts{};  // used, in narrow, in wide, misses, misses in wide
  const auto verifyEnter = AwakeClock::now();
  double encodeMs = 0.0;  // CPU time building the pipelined stage graphs
  uint32_t hcRows = 0, hcStep = 1;
  HyperConnectionParams hcParams{};
  auto setLiveRows = [&] {
    const uint32_t live = buffers.liveRowsPerLane;
    hcRows = lanes == 1 ? live : (live == 1 ? lanes : rows);
    hcStep = lanes != 1 && live == 1 ? ExecutionLimits::targetVerifyRows : 1;
    hcParams = {hcRows, geometry.hiddenSize, geometry.hyperConnectionCount,
                geometry.hyperConnectionLowRank, 1e-6f, 1, hcStep, 0};
  };
  setLiveRows();

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;

  // Set while a pipelined step is being encoded; see the staged loop.
  std::vector<std::function<void()>> *pipelineDeferred = nullptr;
  auto encodePerLayerEmbedding = [&](metal::CommandGraph &g,
                                     uint32_t layerIndex, bool onCpu) {
    constexpr uint32_t laneRows = ExecutionLimits::targetVerifyRows;
    const uint64_t width = uint64_t{geometry.residualWidth()} * 2;
    const uint64_t laneNormalized =
        (uint64_t{geometry.pleHistoryRows} + laneRows) * width;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      addPleGather(weights, backend, g, buffers.ple, geometry,
                   auxiliaryView(backend, geometry,
                                 buffers.currentGdnStates[lane]),
                   lane * laneRows, laneRows, onCpu, buffers.liveRowsPerLane,
                   pipelineDeferred);
    operators.linear().addDecodeBatch(
        g, buffers.ple.embedding, weights.perLayerEmbedding.keyProjection,
        buffers.ple.keys,
        {geometry.residualWidth(), geometry.pleEmbeddingSize}, lanes, stats);
    operators.linear().addDecodeBatch(
        g, buffers.ple.embedding, weights.perLayerEmbedding.valueProjection,
        buffers.ple.values, {geometry.hiddenSize, geometry.pleEmbeddingSize},
        lanes, stats);
    // Which rows to keep is not known until the verifier decides, so the
    // history is written by the state commit, not here.
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      addPleApply(
          weights, backend, g, buffers.ple, geometry,
          auxiliaryView(backend, geometry, buffers.currentGdnStates[lane]),
          backend.view(buffers.ple.normalized, lane * laneNormalized,
                       laneNormalized),
          backend.view(buffers.hidden[layerIndex & 1],
                       uint64_t{lane} * laneRows * width, laneRows * width),
          lane * laneRows, laneRows);
    }
  };

  // priorWorkComplete: every earlier submission of this step has finished,
  // so host-side reads of GPU-written inputs are safe.
  auto encodeAttentionHC = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               bool priorWorkComplete = false) {
    if (geometry.hasPerLayerEmbedding() && layerIndex == geometry.pleLayer)
      encodePerLayerEmbedding(g, layerIndex, priorWorkComplete);
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    addHyperConnection(g, geometry, input, layer.attentionHyperConnection,
                       buffers.normalized, buffers.hyperReduced,
                       buffers.hyperMixed, buffers.hyperInjection, hcRows,
                       hcStep);
  };

  auto encodeMixer = [&](metal::CommandGraph &g, uint32_t layerIndex,
                         uint32_t &gdnIdx, uint32_t &attnIdx) -> metal::MetalBuffer {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer mixerOutBuffer;
    if (std::holds_alternative<QwenGdnWeights>(layer.mixer)) {
      const auto &mixer = std::get<QwenGdnWeights>(layer.mixer);
      operators.linear().addDecodeBatch(
          g, buffers.hyperMixed, mixer.inputProjectionQ8,
          buffers.gdnPacked[gdnIdx], gdnInput, lanes, stats);
      ops::GDN::addDecode(
          g,
          {buffers.gdnPacked[gdnIdx], mixer.convolutionWeights,
           buffers.currentGdnStates, buffers.nextGdnStates,
           buffers.gdnMixed[gdnIdx], mixer.decay, mixer.timeBias,
           buffers.gdnDecay[gdnIdx], buffers.gdnBeta[gdnIdx],
           buffers.recurrent, mixer.mixerNorm, buffers.gdnHidden,
           buffers.arrived, buffers.generation},
          gdnShape(geometry, [] {
            const char *v = std::getenv("SPLASH_TREE_DRAFT");
            return (v != nullptr && std::atoi(v) != 0) ? 0x60132100u : 0u;
          }()), lanes, gdnIdx,
          {geometry.stateLayout.convolutionLayerBytes(),
           geometry.stateLayout.recurrentLayerBytes(),
           geometry.stateLayout.convolutionBytes()});
      operators.linear().addDecodeBatch(g, buffers.gdnHidden,
                                        mixer.outputProjectionQ8,
                                        buffers.gdnOutput, mixerOutput, lanes,
                                        stats);
      mixerOutBuffer = buffers.gdnOutput;
      ++gdnIdx;
    } else {
      const auto &mixer = std::get<QwenAttentionWeights>(layer.mixer);
      const uint32_t tileRows = kv::kPageTokens;
      operators.linear().addDecodeBatch(
          g, buffers.hyperMixed, mixer.inputProjectionQ8, buffers.fullPacked,
          attentionInput, lanes, stats);
      ops::PagedAttention::addVerifyProjection(
          g, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm,
          buffers.ropeCos, buffers.ropeSin, buffers.fullQueries,
          buffers.chunkKeys[attnIdx], buffers.chunkValues[attnIdx],
          ExecutionLimits::targetVerifyRows, tileRows, tileRows,
          geometry.attentionQueryHeads, geometry.kvLayout, lanes);
      ops::PagedAttention::addVerify(
          g, kvLayers[attnIdx],
          {buffers.chunkKeys[attnIdx], buffers.chunkValues[attnIdx],
           buffers.fullQueries, buffers.attentionPartials,
           buffers.attentionStatistics, buffers.fullAttention,
           buffers.pageTables},
          q8, verify, attentionPlan);
      ops::PagedAttention::addVerifyGate(
          g, buffers.fullPacked, buffers.fullAttention,
          buffers.attentionHidden, ExecutionLimits::targetVerifyRows, tileRows,
          tileRows, geometry.attentionQueryHeads, geometry.kvLayout, lanes);
      operators.linear().addDecodeBatch(g, buffers.attentionHidden,
                                        mixer.outputProjectionQ8,
                                        buffers.attentionOutput, mixerOutput,
                                        lanes, stats);
      mixerOutBuffer = buffers.attentionOutput;
      ++attnIdx;
    }
    return mixerOutBuffer;
  };

  auto encodeMixerUpdate = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               metal::MetalBuffer mixerOutBuffer) {
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    g.add("hyper_connection_update",
          {input, mixerOutBuffer, buffers.hyperInjection}, hcParams,
          {64, 1, 1}, {256, 1, 1});
  };

  auto encodeMlpHC = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    addHyperConnection(g, geometry, input, layer.mlpHyperConnection,
                       buffers.normalized, buffers.hyperReduced,
                       buffers.hyperMixed, buffers.hyperInjection, hcRows,
                       hcStep);
  };

  auto encodeMoERoute = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    ops::MoE::addRoute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        layer.ffn, moePlan);
  };

  // Lookahead: layer `target`'s router applied to the input the previous
  // layer's router just read. Only a guess, used to start reads early.
  auto encodePredictRoute = [&](metal::CommandGraph &g, uint32_t target) {
    if (!weights.predictSelected) {
      weights.predictSelected = backend.allocateBuffer(
          buffers.selectedExperts.sizeBytes(), metal::BufferStorage::Shared,
          "qwen4exp-predict-selected");
      weights.predictWeights = backend.allocateBuffer(
          buffers.routingWeights.sizeBytes(), metal::BufferStorage::Shared,
          "qwen4exp-predict-weights");
      weights.predictScratch = backend.allocateBuffer(
          buffers.groupedInput.sizeBytes(), metal::BufferStorage::Shared,
          "qwen4exp-predict-scratch");
    }
    // Scores only; the host picks the likely experts from them.
    ops::MoE::addRouteScores(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         weights.predictSelected, weights.predictWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, weights.predictScratch, buffers.expertIntermediate,
         buffers.expertOutput},
        weights.layers[target].ffn, moePlan);
  };

  auto encodeMoEExecute = [&](metal::CommandGraph &g, const ops::MoeWeights &weightsToUse,
                              bool hostGrouped = false) {
    ops::MoE::addExecute(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weightsToUse, moePlan, /*addResidual=*/false, hostGrouped);
  };

  auto encodeMlpUpdate = [&](metal::CommandGraph &g, uint32_t layerIndex) {
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    g.add("hyper_connection_update_out",
          {input, output, buffers.gdnOutput, buffers.hyperInjection},
          hcParams, {64, 1, 1}, {256, 1, 1});
  };


  auto encodeHead = [&](metal::CommandGraph &g) {
    addHyperConnection(g, geometry, buffers.hidden[geometry.layers & 1],
                       weights.hyperConnectionMixer, buffers.normalized,
                       buffers.hyperReduced, buffers.finalHidden, {}, hcRows,
                       hcStep);
    const ops::LinearMatrix head{geometry.vocabularySize, geometry.hiddenSize};
    operators.linear().addDecodeBatch(g, buffers.finalHidden,
                                      weights.logitsProjection, buffers.logits,
                                      head, lanes, stats);
  };

  const uint32_t R = std::min(weights.residentLayers, geometry.layers);
  const bool useStreamingCache = (R < geometry.layers && weights.streamingCacheGate);

  if (!useStreamingCache) {
    for (uint32_t layerIndex = 0; layerIndex < geometry.layers; ++layerIndex) {
      encodeAttentionHC(graph, layerIndex);
      metal::MetalBuffer mixerOut = encodeMixer(graph, layerIndex, gdnIndex, attentionIndex);
      encodeMixerUpdate(graph, layerIndex, mixerOut);
      encodeMlpHC(graph, layerIndex);
      ops::MoE::add(
          graph,
          {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
           buffers.selectedExperts, buffers.routingWeights,
           buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
           buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
           buffers.expertOutput},
          weights.layers[layerIndex].ffn, moePlan, /*addResidual=*/false);
      encodeMlpUpdate(graph, layerIndex);
    }
    if (gdnIndex != geometry.stateLayout.layers ||
        attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
      throw std::logic_error("Qwen target layer partition mismatch");
    }
    encodeHead(graph);
    return;
  }

  // =========================================================================
  // Staged Streaming Execution Path with Active Expert Caching & Speculation
  // =========================================================================

  uint32_t totalMisses = 0;

  // Layer `geometry.layers` is the MTP head's decoder layer.
  auto layerRef = [&](uint32_t index) -> const Qwen4ExpLayerWeights & {
    return index < geometry.layers ? weights.layers[index] : *weights.mtpLayer;
  };
  // Decode misses waiting to be read (deferred staging), and their slots.
  struct DecodeMiss { uint32_t expert; uint32_t slot; };
  std::vector<DecodeMiss> pendingMisses;
  uint32_t pendingLayer = 0;
  std::bitset<512> pendingMissSlots;
  // deferReads: assign slots but leave the missing experts' reads for
  // readPendingMisses(), so the GPU can run the cached ones meanwhile.
  auto stageActiveExperts = [&](uint32_t layerIndex, bool deferReads = false) -> bool {
    auto &cache = layerRef(layerIndex).expertCache;
    // Predicted experts may still be loading; their slots are already claimed.
    // Predicted experts may still be loading; their slots are claimed. Wait
    // only for the ones this step uses (below, once they are known).
    auto *selPtr = static_cast<uint32_t *>(buffers.selectedExperts.contents());
    const uint32_t routesPerRow = moePlan.shape().routesPerToken();
    const uint32_t expertsPerToken = weights.layout.expertsPerToken;
    const uint32_t totalExperts = weights.layout.experts;
    constexpr uint32_t kMaxExperts = 512;
    logRoutes('D', layerIndex, selPtr, rows, routesPerRow, expertsPerToken,
              buffers.liveRowsPerLane, ExecutionLimits::targetVerifyRows);
    int16_t stepExpertSeen[kMaxExperts];
    std::fill_n(stepExpertSeen, kMaxExperts, -1);
    std::vector<uint32_t> uniqueExperts;
    uniqueExperts.reserve(32);

    for (uint32_t r = 0; r < rows; ++r) {
      // Rows that cannot be kept select no expert at all, so they neither
      // cost expert work nor evict experts the kept rows will need.
      if (r % ExecutionLimits::targetVerifyRows >= buffers.liveRowsPerLane) {
        for (uint32_t k = 0; k < expertsPerToken; ++k)
          selPtr[r * routesPerRow + k] = kMoeSkippedRoute;
        continue;
      }
      for (uint32_t k = 0; k < expertsPerToken; ++k) {
        uint32_t exp = selPtr[r * routesPerRow + k];
        if (exp < totalExperts && stepExpertSeen[exp] == -1) {
          stepExpertSeen[exp] = 1;
          uniqueExperts.push_back(exp);
        }
      }
    }

    if (layerIndex < weights.lastSelectedExperts.size()) {
      weights.lastSelectedExperts[layerIndex] = uniqueExperts;
    }

    if (uniqueExperts.size() > cache.capacity) {
      return false;
    }

    using Miss = DecodeMiss;
    std::vector<Miss> misses;
    misses.reserve(uniqueExperts.size());
    ++cache.clock;

    int16_t expertToAssignedSlot[kMaxExperts];
    std::fill_n(expertToAssignedSlot, kMaxExperts, -1);

    for (uint32_t exp : uniqueExperts)
      countExpertUse(cache, exp);
    if (cache.slotReads) {
      const auto waitStart = AwakeClock::now();
      for (uint32_t exp : uniqueExperts) {
        const int16_t slot = cache.expertToSlot[exp];
        if (slot >= 0)
          while (cache.slotReads[slot].load(std::memory_order_acquire)) {}
      }
      hostParts[0] += std::chrono::duration<double, std::milli>(
                          AwakeClock::now() - waitStart).count();
    }
    if (layerIndex < geometry.layers && predictedWide[layerIndex].any())
      for (uint32_t exp : uniqueExperts) {
        ++predictionCounts[0];
        predictionCounts[1] += predictedNarrow[layerIndex][exp];
        predictionCounts[2] += predictedWide[layerIndex][exp];
        if (cache.expertToSlot[exp] < 0) {
          ++predictionCounts[3];
          predictionCounts[4] += predictedWide[layerIndex][exp];
        }
      }
    for (uint32_t exp : uniqueExperts) {
      int16_t slot = cache.expertToSlot[exp];
      if (slot != -1) {
        // Cache hit: refresh LRU timestamp
        cache.lruTime[slot] = cache.clock;
        expertToAssignedSlot[exp] = slot;
      } else {
        // Cache miss: find a slot
        uint32_t assignSlot = 0;
        if (cache.numCached < cache.capacity) {
          assignSlot = cache.numCached++;
        } else {
          // Evict the least-used slot this step does not need.
          const int32_t victim = pickVictim(cache);
          assignSlot = victim >= 0 ? static_cast<uint32_t>(victim) : 0;
          int16_t evicted = cache.slotToExpert[assignSlot];
          if (evicted != -1) {
            cache.expertToSlot[evicted] = -1;
          }
        }
        cache.slotToExpert[assignSlot] = exp;
        cache.expertToSlot[exp] = assignSlot;
        cache.lruTime[assignSlot] = cache.clock;
        expertToAssignedSlot[exp] = assignSlot;
        misses.push_back({exp, assignSlot});
      }
    }

    totalMisses += static_cast<uint32_t>(misses.size());

    pendingMissSlots.reset();
    for (const Miss &miss : misses) pendingMissSlots.set(miss.slot);
    if (deferReads) {
      pendingMisses = misses;
      pendingLayer = layerIndex;
    } else if (!misses.empty()) {
      const auto &layer = layerRef(layerIndex);
      const uint64_t stride = layer.ffn.expertGate.expertStrideBytes;
      char *cg = static_cast<char *>(cache.cacheGate.contents());
      char *cu = static_cast<char *>(cache.cacheUp.contents());
      char *cd = static_cast<char *>(cache.cacheDown.contents());

      // The expert regions are mapped for random access, so copying a missed
      // expert faults it in one 16 KB page at a time, each its own disk read.
      // Asking for every missed range first lets the reads go out together
      // and in large pieces; the copies below then find the pages in memory.
      const auto readStart = AwakeClock::now();
      readMissedExperts(layer.expertSource, misses.data(), misses.size(),
                        stride, cg, cu, cd);
      hostParts[1] += std::chrono::duration<double, std::milli>(
                          AwakeClock::now() - readStart).count();
    }

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t k = 0; k < expertsPerToken; ++k) {
        uint32_t exp = selPtr[r * routesPerRow + k];
        if (exp < totalExperts && expertToAssignedSlot[exp] >= 0) {
          selPtr[r * routesPerRow + k] = expertToAssignedSlot[exp];
        }
      }
      selPtr[r * routesPerRow + expertsPerToken] = 512;
    }
    return true;
  };

  // Start background reads for the experts lookahead predicts layer `layer`
  // will want. Claims least-recently-used slots of that layer only; nothing
  // on the GPU reads that layer's cache until its own stage.
  auto prefetchPredicted = [&](uint32_t layer) {
    if (!weights.predictSelected || layer >= geometry.layers)
      return;
    auto &cache = weights.layers[layer].expertCache;
    const auto *predictedScores =
        static_cast<const uint16_t *>(weights.predictScratch.contents());
    const uint32_t width = moePlan.shape().routerWidth();
    struct Miss { uint32_t expert; uint32_t slot; };
    std::vector<Miss> misses;
    ++cache.clock;
    constexpr uint32_t kMaxPredicted = 512;
    std::bitset<kMaxPredicted> wanted;
    std::array<uint8_t, kMaxPredicted> votes{};
    std::vector<uint32_t> order;
    for (uint32_t r = 0; r < rows; ++r) {
      if (r % ExecutionLimits::targetVerifyRows >= buffers.liveRowsPerLane)
        continue;
      // The predicted top 6 of each row's 10. The prediction is right for
      // ~2/3 of its top 10, and every wrong guess is a wasted SSD read that
      // competes with the reads that matter: 10 -> 6 cut reads 133 -> 95 a
      // step and gave 38.6 -> 40.3 tok/s on the 10-prompt suite (5-8 tie;
      // 12 and none are slower). SPLASH_LOOKAHEAD_EXPERTS overrides.
      static const uint32_t widened = [] {
        const char *value = std::getenv("SPLASH_LOOKAHEAD_EXPERTS");
        return value ? static_cast<uint32_t>(std::atoi(value)) : 6u;
      }();
      const uint32_t guesses = std::clamp(widened, 1u, 16u);
      uint32_t ids[16];
      topExperts(predictedScores + uint64_t{r} * width, weights.layout.experts,
                 16, ids, nullptr);
      for (uint32_t k = 0; k < 16; ++k) {
        predictedWide[layer].set(ids[k]);
        if (k < guesses) predictedNarrow[layer].set(ids[k]);
        if (k < 10 && ids[k] < kMaxPredicted) ++votes[ids[k]];
      }
      for (uint32_t k = 0; k < guesses; ++k)
        if (ids[k] < kMaxPredicted && !wanted.test(ids[k])) {
          wanted.set(ids[k]);
          order.push_back(ids[k]);
        }
    }
    // SPLASH_LOOKAHEAD_CONSENSUS=n also reads experts that n or more rows
    // predict in their top 10 (0, the default, reads only each row's top k).
    // Tried 2026-09-23: n = 2 or 3, with top 4 or 6 - all within noise of the
    // default (41.2-41.5 tok/s), so it stays off.
    static const uint32_t consensus = [] {
      const char *value = std::getenv("SPLASH_LOOKAHEAD_CONSENSUS");
      return value ? static_cast<uint32_t>(std::atoi(value)) : 0u;
    }();
    if (consensus)
      for (uint32_t e = 0; e < weights.layout.experts && e < kMaxPredicted; ++e)
        if (votes[e] >= consensus && !wanted.test(e)) {
          wanted.set(e);
          order.push_back(e);
        }
    for (const uint32_t expert : order) {
      const int16_t existing = cache.expertToSlot[expert];
      if (existing >= 0) {
        cache.lruTime[existing] = cache.clock;
        continue;
      }
      uint32_t slot = 0;
      if (cache.numCached < cache.capacity) {
        slot = cache.numCached++;
      } else {
        const int32_t victim = pickVictim(cache);
        if (victim < 0)
          break;
        slot = static_cast<uint32_t>(victim);
        if (cache.slotToExpert[slot] >= 0)
          cache.expertToSlot[cache.slotToExpert[slot]] = -1;
      }
      cache.slotToExpert[slot] = static_cast<int16_t>(expert);
      cache.expertToSlot[expert] = static_cast<int16_t>(slot);
      cache.lruTime[slot] = cache.clock;
      misses.push_back({expert, slot});
    }
    if (misses.empty())
      return;
    weights.predictIssued += misses.size();
    if (!cache.inflight)
      cache.inflight = dispatch_group_create();
    if (!cache.slotReads) {
      cache.slotReads.reset(new std::atomic<uint32_t>[cache.capacity]);
      for (uint32_t slot = 0; slot < cache.capacity; ++slot) cache.slotReads[slot] = 0;
    }
    for (const Miss &miss : misses)
      cache.slotReads[miss.slot].fetch_add(3 * readPieces(), std::memory_order_relaxed);
    auto slotReads = cache.slotReads;
    const auto &source = weights.layers[layer].expertSource;
    const uint64_t stride = weights.layers[layer].ffn.expertGate.expertStrideBytes;
    char *gate = static_cast<char *>(cache.cacheGate.contents());
    char *up = static_cast<char *>(cache.cacheUp.contents());
    char *down = static_cast<char *>(cache.cacheDown.contents());
    auto owned = std::make_shared<std::vector<Miss>>(std::move(misses));
    dispatch_group_async(cache.inflight,
                         dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      readMissedExperts(source, owned->data(), owned->size(), stride, gate, up, down,
                        slotReads.get());
    });
    cache.pending = true;
  };

  // Host routing for a pipelined step. The GPU stage ends at the router's
  // scores; the host, already paused there to stage experts, selects and
  // groups. Two single-threadgroup kernels that each cost ~0.1-0.2 ms of
  // latency per layer, a third of decode GPU time, become microseconds here.
  auto encodeRouteScores = [&](metal::CommandGraph &g, uint32_t layerIndex,
                               metal::MetalBuffer scores) {
    ops::MoE::addRouteScores(
        g,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, std::move(scores), buffers.expertIntermediate,
         buffers.expertOutput},
        layerRef(layerIndex).ffn, moePlan);
  };
  const uint32_t hostRoutesPerRow = moePlan.shape().routesPerToken();
  auto isLive = [&](uint32_t row) {
    return row % ExecutionLimits::targetVerifyRows < buffers.liveRowsPerLane;
  };
  auto hostSelect = [&](uint32_t layerIndex) {
    const auto *scores = static_cast<const uint16_t *>(buffers.groupedInput.contents());
    const auto *input = static_cast<const uint16_t *>(buffers.hyperMixed.contents());
    auto *selected = static_cast<uint32_t *>(buffers.selectedExperts.contents());
    auto *routing = static_cast<uint16_t *>(buffers.routingWeights.contents());
    const uint32_t k = weights.layout.expertsPerToken;
    const uint32_t width = moePlan.shape().routerWidth();
    for (uint32_t r = 0; r < rows; ++r) {
      if (!isLive(r)) continue;
      uint32_t ids[16];
      float w[16];
      topExperts(scores + uint64_t{r} * width, weights.layout.experts, k, ids, w);
      for (uint32_t j = 0; j < k; ++j) {
        selected[r * hostRoutesPerRow + j] = ids[j];
        routing[r * hostRoutesPerRow + j] = toBf16(w[j]);
      }
      selected[r * hostRoutesPerRow + k] = weights.layout.experts;
      routing[r * hostRoutesPerRow + k] = toBf16(sharedExpertGate(
          layerRef(layerIndex).ffn.sharedExpertGate,
          input + uint64_t{r} * geometry.hiddenSize, geometry.hiddenSize));
    }
  };
  // As moe_group_routes lays it out: routed tiles in ascending slot order,
  // tile_rows each, padding marked ~0; then the shared expert's tiles. Rows
  // that cannot be kept get no routes at all.
  // waves: order tiles as cached experts, then the shared expert, then the
  // experts still being read (pendingMissSlots), and write the two ranges
  // [0, split) and [split, end) to weights.decodeRanges for the ranged passes.
  auto hostGroup = [&](bool waves = false) {
    const auto *selected = static_cast<const uint32_t *>(buffers.selectedExperts.contents());
    auto *tiles = static_cast<MoeTileDescriptor *>(buffers.tileDescriptors.contents());
    auto *tileCount = static_cast<uint32_t *>(buffers.tileCount.contents());
    auto *grouped = static_cast<uint32_t *>(buffers.groupedRoutes.contents());
    auto *routeRows = static_cast<uint32_t *>(buffers.routeRows.contents());
    const uint32_t k = weights.layout.expertsPerToken;
    const uint32_t tileRows = moePlan.tileRows();
    const uint32_t experts = moePlan.shape().experts;
    // Counting sort of the live routes by slot, without allocating: a
    // step has at most rows x k of them.
    std::array<uint16_t, 513> start{};
    std::array<uint32_t, ExecutionLimits::maximumBatchWidth *
                             ExecutionLimits::targetVerifyRows * 16> sorted;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t j = 0; j <= k; ++j) {
        const uint32_t route = r * hostRoutesPerRow + j;
        routeRows[route] = 0xFFFFFFFFu;
        if (!isLive(r) || j == k) continue;
        const uint32_t e = selected[route];
        if (e < experts) ++start[e + 1];
      }
    for (uint32_t e = 0; e < experts; ++e) start[e + 1] += start[e];
    std::array<uint16_t, 512> fill{};
    for (uint32_t r = 0; r < rows; ++r) {
      if (!isLive(r)) continue;
      for (uint32_t j = 0; j < k; ++j) {
        const uint32_t route = r * hostRoutesPerRow + j;
        const uint32_t e = selected[route];
        if (e < experts) sorted[start[e] + fill[e]++] = route;
      }
    }
    uint32_t tile = 0;
    auto addExpert = [&](uint32_t e) {
      const std::span<const uint32_t> routes(sorted.data() + start[e],
                                             start[e + 1] - start[e]);
      for (uint32_t first = 0; first < routes.size(); first += tileRows) {
        const uint32_t count = std::min<uint32_t>(tileRows, routes.size() - first);
        tiles[tile] = MoeTileDescriptor{e, count};
        for (uint32_t i = 0; i < tileRows; ++i) {
          const uint32_t row = tile * tileRows + i;
          if (i < count) {
            grouped[row] = routes[first + i];
            routeRows[routes[first + i]] = row;
          } else {
            grouped[row] = 0xFFFFFFFFu;
          }
        }
        ++tile;
      }
    };
    for (uint32_t e = 0; e < experts; ++e)
      if (!waves || !pendingMissSlots[e]) addExpert(e);
    std::vector<uint32_t> shared;
    for (uint32_t r = 0; r < rows; ++r)
      if (isLive(r)) shared.push_back(r * hostRoutesPerRow + k);
    for (uint32_t first = 0; first < shared.size(); first += tileRows) {
      const uint32_t count = std::min<uint32_t>(tileRows, shared.size() - first);
      tiles[tile] = MoeTileDescriptor{experts, count};
      for (uint32_t i = 0; i < tileRows; ++i) {
        const uint32_t row = tile * tileRows + i;
        if (i < count) {
          grouped[row] = shared[first + i];
          routeRows[shared[first + i]] = row;
        } else {
          grouped[row] = 0xFFFFFFFFu;
        }
      }
      ++tile;
    }
    uint32_t split = tile;
    if (waves) {
      for (uint32_t e = 0; e < experts; ++e)
        if (pendingMissSlots[e]) addExpert(e);
      auto *ranges = static_cast<uint32_t *>(weights.decodeRanges.contents());
      ranges[0] = 0;
      ranges[1] = split;
      ranges[64] = split;   // second range, 256 bytes on
      ranges[65] = tile;
    }
    *tileCount = tile;
  };

  auto makeCacheWeights = [&](uint32_t layerIndex) -> ops::MoeWeights {
    const auto &layer = layerRef(layerIndex);
    const auto &cache = layer.expertCache;
    const uint64_t stride = layer.ffn.expertGate.expertStrideBytes;
    return ops::MoeWeights{
        .router = layer.ffn.router,
        .expertGate = {cache.cacheGate, cache.capacity,
                       weights.layout.expertIntermediateSize, weights.layout.hiddenSize, stride},
        .expertUp = {cache.cacheUp, cache.capacity,
                     weights.layout.expertIntermediateSize, weights.layout.hiddenSize, stride},
        .expertDown = {cache.cacheDown, cache.capacity,
                       weights.layout.hiddenSize, weights.layout.expertIntermediateSize, stride},
        .sharedGate = layer.ffn.sharedGate,
        .sharedUp = layer.ffn.sharedUp,
        .sharedDown = layer.ffn.sharedDown,
        .sharedExpertGate = layer.ffn.sharedExpertGate,
    };
  };


  // -------------------------------------------------------------------------
  // MTP draft head (lane 0). Runs before this step's verify: its inputs are
  // the previous step's final residual rows, still in Hidden0. Each draft
  // step is two GPU stages around a host pause for the head's experts:
  //   A  combiner, the head's attention layer (13th KV layer), MLP mix, router
  //   B  experts, update, the head's own mixer, the shared output head
  // In shadow mode its guesses are only scored against what the target then
  // produces; nothing is proposed.
  // -------------------------------------------------------------------------
  uint32_t drafted = 0;
  // Guesses per step, at most one per verify row after the anchor.
  struct AdaptiveDraftController {
    uint32_t rollingWindow = 0xFF;
    uint32_t cycles = 0;
    uint32_t reject2Streak = 0;
    uint32_t lastProposed = 0;

    void update(uint32_t retained) {
      if (lastProposed == 0) return;
      uint32_t acceptedDrafts = (retained > 0) ? (retained - 1) : 0;
      bool acceptedFirst = (acceptedDrafts >= 1);
      rollingWindow = ((rollingWindow << 1) | (acceptedFirst ? 1u : 0u)) & 0xFFu;
      cycles++;

      if (lastProposed >= 2) {
        if (acceptedDrafts >= 2) {
          reject2Streak = 0;
        } else if (acceptedDrafts == 1) {
          reject2Streak++;
        }
      }
    }

    uint32_t calculateMaxDrafts(uint32_t configuredMax) const {
      static const int overrideDepth = [] {
        const char *env = std::getenv("SPLASH_MTP_DEPTH");
        return (env && env[0]) ? std::atoi(env) : 0;
      }();
      if (overrideDepth > 0) {
        return static_cast<uint32_t>(std::clamp(overrideDepth, 1, static_cast<int>(configuredMax)));
      }

      static const bool adaptiveEnabled = [] {
        const char *env = std::getenv("SPLASH_MTP_ADAPTIVE_DEPTH");
        return env ? (std::atoi(env) != 0) : true;
      }();
      if (!adaptiveEnabled || configuredMax <= 2) {
        return configuredMax;
      }

      uint32_t bits = 0;
      for (uint32_t i = 0; i < 8; ++i) {
        bits += (rollingWindow >> i) & 1u;
      }

      if (reject2Streak >= 2 || (cycles >= 8 && bits < 4)) {
        return std::min(configuredMax, 2u);
      }
      if (cycles >= 8 && bits < 6) {
        return std::min(configuredMax, 3u);
      }
      return configuredMax;
    }
  };
  static AdaptiveDraftController adaptiveController;
  if (lanes == 1 && buffers.mtp[0].rows > 0) {
    adaptiveController.update(buffers.mtp[0].rows);
  }
  const uint32_t baseMaxDrafts = mtpDraftLimit();
  const uint32_t maxDrafts = adaptiveController.calculateMaxDrafts(baseMaxDrafts);
  std::array<std::array<uint32_t, 16>, 7> draftCandidates{};
  std::array<std::array<float, 16>, 7> draftProbabilities{};
  // SPLASH_TRACE=path appends one JSON line per verify step: the head's
  // top-4 guesses and confidence at each depth, how many guesses the target
  // kept, and the step's timing split. dev/benchmarks/qwen4exp/trace_report.py
  // reads it. A step's line is written when the next step starts, since only
  // then is it known what was kept.
  static FILE *traceFile = [] {
    const char *path = std::getenv("SPLASH_TRACE");
    return path ? std::fopen(path, "a") : nullptr;
  }();
  struct DraftTrace {
    uint32_t depths = 0;
    std::array<std::array<uint32_t, 4>, 7> ids{};
    std::array<std::array<float, 4>, 7> logits{};
    std::array<float, 7> confidence{};
    std::array<uint32_t, 7> chosen{};
  } draftTrace;
  auto runMtpDraft = [&]() -> std::array<uint32_t, 7> {
    std::array<uint32_t, 7> drafts{};
    draftTrace = DraftTrace{};
    const QwenMtpLane &mtp = buffers.mtp[0];
    const auto &head = *weights.mtpLayer;
    const auto &combiner = *weights.mtpCombiner;
    const uint32_t mtpIndex = geometry.layers;
    const uint32_t kvIndex = geometry.kvLayout.attentionLayers - 1;
    constexpr uint32_t kRows = ExecutionLimits::targetVerifyRows;
    const uint32_t hidden = geometry.hiddenSize, width = geometry.residualWidth();
    const uint32_t vocabulary = geometry.vocabularySize;
    auto shared = [&](metal::MetalBuffer &b, uint64_t bytes, const char *label) {
      if (!b) b = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    };
    shared(weights.mtpTokens, kRows * 4, "mtp-tokens");
    shared(weights.mtpEmbed, kRows * hidden * 2, "mtp-embed");
    shared(weights.mtpNorm, kRows * hidden * 2, "mtp-norm");
    shared(weights.mtpE, kRows * hidden * 2, "mtp-e");
    shared(weights.mtpOnes, kRows * 4 * 2, "mtp-ones");
    shared(weights.mtpCos, kRows * geometry.rotaryPairs * 4, "mtp-cos");
    shared(weights.mtpSin, kRows * geometry.rotaryPairs * 4, "mtp-sin");
    shared(weights.mtpHin, kRows * uint64_t{width} * 2, "mtp-h-in");
    std::fill_n(static_cast<uint16_t *>(weights.mtpOnes.contents()), kRows * 4,
                uint16_t{0x3F80});  // bf16 1.0
    auto *hIn = static_cast<uint16_t *>(weights.mtpHin.contents());
    const auto *hidden0 = static_cast<const uint16_t *>(buffers.hidden[0].contents());
    metal::MetalBuffer X = buffers.hidden[1], Y = buffers.hidden[0];
    const HyperConnectionParams all{kRows, hidden, geometry.hyperConnectionCount,
                                    geometry.hyperConnectionLowRank, 1e-6f, 1, 1, 0};
    const HyperConnectionParams single{kRows, hidden, 1,
                                       geometry.hyperConnectionLowRank, 1e-6f, 0, 1, 0};
    const ops::LinearMatrix square{hidden, hidden};
    const ops::LinearMatrix attentionInput{geometry.packedAttentionWidth, hidden};
    const ops::LinearMatrix mixerOutput{hidden, geometry.attentionWidth};

    // Precompute rotary frequencies once outside step:
    std::vector<float> rotaryFrequencies(geometry.rotaryPairs);
    for (uint32_t d = 0; d < geometry.rotaryPairs; ++d) {
      rotaryFrequencies[d] = std::pow(geometry.rotaryTheta,
                                      -float(d) / float(geometry.rotaryPairs));
    }

    // One draft step over `live` rows at positions position..position+live-1,
    // whose inputs are rows 0..live-1 of mtpHin and `tokens`. Returns the
    // head's top token after the last row.
    float lastConfidence = 1.0f;

    std::array<uint32_t, 16> lastCandidates{};
    std::array<float, 16> lastProbabilities{};
    auto step = [&](uint64_t position, uint32_t live,
                    const uint32_t *tokens) -> uint32_t {
      auto *tokenOut = static_cast<uint32_t *>(weights.mtpTokens.contents());
      auto *cosines = static_cast<float *>(weights.mtpCos.contents());
      auto *sines = static_cast<float *>(weights.mtpSin.contents());
      for (uint32_t r = 0; r < kRows; ++r) {
        tokenOut[r] = tokens[std::min(r, live - 1)];
        for (uint32_t d = 0; d < geometry.rotaryPairs; ++d) {
          const float angle = float(position + r) * rotaryFrequencies[d];
          cosines[r * geometry.rotaryPairs + d] = std::cos(angle);
          sines[r * geometry.rotaryPairs + d] = std::sin(angle);
        }
      }
      std::array<kv::Q8ChunkedPrefillParams, ExecutionLimits::maximumBatchWidth> q8m{};
      std::array<kv::Q8VerifyAttentionParams, ExecutionLimits::maximumBatchWidth> vm{};
      std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
      for (uint32_t lane = 0; lane < ExecutionLimits::maximumBatchWidth; ++lane) {
        q8m[lane] = ops::PagedAttention::prefillParams(
            position, kRows, kv::kPageTokens, mtp.pageTable, buffers.kvPageCount);
        vm[lane] = kv::q8VerifyAttentionParams(
            q8m[lane].committed_tokens, q8m[lane].chunk_tokens,
            q8m[lane].chunk_stride, q8m[lane].page_table_entries,
            q8m[lane].physical_page_count);
        histories[lane] = q8m[lane].committed_tokens;
      }
      const auto plan = operators.verifyAttention(
          lanes, geometry.attentionQueryHeads, geometry.kvLayout, histories);

      metal::CommandGraph a;
      ops::Embedding::add(a, weights.mtpTokens, weights.tokenEmbedding,
                          weights.mtpEmbed, kRows);
      a.add("hyper_connection_rms", {weights.mtpEmbed, combiner.embeddingNorm,
                                     weights.mtpNorm},
            single, {kRows, 1, 1}, {256, 1, 1});
      operators.linear().addDecodeBatch(a, weights.mtpNorm, combiner.fcEmbedding,
                                        weights.mtpE, square, lanes, stats);
      a.add("hyper_connection_rms", {weights.mtpHin, combiner.hiddenNorm,
                                     buffers.normalized},
            all, {kRows, geometry.hyperConnectionCount, 1}, {256, 1, 1});
      // fc_hidden on each stream: 8 rows of 4 streams are 32 rows of hidden.
      operators.linear().addDecodeBatch(a, buffers.normalized, combiner.fcHidden,
                                        X, square, 4 * lanes, stats);
      a.add("hyper_connection_update", {X, weights.mtpE, weights.mtpOnes}, all,
            {64, 1, 1}, {256, 1, 1});
      addHyperConnection(a, geometry, X, head.attentionHyperConnection,
                         buffers.normalized, buffers.hyperReduced,
                         buffers.hyperMixed, buffers.hyperInjection, kRows, 1);
      const auto &mixer = std::get<QwenAttentionWeights>(head.mixer);
      operators.linear().addDecodeBatch(a, buffers.hyperMixed, mixer.inputProjectionQ8,
                                        buffers.fullPacked, attentionInput, lanes, stats);
      ops::PagedAttention::addVerifyProjection(
          a, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm, weights.mtpCos,
          weights.mtpSin, buffers.fullQueries, buffers.chunkKeys[kvIndex],
          buffers.chunkValues[kvIndex], kRows, kv::kPageTokens, kv::kPageTokens,
          geometry.attentionQueryHeads, geometry.kvLayout, lanes);
      ops::PagedAttention::addVerify(
          a, kvLayers[kvIndex],
          {buffers.chunkKeys[kvIndex], buffers.chunkValues[kvIndex],
           buffers.fullQueries, buffers.attentionPartials,
           buffers.attentionStatistics, buffers.fullAttention, buffers.pageTables},
          q8m, vm, plan);
      ops::PagedAttention::addVerifyGate(
          a, buffers.fullPacked, buffers.fullAttention, buffers.attentionHidden,
          kRows, kv::kPageTokens, kv::kPageTokens, geometry.attentionQueryHeads,
          geometry.kvLayout, lanes);
      operators.linear().addDecodeBatch(a, buffers.attentionHidden,
                                        mixer.outputProjectionQ8,
                                        buffers.attentionOutput, mixerOutput, lanes, stats);
      a.add("hyper_connection_update", {X, buffers.attentionOutput,
                                        buffers.hyperInjection},
            all, {64, 1, 1}, {256, 1, 1});
      addHyperConnection(a, geometry, X, head.mlpHyperConnection,
                         buffers.normalized, buffers.hyperReduced,
                         buffers.hyperMixed, buffers.hyperInjection, kRows, 1);
      encodeRouteScores(a, mtpIndex, buffers.groupedInput);
      const auto clockA = AwakeClock::now();
      (void)backend.submitCommand(a.dispatches());
      const auto clockStage = AwakeClock::now();

      const uint32_t saved = buffers.liveRowsPerLane;
      buffers.liveRowsPerLane = live;
      hostSelect(mtpIndex);
      if (!stageActiveExperts(mtpIndex))
        throw std::logic_error("MTP head needs more experts than its cache holds");
      hostGroup();
      buffers.liveRowsPerLane = saved;
      const auto clockB = AwakeClock::now();

      metal::CommandGraph b;
      encodeMoEExecute(b, makeCacheWeights(mtpIndex), /*hostGrouped=*/true);
      b.add("hyper_connection_update_out", {X, Y, buffers.gdnOutput,
                                            buffers.hyperInjection},
            all, {64, 1, 1}, {256, 1, 1});
      addHyperConnection(b, geometry, Y, combiner.mixer, buffers.normalized,
                         buffers.hyperReduced, buffers.finalHidden, {}, kRows, 1);
      // With SPLASH_DRAFT_VOCAB the head scores only the most common tokens;
      // picked positions are mapped back to token ids below.
      const bool subset = !weights.draftVocabIds.empty();
      const uint32_t scored = subset ? weights.draftVocabProjection.outputSize : vocabulary;
      operators.linear().addDecodeBatch(b, buffers.finalHidden,
                                        subset ? weights.draftVocabProjection
                                               : weights.draftLogitsProjection,
                                        buffers.logits, {scored, hidden}, lanes, stats);
      // Top candidates per vocabulary slice on the GPU (mtp_pick.metal).
      constexpr uint32_t kSlices = 64, kCandidates = 16;
      shared(weights.mtpPickIds, kSlices * kCandidates * 4, "mtp-pick-ids");
      shared(weights.mtpPickValues, kSlices * kCandidates * 4, "mtp-pick-values");
      shared(weights.mtpPickMass, kSlices * 2 * 4, "mtp-pick-mass");
      b.add("mtp_pick_slices",
            {backend.view(buffers.logits, uint64_t{live - 1} * scored * 2,
                          uint64_t{scored} * 2),
             weights.mtpPickIds, weights.mtpPickValues, weights.mtpPickMass},
            scored, {kSlices, 1, 1}, {256, 1, 1});
      (void)backend.submitCommand(b.dispatches());
      const auto clockPick = AwakeClock::now();
      auto ms = [](auto from, auto to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
      };
      mtpParts[0] += ms(clockA, clockStage);
      mtpParts[1] += ms(clockStage, clockB);
      mtpParts[2] += ms(clockB, clockPick);
      struct PickTimer {
        AwakeClock::time_point start;
        double &sink;
        ~PickTimer() {
          sink += std::chrono::duration<double, std::milli>(
                      AwakeClock::now() - start).count();
        }
      } pickTimer{clockPick, mtpParts[3]};

      // Merge the slices' candidates: the 16 best overall (ties to the lower
      // id), and the exact softmax mass from each slice's max and sum.
      // Greedy keeps one candidate; four when tracing, for the record only.
      const uint32_t width16 = mtp.temperature > 0.0f
          ? std::min<uint32_t>(mtp.topK ? mtp.topK : 16, 16) : 4u;
      const auto *pickIds = static_cast<const uint32_t *>(weights.mtpPickIds.contents());
      const auto *pickValues = static_cast<const float *>(weights.mtpPickValues.contents());
      const auto *pickMass = static_cast<const float *>(weights.mtpPickMass.contents());
      std::array<uint32_t, 16> ids{};
      std::array<float, 16> values{};
      static const bool mtpMaskedDraft = [] {
        const char *v = std::getenv("SPLASH_MTP_MASKED_DRAFT");
        return v ? (std::atoi(v) != 0) : false;
      }();
      const bool hasDraftMask = mtpMaskedDraft && !buffers.draftMask.empty() && (live == 1);
      uint32_t filled = 0;
      for (uint32_t c = 0; c < kSlices * kCandidates; ++c) {
        const uint32_t id = pickIds[c];
        const float value = pickValues[c];
        if (id == UINT32_MAX) continue;
        if (hasDraftMask) {
          const uint32_t realId = subset ? weights.draftVocabIds[id] : id;
          const uint32_t wordIdx = realId / 32;
          const uint32_t bitIdx = realId % 32;
          if (wordIdx < buffers.draftMask.size() &&
              (buffers.draftMask[wordIdx] & (1U << bitIdx)) == 0) {
            continue;
          }
        }
        auto before = [&](uint32_t at) {
          return values[at] > value || (values[at] == value && ids[at] < id);
        };
        if (filled == width16 && before(width16 - 1)) continue;
        uint32_t at = filled < width16 ? filled++ : width16 - 1;
        while (at > 0 && !before(at - 1)) {
          ids[at] = ids[at - 1]; values[at] = values[at - 1]; --at;
        }
        ids[at] = id; values[at] = value;
      }
      if (filled == 0 && hasDraftMask) {
        for (uint32_t c = 0; c < kSlices * kCandidates; ++c) {
          const uint32_t id = pickIds[c];
          const float value = pickValues[c];
          if (id == UINT32_MAX) continue;
          auto before = [&](uint32_t at) {
            return values[at] > value || (values[at] == value && ids[at] < id);
          };
          if (filled == width16 && before(width16 - 1)) continue;
          uint32_t at = filled < width16 ? filled++ : width16 - 1;
          while (at > 0 && !before(at - 1)) {
            ids[at] = ids[at - 1]; values[at] = values[at - 1]; --at;
          }
          ids[at] = id; values[at] = value;
        }
      }
      if (subset)
        for (uint32_t i = 0; i < filled; ++i) ids[i] = weights.draftVocabIds[ids[i]];
      // Confidence: the head's own probability for its top token.
      float total = 0.0f;
      for (uint32_t slice = 0; slice < kSlices; ++slice)
        if (pickMass[2 * slice + 1] > 0.0f)
          total += pickMass[2 * slice + 1] * std::exp(pickMass[2 * slice] - values[0]);
      lastConfidence = 1.0f / total;
      if (traceFile && draftTrace.depths < 7) {
        const uint32_t d = draftTrace.depths++;
        for (uint32_t i = 0; i < 4; ++i) {
          draftTrace.ids[d][i] = i < filled ? ids[i] : UINT32_MAX;
          draftTrace.logits[d][i] = i < filled ? values[i] : 0.0f;
        }
        draftTrace.confidence[d] = lastConfidence;
      }
      lastCandidates.fill(UINT32_MAX);
      lastProbabilities.fill(0.0f);
      for (uint32_t i = 0; i < filled; ++i) {
        lastCandidates[i] = ids[i];
        lastProbabilities[i] = (i == 0) ? 1.0f : 0.0f;
      }
      if (mtp.temperature <= 0.0f) {
        return ids[0];
      }
      std::array<float, 16> q{};
      float sum = 0.0f;
      for (uint32_t i = 0; i < width16; ++i) {
        q[i] = std::exp((values[i] - values[0]) / mtp.temperature);
        sum += q[i];
      }
      uint32_t kept = width16;
      float cumulative = 0.0f;
      for (uint32_t i = 0; i < width16; ++i) {
        cumulative += q[i] / sum;
        if (cumulative >= mtp.topP) { kept = i + 1; break; }
      }
      float keptSum = 0.0f;
      for (uint32_t i = 0; i < kept; ++i) keptSum += q[i];
      static std::mt19937 random(20260922);
      for (uint32_t i = 0; i < kept; ++i) {
        lastCandidates[i] = ids[i];
        lastProbabilities[i] = q[i] / keptSum;
      }
      float draw = std::uniform_real_distribution<float>(0.0f, 1.0f)(random);
      for (uint32_t i = 0; i + 1 < kept; ++i) {
        if (draw < lastProbabilities[i]) return ids[i];
        draw -= lastProbabilities[i];
      }
      return ids[kept - 1];
    };

    // Step 1: refresh the accepted rows with the target's own residuals, and
    // guess the token after the anchor.
    for (uint32_t r = 0; r < mtp.rows; ++r)
      std::memcpy(hIn + uint64_t{r} * width,
                  hidden0 + uint64_t{mtp.firstRow + r} * width, width * 2);
    // Guessing stops once the head is unsure (as llama.cpp's
    // --spec-draft-p-min): an unlikely guess is usually rejected, and every
    // guessed row costs the verifier its own experts.
    static const float confident = [] {
      const char *value = std::getenv("SPLASH_MTP_P_MIN");
      return value ? static_cast<float>(std::atof(value)) : 0.3f;
    }();
    // A guess only pays off if every guess before it is kept too, and each
    // row checked costs ~7 ms (its experts, many read from the SSD). So
    // guessing also stops once the product of confidences so far drops
    // below this. trace_report.py predicted 37.6 -> 39.5 tok/s at 0.35;
    // measured 39.1 (0.25 and 0.45: 39.1, 38.0), same outputs.
    // SPLASH_MTP_CHAIN_MIN=0 turns it off.
    static const float chainConfident = [] {
      const char *value = std::getenv("SPLASH_MTP_CHAIN_MIN");
      return value ? static_cast<float>(std::atof(value)) : 0.35f;
    }();
    // Early-exit factor: token k only pays off if chainConfidence * P(k) >= chainConfident.
    // If chainConfidence * earlyExitFactor < chainConfident, drafting token k is almost
    // guaranteed to fail the chain check, so stop before dispatching MTP to the GPU.
    static const float earlyExitFactor = [] {
      const char *value = std::getenv("SPLASH_MTP_EARLY_EXIT_P");
      return value ? static_cast<float>(std::atof(value)) : 0.85f;
    }();
    static const bool treeDrafting = [] {
      const char *value = std::getenv("SPLASH_TREE_DRAFT");
      return value != nullptr && std::atoi(value) != 0;
    }();

    if (treeDrafting) {
      // Structure A Tree: 7 draft tokens
      // Row 1: drafts[0] = Guess 1A (Top-1 from Anchor, parent 0)
      // Row 2: drafts[1] = Guess 2A (Top-1 from Guess 1A, parent 1)
      // Row 3: drafts[2] = Guess 3A (Top-1 from Guess 2A, parent 2)
      // Row 4: drafts[3] = Guess 4A (Top-1 from Guess 3A, parent 3)
      // Row 5: drafts[4] = Guess 2B (Top-2 from Guess 1A, parent 1)
      // Row 6: drafts[5] = Guess 1B (Top-2 from Anchor, parent 0)
      // Row 7: drafts[6] = Guess 2C (Top-1 from Guess 1B, parent 6)
      const uint64_t anchorPosition = mtp.firstPosition + mtp.rows;
      const auto *y = static_cast<const uint16_t *>(Y.contents());

      drafts[0] = step(mtp.firstPosition, mtp.rows, mtp.tokens.data());
      draftTrace.chosen[0] = drafts[0];
      draftCandidates[0] = lastCandidates;
      draftProbabilities[0] = lastProbabilities;

      uint32_t guess1B = (lastCandidates[1] != UINT32_MAX) ? lastCandidates[1] : drafts[0];
      drafts[5] = guess1B;
      draftCandidates[5] = lastCandidates;
      draftProbabilities[5] = lastProbabilities;

      std::vector<uint16_t> yAnchor(width);
      std::memcpy(yAnchor.data(), y + uint64_t{mtp.rows - 1} * width, width * 2);

      // Guess 1A step -> produces Guess 2A (ids[0]) and Guess 2B (ids[1])
      std::memcpy(hIn, yAnchor.data(), width * 2);
      drafts[1] = step(anchorPosition, 1, &drafts[0]);
      draftTrace.chosen[1] = drafts[1];
      draftCandidates[1] = lastCandidates;
      draftProbabilities[1] = lastProbabilities;

      uint32_t guess2B = (lastCandidates[1] != UINT32_MAX) ? lastCandidates[1] : drafts[1];
      drafts[4] = guess2B;
      draftCandidates[4] = lastCandidates;
      draftProbabilities[4] = lastProbabilities;

      std::vector<uint16_t> y1A(width);
      std::memcpy(y1A.data(), y, width * 2);

      // Guess 2A step -> produces Guess 3A (ids[0])
      std::vector<uint16_t> y2A(width);
      const bool fits2 = (anchorPosition + 1 + kRows + kv::kPageTokens - 1) /
                             kv::kPageTokens <= mtp.pageTable.size();
      if (fits2) {
        std::memcpy(hIn, y1A.data(), width * 2);
        drafts[2] = step(anchorPosition + 1, 1, &drafts[1]);
        std::memcpy(y2A.data(), y, width * 2);
      } else {
        std::memcpy(hIn, yAnchor.data(), width * 2);
        std::memcpy(hIn + uint64_t{1} * width, y1A.data(), width * 2);
        std::array<uint32_t, 2> branch2{drafts[0], drafts[1]};
        drafts[2] = step(anchorPosition, 2, branch2.data());
        std::memcpy(y2A.data(), y + uint64_t{1} * width, width * 2);
      }
      draftTrace.chosen[2] = drafts[2];
      draftCandidates[2] = lastCandidates;
      draftProbabilities[2] = lastProbabilities;

      // Guess 3A step -> produces Guess 4A (ids[0])
      const bool fits3 = (anchorPosition + 2 + kRows + kv::kPageTokens - 1) /
                             kv::kPageTokens <= mtp.pageTable.size();
      if (fits3) {
        std::memcpy(hIn, y2A.data(), width * 2);
        drafts[3] = step(anchorPosition + 2, 1, &drafts[2]);
      } else {
        std::memcpy(hIn, yAnchor.data(), width * 2);
        std::memcpy(hIn + uint64_t{1} * width, y1A.data(), width * 2);
        std::memcpy(hIn + uint64_t{2} * width, y2A.data(), width * 2);
        std::array<uint32_t, 3> branch3{drafts[0], drafts[1], drafts[2]};
        drafts[3] = step(anchorPosition, 3, branch3.data());
      }
      draftTrace.chosen[3] = drafts[3];
      draftCandidates[3] = lastCandidates;
      draftProbabilities[3] = lastProbabilities;

      // Guess 1B step -> produces Guess 2C (ids[0])
      std::memcpy(hIn, yAnchor.data(), width * 2);
      drafts[6] = step(anchorPosition, 1, &drafts[5]);
      draftTrace.chosen[6] = drafts[6];
      draftCandidates[6] = lastCandidates;
      draftProbabilities[6] = lastProbabilities;

      drafted = 7;
      return drafts;
    }

    float chainConfidence = 1.0f;
    drafted = 0;
    drafts[0] = step(mtp.firstPosition, mtp.rows, mtp.tokens.data());
    draftTrace.chosen[0] = drafts[0];
    draftCandidates[0] = lastCandidates;
    draftProbabilities[0] = lastProbabilities;
    chainConfidence *= lastConfidence;
    if (lastConfidence < confident || chainConfidence < chainConfident)
      return drafts;
    drafted = 1;
    // Steps 2 and 3 chain on the head's own residual. Each re-runs the rows
    // before it (same inputs, same keys) so its attention sees them.
    const uint64_t anchorPosition = mtp.firstPosition + mtp.rows;
    std::vector<uint16_t> chain(maxDrafts * uint64_t{width});
    const auto *y = static_cast<const uint16_t *>(Y.contents());
    std::memcpy(chain.data(), y + uint64_t{mtp.rows - 1} * width, width * 2);
    // Each guess runs one new row: the attention step stores every row's
    // keys and values at its position, so the rows guessed before are read
    // from the cache. (Re-running them, SPLASH_MTP_RERUN, costs each earlier
    // row's experts again.) The next step's first guess overwrites these
    // positions with the accepted tokens' real rows.
    static const bool rerun = std::getenv("SPLASH_MTP_RERUN") != nullptr;
    for (uint32_t k = 1; k < maxDrafts; ++k) {
      if (chainConfidence * earlyExitFactor < chainConfident)
        break;
      // One new row writes an 8-row window from its own position; when that
      // runs past the pages the request holds (a decode step reserves its
      // verify window only), re-run from the anchor, which stays inside it.
      const bool fits = (anchorPosition + k - 1 + kRows + kv::kPageTokens - 1) /
                            kv::kPageTokens <= mtp.pageTable.size();
      const bool rerunThis = rerun || !fits;
      if (rerunThis) {
        for (uint32_t r = 0; r < k; ++r)
          std::memcpy(hIn + uint64_t{r} * width, chain.data() + uint64_t{r} * width, width * 2);
        drafts[k] = step(anchorPosition, k, drafts.data());
        draftTrace.chosen[k] = drafts[k];
      } else {
        std::memcpy(hIn, chain.data() + uint64_t{k - 1} * width, width * 2);
        drafts[k] = step(anchorPosition + k - 1, 1, drafts.data() + (k - 1));
        draftTrace.chosen[k] = drafts[k];
      }
      draftCandidates[k] = lastCandidates;
      draftProbabilities[k] = lastProbabilities;
      chainConfidence *= lastConfidence;
      if (lastConfidence < confident || chainConfidence < chainConfident)
        break;
      drafted = k + 1;
      std::memcpy(chain.data() + uint64_t{k} * width,
                  y + uint64_t{rerunThis ? k - 1 : 0} * width, width * 2);
    }
    return drafts;
  };

  // Finish the previous step's trace line: this step's MTP inputs are the
  // rows the target kept then (guesses kept + 1) and the tokens after them,
  // the last being the target's own next token.
  static std::string pendingTrace;
  static uint64_t pendingAnchor = UINT64_MAX;
  if (traceFile && !pendingTrace.empty()) {
    const QwenMtpLane &lane = buffers.mtp[0];
    std::string tail;
    if (lanes == 1 && lane.rows && lane.firstPosition == pendingAnchor)
      tail = ",\"accepted\":" + std::to_string(lane.rows - 1) +
             ",\"next\":" + std::to_string(lane.tokens[lane.rows - 1]);
    std::fprintf(traceFile, "%s%s}\n", pendingTrace.c_str(), tail.c_str());
    std::fflush(traceFile);
    pendingTrace.clear();
  }
  double mtpMs = 0.0;
  if (buffers.mtpDrafted) {
    buffers.liveRowsPerLane = 1 + *buffers.mtpProposedOut;
    setLiveRows();
  } else if (buffers.mtpEnabled && weights.mtpLayer && lanes == 1 && buffers.mtp[0].rows) {
    static const bool pldEnabled = [] {
      const char *v = std::getenv("SPLASH_PROMPT_LOOKUP");
      return v != nullptr && std::atoi(v) != 0;
    }();
    static const uint32_t pldMinMatch = [] {
      const char *v = std::getenv("SPLASH_PLD_MIN_MATCH");
      return v ? static_cast<uint32_t>(std::atoi(v)) : 4u;
    }();
    static const bool pldUnambiguous = [] {
      const char *v = std::getenv("SPLASH_PLD_UNAMBIGUOUS");
      return v ? (std::atoi(v) != 0) : true;
    }();
    static const uint32_t pldMaxDrafts = [] {
      const char *v = std::getenv("SPLASH_PLD_MAX_DRAFTS");
      return v ? static_cast<uint32_t>(std::atoi(v)) : 4u;
    }();
    static const bool adaptiveMode = [] {
      const char *v = std::getenv("SPLASH_ADAPTIVE_MODE");
      return v ? (std::atoi(v) != 0) : true;
    }();
    const bool skipPld = adaptiveMode && buffers.inThinkingPhase;
    std::array<uint32_t, 7> drafts{};
    const uint32_t anchor = buffers.mtp[0].tokens[buffers.mtp[0].rows - 1];

    if (pldEnabled && !skipPld && buffers.promptLookup) {
      std::array<uint32_t, 1> queryTokens{anchor};
      std::array<uint32_t, ExecutionLimits::draftProposalTokens> pldDrafts{};
      const uint32_t needed = std::min(maxDrafts, pldMaxDrafts);
      const uint32_t pldFound = buffers.promptLookup->propose(queryTokens, pldDrafts, needed, pldMinMatch, pldUnambiguous);
      if (pldFound > 0) {
        for (uint32_t j = 0; j < pldFound; ++j) {
          drafts[j] = pldDrafts[j];
          draftCandidates[j][0] = pldDrafts[j];
          draftProbabilities[j][0] = 1.0f;
          for (uint32_t c = 1; c < 16; ++c) {
            draftCandidates[j][c] = UINT32_MAX;
            draftProbabilities[j][c] = 0.0f;
          }
        }
        drafted = pldFound;
      }
    }

    if (drafted == 0) {
      const auto mtpStart = AwakeClock::now();
      drafts = runMtpDraft();
      mtpMs = std::chrono::duration<double, std::milli>(
                  AwakeClock::now() - mtpStart).count();
    }
    if (!buffers.mtpShadow) {
      adaptiveController.lastProposed = drafted;
      buffers.liveRowsPerLane = 1 + drafted;
      setLiveRows();
      if (buffers.mtpProposedOut)
        *buffers.mtpProposedOut = drafted;
      // The verify rows are the anchor then these proposals. Past the third
      // the proposals repeat it: the retained-row cap keeps them from counting.
      auto *proposed = static_cast<uint32_t *>(buffers.proposedTokens.contents());
      auto *candidates = static_cast<uint32_t *>(buffers.proposalCandidates.contents());
      auto *probabilities = static_cast<float *>(buffers.proposalProbabilities.contents());
      for (uint32_t k = 0; k < ExecutionLimits::draftProposalTokens; ++k) {
        const uint32_t source = std::min<uint32_t>(k, maxDrafts - 1);
        proposed[k] = drafts[source];
        // The distribution each guess was drawn from (one point if greedy).
        for (uint32_t c = 0; c < 16; ++c) {
          candidates[k * 16 + c] = draftCandidates[source][c];
          probabilities[k * 16 + c] = draftProbabilities[source][c];
        }
      }
    }
    // Score each guess when the token it guessed is decided. The anchor of
    // this step is the target's token at position firstPosition + rows.
    static std::map<uint64_t, std::array<uint32_t, 3>> guesses;
    static std::array<uint64_t, 3> hits{}, total{};
    const QwenMtpLane &mtp = buffers.mtp[0];
    const uint64_t anchorPosition = mtp.firstPosition + mtp.rows;
    if (auto found = guesses.find(anchorPosition); found != guesses.end()) {
      for (uint32_t k = 0; k < 3; ++k)
        if (found->second[k] != UINT32_MAX) {
          ++total[k];
          hits[k] += found->second[k] == anchor;
        }
      guesses.erase(guesses.begin(), std::next(found));
    }
    for (uint32_t k = 0; k < 3; ++k) {
      auto &slot = guesses[anchorPosition + 1 + k];
      if (slot[0] == 0 && slot[1] == 0 && slot[2] == 0) slot = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
      slot[k] = drafts[k];
    }
    static const bool shadowLog = std::getenv("SPLASH_STEP_TIMING") != nullptr;
    if (shadowLog && total[0] && total[0] % 16 == 0)
      std::cerr << "[MTP shadow] guess 1: " << hits[0] << "/" << total[0]
                << "  guess 2: " << hits[1] << "/" << total[1]
                << "  guess 3: " << hits[2] << "/" << total[2] << "\n";
  }

  if (buffers.mtpDraftOnly)
    return;

  metal::CommandGraph residentGraph = std::move(graph);
  for (uint32_t layerIndex = 0; layerIndex < R; ++layerIndex) {
    encodeAttentionHC(residentGraph, layerIndex);
    metal::MetalBuffer mixerOut = encodeMixer(residentGraph, layerIndex, gdnIndex, attentionIndex);
    encodeMixerUpdate(residentGraph, layerIndex, mixerOut);
    encodeMlpHC(residentGraph, layerIndex);
    ops::MoE::add(
        residentGraph,
        {buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
         buffers.selectedExperts, buffers.routingWeights,
         buffers.tileDescriptors, buffers.tileCount, buffers.groupedRoutes,
         buffers.routeRows, buffers.groupedInput, buffers.expertIntermediate,
         buffers.expertOutput},
        weights.layers[layerIndex].ffn, moePlan, /*addResidual=*/false);
    encodeMlpUpdate(residentGraph, layerIndex);
  }

  // Layer R base up to router
  encodeAttentionHC(residentGraph, R);
  metal::MetalBuffer mixerOutR = encodeMixer(residentGraph, R, gdnIndex, attentionIndex);
  encodeMixerUpdate(residentGraph, R, mixerOutR);
  encodeMlpHC(residentGraph, R);
  encodeMoERoute(residentGraph, R);

  // Submit residentGraph and prefetch streaming experts in background
  auto t0 = AwakeClock::now();
  metal::CommandTicket residentTicket = backend.submitCommandAsync(residentGraph.dispatches());
  // No blanket read-ahead of the previous step's experts: after a long
  // prompt that is nearly every expert, ~68 GB the OS then reads in the
  // background for minutes, stalling decode. Misses are fetched on demand.
  (void)residentTicket.wait();
  auto t1 = AwakeClock::now();
  double residentMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  double totalStageMs = 0.0;
  double totalGpuMs = 0.0;
  double totalPureGpuMs = 0.0;

  const bool pipelined = !backend.dispatchProfiling() &&
                         !std::getenv("SPLASH_NO_PIPELINE") &&
                         weights.layers[R].expertCache.capacity >=
                             rows * weights.layout.expertsPerToken;
  if (pipelined) {
    const bool lookahead = !std::getenv("SPLASH_NO_LOOKAHEAD");
    // Every layer is encoded and committed up front. Stage k runs layer
    // R+k-1's experts and layer R+k up to its router, then raises the
    // pipeline event; the host stages layer R+k's experts (and any deferred
    // work) and signals the GPU on. Nothing waits for a submission; the GPU
    // waits only for experts that were actually missing.
    std::vector<metal::ComputeDispatch> all;
    std::vector<size_t> starts;
    std::vector<std::vector<std::function<void()>>> deferredPerStage;
    // Dispatches point into their graph's parameter storage, so every stage
    // graph must outlive the submission.
    std::deque<metal::CommandGraph> stageGraphs;
    auto append = [&](metal::CommandGraph &g) {
      starts.push_back(all.size());
      for (auto &d : g.dispatches()) all.push_back(d);
    };
    // Stage 0 has already been submitted and completed above (resident
    // graph), so stages here start at layer R's experts.
    // Waves: every layer after the first is two stages. A runs the experts
    // already cached while the host reads the missing ones; B runs those,
    // then the rest of the layer as before. A is quiet (raises no event), so
    // the host starts B as soon as its reads land. Same outputs; 84.6 -> 79.6
    // ms a step on the 10-prompt suite. SPLASH_DECODE_WAVES=0 turns it off.
    static const bool waves = [] {
      const char *value = std::getenv("SPLASH_DECODE_WAVES");
      return !value || std::atoi(value) != 0;
    }();
    // One flag per stage (at most two per layer); std::vector<bool> is packed
    // bits and cannot be viewed as bools.
    std::array<bool, 2 * 64> quiet{};
    size_t quietCount = 0;
    if (waves && !weights.decodeRanges)
      weights.decodeRanges = backend.allocateBuffer(
          512, metal::BufferStorage::Shared, "qwen4exp-decode-ranges");
    const ops::MoeBuffers moeBuffers{
        buffers.hyperMixed, buffers.hyperMixed, buffers.gdnOutput,
        buffers.selectedExperts, buffers.routingWeights, buffers.tileDescriptors,
        buffers.tileCount, buffers.groupedRoutes, buffers.routeRows,
        buffers.groupedInput, buffers.expertIntermediate, buffers.expertOutput};
    const auto encodeStart = AwakeClock::now();
    for (uint32_t L = R; L < geometry.layers - 1; ++L) {
      deferredPerStage.emplace_back();
      pipelineDeferred = &deferredPerStage.back();
      if (waves && L > R) {
        metal::CommandGraph &cached = stageGraphs.emplace_back();
        ops::MoE::addGather(cached, moeBuffers, moePlan);
        ops::MoE::addExpertTilesInRange(cached, moeBuffers, makeCacheWeights(L), moePlan,
                                        backend.view(weights.decodeRanges, 0, 8));
        append(cached);
        quiet[quietCount++] = true;
      }
      metal::CommandGraph &stepGraph = stageGraphs.emplace_back();
      if (waves && L > R) {
        ops::MoE::addExpertTilesInRange(stepGraph, moeBuffers, makeCacheWeights(L), moePlan,
                                        backend.view(weights.decodeRanges, 256, 8));
        ops::MoE::addCombine(stepGraph, moeBuffers, moePlan, /*addResidual=*/false);
      } else {
        encodeMoEExecute(stepGraph, makeCacheWeights(L), /*hostGrouped=*/true);
      }
      encodeMlpUpdate(stepGraph, L);
      encodeAttentionHC(stepGraph, L + 1, /*priorWorkComplete=*/true);
      metal::MetalBuffer mixerOutNext = encodeMixer(stepGraph, L + 1, gdnIndex, attentionIndex);
      encodeMixerUpdate(stepGraph, L + 1, mixerOutNext);
      encodeMlpHC(stepGraph, L + 1);
      encodeRouteScores(stepGraph, L + 1, buffers.groupedInput);
      if (lookahead && L + 2 < geometry.layers)
        encodePredictRoute(stepGraph, L + 2);
      append(stepGraph);
      quiet[quietCount++] = false;
      pipelineDeferred = nullptr;
    }
    // The first stage's host work (layer R's experts) happens before commit:
    // its router result is already in memory. Shift so stage 0 needs none.
    auto ts0 = AwakeClock::now();
    encodeMs = std::chrono::duration<double, std::milli>(ts0 - encodeStart).count();
    if (!stageActiveExperts(R))
      throw std::logic_error("pipelined decode found more experts than the cache holds");
    hostGroup();
    for (auto &work : deferredPerStage[0]) work();
    totalStageMs += std::chrono::duration<double, std::milli>(
                        AwakeClock::now() - ts0).count();
    const uint32_t stages = static_cast<uint32_t>(starts.size());
    const uint64_t base = backend.reservePipelineEvents(stages);
    auto tg0 = AwakeClock::now();
    metal::CommandTicket ticket = backend.submitPipelineAsync(
        all, starts, base, waves ? std::span<const bool>(quiet.data(), quietCount) : std::span<const bool>{});
    static uint32_t tracedSteps = 0;
    const bool trace = std::getenv("SPLASH_TRACE_STAGES") && tracedSteps++ < 3;
    std::vector<double> waits;
    const uint32_t layerStages = geometry.layers - 1 - R;
    for (uint32_t k = 1; k < layerStages; ++k) {
      // Stage indices: layer R+k's A stage (waves) and its main (B) stage.
      const uint32_t mainStage = waves ? 2 * k : k;
      const uint32_t cachedStage = mainStage - 1;
      const uint32_t previous = waves ? (k == 1 ? 0 : 2 * (k - 1)) : k - 1;
      const auto waitStart = AwakeClock::now();
      if (!backend.waitPipelineEvent(base + 2 * previous + 1, 60000))
        throw std::runtime_error("pipelined decode stage timed out");
      if (trace)
        waits.push_back(std::chrono::duration<double, std::milli>(
                            AwakeClock::now() - waitStart).count());
      auto ts = AwakeClock::now();
      const uint32_t layer = R + k;
      const auto hostStart = ts;
      hostSelect(layer);
      if (!stageActiveExperts(layer, /*deferReads=*/waves))
        throw std::logic_error("pipelined decode found more experts than the cache holds");
      hostGroup(waves);
      for (auto &work : deferredPerStage[k]) work();
      if (waves) {
        // Cached experts go now; the missing ones follow once read.
        backend.signalPipelineEvent(base + 2 * cachedStage);
        if (!pendingMisses.empty()) {
          const auto &src = layerRef(pendingLayer);
          auto &cache = src.expertCache;
          const auto readStart = AwakeClock::now();
          readMissedExperts(src.expertSource, pendingMisses.data(), pendingMisses.size(),
                            src.ffn.expertGate.expertStrideBytes,
                            static_cast<char *>(cache.cacheGate.contents()),
                            static_cast<char *>(cache.cacheUp.contents()),
                            static_cast<char *>(cache.cacheDown.contents()));
          hostParts[1] += std::chrono::duration<double, std::milli>(
                              AwakeClock::now() - readStart).count();
        }
      }
      // Stage k-1 also predicted layer + 1; start those reads now so they
      // land while stage k runs.
      if (lookahead)
        prefetchPredicted(layer + 1);
      totalStageMs += std::chrono::duration<double, std::milli>(
                          AwakeClock::now() - ts).count();
      hostParts[2] += std::chrono::duration<double, std::milli>(
                          AwakeClock::now() - hostStart).count();
      backend.signalPipelineEvent(base + 2 * mainStage);
    }
    metal::CommandTiming timing = ticket.wait();
    totalGpuMs += std::chrono::duration<double, std::milli>(
                      AwakeClock::now() - tg0).count();
    totalPureGpuMs += timing.gpuSeconds * 1000.0;
    if (trace) {
      std::cerr << "[stage waits ms]";
      for (double w : waits) std::cerr << ' ' << std::lround(w * 10) / 10.0;
      std::cerr << '\n';
    }
  } else {
    // Loop through streaming layers R .. geometry.layers - 2
    for (uint32_t L = R; L < geometry.layers - 1; ++L) {
      auto ts0 = AwakeClock::now();
      bool staged = stageActiveExperts(L);
      auto ts1 = AwakeClock::now();
      totalStageMs += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

      ops::MoeWeights cacheW = staged ? makeCacheWeights(L) : weights.layers[L].ffn;

      metal::CommandGraph stepGraph;
      encodeMoEExecute(stepGraph, cacheW);
      encodeMlpUpdate(stepGraph, L);

      // Layer L + 1 base up to router
      encodeAttentionHC(stepGraph, L + 1, /*priorWorkComplete=*/true);
      metal::MetalBuffer mixerOutNext = encodeMixer(stepGraph, L + 1, gdnIndex, attentionIndex);
      encodeMixerUpdate(stepGraph, L + 1, mixerOutNext);
      encodeMlpHC(stepGraph, L + 1);
      encodeMoERoute(stepGraph, L + 1);

      auto tg0 = AwakeClock::now();
      metal::CommandTiming timing = backend.submitCommandAsync(stepGraph.dispatches()).wait();
      auto tg1 = AwakeClock::now();
      totalGpuMs += std::chrono::duration<double, std::milli>(tg1 - tg0).count();
      totalPureGpuMs += timing.gpuSeconds * 1000.0;
    }
  }

  const uint32_t lastL = geometry.layers - 1;
  auto ts0 = AwakeClock::now();
  // The pipeline's last stage stopped at the last layer's router scores.
  if (pipelined)
    hostSelect(lastL);
  bool stagedLast = stageActiveExperts(lastL);
  if (pipelined)
    hostGroup();
  auto ts1 = AwakeClock::now();
  totalStageMs += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

  ops::MoeWeights cacheWLast = stagedLast ? makeCacheWeights(lastL) : weights.layers[lastL].ffn;

  if (gdnIndex != geometry.stateLayout.layers ||
      attentionIndex + geometry.extraKvLayers != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  // Encode final layer's MoE, MLP update and head into the caller's graph
  encodeMoEExecute(graph, cacheWLast, /*hostGrouped=*/pipelined);
  encodeMlpUpdate(graph, lastL);
  encodeHead(graph);

  if (traceFile && lanes == 1) {
    const QwenMtpLane &lane = buffers.mtp[0];
    std::ostringstream line;
    line.precision(5);
    auto list = [&](auto first, uint32_t count) {
      line << '[';
      for (uint32_t i = 0; i < count; ++i) line << (i ? "," : "") << first[i];
      line << ']';
    };
    line << "{\"anchor_pos\":" << lane.firstPosition + lane.rows
         << ",\"temp\":" << lane.temperature
         << ",\"live\":" << buffers.liveRowsPerLane << ",\"drafted\":" << drafted
         << ",\"chosen\":";
    list(draftTrace.chosen.data(), draftTrace.depths);
    line << ",\"cand\":[";
    for (uint32_t d = 0; d < draftTrace.depths; ++d) {
      line << (d ? "," : "");
      list(draftTrace.ids[d].data(), 4);
    }
    line << "],\"logit\":[";
    for (uint32_t d = 0; d < draftTrace.depths; ++d) {
      line << (d ? "," : "");
      list(draftTrace.logits[d].data(), 4);
    }
    line << "],\"conf\":";
    list(draftTrace.confidence.data(), draftTrace.depths);
    line << ",\"misses\":" << totalMisses << ",\"ms\":{\"verify\":"
         << std::chrono::duration<double, std::milli>(
                AwakeClock::now() - verifyEnter).count()
         << ",\"mtp\":" << mtpMs << ",\"mtp_gpu_a\":" << mtpParts[0]
         << ",\"mtp_stage\":" << mtpParts[1] << ",\"mtp_gpu_b\":" << mtpParts[2]
         << ",\"staging\":" << totalStageMs << ",\"gpu_wall\":" << totalGpuMs
         << ",\"pure_gpu\":" << totalPureGpuMs << ",\"prefetch_wait\":" << hostParts[0]
         << ",\"miss_reads\":" << hostParts[1] << ",\"host_stage\":" << hostParts[2]
         << ",\"encode\":" << encodeMs << "}";
    pendingTrace = line.str();
    pendingAnchor = lane.firstPosition + lane.rows;
  }
  static const bool everyStep = std::getenv("SPLASH_STEP_TIMING") != nullptr;
  if (everyStep) {
    std::cerr << "[Verify Timing] Resident 0.." << R << ": " << residentMs << " ms | Staging: "
              << totalStageMs << " ms (misses: " << totalMisses << ") | GPU Wall: " << totalGpuMs << " ms (pure GPU: " << totalPureGpuMs << " ms) | MTP: " << mtpMs << " ms"
              << " (gpu A " << mtpParts[0] << ", stage " << mtpParts[1] << ", gpu B "
              << mtpParts[2] << ", pick " << mtpParts[3] << ")"
              << " | host: prefetch wait " << hostParts[0] << ", miss reads "
              << hostParts[1] << ", per-stage total " << hostParts[2]
              << ", predicted " << weights.predictIssued << " | prediction: used "
              << predictionCounts[0] << " in top-10 " << predictionCounts[1] << " in top-16 "
              << predictionCounts[2] << " misses " << predictionCounts[3]
              << " of which in top-16 " << predictionCounts[4] << " | encode " << encodeMs
              << " | inside verify " << std::chrono::duration<double, std::milli>(
                     AwakeClock::now() - verifyEnter).count() << "\n";
  }
}

} // namespace splash::model
