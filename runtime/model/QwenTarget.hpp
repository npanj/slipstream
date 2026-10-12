#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PromptLookup.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace splash::model {

struct Qwen4ExpWeights;
struct Qwen3_8Weights;
struct Qwen3_8Q8Weights;
class Qwen3_8Target;

enum class QwenFfnKind : uint8_t { Dense, SparseMoe };

// Both supported targets bind the same mixer tensors per hybrid layer; only
// the FFN differs between them.
// A package stores each projection as 4-bit or 8-bit; qwen4exp stores its
// mixer projections 8-bit (the `...Q8` fields) and leaves the 4-bit ones
// empty. Every other target is 4-bit throughout.
struct QwenGdnWeights final {
  ops::Q4Projection inputProjection;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer decay;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer mixerNorm;
  ops::Q4Projection outputProjection;
  ops::Q8Projection inputProjectionQ8;
  ops::Q8Projection outputProjectionQ8;
};

struct QwenAttentionWeights final {
  ops::Q4Projection inputProjection;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Q4Projection outputProjection;
  ops::Q8Projection inputProjectionQ8;
  ops::Q8Projection outputProjectionQ8;
};

using QwenMixerWeights = std::variant<QwenGdnWeights, QwenAttentionWeights>;

// Sizes of the mixer sections in a packed layer file.
struct QwenMixerGeometry final {
  uint32_t hiddenSize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedAttentionWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionHeadDimension = 0;
};

// Reads the mixer sections that follow a layer's input norm, in file order.
[[nodiscard]] QwenMixerWeights readQwenMixer(WeightFile &file,
                                             metal::MetalBackend &backend,
                                             const QwenMixerGeometry &geometry,
                                             bool fullAttention,
                                             bool eightBit = false);

inline constexpr std::string_view kEmbeddingMagic = "MDFE0001";
inline constexpr std::string_view kEmbeddingQ8Magic = "MDFE0008";

// Reads a packed target directory: one file per hybrid layer (input norm,
// mixer, post-attention norm, then the architecture's FFN through readFfn),
// head.bin and embedding.bin. Weights is the architecture's weight struct.
template <class Weights, class Layout, class ReadFfn>
[[nodiscard]] Weights
loadQwenTargetWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      const Layout &layout, std::string_view headMagic,
                      ReadFfn readFfn) {
  constexpr bool isQ8 = std::is_same_v<decltype(Weights{}.logitsProjection), ops::Q8Projection>;
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Weights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  const uint64_t hiddenBytes = checkedWeightMultiply(
      layout.hiddenSize, kBFloat16Bytes, "Qwen norm bytes");
  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    const std::string filename =
        "layer-" + std::to_string(layerIndex) + ".bin";
    WeightFile file(backend, directory / filename, "target/" + filename,
                    Layout::layerMagic, layerIndex, fullAttention ? 1U : 0U);
    auto &layer = result.layers.emplace_back();
    layer.inputNorm = file.section(hiddenBytes, "input-norm");
    layer.mixer =
        readQwenMixer(file, backend, layout.mixerGeometry(), fullAttention, isQ8);
    layer.postAttentionNorm =
        file.section(hiddenBytes, "post-attention-norm");
    readFfn(file, layer);
    file.finish();
    result.files.push_back(file.record());
  }

  {
    WeightFile file(backend, directory / "head.bin", "target/head.bin",
                    headMagic, layout.layers, 2);
    result.finalNorm = file.section(hiddenBytes, "final-norm");
    if constexpr (isQ8) {
      result.logitsProjection = readQ8Projection(
          file, backend, layout.vocabularySize, layout.hiddenSize, "logits");
    } else {
      result.logitsProjection = readQ4Projection(
          file, backend, layout.vocabularySize, layout.hiddenSize, "logits");
    }
    file.finish();
    result.files.push_back(file.record());
  }
  {
    constexpr std::string_view embeddingMagic = isQ8 ? kEmbeddingQ8Magic : kEmbeddingMagic;
    WeightFile file(backend, directory / "embedding.bin",
                    "target/embedding.bin", embeddingMagic,
                    layout.vocabularySize, layout.hiddenSize);
    if constexpr (isQ8) {
      result.tokenEmbedding = readQ8ProjectionComponents(
          file, layout.vocabularySize, layout.hiddenSize, "embedding");
    } else {
      result.tokenEmbedding = readQ4ProjectionComponents(
          file, layout.vocabularySize, layout.hiddenSize, "embedding");
    }
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

// Runtime-visible tensor geometry shared by the supported Qwen hybrid
// targets. It describes semantics only; operators remain responsible for
// choosing device-specific Metal pipelines and compute tiles.
struct QwenTargetGeometry final {
  uint32_t maximumContextTokens = 0;
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedAttentionWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionQueryHeads = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  uint32_t denseIntermediateSize = 0;
  ops::MoeShape moe{};
  QwenFfnKind ffnKind = QwenFfnKind::Dense;
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens{};
  kv::Q8Layout kvLayout{};
  GdnStateLayout stateLayout{};
  uint32_t hyperConnectionCount = 1;
  uint32_t hyperConnectionLowRank = 0;
  // qwen4exp's per-layer embedding, added to the residual entering pleLayer.
  // pleEmbeddingSize is zero for models without one.
  uint32_t pleLayer = 0;
  uint32_t pleEmbeddingSize = 0;
  uint32_t pleHistoryRows = 0;
  uint32_t pleEndToken = 0;
  // Key/value layers past the trunk's attention layers (the MTP head's).
  uint32_t extraKvLayers = 0;

  [[nodiscard]] constexpr bool hasPerLayerEmbedding() const noexcept {
    return pleEmbeddingSize != 0;
  }

  [[nodiscard]] constexpr uint32_t residualWidth() const noexcept {
    return hiddenSize * (hyperConnectionCount ? hyperConnectionCount : 1);
  }

  [[nodiscard]] constexpr uint32_t gdnKeyWidth() const noexcept {
    return gdnKeyHeads * gdnHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t ffnScratchWidth() const noexcept {
    return ffnKind == QwenFfnKind::Dense ? denseIntermediateSize
                                         : moe.expertIntermediateSize;
  }
  [[nodiscard]] constexpr ops::GdnShape gdnShape() const noexcept {
    return {gdnKeyHeads, gdnValueHeads, gdnHeadDimension,
            convolutionDimension, packedGdnWidth};
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return maximumContextTokens && layers && hiddenSize && vocabularySize &&
           packedGdnWidth && packedAttentionWidth && convolutionDimension &&
           gdnKeyHeads && gdnValueHeads && gdnHeadDimension &&
           attentionWidth && attentionQueryHeads && attentionKvHeads &&
           attentionHeadDimension && rotaryPairs && rotaryTheta > 0.0F &&
           kvLayout.valid() && stateLayout.valid() &&
           stateLayout.layers + kvLayout.attentionLayers - extraKvLayers ==
               layers &&
           gdnKeyWidth() * 2 + attentionWidth <= packedGdnWidth &&
           attentionWidth == attentionQueryHeads * attentionHeadDimension &&
           kvLayout.kvHeads == attentionKvHeads &&
           kvLayout.headDimension == attentionHeadDimension &&
           ((ffnKind == QwenFfnKind::Dense && denseIntermediateSize) ||
            (ffnKind == QwenFfnKind::SparseMoe && moe.valid()));
  }
};

// Scratch for qwen4exp's per-layer embedding, and the token ids it hashes.
// Empty for models without one.
struct QwenTargetPleBuffers final {
  metal::MetalBuffer tokens;     // uint32 per row
  metal::MetalBuffer shifted;    // uint32 [3][rows], per sequence
  metal::MetalBuffer embedding;  // bf16 rows x pleEmbeddingSize
  metal::MetalBuffer keys;       // bf16 rows x residual width
  metal::MetalBuffer values;     // bf16 rows x hidden
  metal::MetalBuffer gated;      // bf16 rows x residual width
  metal::MetalBuffer normalized; // bf16 (history + rows) x width, per sequence
};

struct QwenTargetPrefillSequence final {
  uint32_t rowBegin = 0;
  uint32_t rows = 0;
  uint32_t attentionStride = 0;
  uint64_t queryOffset = 0;
  uint64_t kvOffset = 0;
  kv::Q8ChunkedPrefillParams q8;
  metal::MetalBuffer pageTable;
  std::span<const metal::MetalBuffer> convolutionIn;
  std::span<const metal::MetalBuffer> convolutionOut;
  std::span<const metal::MetalBuffer> recurrentIn;
  std::span<const metal::MetalBuffer> recurrentOut;
  // The GDN cell's auxiliary region, current and next parity.
  metal::MetalBuffer auxiliaryIn;
  metal::MetalBuffer auxiliaryOut;
};

struct QwenTargetPrefillBuffers final {
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer gdnPacked;
  metal::MetalBuffer gdnQueries;
  metal::MetalBuffer gdnKeys;
  metal::MetalBuffer gdnValues;
  metal::MetalBuffer gdnDecay;
  metal::MetalBuffer gdnBeta;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseGateScratch;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer downProjectionSums;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
  metal::MetalBuffer hyperReduced;
  metal::MetalBuffer hyperInjection;
  metal::MetalBuffer hyperMixed;
  QwenTargetPleBuffers ple;
};

// What the MTP draft head consumes at the start of a decode cycle for one
// lane: rows of the previous step's final residual (decode Hidden0, lane
// local) at consecutive positions, each paired with the token that followed
// it. After a verify step that is its retained rows; after a prompt, the
// prompt's last row. The last row's token is the current anchor.
struct QwenMtpLane final {
  uint32_t rows = 0;
  uint32_t firstRow = 0;
  uint64_t firstPosition = 0;
  std::array<uint32_t, ExecutionLimits::targetVerifyRows> tokens{};
  std::span<const uint32_t> pageTable;
  // The request's sampling; temperature 0 is greedy.
  float temperature = 0.0f;
  float topP = 1.0f;
  uint32_t topK = 0;
};

struct QwenTargetVerifyBuffers final {
  // Rows per lane whose output can be kept. Below targetVerifyRows when the
  // draft is a placeholder: the other rows are computed only as far as the
  // cheap layers go, and their expert work is skipped.
  uint32_t liveRowsPerLane = ExecutionLimits::targetVerifyRows;
  // MTP drafting. `mtpShadow` runs the head and measures its guesses without
  // proposing them.
  bool mtpEnabled = false;
  bool mtpShadow = true;
  uint32_t kvPageCount = 0;
  std::array<QwenMtpLane, ExecutionLimits::maximumBatchWidth> mtp{};
  const ops::PromptLookup *promptLookup = nullptr;
  std::span<const uint32_t> draftMask;
  bool inThinkingPhase = false;
  metal::MetalBuffer proposedTokens;
  // Out: proposals the MTP head actually made for lane 0 (it stops when
  // unsure), so the runtime can cap the rows the verifier keeps.
  uint32_t *mtpProposedOut = nullptr;
  // A constrained request needs the proposals before the step, to build its
  // grammar masks. `mtpDraftOnly` runs just the head and returns; the step
  // then runs with `mtpDrafted`, reusing those proposals (and
  // *mtpProposedOut) instead of drafting again.
  bool mtpDraftOnly = false;
  bool mtpDrafted = false;
  // The draft's distribution per proposal as the acceptance reads it: 16
  // candidate ids and their probabilities.
  metal::MetalBuffer proposalCandidates;
  metal::MetalBuffer proposalProbabilities;
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer arrived;
  metal::MetalBuffer generation;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer denseGateScratch;
  std::span<const metal::MetalBuffer> gdnPacked;
  std::span<const metal::MetalBuffer> gdnMixed;
  std::span<const metal::MetalBuffer> gdnDecay;
  std::span<const metal::MetalBuffer> gdnBeta;
  std::span<const metal::MetalBuffer> chunkKeys;
  std::span<const metal::MetalBuffer> chunkValues;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      pageTables;
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
  metal::MetalBuffer hyperReduced;
  metal::MetalBuffer hyperInjection;
  metal::MetalBuffer hyperMixed;
  QwenTargetPleBuffers ple;
};

struct QwenTargetCommitBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextStates;
  metal::MetalBuffer retainedCounts;
  // The verify pass's per-layer embedding rows, for choosing what to keep.
  metal::MetalBuffer pleTokens;
  metal::MetalBuffer pleNormalized;
};

[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen4ExpWeights &weights);
[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen3_8Weights &weights);
[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen3_8Q8Weights &weights);

// Measurement only: the residual rows a model kept from the last prompt chunk
// (SPLASH_CAPTURE_LAYERS), one bf16 [rows x width] block per layer.
struct CapturedPrefillLayers {
  std::vector<uint32_t> layers;
  std::vector<const uint16_t *> data;
  uint32_t rows = 0;
  uint32_t width = 0;
};

// Builds the shared Qwen GDN/attention layer graph with the target's dense
// or sparse-MoE FFN. Architecture-specific loaders supply the package tensors.
class QwenTarget final {
public:
  QwenTarget(const Qwen4ExpWeights &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators);
  QwenTarget(const Qwen3_8Weights &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators);
  QwenTarget(const Qwen3_8Q8Weights &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators);
  ~QwenTarget();
  QwenTarget(QwenTarget &&) noexcept;
  QwenTarget(const QwenTarget &) = delete;
  QwenTarget &operator=(const QwenTarget &) = delete;

  [[nodiscard]] const QwenTargetGeometry &geometry() const noexcept {
    return geometry_;
  }

  void addPrefill(
      metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const kv::Q8LayerStorage> kvLayers) const;
  void addVerify(
      metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
      std::span<const kv::Q8LayerStorage> kvLayers,
      std::span<const kv::Q8ChunkedPrefillParams> q8,
      std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
      ops::Q4DispatchStats &stats) const;
  void addHead(metal::CommandGraph &graph, metal::MetalBuffer hidden,
               metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
               uint32_t normalizedRows) const;
  void addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    metal::MetalBuffer hidden, uint32_t rows) const;
  void addStateCommit(metal::CommandGraph &graph,
                      QwenTargetCommitBuffers buffers, uint32_t lanes) const;
  [[nodiscard]] CapturedPrefillLayers capturedPrefillLayers() const;

private:
  using WeightView =
      std::variant<const Qwen4ExpWeights *, const Qwen3_8Weights *,
                   const Qwen3_8Q8Weights *>;

  WeightView weights_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  metal::MetalBuffer embeddingScratch_;
  metal::MetalBuffer headNormalized_;
  metal::MetalBuffer headReduced_;
  std::unique_ptr<Qwen3_8Target> qwen38Target_;
};

} // namespace splash::model
