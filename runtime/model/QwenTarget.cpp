#include "model/QwenTarget.hpp"

#include "models/qwen4exp/Qwen4Exp.hpp"
#include "models/qwen4exp/Qwen4ExpTarget.hpp"
#include "models/qwen38/Qwen3_8.hpp"
#include "models/qwen38/Qwen3_8Target.hpp"

#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace splash::model {
namespace {

template <class Layout>
QwenTargetGeometry commonGeometry(const Layout &layout) {
  QwenTargetGeometry result;
  result.maximumContextTokens = layout.maximumContextTokens;
  result.layers = layout.layers;
  result.hiddenSize = layout.hiddenSize;
  result.vocabularySize = layout.vocabularySize;
  result.packedGdnWidth = layout.packedGdnWidth;
  result.packedAttentionWidth = layout.packedFullWidth;
  result.convolutionDimension = layout.convolutionDimension;
  result.attentionWidth = layout.attentionWidth;
  result.attentionQueryHeads = layout.attentionQueryHeads;
  result.attentionKvHeads = layout.attentionKvHeads;
  result.attentionHeadDimension = layout.attentionHeadDimension;
  result.rotaryPairs = layout.rotaryPairs;
  result.rotaryTheta = layout.rotaryTheta;
  result.gdnKeyHeads = layout.gdnKeyHeads;
  result.gdnValueHeads = layout.gdnValueHeads;
  result.gdnHeadDimension = layout.gdnHeadDimension;
  result.maskToken = layout.maskToken;
  result.stopTokens = layout.stopTokens;
  result.kvLayout = layout.q8Layout();
  result.stateLayout = layout.gdnStateLayout();
  return result;
}

QwenTargetGeometry geometryFor(const Qwen4ExpLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.moe = {layout.hiddenSize, layout.experts, layout.expertsPerToken,
                layout.expertIntermediateSize, layout.expertStorageN};
  result.ffnKind = QwenFfnKind::SparseMoe;
  result.hyperConnectionCount = layout.hyperConnectionCount;
  result.hyperConnectionLowRank = layout.hyperConnectionLowRank;
  result.pleLayer = layout.ngramLayer;
  result.pleEmbeddingSize = layout.ngramEmbeddingSize;
  result.pleHistoryRows = layout.pleConvolutionState();
  // The reference's end-of-sequence token, which both fills a fresh n-gram
  // history and restarts the window: config eos_token_id, the first stop.
  result.pleEndToken = layout.stopTokens[0];
  result.extraKvLayers = layout.mtpLayers;
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_8Layout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.ffnKind = QwenFfnKind::Dense;
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_8Q8Layout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.ffnKind = QwenFfnKind::Dense;
  return result;
}

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers + geometry.extraKvLayers !=
          geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

template <class Layer>
constexpr bool hasDenseFfn = requires(const Layer &layer) {
  layer.gateProjection;
  layer.upProjection;
  layer.downProjection;
};

template <class Mixer>
constexpr bool isGdnMixer =
    std::is_same_v<std::remove_cvref_t<Mixer>, QwenGdnWeights>;

} // namespace

QwenMixerWeights readQwenMixer(WeightFile &file, metal::MetalBackend &backend,
                               const QwenMixerGeometry &geometry,
                               bool fullAttention, bool eightBit) {
  constexpr uint64_t kFloat32Bytes = 4;
  auto projection = [&](auto &q4, auto &q8, uint32_t out, uint32_t in,
                        const char *label) {
    if (eightBit)
      q8 = readQ8Projection(file, backend, out, in, label);
    else
      q4 = readQ4Projection(file, backend, out, in, label);
  };
  if (fullAttention) {
    QwenAttentionWeights attention;
    projection(attention.inputProjection, attention.inputProjectionQ8,
               geometry.packedAttentionWidth, geometry.hiddenSize,
               "attention-input");
    const uint64_t headNormBytes = checkedWeightMultiply(
        geometry.attentionHeadDimension, kBFloat16Bytes, "head norm bytes");
    attention.queryNorm = file.section(headNormBytes, "query-norm");
    attention.keyNorm = file.section(headNormBytes, "key-norm");
    projection(attention.outputProjection, attention.outputProjectionQ8,
               geometry.hiddenSize, geometry.attentionWidth,
               "attention-output");
    return attention;
  }
  QwenGdnWeights gdn;
  projection(gdn.inputProjection, gdn.inputProjectionQ8, geometry.packedGdnWidth,
             geometry.hiddenSize, "gdn-input");
  gdn.convolutionWeights = file.section(
      checkedWeightMultiply(
          checkedWeightMultiply(geometry.convolutionDimension, kGdnConvolutionTaps,
                                "convolution elements"),
          kBFloat16Bytes, "convolution bytes"),
      "gdn-convolution");
  gdn.decay = file.section(checkedWeightMultiply(geometry.gdnValueHeads,
                                                 kFloat32Bytes,
                                                 "GDN decay bytes"),
                           "gdn-decay");
  gdn.timeBias = file.section(
      checkedWeightMultiply(geometry.gdnValueHeads, kBFloat16Bytes,
                            "GDN time bias bytes"),
      "gdn-time-bias");
  gdn.mixerNorm = file.section(
      checkedWeightMultiply(geometry.gdnHeadDimension, kBFloat16Bytes,
                            "GDN norm bytes"),
      "gdn-norm");
  projection(gdn.outputProjection, gdn.outputProjectionQ8, geometry.hiddenSize,
             geometry.attentionWidth, "gdn-output");
  return gdn;
}

QwenTarget::QwenTarget(const Qwen4ExpWeights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
  embeddingScratch_ = backend_.allocateBuffer(
      uint64_t{ExecutionLimits::prefillTokenBudget} * geometry_.hiddenSize * sizeof(uint16_t),
      metal::BufferStorage::Private, "qwen4exp-embedding-scratch");
  headNormalized_ = backend_.allocateBuffer(
      uint64_t{ExecutionLimits::targetVerifyRows} * geometry_.residualWidth() * sizeof(uint16_t),
      metal::BufferStorage::Private, "qwen4exp-head-normalized");
  headReduced_ = backend_.allocateBuffer(
      uint64_t{ExecutionLimits::targetVerifyRows} * geometry_.hyperConnectionLowRank * sizeof(uint16_t),
      metal::BufferStorage::Private, "qwen4exp-head-reduced");
}

QwenTarget::QwenTarget(const Qwen3_8Weights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators),
      qwen38Target_(std::make_unique<Qwen3_8Target>(weights, geometry_, backend, operators)) {
  requireWeights(weights, geometry_);
}

QwenTarget::QwenTarget(const Qwen3_8Q8Weights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), geometry_(qwenTargetGeometry(weights)),
      backend_(backend), operators_(operators),
      qwen38Target_(std::make_unique<Qwen3_8Target>(weights, geometry_, backend, operators)) {
  requireWeights(weights, geometry_);
}

QwenTarget::~QwenTarget() = default;
QwenTarget::QwenTarget(QwenTarget &&) noexcept = default;

QwenTargetGeometry qwenTargetGeometry(const Qwen4ExpWeights &weights) {
  return geometryFor(weights.layout);
}

QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &weights) {
  return geometryFor(weights.layout);
}

QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Q8Weights &weights) {
  return geometryFor(weights.layout);
}

void QwenTarget::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::Q8LayerStorage> kvLayers) const {
  if (qwen38Target_) {
    qwen38Target_->addPrefill(graph, std::move(buffers), sequences, rows, kvLayers);
    return;
  }
  Qwen4ExpTarget::addPrefill(*std::get<const Qwen4ExpWeights *>(weights_),
                             geometry_, backend_, operators_, graph,
                             std::move(buffers), sequences, rows, kvLayers);
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const kv::Q8LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  if (qwen38Target_) {
    qwen38Target_->addVerify(graph, std::move(buffers), kvLayers, q8, verify, lanes, stats);
    return;
  }
  Qwen4ExpTarget::addVerify(*std::get<const Qwen4ExpWeights *>(weights_),
                             geometry_, backend_, operators_, graph,
                             std::move(buffers), kvLayers, q8, verify, lanes,
                             stats);
}

void QwenTarget::addHead(metal::CommandGraph &graph,
                         metal::MetalBuffer hidden,
                         metal::MetalBuffer finalHidden,
                         metal::MetalBuffer logits,
                         uint32_t normalizedRows) const {
  if (!normalizedRows ||
      normalizedRows > ExecutionLimits::targetVerifyRows) {
    throw std::invalid_argument("invalid Qwen head row count");
  }
  if (qwen38Target_) {
    qwen38Target_->addHead(graph, std::move(hidden), std::move(finalHidden), std::move(logits), normalizedRows);
    return;
  }
  Qwen4ExpTarget::addHead(*std::get<const Qwen4ExpWeights *>(weights_),
                          geometry_, operators_, graph, std::move(hidden),
                          std::move(finalHidden), std::move(logits),
                          headNormalized_, headReduced_, normalizedRows);
}

CapturedPrefillLayers QwenTarget::capturedPrefillLayers() const {
  return Qwen4ExpTarget::capturedPrefillLayers(geometry_);
}

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  if (qwen38Target_) {
    qwen38Target_->addEmbedding(graph, std::move(tokens), std::move(hidden), rows);
    return;
  }
  if (std::holds_alternative<const Qwen4ExpWeights *>(weights_)) {
    const auto *exp = std::get<const Qwen4ExpWeights *>(weights_);
    Qwen4ExpTarget::addEmbedding(*exp, geometry_, graph, std::move(tokens),
                                 std::move(hidden), embeddingScratch_, rows);
    return;
  }
  std::visit(
      [&](const auto *weights) {
        ops::Embedding::add(graph, std::move(tokens), weights->tokenEmbedding,
                            std::move(hidden), rows);
      },
      weights_);
}

void QwenTarget::addStateCommit(metal::CommandGraph &graph,
                                QwenTargetCommitBuffers buffers,
                                uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  if (qwen38Target_) {
    qwen38Target_->addStateCommit(graph, std::move(buffers), lanes);
    return;
  }
  if (geometry_.hasPerLayerEmbedding())
    Qwen4ExpTarget::addPerLayerEmbeddingCommit(geometry_, backend_, graph,
                                               buffers, lanes);
  ops::GdnShape gdnShape = geometry_.gdnShape();
  const char *treeEnv = std::getenv("SPLASH_TREE_DRAFT");
  if (treeEnv != nullptr && std::atoi(treeEnv) != 0) {
    gdnShape.treeParents = 0x60132100u;
  }
  ops::GDN::addCommit(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      gdnShape, geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

} // namespace splash::model
