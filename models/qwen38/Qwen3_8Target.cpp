#include "models/qwen38/Qwen3_8Target.hpp"

#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace splash::model {
namespace {

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

template <class Mixer>
constexpr bool isGdnMixer =
    std::is_same_v<std::remove_cvref_t<Mixer>, QwenGdnWeights>;

} // namespace

Qwen3_8Target::Qwen3_8Target(const Qwen3_8Weights &weights,
                             const QwenTargetGeometry &geometry,
                             metal::MetalBackend &backend,
                             const ops::ExecutionPlans &operators)
    : weights_(&weights), geometry_(geometry), backend_(backend),
      operators_(operators) {
  requireWeights(weights, geometry_);
}

Qwen3_8Target::Qwen3_8Target(const Qwen3_8Q8Weights &weights,
                             const QwenTargetGeometry &geometry,
                             metal::MetalBackend &backend,
                             const ops::ExecutionPlans &operators)
    : weights_(&weights), geometry_(geometry), backend_(backend),
      operators_(operators) {
  requireWeights(weights, geometry_);
}

void Qwen3_8Target::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::Q8LayerStorage> kvLayers) const {
  std::visit(
      [&](const auto *weights) {
        addPrefillImpl(*weights, graph, std::move(buffers), sequences, rows,
                       kvLayers);
      },
      weights_);
}

template <class Weights>
void Qwen3_8Target::addPrefillImpl(
    const Weights &weights, metal::CommandGraph &graph,
    QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const kv::Q8LayerStorage> kvLayers) const {
  constexpr bool isQ8 = std::is_same_v<Weights, Qwen3_8Q8Weights>;
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }
  const ops::LinearMatrix gdnInput{geometry_.packedGdnWidth,
                                   geometry_.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry_.packedAttentionWidth,
                                         geometry_.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry_.hiddenSize,
                                      geometry_.attentionWidth};

  auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(uint16_t),
                         uint64_t{count} * width * sizeof(uint16_t));
  };
  auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                 uint32_t count, uint32_t width) {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(float),
                         uint64_t{count} * width * sizeof(float));
  };

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;
  for (uint32_t layerIndex = 0; layerIndex < geometry_.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    ops::Normalization::addRmsWithQ4Sums(
        graph, input, layer.inputNorm, buffers.normalized,
        buffers.projectionSums, geometry_.hiddenSize, rows);

    metal::MetalBuffer residual;
    std::visit(
        [&](const auto &mixer) {
          if constexpr (isGdnMixer<decltype(mixer)>) {
            if constexpr (isQ8) {
              operators_.linear().addPrefill(graph, buffers.normalized,
                             mixer.inputProjectionQ8, buffers.gdnPacked,
                             buffers.projectionSums, gdnInput, rows);
            } else {
              operators_.linear().addPrefill(graph, buffers.normalized,
                             mixer.inputProjection, buffers.gdnPacked,
                             buffers.projectionSums, gdnInput, rows);
            }
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              ops::GDN::addPrefill(
                  graph,
                  {u16(buffers.gdnPacked, sequence.rowBegin, sequence.rows,
                       geometry_.packedGdnWidth),
                   mixer.convolutionWeights, sequence.convolutionIn[gdnIndex],
                   sequence.convolutionOut[gdnIndex],
                   u16(buffers.gdnQueries, sequence.rowBegin, sequence.rows,
                       geometry_.gdnKeyWidth()),
                   u16(buffers.gdnKeys, sequence.rowBegin, sequence.rows,
                       geometry_.gdnKeyWidth()),
                   u16(buffers.gdnValues, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth),
                   mixer.decay, mixer.timeBias,
                   f32(buffers.gdnDecay, sequence.rowBegin, sequence.rows,
                       geometry_.gdnValueHeads),
                   u16(buffers.gdnBeta, sequence.rowBegin, sequence.rows,
                       geometry_.gdnValueHeads),
                   sequence.recurrentIn[gdnIndex],
                   sequence.recurrentOut[gdnIndex],
                   u16(buffers.recurrent, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth),
                   mixer.mixerNorm,
                   u16(buffers.gdnHidden, sequence.rowBegin, sequence.rows,
                       geometry_.attentionWidth)},
                  geometry_.gdnShape(), sequence.rows);
            }
            operators_.linear().addPrefillSums(graph, buffers.gdnHidden,
                               buffers.projectionSums, mixerOutput, rows);
            if constexpr (isQ8) {
              operators_.linear().addPrefillResidual(
                  graph, buffers.gdnHidden, mixer.outputProjectionQ8, input,
                  buffers.gdnOutput, buffers.projectionSums, mixerOutput,
                  rows);
            } else {
              operators_.linear().addPrefillResidual(
                  graph, buffers.gdnHidden, mixer.outputProjection, input,
                  buffers.gdnOutput, buffers.projectionSums, mixerOutput,
                  rows);
            }
            residual = buffers.gdnOutput;
            ++gdnIndex;
          } else {
            if constexpr (isQ8) {
              operators_.linear().addPrefill(graph, buffers.normalized,
                             mixer.inputProjectionQ8, buffers.fullPacked,
                             buffers.projectionSums, attentionInput, rows);
            } else {
              operators_.linear().addPrefill(graph, buffers.normalized,
                             mixer.inputProjection, buffers.fullPacked,
                             buffers.projectionSums, attentionInput, rows);
            }
            for (const QwenTargetPrefillSequence &sequence : sequences) {
              const uint64_t queryBytes =
                  uint64_t{geometry_.attentionQueryHeads} *
                  sequence.attentionStride * geometry_.attentionHeadDimension *
                  sizeof(uint16_t);
              const uint64_t kvBytes =
                  uint64_t{geometry_.attentionKvHeads} *
                  sequence.attentionStride * geometry_.attentionHeadDimension *
                  sizeof(uint16_t);
              metal::MetalBuffer queries = backend_.view(
                  buffers.fullQueries, sequence.queryOffset, queryBytes);
              metal::MetalBuffer attentionRows = backend_.view(
                  buffers.fullAttention, sequence.queryOffset, queryBytes);
              metal::MetalBuffer keys = backend_.view(
                  buffers.chunkKeys, sequence.kvOffset, kvBytes);
              metal::MetalBuffer values = backend_.view(
                  buffers.chunkValues, sequence.kvOffset, kvBytes);
              ops::PagedAttention::addPrefillProjection(
                  graph,
                  u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                      geometry_.packedAttentionWidth),
                  mixer.queryNorm, mixer.keyNorm,
                  f32(buffers.ropeCos, sequence.rowBegin, sequence.rows,
                      geometry_.rotaryPairs),
                  f32(buffers.ropeSin, sequence.rowBegin, sequence.rows,
                      geometry_.rotaryPairs),
                  queries, keys, values, sequence.rows,
                  sequence.attentionStride, sequence.attentionStride,
                  geometry_.attentionQueryHeads, geometry_.kvLayout);
              ops::PagedAttention::addPrefillStore(
                  graph, kvLayers[attentionIndex], keys, values,
                  sequence.pageTable, sequence.q8, geometry_.kvLayout);
              ops::PagedAttention::addPrefill(
                  graph, kvLayers[attentionIndex], queries, attentionRows,
                  buffers.attentionPartials, buffers.attentionStatistics,
                  sequence.pageTable, sequence.q8,
                  operators_.prefillAttention(
                      sequence.rows, geometry_.attentionQueryHeads,
                      geometry_.kvLayout, sequence.q8.committed_tokens));
              ops::PagedAttention::addPrefillGate(
                  graph,
                  u16(buffers.fullPacked, sequence.rowBegin, sequence.rows,
                      geometry_.packedAttentionWidth),
                  attentionRows,
                  u16(buffers.attentionHidden, sequence.rowBegin,
                      sequence.rows, geometry_.attentionWidth),
                  sequence.rows, sequence.attentionStride,
                  sequence.attentionStride, geometry_.attentionQueryHeads,
                  geometry_.kvLayout);
            }
            operators_.linear().addPrefillSums(graph, buffers.attentionHidden,
                               buffers.projectionSums, mixerOutput, rows);
            if constexpr (isQ8) {
              operators_.linear().addPrefillResidual(
                  graph, buffers.attentionHidden, mixer.outputProjectionQ8, input,
                  buffers.attentionOutput, buffers.projectionSums, mixerOutput,
                  rows);
            } else {
              operators_.linear().addPrefillResidual(
                  graph, buffers.attentionHidden, mixer.outputProjection, input,
                  buffers.attentionOutput, buffers.projectionSums, mixerOutput,
                  rows);
            }
            residual = buffers.attentionOutput;
            ++attentionIndex;
          }
        },
        layer.mixer);

    ops::Normalization::addRmsWithQ4Sums(
        graph, residual, layer.postAttentionNorm, buffers.normalized,
        buffers.projectionSums, geometry_.hiddenSize, rows);

    const ops::LinearMatrix up{geometry_.denseIntermediateSize,
                               geometry_.hiddenSize};
    const ops::LinearMatrix down{geometry_.hiddenSize,
                                 geometry_.denseIntermediateSize};
    operators_.linear().addPrefill(graph, buffers.normalized, layer.gateProjection,
                   buffers.denseGateScratch, buffers.projectionSums, up,
                   rows);
    operators_.linear().addPrefillUpWithGate(
        graph, buffers.normalized, layer.upProjection,
        buffers.denseGateScratch, buffers.denseIntermediate,
        buffers.projectionSums, buffers.downProjectionSums, up, rows);
    operators_.linear().addPrefillResidual(
        graph, buffers.denseIntermediate, layer.downProjection, residual,
        output, buffers.downProjectionSums, down, rows);
  }
  if (gdnIndex != geometry_.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }
}

void Qwen3_8Target::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const kv::Q8LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  std::visit(
      [&](const auto *weights) {
        addVerifyImpl(*weights, graph, std::move(buffers), kvLayers, q8,
                      verify, lanes, stats);
      },
      weights_);
}

template <class Weights>
void Qwen3_8Target::addVerifyImpl(
    const Weights &weights, metal::CommandGraph &graph,
    QwenTargetVerifyBuffers buffers,
    std::span<const kv::Q8LayerStorage> kvLayers,
    std::span<const kv::Q8ChunkedPrefillParams> q8,
    std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
    ops::Q4DispatchStats &stats) const {
  constexpr bool isQ8 = std::is_same_v<Weights, Qwen3_8Q8Weights>;
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      q8.size() != ExecutionLimits::maximumBatchWidth ||
      verify.size() != ExecutionLimits::maximumBatchWidth ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = verify[lane].committed_tokens;
  const auto attentionPlan = operators_.verifyAttention(
      lanes, geometry_.attentionQueryHeads, geometry_.kvLayout, histories);
  const ops::LinearMatrix gdnInput{geometry_.packedGdnWidth, geometry_.hiddenSize};
  const ops::LinearMatrix attentionInput{geometry_.packedAttentionWidth, geometry_.hiddenSize};
  const ops::LinearMatrix mixerOutput{geometry_.hiddenSize, geometry_.attentionWidth};
  constexpr uint32_t tileRows = kv::kPageTokens;

  uint32_t gdnIndex = 0;
  uint32_t attentionIndex = 0;
  for (uint32_t layerIndex = 0; layerIndex < geometry_.layers; ++layerIndex) {
    const auto &layer = weights.layers[layerIndex];
    metal::MetalBuffer input = buffers.hidden[layerIndex & 1];
    metal::MetalBuffer output = buffers.hidden[(layerIndex & 1) ^ 1];
    ops::Normalization::addRms(graph, input, layer.inputNorm,
                               buffers.normalized, geometry_.hiddenSize, rows);

    metal::MetalBuffer residual;
    std::visit(
        [&](const auto &mixer) {
          if constexpr (isGdnMixer<decltype(mixer)>) {
            if constexpr (isQ8) {
              operators_.linear().addDecodeBatch(graph,
                                 buffers.normalized, mixer.inputProjectionQ8,
                                 buffers.gdnPacked[gdnIndex], gdnInput, lanes,
                                 stats);
            } else {
              operators_.linear().addDecodeBatch(graph,
                                 buffers.normalized, mixer.inputProjection,
                                 buffers.gdnPacked[gdnIndex], gdnInput, lanes,
                                 stats);
            }
            ops::GDN::addDecode(
                graph,
                {buffers.gdnPacked[gdnIndex], mixer.convolutionWeights,
                 buffers.currentGdnStates, buffers.nextGdnStates,
                 buffers.gdnMixed[gdnIndex], mixer.decay, mixer.timeBias,
                 buffers.gdnDecay[gdnIndex], buffers.gdnBeta[gdnIndex],
                 buffers.recurrent, mixer.mixerNorm, buffers.gdnHidden,
                 buffers.arrived, buffers.generation},
                geometry_.gdnShape(), lanes, gdnIndex,
                {geometry_.stateLayout.convolutionLayerBytes(),
                 geometry_.stateLayout.recurrentLayerBytes(),
                 geometry_.stateLayout.convolutionBytes()});
            if constexpr (isQ8) {
              operators_.linear().addResidualBatch(
                  graph, buffers.gdnHidden,
                  mixer.outputProjectionQ8, input, buffers.gdnOutput, mixerOutput,
                  lanes, stats);
            } else {
              operators_.linear().addResidualBatch(
                  graph, buffers.gdnHidden,
                  mixer.outputProjection, input, buffers.gdnOutput, mixerOutput,
                  lanes, stats);
            }
            residual = buffers.gdnOutput;
            ++gdnIndex;
          } else {
            if constexpr (isQ8) {
              operators_.linear().addDecodeBatch(graph,
                                 buffers.normalized, mixer.inputProjectionQ8,
                                 buffers.fullPacked, attentionInput, lanes,
                                 stats);
            } else {
              operators_.linear().addDecodeBatch(graph,
                                 buffers.normalized, mixer.inputProjection,
                                 buffers.fullPacked, attentionInput, lanes,
                                 stats);
            }
            ops::PagedAttention::addVerifyProjection(
                graph, buffers.fullPacked, mixer.queryNorm, mixer.keyNorm,
                buffers.ropeCos, buffers.ropeSin, buffers.fullQueries,
                buffers.chunkKeys[attentionIndex],
                buffers.chunkValues[attentionIndex],
                ExecutionLimits::targetVerifyRows, tileRows, tileRows,
                geometry_.attentionQueryHeads, geometry_.kvLayout, lanes);
            ops::PagedAttention::addVerify(
                graph, kvLayers[attentionIndex],
                {buffers.chunkKeys[attentionIndex],
                 buffers.chunkValues[attentionIndex], buffers.fullQueries,
                 buffers.attentionPartials, buffers.attentionStatistics,
                 buffers.fullAttention, buffers.pageTables},
                q8, verify, attentionPlan);
            ops::PagedAttention::addVerifyGate(
                graph, buffers.fullPacked, buffers.fullAttention,
                buffers.attentionHidden, ExecutionLimits::targetVerifyRows,
                tileRows, tileRows, geometry_.attentionQueryHeads,
                geometry_.kvLayout, lanes);
            if constexpr (isQ8) {
              operators_.linear().addResidualBatch(
                  graph, buffers.attentionHidden,
                  mixer.outputProjectionQ8, input, buffers.attentionOutput,
                  mixerOutput, lanes, stats);
            } else {
              operators_.linear().addResidualBatch(
                  graph, buffers.attentionHidden,
                  mixer.outputProjection, input, buffers.attentionOutput,
                  mixerOutput, lanes, stats);
            }
            residual = buffers.attentionOutput;
            ++attentionIndex;
          }
        },
        layer.mixer);

    ops::Normalization::addRms(graph, residual, layer.postAttentionNorm,
                               buffers.normalized, geometry_.hiddenSize, rows);
    const ops::LinearMatrix up{geometry_.denseIntermediateSize, geometry_.hiddenSize};
    const ops::LinearMatrix down{geometry_.hiddenSize, geometry_.denseIntermediateSize};
    operators_.linear().addGateUpBatch(graph, buffers.normalized, layer.gateProjection,
                       layer.upProjection, buffers.denseGateScratch,
                       buffers.denseIntermediate, up, lanes, stats);
    operators_.linear().addResidualBatch(
        graph, buffers.denseIntermediate,
        layer.downProjection, residual, output, down, lanes, stats);
  }
  if (gdnIndex != geometry_.stateLayout.layers ||
      attentionIndex != kvLayers.size()) {
    throw std::logic_error("Qwen target layer partition mismatch");
  }

  ops::Normalization::addRms(graph, buffers.hidden[geometry_.layers & 1],
                             weights.finalNorm,
                             buffers.finalHidden, geometry_.hiddenSize, rows);
  const ops::LinearMatrix head{geometry_.vocabularySize, geometry_.hiddenSize};
  operators_.linear().addDecodeBatch(graph, buffers.finalHidden,
                     weights.logitsProjection, buffers.logits, head, lanes,
                     stats);
}

void Qwen3_8Target::addHead(metal::CommandGraph &graph,
                            metal::MetalBuffer hidden,
                            metal::MetalBuffer finalHidden,
                            metal::MetalBuffer logits,
                            uint32_t normalizedRows) const {
  if (!normalizedRows ||
      normalizedRows > ExecutionLimits::targetVerifyRows) {
    throw std::invalid_argument("invalid Qwen head row count");
  }
  std::visit([&](const auto *weights) {
    ops::Normalization::addRms(graph, std::move(hidden), weights->finalNorm,
                               finalHidden, geometry_.hiddenSize, normalizedRows);
    const ops::LinearMatrix head{geometry_.vocabularySize, geometry_.hiddenSize};
    operators_.linear().addDecode(graph, std::move(finalHidden),
                                  weights->logitsProjection, std::move(logits), head);
  }, weights_);
}

void Qwen3_8Target::addEmbedding(metal::CommandGraph &graph,
                                 metal::MetalBuffer tokens,
                                 metal::MetalBuffer hidden,
                                 uint32_t rows) const {
  std::visit([&](const auto *weights) {
    ops::Embedding::add(graph, std::move(tokens), weights->tokenEmbedding,
                        std::move(hidden), rows);
  }, weights_);
}

void Qwen3_8Target::addStateCommit(metal::CommandGraph &graph,
                                   QwenTargetCommitBuffers buffers,
                                   uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  ops::GDN::addCommit(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      geometry_.gdnShape(), geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

} // namespace splash::model
