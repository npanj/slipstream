#pragma once

#include "models/qwen38/Qwen3_8.hpp"
#include "model/QwenTarget.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"

#include <span>
#include <variant>

namespace splash::model {

class Qwen3_8Target final {
public:
  Qwen3_8Target(const Qwen3_8Weights &weights,
                const QwenTargetGeometry &geometry,
                metal::MetalBackend &backend,
                const ops::ExecutionPlans &operators);
  Qwen3_8Target(const Qwen3_8Q8Weights &weights,
                const QwenTargetGeometry &geometry,
                metal::MetalBackend &backend,
                const ops::ExecutionPlans &operators);

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

private:
  using WeightView =
      std::variant<const Qwen3_8Weights *, const Qwen3_8Q8Weights *>;

  template <class Weights>
  void addPrefillImpl(
      const Weights &weights, metal::CommandGraph &graph,
      QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const kv::Q8LayerStorage> kvLayers) const;

  template <class Weights>
  void addVerifyImpl(
      const Weights &weights, metal::CommandGraph &graph,
      QwenTargetVerifyBuffers buffers,
      std::span<const kv::Q8LayerStorage> kvLayers,
      std::span<const kv::Q8ChunkedPrefillParams> q8,
      std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
      ops::Q4DispatchStats &stats) const;

  WeightView weights_;
  const QwenTargetGeometry &geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
};

} // namespace splash::model
