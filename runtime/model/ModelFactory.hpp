#pragma once

#include "ModelDescriptor.hpp"
#include "model/DFlashDraft.hpp"
#include "model/QwenVision.hpp"
#include "models/qwen4exp/Qwen4Exp.hpp"
#include "ops/Q8PageStorage.hpp"
#include "ops/ExecutionPlans.hpp"

#include <filesystem>
#include <string>
#include <variant>

namespace splash::model {

// Flash-Next only: 27B support lives on the 27b branch. The single-variant
// form keeps every std::visit/holds_alternative call site compiling.
using TargetWeights = std::variant<Qwen4ExpWeights>;

struct ModelPackage final {
  ModelDescriptor descriptor;
  TargetWeights target;
  QwenVisionWeights vision;
  DFlashDraftWeights draft;
  std::string manifestFingerprintSha256;

  [[nodiscard]] const std::string &name() const noexcept {
    return descriptor.name;
  }
  [[nodiscard]] kv::Q8Layout targetKvLayout() const noexcept {
    return descriptor.targetKvLayout;
  }
  [[nodiscard]] CompositeStateLayout stateLayout() const noexcept {
    return descriptor.stateLayout;
  }
  [[nodiscard]] uint32_t maximumContextTokens() const noexcept {
    return descriptor.capabilities.maximumContextTokens;
  }
  [[nodiscard]] uint64_t targetActualAllocatedBytes() const noexcept {
    return std::visit([](const auto &weights) {
      return weights.actualAllocatedBytes;
    }, target);
  }
  [[nodiscard]] const std::string &targetManifestFingerprint() const noexcept {
    return std::visit([](const auto &weights) -> const std::string & {
      return weights.manifestFingerprintSha256;
    }, target);
  }
  // Zero for targets that are planned fully resident; the two shipped models
  // are, so nothing about their planning changes.
  [[nodiscard]] uint64_t streamableWeightBytes() const noexcept {
    return std::visit([](const auto &weights) -> uint64_t {
      using W = std::remove_cvref_t<decltype(weights)>;
      if constexpr (std::is_same_v<W, Qwen4ExpWeights>)
        return weights.layout.streamableWeightBytes();
      else
        return 0;
    }, target);
  }
  [[nodiscard]] uint64_t streamCacheActualAllocatedBytes() const noexcept {
    return std::visit([](const auto &weights) -> uint64_t {
      using W = std::remove_cvref_t<decltype(weights)>;
      if constexpr (std::is_same_v<W, Qwen4ExpWeights>)
        return weights.expertCacheActualAllocatedBytes();
      else
        return 0;
    }, target);
  }
  [[nodiscard]] std::span<const WeightFileRecord> targetFiles() const noexcept {
    return std::visit([](const auto &weights) ->
                          std::span<const WeightFileRecord> {
      return weights.files;
    }, target);
  }
  [[nodiscard]] bool hasVision() const noexcept {
    return vision.actualAllocatedBytes > 0;
  }
  // Phase 1 loads the draft weights; the draft forward arrives in phase 2.
  [[nodiscard]] bool hasDraft() const noexcept {
    return draft.actualAllocatedBytes > 0;
  }
};

// Model execution resources; physical memory admission remains governed by
// the engine through admitAllocation.
struct RuntimeContext final {
  metal::MetalBackend &backend;
  metal::AllocationAdmission admitAllocation;
  const ModelPackage &package;
  kv::Q8PageStorage &kvPages;
  StateStorage &stateStorage;
  const ops::ExecutionPlans &operators;
  uint32_t maximumImagePatches = ops::kMaximumImagePatches;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;
};

// Validates only the interface between independently defined target and draft
// architectures. Each architecture validates its own tensor and state layout.
void requireCompatibleModelPackage(const ModelPackage &package);

// Production loading is selected by the validated package descriptor. There
// is one shared engine and DFlash controller; only model execution differs.
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const std::filesystem::path &root);
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const std::filesystem::path &root,
                 const ModelDescriptor &descriptor);

[[nodiscard]] ModelMemoryPlan
plannedRuntimeMemory(const DeviceCapabilities &device,
                     const ModelPackage &package,
                     const ops::ExecutionPlans &operators);
[[nodiscard]] std::unique_ptr<StateStorage>
createStateStorage(metal::MetalBackend &backend,
                   metal::AllocationAdmission admitAllocation,
                   const ModelPackage &package);
[[nodiscard]] std::unique_ptr<RuntimeModel>
createRuntime(RuntimeContext context);

} // namespace splash::model
