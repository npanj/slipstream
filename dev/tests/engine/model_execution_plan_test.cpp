#include "model/ModelFactory.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {

using namespace splash;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

uint64_t aligned(uint64_t bytes) {
  constexpr uint64_t alignment = 16 * 1024;
  return (bytes + alignment - 1) & ~(alignment - 1);
}

template <class Weights>
model::ModelPackage package() {
  model::ModelPackage result;
  Weights target;
  result.descriptor =
      model::makeModelDescriptor("operator workspace test", target.layout);
  result.target = std::move(target);
  return result;
}

void checkPackage(const model::ModelPackage &package, uint32_t family) {
  DeviceCapabilities device;
  device.appleGpuFamily = family;
  ops::ExecutionPlans baseline(device);
  const auto before = model::plannedRuntimeMemory(device, package, baseline);
  const auto geometry = std::visit([](const auto &weights) {
    return model::qwenTargetGeometry(weights);
  }, package.target);
  const ops::AttentionShape attention{geometry.attentionQueryHeads,
                                      geometry.attentionKvHeads,
                                      geometry.attentionHeadDimension};
  ops::OperatorChoices choices;
  choices.prefillAttention.push_back(
      {{attention}, {ops::PrefillSplitMultiplier::Two}});
  if (geometry.ffnKind == model::QwenFfnKind::SparseMoe)
    choices.moe.push_back({{geometry.moe, 24, ops::MoePhase::Decode},
                           {ops::MoeExpertTile::M32}});
  ops::ExecutionPlans selected(device);
  selected.install(choices);
  const auto after = model::plannedRuntimeMemory(device, package, selected);
  const auto prefillBefore = baseline.prefillAttentionWorkspace(
      SPLASH_PREFILL_TOKEN_BUDGET, attention.queryHeads, geometry.kvLayout);
  const auto prefillAfter = selected.prefillAttentionWorkspace(
      SPLASH_PREFILL_TOKEN_BUDGET, attention.queryHeads, geometry.kvLayout);
  const uint64_t prefillGrowth =
      aligned(prefillAfter.partialsBytes) - aligned(prefillBefore.partialsBytes) +
      aligned(prefillAfter.statisticsBytes) - aligned(prefillBefore.statisticsBytes);
  // The selected split count and the fallback baseline share an arena whose
  // governed bound includes the larger candidate's exact scratch requirement.
  require(prefillGrowth > 0 &&
              after.sharedPrefillPlannedAllocatedBytes ==
                  before.sharedPrefillPlannedAllocatedBytes + prefillGrowth,
          "runtime prefill allocation lost the selected split workspace bound");
  const auto selectedPrefill = selected.prefillAttention(
      SPLASH_PREFILL_TOKEN_BUDGET, attention.queryHeads, geometry.kvLayout, 131072);
  require(selectedPrefill.configuration.splitMultiplier == ops::PrefillSplitMultiplier::Two &&
              selectedPrefill.workspace.partialsBytes ==
                  2 * baseline.prefillAttention(SPLASH_PREFILL_TOKEN_BUDGET, attention.queryHeads,
                                             geometry.kvLayout, 131072)
                      .workspace.partialsBytes,
          "runtime did not install the selected prefill split plan");

  uint64_t decodeGrowth = 0;
  if (geometry.ffnKind == model::QwenFfnKind::SparseMoe) {
    constexpr std::array fields{
        &ops::MoeWorkspace::selectedExpertsBytes,
        &ops::MoeWorkspace::routingWeightsBytes,
        &ops::MoeWorkspace::tileDescriptorsBytes,
        &ops::MoeWorkspace::tileCountBytes,
        &ops::MoeWorkspace::groupedRoutesBytes,
        &ops::MoeWorkspace::routeRowsBytes,
        &ops::MoeWorkspace::groupedInputBytes,
        &ops::MoeWorkspace::expertIntermediateBytes,
        &ops::MoeWorkspace::expertOutputBytes};
    const auto oldMoe = baseline.moeDecodeWorkspacePerLane(geometry.moe);
    const auto newMoe = selected.moeDecodeWorkspacePerLane(geometry.moe);
    for (auto field : fields)
      decodeGrowth += aligned(4 * (newMoe.*field)) - aligned(4 * (oldMoe.*field));
    require(decodeGrowth > 0, "M24 expert plan did not reserve larger scratch");
  }
  require(after.sharedDecodePlannedAllocatedBytes ==
              before.sharedDecodePlannedAllocatedBytes + decodeGrowth,
          "runtime decode allocation does not use all selected width bounds");
  require(after.activeStateCellPlannedAllocatedBytes ==
              before.activeStateCellPlannedAllocatedBytes &&
              after.pipelineReserveBytes == before.pipelineReserveBytes &&
              after.runtimeOverheadReserveBytes == before.runtimeOverheadReserveBytes,
          "kernel selection changed state or unrelated memory reserves");
  selected.install({});
  const auto reset = model::plannedRuntimeMemory(device, package, selected);
  require(reset.sharedPrefillPlannedAllocatedBytes ==
              before.sharedPrefillPlannedAllocatedBytes &&
              reset.sharedDecodePlannedAllocatedBytes ==
              before.sharedDecodePlannedAllocatedBytes,
          "reset left stale selected workspace");
}

} // namespace

int main() {
  try {
    const auto qwen4exp = package<model::Qwen4ExpWeights>();
    for (uint32_t family : {9U, 10U})
      checkPackage(qwen4exp, family);
    std::cout << "model execution plans: PASS (qwen4exp)\n";

    const auto qwen38 = package<model::Qwen3_8Weights>();
    for (uint32_t family : {9U, 10U})
      checkPackage(qwen38, family);
    std::cout << "model execution plans: PASS (qwen38)\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
