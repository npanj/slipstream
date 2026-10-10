#include "ops/AneDraftOffload.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace splash::ops::ane;

void testAneDraftOffloading() {
  std::cout << "[testAneDraftOffloading] Testing Apple Neural Engine heterogeneous draft dispatch..." << std::endl;

  AneDraftConfig cfg;
  cfg.draftLayers = 5;
  cfg.hiddenSize = 5120;
  cfg.maxDraftTokens = 4;
  cfg.useAne = true;

  AneDraftOffloader offloader(cfg);
  assert(offloader.target() == AneExecutionTarget::AppleNeuralEngine);
  assert(offloader.draftLayers() == 5);
  assert(offloader.maxDraftTokens() == 4);

  std::vector<uint32_t> context = {101, 202, 303, 404, 505};
  std::vector<float> targetHidden(5120, 0.42f);

  // 1. Synchronous dispatch verification
  auto drafts = offloader.executeDraft(context, targetHidden.data(), targetHidden.size());
  assert(drafts.size() == 4);
  assert(offloader.totalDraftDispatches() == 1);
  assert(offloader.aneOffloadHits() == 1);

  // 2. Heterogeneous Concurrent Execution:
  // Metal GPU target verification simulated on current thread while ANE executes draft concurrently
  auto t0 = std::chrono::high_resolution_clock::now();
  auto futureDrafts = offloader.dispatchDraftAsync(context, targetHidden.data(), targetHidden.size());

  // Simulate GPU target verification (e.g. 5 ms)
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Harvest draft tokens from ANE
  auto asyncDrafts = futureDrafts.get();
  auto t1 = std::chrono::high_resolution_clock::now();

  assert(asyncDrafts.size() == 4);
  assert(asyncDrafts == drafts);
  assert(offloader.totalDraftDispatches() == 2);
  assert(offloader.aneOffloadHits() == 2);

  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Concurrent GPU Target Verification + ANE Draft completed in " << elapsed_ms << " ms" << std::endl;
  std::cout << "  Draft Tokens generated on ANE: [";
  for (size_t i = 0; i < asyncDrafts.size(); ++i) {
    std::cout << asyncDrafts[i] << (i + 1 < asyncDrafts.size() ? ", " : "");
  }
  std::cout << "]" << std::endl;
  std::cout << "[testAneDraftOffloading] PASS" << std::endl;
}

int main() {
  std::cout << "=== Running Apple Neural Engine Heterogeneous Draft Offload Tests ===" << std::endl;
  testAneDraftOffloading();
  std::cout << "=== All ANE Heterogeneous Draft Tests Passed! ===" << std::endl;
  return 0;
}
