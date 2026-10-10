#include "model/Model.hpp"

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using namespace splash;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename F> void requireThrows(F &&run, const char *message) {
  bool rejected = false;
  try {
    static_cast<void>(run());
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, message);
}

} // namespace

int main() {
  constexpr uint32_t window = 2048; // draftContextTokens
  // Fresh short prompt: one capture span over the whole prompt with a reset,
  // all rows active.
  {
    const DraftContextPlan plan = planDraftContext(0, 500, std::nullopt, {});
    require(plan.captureSpans.size() == 1, "fresh prompt needs one span");
    require(plan.captureSpans[0].begin == 0 && plan.captureSpans[0].end == 500,
            "fresh span should cover the prompt");
    require(plan.captureSpans[0].resetDraftState, "fresh span must reset");
    require(plan.draftContextRowsActive == 500, "all rows should be active");
    require(plan.draftStateResets == 1, "fresh prompt resets once");
  }
  // Fresh long prompt: capture covers only the trailing window.
  {
    const DraftContextPlan plan = planDraftContext(0, 5000, std::nullopt, {});
    require(plan.captureSpans.size() == 1, "long prompt needs one span");
    require(plan.captureSpans[0].begin == 5000 - window &&
                plan.captureSpans[0].end == 5000,
            "long span should cover the trailing window");
    require(plan.draftContextRowsAvoided == 5000 - window,
            "avoided rows miscounted");
  }
  // Restored state within the window continues without a reset.
  {
    const DraftContextPlan plan =
        planDraftContext(400, 900, std::optional<uint32_t>(400), {});
    require(plan.captureSpans.size() == 1, "continued span missing");
    require(!plan.captureSpans[0].resetDraftState,
            "continued span must not reset");
    require(plan.draftStateResets == 0, "continued span must not reset");
  }
  // Materialization boundary splits active from materialization rows.
  {
    const std::vector<uint32_t> materialization{300};
    const DraftContextPlan plan =
        planDraftContext(0, 900, std::nullopt, materialization);
    require(plan.draftContextRowsMaterialization == 300,
            "materialization rows miscounted");
    require(plan.draftContextRowsActive == 600, "active rows miscounted");
    const DispatchDraftCapturePlan dispatch =
        draftCaptureSpansForDispatch(plan, 0, 900);
    require(dispatch.size() == 1, "dispatch should see one span");
    require(dispatch[0].activeRows == 600 &&
                dispatch[0].materializationRows == 300,
            "dispatch row split is wrong");
  }
  // Dispatch windows clip spans to the dispatched range.
  {
    const DraftContextPlan plan = planDraftContext(0, 3000, std::nullopt, {});
    const DispatchDraftCapturePlan first =
        draftCaptureSpansForDispatch(plan, 952, 2000);
    require(first.size() == 1 && first[0].absoluteBegin == 952 &&
                first[0].absoluteEnd == 2000,
            "dispatch clipping is wrong");
  }
  // Invalid plans are rejected.
  {
    requireThrows(
        [] { static_cast<void>(planDraftContext(100, 50, std::nullopt, {})); },
        "reversed range accepted");
    requireThrows(
        [] {
          static_cast<void>(
              planDraftContext(100, 200, std::optional<uint32_t>(50), {}));
        },
        "restore mismatch accepted");
    requireThrows(
        [] { static_cast<void>(planDraftContext(100, 200, std::nullopt, {})); },
        "nonzero replay without restore accepted");
    const std::vector<uint32_t> unsorted{400, 300};
    requireThrows(
        [&] {
          static_cast<void>(planDraftContext(0, 900, std::nullopt, unsorted));
        },
        "unsorted boundaries accepted");
    const DraftContextPlan plan = planDraftContext(0, 500, std::nullopt, {});
    requireThrows(
        [&] {
          static_cast<void>(draftCaptureSpansForDispatch(plan, 0, 600));
        },
        "oversize dispatch accepted");
  }

  std::cout << "draft-context-plan-test: ok\n";
  return 0;
}
