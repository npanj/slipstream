#include "model/DFlashDraft.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using namespace splash::model;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void requireRejects(DFlashDraftLayout layout, const char *message) {
  bool rejected = false;
  try {
    validateDFlashDraftLayout(layout);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, message);
}

} // namespace

int main() {
  // The shipped 27B draft layout validates as-is.
  {
    DFlashDraftLayout layout;
    validateDFlashDraftLayout(layout);
    require(layout.layers == 5, "draft should have 5 layers");
    require(layout.hiddenSize == 5120, "draft hidden size should be 5120");
    require(layout.vocabularySize == 248320, "draft vocab should be 248320");
    require(layout.targetHiddenSize == 25600,
            "draft target hidden should be 25600");
    require(layout.selectorRank == 256, "draft selector rank should be 256");
    require(layout.kvHeads == 8, "draft should have 8 kv heads");
  }

  // Zero dimensions are rejected.
  {
    DFlashDraftLayout zeroLayers;
    zeroLayers.layers = 0;
    requireRejects(zeroLayers, "zero layers should be rejected");
    DFlashDraftLayout zeroHidden;
    zeroHidden.hiddenSize = 0;
    requireRejects(zeroHidden, "zero hidden size should be rejected");
    DFlashDraftLayout zeroTheta;
    zeroTheta.rotaryTheta = 0.0F;
    requireRejects(zeroTheta, "zero rotary theta should be rejected");
  }

  // The selector kernels are compiled for rank 256 only.
  {
    DFlashDraftLayout badRank;
    badRank.selectorRank = 128;
    requireRejects(badRank, "non-256 selector rank should be rejected");
  }

  // Q4 shapes the dequant kernels cannot bind are rejected.
  {
    DFlashDraftLayout badQ4;
    badQ4.dynamicSize = 1000;
    requireRejects(badQ4, "misaligned Q4 width should be rejected");
  }

  std::cout << "draft-weights-test: ok\n";
  return 0;
}
