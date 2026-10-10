// Draft weights load check for the 27B package.
//
//   draft-weights-load METALLIB PACKAGE_ROOT
//
// Loads the packed DFlash draft role through the production loader and checks
// layer/section geometry against the layout: 5 layers, bf16 norm widths, and
// full vocabulary codebooks. Prints allocated bytes and the file fingerprint.

#include "metal/MetalBackend.hpp"
#include "model/DFlashDraft.hpp"

#import <Foundation/Foundation.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

using splash::metal::MetalBackend;
using splash::model::DFlashDraftLayout;
using splash::model::loadDFlashDraftWeights;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: draft-weights-load METALLIB PACKAGE_ROOT\n";
    return 2;
  }
  @autoreleasepool {
    try {
      MetalBackend backend(argv[1]);
      const auto weights = loadDFlashDraftWeights(
          backend, std::filesystem::path(argv[2]) / "draft",
          DFlashDraftLayout{});
      require(weights.layers.size() == 5, "draft should load 5 layers");
      require(weights.layout.hiddenSize == 5120, "unexpected hidden size");
      require(weights.layers[0].inputNorm.sizeBytes() == 5120 * 2,
              "input-norm section has wrong bytes");
      require(weights.predecessorCodebook.sizeBytes() ==
                  uint64_t{248320} * 256 * 2,
              "predecessor codebook has wrong bytes");
      require(weights.successorCodebook.sizeBytes() ==
                  uint64_t{248320} * 256 * 2,
              "successor codebook has wrong bytes");
      require(weights.actualAllocatedBytes > 0, "draft allocated nothing");
      require(!weights.manifestFingerprintSha256.empty(),
              "draft fingerprint is empty");
      require(weights.files.size() == 6, "draft should record 6 files");
      std::cout << "draft layers: " << weights.layers.size() << "\n"
                << "allocated bytes: " << weights.actualAllocatedBytes << "\n"
                << "fingerprint: " << weights.manifestFingerprintSha256 << "\n"
                << "draft-weights-load: ok\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "draft-weights-load failed: " << error.what() << "\n";
      return 1;
    }
  }
}
