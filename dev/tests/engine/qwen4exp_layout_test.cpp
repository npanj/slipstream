#include "models/qwen4exp/Qwen4Exp.hpp"
#include "model/WeightStore.hpp"
#include "ops/GDN.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace splash;
using namespace splash::model;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// Mirrors validateQ4Layout, which lives in WeightStore.cpp and would pull
// Metal into what is otherwise a pure arithmetic test.
constexpr bool q4LayoutIsValid(uint32_t outputSize, uint32_t inputSize,
                               uint32_t storageN = kQ4StorageN) {
  return outputSize && inputSize && inputSize % kQ4GroupElements == 0 &&
         outputSize % storageN == 0;
}

// The packed widths are derived, not transcribed: a concatenation padded up
// to StorageN. Checking them here catches a bad constant before any weight
// file is written against it.
void packedWidthsFollowFromTheConcatenation() {
  constexpr Qwen4ExpLayout layout;
  static_assert(layout.actualGdnWidth() == 16480,
                "qkv + z + b + a must be 16480 rows");
  static_assert(layout.packedGdnWidth == 16640,
                "16480 padded up to StorageN 256 is 16640");
  static_assert(layout.packedGdnWidth % kQ4StorageN == 0);
  static_assert(layout.packedGdnWidth - layout.actualGdnWidth() == 160,
                "the last GDN tile carries 160 zero rows");

  static_assert(layout.actualFullWidth() == 13312,
                "2*q + k + v with a gated query is 13312 rows");
  static_assert(layout.packedFullWidth == layout.actualFullWidth(),
                "the attention concatenation needs no padding");
  static_assert(layout.packedFullWidth % kQ4StorageN == 0);

  static_assert(layout.attentionWidth ==
                    layout.gdnValueHeads * layout.gdnHeadDimension,
                "attention width is value heads by head dimension");
  static_assert(layout.convolutionDimension ==
                    2 * layout.gdnKeyHeads * layout.gdnHeadDimension +
                        layout.gdnValueHeads * layout.gdnHeadDimension,
                "conv width is 2*k + v head widths");
}

// Three layers in four are linear attention (the shared GDN kernels), one in
// four full attention.
void linearAttentionLayers() {
  constexpr Qwen4ExpLayout next;

  // 48 layers at period 4 gives 12 attention layers and 36 linear ones.
  static_assert(next.attentionLayerCount() == 12);
  static_assert(next.layers - next.attentionLayerCount() == 36);
  static_assert(!next.isFullAttentionLayer(0));
  static_assert(next.isFullAttentionLayer(3));
  static_assert(next.isFullAttentionLayer(47));
}

// The linear-attention block needs no new kernel. ops::GDN already compiles
// for this exact shape - see kernelShape() in ops/GDN.cpp - because Qwen3.8
// has the same head geometry, and the GDN kernels are templated on head
// counts rather than hidden size.
void gdnShapeIsAlreadyCompiled() {
  constexpr Qwen4ExpLayout l;
  constexpr ops::GdnShape shape{l.gdnKeyHeads, l.gdnValueHeads,
                                l.gdnHeadDimension, l.convolutionDimension,
                                l.packedGdnWidth};
  static_assert(shape.valid(), "the qwen4exp GDN shape must be well formed");
  static_assert(shape == ops::GdnShape{16, 48, 128, 10240, 16640},
                "qwen4exp must match a GDN shape the engine already compiles");
}

// Every projection the engine can already express must satisfy the Q4 rules.
void supportedProjectionsAreQ4Aligned() {
  constexpr Qwen4ExpLayout layout;
  static_assert(q4LayoutIsValid(layout.packedGdnWidth, layout.hiddenSize));
  static_assert(q4LayoutIsValid(layout.packedFullWidth, layout.hiddenSize));
  static_assert(q4LayoutIsValid(layout.hiddenSize, layout.attentionWidth));
  static_assert(q4LayoutIsValid(layout.vocabularySize, layout.hiddenSize));
  require(layout.hiddenSize % kQ4GroupElements == 0,
          "hidden size must be group-aligned as a projection input");
  require(layout.hiddenSize % kQ4StorageN == 0,
          "hidden size must be StorageN-aligned as a projection output");
}

// Expert projections tile narrower than everything else. This is the whole
// reason kQ4ExpertStorageN exists, so pin both directions.
void expertProjectionsTileAt128() {
  constexpr Qwen4ExpLayout layout;
  static_assert(layout.expertStorageN == kQ4ExpertStorageN);

  // 640 is not expressible at the usual width, which is why it tiles narrower.
  static_assert(!q4LayoutIsValid(layout.expertIntermediateSize,
                                 layout.hiddenSize, kQ4StorageN),
                "640 must not be expressible at StorageN 256");
  static_assert(q4LayoutIsValid(layout.expertIntermediateSize,
                                layout.hiddenSize, kQ4ExpertStorageN),
                "640 must be expressible at StorageN 128");
  static_assert(q4LayoutIsValid(layout.hiddenSize,
                                layout.expertIntermediateSize,
                                kQ4ExpertStorageN),
                "the down projection must tile at 128 as well");

  // The narrower tile must still be a whole number of 32-wide matmul slices.
  static_assert(kQ4ExpertStorageN % 32 == 0);
  static_assert(kQ4StorageN % kQ4ExpertStorageN == 0);

  // What padding to the usual width would have cost, for the record: three
  // projections per expert, 512 experts, 48 layers, at 9 bytes per 16 weights.
  constexpr uint64_t stored =
      uint64_t(layout.expertIntermediateSize) * layout.hiddenSize * 3 *
      layout.experts * layout.layers * 9 / 16;
  constexpr uint64_t padded = uint64_t(768) * layout.hiddenSize * 3 *
                              layout.experts * layout.layers * 9 / 16;
  require(padded - stored > 12ULL * 1024 * 1024 * 1024,
          "padding to 768 would have cost more than 12 GiB");
}

// The MoE shape this layout produces must be one ops::MoE accepts. This used
// to assert the opposite - that the expert count was over the ceiling - and
// tightening the limit is what brought it here.
void moeShapeIsAccepted() {
  constexpr Qwen4ExpLayout layout;
  constexpr ops::MoeShape shape{layout.hiddenSize, layout.experts,
                                layout.expertsPerToken,
                                layout.expertIntermediateSize,
                                layout.expertStorageN};
  static_assert(layout.experts == kQwen4ExpExpertCount);
  static_assert(shape.valid(), "ops::MoE must accept the qwen4exp MoE shape");
  // Ten of five hundred and twelve, in 128-wide tiles, and the router pads to
  // whole 256-wide tiles.
  static_assert(shape.routerWidth() == 512);
  static_assert(shape.expertIntermediateSize % shape.storageN == 0);
}

// The per-layer embedding's shape was wrong once in a way the totals hid:
// sixteen heads of 160 and eight of 2560 have the same element count, so the
// footprint agreed while the structure did not. Pin the structure.
void ngramGeometryIsPinned() {
  constexpr Qwen4ExpLayout layout;
  static_assert(layout.ngramHeads() == 16,
                "one head per context position per hash");
  static_assert(layout.ngramHeadDimension() == 160,
                "the heads partition the embedding width");
  static_assert(layout.ngramHeads() * layout.ngramHeadDimension() ==
                    layout.ngramEmbeddingSize,
                "the heads must tile the embedding exactly");

  // 160 is not a whole number of ordinary groups, which is the whole reason
  // this table is quantized in finer ones.
  static_assert(layout.ngramHeadDimension() % kQ4GroupElements != 0);
  static_assert(layout.ngramHeadDimension() % kQ4FineGroupElements == 0);
  static_assert(layout.ngramVocabularySize % layout.ngramShards == 0,
                "the table must divide evenly into its shards");

  // Rows are prime-sized per head, above the base, then padded; the padded
  // total must still cover every head.
  static_assert(layout.ngramVocabularySize >
                    uint64_t(layout.ngramHeads()) * layout.ngramVocabularyBase,
                "the padded vocabulary must hold every head's table");
}

void stateLayoutsAreConsistent() {
  constexpr Qwen4ExpLayout layout;
  const auto kv = layout.q8Layout();
  // The trunk's 12 attention layers, and one for the MTP draft head.
  require(kv.attentionLayers == 12 + layout.mtpLayers && layout.mtpLayers == 1,
          "KV cache covers the attention layers and the MTP head");
  require(kv.kvHeads == layout.attentionKvHeads, "KV heads mismatch");
  require(kv.headDimension == layout.attentionHeadDimension,
          "KV head dimension mismatch");

  const auto gdn = layout.gdnStateLayout();
  require(gdn.layers == 36, "GDN state covers only the linear layers");
  require(gdn.convolutionChannels == layout.convolutionDimension,
          "GDN conv channel count mismatch");
  require(gdn.convolutionHistory == kGdnConvolutionTaps - 1,
          "GDN conv history must be taps minus one");
}

// Decode-feature defaults are set by measurement (see the switches in
// Qwen4Exp.hpp): the horizon advisory regressed pinned-cache decode by
// ~20% and must stay opt-in, the eviction guard stays on, and EAGLE-2 tree
// drafting stays opt-in. This pins the polarity so a drive-by edit cannot
// silently re-enable a measured cost. (The engine reads each switch once at
// first decode, so this tests the predicate, not a live toggle.)
void decodeFeatureSwitchDefaults() {
  unsetenv("SPLASH_HORIZON_ADVISORY");
  unsetenv("SPLASH_NO_HORIZON_ADVISORY");
  unsetenv("SPLASH_NO_EVICTION_GUARD");
  unsetenv("SPLASH_EAGLE2_DRAFT");
  require(!horizonAdvisoryEnabled(), "horizon advisory must default off");
  require(evictionGuardEnabled(), "eviction guard must default on");
  require(!eagle2DraftEnabled(), "eagle-2 drafting must default off");
  setenv("SPLASH_HORIZON_ADVISORY", "1", 1);
  require(horizonAdvisoryEnabled(), "SPLASH_HORIZON_ADVISORY=1 must opt in");
  setenv("SPLASH_NO_HORIZON_ADVISORY", "1", 1);
  require(!horizonAdvisoryEnabled(), "explicit off must win over opt-in");
  unsetenv("SPLASH_HORIZON_ADVISORY");
  unsetenv("SPLASH_NO_HORIZON_ADVISORY");
  setenv("SPLASH_NO_EVICTION_GUARD", "1", 1);
  require(!evictionGuardEnabled(), "SPLASH_NO_EVICTION_GUARD=1 must opt out");
  unsetenv("SPLASH_NO_EVICTION_GUARD");
  setenv("SPLASH_EAGLE2_DRAFT", "1", 1);
  require(eagle2DraftEnabled(), "SPLASH_EAGLE2_DRAFT=1 must opt in");
  setenv("SPLASH_EAGLE2_DRAFT", "0", 1);
  require(!eagle2DraftEnabled(), "SPLASH_EAGLE2_DRAFT=0 must stay off");
  unsetenv("SPLASH_EAGLE2_DRAFT");
}

} // namespace

int main() {
  try {
    decodeFeatureSwitchDefaults();
    packedWidthsFollowFromTheConcatenation();
    linearAttentionLayers();
    supportedProjectionsAreQ4Aligned();
    gdnShapeIsAlreadyCompiled();
    expertProjectionsTileAt128();
    moeShapeIsAccepted();
    ngramGeometryIsPinned();
    stateLayoutsAreConsistent();
  } catch (const std::exception &error) {
    std::cerr << "qwen4exp layout test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "qwen4exp layout test passed\n";
  return 0;
}
