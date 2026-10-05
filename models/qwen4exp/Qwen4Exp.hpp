#pragma once

#include <atomic>
#include <memory>

#include <dispatch/dispatch.h>

#include "model/QwenTarget.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightStore.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

// Qwen3.8-Flash-Next, upstream architecture name `qwen4exp`. Its linear
// attention is dimensionally identical to the Qwen3.8 GDN block, so three
// layers in four reuse the existing mixer geometry; the fourth is sparse
// attention with a routing indexer, which has no operator yet.
//
// Numbers below come from the published config.json. The fields under
// "no Splash equivalent" describe modules this engine cannot execute today;
// they are recorded here so the package validator can check them and so the
// gaps are visible in one place. See kUnsupported* below.
struct Qwen4ExpLayout final {
  // 0011: mixer projections 8-bit (they were 4-bit in 0001; see
  // dev/benchmarks/qwen4exp/reference_logits.py --quant for why).
  // 0031: hyper-connection mix weights 8-bit, in the tiled matrix layout.
  static constexpr std::string_view layerMagic = "MDFN0031";

  uint32_t maximumContextTokens = kv::kMaximumLogicalTokens;
  uint32_t layers = 48;
  uint32_t hiddenSize = 2560;
  uint32_t vocabularySize = 248320;
  uint32_t packedGdnWidth = 16640;
  uint32_t packedFullWidth = 13312;
  uint32_t convolutionDimension = 10240;
  uint32_t gdnKeyHeads = 16;
  uint32_t gdnValueHeads = 48;
  uint32_t gdnHeadDimension = 128;
  uint32_t attentionWidth = 6144;
  uint32_t attentionQueryHeads = 24;
  uint32_t attentionKvHeads = 2;
  uint32_t attentionHeadDimension = 256;
  uint32_t rotaryPairs = 32;
  float rotaryTheta = 10'000'000.0F;
  uint32_t fullAttentionPeriod = 4;
  uint32_t experts = 512;
  uint32_t expertsPerToken = 10;
  uint32_t expertIntermediateSize = 640;
  // 640 is not a multiple of 256 but is a multiple of 128, so the expert
  // projections tile narrower than everything else rather than padding to
  // 768. See kQ4ExpertStorageN.
  uint32_t expertStorageN = kQ4ExpertStorageN;

  // TODO(qwen4exp): confirm against the shipped tokenizer; the vocabulary
  // matches Qwen3.8 exactly, so these are carried over as placeholders.
  uint32_t maskToken = 248070;
  std::array<uint32_t, 2> stopTokens{248044, 248046};

  // ---- no operator yet; the weights are still described and mapped ------
  uint32_t hyperConnectionCount = 4;
  uint32_t hyperConnectionLowRank = 320;
  uint32_t indexerHeads = 4;
  uint32_t indexerKvHeads = 1;
  uint32_t indexerHeadDimension = 128;
  uint32_t indexerBudget = 2048;
  // The n-gram table is a Q4 lookup table on one layer, not a per-layer cost:
  // 20,000,000 rows of 2560 at nine bytes per sixteen weights is 26.8 GiB,
  // which is what this model measures on disk.
  uint32_t ngramSize = 3;
  uint32_t ngramLayer = 1;
  // The table is one row per hashed n-gram per head, not one per token. Each
  // head gets its own prime-sized vocabulary just above ngramVocabularyBase,
  // and the padded total is what the packed file holds. Sixteen heads of 160
  // multiply out to the same element count as eight of 2560, which is why a
  // wrong shape can still produce the right total.
  uint32_t ngramVocabularyBase = 20'000'000;
  uint32_t ngramVocabularySize = 320'001'536;
  uint32_t ngramEmbeddingSize = 2560;
  uint32_t ngramHeadsPerOrder = 8;
  uint32_t ngramShards = 128;
  uint32_t pleConvolutionTaps = 4;

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return fullAttentionPeriod && (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    return fullAttentionPeriod ? layers / fullAttentionPeriod : 0;
  }
  // Packed projections are concatenated then padded up to StorageN.
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept {
    return convolutionDimension + attentionWidth + 2 * gdnValueHeads;
  }
  // Query width is doubled for the output gate, as in Qwen3.8.
  [[nodiscard]] constexpr uint32_t actualFullWidth() const noexcept {
    return 2 * attentionQueryHeads * attentionHeadDimension +
           2 * attentionKvHeads * attentionHeadDimension;
  }
  // One key/value layer past the trunk's for the MTP draft head, whose one
  // decoder layer has the trunk's attention shape. Unused without the head.
  uint32_t mtpLayers = 1;
  [[nodiscard]] constexpr kv::Q8Layout q8Layout() const noexcept {
    return {attentionLayerCount() + mtpLayers, attentionKvHeads,
            attentionHeadDimension};
  }
  // The per-layer embedding carries two pieces of history between steps: the
  // last ngramSize - 1 tokens, for the n-gram hash, and the last
  // pleConvolutionState() normalized rows, for its dilated convolution. They
  // live in the GDN state cell's auxiliary region:
  //
  //   uint32 valid, older token, newer token, padding     16 bytes
  //   bf16   [pleConvolutionState()][hyperConnectionWidth()]
  //
  // A zeroed region (valid = 0) is a fresh sequence, which the reference
  // treats as two end-of-sequence tokens and a zero convolution history.
  static constexpr uint32_t pleHistoryHeaderBytes = 16;
  [[nodiscard]] constexpr uint64_t pleAuxiliaryBytes() const noexcept {
    return pleHistoryHeaderBytes + uint64_t{pleConvolutionState()} *
                                       hyperConnectionWidth() * 2;
  }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {layers - attentionLayerCount(), kGdnConvolutionTaps - 1,
            convolutionDimension, gdnValueHeads, gdnHeadDimension,
            gdnHeadDimension, pleAuxiliaryBytes()};
  }
  [[nodiscard]] constexpr QwenMixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,     packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads,  gdnHeadDimension,
            attentionWidth, attentionHeadDimension};
  }
  // Four residual streams, so every hyper-connection tensor is this wide.
  [[nodiscard]] constexpr uint32_t hyperConnectionWidth() const noexcept {
    return hiddenSize * hyperConnectionCount;
  }
  // Query and key-value share one indexer projection, as in the reference.
  [[nodiscard]] constexpr uint32_t indexerProjectionWidth() const noexcept {
    return (indexerHeads + indexerKvHeads) * indexerHeadDimension;
  }
  // The PLE convolution is dilated by the n-gram order, so its state spans
  // more positions than it has taps.
  [[nodiscard]] constexpr uint32_t pleConvolutionState() const noexcept {
    return (pleConvolutionTaps - 1) * ngramSize;
  }
  // One head per context position per hash, so the orders below the full
  // n-gram each contribute a set.
  [[nodiscard]] constexpr uint32_t ngramHeads() const noexcept {
    return (ngramSize - 1) * ngramHeadsPerOrder;
  }
  // The heads partition the embedding width between them.
  [[nodiscard]] constexpr uint32_t ngramHeadDimension() const noexcept {
    return ngramHeads() ? ngramEmbeddingSize / ngramHeads() : 0;
  }

  // Weights this model can be planned without holding all of: the routed
  // experts, which are touched ten at a time out of five hundred and twelve,
  // and the per-layer embedding table, which is one gathered row per head.
  // Everything else - attention, the mixers, the shared expert, the norms -
  // is read for every token and stays resident.
  [[nodiscard]] constexpr uint64_t streamableWeightBytes() const noexcept {
    const uint64_t perExpert =
        (2ULL * expertIntermediateSize * hiddenSize +
         uint64_t{hiddenSize} * expertIntermediateSize) *
        9 / 16;
    const uint64_t experts_ = perExpert * experts * layers;
    const uint64_t table = uint64_t{ngramVocabularySize} *
                           ngramHeadDimension() * 9 / 16;
    return experts_ + table;
  }

  bool operator==(const Qwen4ExpLayout &) const = default;
};

// How many experts each streamed layer caches, and the bytes that pins at
// load (caches plus the prompt staging buffer). The loader allocates exactly
// this; startup checks it against free memory first, so a second engine
// started beside a running one refuses instead of freezing the machine.
// Reads SPLASH_RESIDENT_LAYERS, SPLASH_EXPERT_CACHE_GIB,
// SPLASH_EXPERT_CACHE_CAPACITY and SPLASH_EXPERT_SLOTS (a per-layer profile:
// 48 comma-separated weights; each layer gets capacity x weight / mean weight,
// so the total stays the same).
struct Qwen4ExpCachePlan final {
  uint32_t residentLayers = 0;
  uint32_t capacity = 0;  // the average per cached layer
  // Slots per layer: trunk layers 0..47, then the MTP head's layer.
  std::vector<uint32_t> layerCapacity;
  uint64_t pinnedBytes = 0;
};
[[nodiscard]] Qwen4ExpCachePlan planQwen4ExpExpertCache(const Qwen4ExpLayout &layout,
                                                        bool mtpLayer);

// Hyper-connections replace the usual input and post-attention norms. The
// residual is four parallel streams carried as one row of hc*hidden, and each
// block reads a mix of them and writes back a gated injection.
//
// Over a row x of width H = hc*hidden, with d = hidden (from the reference
// implementation, Qwen4ExpTextGatedResidual):
//
//   xn    = hc_norm(x)                  RMS per stream, gain over all of H
//   w     = silu(down(xn) / hc)         H -> lowRank
//   w     = sigmoid(up(w))              lowRank -> H
//   input = mean over hc of             the d-wide block input
//             reshape(w,[hc,d]) * reshape(xn,[hc,d])
//   inj   = 2 * sigmoid(block_inject(xn) / hc)      H -> hc
//
// and after the block produces y of width d, the residual update is an outer
// product against the *unnormalized* input:
//
//   x = x + outer(inj, y)               [hc,d], flattened back to H
//
// The mix projections are 8-bit in the matrix kernels' tiled layout, so a
// prompt can run them as matrix products while decode reads them with its
// own kernels. The down projection's 320 outputs are padded to 512 (a whole
// number of 256-row tiles); decode never reads the padding.
inline constexpr uint32_t kHyperDownPadded = 512;
// The down projection splits its 10240-long dot products across this many
// threadgroups when a call has at most kHyperDownSplitRows rows.
inline constexpr uint32_t kHyperDownSplits = 10;
inline constexpr uint32_t kPromptStagingExperts = 144;
inline constexpr uint32_t kHyperDownSplitRows = 32;

struct Qwen4ExpHyperConnection final {
  metal::MetalBuffer norm;         // hyperConnectionWidth
  ops::Q8Projection mixDown;       // kHyperDownPadded x hyperConnectionWidth
  ops::Q8Projection mixUp;         // hyperConnectionWidth x lowRank
  // Absent on the final mixer, which collapses the streams before the head
  // and never injects a block result back (use_combine=False upstream).
  std::optional<metal::MetalBuffer> blockInject;  // count x width
  // Scratch for the split down projection's partial sums, shared by every
  // hyper-connection (they run one at a time).
  metal::MetalBuffer downPartials;
  // Prompt-path scratch, shared likewise: input sums for the products, the
  // padded down outputs, and the up logits (prefill budget rows each).
  metal::MetalBuffer promptSums;
  metal::MetalBuffer promptDown;
  metal::MetalBuffer promptLogits;
};

// The per-layer embedding on layer ngramLayer. It gathers a hashed n-gram
// embedding, projects it to one value and one key per residual stream, gates
// the value by the key against the normalized stream, and adds a dilated
// depthwise convolution of the result. It writes the full hc*hidden width, so
// it feeds every stream.
//
// The embedding table itself is sharded, with a vocabulary size and offset per
// head, which is why the head tables travel with it.
struct Qwen4ExpPerLayerEmbedding final {
  // Rows are one head wide, and quantized in finer groups than everything
  // else because 160 is not a whole number of 64-element groups.
  ops::Q4Projection table;            // ngram vocabulary x head dimension
  metal::MetalBuffer headOffsets;     // ngramHeads
  metal::MetalBuffer headVocabularySizes;
  metal::MetalBuffer layerMultipliers;
  ops::Q4Projection keyProjection;    // hc width x embedding size
  ops::Q4Projection valueProjection;  // hidden x embedding size
  metal::MetalBuffer keyNorm;         // hc width
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer convolutionNorm;
  metal::MetalBuffer convolutionWeights;  // hc width x taps
};

// Sparse attention keeps an indexer that scores which keys to read.
struct Qwen4ExpIndexer final {
  ops::Q4Projection queryKeyProjection;  // indexerProjectionWidth x hidden
  metal::MetalBuffer queryNorm;          // indexerHeadDimension
  metal::MetalBuffer keyNorm;            // indexerHeadDimension
};

struct Qwen4ExpLayerExpertCache final {
  metal::MetalBuffer cacheGate;
  metal::MetalBuffer cacheUp;
  metal::MetalBuffer cacheDown;
  uint32_t capacity = 0;
  uint32_t numCached = 0;
  uint32_t clock = 0;
  std::vector<int16_t> expertToSlot;
  std::vector<int16_t> slotToExpert;
  std::vector<uint32_t> lruTime;
  // Use counts per expert, halved every kFrequencyHalfLife steps. Eviction
  // takes the least-used slot, not only the least recent: replaying real
  // routing at this cache size, that misses ~17% less than pure LRU.
  std::vector<uint32_t> frequency;
  uint32_t nextHalving = 0;
  // Background reads of predicted experts (lookahead). Staging the layer
  // waits for them first; the slots are already claimed in expertToSlot.
  dispatch_group_t inflight = nullptr;
  bool pending = false;
  // Background reads still writing each slot (three per expert, one per
  // matrix). Decode waits only for the slots it needs, and never evicts a
  // slot whose read has not landed.
  std::shared_ptr<std::atomic<uint32_t>[]> slotReads;
};

// Section order per layer file, which the packer must follow exactly:
//
//   attention hyper-connection   norm, inject, mix down, mix up
//   mixer                        GDN or sparse attention, as Qwen3.8
//   indexer                      attention layers only
//   mlp hyper-connection         norm, inject, mix down, mix up
//   experts                      router, gate, up, down, shared x3, gate
// Where a streaming layer's routed experts sit in its file, for reading
// cache misses directly (file cache off) instead of faulting them in.
struct Qwen4ExpExpertSource final {
  int fd = -1;
  uint64_t gate = 0; // file offsets of expert 0 of each matrix
  uint64_t up = 0;
  uint64_t down = 0;
};

struct Qwen4ExpLayerWeights final {
  Qwen4ExpHyperConnection attentionHyperConnection;
  QwenMixerWeights mixer;
  std::optional<Qwen4ExpIndexer> indexer;
  Qwen4ExpHyperConnection mlpHyperConnection;
  ops::MoeWeights ffn;
  mutable Qwen4ExpLayerExpertCache expertCache;
  Qwen4ExpExpertSource expertSource;
};

// The multi-token-prediction head's combiner and closing mixer. Its decoder
// layer is an ordinary attention layer (mtpLayer). Per draft step:
//
//   e = fcEmbedding(rms(embed(token)) * (1 + embeddingNorm))
//   h = fcHidden applied to each stream of rms_per_stream(h_wide) * (1 + hiddenNorm)
//   residual = h + e broadcast to every stream  -> mtpLayer -> mixer -> lm_head
//
// following the llama.cpp fork's qwen4exp graph_mtp ("combiner variant B").
struct Qwen4ExpMtpCombiner final {
  metal::MetalBuffer embeddingNorm; // hidden
  metal::MetalBuffer hiddenNorm;    // hc width
  ops::Q4Projection fcEmbedding;    // hidden x hidden
  ops::Q4Projection fcHidden;       // hidden x hidden, per stream
  Qwen4ExpHyperConnection mixer;    // no injection
};

struct Qwen4ExpWeights final {
  Qwen4ExpLayout layout;
  std::vector<Qwen4ExpLayerWeights> layers;
  Qwen4ExpHyperConnection hyperConnectionMixer;
  metal::MetalBuffer finalNorm;
  ops::Q8Projection logitsProjection;
  // The same head at 4 bits, for the MTP head's guesses only.
  ops::Q4Projection draftLogitsProjection;
  // SPLASH_DRAFT_VOCAB=N: the rows of draftLogitsProjection for the N most
  // common tokens, and their token ids. Empty when off.
  ops::Q4Projection draftVocabProjection;
  std::vector<uint32_t> draftVocabIds;
  ops::Q8Projection tokenEmbedding;
  // Its own file: the embedding table alone is 26.8 GiB, which does not
  // belong inside a layer file.
  Qwen4ExpPerLayerEmbedding perLayerEmbedding;
  // Present when the package carries the MTP draft head.
  std::optional<Qwen4ExpLayerWeights> mtpLayer;
  std::optional<Qwen4ExpMtpCombiner> mtpCombiner;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
  uint32_t residentLayers = 0;
  metal::MetalBuffer streamingCacheGate;
  metal::MetalBuffer streamingCacheUp;
  metal::MetalBuffer streamingCacheDown;
  uint32_t streamingCacheCapacity = 0;
  mutable std::vector<std::vector<uint32_t>> lastSelectedExperts;
  // Lookahead routing: the router of layer L+2 applied to layer L+1's input,
  // a guess at which experts L+2 will want. Sized on first use.
  mutable metal::MetalBuffer predictSelected;
  mutable metal::MetalBuffer predictWeights;
  mutable metal::MetalBuffer predictScratch;
  // Prompt expert waves: one tile count per wave, 256 bytes apart.
  mutable metal::MetalBuffer prefillRangeCounts;
  // Decode expert waves: [first, end) of the cached tiles at byte 0 and of
  // the tiles still being read at byte 256.
  mutable metal::MetalBuffer decodeRanges;
  // The MTP pick's per-slice candidates and softmax mass (mtp_pick.metal).
  mutable metal::MetalBuffer mtpPickIds, mtpPickValues, mtpPickMass;
  // The split prompt kernels' gate outputs, kept apart from expertOutput.
  mutable metal::MetalBuffer prefillGateScratch;
  // MTP draft scratch, allocated on first use.
  mutable metal::MetalBuffer mtpTokens, mtpEmbed, mtpNorm, mtpE, mtpOnes;
  mutable metal::MetalBuffer mtpCos, mtpSin, mtpHin;
  mutable metal::MetalBuffer mtpPrefillOnes;
  mutable uint64_t predictIssued = 0;
  mutable uint64_t predictUseful = 0;

  void prefetchStreamingExperts() const noexcept;
  void evictStreamingExperts() const noexcept;

  // Where a prompt's missing experts are read, so they never evict decode's
  // cached ones: two halves of kPromptStagingExperts / 2 experts each.
  mutable Qwen4ExpLayerExpertCache promptStaging;

  [[nodiscard]] uint64_t expertCacheActualAllocatedBytes() const noexcept {
    uint64_t total = promptStaging.cacheGate.sizeBytes() +
                     promptStaging.cacheUp.sizeBytes() +
                     promptStaging.cacheDown.sizeBytes();
    for (const auto &layer : layers) {
      total += layer.expertCache.cacheGate.sizeBytes();
      total += layer.expertCache.cacheUp.sizeBytes();
      total += layer.expertCache.cacheDown.sizeBytes();
    }
    if (mtpLayer) {
      total += mtpLayer->expertCache.cacheGate.sizeBytes();
      total += mtpLayer->expertCache.cacheUp.sizeBytes();
      total += mtpLayer->expertCache.cacheDown.sizeBytes();
    }
    return total;
  }
};

// Defined in QwenTarget.cpp beside the other two, so all three share
// commonGeometry.
[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen4ExpWeights &weights);

[[nodiscard]] Qwen4ExpWeights
loadQwen4ExpWeights(metal::MetalBackend &backend,
                    const std::filesystem::path &directory,
                    Qwen4ExpLayout layout = {});

// Both of this architecture's MoE limits are now lifted. The expert width is
// expressible at kQ4ExpertStorageN = 128, the packed tiles are instantiated at
// that width, and ops::MoE routes at a padded router width rather than a fixed
// 256. What remains for this block is not a width problem.
inline constexpr uint32_t kQwen4ExpExpertCount = 512;

} // namespace splash::model
