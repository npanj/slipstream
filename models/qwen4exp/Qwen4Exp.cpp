#include <unistd.h>
#include <fcntl.h>
#include "Qwen4Exp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/sysctl.h>

namespace splash::model {
namespace {

constexpr std::string_view kNextHeadMagic = "MDFN0034";  // 8-bit logits and tiled mixer, 4-bit draft copy
constexpr std::string_view kNextEmbeddingMagic = "MDFN0013";  // 8-bit rows
constexpr std::string_view kNextNgramMagic = "MDFN0004";

// Memory macOS could hand out now: physical minus used pages, crediting
// file-backed and purgeable pages. Same estimate as
// engine::queryHostAvailableMemory; repeated here because model code does
// not link the engine.
uint64_t hostAvailableBytes() {
  uint64_t physical = 0;
  size_t length = sizeof(physical);
  if (sysctlbyname("hw.memsize", &physical, &length, nullptr, 0) != 0) return 0;
  mach_port_t host = mach_host_self();
  vm_size_t pageSize = 0;
  vm_statistics64_data_t stats{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  const bool ok = host_page_size(host, &pageSize) == KERN_SUCCESS &&
      host_statistics64(host, HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&stats), &count) == KERN_SUCCESS;
  mach_port_deallocate(mach_task_self(), host);
  if (!ok || !pageSize) return 0;
  uint64_t used = uint64_t{stats.active_count} + stats.inactive_count +
                  stats.speculative_count + stats.wire_count +
                  stats.compressor_page_count;
  const uint64_t credit = uint64_t{stats.external_page_count} + stats.purgeable_count;
  used = used > credit ? used - credit : 0;
  const uint64_t usedBytes = used * pageSize;
  return usedBytes < physical ? physical - usedBytes : 0;
}

void requireLayout(const Qwen4ExpLayout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.attentionQueryHeads || !layout.attentionKvHeads ||
      !layout.attentionHeadDimension || !layout.rotaryPairs ||
      !(layout.rotaryTheta > 0.0F) || !layout.fullAttentionPeriod ||
      !layout.experts || !layout.expertsPerToken ||
      !layout.expertIntermediateSize || !layout.expertStorageN ||
      !layout.hyperConnectionCount || !layout.hyperConnectionLowRank ||
      !layout.indexerHeads || !layout.indexerKvHeads ||
      !layout.indexerHeadDimension || !layout.ngramVocabularySize ||
      !layout.ngramEmbeddingSize || !layout.ngramHeadsPerOrder ||
      !layout.ngramShards || !layout.pleConvolutionTaps ||
      !layout.ngramVocabularyBase || !layout.ngramHeads() ||
      !layout.ngramHeadDimension()) {
    throw WeightStoreError("qwen4exp layout contains a zero dimension");
  }
  if (layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension !=
          (2 * layout.gdnKeyHeads + layout.gdnValueHeads) *
              layout.gdnHeadDimension ||
      layout.attentionWidth !=
          layout.gdnValueHeads * layout.gdnHeadDimension ||
      layout.packedFullWidth != layout.actualFullWidth() ||
      layout.packedGdnWidth < layout.actualGdnWidth() ||
      layout.packedGdnWidth % kQ4StorageN ||
      layout.expertsPerToken > layout.experts ||
      layout.ngramLayer >= layout.layers ||
      // The history kernels keep exactly two tokens: a window of three.
      layout.ngramSize != 3 ||
      !layout.q8Layout().valid() || !layout.gdnStateLayout().valid()) {
    throw WeightStoreError("qwen4exp layout is inconsistent");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
  // Experts and the indexer tile narrower; both are multiples of 128 only.
  validateQ4Layout(layout.expertIntermediateSize, layout.hiddenSize,
                   layout.expertStorageN);
  validateQ4Layout(layout.hiddenSize, layout.expertIntermediateSize,
                   layout.expertStorageN);
  validateQ4Layout(layout.indexerProjectionWidth(), layout.hiddenSize,
                   layout.expertStorageN);
  validateQ4Layout(layout.hyperConnectionWidth(), layout.ngramEmbeddingSize);
  validateQ4Layout(layout.hiddenSize, layout.ngramEmbeddingSize);
  if (layout.ngramEmbeddingSize % layout.ngramHeads() ||
      layout.ngramHeadDimension() % kQ4FineGroupElements ||
      layout.ngramVocabularySize % layout.ngramShards) {
    throw WeightStoreError("qwen4exp n-gram geometry is inconsistent");
  }
}

Qwen4ExpHyperConnection readHyperConnection(WeightFile &file,
                                            metal::MetalBackend &backend,
                                            const Qwen4ExpLayout &layout,
                                            std::string_view label,
                                            bool withInject) {
  const uint64_t width =
      checkedWeightMultiply(layout.hyperConnectionWidth(), kBFloat16Bytes,
                            "hyper-connection width bytes");
  const std::string prefix(label);
  Qwen4ExpHyperConnection result;
  result.norm = file.section(width, prefix + "-norm");
  result.mixDown = readQ8Projection(file, backend, kHyperDownPadded,
                                    layout.hyperConnectionWidth(), prefix + "-mix-down");
  result.mixUp = readQ8Projection(file, backend, layout.hyperConnectionWidth(),
                                  layout.hyperConnectionLowRank, prefix + "-mix-up");
  if (withInject) {
    result.blockInject =
        file.section(checkedWeightMultiply(layout.hyperConnectionCount, width,
                                           "hyper-connection inject bytes"),
                     prefix + "-inject");
  }
  return result;
}

Qwen4ExpIndexer readIndexer(WeightFile &file, metal::MetalBackend &backend,
                            const Qwen4ExpLayout &layout) {
  const uint64_t normBytes =
      checkedWeightMultiply(layout.indexerHeadDimension, kBFloat16Bytes,
                            "indexer norm bytes");
  ops::Q4Projection projection = readQ4Projection(
      file, backend, layout.indexerProjectionWidth(), layout.hiddenSize,
      "indexer-qk", layout.expertStorageN);
  return {projection, file.section(normBytes, "indexer-query-norm"),
          file.section(normBytes, "indexer-key-norm")};
}

void readExperts(WeightFile &file, metal::MetalBackend &backend,
                 const Qwen4ExpLayout &layout, ops::MoeWeights &ffn) {
  const uint32_t n = layout.expertStorageN;
  ffn.router = readQ8Projection(file, backend, layout.experts,
                                layout.hiddenSize, "router");
  ffn.expertGate = readExpertQ4Projection(
      file, layout.experts, layout.expertIntermediateSize, layout.hiddenSize,
      "experts-gate", n);
  ffn.expertUp = readExpertQ4Projection(
      file, layout.experts, layout.expertIntermediateSize, layout.hiddenSize,
      "experts-up", n);
  ffn.expertDown = readExpertQ4Projection(
      file, layout.experts, layout.hiddenSize, layout.expertIntermediateSize,
      "experts-down", n);
  ffn.sharedGate = readExpertQ4Projection(
      file, 1, layout.expertIntermediateSize, layout.hiddenSize,
      "shared-expert-gate", n);
  ffn.sharedUp = readExpertQ4Projection(
      file, 1, layout.expertIntermediateSize, layout.hiddenSize,
      "shared-expert-up", n);
  ffn.sharedDown = readExpertQ4Projection(
      file, 1, layout.hiddenSize, layout.expertIntermediateSize,
      "shared-expert-down", n);
  ffn.sharedExpertGate = readQ8Projection(
      file, backend, kQ4StorageN, layout.hiddenSize,
      "shared-expert-scalar-gate");
}

metal::MetalBuffer detachBuffer(metal::MetalBackend &backend, const metal::MetalBuffer &view, const char *label) {
  if (!view) return {};
  metal::MetalBuffer copy = backend.allocateBuffer(view.sizeBytes(), metal::BufferStorage::Shared, label);
  std::memcpy(copy.contents(), view.contents(), view.sizeBytes());
  return copy;
}

ops::Q4Projection detachQ4(metal::MetalBackend &backend, const ops::Q4Projection &p, const char *label) {
  return ops::Q4Projection{
      .weights = detachBuffer(backend, p.weights, (std::string(label) + "-w").c_str()),
      .scales = detachBuffer(backend, p.scales, (std::string(label) + "-s").c_str()),
      .biases = detachBuffer(backend, p.biases, (std::string(label) + "-b").c_str()),
      .outputSize = p.outputSize,
      .inputSize = p.inputSize,
  };
}

ops::Q8Projection detachQ8(metal::MetalBackend &backend, const ops::Q8Projection &p, const char *label) {
  return ops::Q8Projection{
      .weights = detachBuffer(backend, p.weights, (std::string(label) + "-w").c_str()),
      .scales = detachBuffer(backend, p.scales, (std::string(label) + "-s").c_str()),
      .biases = detachBuffer(backend, p.biases, (std::string(label) + "-b").c_str()),
      .outputSize = p.outputSize,
      .inputSize = p.inputSize,
  };
}

ops::ExpertQ4Projection detachExpert(metal::MetalBackend &backend, const ops::ExpertQ4Projection &p, const char *label) {
  return ops::ExpertQ4Projection{
      .packed = detachBuffer(backend, p.packed, label),
      .experts = p.experts,
      .outputSize = p.outputSize,
      .inputSize = p.inputSize,
      .expertStrideBytes = p.expertStrideBytes,
  };
}

[[maybe_unused]] void detachStreamingLayer(metal::MetalBackend &backend, Qwen4ExpLayerWeights &layer) {
  // 1. Attention Hyper-connection
  layer.attentionHyperConnection.norm = detachBuffer(backend, layer.attentionHyperConnection.norm, "att-norm");
  layer.attentionHyperConnection.mixDown = detachQ8(backend, layer.attentionHyperConnection.mixDown, "att-down");
  layer.attentionHyperConnection.mixUp = detachQ8(backend, layer.attentionHyperConnection.mixUp, "att-up");
  if (layer.attentionHyperConnection.blockInject) {
    layer.attentionHyperConnection.blockInject = detachBuffer(backend, *layer.attentionHyperConnection.blockInject, "att-inject");
  }

  // 2. Mixer
  if (std::holds_alternative<QwenGdnWeights>(layer.mixer)) {
    auto &gdn = std::get<QwenGdnWeights>(layer.mixer);
    gdn.inputProjectionQ8 = detachQ8(backend, gdn.inputProjectionQ8, "gdn-in");
    gdn.convolutionWeights = detachBuffer(backend, gdn.convolutionWeights, "gdn-conv");
    gdn.decay = detachBuffer(backend, gdn.decay, "gdn-decay");
    gdn.timeBias = detachBuffer(backend, gdn.timeBias, "gdn-bias");
    gdn.mixerNorm = detachBuffer(backend, gdn.mixerNorm, "gdn-norm");
    gdn.outputProjectionQ8 = detachQ8(backend, gdn.outputProjectionQ8, "gdn-out");
  } else {
    auto &attn = std::get<QwenAttentionWeights>(layer.mixer);
    attn.inputProjectionQ8 = detachQ8(backend, attn.inputProjectionQ8, "attn-in");
    attn.queryNorm = detachBuffer(backend, attn.queryNorm, "attn-qnorm");
    attn.keyNorm = detachBuffer(backend, attn.keyNorm, "attn-knorm");
    attn.outputProjectionQ8 = detachQ8(backend, attn.outputProjectionQ8, "attn-out");
  }

  // 3. Indexer
  if (layer.indexer) {
    layer.indexer->queryKeyProjection = detachQ4(backend, layer.indexer->queryKeyProjection, "idx-qk");
    layer.indexer->queryNorm = detachBuffer(backend, layer.indexer->queryNorm, "idx-qnorm");
    layer.indexer->keyNorm = detachBuffer(backend, layer.indexer->keyNorm, "idx-knorm");
  }

  // 4. MLP Hyper-connection
  layer.mlpHyperConnection.norm = detachBuffer(backend, layer.mlpHyperConnection.norm, "mlp-norm");
  layer.mlpHyperConnection.mixDown = detachQ8(backend, layer.mlpHyperConnection.mixDown, "mlp-down");
  layer.mlpHyperConnection.mixUp = detachQ8(backend, layer.mlpHyperConnection.mixUp, "mlp-up");
  if (layer.mlpHyperConnection.blockInject) {
    layer.mlpHyperConnection.blockInject = detachBuffer(backend, *layer.mlpHyperConnection.blockInject, "mlp-inject");
  }

  // 5. Router and Shared Experts
  layer.ffn.router = detachQ8(backend, layer.ffn.router, "router");
  layer.ffn.sharedGate = detachExpert(backend, layer.ffn.sharedGate, "shared-gate");
  layer.ffn.sharedUp = detachExpert(backend, layer.ffn.sharedUp, "shared-up");
  layer.ffn.sharedDown = detachExpert(backend, layer.ffn.sharedDown, "shared-down");
  layer.ffn.sharedExpertGate = detachQ8(backend, layer.ffn.sharedExpertGate, "shared-scalar");
}

} // namespace

Qwen4ExpCachePlan planQwen4ExpExpertCache(const Qwen4ExpLayout &layout,
                                          bool mtpLayer) {
  Qwen4ExpCachePlan plan;
  if (const char *envResident = getenv("SPLASH_RESIDENT_LAYERS"))
    plan.residentLayers = std::min<uint32_t>(layout.layers, std::max(0, std::atoi(envResident)));
  if (plan.residentLayers >= layout.layers) return plan;
  const uint64_t expertStride = uint64_t{layout.expertIntermediateSize} * layout.hiddenSize * 9 / 16;
  // The expert cache gets the budget llama.cpp's --moe-stream-cache uses on
  // this machine: everything but ~28 GiB, i.e. 36 GiB of 64. Measured on a
  // 64 GB M5 Pro: 16 GiB (128 per layer) 10-13 tok/s, 36 GiB 14-19. Past
  // ~38 GiB both engines swap. SPLASH_EXPERT_CACHE_GIB or _CAPACITY override.
  uint64_t physical = 0;
  size_t length = sizeof(physical);
  (void)sysctlbyname("hw.memsize", &physical, &length, nullptr, 0);
  const double physicalGiB = double(physical) / double(1ULL << 30);
  double budgetGiB = std::max(8.0, physicalGiB - 28.0);
  if (const char *envGiB = getenv("SPLASH_EXPERT_CACHE_GIB"))
    budgetGiB = std::max(1.0, std::atof(envGiB));
  // The prompt staging buffer comes out of the same budget.
  budgetGiB -= double(kPromptStagingExperts) * 3 * double(expertStride) /
               double(1ULL << 30);
  const uint64_t perSlot =
      3 * expertStride * uint64_t(layout.layers - plan.residentLayers);
  plan.capacity = static_cast<uint32_t>(std::max(0.0, std::min<double>(
      layout.experts, budgetGiB * double(1ULL << 30) / double(perSlot))));
  if (const char *envCap = getenv("SPLASH_EXPERT_CACHE_CAPACITY"))
    plan.capacity = std::max(16, std::atoi(envCap));
  plan.capacity = std::max<uint32_t>(plan.capacity, 16);
  plan.layerCapacity.assign(layout.layers + 1, plan.capacity);
  // A check step of 8 rows can route to 80 distinct experts in one layer.
  constexpr uint32_t kMinimumSlots = 96;
  // Some layers spread their routing over many more experts than others, so
  // the same memory misses less when split unevenly. This profile was fitted
  // by replaying Nitin's coding sessions (models/qwen4exp/bench/cache_plan.py)
  // and checked on the 10-prompt suite, which it never saw: SSD reads 95 ->
  // 87 a step, 39.7 -> 41.5 tok/s, same memory. SPLASH_EXPERT_SLOTS overrides;
  // SPLASH_EXPERT_SLOTS=even gives every layer the same.
  static constexpr std::array<uint16_t, 48> kSlotProfile = {
      416, 352, 368, 272, 304, 304, 336, 336, 272, 272, 304, 272, 288, 272, 240, 192,
      208, 256, 304, 240, 256, 288, 336, 272, 272, 256, 272, 224, 288, 288, 240, 208,
      208, 256, 304, 240, 240, 272, 336, 224, 224, 256, 272, 224, 240, 256, 240, 256};
  std::vector<double> weights(kSlotProfile.begin(), kSlotProfile.end());
  const char *profile = getenv("SPLASH_EXPERT_SLOTS");
  if (profile && std::string_view(profile) == "even") {
    weights.assign(layout.layers, 1.0);
  } else if (profile) {
    weights.clear();
    for (const char *cursor = profile; *cursor;) {
      char *end = nullptr;
      weights.push_back(std::strtod(cursor, &end));
      if (end == cursor)
        throw std::invalid_argument("SPLASH_EXPERT_SLOTS: expected numbers");
      cursor = *end == ',' ? end + 1 : end;
    }
  }
  if (weights.size() == layout.layers) {
    double mean = 0.0;
    for (uint32_t l = plan.residentLayers; l < layout.layers; ++l) mean += weights[l];
    mean /= double(layout.layers - plan.residentLayers);
    for (uint32_t l = plan.residentLayers; l < layout.layers; ++l)
      plan.layerCapacity[l] = std::clamp<uint32_t>(
          static_cast<uint32_t>(std::lround(plan.capacity * weights[l] / mean)),
          kMinimumSlots, layout.experts);
  } else if (profile) {
    throw std::invalid_argument("SPLASH_EXPERT_SLOTS: need one weight per layer");
  }
  uint64_t slots = kPromptStagingExperts;
  for (uint32_t l = plan.residentLayers; l < layout.layers; ++l)
    slots += plan.layerCapacity[l];
  if (mtpLayer)
    slots += plan.layerCapacity[layout.layers];
  plan.pinnedBytes = 3 * expertStride * slots;
  return plan;
}

Qwen4ExpWeights loadQwen4ExpWeights(metal::MetalBackend &backend,
                                    const std::filesystem::path &directory,
                                    Qwen4ExpLayout layout) {
  requireLayout(layout);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Qwen4ExpWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  const uint32_t residentLayers = planQwen4ExpExpertCache(layout, false).residentLayers;
  result.residentLayers = residentLayers;
  std::cerr << "[Qwen4Exp] resident layers: " << residentLayers << " / " << layout.layers << "\n";

  // One decoder layer file: the trunk's 48, and the MTP head's, which is an
  // ordinary attention layer.
  auto readLayer = [&](const std::string &filename, uint32_t layerIndex,
                       bool isFull) -> Qwen4ExpLayerWeights {
    const bool fullAttention = isFull;
    WeightFile file(backend, directory / filename, "target/" + filename,
                    Qwen4ExpLayout::layerMagic, layerIndex,
                    fullAttention ? 1U : 0U);
    if (layerIndex < residentLayers) {
      file.advise(MemoryAdvice::WillNeed);
      file.prefetch(true);
    } else {
      constexpr uint64_t kNonExpertHeaderBytes = 80ULL * 1024 * 1024;
      file.adviseRange(0, kNonExpertHeaderBytes, MemoryAdvice::WillNeed);
      if (file.bytes() > kNonExpertHeaderBytes) {
        file.adviseRange(kNonExpertHeaderBytes, file.bytes() - kNonExpertHeaderBytes,
                         MemoryAdvice::Random);
      }
    }
    Qwen4ExpLayerWeights layer;
    layer.attentionHyperConnection =
        readHyperConnection(file, backend, layout, "attention-hyper", true);
    layer.mixer =
        readQwenMixer(file, backend, layout.mixerGeometry(), fullAttention,
                      /*eightBit=*/true);
    if (fullAttention) layer.indexer = readIndexer(file, backend, layout);
    layer.mlpHyperConnection = readHyperConnection(file, backend, layout, "mlp-hyper", true);
    readExperts(file, backend, layout, layer.ffn);
    if (layerIndex >= residentLayers) {
      const uint8_t *base = file.mappedBase();
      auto offset = [&](const metal::MetalBuffer &view) {
        return uint64_t(static_cast<const uint8_t *>(view.contents()) - base);
      };
      layer.expertSource.gate = offset(layer.ffn.expertGate.packed);
      layer.expertSource.up = offset(layer.ffn.expertUp.packed);
      layer.expertSource.down = offset(layer.ffn.expertDown.packed);
      layer.expertSource.fd = ::open((directory / filename).c_str(), O_RDONLY);
      if (layer.expertSource.fd < 0)
        throw WeightStoreError("cannot open " + filename + " for expert reads");
      // Misses go around the file cache, as llama.cpp's --moe-stream-direct
      // does: the expert cache already holds them, so a second copy in the
      // file cache only crowds memory.
      if (!std::getenv("SPLASH_EXPERT_READS_CACHED"))
        (void)::fcntl(layer.expertSource.fd, F_NOCACHE, 1);
    }
    file.finish();
    if (layerIndex >= residentLayers) {
      detachStreamingLayer(backend, layer);
    }
    result.files.push_back(file.record());
    return layer;
  };
  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex)
    result.layers.push_back(readLayer("layer-" + std::to_string(layerIndex) + ".bin",
                                      layerIndex,
                                      layout.isFullAttentionLayer(layerIndex)));
  if (std::filesystem::exists(directory / "mtp-layer.bin") &&
      std::filesystem::exists(directory / "mtp-combiner.bin")) {
    result.mtpLayer = readLayer("mtp-layer.bin", layout.layers, true);
    WeightFile file(backend, directory / "mtp-combiner.bin",
                    "target/mtp-combiner.bin", "MDFN0035", layout.layers, 3);
    Qwen4ExpMtpCombiner combiner;
    const uint64_t hidden = uint64_t{layout.hiddenSize} * kBFloat16Bytes;
    combiner.embeddingNorm = detachBuffer(
        backend, file.section(hidden, "mtp-enorm"), "mtp-enorm");
    combiner.hiddenNorm = detachBuffer(
        backend, file.section(hidden * layout.hyperConnectionCount, "mtp-hnorm"),
        "mtp-hnorm");
    combiner.fcEmbedding = detachQ4(
        backend, readQ4Projection(file, backend, layout.hiddenSize,
                                  layout.hiddenSize, "mtp-fc-embedding"),
        "mtp-fc-embedding");
    combiner.fcHidden = detachQ4(
        backend, readQ4Projection(file, backend, layout.hiddenSize,
                                  layout.hiddenSize, "mtp-fc-hidden"),
        "mtp-fc-hidden");
    combiner.mixer = readHyperConnection(file, backend, layout, "mtp-mixer", false);
    combiner.mixer.norm = detachBuffer(backend, combiner.mixer.norm, "mtp-mix-norm");
    combiner.mixer.mixDown = detachQ8(backend, combiner.mixer.mixDown, "mtp-mix-down");
    combiner.mixer.mixUp = detachQ8(backend, combiner.mixer.mixUp, "mtp-mix-up");
    file.finish();
    result.files.push_back(file.record());
    result.mtpCombiner = std::move(combiner);
    std::cerr << "[Qwen4Exp] MTP draft head loaded\n";
  }

  {
    WeightFile file(backend, directory / "head.bin", "target/head.bin",
                    kNextHeadMagic, layout.layers, 2);
    file.advise(MemoryAdvice::WillNeed);
    file.prefetch(false);
    result.hyperConnectionMixer =
        readHyperConnection(file, backend, layout, "hyper-mixer", false);
    result.finalNorm = file.section(
        checkedWeightMultiply(layout.hiddenSize, kBFloat16Bytes,
                              "qwen4exp norm bytes"),
        "final-norm");
    result.logitsProjection = readQ8Projection(
        file, backend, layout.vocabularySize, layout.hiddenSize, "logits");
    result.draftLogitsProjection = readQ4Projection(
        file, backend, layout.vocabularySize, layout.hiddenSize, "draft-logits");
    file.finish();
    result.files.push_back(file.record());
  }
  // The draft head scores only the most common tokens when the package has
  // a ranking (target/draft-vocab.bin, written by dev/tools/draft_vocab.py).
  // Guesses are still checked by the full head, so this changes speed, never
  // answers. 65536 of 248320 cut the head's time 12.3 -> 8.4 ms a step and
  // kept tokens per step (32768 made it overconfident: more rejected
  // guesses). SPLASH_DRAFT_VOCAB=N overrides; 0 scores every token. Rows are
  // copied out of the 4-bit tiled layout: 256-row tiles, 64-input groups, rows.
  const auto rankingPath = directory / "draft-vocab.bin";
  const char *envVocab = getenv("SPLASH_DRAFT_VOCAB");
  const int wantedVocab = envVocab ? std::atoi(envVocab)
                                   : (std::filesystem::exists(rankingPath) ? 65536 : 0);
  if (wantedVocab > 0 && static_cast<uint32_t>(wantedVocab) < layout.vocabularySize) {
    const uint32_t tileRows = kQ4StorageN;
    const uint32_t rows = std::min<uint32_t>(
        layout.vocabularySize,
        (static_cast<uint32_t>(wantedVocab) + tileRows - 1) / tileRows * tileRows);
    std::vector<uint32_t> ids(rows);
    {
      const int fd = ::open(rankingPath.c_str(), O_RDONLY);
      const ssize_t want = static_cast<ssize_t>(rows * sizeof(uint32_t));
      const ssize_t got = fd >= 0 ? ::pread(fd, ids.data(), want, 0) : -1;
      if (fd >= 0) ::close(fd);
      if (got != want)
        throw std::runtime_error("SPLASH_DRAFT_VOCAB needs " + rankingPath.string() +
                                 " (dev/tools/draft_vocab.py) with at least " +
                                 std::to_string(rows) + " ids");
    }
    for (uint32_t id : ids)
      if (id >= layout.vocabularySize)
        throw std::runtime_error(rankingPath.string() + " holds an id past the vocabulary");
    const ops::Q4Projection &full = result.draftLogitsProjection;
    const uint32_t groups = layout.hiddenSize / 64;
    const uint64_t cells = uint64_t{rows} * groups;
    ops::Q4Projection subset{
        backend.allocateBuffer(cells * 32, metal::BufferStorage::Shared, "draft-vocab-weights"),
        backend.allocateBuffer(cells * 2, metal::BufferStorage::Shared, "draft-vocab-scales"),
        backend.allocateBuffer(cells * 2, metal::BufferStorage::Shared, "draft-vocab-biases"),
        rows, layout.hiddenSize};
    const auto *fromWeights = static_cast<const uint8_t *>(full.weights.contents());
    const auto *fromScales = static_cast<const uint16_t *>(full.scales.contents());
    const auto *fromBiases = static_cast<const uint16_t *>(full.biases.contents());
    auto *toWeights = static_cast<uint8_t *>(subset.weights.contents());
    auto *toScales = static_cast<uint16_t *>(subset.scales.contents());
    auto *toBiases = static_cast<uint16_t *>(subset.biases.contents());
    for (uint32_t row = 0; row < rows; ++row) {
      const uint32_t source = ids[row];
      for (uint32_t group = 0; group < groups; ++group) {
        const uint64_t from = (uint64_t{source / tileRows} * groups + group) * tileRows + source % tileRows;
        const uint64_t to = (uint64_t{row / tileRows} * groups + group) * tileRows + row % tileRows;
        std::memcpy(toWeights + to * 32, fromWeights + from * 32, 32);
        toScales[to] = fromScales[from];
        toBiases[to] = fromBiases[from];
      }
    }
    result.draftVocabProjection = std::move(subset);
    result.draftVocabIds = std::move(ids);
    std::cerr << "[Qwen4Exp] draft head scores the " << rows << " most common tokens\n";
  }
  {
    WeightFile file(backend, directory / "embedding.bin",
                    "target/embedding.bin", kNextEmbeddingMagic,
                    layout.vocabularySize, layout.hiddenSize);
    file.advise(MemoryAdvice::WillNeed);
    file.prefetch(false);
    // Rows of hidden bytes, then a bf16 scale and bias per 64 (the layout
    // embedding_q8 reads).
    const uint64_t elements = uint64_t{layout.vocabularySize} * layout.hiddenSize;
    result.tokenEmbedding = {
        file.section(elements, "embedding-weights"),
        file.section(elements / 32, "embedding-scales"),
        file.section(elements / 32, "embedding-biases"),
        layout.vocabularySize, layout.hiddenSize};
    file.finish();
    result.files.push_back(file.record());
  }
  {
    // The per-layer embedding, in its own file: the table is gathered per
    // token like the token embedding, so it is stored the same way, and at
    // 26.8 GiB it does not belong inside the layer it serves.
    WeightFile file(backend, directory / "ngram.bin", "target/ngram.bin",
                    kNextNgramMagic, layout.ngramShards,
                    layout.ngramHeadDimension());
    const uint64_t tableBytes = q4PackedBytes(
        layout.ngramVocabularySize, layout.ngramHeadDimension(),
        kQ4FineGroupElements);
    file.adviseRange(0, tableBytes, MemoryAdvice::Random);
    if (file.bytes() > tableBytes) {
      file.adviseRange(tableBytes, file.bytes() - tableBytes, MemoryAdvice::WillNeed);
    }
    auto &ple = result.perLayerEmbedding;
    // One row per hashed n-gram per head, a head wide, in finer groups.
    ple.table = readQ4ProjectionComponents(
        file, layout.ngramVocabularySize, layout.ngramHeadDimension(), "ngram",
        kQ4FineGroupElements);
    const uint64_t headBytes = checkedWeightMultiply(
        layout.ngramHeads(), 8, "n-gram head table bytes");
    ple.headOffsets = file.section(headBytes, "ngram-head-offsets");
    ple.headVocabularySizes =
        file.section(headBytes, "ngram-head-vocabulary-sizes");
    ple.layerMultipliers = file.section(
        checkedWeightMultiply(layout.ngramSize, 8,
                              "n-gram layer multiplier bytes"),
        "ngram-layer-multipliers");
    ple.keyProjection = readQ4Projection(
        file, backend, layout.hyperConnectionWidth(),
        layout.ngramEmbeddingSize, "ple-key");
    ple.valueProjection = readQ4Projection(
        file, backend, layout.hiddenSize, layout.ngramEmbeddingSize,
        "ple-value");
    const uint64_t width = checkedWeightMultiply(
        layout.hyperConnectionWidth(), kBFloat16Bytes, "PLE norm bytes");
    ple.keyNorm = file.section(width, "ple-key-norm");
    ple.queryNorm = file.section(width, "ple-query-norm");
    ple.convolutionNorm = file.section(width, "ple-conv-norm");
    // Depthwise, so one row of taps per channel.
    ple.convolutionWeights = file.section(
        checkedWeightMultiply(width, layout.pleConvolutionTaps,
                              "PLE convolution bytes"),
        "ple-convolution");
    // Everything but the table leaves the file's buffer. A kernel that binds
    // any view of it makes the GPU keep all 32 GB resident, which cost
    // seconds per step; the table itself is gathered on the CPU instead.
    ple.keyProjection = detachQ4(backend, ple.keyProjection, "ple-key");
    ple.valueProjection = detachQ4(backend, ple.valueProjection, "ple-value");
    ple.keyNorm = detachBuffer(backend, ple.keyNorm, "ple-key-norm");
    ple.queryNorm = detachBuffer(backend, ple.queryNorm, "ple-query-norm");
    ple.convolutionNorm = detachBuffer(backend, ple.convolutionNorm, "ple-conv-norm");
    ple.convolutionWeights =
        detachBuffer(backend, ple.convolutionWeights, "ple-convolution");
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);

  {
    // Partial sums of the split down projection: splits x rows x outputs.
    const metal::MetalBuffer partials = backend.allocateBuffer(
        uint64_t{kHyperDownSplits} * kHyperDownSplitRows *
            (layout.hyperConnectionLowRank + layout.hyperConnectionCount) * sizeof(float),
        metal::BufferStorage::Private, "qwen4exp-hyper-partials");
    // Prompt-path scratch: rows padded to whole 32-row tiles.
    const uint64_t rows = (ExecutionLimits::prefillTokenBudget + 31) / 32 * 32;
    const uint64_t width = layout.hyperConnectionWidth();
    const metal::MetalBuffer sums = backend.allocateBuffer(
        rows * (width / 64) * sizeof(float), metal::BufferStorage::Private,
        "qwen4exp-hyper-prompt-sums");
    const metal::MetalBuffer down = backend.allocateBuffer(
        rows * kHyperDownPadded * 2, metal::BufferStorage::Private,
        "qwen4exp-hyper-prompt-down");
    const metal::MetalBuffer logits = backend.allocateBuffer(
        rows * width * 2, metal::BufferStorage::Private, "qwen4exp-hyper-prompt-logits");
    auto share = [&](Qwen4ExpHyperConnection &hc) {
      hc.downPartials = partials;
      hc.promptSums = sums;
      hc.promptDown = down;
      hc.promptLogits = logits;
    };
    for (auto &layer : result.layers) {
      share(layer.attentionHyperConnection);
      share(layer.mlpHyperConnection);
    }
    if (result.mtpLayer) {
      share(result.mtpLayer->attentionHyperConnection);
      share(result.mtpLayer->mlpHyperConnection);
    }
    if (result.mtpCombiner)
      share(result.mtpCombiner->mixer);
    share(result.hyperConnectionMixer);
  }
  result.lastSelectedExperts.resize(layout.layers);
  if (residentLayers < layout.layers) {
    const uint64_t expertStride = uint64_t{layout.expertIntermediateSize} * layout.hiddenSize * 9 / 16;
    const Qwen4ExpCachePlan cachePlan =
        planQwen4ExpExpertCache(layout, static_cast<bool>(result.mtpLayer));
    const uint32_t cacheCapacity = cachePlan.capacity;
    auto capacityOf = [&](uint32_t l) { return cachePlan.layerCapacity[std::min(l, layout.layers)]; };
    // The cache is pinned: macOS cannot page it out. If it does not fit in
    // free memory, the machine freezes until its watchdog restarts it (twice
    // on 2026-09-22: two engines at once, then one asked for 60 GiB). The
    // margin is small on purpose: the default 34 GiB cache normally leaves
    // ~4 GiB, and the runtime governor guards what follows; a second engine
    // finds ~4 GiB against 35 needed and stops here.
    {
      uint64_t physical = 0;
      size_t length = sizeof(physical);
      (void)sysctlbyname("hw.memsize", &physical, &length, nullptr, 0);
      const uint64_t reserve = physical / 32;
      const uint64_t available = hostAvailableBytes();
      constexpr double kGiB = double(1ULL << 30);
      std::cerr << "[Qwen4Exp] expert cache: " << cacheCapacity
                << " experts per layer on average, pins " << double(cachePlan.pinnedBytes) / kGiB
                << " GiB; free now " << double(available) / kGiB << " GiB\n";
      if (available < cachePlan.pinnedBytes + reserve) {
        std::ostringstream message;
        message << "expert cache needs " << double(cachePlan.pinnedBytes) / kGiB
                << " GiB plus " << double(reserve) / kGiB
                << " GiB kept free for macOS, but only " << double(available) / kGiB
                << " GiB is free. Is another model running? Otherwise lower "
                   "SPLASH_EXPERT_CACHE_GIB.";
        throw std::runtime_error(message.str());
      }
    }
    {
      auto &staging = result.promptStaging;
      staging.capacity = kPromptStagingExperts;
      const uint64_t bytes = uint64_t{kPromptStagingExperts} * expertStride;
      staging.cacheGate = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "prompt-staging-gate");
      staging.cacheUp = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "prompt-staging-up");
      staging.cacheDown = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "prompt-staging-down");
    }
    auto streamingLayer = [&](uint32_t l) -> Qwen4ExpLayerWeights & {
      return l < layout.layers ? result.layers[l] : *result.mtpLayer;
    };
    const uint32_t cachedLayers = layout.layers + (result.mtpLayer ? 1 : 0);
    for (uint32_t l = residentLayers; l < cachedLayers; ++l) {
      auto &cache = streamingLayer(l).expertCache;
      cache.capacity = capacityOf(l);
      const uint64_t cacheBytes = uint64_t{cache.capacity} * expertStride;
      cache.cacheGate = backend.allocateBuffer(cacheBytes, metal::BufferStorage::Shared, "layer-expert-gate");
      cache.cacheUp = backend.allocateBuffer(cacheBytes, metal::BufferStorage::Shared, "layer-expert-up");
      cache.cacheDown = backend.allocateBuffer(cacheBytes, metal::BufferStorage::Shared, "layer-expert-down");
      cache.expertToSlot.assign(layout.experts, -1);
      cache.slotToExpert.assign(cache.capacity, -1);
      cache.lruTime.assign(cache.capacity, 0);
      cache.numCached = 0;
      cache.clock = 0;
    }
    const uint32_t streamingLayersCount = cachedLayers - residentLayers;
    // Loading experts 0..N at startup is an arbitrary guess that costs a read
    // of the whole cache; the first request fills it with the right ones.
    uint32_t prewarmCount = 0;
    if (const char *envPre = getenv("SPLASH_PREWARM_EXPERTS")) {
      prewarmCount = std::atoi(envPre);
    }
    dispatch_apply(streamingLayersCount, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t idx) {
      const uint32_t l = residentLayers + static_cast<uint32_t>(idx);
      auto &cache = streamingLayer(l).expertCache;
      const auto &layer = streamingLayer(l);
      const uint64_t cacheBytes = uint64_t{cache.capacity} * expertStride;
      const uint32_t prewarm = std::min(cache.capacity, prewarmCount);
      char *cg = static_cast<char *>(cache.cacheGate.contents());
      char *cu = static_cast<char *>(cache.cacheUp.contents());
      char *cd = static_cast<char *>(cache.cacheDown.contents());
      const char *fg = static_cast<const char *>(layer.ffn.expertGate.packed.contents());
      const char *fu = static_cast<const char *>(layer.ffn.expertUp.packed.contents());
      const char *fd = static_cast<const char *>(layer.ffn.expertDown.packed.contents());
      // Pre-fault all pages across the cache buffers to eliminate GPU first-dispatch MMU faulting
      for (uint64_t off = 0; off < cacheBytes; off += 16384) {
        cg[off] = 0;
        cu[off] = 0;
        cd[off] = 0;
      }
      // Pin the cache, as llama.cpp's is. Pageable, macOS compresses it under
      // pressure, and every hit then pays to decompress: at a 36 GiB cache
      // decode stalled for minutes with ~28 GB in the compressor. Pinned, the
      // file cache is what gives way instead.
      if (!getenv("SPLASH_EXPERT_CACHE_UNPINNED")) {
        if (::mlock(cg, cacheBytes) || ::mlock(cu, cacheBytes) ||
            ::mlock(cd, cacheBytes))
          std::cerr << "[Qwen4Exp] could not pin expert cache of layer " << l
                    << "; it may be compressed under memory pressure\n";
      }
      for (uint32_t s = 0; s < prewarm; ++s) {
        cache.slotToExpert[s] = s;
        cache.expertToSlot[s] = s;
        cache.lruTime[s] = 1;
        std::memcpy(cg + uint64_t{s} * expertStride, fg + uint64_t{s} * expertStride, expertStride);
        std::memcpy(cu + uint64_t{s} * expertStride, fu + uint64_t{s} * expertStride, expertStride);
        std::memcpy(cd + uint64_t{s} * expertStride, fd + uint64_t{s} * expertStride, expertStride);
      }
      cache.numCached = prewarm;
      cache.clock = 1;
    });
    // Keep every expert cache resident on the GPU. Otherwise the driver maps
    // each layer's cache on first use - ~14 ms a layer, ~0.7 s for a step -
    // and again after a long prompt, streamed through whole layer files,
    // evicted them.
    {
      std::vector<metal::MetalBuffer> resident;
      for (uint32_t l = residentLayers; l < cachedLayers; ++l) {
        const auto &cache = streamingLayer(l).expertCache;
        resident.insert(resident.end(), {cache.cacheGate, cache.cacheUp, cache.cacheDown});
      }
      const auto &staging = result.promptStaging;
      resident.insert(resident.end(), {staging.cacheGate, staging.cacheUp, staging.cacheDown});
      for (const auto &buffer : {staging.cacheGate, staging.cacheUp, staging.cacheDown})
        (void)::mlock(buffer.contents(), buffer.sizeBytes());
      if (!getenv("SPLASH_NO_RESIDENCY_SET"))
        backend.keepResident(resident);
    }
    result.streamingCacheCapacity = cacheCapacity;
    result.streamingCacheGate = result.layers[residentLayers].expertCache.cacheGate;
    result.streamingCacheUp = result.layers[residentLayers].expertCache.cacheUp;
    result.streamingCacheDown = result.layers[residentLayers].expertCache.cacheDown;
  }

  return result;
}

void Qwen4ExpWeights::prefetchStreamingExperts() const noexcept {
  for (uint32_t l = residentLayers; l < layers.size(); ++l) {
    if (l >= lastSelectedExperts.size() || lastSelectedExperts[l].empty())
      continue;
    const auto &ffn = layers[l].ffn;
    const uint64_t stride = ffn.expertGate.expertStrideBytes;
    if (!stride) continue;
    char *gBase = static_cast<char *>(ffn.expertGate.packed.contents());
    char *uBase = static_cast<char *>(ffn.expertUp.packed.contents());
    char *dBase = static_cast<char *>(ffn.expertDown.packed.contents());
    for (uint32_t exp : lastSelectedExperts[l]) {
      if (gBase) (void)madvise(gBase + uint64_t{exp} * stride, stride, MADV_WILLNEED);
      if (uBase) (void)madvise(uBase + uint64_t{exp} * stride, stride, MADV_WILLNEED);
      if (dBase) (void)madvise(dBase + uint64_t{exp} * stride, stride, MADV_WILLNEED);
    }
  }
}

void Qwen4ExpWeights::evictStreamingExperts() const noexcept {
  for (uint32_t l = residentLayers; l < layers.size(); ++l) {
    const auto &ffn = layers[l].ffn;
    if (void *ptr = ffn.expertGate.packed.contents()) {
      madvise(ptr, ffn.expertGate.packed.sizeBytes(), MADV_DONTNEED);
    }
    if (void *ptr = ffn.expertUp.packed.contents()) {
      madvise(ptr, ffn.expertUp.packed.sizeBytes(), MADV_DONTNEED);
    }
    if (void *ptr = ffn.expertDown.packed.contents()) {
      madvise(ptr, ffn.expertDown.packed.sizeBytes(), MADV_DONTNEED);
    }
  }
  if (mtpLayer && residentLayers <= layers.size()) {
    const auto &ffn = mtpLayer->ffn;
    if (void *ptr = ffn.expertGate.packed.contents()) {
      madvise(ptr, ffn.expertGate.packed.sizeBytes(), MADV_DONTNEED);
    }
    if (void *ptr = ffn.expertUp.packed.contents()) {
      madvise(ptr, ffn.expertUp.packed.sizeBytes(), MADV_DONTNEED);
    }
    if (void *ptr = ffn.expertDown.packed.contents()) {
      madvise(ptr, ffn.expertDown.packed.sizeBytes(), MADV_DONTNEED);
    }
  }
}

} // namespace splash::model
