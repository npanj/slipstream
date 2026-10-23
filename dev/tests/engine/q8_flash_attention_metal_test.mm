#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ops/PagedAttention.hpp"
#include "Q8PageFormatReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash::kv;

namespace {

constexpr uint32_t kStride = 32;
constexpr uint32_t kQueryStride = kStride;
constexpr uint32_t kRows = kQ8VerifyMaximumRows;

// The two production GQA geometries. The group size selects the kernel
// specialization; the suffix names its pipelines.
struct Shape {
  uint32_t kvHeads;
  uint32_t queryHeadsPerKvHead;
  const char *suffix;
  uint32_t queryHeads() const { return kvHeads * queryHeadsPerKvHead; }
  uint32_t fusedRows() const { return kRows * queryHeadsPerKvHead; }
  Q8Layout layout() const { return {1, kvHeads, kHeadDimension}; }
};
// qwen4exp is the last: two KV heads of twelve query heads each, where a
// fused row group no longer fits one pass of the 256-thread softmax.
constexpr std::array<Shape, 3> kShapes{
    {{4, 6, ""}, {2, 8, "_kv2_g8"}, {2, 12, "_kv2_g12"}}};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void testContract() {
  Q8VerifyAttentionParams finalCycle{
      splash::kv::kMaximumLogicalTokens - 1,
      splash::kv::kQ8VerifyMaximumRows,
      32,
      (splash::kv::kMaximumPhysicalTokens + kPageTokens - 1) / kPageTokens,
      (splash::kv::kMaximumPhysicalTokens + kPageTokens - 1) / kPageTokens,
      kQ8VerifySplits,
      kQ8VerifySplits,
      0};
  require(q8VerifyAttentionValidationError(finalCycle).empty(),
          "final fixed-eight verification rows exceeded physical KV scratch");
  Q8VerifyAttentionParams scaled = finalCycle;
  scaled.split_count = kQ8VerifyMaximumSplits;
  scaled.slot_splits = kQ8VerifyMaximumSplits;
  require(q8VerifyAttentionValidationError(scaled).empty(),
          "maximum verify split count was rejected");
  ++scaled.split_count;
  ++scaled.slot_splits;
  require(q8VerifyAttentionValidationError(scaled) == "split_count_invalid",
          "split count above the workspace maximum was accepted");
  scaled = finalCycle;
  scaled.slot_splits = scaled.split_count - 1;
  require(q8VerifyAttentionValidationError(scaled) == "slot_splits_invalid",
          "slot stride below the lane's split count was accepted");
  require(q8VerifyAttentionSplits(kQ8VerifySplits, 0, 8) == kQ8VerifySplits &&
              q8VerifyAttentionSplits(kQ8VerifySplits, 16 * 1024, 8) ==
                  kQ8VerifySplits + 1 &&
              q8VerifyAttentionSplits(kQ8VerifySplits, 131072, 8) ==
                  kQ8VerifyMaximumSplits &&
              q8VerifyAttentionSplits(1, 0, 8) == 1,
          "verify split scaling departed from one split per 16 visible pages");
  ++finalCycle.committed_tokens;
  require(q8VerifyAttentionValidationError(finalCycle) ==
              "context_out_of_range",
          "physical KV scratch exceeded its fixed seven-row allowance");
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, uint64_t bytes) {
  id<MTLBuffer> result =
      [device newBufferWithLength:std::max<uint64_t>(bytes, 1)
                          options:MTLResourceStorageModeShared];
  if (!result)
    throw std::runtime_error("Metal buffer allocation failed");
  std::memset(result.contents, 0, result.length);
  return result;
}

id<MTLComputePipelineState>
makePipeline(id<MTLDevice> device, id<MTLLibrary> library,
             const std::string &name) {
  id<MTLFunction> function =
      [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!function)
    throw std::runtime_error("missing Metal kernel: " + name);
  NSError *error = nil;
  id<MTLComputePipelineState> result =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!result)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return result;
}

// Shader validation (MTL_SHADER_VALIDATION=1, how make test-engine runs this
// test) instruments every pipeline's threadgroup memory (2x-12 bytes for
// these tiles), so the printed lengths are the kernels' own only when it is
// off. The checks below compare pipelines with each other, which holds
// either way; an absolute bound would only ever be checked uninstrumented.
bool shaderValidationEnabled() {
  const char *value = std::getenv("MTL_SHADER_VALIDATION");
  return value && std::strcmp(value, "0") != 0;
}

void finish(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::string message = "Metal command failed";
    if (command.error) {
      message += ": ";
      message += command.error.localizedDescription.UTF8String;
    }
    throw std::runtime_error(message);
  }
}

float keyPattern(uint32_t token, uint32_t head, uint32_t dimension) {
  int32_t centered =
      int32_t((uint64_t{token} * 37 + head * 101 + dimension * 17 +
               uint64_t{token} * dimension * 3) %
              2003) -
      1001;
  return float(centered) / 2002.0f;
}

float valuePattern(uint32_t token, uint32_t head, uint32_t dimension) {
  int32_t centered =
      int32_t((uint64_t{token} * 53 + head * 79 + dimension * 29 +
               uint64_t{token} * dimension * 5) %
              2011) -
      1005;
  return float(centered) / 1005.0f;
}

float queryPattern(uint32_t row, uint32_t head, uint32_t dimension) {
  int32_t centered = int32_t((uint64_t{row} * 43 + head * 67 + dimension * 11 +
                              uint64_t{head} * dimension * 7) %
                             1019) -
                     509;
  return float(centered) / 2036.0f;
}

struct Case {
  Shape shape;
  Q8VerifyAttentionParams params;
  std::vector<uint32_t> pageTable;
  id<MTLBuffer> pageTableBuffer;
  id<MTLBuffer> q8Keys;
  id<MTLBuffer> keyScales;
  id<MTLBuffer> q8Values;
  id<MTLBuffer> valueScales;
  id<MTLBuffer> queries;

  uint64_t queryIndex(uint32_t head, uint32_t row, uint32_t dimension) const {
    const uint32_t kvHead = head / shape.queryHeadsPerKvHead;
    const uint32_t localHead = head % shape.queryHeadsPerKvHead;
    return ((uint64_t{kvHead} * kQueryStride + row) * shape.queryHeadsPerKvHead +
            localHead) *
               kHeadDimension +
           dimension;
  }
};

Case makeCase(id<MTLDevice> device, Shape shape, uint32_t committed,
              uint32_t activeRows, uint32_t splits) {
  Case result;
  result.shape = shape;
  const Q8Layout layout = shape.layout();
  uint32_t pages = (committed + activeRows + kPageTokens - 1) / kPageTokens;
  uint32_t physicalPages = pages * 2 + 1;
  result.params = {committed,     activeRows, kStride, pages,
                   physicalPages, splits,     splits,  0};
  result.pageTable.resize(pages);
  for (uint32_t logical = 0; logical < pages; ++logical)
    result.pageTable[logical] = (logical * 2 + 1) % physicalPages;
  result.pageTableBuffer = makeBuffer(device, pages * sizeof(uint32_t));
  std::memcpy(result.pageTableBuffer.contents, result.pageTable.data(),
              pages * sizeof(uint32_t));
  result.q8Keys =
      makeBuffer(device, uint64_t{physicalPages} * layout.dataBytesPerLayerPage());
  result.keyScales =
      makeBuffer(device, uint64_t{physicalPages} * layout.scaleBytesPerLayerPage());
  result.q8Values =
      makeBuffer(device, uint64_t{physicalPages} * layout.dataBytesPerLayerPage());
  result.valueScales =
      makeBuffer(device, uint64_t{physicalPages} * layout.scaleBytesPerLayerPage());
  uint64_t queryElements =
      uint64_t{shape.queryHeads()} * kQueryStride * kHeadDimension;
  result.queries = makeBuffer(device, queryElements * sizeof(BFloat16Bits));
  return result;
}

// The page format is head-major inside a page, so a kv2_g8 page is the first
// two heads of the four-head oracle page: quantize through the shared oracle
// and store the layout's own head count.
void fill(Case &data) {
  const Shape shape = data.shape;
  const Q8Layout layout = shape.layout();
  auto *storedKeys = static_cast<int8_t *>(data.q8Keys.contents);
  auto *storedKeyScales = static_cast<float *>(data.keyScales.contents);
  auto *storedValues = static_cast<int8_t *>(data.q8Values.contents);
  auto *storedValueScales = static_cast<float *>(data.valueScales.contents);
  std::vector<float> pageKeys;
  std::vector<float> pageValues;
  auto quantized = std::make_unique<Q8LayerPage>();
  uint32_t visibleTokens = data.params.committed_tokens + data.params.active_rows;
  uint32_t visiblePages = (visibleTokens + kPageTokens - 1) / kPageTokens;
  for (uint32_t logicalPage = 0; logicalPage < visiblePages; ++logicalPage) {
    uint32_t valid =
        std::min(kPageTokens, visibleTokens - logicalPage * kPageTokens);
    pageKeys.assign(uint64_t{valid} * kKvHeads * kHeadDimension, 0.0f);
    pageValues.assign(uint64_t{valid} * kKvHeads * kHeadDimension, 0.0f);
    for (uint32_t token = 0; token < valid; ++token) {
      uint32_t global = logicalPage * kPageTokens + token;
      for (uint32_t head = 0; head < shape.kvHeads; ++head) {
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          uint64_t index = logicalIndex(token, head, dimension);
          pageKeys[index] = bfloat16ToFloat(
              floatToBFloat16(keyPattern(global, head, dimension)));
          pageValues[index] = bfloat16ToFloat(
              floatToBFloat16(valuePattern(global, head, dimension)));
        }
      }
    }
    quantizeLayerPage(pageKeys, pageValues, valid, *quantized);
    uint32_t physical = data.pageTable[logicalPage];
    const uint64_t elements = layout.elementsPerLayerPage();
    const uint64_t scales = layout.scalesPerTensorLayerPage();
    std::copy_n(quantized->keys.begin(), elements,
                storedKeys + uint64_t{physical} * elements);
    std::copy_n(quantized->keyScales.begin(), scales,
                storedKeyScales + uint64_t{physical} * scales);
    std::copy_n(quantized->values.begin(), elements,
                storedValues + uint64_t{physical} * elements);
    std::copy_n(quantized->valueScales.begin(), scales,
                storedValueScales + uint64_t{physical} * scales);
  }

  auto *queries = static_cast<BFloat16Bits *>(data.queries.contents);
  for (uint32_t head = 0; head < shape.queryHeads(); ++head)
    for (uint32_t row = 0; row < data.params.active_rows; ++row)
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
        queries[data.queryIndex(head, row, dimension)] =
            floatToBFloat16(queryPattern(row, head, dimension));
}

float loadKey(const Case &data, uint32_t token, uint32_t head,
              uint32_t dimension) {
  const Q8Layout layout = data.shape.layout();
  uint32_t physical = data.pageTable[token / kPageTokens];
  uint32_t pageToken = token % kPageTokens;
  float scale = static_cast<const float *>(
      data.keyScales.contents)[uint64_t{physical} * layout.scalesPerTensorLayerPage() +
                               keyScaleIndex(head, pageToken)];
  return float(static_cast<const int8_t *>(
             data.q8Keys.contents)[uint64_t{physical} * layout.elementsPerLayerPage() +
                                   keyDataIndex(head, pageToken, dimension)]) *
         scale;
}

float loadValue(const Case &data, uint32_t token, uint32_t head,
                uint32_t dimension) {
  const Q8Layout layout = data.shape.layout();
  uint32_t physical = data.pageTable[token / kPageTokens];
  uint32_t pageToken = token % kPageTokens;
  float scale = static_cast<const float *>(
      data.valueScales
          .contents)[uint64_t{physical} * layout.scalesPerTensorLayerPage() +
                     valueScaleIndex(head, pageToken)];
  return float(static_cast<const int8_t *>(
             data.q8Values
                 .contents)[uint64_t{physical} * layout.elementsPerLayerPage() +
                            valueDataIndex(head, pageToken, dimension)]) *
         scale;
}

// Double-precision softmax attention over stored Q8 pages or the unquantized
// BF16 pattern, with rounding to BF16 only at the final output.
std::vector<BFloat16Bits> cpuReference(const Case &data, bool quantized) {
  const Shape shape = data.shape;
  std::vector<BFloat16Bits> output(
      uint64_t{shape.queryHeads()} * kQueryStride * kHeadDimension,
      BFloat16Bits{0});
  const auto *queries =
      static_cast<const BFloat16Bits *>(data.queries.contents);
  for (uint32_t queryHead = 0; queryHead < shape.queryHeads(); ++queryHead) {
    uint32_t kvHead = queryHead / shape.queryHeadsPerKvHead;
    for (uint32_t row = 0; row < data.params.active_rows; ++row) {
      uint32_t visible = data.params.committed_tokens + row + 1;
      std::vector<double> weights(visible);
      double maximum = -std::numeric_limits<double>::infinity();
      for (uint32_t token = 0; token < visible; ++token) {
        double score = 0.0;
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const float query = bfloat16ToFloat(
              queries[data.queryIndex(queryHead, row, dimension)]);
          const float key =
              quantized
                  ? loadKey(data, token, kvHead, dimension)
                  : bfloat16ToFloat(floatToBFloat16(
                        keyPattern(token, kvHead, dimension)));
          score += double(query) * key;
        }
        weights[token] = score * 0.0625;
        maximum = std::max(maximum, weights[token]);
      }
      double denominator = 0.0;
      for (double &weight : weights) {
        weight = std::exp(weight - maximum);
        denominator += weight;
      }
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        double value = 0.0;
        for (uint32_t token = 0; token < visible; ++token) {
          const float storedValue =
              quantized
                  ? loadValue(data, token, kvHead, dimension)
                  : bfloat16ToFloat(floatToBFloat16(
                        valuePattern(token, kvHead, dimension)));
          value += weights[token] * storedValue;
        }
        output[data.queryIndex(queryHead, row, dimension)] =
            floatToBFloat16(float(value / denominator));
      }
    }
  }
  return output;
}

// One scale placement of one geometry, with the shared reduction.
struct Pipelines {
  std::string splitName;
  id<MTLComputePipelineState> split;
  id<MTLComputePipelineState> reduce;
};

Pipelines makePipelines(id<MTLDevice> device, id<MTLLibrary> library,
                        Shape shape, bool cooperativeScale) {
  const std::string variant =
      std::string(cooperativeScale ? "_cooperative_scale" : "") + shape.suffix;
  Pipelines result;
  result.splitName = "verify_attention_q8_split" + variant;
  result.split = makePipeline(device, library, result.splitName);
  result.reduce = makePipeline(
      device, library, std::string("verify_attention_q8_reduce") + shape.suffix);
  const uint64_t scratch = result.split.staticThreadgroupMemoryLength;
  std::cout << "pipeline=" << result.splitName << " threadgroup_bytes=" << scratch
            << (shaderValidationEnabled() ? " (instrumented by shader validation)" : "")
            << '\n';
  require(scratch == makePipeline(device, library,
                                  "prefill_attention_q8_split" + variant)
                         .staticThreadgroupMemoryLength,
          "verify tile left the shared score/probability footprint");
  return result;
}

// Instrumented, the kv2_g12 tile (19,600 bytes) grows to 39,188, past a 32 KB
// device limit. Metal still builds the pipeline, but its dispatches overrun
// threadgroup memory and return garbage that changes run to run, so a shape
// that only exceeds the limit under shader validation is checked
// uninstrumented. Uninstrumented, exceeding it is a kernel defect.
bool fitsThreadgroupMemory(id<MTLDevice> device,
                           const std::array<Pipelines, 2> &pipelines) {
  const uint64_t limit = device.maxThreadgroupMemoryLength;
  bool fits = true;
  for (const Pipelines &placement : pipelines) {
    const uint64_t scratch = placement.split.staticThreadgroupMemoryLength;
    if (scratch <= limit)
      continue;
    require(shaderValidationEnabled(),
            "verify tile exceeds the device's threadgroup memory");
    std::cout << "skipping " << placement.splitName << ": shader validation "
              << "instruments it to " << scratch << " threadgroup bytes, past "
              << "the device's " << limit << "; run without "
              << "MTL_SHADER_VALIDATION to check it\n";
    fits = false;
  }
  return fits;
}

// The split, reduce and every scratch buffer of one production-sized verify
// dispatch: partials and statistics exactly cover [lane][KV head][split].
struct Dispatch {
  std::vector<uint8_t> output;
  std::vector<uint8_t> partials;
  std::vector<uint8_t> statistics;
};

std::vector<uint8_t> copyOf(id<MTLBuffer> buffer) {
  const auto *bytes = static_cast<const uint8_t *>(buffer.contents);
  return {bytes, bytes + buffer.length};
}

Dispatch dispatch(id<MTLDevice> device, id<MTLCommandQueue> queue,
                  id<MTLComputePipelineState> split,
                  id<MTLComputePipelineState> reduce, const Case &data,
                  uint32_t width) {
  const Shape shape = data.shape;
  const uint32_t splits = data.params.split_count;
  const uint64_t laneBytes = data.queries.length;
  id<MTLBuffer> queries = makeBuffer(device, width * laneBytes);
  id<MTLBuffer> output = makeBuffer(device, width * laneBytes);
  const uint64_t slots = uint64_t{width} * shape.kvHeads * splits;
  id<MTLBuffer> partials = makeBuffer(
      device, slots * shape.fusedRows() * kHeadDimension * sizeof(float));
  id<MTLBuffer> statistics =
      makeBuffer(device, slots * shape.fusedRows() * 2 * sizeof(float));
  for (uint32_t lane = 0; lane < width; ++lane) {
    std::memcpy(static_cast<uint8_t *>(queries.contents) + lane * laneBytes,
                data.queries.contents, laneBytes);
  }
  std::array<Q8VerifyAttentionParams, 4> params{};
  params.fill(data.params);
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:split];
  [encoder setBuffer:queries offset:0 atIndex:0];
  [encoder setBuffer:data.q8Keys offset:0 atIndex:1];
  [encoder setBuffer:data.keyScales offset:0 atIndex:2];
  [encoder setBuffer:data.q8Values offset:0 atIndex:3];
  [encoder setBuffer:data.valueScales offset:0 atIndex:4];
  [encoder setBuffer:partials offset:0 atIndex:5];
  [encoder setBuffer:statistics offset:0 atIndex:6];
  for (uint32_t index = 7; index < 11; ++index)
    [encoder setBuffer:data.pageTableBuffer offset:0 atIndex:index];
  [encoder setBytes:params.data()
              length:sizeof(Q8VerifyAttentionParams) * params.size()
             atIndex:11];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, splits, width)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:partials offset:0 atIndex:0];
  [encoder setBuffer:statistics offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:params.data()
              length:sizeof(Q8VerifyAttentionParams) * params.size()
             atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, shape.fusedRows(), width)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  finish(command);
  return {copyOf(output), copyOf(partials), copyOf(statistics)};
}

// The BF16 quality gate measures Q8 quantization fidelity on the synthetic
// pattern and is calibrated for short histories; long-history cases exercise
// the split geometry against the Q8 reference only.
void checkOutput(const Case &data, uint32_t width,
                 const std::vector<BFloat16Bits> &expectedQ8,
                 const std::vector<BFloat16Bits> &expectedBf16,
                 const std::vector<uint8_t> &outputBytes, bool qualityGate,
                 const std::string &label) {
  const Shape shape = data.shape;
  const uint32_t activeRows = data.params.active_rows;
  const auto *actual =
      reinterpret_cast<const BFloat16Bits *>(outputBytes.data());
  const uint64_t laneElements = expectedQ8.size();
  require(outputBytes.size() == width * laneElements * sizeof(BFloat16Bits),
          "verify output size departed from its lanes");
  double dot = 0.0;
  double actualSquared = 0.0;
  double expectedSquared = 0.0;
  float maximumAbsolute = 0.0f;
  double qualityDot = 0.0;
  double qualityActualSquared = 0.0;
  double qualityBf16Squared = 0.0;
  float qualityMaximumAbsolute = 0.0f;
  uint32_t top1Matches = 0;
  uint32_t top1Rows = 0;
  for (uint32_t lane = 0; lane < width; ++lane) {
    for (uint32_t head = 0; head < shape.queryHeads(); ++head) {
      for (uint32_t row = 0; row < activeRows; ++row) {
        uint32_t actualTop = 0;
        float actualTopValue = -std::numeric_limits<float>::infinity();
        float bf16TopValue = -std::numeric_limits<float>::infinity();
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const uint64_t local = data.queryIndex(head, row, dimension);
          const uint64_t index = uint64_t{lane} * laneElements + local;
          const float observed = bfloat16ToFloat(actual[index]);
          const float q8Reference = bfloat16ToFloat(expectedQ8[local]);
          const float bf16Reference = bfloat16ToFloat(expectedBf16[local]);
          maximumAbsolute =
              std::max(maximumAbsolute, std::abs(observed - q8Reference));
          dot += double(observed) * q8Reference;
          actualSquared += double(observed) * observed;
          expectedSquared += double(q8Reference) * q8Reference;
          qualityMaximumAbsolute = std::max(
              qualityMaximumAbsolute, std::abs(observed - bf16Reference));
          qualityDot += double(observed) * bf16Reference;
          qualityActualSquared += double(observed) * observed;
          qualityBf16Squared += double(bf16Reference) * bf16Reference;
          if (observed > actualTopValue) {
            actualTopValue = observed;
            actualTop = dimension;
          }
          bf16TopValue = std::max(bf16TopValue, bf16Reference);
        }
        // The reference can hold an exact bf16 tie between dimensions; any
        // dimension at its maximum is the row's top-1.
        top1Matches += bfloat16ToFloat(expectedBf16[data.queryIndex(
                           head, row, actualTop)]) == bf16TopValue;
        ++top1Rows;
      }
    }
  }
  double cosine = dot / std::sqrt(actualSquared * expectedSquared);
  const double qualityCosine =
      qualityDot / std::sqrt(qualityActualSquared * qualityBf16Squared);
  const double top1Agreement = double(top1Matches) / top1Rows;
  std::cout << label << " verify_batch_width=" << width
            << " active_rows=" << activeRows
            << " committed=" << data.params.committed_tokens
            << " splits=" << data.params.split_count << " q8_cosine=" << cosine
            << " bf16_cosine=" << qualityCosine
            << " bf16_max_absolute=" << qualityMaximumAbsolute
            << " bf16_top1=" << top1Agreement << '\n';
  require(cosine > 0.9995,
          "production batch attention differs from its Q8 reference");
  require(maximumAbsolute < 0.02f,
          "production batch attention exceeds its Q8 error bound");
  require(!qualityGate ||
              (qualityCosine > 0.9999 && qualityMaximumAbsolute < 0.01f &&
               top1Agreement >= 0.98),
          "Q8 production attention failed its BF16 quality gate");
  for (uint32_t lane = 0; lane < width; ++lane)
    for (uint32_t head = 0; head < shape.queryHeads(); ++head)
      for (uint32_t row = activeRows; row < kRows; ++row)
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
          require(actual[uint64_t{lane} * laneElements +
                         data.queryIndex(head, row, dimension)] ==
                      BFloat16Bits{0},
                  "inactive verify row was not zeroed");
}

// Independent submissions of the same graph must preserve all outputs and
// scratch results. Report the first differing element before failing.
void requireIdentical(const Pipelines &placement, const Case &data,
                      uint32_t width, const Dispatch &first,
                      const Dispatch &repeat) {
  struct Buffer {
    const char *name;
    const std::vector<uint8_t> Dispatch::*bytes;
    size_t elementBytes;
  };
  constexpr std::array<Buffer, 3> buffers{
      {{"output", &Dispatch::output, sizeof(BFloat16Bits)},
       {"partials", &Dispatch::partials, sizeof(float)},
       {"statistics", &Dispatch::statistics, sizeof(float)}}};
  for (const Buffer &buffer : buffers) {
    const std::vector<uint8_t> &left = first.*buffer.bytes;
    const std::vector<uint8_t> &right = repeat.*buffer.bytes;
    const auto [leftAt, rightAt] =
        std::mismatch(left.begin(), left.end(), right.begin(), right.end());
    if (leftAt == left.end() && rightAt == right.end())
      continue;
    const size_t element = size_t(leftAt - left.begin()) / buffer.elementBytes;
    const auto value = [&](const std::vector<uint8_t> &bytes) {
      const size_t at = element * buffer.elementBytes;
      if (at + buffer.elementBytes > bytes.size())
        return std::string("absent");
      if (buffer.elementBytes == sizeof(float)) {
        float decoded = 0.0f;
        std::memcpy(&decoded, bytes.data() + at, sizeof decoded);
        return std::to_string(decoded);
      }
      BFloat16Bits decoded{};
      std::memcpy(&decoded, bytes.data() + at, sizeof decoded);
      return std::to_string(bfloat16ToFloat(decoded));
    };
    size_t differing = 0;
    for (size_t i = 0; i < std::min(left.size(), right.size()); ++i)
      differing += left[i] != right[i];
    differing += std::max(left.size(), right.size()) - std::min(left.size(), right.size());
    std::cout << "verify repeat differs from the first submission: pipeline="
              << placement.splitName << " kv_heads=" << data.shape.kvHeads
              << " group=" << data.shape.queryHeadsPerKvHead
              << " committed=" << data.params.committed_tokens
              << " active_rows=" << data.params.active_rows
              << " verify_batch_width=" << width
              << " splits=" << data.params.split_count << " buffer=" << buffer.name
              << " first_differing_element=" << element << " first=" << value(left)
              << " repeat=" << value(right) << " differing_bytes=" << differing
              << " of " << left.size() << '\n';
    throw std::runtime_error(
        "verify repeat is not bit-identical to the first submission");
  }
}

// The sparse walk visits a list of logical pages and masks the tokens the
// indexer dropped. Handing it every page with every token kept must reproduce
// the dense result exactly: same positions, same causal limit, same softmax,
// only a different way of arriving at the pages. Anything less than bit
// equality here means the walk moved something it should not have.
Dispatch dispatchSparse(id<MTLDevice> device, id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> split,
                        id<MTLComputePipelineState> reduce, const Case &data,
                        uint32_t width, const std::vector<uint32_t> &pages,
                        const std::vector<uint32_t> &masks) {
  const Shape shape = data.shape;
  const uint32_t splits = data.params.split_count;
  const uint64_t laneBytes = data.queries.length;
  id<MTLBuffer> queries = makeBuffer(device, width * laneBytes);
  id<MTLBuffer> output = makeBuffer(device, width * laneBytes);
  const uint64_t slots = uint64_t{width} * shape.kvHeads * splits;
  id<MTLBuffer> partials = makeBuffer(
      device, slots * shape.fusedRows() * kHeadDimension * sizeof(float));
  id<MTLBuffer> statistics =
      makeBuffer(device, slots * shape.fusedRows() * 2 * sizeof(float));
  for (uint32_t lane = 0; lane < width; ++lane) {
    std::memcpy(static_cast<uint8_t *>(queries.contents) + lane * laneBytes,
                data.queries.contents, laneBytes);
  }
  // One lane's selection: the count, then the pages, then their masks.
  const uint32_t stride = uint32_t(1 + pages.size() + masks.size());
  id<MTLBuffer> selection =
      makeBuffer(device, uint64_t{stride} * width * sizeof(uint32_t));
  uint32_t *cursor = static_cast<uint32_t *>(selection.contents);
  for (uint32_t lane = 0; lane < width; ++lane) {
    uint32_t *base = cursor + uint64_t{lane} * stride;
    base[0] = uint32_t(pages.size());
    std::memcpy(base + 1, pages.data(), pages.size() * sizeof(uint32_t));
    std::memcpy(base + 1 + pages.size(), masks.data(),
                masks.size() * sizeof(uint32_t));
  }
  std::array<Q8VerifyAttentionParams, 4> params{};
  params.fill(data.params);
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:split];
  [encoder setBuffer:queries offset:0 atIndex:0];
  [encoder setBuffer:data.q8Keys offset:0 atIndex:1];
  [encoder setBuffer:data.keyScales offset:0 atIndex:2];
  [encoder setBuffer:data.q8Values offset:0 atIndex:3];
  [encoder setBuffer:data.valueScales offset:0 atIndex:4];
  [encoder setBuffer:partials offset:0 atIndex:5];
  [encoder setBuffer:statistics offset:0 atIndex:6];
  for (uint32_t index = 7; index < 11; ++index)
    [encoder setBuffer:data.pageTableBuffer offset:0 atIndex:index];
  [encoder setBytes:params.data()
              length:sizeof(Q8VerifyAttentionParams) * params.size()
             atIndex:11];
  [encoder setBuffer:selection offset:0 atIndex:12];
  [encoder setBytes:&stride length:sizeof(uint32_t) atIndex:13];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, splits, width)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:partials offset:0 atIndex:0];
  [encoder setBuffer:statistics offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:params.data()
              length:sizeof(Q8VerifyAttentionParams) * params.size()
             atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, shape.fusedRows(), width)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  finish(command);
  return {copyOf(output), copyOf(partials), copyOf(statistics)};
}

// Keep the CPU reference gates for both scale placements and verify that a
// second independent submission produces identical output and scratch.
void runCase(id<MTLDevice> device, id<MTLCommandQueue> queue,
             const std::array<Pipelines, 2> &pipelines, Shape shape,
             uint32_t committed, uint32_t activeRows, uint32_t width,
             bool qualityGate = true, uint32_t splits = kQ8VerifySplits) {
  require(width >= 1 && width <= 4, "invalid verify batch width");
  Case data = makeCase(device, shape, committed, activeRows, splits);
  fill(data);
  const std::vector<BFloat16Bits> expectedQ8 = cpuReference(data, true);
  const std::vector<BFloat16Bits> expectedBf16 = cpuReference(data, false);
  for (const Pipelines &placement : pipelines) {
    const Dispatch first =
        dispatch(device, queue, placement.split, placement.reduce, data, width);
    checkOutput(data, width, expectedQ8, expectedBf16, first.output,
                qualityGate, placement.splitName);
    const Dispatch repeat =
        dispatch(device, queue, placement.split, placement.reduce, data, width);
    checkOutput(data, width, expectedQ8, expectedBf16, repeat.output,
                qualityGate, placement.splitName + "_repeat");
    requireIdentical(placement, data, width, first, repeat);
  }
}

void run(const char *libraryPath) {
  testContract();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device)
    throw std::runtime_error("Metal device unavailable");
  NSError *error = nil;
  NSURL *url =
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryPath]];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (!library)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLCommandQueue> queue = [device newCommandQueue];
  for (const Shape shape : kShapes) {
    const std::array<Pipelines, 2> pipelines{
        makePipelines(device, library, shape, false),
        makePipelines(device, library, shape, true)};
    if (!fitsThreadgroupMemory(device, pipelines))
      continue;
    for (uint32_t width = 1; width <= 4; ++width)
      runCase(device, queue, pipelines, shape, 127, 8, width);
    runCase(device, queue, pipelines, shape, 0, 8, 4);
    for (uint32_t activeRows = 1; activeRows <= 8; ++activeRows)
      runCase(device, queue, pipelines, shape, 127, activeRows, 4);
    runCase(device, queue, pipelines, shape, 129, 8, 4);
    runCase(device, queue, pipelines, shape, 421, 3, 4);
    // Several pages per split, a partially filled last page, and a history
    // that leaves some splits empty.
    runCase(device, queue, pipelines, shape, 1'100, 8, 2, false);
    runCase(device, queue, pipelines, shape, 4'093, 5, 1, false);
    // History-scaled partitions: every split count the plan can produce must
    // agree with the reference through the same store/split/reduce path,
    // including counts that leave a ragged last split.
    runCase(device, queue, pipelines, shape, 4'093, 5, 1, false, 64);
    runCase(device, queue, pipelines, shape, 4'093, 6, 3, false, 65);
    runCase(device, queue, pipelines, shape, 4'093, 8, 2, false,
            kQ8VerifyMaximumSplits);
    runCase(device, queue, pipelines, shape, 1'100, 8, 1, false, 1);
    runCase(device, queue, pipelines, shape, 8'192, 5, 4, false);
    runCase(device, queue, pipelines, shape, 32'768, 7, 4, false);
    runCase(device, queue, pipelines, shape, 32'768, 8, 2, false, 65);
  }
  // The sparse walk, on the shape it is built for. Selecting every page with
  // every token kept must land on the dense answer bit for bit.
  {
    const Shape shape = kShapes[1];
    const Pipelines dense = makePipelines(device, library, shape, true);
    id<MTLComputePipelineState> sparse =
        makePipeline(device, library, "verify_attention_q8_sparse_kv2_g8");
    for (const uint32_t committed : {127u, 1'100u, 4'093u}) {
      Case data = makeCase(device, shape, committed, 8, kQ8VerifySplits);
      fill(data);
      const uint32_t pageCount =
          (committed + 8 + kPageTokens - 1) / kPageTokens;
      std::vector<uint32_t> pages(pageCount), masks(pageCount, ~0u);
      for (uint32_t page = 0; page < pageCount; ++page) pages[page] = page;
      const Dispatch want =
          dispatch(device, queue, dense.split, dense.reduce, data, 2);
      const Dispatch got = dispatchSparse(device, queue, sparse, dense.reduce,
                                          data, 2, pages, masks);
      require(want.output.size() == got.output.size(),
              "sparse output size differs from dense");
      {
        size_t differing = 0, firstAt = 0;
        for (size_t i = 0; i < want.output.size(); ++i) {
          if (want.output[i] != got.output[i]) {
            if (!differing) firstAt = i;
            ++differing;
          }
        }
        if (differing) {
          std::cout << "  sparse differs at " << differing << " of "
                    << want.output.size() << " elements, first at " << firstAt
                    << " dense=" << std::hex << unsigned(want.output[firstAt])
                    << " sparse=" << unsigned(got.output[firstAt]) << std::dec
                    << '\n';
        }
        require(differing == 0,
                "a full selection did not reproduce the dense result exactly");
      }
      // Dropping pages must change the answer, or the mask is doing nothing
      // and the comparison above proves nothing either.
      if (pageCount > 2) {
        std::vector<uint32_t> fewer(pages.begin(), pages.end() - 1);
        std::vector<uint32_t> fewerMasks(fewer.size(), ~0u);
        const Dispatch partial = dispatchSparse(
            device, queue, sparse, dense.reduce, data, 2, fewer, fewerMasks);
        require(std::memcmp(want.output.data(), partial.output.data(),
                            want.output.size()) != 0,
                "dropping a page left the result unchanged");
      }
      std::cout << "sparse_walk committed=" << committed
                << " pages=" << pageCount << " matches dense exactly\n";
    }
  }
  std::cout << "q8_flash_attention_metal_test: ok\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc != 2)
        throw std::runtime_error(
            "usage: q8_flash_attention_metal_test <metallib>");
      run(argv[1]);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "q8_flash_attention_metal_test: " << error.what() << '\n';
      return 1;
    }
  }
}
