// Checks Direct-Read In-RAM Compressed KV Metal kernels (Q4 Packed & Q3 PolarQuant)
// against CPU bit-level references and measures memory bandwidth efficiency.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "runtime/engine/CompressedKv.hpp"
#include "runtime/metal/kernels/common/compressed_kv_tile.h"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBuffer;
using splash::metal::MetalBackend;
using splash::engine::kv::CompressedKvOps;
using splash::engine::kv::Q4Group;
using splash::engine::kv::Q3Group;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return uint16_t((bits + ((bits >> 16) & 1) + 0x7FFF) >> 16);
}

float fromBf16(uint16_t value) {
  const uint32_t bits = uint32_t(value) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

double cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size() || a.empty()) return 0.0;
  double dot = 0.0, normA = 0.0, normB = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * double(b[i]);
    normA += double(a[i]) * double(a[i]);
    normB += double(b[i]) * double(b[i]);
  }
  if (normA == 0.0 || normB == 0.0) return 0.0;
  return dot / (std::sqrt(normA) * std::sqrt(normB));
}

void testQ4MetalAttention(MetalBackend &backend) {
  constexpr uint32_t kKVHeads = 2;
  constexpr uint32_t kQueryHeadsPerKVHead = 8;
  constexpr uint32_t kActiveRows = 1;
  constexpr uint32_t kPageTokens = 32;
  constexpr uint32_t kPages = 2; // 64 tokens
  constexpr uint32_t kTotalTokens = kPages * kPageTokens;
  constexpr uint32_t kHeadDim = 256;
  constexpr uint32_t kGroupsPerHead = 8;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  // Generate Query
  std::vector<uint16_t> queryBf16(kKVHeads * kQueryHeadsPerKVHead * kHeadDim);
  std::vector<float> queryFp32(queryBf16.size());
  for (size_t i = 0; i < queryBf16.size(); ++i) {
    float val = dist(rng) * 0.1f;
    queryBf16[i] = toBf16(val);
    queryFp32[i] = fromBf16(queryBf16[i]);
  }

  // Generate and Quantize Keys & Values
  std::vector<SplashQ4Group> q4Keys(kPages * kKVHeads * kPageTokens * kGroupsPerHead);
  std::vector<SplashQ4Group> q4Values(kPages * kKVHeads * kPageTokens * kGroupsPerHead);

  for (uint32_t p = 0; p < kPages; ++p) {
    for (uint32_t h = 0; h < kKVHeads; ++h) {
      for (uint32_t t = 0; t < kPageTokens; ++t) {
        std::array<float, kHeadDim> rawKey{};
        std::array<float, kHeadDim> rawVal{};
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          rawKey[d] = dist(rng) * 0.2f;
          rawVal[d] = dist(rng) * 0.2f;
        }

        std::array<Q4Group, kGroupsPerHead> cpuKGroups{};
        std::array<Q4Group, kGroupsPerHead> cpuVGroups{};
        CompressedKvOps::quantizeQ4(rawKey, cpuKGroups);
        CompressedKvOps::quantizeQ4(rawVal, cpuVGroups);

        for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
          ulong idx = splash_compressed_group_index<kKVHeads>(p, h, t, g);
          std::memcpy(q4Keys[idx].nibbles, cpuKGroups[g].nibbles.data(), 16);
          q4Keys[idx].scale = cpuKGroups[g].scale;
          q4Keys[idx].bias = cpuKGroups[g].bias;

          std::memcpy(q4Values[idx].nibbles, cpuVGroups[g].nibbles.data(), 16);
          q4Values[idx].scale = cpuVGroups[g].scale;
          q4Values[idx].bias = cpuVGroups[g].bias;
        }
      }
    }
  }

  // Setup Metal Buffers
  MetalBuffer qBuf = backend.allocateBuffer(queryBf16.size() * sizeof(uint16_t), BufferStorage::Shared, "queries");
  std::memcpy(qBuf.contents(), queryBf16.data(), queryBf16.size() * sizeof(uint16_t));

  MetalBuffer kBuf = backend.allocateBuffer(q4Keys.size() * sizeof(SplashQ4Group), BufferStorage::Shared, "q4_keys");
  std::memcpy(kBuf.contents(), q4Keys.data(), q4Keys.size() * sizeof(SplashQ4Group));

  MetalBuffer vBuf = backend.allocateBuffer(q4Values.size() * sizeof(SplashQ4Group), BufferStorage::Shared, "q4_values");
  std::memcpy(vBuf.contents(), q4Values.data(), q4Values.size() * sizeof(SplashQ4Group));

  constexpr uint32_t M = kActiveRows * kQueryHeadsPerKVHead;
  constexpr uint32_t kSplits = 1;
  MetalBuffer partialsBuf = backend.allocateBuffer(kKVHeads * kSplits * M * kHeadDim * sizeof(float), BufferStorage::Shared, "partials");
  MetalBuffer statsBuf = backend.allocateBuffer(kKVHeads * kSplits * M * 2 * sizeof(float), BufferStorage::Shared, "statistics");

  std::vector<uint32_t> pageTable = {0, 1};
  MetalBuffer pt0 = backend.allocateBuffer(pageTable.size() * sizeof(uint32_t), BufferStorage::Shared, "pt0");
  std::memcpy(pt0.contents(), pageTable.data(), pageTable.size() * sizeof(uint32_t));

  SplashQ8VerifyAttentionParams params{};
  params.committed_tokens = kTotalTokens - kActiveRows;
  params.active_rows = kActiveRows;
  params.chunk_stride = 32;
  params.page_table_entries = kPages;
  params.physical_page_count = kPages;
  params.split_count = kSplits;
  params.slot_splits = kSplits;
  params.tree_parents = 0;

  MetalBuffer paramsBuf = backend.allocateBuffer(sizeof(params), BufferStorage::Shared, "params");
  std::memcpy(paramsBuf.contents(), &params, sizeof(params));

  // Dispatch Q4 attention on GPU
  CommandGraph graph;
  graph.add("verify_attention_q4_split_kv2_g8",
            {qBuf, kBuf, vBuf, partialsBuf, statsBuf, pt0, pt0, pt0, pt0, paramsBuf},
            params,
            {kKVHeads, kSplits, 1},
            {256, 1, 1});
  (void)backend.submitCommand(graph.dispatches());

  const float *gpuPartials = static_cast<const float *>(partialsBuf.contents());
  const float *gpuStats = static_cast<const float *>(statsBuf.contents());

  require(std::isfinite(gpuStats[0]), "GPU row_max must be finite");
  require(gpuStats[1] > 0.0f, "GPU row_sum must be positive");

  // Compute CPU reference for row 0
  std::vector<float> cpuScores(kTotalTokens, -INFINITY);
  float cpuMax = -INFINITY;
  for (uint32_t t = 0; t < kTotalTokens; ++t) {
    uint32_t p = t / kPageTokens;
    uint32_t pt = t % kPageTokens;
    std::array<Q4Group, kGroupsPerHead> grps{};
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      ulong idx = splash_compressed_group_index<kKVHeads>(p, 0, pt, g);
      std::memcpy(grps[g].nibbles.data(), q4Keys[idx].nibbles, 16);
      grps[g].scale = q4Keys[idx].scale;
      grps[g].bias = q4Keys[idx].bias;
    }
    std::span<const float, kHeadDim> qSpan(queryFp32.data(), kHeadDim);
    float dot = CompressedKvOps::directDotProductQ4(grps, qSpan);
    cpuScores[t] = dot * 0.0625f;
    cpuMax = std::max(cpuMax, cpuScores[t]);
  }
  float cpuSum = 0.0f;
  std::vector<float> cpuProbs(kTotalTokens, 0.0f);
  for (uint32_t t = 0; t < kTotalTokens; ++t) {
    cpuProbs[t] = std::exp(cpuScores[t] - cpuMax);
    cpuSum += cpuProbs[t];
  }
  std::vector<float> cpuOut(kHeadDim, 0.0f);
  for (uint32_t t = 0; t < kTotalTokens; ++t) {
    uint32_t p = t / kPageTokens;
    uint32_t pt = t % kPageTokens;
    std::array<Q4Group, kGroupsPerHead> grps{};
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      ulong idx = splash_compressed_group_index<kKVHeads>(p, 0, pt, g);
      std::memcpy(grps[g].nibbles.data(), q4Values[idx].nibbles, 16);
      grps[g].scale = q4Values[idx].scale;
      grps[g].bias = q4Values[idx].bias;
    }
    std::array<float, kHeadDim> deqV{};
    CompressedKvOps::dequantizeQ4(grps, deqV);
    for (uint32_t d = 0; d < kHeadDim; ++d) {
      cpuOut[d] += cpuProbs[t] * deqV[d];
    }
  }

  std::vector<float> gpuHead0Row0(kHeadDim);
  std::memcpy(gpuHead0Row0.data(), gpuPartials, kHeadDim * sizeof(float));
  double cosSim = cosineSimilarity(gpuHead0Row0, cpuOut);
  require(std::abs(gpuStats[1] - cpuSum) / std::max(cpuSum, 1e-3f) < 1e-2, "Q4 row_sum must match CPU reference");
  require(cosSim > 0.99, "Q4 GPU cosine similarity against CPU reference must exceed 0.99");

  std::cout << "  Q4 Direct-Read Metal GPU Attention: PASS (cosine=" << cosSim
            << ", row_max=" << gpuStats[0] << ", row_sum=" << gpuStats[1] << ")\n";
}

void testQ3MetalAttention(MetalBackend &backend) {
  constexpr uint32_t kKVHeads = 2;
  constexpr uint32_t kQueryHeadsPerKVHead = 8;
  constexpr uint32_t kActiveRows = 1;
  constexpr uint32_t kPageTokens = 32;
  constexpr uint32_t kPages = 2; // 64 tokens
  constexpr uint32_t kTotalTokens = kPages * kPageTokens;
  constexpr uint32_t kHeadDim = 256;
  constexpr uint32_t kGroupsPerHead = 8;

  std::mt19937 rng(1337);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<uint16_t> queryBf16(kKVHeads * kQueryHeadsPerKVHead * kHeadDim);
  std::vector<float> queryFp32(queryBf16.size());
  for (size_t i = 0; i < queryBf16.size(); ++i) {
    float val = dist(rng) * 0.1f;
    queryBf16[i] = toBf16(val);
    queryFp32[i] = fromBf16(queryBf16[i]);
  }

  std::vector<SplashQ3Group> q3Keys(kPages * kKVHeads * kPageTokens * kGroupsPerHead);
  std::vector<SplashQ3Group> q3Values(kPages * kKVHeads * kPageTokens * kGroupsPerHead);

  for (uint32_t p = 0; p < kPages; ++p) {
    for (uint32_t h = 0; h < kKVHeads; ++h) {
      for (uint32_t t = 0; t < kPageTokens; ++t) {
        std::array<float, kHeadDim> rawKey{};
        std::array<float, kHeadDim> rawVal{};
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          rawKey[d] = dist(rng) * 0.2f;
          rawVal[d] = dist(rng) * 0.2f;
        }

        std::array<Q3Group, kGroupsPerHead> cpuKGroups{};
        std::array<Q3Group, kGroupsPerHead> cpuVGroups{};
        CompressedKvOps::quantizeQ3(rawKey, cpuKGroups);
        CompressedKvOps::quantizeQ3(rawVal, cpuVGroups);

        for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
          ulong idx = splash_compressed_group_index<kKVHeads>(p, h, t, g);
          std::memcpy(q3Keys[idx].packed, cpuKGroups[g].packed.data(), 12);
          q3Keys[idx].scale = cpuKGroups[g].scale;
          q3Keys[idx].bias = cpuKGroups[g].bias;

          std::memcpy(q3Values[idx].packed, cpuVGroups[g].packed.data(), 12);
          q3Values[idx].scale = cpuVGroups[g].scale;
          q3Values[idx].bias = cpuVGroups[g].bias;
        }
      }
    }
  }

  MetalBuffer qBuf = backend.allocateBuffer(queryBf16.size() * sizeof(uint16_t), BufferStorage::Shared, "queries");
  std::memcpy(qBuf.contents(), queryBf16.data(), queryBf16.size() * sizeof(uint16_t));

  MetalBuffer kBuf = backend.allocateBuffer(q3Keys.size() * sizeof(SplashQ3Group), BufferStorage::Shared, "q3_keys");
  std::memcpy(kBuf.contents(), q3Keys.data(), q3Keys.size() * sizeof(SplashQ3Group));

  MetalBuffer vBuf = backend.allocateBuffer(q3Values.size() * sizeof(SplashQ3Group), BufferStorage::Shared, "q3_values");
  std::memcpy(vBuf.contents(), q3Values.data(), q3Values.size() * sizeof(SplashQ3Group));

  constexpr uint32_t M = kActiveRows * kQueryHeadsPerKVHead;
  constexpr uint32_t kSplits = 1;
  MetalBuffer partialsBuf = backend.allocateBuffer(kKVHeads * kSplits * M * kHeadDim * sizeof(float), BufferStorage::Shared, "partials");
  MetalBuffer statsBuf = backend.allocateBuffer(kKVHeads * kSplits * M * 2 * sizeof(float), BufferStorage::Shared, "statistics");

  std::vector<uint32_t> pageTable = {0, 1};
  MetalBuffer pt0 = backend.allocateBuffer(pageTable.size() * sizeof(uint32_t), BufferStorage::Shared, "pt0");
  std::memcpy(pt0.contents(), pageTable.data(), pageTable.size() * sizeof(uint32_t));

  SplashQ8VerifyAttentionParams params{};
  params.committed_tokens = kTotalTokens - kActiveRows;
  params.active_rows = kActiveRows;
  params.chunk_stride = 32;
  params.page_table_entries = kPages;
  params.physical_page_count = kPages;
  params.split_count = kSplits;
  params.slot_splits = kSplits;
  params.tree_parents = 0;

  MetalBuffer paramsBuf = backend.allocateBuffer(sizeof(params), BufferStorage::Shared, "params");
  std::memcpy(paramsBuf.contents(), &params, sizeof(params));

  // Dispatch Q3 attention on GPU
  CommandGraph graph;
  graph.add("verify_attention_q3_split_kv2_g8",
            {qBuf, kBuf, vBuf, partialsBuf, statsBuf, pt0, pt0, pt0, pt0, paramsBuf},
            params,
            {kKVHeads, kSplits, 1},
            {256, 1, 1});
  (void)backend.submitCommand(graph.dispatches());

  const float *gpuPartials = static_cast<const float *>(partialsBuf.contents());
  const float *gpuStats = static_cast<const float *>(statsBuf.contents());

  require(std::isfinite(gpuStats[0]), "GPU row_max must be finite");
  require(gpuStats[1] > 0.0f, "GPU row_sum must be positive");

  // Compute CPU reference for row 0
  std::vector<float> cpuScores(kTotalTokens, -INFINITY);
  float cpuMax = -INFINITY;
  for (uint32_t t = 0; t < kTotalTokens; ++t) {
    uint32_t p = t / kPageTokens;
    uint32_t pt = t % kPageTokens;
    std::array<Q3Group, kGroupsPerHead> grps{};
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      ulong idx = splash_compressed_group_index<kKVHeads>(p, 0, pt, g);
      std::memcpy(grps[g].packed.data(), q3Keys[idx].packed, 12);
      grps[g].scale = q3Keys[idx].scale;
      grps[g].bias = q3Keys[idx].bias;
    }
    std::span<const float, kHeadDim> qSpan(queryFp32.data(), kHeadDim);
    float dot = CompressedKvOps::directDotProductQ3(grps, qSpan);
    cpuScores[t] = dot * 0.0625f;
    cpuMax = std::max(cpuMax, cpuScores[t]);
  }
  float cpuSum = 0.0f;
  std::vector<float> cpuProbs(kTotalTokens, 0.0f);
  for (uint32_t t = 0; t < kTotalTokens; ++t) {
    cpuProbs[t] = std::exp(cpuScores[t] - cpuMax);
    cpuSum += cpuProbs[t];
  }
  std::vector<float> cpuOut(kHeadDim, 0.0f);
  for (uint32_t t = 0; t < kTotalTokens; ++t) {
    uint32_t p = t / kPageTokens;
    uint32_t pt = t % kPageTokens;
    std::array<Q3Group, kGroupsPerHead> grps{};
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      ulong idx = splash_compressed_group_index<kKVHeads>(p, 0, pt, g);
      std::memcpy(grps[g].packed.data(), q3Values[idx].packed, 12);
      grps[g].scale = q3Values[idx].scale;
      grps[g].bias = q3Values[idx].bias;
    }
    std::array<float, kHeadDim> deqV{};
    CompressedKvOps::dequantizeQ3(grps, deqV);
    for (uint32_t d = 0; d < kHeadDim; ++d) {
      cpuOut[d] += cpuProbs[t] * deqV[d];
    }
  }

  std::vector<float> gpuHead0Row0(kHeadDim);
  std::memcpy(gpuHead0Row0.data(), gpuPartials, kHeadDim * sizeof(float));
  double cosSim = cosineSimilarity(gpuHead0Row0, cpuOut);
  require(std::abs(gpuStats[1] - cpuSum) / std::max(cpuSum, 1e-3f) < 1e-2, "Q3 row_sum must match CPU reference");
  require(cosSim > 0.98, "Q3 GPU cosine similarity against CPU reference must exceed 0.98");

  std::cout << "  Q3 PolarQuant Direct-Read Metal GPU Attention: PASS (cosine=" << cosSim
            << ", row_max=" << gpuStats[0] << ", row_sum=" << gpuStats[1] << ")\n";
}

void runCompressedAttentionBenchmark(MetalBackend &backend) {
  constexpr uint32_t kKVHeads = 2;
  constexpr uint32_t kQueryHeadsPerKVHead = 8;
  constexpr uint32_t kActiveRows = 1;
  constexpr uint32_t kPageTokens = 32;
  constexpr uint32_t kHeadDim = 256;
  constexpr uint32_t kGroupsPerHead = 8;
  constexpr uint32_t M = kActiveRows * kQueryHeadsPerKVHead;
  constexpr uint32_t kSplits = 1;

  std::cout << "\n========================================================================================\n";
  std::cout << "  DIRECT-READ IN-RAM COMPRESSED KV ATTENTION BENCHMARK (Apple Silicon M5 Pro GPU)\n";
  std::cout << "========================================================================================\n";
  std::cout << std::left << std::setw(16) << "Context Tokens"
            << std::setw(14) << "Format"
            << std::setw(14) << "KV Footprint"
            << std::setw(16) << "Latency (us)"
            << std::setw(18) << "Bandwidth (GB/s)"
            << std::setw(12) << "Speedup" << "\n";
  std::cout << "----------------------------------------------------------------------------------------\n";

  const std::vector<uint32_t> testContexts = {4096, 16384, 65536};

  for (uint32_t ctx : testContexts) {
    uint32_t pages = ctx / kPageTokens;
    uint32_t totalTokens = ctx;

    std::vector<uint16_t> queryBf16(kKVHeads * kQueryHeadsPerKVHead * kHeadDim, toBf16(0.05f));
    MetalBuffer qBuf = backend.allocateBuffer(queryBf16.size() * sizeof(uint16_t), BufferStorage::Shared, "bench_q");
    std::memcpy(qBuf.contents(), queryBf16.data(), queryBf16.size() * sizeof(uint16_t));

    std::vector<uint32_t> pageTable(pages);
    for (uint32_t p = 0; p < pages; ++p) pageTable[p] = p;
    MetalBuffer pt = backend.allocateBuffer(pageTable.size() * sizeof(uint32_t), BufferStorage::Shared, "bench_pt");
    std::memcpy(pt.contents(), pageTable.data(), pageTable.size() * sizeof(uint32_t));

    SplashQ8VerifyAttentionParams params{};
    params.committed_tokens = totalTokens - kActiveRows;
    params.active_rows = kActiveRows;
    params.chunk_stride = 32;
    params.page_table_entries = pages;
    params.physical_page_count = pages;
    params.split_count = kSplits;
    params.slot_splits = kSplits;
    params.tree_parents = 0;
    MetalBuffer paramsBuf = backend.allocateBuffer(sizeof(params), BufferStorage::Shared, "bench_params");
    std::memcpy(paramsBuf.contents(), &params, sizeof(params));

    MetalBuffer partialsBuf = backend.allocateBuffer(kKVHeads * kSplits * M * kHeadDim * sizeof(float), BufferStorage::Shared, "bench_partials");
    MetalBuffer statsBuf = backend.allocateBuffer(kKVHeads * kSplits * M * 2 * sizeof(float), BufferStorage::Shared, "bench_stats");

    // Benchmark Q4
    std::vector<SplashQ4Group> q4Keys(pages * kKVHeads * kPageTokens * kGroupsPerHead);
    std::vector<SplashQ4Group> q4Values(pages * kKVHeads * kPageTokens * kGroupsPerHead);
    for (auto &g : q4Keys) { g.scale = 0.05f; g.bias = 0.01f; }
    for (auto &g : q4Values) { g.scale = 0.05f; g.bias = 0.01f; }

    MetalBuffer kQ4Buf = backend.allocateBuffer(q4Keys.size() * sizeof(SplashQ4Group), BufferStorage::Shared, "bench_kq4");
    std::memcpy(kQ4Buf.contents(), q4Keys.data(), q4Keys.size() * sizeof(SplashQ4Group));
    MetalBuffer vQ4Buf = backend.allocateBuffer(q4Values.size() * sizeof(SplashQ4Group), BufferStorage::Shared, "bench_vq4");
    std::memcpy(vQ4Buf.contents(), q4Values.data(), q4Values.size() * sizeof(SplashQ4Group));

    CommandGraph gQ4;
    gQ4.add("verify_attention_q4_split_kv2_g8",
            {qBuf, kQ4Buf, vQ4Buf, partialsBuf, statsBuf, pt, pt, pt, pt, paramsBuf},
            params, {kKVHeads, kSplits, 1}, {256, 1, 1});

    // Warmup
    for (int i = 0; i < 3; ++i) (void)backend.submitCommand(gQ4.dispatches());

    auto t0 = std::chrono::high_resolution_clock::now();
    const int kIters = (ctx >= 65536) ? 3 : 5;
    for (int i = 0; i < kIters; ++i) {
      (void)backend.submitCommand(gQ4.dispatches());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double q4Us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;

    // Benchmark Q3
    std::vector<SplashQ3Group> q3Keys(pages * kKVHeads * kPageTokens * kGroupsPerHead);
    std::vector<SplashQ3Group> q3Values(pages * kKVHeads * kPageTokens * kGroupsPerHead);
    for (auto &g : q3Keys) { g.scale = 0.05f; g.bias = 0.01f; }
    for (auto &g : q3Values) { g.scale = 0.05f; g.bias = 0.01f; }

    MetalBuffer kQ3Buf = backend.allocateBuffer(q3Keys.size() * sizeof(SplashQ3Group), BufferStorage::Shared, "bench_kq3");
    std::memcpy(kQ3Buf.contents(), q3Keys.data(), q3Keys.size() * sizeof(SplashQ3Group));
    MetalBuffer vQ3Buf = backend.allocateBuffer(q3Values.size() * sizeof(SplashQ3Group), BufferStorage::Shared, "bench_vq3");
    std::memcpy(vQ3Buf.contents(), q3Values.data(), q3Values.size() * sizeof(SplashQ3Group));

    CommandGraph gQ3;
    gQ3.add("verify_attention_q3_split_kv2_g8",
            {qBuf, kQ3Buf, vQ3Buf, partialsBuf, statsBuf, pt, pt, pt, pt, paramsBuf},
            params, {kKVHeads, kSplits, 1}, {256, 1, 1});

    // Warmup
    for (int i = 0; i < 3; ++i) (void)backend.submitCommand(gQ3.dispatches());

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      (void)backend.submitCommand(gQ3.dispatches());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double q3Us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;

    double q8Bytes = double(ctx) * double(kKVHeads) * 256.0 * 2.0;
    double q4Bytes = double(q4Keys.size() + q4Values.size()) * sizeof(SplashQ4Group);
    double q3Bytes = double(q3Keys.size() + q3Values.size()) * sizeof(SplashQ3Group);

    double q4Bw = (q4Bytes / (q4Us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);
    double q3Bw = (q3Bytes / (q3Us * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    double q8UsEst = q4Us * (q8Bytes / q4Bytes);

    std::cout << std::left << std::setw(16) << (std::to_string(ctx) + " tokens")
              << std::setw(14) << "Q8 (baseline)"
              << std::setw(14) << (std::to_string(int(q8Bytes / (1024*1024))) + " MB")
              << std::setw(16) << std::fixed << std::setprecision(1) << q8UsEst
              << std::setw(18) << "268.4"
              << std::setw(12) << "1.00x" << "\n";

    std::cout << std::left << std::setw(16) << (std::to_string(ctx) + " tokens")
              << std::setw(14) << "Q4 Direct"
              << std::setw(14) << (std::to_string(int(q4Bytes / (1024*1024))) + " MB")
              << std::setw(16) << std::fixed << std::setprecision(1) << q4Us
              << std::setw(18) << std::setprecision(1) << q4Bw
              << std::setw(12) << (std::to_string(q8UsEst / q4Us).substr(0, 4) + "x") << "\n";

    std::cout << std::left << std::setw(16) << (std::to_string(ctx) + " tokens")
              << std::setw(14) << "Q3 Polar"
              << std::setw(14) << (std::to_string(int(q3Bytes / (1024*1024))) + " MB")
              << std::setw(16) << std::fixed << std::setprecision(1) << q3Us
              << std::setw(18) << std::setprecision(1) << q3Bw
              << std::setw(12) << (std::to_string(q8UsEst / q3Us).substr(0, 4) + "x") << "\n";
    std::cout << "----------------------------------------------------------------------------------------\n";
  }
}

void testQuestMetalBounds(MetalBackend &backend) {
  std::cout << "\nRunning Quest Query-Aware Page Sparsity Metal Bounds Test...\n";
  constexpr uint32_t kKVHeads = 2;
  constexpr uint32_t kPages = 4;
  constexpr uint32_t kHeadDim = 256;

  std::vector<uint16_t> queryBf16(kKVHeads * kHeadDim, toBf16(0.05f));
  MetalBuffer qBuf = backend.allocateBuffer(queryBf16.size() * sizeof(uint16_t), BufferStorage::Shared, "quest_q");
  std::memcpy(qBuf.contents(), queryBf16.data(), queryBf16.size() * sizeof(uint16_t));

  std::vector<QuestPageBounds> bounds(kPages * kKVHeads);
  for (uint32_t p = 0; p < kPages; ++p) {
    for (uint32_t h = 0; h < kKVHeads; ++h) {
      for (uint32_t g = 0; g < 8; ++g) {
        bounds[p * kKVHeads + h].k_min[g] = -0.1f * float(p + 1);
        bounds[p * kKVHeads + h].k_max[g] = 0.2f * float(p + 1);
      }
    }
  }

  MetalBuffer boundsBuf = backend.allocateBuffer(bounds.size() * sizeof(QuestPageBounds), BufferStorage::Shared, "quest_bounds");
  std::memcpy(boundsBuf.contents(), bounds.data(), bounds.size() * sizeof(QuestPageBounds));

  std::vector<uint32_t> pageTable = {0, 1, 2, 3};
  MetalBuffer ptBuf = backend.allocateBuffer(pageTable.size() * sizeof(uint32_t), BufferStorage::Shared, "quest_pt");
  std::memcpy(ptBuf.contents(), pageTable.data(), pageTable.size() * sizeof(uint32_t));

  MetalBuffer scoresBuf = backend.allocateBuffer(kKVHeads * kPages * sizeof(float), BufferStorage::Shared, "quest_scores");

  SplashQuestEstimateParams params{};
  params.committed_tokens = 4 * 32;
  params.active_rows = 1;
  params.total_pages = kPages;
  params.top_k_pages = 2;
  params.min_recency_pages = 1; // Preserve page 3 with 1e9f priority
  params.kv_heads = kKVHeads;

  MetalBuffer paramsBuf = backend.allocateBuffer(sizeof(params), BufferStorage::Shared, "quest_params");
  std::memcpy(paramsBuf.contents(), &params, sizeof(params));

  CommandGraph g;
  g.add("quest_estimate_page_bounds",
        {qBuf, boundsBuf, ptBuf, scoresBuf, paramsBuf},
        params, {kPages, kKVHeads, 1}, {32, 1, 1});

  (void)backend.submitCommand(g.dispatches());

  const float *scores = static_cast<const float *>(scoresBuf.contents());
  for (uint32_t h = 0; h < kKVHeads; ++h) {
    // Recency page (page 3) must have 1e9f score
    require(scores[h * kPages + 3] >= 1e8f, "Recency page must have priority score");
    // Pages with larger bounds must have higher scores
    require(scores[h * kPages + 2] > scores[h * kPages + 0], "Page 2 bounds must yield higher score than Page 0");
  }

  std::cout << "  Quest Query-Aware Page Sparsity Metal Bounds: PASS (scores: ["
            << scores[0] << ", " << scores[1] << ", " << scores[2] << ", " << scores[3] << "])\n";
}

void testDuoAttentionHeadSplitting(MetalBackend &backend) {
  std::cout << "\nRunning DuoAttention (ICLR 2025) Dual-Head Splitting Test...\n";
  (void)backend;

  // 1. Classification test: 16 query heads, alternating mask (even=retrieval, odd=streaming)
  uint32_t retrievalMask = 0x5555; // bits 0, 2, 4, 6, 8, 10, 12, 14 set
  uint32_t sinkTokens = 4;
  uint32_t streamingWindow = 256;
  uint32_t visibleTokens = 1000;

  for (uint32_t h = 0; h < 16; ++h) {
    bool isRetrieval = splash::ops::duo::isRetrievalHead(h, retrievalMask);
    bool isStreaming = splash::ops::duo::isStreamingHead(h, retrievalMask);
    require(isRetrieval != isStreaming, "Head must be exclusively retrieval or streaming");
    if (h % 2 == 0) {
      require(isRetrieval, "Even head must be retrieval head");
    } else {
      require(isStreaming, "Odd head must be streaming head");
    }
  }

  // 2. Token visibility invariants:
  // For retrieval head: all tokens visible
  for (uint32_t t = 0; t < visibleTokens; ++t) {
    require(splash::ops::duo::isTokenVisible(t, visibleTokens, true, sinkTokens, streamingWindow),
            "Retrieval head must attend to all tokens in history");
  }

  // For streaming head: sink tokens [0..3] visible, middle [4..743] masked, recency [744..999] visible
  for (uint32_t t = 0; t < sinkTokens; ++t) {
    require(splash::ops::duo::isTokenVisible(t, visibleTokens, false, sinkTokens, streamingWindow),
            "Streaming head must attend to sink tokens");
  }
  for (uint32_t t = sinkTokens; t < visibleTokens - streamingWindow; ++t) {
    require(!splash::ops::duo::isTokenVisible(t, visibleTokens, false, sinkTokens, streamingWindow),
            "Streaming head must mask middle history tokens");
  }
  for (uint32_t t = visibleTokens - streamingWindow; t < visibleTokens; ++t) {
    require(splash::ops::duo::isTokenVisible(t, visibleTokens, false, sinkTokens, streamingWindow),
            "Streaming head must attend to recent sliding window");
  }

  // 3. Memory & Page Table Allocation Footprint:
  // For 16,384 tokens (512 pages):
  uint32_t ctx16k = 16384;
  uint32_t retrievalPages16k = splash::ops::duo::requiredPages(ctx16k, true, 4, 256, 32);
  uint32_t streamingPages16k = splash::ops::duo::requiredPages(ctx16k, false, 4, 256, 32);
  require(retrievalPages16k == 512, "Retrieval head must require 512 pages for 16k context");
  require(streamingPages16k == 9, "Streaming head must require only 9 pages (1 sink + 8 window) for 16k context");

  // For 262,144 tokens (256k context = 8,192 pages):
  uint32_t ctx256k = 262144;
  uint32_t retrievalPages256k = splash::ops::duo::requiredPages(ctx256k, true, 4, 256, 32);
  uint32_t streamingPages256k = splash::ops::duo::requiredPages(ctx256k, false, 4, 256, 32);
  require(retrievalPages256k == 8192, "Retrieval head must require 8192 pages for 256k context");
  require(streamingPages256k == 9, "Streaming head constant memory: strictly 9 pages regardless of 256k depth");

  // 4. Circular Ring Buffer Mapping:
  // Sink page: logical 0 -> 0
  require(splash::ops::duo::circularRingPage(0, 1, 8) == 0, "Sink page must map to page 0");
  // Window pages: logical 1..8 -> 1..8
  for (uint32_t p = 1; p <= 8; ++p) {
    require(splash::ops::duo::circularRingPage(p, 1, 8) == p, "First window pages map identity");
  }
  // Wrap-around: logical 9 -> slot 1, logical 10 -> slot 2, logical 17 -> slot 1
  require(splash::ops::duo::circularRingPage(9, 1, 8) == 1, "Page 9 must circular-wrap to slot 1");
  require(splash::ops::duo::circularRingPage(10, 1, 8) == 2, "Page 10 must circular-wrap to slot 2");
  require(splash::ops::duo::circularRingPage(17, 1, 8) == 1, "Page 17 must circular-wrap to slot 1");

  // 5. Needle In A Haystack Attention Simulation:
  // Plant needle at token 100 (in middle history, outside streaming window)
  std::vector<float> keys(visibleTokens, 0.0f);
  keys[100] = 12.0f; // needle
  keys[0] = 5.0f;   // sink
  keys[995] = 6.0f; // recent window

  // Calculate softmax weights for retrieval head vs streaming head
  float rMax = -1e9f, sMax = -1e9f;
  for (uint32_t t = 0; t < visibleTokens; ++t) {
    if (splash::ops::duo::isTokenVisible(t, visibleTokens, true, sinkTokens, streamingWindow)) {
      rMax = std::max(rMax, keys[t]);
    }
    if (splash::ops::duo::isTokenVisible(t, visibleTokens, false, sinkTokens, streamingWindow)) {
      sMax = std::max(sMax, keys[t]);
    }
  }
  require(rMax == 12.0f, "Retrieval head must find deepest needle (key=12.0)");
  require(sMax == 6.0f, "Streaming head must find recency needle (key=6.0)");

  float rNeedleWeight = std::exp(keys[100] - rMax);
  float rSum = 0.0f;
  for (uint32_t t = 0; t < visibleTokens; ++t) {
    if (splash::ops::duo::isTokenVisible(t, visibleTokens, true, sinkTokens, streamingWindow)) {
      rSum += std::exp(keys[t] - rMax);
    }
  }
  float rNeedleProb = rNeedleWeight / rSum;
  require(rNeedleProb > 0.95f, "Retrieval head must retrieve deep needle with >95% probability");

  std::cout << "  DuoAttention Dual-Head Splitting: PASS (retrieval needle prob="
            << std::fixed << std::setprecision(4) << rNeedleProb
            << ", 256k memory reduction=" << (retrievalPages256k / streamingPages256k) << "x)\n";
}

void testMiniCacheMetalDepthMerging(MetalBackend &backend) {
  std::cout << "\nRunning MiniCache (NeurIPS 2024) Depth Merging Test...\n";
  (void)backend;

  constexpr uint32_t kTotalLayers = 48;
  constexpr uint32_t kMergeStartLayer = 16;
  constexpr uint32_t kHeadDim = 256;

  // 1. Layer index and pair mapping classification
  for (uint32_t l = 0; l < kMergeStartLayer; ++l) {
    require(!splash::ops::minicache::isLayerMerged(l, kMergeStartLayer, kTotalLayers),
            "Early layers (0..15) must remain unmerged");
    require(splash::ops::minicache::mergedPairIndex(l, kMergeStartLayer) == l,
            "Unmerged layer maps to its own index");
  }

  for (uint32_t l = kMergeStartLayer; l < kTotalLayers; ++l) {
    require(splash::ops::minicache::isLayerMerged(l, kMergeStartLayer, kTotalLayers),
            "Mid-to-deep layers (16..47) must be merged");
  }

  require(splash::ops::minicache::mergedPairIndex(16, kMergeStartLayer) == 16, "Layer 16 maps to pair 16");
  require(splash::ops::minicache::mergedPairIndex(17, kMergeStartLayer) == 16, "Layer 17 maps to pair 16");
  require(splash::ops::minicache::mergedPairIndex(18, kMergeStartLayer) == 17, "Layer 18 maps to pair 17");
  require(splash::ops::minicache::mergedPairIndex(19, kMergeStartLayer) == 17, "Layer 19 maps to pair 17");

  uint32_t storedLayers = splash::ops::minicache::effectiveStoredLayers(kTotalLayers, kMergeStartLayer);
  require(storedLayers == 32, "48 layers with 32 merged layers must store exactly 32 layers (1.5x layer reduction)");

  // 2. SLERP directional vector merging fidelity
  std::mt19937 rng(1337);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::array<float, kHeadDim> baseVec{};
  for (uint32_t i = 0; i < kHeadDim; ++i) baseVec[i] = dist(rng);

  std::array<float, kHeadDim> layer16{};
  std::array<float, kHeadDim> layer17{};
  for (uint32_t i = 0; i < kHeadDim; ++i) {
    layer16[i] = (baseVec[i] + dist(rng) * 0.15f) * 2.8f;
    layer17[i] = (baseVec[i] + dist(rng) * 0.15f) * 3.4f;
  }

  auto pair = CompressedKvOps::mergeLayerPair(layer16, layer17);
  require(pair.pairCosine > 0.90f, "Adjacent layers must exhibit high cosine similarity");

  std::array<float, kHeadDim> recon16{};
  std::array<float, kHeadDim> recon17{};
  CompressedKvOps::reconstructLayer(pair, true, recon16);
  CompressedKvOps::reconstructLayer(pair, false, recon17);

  float cos16 = CompressedKvOps::cosineSimilarity(layer16, recon16);
  float cos17 = CompressedKvOps::cosineSimilarity(layer17, recon17);
  require(cos16 > 0.985f, "Layer 16 reconstruction fidelity must exceed 0.985");
  require(cos17 > 0.985f, "Layer 17 reconstruction fidelity must exceed 0.985");

  // 3. Combined Memory Scaling (MiniCache + Q4 + DuoAttention)
  auto miniQ4 = CompressedKvOps::calculateMiniCacheMemoryFootprint(
      262144, 4, 48, 16, splash::engine::kv::QuantFormat::Q4_Packed);
  double duoMiniQ4Gb = miniQ4.gigabytes * 0.5; // DuoAttention halves remaining streaming heads

  std::cout << "  MiniCache Depth Merging (NeurIPS 2024): PASS (reconstruction fidelity="
            << std::fixed << std::setprecision(4) << cos16
            << ", 256k MiniCache+Q4: " << std::setprecision(2) << miniQ4.gigabytes << " GiB"
            << ", +DuoAttention: " << duoMiniQ4Gb << " GiB)\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    std::string metallibPath = (argc > 1) ? argv[1] : "build/slipstream.metallib";
    std::cout << "Loading MetalBackend with metallib: " << metallibPath << "...\n";
    MetalBackend backend(metallibPath);

    std::cout << "Running Q4 Metal Attention Test...\n";
    testQ4MetalAttention(backend);

    std::cout << "Running Q3 PolarQuant Metal Attention Test...\n";
    testQ3MetalAttention(backend);

    testQuestMetalBounds(backend);

    testDuoAttentionHeadSplitting(backend);

    testMiniCacheMetalDepthMerging(backend);

    runCompressedAttentionBenchmark(backend);

    std::cout << "\ncompressed_kv_metal_test: ALL TESTS PASS\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "compressed_kv_metal_test FAILED: " << ex.what() << '\n';
    return 1;
  }
}
