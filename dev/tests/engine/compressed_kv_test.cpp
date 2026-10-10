#include "runtime/engine/CompressedKv.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace splash::engine::kv {

namespace {

void testQ4QuantizeAndDirectDotProduct() {
  std::cout << "[testQ4QuantizeAndDirectDotProduct] Starting..." << std::endl;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::array<float, kHeadDimension> kvVector{};
  std::array<float, kHeadDimension> queryVector{};
  for (uint32_t i = 0; i < kHeadDimension; ++i) {
    kvVector[i] = dist(rng);
    queryVector[i] = dist(rng);
  }

  std::array<Q4Group, kGroupsPerHead> q4Groups{};
  CompressedKvOps::quantizeQ4(kvVector, q4Groups);

  std::array<float, kHeadDimension> dequantVector{};
  CompressedKvOps::dequantizeQ4(q4Groups, dequantVector);

  // Compute reconstruction error and cosine similarity
  float dotOriginalDequant = 0.0f;
  float normOrig = 0.0f;
  float normDequant = 0.0f;
  float maxDiff = 0.0f;

  for (uint32_t i = 0; i < kHeadDimension; ++i) {
    float diff = std::abs(kvVector[i] - dequantVector[i]);
    maxDiff = std::max(maxDiff, diff);
    dotOriginalDequant += kvVector[i] * dequantVector[i];
    normOrig += kvVector[i] * kvVector[i];
    normDequant += dequantVector[i] * dequantVector[i];
  }

  float cosineSim = dotOriginalDequant / (std::sqrt(normOrig) * std::sqrt(normDequant));
  std::cout << "  Q4 Cosine similarity: " << cosineSim << " (max abs diff: " << maxDiff << ")" << std::endl;
  assert(cosineSim > 0.985f && "Q4 cosine similarity must exceed 0.985");

  // Verify directDotProductQ4 in registers matches dequantized dot product
  float expectedDot = 0.0f;
  for (uint32_t i = 0; i < kHeadDimension; ++i) {
    expectedDot += dequantVector[i] * queryVector[i];
  }

  float directDot = CompressedKvOps::directDotProductQ4(q4Groups, queryVector);
  float dotDiff = std::abs(directDot - expectedDot);
  std::cout << "  Direct Dot: " << directDot << ", Expected Dot: " << expectedDot
            << " (diff: " << dotDiff << ")" << std::endl;
  assert(dotDiff < 1e-4f && "directDotProductQ4 must exactly match dequantized dot product");

  std::cout << "[testQ4QuantizeAndDirectDotProduct] PASS" << std::endl;
}

void testQ3PolarQuantAndDirectDotProduct() {
  std::cout << "[testQ3PolarQuantAndDirectDotProduct] Starting..." << std::endl;

  std::mt19937 rng(1337);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::array<float, kHeadDimension> kvVector{};
  std::array<float, kHeadDimension> queryVector{};
  for (uint32_t i = 0; i < kHeadDimension; ++i) {
    kvVector[i] = dist(rng);
    queryVector[i] = dist(rng);
  }

  std::array<Q3Group, kGroupsPerHead> q3Groups{};
  CompressedKvOps::quantizeQ3(kvVector, q3Groups);

  std::array<float, kHeadDimension> dequantVector{};
  CompressedKvOps::dequantizeQ3(q3Groups, dequantVector);

  float dotOriginalDequant = 0.0f;
  float normOrig = 0.0f;
  float normDequant = 0.0f;
  float maxDiff = 0.0f;

  for (uint32_t i = 0; i < kHeadDimension; ++i) {
    float diff = std::abs(kvVector[i] - dequantVector[i]);
    maxDiff = std::max(maxDiff, diff);
    dotOriginalDequant += kvVector[i] * dequantVector[i];
    normOrig += kvVector[i] * kvVector[i];
    normDequant += dequantVector[i] * dequantVector[i];
  }

  float cosineSim = dotOriginalDequant / (std::sqrt(normOrig) * std::sqrt(normDequant));
  std::cout << "  Q3 Cosine similarity: " << cosineSim << " (max abs diff: " << maxDiff << ")" << std::endl;
  assert(cosineSim > 0.95f && "Q3 PolarQuant cosine similarity must exceed 0.95");

  float expectedDot = 0.0f;
  for (uint32_t i = 0; i < kHeadDimension; ++i) {
    expectedDot += dequantVector[i] * queryVector[i];
  }

  float directDot = CompressedKvOps::directDotProductQ3(q3Groups, queryVector);
  float dotDiff = std::abs(directDot - expectedDot);
  std::cout << "  Direct Dot: " << directDot << ", Expected Dot: " << expectedDot
            << " (diff: " << dotDiff << ")" << std::endl;
  assert(dotDiff < 1e-4f && "directDotProductQ3 must match dequantized dot product");

  std::cout << "[testQ3PolarQuantAndDirectDotProduct] PASS" << std::endl;
}

void testMemoryFootprintScaling() {
  std::cout << "[testMemoryFootprintScaling] Starting..." << std::endl;

  // 262k context tokens, 4 KV heads per layer, 64 layers (Qwen3.8-27B geometry)
  constexpr uint32_t kContextTokens = 262144;
  constexpr uint32_t kKvHeads = 4;
  constexpr uint32_t kNumLayers = 64;

  auto q8Footprint = CompressedKvOps::calculateMemoryFootprint(
      kContextTokens, kKvHeads, kNumLayers, QuantFormat::Q8_Standard);
  auto q4Footprint = CompressedKvOps::calculateMemoryFootprint(
      kContextTokens, kKvHeads, kNumLayers, QuantFormat::Q4_Packed);
  auto q3Footprint = CompressedKvOps::calculateMemoryFootprint(
      kContextTokens, kKvHeads, kNumLayers, QuantFormat::Q3_PolarQuant);

  std::cout << "  Q8 Standard Footprint: " << q8Footprint.gigabytes << " GiB ("
            << q8Footprint.compressionRatio << "x)" << std::endl;
  std::cout << "  Q4 Packed Footprint:   " << q4Footprint.gigabytes << " GiB ("
            << q4Footprint.compressionRatio << "x)" << std::endl;
  std::cout << "  Q3 PolarQuant Footprint: " << q3Footprint.gigabytes << " GiB ("
            << q3Footprint.compressionRatio << "x)" << std::endl;

  assert(q8Footprint.gigabytes > 30.0 && "Q8 at 262k tokens exceeds 30 GiB baseline");
  assert(q4Footprint.gigabytes < 19.0 && "Q4 packed must reduce memory below 19 GiB");
  assert(q3Footprint.gigabytes < 14.0 && "Q3 PolarQuant must reduce memory below 14 GiB");
  assert(q3Footprint.compressionRatio > 2.4 && "Q3 PolarQuant compression ratio must exceed 2.4x");

  std::cout << "[testMemoryFootprintScaling] PASS" << std::endl;
}

void testMiniCacheDepthMerging() {
  std::cout << "[testMiniCacheDepthMerging] Starting MiniCache (NeurIPS 2024) Depth Merging..." << std::endl;

  std::mt19937 rng(2025);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  // 1. Generate adjacent layer states with high cosine similarity (simulating mid-to-deep layers)
  std::array<float, kHeadDimension> baseDir{};
  for (uint32_t i = 0; i < kHeadDimension; ++i) {
    baseDir[i] = dist(rng);
  }

  std::array<float, kHeadDimension> layerL{};
  std::array<float, kHeadDimension> layerLPlus1{};
  float scaleL = 2.5f;
  float scaleLPlus1 = 3.2f;

  for (uint32_t i = 0; i < kHeadDimension; ++i) {
    // Add small perturbation to simulate adjacent layer shift (signal var 1.0, noise var 0.04 -> cosine ~0.96)
    layerL[i] = (baseDir[i] + dist(rng) * 0.2f) * scaleL;
    layerLPlus1[i] = (baseDir[i] + dist(rng) * 0.2f) * scaleLPlus1;
  }

  float initialCosine = CompressedKvOps::cosineSimilarity(layerL, layerLPlus1);
  std::cout << "  Adjacent Layer Initial Cosine Similarity: " << initialCosine << std::endl;
  assert(initialCosine > 0.90f && "Adjacent layer cosine similarity must exceed 0.90 for MiniCache test");

  // 2. Perform MiniCache layer pair merging using SLERP
  auto pair = CompressedKvOps::mergeLayerPair(layerL, layerLPlus1);
  float sharedNorm = CompressedKvOps::l2Norm(pair.sharedDirection);
  std::cout << "  Shared Direction Vector L2 Norm: " << sharedNorm << std::endl;
  assert(std::abs(sharedNorm - 1.0f) < 1e-5f && "Shared directional vector must have unit L2 norm");

  // 3. Reconstruct vectors and verify fidelity
  std::array<float, kHeadDimension> reconL{};
  std::array<float, kHeadDimension> reconLPlus1{};
  CompressedKvOps::reconstructLayer(pair, true, reconL);
  CompressedKvOps::reconstructLayer(pair, false, reconLPlus1);

  float cosReconL = CompressedKvOps::cosineSimilarity(layerL, reconL);
  float cosReconLPlus1 = CompressedKvOps::cosineSimilarity(layerLPlus1, reconLPlus1);

  std::cout << "  Layer L Reconstruction Cosine Fidelity:      " << cosReconL << std::endl;
  std::cout << "  Layer L+1 Reconstruction Cosine Fidelity:    " << cosReconLPlus1 << std::endl;
  assert(cosReconL > 0.98f && "Layer L reconstruction cosine must exceed 0.98");
  assert(cosReconLPlus1 > 0.98f && "Layer L+1 reconstruction cosine must exceed 0.98");

  // Norm preservation check
  float normReconL = CompressedKvOps::l2Norm(reconL);
  float normOrigL = CompressedKvOps::l2Norm(layerL);
  assert(std::abs(normReconL - normOrigL) / normOrigL < 1e-4f && "Reconstructed norm must match original norm");

  // 4. Test MiniCache Memory Footprint Scaling across a 48-layer model
  constexpr uint32_t kContextTokens = 262144; // 256k tokens
  constexpr uint32_t kKvHeads = 4;
  constexpr uint32_t kTotalLayers = 48;
  constexpr uint32_t kMergeStartLayer = 16; // Merge back 32 layers (16..47) into 16 shared pairs

  auto miniQ4 = CompressedKvOps::calculateMiniCacheMemoryFootprint(
      kContextTokens, kKvHeads, kTotalLayers, kMergeStartLayer, QuantFormat::Q4_Packed);
  auto miniQ3 = CompressedKvOps::calculateMiniCacheMemoryFootprint(
      kContextTokens, kKvHeads, kTotalLayers, kMergeStartLayer, QuantFormat::Q3_PolarQuant);

  std::cout << "  MiniCache + Q4 Footprint (256k tokens, 48L): " << miniQ4.gigabytes << " GiB ("
            << miniQ4.compressionRatio << "x)" << std::endl;
  std::cout << "  MiniCache + Q3 Footprint (256k tokens, 48L): " << miniQ3.gigabytes << " GiB ("
            << miniQ3.compressionRatio << "x)" << std::endl;

  assert(miniQ4.gigabytes < 10.0 && "MiniCache + Q4 must drop 256k footprint below 10 GiB");
  assert(miniQ3.gigabytes < 7.0 && "MiniCache + Q3 must drop 256k footprint below 7 GiB");
  assert(miniQ4.compressionRatio > 2.5 && "MiniCache + Q4 compression ratio must exceed 2.5x");
  assert(miniQ3.compressionRatio > 3.5 && "MiniCache + Q3 compression ratio must exceed 3.5x");

  std::cout << "[testMiniCacheDepthMerging] PASS" << std::endl;
}

} // namespace

} // namespace splash::engine::kv

int main() {
  std::cout << "=== Running Compressed KV In-RAM Attention Tests ===" << std::endl;
  splash::engine::kv::testQ4QuantizeAndDirectDotProduct();
  splash::engine::kv::testQ3PolarQuantAndDirectDotProduct();
  splash::engine::kv::testMemoryFootprintScaling();
  splash::engine::kv::testMiniCacheDepthMerging();
  std::cout << "=== All Compressed KV Tests Passed! ===" << std::endl;
  return 0;
}
