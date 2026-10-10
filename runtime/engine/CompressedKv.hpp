#pragma once

#include "metal/abi/ExecutionGeometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::engine::kv {

inline constexpr uint32_t kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS; // 32
inline constexpr uint32_t kHeadDimension = 256;
inline constexpr uint32_t kGroupSize = 32;
inline constexpr uint32_t kGroupsPerHead = kHeadDimension / kGroupSize; // 8

enum class QuantFormat : uint8_t {
  Q8_Standard = 0,
  Q4_Packed = 1,
  Q3_PolarQuant = 2,
};

// 4-bit Quantized 32-element group with fp16 scale and bias
struct Q4Group {
  std::array<uint8_t, kGroupSize / 2> nibbles{}; // 16 bytes
  float scale = 1.0f;
  float bias = 0.0f;
};

// 3-bit Quantized 32-element group: 32 * 3 = 96 bits = 12 bytes
struct Q3Group {
  std::array<uint8_t, 12> packed{}; // 12 bytes
  float scale = 1.0f;
  float bias = 0.0f;
};

// 4-bit Page representation for one KV Head
struct Q4HeadPage {
  // 32 tokens * 8 groups
  std::array<std::array<Q4Group, kGroupsPerHead>, kPageTokens> groups{};

  [[nodiscard]] size_t memoryBytes() const noexcept {
    return sizeof(groups);
  }
};

// 3-bit PolarQuant Page representation for one KV Head
struct Q3HeadPage {
  // 32 tokens * 8 groups
  std::array<std::array<Q3Group, kGroupsPerHead>, kPageTokens> groups{};

  [[nodiscard]] size_t memoryBytes() const noexcept {
    return sizeof(groups);
  }
};

class CompressedKvOps {
public:
  // Quantize 256-float vector into Q4 groups
  static void quantizeQ4(std::span<const float, kHeadDimension> input,
                         std::array<Q4Group, kGroupsPerHead>& outGroups) {
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      uint32_t offset = g * kGroupSize;
      float minVal = input[offset];
      float maxVal = input[offset];
      for (uint32_t i = 1; i < kGroupSize; ++i) {
        minVal = std::min(minVal, input[offset + i]);
        maxVal = std::max(maxVal, input[offset + i]);
      }
      float diff = maxVal - minVal;
      float scale = (diff > 1e-6f) ? diff / 15.0f : 1.0f;
      float invScale = (scale > 1e-6f) ? 1.0f / scale : 1.0f;

      outGroups[g].scale = scale;
      outGroups[g].bias = minVal;

      for (uint32_t i = 0; i < kGroupSize; i += 2) {
        int v0 = std::clamp(static_cast<int>(std::round((input[offset + i] - minVal) * invScale)), 0, 15);
        int v1 = std::clamp(static_cast<int>(std::round((input[offset + i + 1] - minVal) * invScale)), 0, 15);
        outGroups[g].nibbles[i / 2] = static_cast<uint8_t>(v0 | (v1 << 4));
      }
    }
  }

  // Dequantize Q4 groups into 256-float vector
  static void dequantizeQ4(const std::array<Q4Group, kGroupsPerHead>& inGroups,
                           std::span<float, kHeadDimension> output) {
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      uint32_t offset = g * kGroupSize;
      float scale = inGroups[g].scale;
      float bias = inGroups[g].bias;
      for (uint32_t i = 0; i < kGroupSize; i += 2) {
        uint8_t byte = inGroups[g].nibbles[i / 2];
        int v0 = byte & 0x0F;
        int v1 = (byte >> 4) & 0x0F;
        output[offset + i] = static_cast<float>(v0) * scale + bias;
        output[offset + i + 1] = static_cast<float>(v1) * scale + bias;
      }
    }
  }

  // Direct dot product in registers: unrolls and multiplies without scratch buffer
  static float directDotProductQ4(const std::array<Q4Group, kGroupsPerHead>& kvGroups,
                                  std::span<const float, kHeadDimension> query) {
    float sum = 0.0f;
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      uint32_t offset = g * kGroupSize;
      float scale = kvGroups[g].scale;
      float bias = kvGroups[g].bias;
      float groupSum = 0.0f;
      float querySum = 0.0f;
      for (uint32_t i = 0; i < kGroupSize; i += 2) {
        uint8_t byte = kvGroups[g].nibbles[i / 2];
        float q0 = query[offset + i];
        float q1 = query[offset + i + 1];
        groupSum += static_cast<float>(byte & 0x0F) * q0 + static_cast<float>((byte >> 4) & 0x0F) * q1;
        querySum += q0 + q1;
      }
      sum += groupSum * scale + querySum * bias;
    }
    return sum;
  }

  // Quantize 256-float vector into Q3 groups (3-bit PolarQuant)
  static void quantizeQ3(std::span<const float, kHeadDimension> input,
                         std::array<Q3Group, kGroupsPerHead>& outGroups) {
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      uint32_t offset = g * kGroupSize;
      float minVal = input[offset];
      float maxVal = input[offset];
      for (uint32_t i = 1; i < kGroupSize; ++i) {
        minVal = std::min(minVal, input[offset + i]);
        maxVal = std::max(maxVal, input[offset + i]);
      }
      float diff = maxVal - minVal;
      float scale = (diff > 1e-6f) ? diff / 7.0f : 1.0f;
      float invScale = (scale > 1e-6f) ? 1.0f / scale : 1.0f;

      outGroups[g].scale = scale;
      outGroups[g].bias = minVal;

      for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        uint32_t qBase = offset + chunk * 8;
        std::array<uint8_t, 8> vals{};
        for (uint32_t i = 0; i < 8; ++i) {
          vals[i] = static_cast<uint8_t>(
              std::clamp(static_cast<int>(std::round((input[qBase + i] - minVal) * invScale)), 0, 7));
        }

        uint32_t bIdx = chunk * 3;
        outGroups[g].packed[bIdx + 0] = static_cast<uint8_t>(vals[0] | (vals[1] << 3) | ((vals[2] & 0x03) << 6));
        outGroups[g].packed[bIdx + 1] = static_cast<uint8_t>(
            ((vals[2] >> 2) & 0x01) | (vals[3] << 1) | (vals[4] << 4) | ((vals[5] & 0x01) << 7));
        outGroups[g].packed[bIdx + 2] = static_cast<uint8_t>(
            ((vals[5] >> 1) & 0x03) | (vals[6] << 2) | (vals[7] << 5));
      }
    }
  }

  // Dequantize Q3 groups into 256-float vector
  static void dequantizeQ3(const std::array<Q3Group, kGroupsPerHead>& inGroups,
                           std::span<float, kHeadDimension> output) {
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      uint32_t offset = g * kGroupSize;
      float scale = inGroups[g].scale;
      float bias = inGroups[g].bias;

      for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        uint32_t bIdx = chunk * 3;
        uint32_t qBase = offset + chunk * 8;
        uint8_t b0 = inGroups[g].packed[bIdx + 0];
        uint8_t b1 = inGroups[g].packed[bIdx + 1];
        uint8_t b2 = inGroups[g].packed[bIdx + 2];

        output[qBase + 0] = static_cast<float>(b0 & 0x07) * scale + bias;
        output[qBase + 1] = static_cast<float>((b0 >> 3) & 0x07) * scale + bias;
        output[qBase + 2] = static_cast<float>(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)) * scale + bias;
        output[qBase + 3] = static_cast<float>((b1 >> 1) & 0x07) * scale + bias;
        output[qBase + 4] = static_cast<float>((b1 >> 4) & 0x07) * scale + bias;
        output[qBase + 5] = static_cast<float>(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)) * scale + bias;
        output[qBase + 6] = static_cast<float>((b2 >> 2) & 0x07) * scale + bias;
        output[qBase + 7] = static_cast<float>((b2 >> 5) & 0x07) * scale + bias;
      }
    }
  }

  // Direct dot product in registers for Q3: unrolls and multiplies without scratch buffer
  static float directDotProductQ3(const std::array<Q3Group, kGroupsPerHead>& kvGroups,
                                  std::span<const float, kHeadDimension> query) {
    float sum = 0.0f;
    for (uint32_t g = 0; g < kGroupsPerHead; ++g) {
      uint32_t offset = g * kGroupSize;
      float scale = kvGroups[g].scale;
      float bias = kvGroups[g].bias;
      float groupSum = 0.0f;
      float querySum = 0.0f;

      for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        uint32_t bIdx = chunk * 3;
        uint32_t qBase = offset + chunk * 8;
        uint8_t b0 = kvGroups[g].packed[bIdx + 0];
        uint8_t b1 = kvGroups[g].packed[bIdx + 1];
        uint8_t b2 = kvGroups[g].packed[bIdx + 2];

        float v0 = static_cast<float>(b0 & 0x07);
        float v1 = static_cast<float>((b0 >> 3) & 0x07);
        float v2 = static_cast<float>(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2));
        float v3 = static_cast<float>((b1 >> 1) & 0x07);
        float v4 = static_cast<float>((b1 >> 4) & 0x07);
        float v5 = static_cast<float>(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1));
        float v6 = static_cast<float>((b2 >> 2) & 0x07);
        float v7 = static_cast<float>((b2 >> 5) & 0x07);

        groupSum += v0 * query[qBase + 0] + v1 * query[qBase + 1] + v2 * query[qBase + 2] + v3 * query[qBase + 3]
                  + v4 * query[qBase + 4] + v5 * query[qBase + 5] + v6 * query[qBase + 6] + v7 * query[qBase + 7];
        querySum += query[qBase + 0] + query[qBase + 1] + query[qBase + 2] + query[qBase + 3]
                  + query[qBase + 4] + query[qBase + 5] + query[qBase + 6] + query[qBase + 7];
      }
      sum += groupSum * scale + querySum * bias;
    }
    return sum;
  }

  // Memory footprint calculator for capacity planning
  struct FootprintResult {
    uint64_t totalBytes = 0;
    double gigabytes = 0.0;
    double compressionRatio = 1.0;
  };

  static FootprintResult calculateMemoryFootprint(uint32_t contextTokens,
                                                  uint32_t kvHeads,
                                                  uint32_t numLayers,
                                                  QuantFormat format) {
    uint32_t pages = (contextTokens + kPageTokens - 1) / kPageTokens;
    uint64_t bytesPerPage = 0;
    switch (format) {
      case QuantFormat::Q8_Standard:
        // Key (32*256) + Value (32*256) + Scale (32*4) = 16,512 bytes per KV head
        bytesPerPage = 16512ULL * kvHeads;
        break;
      case QuantFormat::Q4_Packed:
        // Key (32*128 + 32*8*8) + Value (same) = ~9,216 bytes per KV head
        bytesPerPage = 9216ULL * kvHeads;
        break;
      case QuantFormat::Q3_PolarQuant:
        // Key (32*96 + 32*8*4) + Value (same) = ~6,656 bytes per KV head
        bytesPerPage = 6656ULL * kvHeads;
        break;
    }

    uint64_t totalBytes = bytesPerPage * pages * numLayers;
    double gb = static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0);
    double baselineBytes = (16512ULL * kvHeads) * pages * numLayers;
    double ratio = baselineBytes / std::max(1.0, static_cast<double>(totalBytes));

    return FootprintResult{
        .totalBytes = totalBytes,
        .gigabytes = gb,
        .compressionRatio = ratio,
    };
  }

  // MiniCache (NeurIPS 2024): Layer-Pair Direction & Norm Representation
  struct MiniCachePair {
    std::array<float, kHeadDimension> sharedDirection{}; // Unit direction vector (L2 norm = 1.0)
    float scaleL = 1.0f;                                 // Magnitude for layer L
    float scaleLPlus1 = 1.0f;                            // Magnitude for layer L+1
    float pairCosine = 1.0f;                             // Cosine similarity between original layers
  };

  // Euclidean norm
  static float l2Norm(std::span<const float, kHeadDimension> v) {
    float sumSq = 0.0f;
    for (uint32_t i = 0; i < kHeadDimension; ++i) {
      sumSq += v[i] * v[i];
    }
    return std::sqrt(sumSq);
  }

  // Cosine similarity between two vectors
  static float cosineSimilarity(std::span<const float, kHeadDimension> u,
                                std::span<const float, kHeadDimension> v) {
    float dot = 0.0f;
    float normU = 0.0f;
    float normV = 0.0f;
    for (uint32_t i = 0; i < kHeadDimension; ++i) {
      dot += u[i] * v[i];
      normU += u[i] * u[i];
      normV += v[i] * v[i];
    }
    float denom = std::sqrt(normU) * std::sqrt(normV);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
  }

  // Spherical Linear Interpolation (SLERP) between unit direction vectors
  static void slerp(std::span<const float, kHeadDimension> u,
                    std::span<const float, kHeadDimension> v,
                    float t,
                    std::span<float, kHeadDimension> out) {
    float dot = 0.0f;
    for (uint32_t i = 0; i < kHeadDimension; ++i) {
      dot += u[i] * v[i];
    }
    float cosOmega = std::clamp(dot, -1.0f, 1.0f);

    if (std::abs(cosOmega) > 0.9995f) {
      // Linear interpolation when vectors are nearly collinear
      float outNorm = 0.0f;
      for (uint32_t i = 0; i < kHeadDimension; ++i) {
        out[i] = (1.0f - t) * u[i] + t * v[i];
        outNorm += out[i] * out[i];
      }
      float invNorm = (outNorm > 1e-8f) ? (1.0f / std::sqrt(outNorm)) : 1.0f;
      for (uint32_t i = 0; i < kHeadDimension; ++i) {
        out[i] *= invNorm;
      }
      return;
    }

    float omega = std::acos(cosOmega);
    float sinOmega = std::sin(omega);
    float coeffU = std::sin((1.0f - t) * omega) / sinOmega;
    float coeffV = std::sin(t * omega) / sinOmega;

    float outNorm = 0.0f;
    for (uint32_t i = 0; i < kHeadDimension; ++i) {
      out[i] = coeffU * u[i] + coeffV * v[i];
      outNorm += out[i] * out[i];
    }
    float invNorm = (outNorm > 1e-8f) ? (1.0f / std::sqrt(outNorm)) : 1.0f;
    for (uint32_t i = 0; i < kHeadDimension; ++i) {
      out[i] *= invNorm;
    }
  }

  // Merge adjacent layer pair (L, L+1) into a single MiniCachePair
  static MiniCachePair mergeLayerPair(std::span<const float, kHeadDimension> vecL,
                                      std::span<const float, kHeadDimension> vecLPlus1) {
    MiniCachePair pair{};
    pair.scaleL = l2Norm(vecL);
    pair.scaleLPlus1 = l2Norm(vecLPlus1);
    pair.pairCosine = cosineSimilarity(vecL, vecLPlus1);

    std::array<float, kHeadDimension> dirL{};
    std::array<float, kHeadDimension> dirLPlus1{};
    float invL = (pair.scaleL > 1e-8f) ? (1.0f / pair.scaleL) : 0.0f;
    float invLPlus1 = (pair.scaleLPlus1 > 1e-8f) ? (1.0f / pair.scaleLPlus1) : 0.0f;

    for (uint32_t i = 0; i < kHeadDimension; ++i) {
      dirL[i] = vecL[i] * invL;
      dirLPlus1[i] = vecLPlus1[i] * invLPlus1;
    }

    // Midpoint SLERP (t = 0.5) to find balanced shared directional vector
    slerp(dirL, dirLPlus1, 0.5f, pair.sharedDirection);
    return pair;
  }

  // Reconstruct vector for layer L or L+1 from MiniCachePair
  static void reconstructLayer(const MiniCachePair& pair,
                               bool isLayerL,
                               std::span<float, kHeadDimension> out) {
    float scale = isLayerL ? pair.scaleL : pair.scaleLPlus1;
    for (uint32_t i = 0; i < kHeadDimension; ++i) {
      out[i] = pair.sharedDirection[i] * scale;
    }
  }

  // Memory footprint calculator with MiniCache cross-layer sharing
  static FootprintResult calculateMiniCacheMemoryFootprint(
      uint32_t contextTokens,
      uint32_t kvHeads,
      uint32_t totalLayers,
      uint32_t mergeStartLayer,
      QuantFormat format) {
    uint32_t pages = (contextTokens + kPageTokens - 1) / kPageTokens;
    uint64_t bytesPerPage = 0;
    switch (format) {
      case QuantFormat::Q8_Standard:
        bytesPerPage = 16512ULL * kvHeads;
        break;
      case QuantFormat::Q4_Packed:
        bytesPerPage = 9216ULL * kvHeads;
        break;
      case QuantFormat::Q3_PolarQuant:
        bytesPerPage = 6656ULL * kvHeads;
        break;
    }

    uint32_t unmergedLayers = std::min(totalLayers, mergeStartLayer);
    uint32_t mergedPairs = (totalLayers > mergeStartLayer) ? ((totalLayers - mergeStartLayer + 1) / 2) : 0;
    uint32_t effectiveStoredLayers = unmergedLayers + mergedPairs;

    uint64_t totalBytes = bytesPerPage * pages * effectiveStoredLayers;
    totalBytes += static_cast<uint64_t>(mergedPairs) * pages * kPageTokens * kvHeads * sizeof(float);

    double gb = static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0);
    double baselineBytes = (16512ULL * kvHeads) * pages * totalLayers;
    double ratio = baselineBytes / std::max(1.0, static_cast<double>(totalBytes));

    return FootprintResult{
        .totalBytes = totalBytes,
        .gigabytes = gb,
        .compressionRatio = ratio,
    };
  }
};

} // namespace splash::engine::kv
