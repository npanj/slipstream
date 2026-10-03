#pragma once

#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::model {

inline constexpr uint32_t kQ4GroupElements = 64;

// The per-layer embedding's rows are 160 wide, which is not a whole number of
// 64-element groups. Finer groups divide it exactly, keep each row's scales to
// itself, and leave row starts group-aligned; the alternatives were sharing a
// scale between unrelated vocabulary rows or padding every row to 192.
inline constexpr uint32_t kQ4FineGroupElements = 32;
inline constexpr uint64_t kBFloat16Bytes = 2;

inline constexpr uint64_t kWeightFileAlignment = 16 * 1024;
inline constexpr uint32_t kQ4StorageN = 256;

// Expert projections tile at a narrower width. Qwen3.8-Flash-Next's experts
// are 640 wide, which is not a multiple of 256 but is a multiple of 128, so
// tiling them at 128 stores them exactly instead of padding 640 up to 768 -
// about 12.7 GiB of zeros across that model's 48 layers. The Metal tiles
// already take this width as a template parameter and 128 is a whole number
// of TileN=32 slices, so only the packed layout and this check change.
inline constexpr uint32_t kQ4ExpertStorageN = 128;

class WeightStoreError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct WeightFileRecord final {
  std::string relativePath;
  std::string magic;
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t declaredBytes = 0;
};

enum class MemoryAdvice {
  Normal,
  Random,
  Sequential,
  WillNeed,
  DontNeed,
};

// A read-only mmap with one no-copy Metal base buffer.  Sections are checked,
// aligned views that retain the mapping; no model loader owns raw mmap state.
class WeightFile final {
public:
  WeightFile(metal::MetalBackend &backend, std::filesystem::path path,
             std::string relativePath, std::string_view expectedMagic,
             uint32_t expectedLayer, uint32_t expectedType);
  ~WeightFile();

  WeightFile(const WeightFile &) = delete;
  WeightFile &operator=(const WeightFile &) = delete;

  [[nodiscard]] metal::MetalBuffer section(uint64_t bytes,
                                            std::string_view label = {});
  void finish();
  [[nodiscard]] const WeightFileRecord &record() const noexcept;
  [[nodiscard]] uint64_t bytes() const noexcept;
  // Start of the file's mapping; a section's file offset is its contents()
  // minus this.
  [[nodiscard]] const uint8_t *mappedBase() const noexcept;
  void advise(MemoryAdvice advice) const noexcept;
  void adviseRange(uint64_t offset, uint64_t bytes, MemoryAdvice advice) const noexcept;
  void prefetch(bool touch = false) const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t checkedWeightMultiply(uint64_t left, uint64_t right,
                                             std::string_view description);
[[nodiscard]] uint64_t
q4PackedBytes(uint32_t outputSize, uint32_t inputSize,
              uint32_t groupElements = kQ4GroupElements);
void validateQ4Layout(uint32_t outputSize, uint32_t inputSize,
                      uint32_t storageN = kQ4StorageN);

[[nodiscard]] ops::Q4Projection
readQ4Projection(WeightFile &file, metal::MetalBackend &backend,
                 uint32_t outputSize, uint32_t inputSize,
                 std::string_view label,
                 uint32_t storageN = kQ4StorageN);

// Embedding weights, scales and biases are independently aligned sections
// so token gather can bind each table directly.
[[nodiscard]] ops::Q4Projection
readQ4ProjectionComponents(WeightFile &file, uint32_t outputSize,
                           uint32_t inputSize, std::string_view label,
                           uint32_t groupElements = kQ4GroupElements);

[[nodiscard]] ops::Q8Projection
readQ8Projection(WeightFile &file, metal::MetalBackend &backend,
                 uint32_t outputSize, uint32_t inputSize,
                 std::string_view label);

[[nodiscard]] ops::Q8Projection
readQ8ProjectionComponents(WeightFile &file, uint32_t outputSize,
                           uint32_t inputSize, std::string_view label);

[[nodiscard]] ops::ExpertQ4Projection
readExpertQ4Projection(WeightFile &file, uint32_t experts,
                       uint32_t outputSize, uint32_t inputSize,
                       std::string_view label,
                       uint32_t storageN = kQ4StorageN);

[[nodiscard]] std::string
weightManifestFingerprint(std::span<const WeightFileRecord> records);

} // namespace splash::model
