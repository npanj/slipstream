#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

namespace splash::engine {

#pragma pack(push, 1)
struct PersistentGdnExtentHeader {
  uint32_t magic = 0x5347444E; // 'SGDN' (Slipstream GDN)
  uint32_t version = 1;
  uint64_t kvBlockId = 0;
  uint32_t layerCount = 0;
  uint32_t boundaryTokens = 0;
  uint64_t convolutionBytes = 0;
  uint64_t recurrentBytes = 0;
  uint64_t checksum = 0;
};
#pragma pack(pop)

/**
 * Persistent GDN Recurrent State Splicer
 *
 * Persists and restores recurrent GDN linear attention states and convolution
 * buffers to/from APFS disk-backed extents, enabling persistent prefix caching
 * across server restarts with zero GPU prefill recomputation.
 */
class PersistentGdnExtent final {
public:
  static constexpr uint32_t kMagic = 0x5347444E;
  static constexpr uint32_t kCurrentVersion = 1;

  PersistentGdnExtent() = default;

  PersistentGdnExtent(uint64_t kvBlockId, uint32_t layers, uint32_t boundaryTokens,
                      size_t convBytes, size_t recBytes)
      : kvBlockId_(kvBlockId), layers_(layers), boundaryTokens_(boundaryTokens),
        convBytes_(convBytes), recBytes_(recBytes) {
    totalPayloadBytes_ = (convBytes_ + recBytes_) * layers_;
    storage_.resize(sizeof(PersistentGdnExtentHeader) + totalPayloadBytes_, 0);
    initHeader();
  }

  [[nodiscard]] uint64_t kvBlockId() const noexcept { return kvBlockId_; }
  [[nodiscard]] uint32_t layers() const noexcept { return layers_; }
  [[nodiscard]] uint32_t boundaryTokens() const noexcept { return boundaryTokens_; }
  [[nodiscard]] size_t convBytesPerLayer() const noexcept { return convBytes_; }
  [[nodiscard]] size_t recBytesPerLayer() const noexcept { return recBytes_; }
  [[nodiscard]] size_t totalBytes() const noexcept { return storage_.size(); }

  void* layerConvData(uint32_t layer) noexcept {
    if (layer >= layers_) return nullptr;
    size_t offset = sizeof(PersistentGdnExtentHeader) + layer * (convBytes_ + recBytes_);
    return storage_.data() + offset;
  }

  const void* layerConvData(uint32_t layer) const noexcept {
    if (layer >= layers_) return nullptr;
    size_t offset = sizeof(PersistentGdnExtentHeader) + layer * (convBytes_ + recBytes_);
    return storage_.data() + offset;
  }

  void* layerRecurrentData(uint32_t layer) noexcept {
    if (layer >= layers_) return nullptr;
    size_t offset = sizeof(PersistentGdnExtentHeader) + layer * (convBytes_ + recBytes_) + convBytes_;
    return storage_.data() + offset;
  }

  const void* layerRecurrentData(uint32_t layer) const noexcept {
    if (layer >= layers_) return nullptr;
    size_t offset = sizeof(PersistentGdnExtentHeader) + layer * (convBytes_ + recBytes_) + convBytes_;
    return storage_.data() + offset;
  }

  bool saveToFile(const std::string& path) const {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    // Fast kernel write
    ssize_t written = ::write(fd, storage_.data(), storage_.size());
    ::close(fd);
    return written == static_cast<ssize_t>(storage_.size());
  }

  bool loadFromFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(PersistentGdnExtentHeader))) {
      ::close(fd);
      return false;
    }

    storage_.resize(static_cast<size_t>(st.st_size));
    ssize_t rd = ::read(fd, storage_.data(), storage_.size());
    ::close(fd);

    if (rd != static_cast<ssize_t>(storage_.size())) {
      storage_.clear();
      return false;
    }

    const auto* hdr = reinterpret_cast<const PersistentGdnExtentHeader*>(storage_.data());
    if (hdr->magic != kMagic || hdr->version != kCurrentVersion) {
      storage_.clear();
      return false;
    }

    kvBlockId_ = hdr->kvBlockId;
    layers_ = hdr->layerCount;
    boundaryTokens_ = hdr->boundaryTokens;
    convBytes_ = static_cast<size_t>(hdr->convolutionBytes);
    recBytes_ = static_cast<size_t>(hdr->recurrentBytes);
    totalPayloadBytes_ = (convBytes_ + recBytes_) * layers_;

    return true;
  }

  [[nodiscard]] bool verifyIntegrity() const noexcept {
    if (storage_.size() < sizeof(PersistentGdnExtentHeader)) return false;
    const auto* hdr = reinterpret_cast<const PersistentGdnExtentHeader*>(storage_.data());
    if (hdr->magic != kMagic || hdr->version != kCurrentVersion) return false;
    return storage_.size() == sizeof(PersistentGdnExtentHeader) + totalPayloadBytes_;
  }

private:
  void initHeader() noexcept {
    auto* hdr = reinterpret_cast<PersistentGdnExtentHeader*>(storage_.data());
    hdr->magic = kMagic;
    hdr->version = kCurrentVersion;
    hdr->kvBlockId = kvBlockId_;
    hdr->layerCount = layers_;
    hdr->boundaryTokens = boundaryTokens_;
    hdr->convolutionBytes = convBytes_;
    hdr->recurrentBytes = recBytes_;
    hdr->checksum = 0;
  }

  uint64_t kvBlockId_ = 0;
  uint32_t layers_ = 0;
  uint32_t boundaryTokens_ = 0;
  size_t convBytes_ = 0;
  size_t recBytes_ = 0;
  size_t totalPayloadBytes_ = 0;
  std::vector<uint8_t> storage_;
};

} // namespace splash::engine
