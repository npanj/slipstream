#pragma once

#include <cstdint>
#include <fcntl.h>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace splash::ops {

#pragma pack(push, 1)
struct RepoTokenIndexHeader {
  uint32_t magic = 0x534C5452; // 'SLTR' (Slipstream Token Repository)
  uint32_t version = 1;
  uint32_t totalTokens = 0;
  uint32_t fileCount = 0;
  uint64_t timestamp = 0;
};
#pragma pack(pop)

class RepoTokenIndex final {
public:
  static constexpr uint32_t kMagic = 0x534C5452;
  static constexpr uint32_t kCurrentVersion = 1;

  RepoTokenIndex() = default;
  ~RepoTokenIndex() { close(); }

  RepoTokenIndex(const RepoTokenIndex&) = delete;
  RepoTokenIndex& operator=(const RepoTokenIndex&) = delete;

  RepoTokenIndex(RepoTokenIndex&& other) noexcept {
    *this = std::move(other);
  }
  RepoTokenIndex& operator=(RepoTokenIndex&& other) noexcept {
    if (this != &other) {
      close();
      fd_ = other.fd_;
      data_ = other.data_;
      size_ = other.size_;
      header_ = other.header_;
      other.fd_ = -1;
      other.data_ = nullptr;
      other.size_ = 0;
      other.header_ = nullptr;
    }
    return *this;
  }

  bool open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      return false;
    }
    struct stat st{};
    if (::fstat(fd_, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(RepoTokenIndexHeader))) {
      close();
      return false;
    }
    size_ = static_cast<size_t>(st.st_size);
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      close();
      return false;
    }
    header_ = reinterpret_cast<const RepoTokenIndexHeader*>(data_);
    if (header_->magic != kMagic || header_->version != kCurrentVersion) {
      close();
      return false;
    }
    size_t expectedSize = sizeof(RepoTokenIndexHeader) + static_cast<size_t>(header_->totalTokens) * sizeof(uint32_t);
    if (size_ < expectedSize) {
      close();
      return false;
    }
    return true;
  }

  void close() noexcept {
    if (data_ && data_ != MAP_FAILED) {
      ::munmap(data_, size_);
      data_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    size_ = 0;
    header_ = nullptr;
  }

  [[nodiscard]] bool isOpen() const noexcept { return data_ != nullptr; }
  [[nodiscard]] uint32_t totalTokens() const noexcept { return header_ ? header_->totalTokens : 0; }
  [[nodiscard]] uint32_t fileCount() const noexcept { return header_ ? header_->fileCount : 0; }
  [[nodiscard]] uint64_t timestamp() const noexcept { return header_ ? header_->timestamp : 0; }

  [[nodiscard]] std::span<const uint32_t> tokens() const noexcept {
    if (!isOpen() || !header_ || header_->totalTokens == 0) {
      return {};
    }
    const uint32_t* tokenData = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(data_) + sizeof(RepoTokenIndexHeader));
    return std::span<const uint32_t>(tokenData, header_->totalTokens);
  }

private:
  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
  const RepoTokenIndexHeader* header_ = nullptr;
};

} // namespace splash::ops
