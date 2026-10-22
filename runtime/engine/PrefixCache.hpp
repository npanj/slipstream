#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// SSD prefix cache: the store half of L1 in docs/roadmap.md.
//
// A conversation's KV rows and its recurrent/draft states are written as files
// so that a later request whose prompt starts with that conversation reads them
// back instead of prefilling again - including after the server restarts or
// after the memory governor evicted the live copy.
//
// This file is deliberately CPU only and model agnostic: it knows keys, files,
// checksums and space, never tensors. The engine decides what to save and when
// and hands the bytes over; that keeps the store testable without a GPU
// (dev/tests/engine/prefix_cache_test.cpp) and keeps the policy in one place.
//
// Layout under the root, one directory per namespace so that no model can read
// another's cache:
//   <root>/<ns>/lock        held (flock) by the server using the namespace
//   <root>/<ns>/kv/<key>    one block of KV rows
//   <root>/<ns>/st/<key>    one saved state (recurrent state, draft ring)
//   <root>/<ns>/e/<key>     one entry: a conversation's tokens and its parts
//
// Every file is a 64-byte header (magic, version, kind, namespace, key, payload
// size, CRC-32C of the payload and of the header) followed by the payload. A
// file is written to a temporary name and renamed into place, so a reader never
// sees a half-written entry; every read verifies both checksums, and any
// mismatch is a miss rather than bad data. Reads and writes use F_NOCACHE on
// macOS so that a 30 GB cache does not evict the weights it is trying to help.
namespace splash::engine::pcache {

constexpr uint64_t kGiB = uint64_t{1} << 30;
constexpr uint64_t kAutoQuota = 32 * kGiB;
// Free space the cache never counts on: the model, the swap file and macOS
// itself live outside this cache and must not be squeezed by it.
constexpr uint64_t kFloor = 10 * kGiB;

// A 128-bit content key (file names: 32 hex digits).
struct Key final {
  uint64_t hi = 0;
  uint64_t lo = 0;

  [[nodiscard]] bool operator==(const Key &other) const noexcept {
    return hi == other.hi && lo == other.lo;
  }
  [[nodiscard]] bool operator!=(const Key &other) const noexcept {
    return !(*this == other);
  }
  [[nodiscard]] bool operator<(const Key &other) const noexcept {
    return hi != other.hi ? hi < other.hi : lo < other.lo;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return hi || lo; }

  [[nodiscard]] std::string hex() const;
  [[nodiscard]] static bool parse(std::string_view text, Key *key) noexcept;
};

// Incremental 128-bit hash: two independent 64-bit lanes, for content keys
// rather than security. Lengths are mixed in so that {"ab","c"} and {"a","bc"}
// are different keys.
class Hasher final {
public:
  Hasher &u64(uint64_t value) noexcept;
  Hasher &u32s(std::span<const uint32_t> values) noexcept;
  Hasher &bytes(const void *data, size_t size) noexcept;
  Hasher &str(std::string_view text) noexcept { return u64(text.size()).bytes(text.data(), text.size()); }
  Hasher &key(const Key &value) noexcept { return u64(value.hi).u64(value.lo); }
  [[nodiscard]] Key done() const noexcept;

private:
  uint64_t a_ = 0x9E3779B97F4A7C15ull;
  uint64_t b_ = 0xC2B2AE3D27D4EB4Full;
  uint64_t count_ = 0;
};

// CRC-32C (Castagnoli) over one buffer; hardware instructions on Apple silicon
// and on x86-64 with SSE4.2, a table otherwise. `crc` is the running value, so
// crc32c(0, p, n) covers a whole buffer.
uint32_t crc32c(uint32_t crc, const void *data, size_t size) noexcept;

enum class Kind : uint32_t {
  KvBlock = 1,
  State = 2,
  Entry = 3,
};

// The identity of a cache namespace: a digest over what a stored byte assumes
// about the engine (model files, drafter, state layouts, format version). A
// change that alters any of them must alter the digest, or old files would be
// read back as if they were new.
[[nodiscard]] Key namespaceDigest(std::initializer_list<std::string_view> parts);

// The key of a conversation prefix: every token up to and including `end`.
[[nodiscard]] Key tokenKey(std::span<const uint32_t> tokens, size_t end);

// "16G", "512M", "1.5T", "4GiB", or a bare byte count. False for anything else.
[[nodiscard]] bool parseSize(std::string_view text, uint64_t *bytes) noexcept;
// Human text for a byte count: "6.5 GiB", or "512.0 MiB" below 1 GiB.
[[nodiscard]] std::string gib(uint64_t bytes);

// The cache's effective limit: min(quota, (avail + held - floor) / 4), where
// floor is min(kFloor, volume / 20). `held` is what the cache's own files take,
// so its writes do not shrink its own limit, while other files filling the
// volume do. Clamped at zero.
[[nodiscard]] uint64_t spaceLimit(uint64_t quota, uint64_t available,
                                  uint64_t held, uint64_t volume) noexcept;

// What the store has done: enough to see whether the cache pays for itself.
struct Stats final {
  uint64_t writes = 0;      // entries written
  uint64_t writeFailures = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t corrupt = 0;     // files rejected by a checksum or a format check
  uint64_t evictions = 0;   // files removed to stay under the limit
  uint64_t bytesWritten = 0;
  uint64_t bytesEvicted = 0;
};

// A store over one namespace directory. Thread safe. Every I/O failure is
// reported as a miss or a false return, never as an exception: a broken cache
// must never fail a request, and must never stop the server from starting.
// The constructor takes the namespace's flock; a second process that finds the
// lock held gets a disabled store rather than a corrupted cache.
class Store final {
public:
  struct Options final {
    std::filesystem::path root;
    Key ns;
    uint64_t quota = kAutoQuota;
    bool enabled = true;
  };

  explicit Store(const Options &options);
  ~Store();

  Store(const Store &) = delete;
  Store &operator=(const Store &) = delete;
  Store(Store &&) = delete;
  Store &operator=(Store &&) = delete;

  // False when the cache is switched off, its root cannot be used, or another
  // process holds this namespace. Callers then simply work in memory.
  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  // Writes `payload` under (kind, key), replacing any previous copy. Returns
  // false when the write failed; the previous copy, if any, is left intact.
  bool put(Kind kind, const Key &key, std::span<const uint8_t> payload);

  // Reads an entry back. Nullopt for a miss, a checksum mismatch, a truncated
  // file or a file written by another namespace or format version.
  [[nodiscard]] std::optional<std::vector<uint8_t>> get(Kind kind, const Key &key) const;

  bool erase(Kind kind, const Key &key);
  void clear();

  // Keys present, for the engine's lookup index. Sorted for deterministic
  // behaviour in tests and logs.
  [[nodiscard]] std::vector<Key> keys(Kind kind) const;

  [[nodiscard]] uint64_t bytes() const noexcept;
  [[nodiscard]] Stats stats() const noexcept;
  [[nodiscard]] uint64_t quota() const noexcept;
  void setQuota(uint64_t bytes);

  // The limit currently in force: the quota narrowed by free space. Recomputed
  // by every write so that a filling volume shrinks the cache.
  [[nodiscard]] uint64_t limit() const;

  // Removes entries until the store is within `limit`.
  void enforceLimit();

private:
  struct Entry final {
    Kind kind = Kind::KvBlock;
    Key key;
    uint64_t bytes = 0;
    uint64_t stamp = 0;  // monotonic order of use; eviction takes the smallest
  };

  [[nodiscard]] std::filesystem::path directory(Kind kind) const;
  [[nodiscard]] std::filesystem::path file(Kind kind, const Key &key) const;
  void scan();
  void evictTo(uint64_t target);
  void touch(Entry &entry) const noexcept;
  [[nodiscard]] static uint64_t entryHash(Kind kind, const Key &key) noexcept;

  std::filesystem::path root_;
  Key ns_;
  uint64_t quota_;
  bool enabled_;
  int lockFd_ = -1;
  mutable std::mutex mutex_;
  // Bookkeeping the const readers update: a hit is a use, and an index and byte
  // count are derived from the directory, not part of the store's identity.
  mutable std::unordered_map<uint64_t, Entry> index_;
  mutable uint64_t bytes_ = 0;
  mutable uint64_t stamp_ = 0;
  mutable Stats stats_;
};

} // namespace splash::engine::pcache
