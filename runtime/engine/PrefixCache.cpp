#include "engine/PrefixCache.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace splash::engine::pcache {
namespace {

constexpr char kMagic[8] = {'S', 'L', 'S', 'P', 'C', 'F', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 64;
// A payload larger than this is treated as a corrupt length field rather than
// as an allocation request: no KV block or state comes close (the largest saved
// state on this engine is a few tens of MiB).
constexpr uint64_t kMaximumPayloadBytes = uint64_t{1} << 31;

void storeU32(uint8_t *out, uint32_t value) noexcept {
  out[0] = static_cast<uint8_t>(value);
  out[1] = static_cast<uint8_t>(value >> 8);
  out[2] = static_cast<uint8_t>(value >> 16);
  out[3] = static_cast<uint8_t>(value >> 24);
}

void storeU64(uint8_t *out, uint64_t value) noexcept {
  for (unsigned byte = 0; byte < 8; ++byte)
    out[byte] = static_cast<uint8_t>(value >> (8 * byte));
}

uint32_t loadU32(const uint8_t *in) noexcept {
  return uint32_t{in[0]} | (uint32_t{in[1]} << 8) | (uint32_t{in[2]} << 16) |
         (uint32_t{in[3]} << 24);
}

uint64_t loadU64(const uint8_t *in) noexcept {
  uint64_t value = 0;
  for (unsigned byte = 0; byte < 8; ++byte)
    value |= uint64_t{in[byte]} << (8 * byte);
  return value;
}

struct Header final {
  Kind kind = Kind::KvBlock;
  Key ns;
  Key key;
  uint64_t payloadBytes = 0;
  uint32_t payloadCrc = 0;
};

// Header layout (little endian, 64 bytes):
//   0  magic[8]   8  version   12 kind
//   16 ns.hi      24 ns.lo     32 key.hi   40 key.lo
//   48 payload bytes            56 payload CRC    60 header CRC
std::array<uint8_t, kHeaderBytes> encodeHeader(const Header &header) noexcept {
  std::array<uint8_t, kHeaderBytes> out{};
  std::memcpy(out.data(), kMagic, sizeof(kMagic));
  storeU32(out.data() + 8, kVersion);
  storeU32(out.data() + 12, static_cast<uint32_t>(header.kind));
  storeU64(out.data() + 16, header.ns.hi);
  storeU64(out.data() + 24, header.ns.lo);
  storeU64(out.data() + 32, header.key.hi);
  storeU64(out.data() + 40, header.key.lo);
  storeU64(out.data() + 48, header.payloadBytes);
  storeU32(out.data() + 56, header.payloadCrc);
  // The header's own checksum covers everything before it, with its own field
  // left zero, so a bit flip anywhere else in the header is caught.
  storeU32(out.data() + 60, 0);
  storeU32(out.data() + 60, crc32c(0, out.data(), kHeaderBytes));
  return out;
}

// False when the bytes are not this format at all; `header` is otherwise filled
// in but its payload checksum is only checked by the caller, against the bytes
// actually read.
bool decodeHeader(const std::array<uint8_t, kHeaderBytes> &raw, Header *header) noexcept {
  if (std::memcmp(raw.data(), kMagic, sizeof(kMagic)) != 0)
    return false;
  std::array<uint8_t, kHeaderBytes> checked = raw;
  const uint32_t storedCrc = loadU32(raw.data() + 60);
  storeU32(checked.data() + 60, 0);
  if (crc32c(0, checked.data(), kHeaderBytes) != storedCrc)
    return false;
  if (loadU32(raw.data() + 8) != kVersion)
    return false;
  const uint32_t kind = loadU32(raw.data() + 12);
  if (kind < static_cast<uint32_t>(Kind::KvBlock) ||
      kind > static_cast<uint32_t>(Kind::Entry))
    return false;
  header->kind = static_cast<Kind>(kind);
  header->ns = {loadU64(raw.data() + 16), loadU64(raw.data() + 24)};
  header->key = {loadU64(raw.data() + 32), loadU64(raw.data() + 40)};
  header->payloadBytes = loadU64(raw.data() + 48);
  header->payloadCrc = loadU32(raw.data() + 56);
  return header->payloadBytes <= kMaximumPayloadBytes;
}

int openRead(const std::filesystem::path &path) noexcept {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return -1;
#if defined(__APPLE__)
  // Do not let a 30 GB cache push the weights out of the page cache.
  (void)::fcntl(fd, F_NOCACHE, 1);
#endif
  return fd;
}

bool readExactly(int fd, void *buffer, size_t size, off_t offset) noexcept {
  uint8_t *cursor = static_cast<uint8_t *>(buffer);
  size_t done = 0;
  while (done < size) {
    const ssize_t got =
        ::pread(fd, cursor + done, size - done, offset + static_cast<off_t>(done));
    if (got < 0 && errno == EINTR)
      continue;
    if (got <= 0)
      return false;
    done += static_cast<size_t>(got);
  }
  return true;
}

bool writeExactly(int fd, const void *buffer, size_t size, off_t offset) noexcept {
  const uint8_t *cursor = static_cast<const uint8_t *>(buffer);
  size_t done = 0;
  while (done < size) {
    const ssize_t wrote =
        ::pwrite(fd, cursor + done, size - done, offset + static_cast<off_t>(done));
    if (wrote < 0 && errno == EINTR)
      continue;
    if (wrote <= 0)
      return false;
    done += static_cast<size_t>(wrote);
  }
  return true;
}

const char *kindDirectory(Kind kind) noexcept {
  switch (kind) {
  case Kind::KvBlock:
    return "kv";
  case Kind::State:
    return "st";
  case Kind::Entry:
    return "e";
  }
  return "e";
}

} // namespace

// ---------------------------------------------------------------- CRC-32C

uint32_t crc32c(uint32_t crc, const void *data, size_t size) noexcept {
#if defined(__ARM_FEATURE_CRC32)
  // Apple silicon's crc32 instructions: about an order of magnitude faster than
  // the table below, which matters when a 16K-turn's blocks are written out.
  const uint8_t *cursor = static_cast<const uint8_t *>(data);
  uint32_t value = crc;
  while (size >= 8) {
    uint64_t chunk = 0;
    std::memcpy(&chunk, cursor, sizeof(chunk));
    value = __crc32cd(value, chunk);
    cursor += 8;
    size -= 8;
  }
  while (size--)
    value = __crc32cb(value, *cursor++);
  return value;
#else
  // Slice-by-1 table, identical on every platform: the format must not depend
  // on which instructions a build happened to enable. A few hundred MB/s is
  // far above what an SSD-backed prefix cache moves.
  static const auto table = [] {
    std::array<uint32_t, 256> entries{};
    for (uint32_t index = 0; index < 256; ++index) {
      uint32_t value = index;
      for (unsigned bit = 0; bit < 8; ++bit)
        value = (value & 1u) ? (value >> 1) ^ 0x82F63B78u : value >> 1;
      entries[index] = value;
    }
    return entries;
  }();
  const uint8_t *cursor = static_cast<const uint8_t *>(data);
  uint32_t value = ~crc;
  for (size_t index = 0; index < size; ++index)
    value = table[(value ^ cursor[index]) & 0xFFu] ^ (value >> 8);
  return ~value;
#endif
}

// ---------------------------------------------------------------- Key

std::string Key::hex() const {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string text(32, '0');
  for (unsigned index = 0; index < 32; ++index) {
    const uint64_t source = index < 16 ? hi : lo;
    const unsigned shift = 60 - 4 * (index % 16);
    text[index] = kDigits[(source >> shift) & 0xFu];
  }
  return text;
}

bool Key::parse(std::string_view text, Key *key) noexcept {
  if (!key || text.size() != 32)
    return false;
  Key parsed;
  for (unsigned index = 0; index < 32; ++index) {
    const char digit = text[index];
    uint64_t value = 0;
    if (digit >= '0' && digit <= '9')
      value = static_cast<uint64_t>(digit - '0');
    else if (digit >= 'a' && digit <= 'f')
      value = static_cast<uint64_t>(digit - 'a' + 10);
    else if (digit >= 'A' && digit <= 'F')
      value = static_cast<uint64_t>(digit - 'A' + 10);
    else
      return false;
    if (index < 16)
      parsed.hi = (parsed.hi << 4) | value;
    else
      parsed.lo = (parsed.lo << 4) | value;
  }
  *key = parsed;
  return true;
}

// ---------------------------------------------------------------- Hasher

namespace {

inline uint64_t mix64(uint64_t value) noexcept {
  value ^= value >> 30;
  value *= 0xBF58476D1CE4E5B9ull;
  value ^= value >> 27;
  value *= 0x94D049BB133111EBull;
  value ^= value >> 31;
  return value;
}

} // namespace

Hasher &Hasher::u64(uint64_t value) noexcept {
  a_ = mix64(a_ ^ (value + 0x9E3779B97F4A7C15ull + count_));
  b_ = mix64(b_ + value * 0xC2B2AE3D27D4EB4Full + count_);
  ++count_;
  return *this;
}

Hasher &Hasher::u32s(std::span<const uint32_t> values) noexcept {
  u64(values.size());
  for (const uint32_t value : values)
    u64(value);
  return *this;
}

Hasher &Hasher::bytes(const void *data, size_t size) noexcept {
  const uint8_t *cursor = static_cast<const uint8_t *>(data);
  // Eight bytes at a time where possible, then the remainder.
  size_t done = 0;
  while (done + 8 <= size) {
    uint64_t chunk = 0;
    std::memcpy(&chunk, cursor + done, sizeof(chunk));
    u64(chunk);
    done += 8;
  }
  if (done < size) {
    uint64_t chunk = 0;
    std::memcpy(&chunk, cursor + done, size - done);
    u64(chunk);
  }
  return *this;
}

Key Hasher::done() const noexcept {
  // Fold the lane and the input count so that a prefix of another input cannot
  // collide with a complete one by construction.
  const uint64_t folded = mix64(a_ ^ mix64(b_ + count_ * 0x9E3779B97F4A7C15ull));
  return {folded, mix64(folded ^ 0xD6E8FEB86659FD93ull)};
}

Key namespaceDigest(std::initializer_list<std::string_view> parts) {
  Hasher hasher;
  for (const std::string_view part : parts)
    hasher.str(part);
  return hasher.done();
}

Key tokenKey(std::span<const uint32_t> tokens, size_t end) {
  const size_t bounded = std::min(end, tokens.size());
  Hasher hasher;
  hasher.u32s(tokens.first(bounded));
  return hasher.done();
}

// ---------------------------------------------------------------- sizes

bool parseSize(std::string_view text, uint64_t *bytes) noexcept {
  if (!bytes || text.empty())
    return false;
  uint64_t scale = 1;
  std::string_view number = text;
  const std::string_view suffixes[] = {"KiB", "MiB", "GiB", "TiB", "K", "M", "G", "T"};
  const uint64_t scales[] = {kGiB / (1024 * 1024), kGiB / 1024, kGiB, kGiB * 1024,
                             kGiB / (1024 * 1024), kGiB / 1024, kGiB, kGiB * 1024};
  for (size_t index = 0; index < std::size(suffixes); ++index) {
    const std::string_view suffix = suffixes[index];
    if (number.size() > suffix.size() &&
        number.substr(number.size() - suffix.size()) == suffix) {
      scale = scales[index];
      number.remove_suffix(suffix.size());
      break;
    }
  }
  if (number.empty() || number.find_first_not_of("0123456789.") != std::string_view::npos)
    return false;
  const std::string asString(number);
  errno = 0;
  char *end = nullptr;
  const double value = std::strtod(asString.c_str(), &end);
  if (errno != 0 || !end || *end != '\0' || value < 0.0)
    return false;
  const double scaled = value * static_cast<double>(scale);
  if (scaled > static_cast<double>(std::numeric_limits<uint64_t>::max()))
    return false;
  *bytes = static_cast<uint64_t>(scaled);
  return true;
}

std::string gib(uint64_t bytes) {
  char buffer[64];
  if (bytes >= kGiB) {
    std::snprintf(buffer, sizeof(buffer), "%.1f GiB",
                  static_cast<double>(bytes) / static_cast<double>(kGiB));
  } else {
    std::snprintf(buffer, sizeof(buffer), "%.1f MiB",
                  static_cast<double>(bytes) / static_cast<double>(kGiB / 1024));
  }
  return std::string(buffer);
}

uint64_t spaceLimit(uint64_t quota, uint64_t available, uint64_t held,
                    uint64_t volume) noexcept {
  const uint64_t floor = std::min(kFloor, volume / 20);
  const uint64_t spendable = std::min(available + held, std::numeric_limits<uint64_t>::max() - floor);
  const uint64_t headroom = spendable > floor ? spendable - floor : 0;
  return std::min(quota, headroom / 4);
}

// ---------------------------------------------------------------- Store

uint64_t Store::entryHash(Kind kind, const Key &key) noexcept {
  return mix64(key.hi ^ mix64(key.lo + static_cast<uint64_t>(kind) * 0x9E3779B97F4A7C15ull));
}

Store::Store(const Options &options)
    : root_(options.root / options.ns.hex()), ns_(options.ns),
      quota_(options.quota), enabled_(options.enabled) {
  if (!enabled_)
    return;
  std::error_code error;
  std::filesystem::create_directories(root_, error);
  if (error) {
    enabled_ = false;
    return;
  }
  for (const Kind kind : {Kind::KvBlock, Kind::State, Kind::Entry}) {
    std::filesystem::create_directories(directory(kind), error);
    if (error) {
      enabled_ = false;
      return;
    }
  }
  // One server per namespace. A second one runs with the cache off instead of
  // interleaving writes into files it does not own.
  lockFd_ = ::open((root_ / "lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (lockFd_ < 0 || ::flock(lockFd_, LOCK_EX | LOCK_NB) != 0) {
    if (lockFd_ >= 0)
      (void)::close(lockFd_);
    lockFd_ = -1;
    enabled_ = false;
    return;
  }
  scan();
}

Store::~Store() {
  if (lockFd_ >= 0) {
    (void)::flock(lockFd_, LOCK_UN);
    (void)::close(lockFd_);
  }
}

std::filesystem::path Store::directory(Kind kind) const {
  return root_ / kindDirectory(kind);
}

std::filesystem::path Store::file(Kind kind, const Key &key) const {
  return directory(kind) / key.hex();
}

void Store::scan() {
  std::lock_guard<std::mutex> guard(mutex_);
  bytes_ = 0;
  index_.clear();
  for (const Kind kind : {Kind::KvBlock, Kind::State, Kind::Entry}) {
    std::error_code error;
    for (const auto &item : std::filesystem::directory_iterator(directory(kind), error)) {
      if (error)
        break;
      if (!item.is_regular_file(error) || error)
        continue;
      Key key;
      if (!Key::parse(item.path().filename().string(), &key))
        continue;  // a temporary or a stray file is not an entry
      const uint64_t size = item.file_size(error);
      if (error)
        continue;
      Entry entry;
      entry.kind = kind;
      entry.key = key;
      entry.bytes = size;
      entry.stamp = ++stamp_;
      bytes_ += size;
      index_.emplace(entryHash(kind, key), entry);
    }
  }
}

void Store::touch(Entry &entry) const noexcept { entry.stamp = ++stamp_; }

bool Store::put(Kind kind, const Key &key, std::span<const uint8_t> payload) {
  if (!enabled_)
    return false;
  if (payload.size() > kMaximumPayloadBytes)
    return false;
  Header header;
  header.kind = kind;
  header.ns = ns_;
  header.key = key;
  header.payloadBytes = payload.size();
  header.payloadCrc = crc32c(0, payload.data(), payload.size());
  const auto raw = encodeHeader(header);

  const std::filesystem::path target = file(kind, key);
  std::string temporary = (directory(kind) / ".tmp-XXXXXX").string();
  const int fd = ::mkstemp(temporary.data());
  if (fd < 0) {
    std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.writeFailures;
    return false;
  }
#if defined(__APPLE__)
  (void)::fcntl(fd, F_NOCACHE, 1);
#endif
  bool ok = writeExactly(fd, raw.data(), raw.size(), 0) &&
            writeExactly(fd, payload.data(), payload.size(), kHeaderBytes);
  if (ok)
    ok = ::fsync(fd) == 0;
  const int closeResult = ::close(fd);
  ok = ok && closeResult == 0;
  if (!ok) {
    ::unlink(temporary.c_str());
    std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.writeFailures;
    return false;
  }
  // Rename is atomic within a directory: a reader sees the old file or the new
  // one, never a partial write.
  if (::rename(temporary.c_str(), target.c_str()) != 0) {
    ::unlink(temporary.c_str());
    std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.writeFailures;
    return false;
  }
  {
    std::lock_guard<std::mutex> guard(mutex_);
    const uint64_t hash = entryHash(kind, key);
    const auto found = index_.find(hash);
    if (found != index_.end()) {
      bytes_ -= std::min(bytes_, found->second.bytes);
      found->second.bytes = kHeaderBytes + payload.size();
      bytes_ += found->second.bytes;
      found->second.kind = kind;
      found->second.key = key;
      touch(found->second);
    } else {
      Entry entry;
      entry.kind = kind;
      entry.key = key;
      entry.bytes = kHeaderBytes + payload.size();
      entry.stamp = ++stamp_;
      bytes_ += entry.bytes;
      index_.emplace(hash, entry);
    }
    ++stats_.writes;
    stats_.bytesWritten += kHeaderBytes + payload.size();
  }
  enforceLimit();
  return true;
}

std::optional<std::vector<uint8_t>> Store::get(Kind kind, const Key &key) const {
  if (!enabled_)
    return std::nullopt;
  const std::filesystem::path path = file(kind, key);
  const int fd = openRead(path);
  if (fd < 0) {
    std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.misses;
    return std::nullopt;
  }
  std::array<uint8_t, kHeaderBytes> raw{};
  Header header;
  bool valid = readExactly(fd, raw.data(), raw.size(), 0) &&
               decodeHeader(raw, &header) &&
               header.kind == kind && header.ns == ns_ && header.key == key;
  std::vector<uint8_t> payload;
  if (valid) {
    payload.resize(static_cast<size_t>(header.payloadBytes));
    valid = readExactly(fd, payload.data(), payload.size(), kHeaderBytes) &&
            crc32c(0, payload.data(), payload.size()) == header.payloadCrc;
  }
  (void)::close(fd);
  if (!valid) {
    std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.corrupt;
    return std::nullopt;
  }
  {
    std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.hits;
    const auto found = index_.find(entryHash(kind, key));
    if (found != index_.end())
      touch(found->second);
  }
  return payload;
}

bool Store::erase(Kind kind, const Key &key) {
  if (!enabled_)
    return false;
  std::error_code error;
  const bool removed = std::filesystem::remove(file(kind, key), error);
  if (!removed || error)
    return false;
  std::lock_guard<std::mutex> guard(mutex_);
  const auto found = index_.find(entryHash(kind, key));
  if (found != index_.end()) {
    bytes_ -= std::min(bytes_, found->second.bytes);
    index_.erase(found);
  }
  return true;
}

void Store::clear() {
  if (!enabled_)
    return;
  for (const Kind kind : {Kind::KvBlock, Kind::State, Kind::Entry}) {
    std::error_code error;
    for (const auto &item : std::filesystem::directory_iterator(directory(kind), error)) {
      if (error)
        break;
      std::error_code ignored;
      std::filesystem::remove(item.path(), ignored);
    }
  }
  std::lock_guard<std::mutex> guard(mutex_);
  index_.clear();
  bytes_ = 0;
}

std::vector<Key> Store::keys(Kind kind) const {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<Key> result;
  for (const auto &[hash, entry] : index_) {
    if (entry.kind == kind)
      result.push_back(entry.key);
  }
  std::sort(result.begin(), result.end());
  return result;
}

uint64_t Store::bytes() const noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  return bytes_;
}

Stats Store::stats() const noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  return stats_;
}

uint64_t Store::quota() const noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  return quota_;
}

void Store::setQuota(uint64_t bytes) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    quota_ = bytes;
  }
  enforceLimit();
}

uint64_t Store::limit() const {
  uint64_t quota = 0;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    quota = quota_;
  }
  struct statvfs status {};
  if (::statvfs(root_.c_str(), &status) != 0)
    return quota;
  const uint64_t available = uint64_t{status.f_bavail} * status.f_frsize;
  const uint64_t volume = uint64_t{status.f_blocks} * status.f_frsize;
  return spaceLimit(quota, available, bytes(), volume);
}

void Store::enforceLimit() {
  if (!enabled_)
    return;
  const uint64_t target = limit();
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (bytes_ <= target)
      return;
  }
  evictTo(target);
}

void Store::evictTo(uint64_t target) {
  // Oldest use first. The vector of candidates is small (thousands at most) and
  // eviction only happens when a write crosses the limit, so the scan is not on
  // any hot path.
  for (;;) {
    Entry victim;
    bool found = false;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (bytes_ <= target || index_.empty())
        return;
      for (const auto &[hash, entry] : index_) {
        if (!found || entry.stamp < victim.stamp) {
          victim = entry;
          found = true;
        }
      }
    }
    if (!found)
      return;
    std::error_code error;
    const bool removed = std::filesystem::remove(file(victim.kind, victim.key), error);
    std::lock_guard<std::mutex> guard(mutex_);
    const auto foundEntry = index_.find(entryHash(victim.kind, victim.key));
    // Only drop the entry we chose: a write that landed meanwhile owns a newer
    // stamp and its file must not be removed from under it.
    if (foundEntry != index_.end() && foundEntry->second.stamp == victim.stamp) {
      bytes_ -= std::min(bytes_, foundEntry->second.bytes);
      index_.erase(foundEntry);
    }
    if (removed && !error) {
      ++stats_.evictions;
      stats_.bytesEvicted += victim.bytes;
    }
  }
}

} // namespace splash::engine::pcache
