#include "engine/PrefixCache.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using namespace splash;
using namespace splash::engine;
using namespace splash::engine::pcache;

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void requireThrows(Function &&function, const char *message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

std::filesystem::path temporaryRoot(const char *name) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("slipstream-prefix-cache-" + std::string(name) + "-" +
                     std::to_string(::getpid()));
  std::error_code error;
  std::filesystem::remove_all(root, error);
  return root;
}

Key salt(uint64_t value) { return {0x0123456789ABCDEFull, value}; }

std::vector<uint8_t> payloadOf(uint64_t seed, size_t bytes) {
  std::vector<uint8_t> payload(bytes);
  uint64_t state = seed * 6364136223846793005ull + 1442695040888963407ull;
  for (auto &byte : payload) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    byte = static_cast<uint8_t>(state >> 33);
  }
  return payload;
}

Store::Options options(const std::filesystem::path &root, uint64_t quota = 64 * 1024 * 1024) {
  Store::Options result;
  result.root = root;
  result.ns = namespaceDigest({"test-model", "test-drafter", "v1"});
  result.quota = quota;
  return result;
}

// ------------------------------------------------------------------ keys

void testKeysAreStableAndDistinct() {
  const std::vector<uint32_t> tokens{1, 2, 3, 4, 5};
  const Key first = tokenKey(tokens, tokens.size());
  const Key second = tokenKey(tokens, tokens.size());
  require(first == second, "the same tokens must give the same key");
  require(tokenKey(tokens, 4) != first,
          "a shorter prefix must not collide with the whole sequence");
  const std::vector<uint32_t> other{1, 2, 3, 4, 6};
  require(tokenKey(other, other.size()) != first,
          "different tokens must give different keys");
  // Length mixing: {"ab","c"} and {"a","bc"} must differ.
  require(namespaceDigest({"ab", "c"}) != namespaceDigest({"a", "bc"}),
          "namespace digest ignored field boundaries");
  require(namespaceDigest({"x"}) == namespaceDigest({"x"}),
          "namespace digest is not deterministic");
}

void testHexRoundTrip() {
  for (uint64_t value = 0; value < 32; ++value) {
    const Key key = salt(value);
    const std::string text = key.hex();
    require(text.size() == 32, "key hex must be 32 characters");
    Key parsed;
    require(Key::parse(text, &parsed), "a written key must parse");
    require(parsed == key, "key hex round trip changed the key");
  }
  Key ignored;
  require(!Key::parse("", &ignored), "an empty key must not parse");
  require(!Key::parse("xyz", &ignored), "a short key must not parse");
  require(!Key::parse(std::string(32, 'g'), &ignored), "a non-hex key must not parse");
  require(!Key::parse(std::string(31, 'a'), &ignored), "a truncated key must not parse");
  Key upper;
  require(Key::parse(std::string(32, 'A'), &upper) && upper.hi != 0,
          "uppercase hex must parse");
}

// ------------------------------------------------------------------ store

void testRoundTrip() {
  const auto root = temporaryRoot("roundtrip");
  {
    Store store(options(root));
    require(store.enabled(), "the store did not open");
    require(store.bytes() == 0, "a fresh store is not empty");
    for (uint64_t index = 0; index < 8; ++index) {
      const auto payload = payloadOf(index, 1000 + 137 * index);
      require(store.put(Kind::KvBlock, salt(index), payload), "put failed");
      require(store.bytes() == store.stats().bytesWritten,
              "byte accounting disagrees with what was written");
    }
    for (uint64_t index = 0; index < 8; ++index) {
      const auto expected = payloadOf(index, 1000 + 137 * index);
      const auto loaded = store.get(Kind::KvBlock, salt(index));
      require(loaded.has_value(), "a written entry was not found");
      require(*loaded == expected, "an entry changed on the way to disk");
    }
    require(store.stats().hits == 8, "hits were not counted");
    require(store.keys(Kind::KvBlock).size() == 8, "keys() lost entries");
    require(store.keys(Kind::State).empty(), "kinds leaked into each other");
    // Kinds are separate namespaces even for the same key.
    require(!store.get(Kind::State, salt(0)).has_value(),
            "a KV block was found under the state kind");
    require(!store.get(Kind::Entry, salt(0)).has_value(),
            "a KV block was found under the entry kind");
  }
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testMissesAndErase() {
  const auto root = temporaryRoot("miss");
  Store store(options(root));
  require(!store.get(Kind::KvBlock, salt(1)).has_value(), "an empty store returned data");
  require(store.stats().misses == 1, "a miss was not counted");
  require(store.put(Kind::State, salt(1), payloadOf(1, 64)), "put failed");
  require(store.erase(Kind::State, salt(1)), "erase failed");
  require(!store.erase(Kind::State, salt(1)), "erasing twice must report false");
  require(!store.get(Kind::State, salt(1)).has_value(), "an erased entry came back");
  require(store.bytes() == 0, "bytes were not returned after erase");
  store.put(Kind::Entry, salt(2), payloadOf(2, 32));
  store.clear();
  require(store.bytes() == 0 && store.keys(Kind::Entry).empty(),
          "clear() left entries behind");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testOverwriteKeepsAccounting() {
  const auto root = temporaryRoot("overwrite");
  Store store(options(root));
  const auto small = payloadOf(1, 100);
  const auto large = payloadOf(2, 5000);
  require(store.put(Kind::KvBlock, salt(1), small), "put failed");
  const uint64_t afterSmall = store.bytes();
  require(store.put(Kind::KvBlock, salt(1), large), "overwrite failed");
  require(store.bytes() > afterSmall, "an overwrite did not grow the store");
  const auto loaded = store.get(Kind::KvBlock, salt(1));
  require(loaded && *loaded == large, "an overwrite returned the old payload");
  require(store.keys(Kind::KvBlock).size() == 1, "an overwrite duplicated a key");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testEmptyPayloadAndLimits() {
  const auto root = temporaryRoot("limits");
  Store store(options(root));
  require(store.put(Kind::KvBlock, salt(1), {}), "an empty payload must round trip");
  const auto loaded = store.get(Kind::KvBlock, salt(1));
  require(loaded && loaded->empty(), "an empty payload came back wrong");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

// ------------------------------------------------------------- durability

void testCorruptionIsAMiss() {
  const auto root = temporaryRoot("corrupt");
  const Key key = salt(7);
  {
    Store store(options(root));
    require(store.put(Kind::KvBlock, key, payloadOf(7, 4096)), "put failed");
  }
  const auto path = root / options(root).ns.hex() / "kv" / key.hex();

  // Flip one payload byte.
  {
    std::FILE *file = std::fopen(path.c_str(), "r+b");
    require(file != nullptr, "the entry file is missing");
    require(std::fseek(file, 4096, SEEK_SET) == 0, "seek failed");
    std::fputc(0x5A, file);
    std::fclose(file);
  }
  {
    Store store(options(root));
    require(!store.get(Kind::KvBlock, key).has_value(),
            "a corrupt payload was returned as data");
    require(store.stats().corrupt == 1, "corruption was not counted");
  }

  // Change a header field without touching its checksum.
  {
    Store store(options(root));
    require(store.put(Kind::KvBlock, key, payloadOf(7, 4096)), "rewrite failed");
  }
  {
    std::FILE *file = std::fopen(path.c_str(), "r+b");
    require(file != nullptr, "the entry file is missing");
    require(std::fseek(file, 32, SEEK_SET) == 0, "seek failed");
    std::fputc(0x01, file);  // key.hi
    std::fclose(file);
  }
  {
    Store store(options(root));
    require(!store.get(Kind::KvBlock, key).has_value(),
            "a corrupt header was returned as data");
  }

  // A truncated file (header says more than the disk holds).
  {
    Store store(options(root));
    require(store.put(Kind::KvBlock, key, payloadOf(7, 4096)), "rewrite failed");
  }
  std::error_code error;
  std::filesystem::resize_file(path, 128, error);
  require(!error, "the test could not truncate the entry");
  {
    Store store(options(root));
    require(!store.get(Kind::KvBlock, key).has_value(),
            "a truncated file was returned as data");
  }
  std::filesystem::remove_all(root, error);
}

void testForeignNamespaceAndFormat() {
  const auto root = temporaryRoot("foreign");
  const Key key = salt(3);
  // Written by a store that thinks it is another model.
  {
    auto other = options(root);
    other.ns = namespaceDigest({"another-model", "test-drafter", "v1"});
    Store store(other);
    require(store.put(Kind::KvBlock, key, payloadOf(3, 256)), "put failed");
  }
  {
    Store store(options(root));
    require(!store.get(Kind::KvBlock, key).has_value(),
            "a file from another namespace was read back");
  }
  // An unknown format version is a miss, not a crash.
  const auto path = root / options(root).ns.hex() / "kv" / key.hex();
  std::filesystem::create_directories(path.parent_path());
  std::FILE *file = std::fopen(path.c_str(), "wb");
  require(file != nullptr, "could not create a file");
  std::fputs("not a cache file at all", file);
  std::fclose(file);
  {
    Store store(options(root));
    require(!store.get(Kind::KvBlock, key).has_value(),
            "a foreign file was read back");
    require(store.stats().corrupt == 1, "a foreign file was not counted as corrupt");
  }
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testReopenFindsEntries() {
  const auto root = temporaryRoot("reopen");
  const Key key = salt(11);
  std::vector<uint8_t> expected;
  {
    Store store(options(root));
    expected = payloadOf(11, 2048);
    require(store.put(Kind::State, key, expected), "put failed");
  }
  {
    // The server restarted: the index is rebuilt from the directory.
    Store store(options(root));
    require(store.enabled(), "the store did not reopen");
    require(store.keys(Kind::State).size() == 1, "a restart lost the entry");
    const auto loaded = store.get(Kind::State, key);
    require(loaded && *loaded == expected, "a restart lost the payload");
    // 64 bytes of header and the payload, counted from the directory rather
    // than remembered from this process's writes.
    require(store.bytes() == 64 + expected.size(),
            "the rebuilt byte count is wrong");
  }
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testNoTemporaryFilesRemain() {
  const auto root = temporaryRoot("temporary");
  {
    Store store(options(root));
    for (uint64_t index = 0; index < 4; ++index)
      require(store.put(Kind::Entry, salt(index), payloadOf(index, 128)), "put failed");
  }
  size_t files = 0;
  for (const auto &item :
       std::filesystem::directory_iterator(root / options(root).ns.hex() / "e"))
    files += item.is_regular_file() ? 1 : 0;
  require(files == 4, "a temporary file was left behind");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testCrashLeftoversAreIgnored() {
  const auto root = temporaryRoot("leftover");
  const Key key = salt(9);
  {
    Store store(options(root));
    require(store.put(Kind::KvBlock, key, payloadOf(9, 512)), "put failed");
  }
  // A process killed mid-write leaves its temporary file behind. It is not an
  // entry: it must not be listed, counted, or served.
  const auto directory = root / options(root).ns.hex() / "kv";
  {
    std::FILE *leftover = std::fopen((directory / ".tmp-deadbeef").c_str(), "wb");
    require(leftover != nullptr, "could not create a leftover file");
    std::fputs("partial write from a killed process", leftover);
    std::fclose(leftover);
  }
  {
    Store store(options(root));
    require(store.keys(Kind::KvBlock).size() == 1,
            "a leftover temporary file was listed as an entry");
    require(store.bytes() == 64 + 512,
            "a leftover temporary file was counted as cached bytes");
    const auto loaded = store.get(Kind::KvBlock, key);
    require(loaded && loaded->size() == 512, "the real entry was disturbed");
  }
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testLargePayloadAndAccounting() {
  const auto root = temporaryRoot("large");
  Store store(options(root, 64 * 1024 * 1024));
  // A payload the size of a saved recurrent state rather than a KV row.
  const auto big = payloadOf(0x5EED, 8 * 1024 * 1024);
  require(store.put(Kind::State, salt(1), big), "a large payload failed to write");
  const auto loaded = store.get(Kind::State, salt(1));
  require(loaded && *loaded == big, "a large payload changed on the way to disk");
  // The byte count is what the directory holds, not what the process thinks it
  // wrote: the two must agree after a reopen.
  uint64_t onDisk = 0;
  for (const auto &item : std::filesystem::recursive_directory_iterator(
           root / options(root).ns.hex())) {
    if (item.is_regular_file() && item.path().filename().string().size() == 32)
      onDisk += item.file_size();
  }
  require(onDisk == store.bytes(), "bytes() disagrees with the directory");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testSecondServerGetsADisabledStore() {
  const auto root = temporaryRoot("lock");
  Store first(options(root));
  require(first.enabled(), "the first store did not open");
  Store second(options(root));
  require(!second.enabled(),
          "a second store opened a namespace another process is using");
  require(!second.put(Kind::KvBlock, salt(1), payloadOf(1, 16)),
          "a disabled store accepted a write");
  require(!second.get(Kind::KvBlock, salt(1)).has_value(),
          "a disabled store served a read");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

// ------------------------------------------------------------------ quota

void testQuotaEvictsOldestFirst() {
  const auto root = temporaryRoot("quota");
  // Room for about three 4 KiB entries plus their headers.
  const uint64_t entryBytes = 4096 + 64;
  Store store(options(root, entryBytes * 3 + entryBytes / 2));
  for (uint64_t index = 0; index < 6; ++index) {
    require(store.put(Kind::KvBlock, salt(index), payloadOf(index, 4096)), "put failed");
    require(store.bytes() <= entryBytes * 3 + entryBytes / 2,
            "the store grew past its quota");
  }
  const auto remaining = store.keys(Kind::KvBlock);
  require(remaining.size() == 3, "the quota did not bound the entry count");
  // The three newest survive; the three oldest were evicted.
  for (uint64_t index = 0; index < 3; ++index)
    require(std::find(remaining.begin(), remaining.end(), salt(index)) == remaining.end(),
            "the oldest entry was not the one evicted");
  for (uint64_t index = 3; index < 6; ++index)
    require(std::find(remaining.begin(), remaining.end(), salt(index)) != remaining.end(),
            "a recent entry was evicted before an older one");
  require(store.stats().evictions == 3, "evictions were not counted");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testReadingRefreshesEvictionOrder() {
  const auto root = temporaryRoot("recency");
  const uint64_t entryBytes = 1024 + 64;
  Store store(options(root, entryBytes * 2 + entryBytes / 2));
  require(store.put(Kind::KvBlock, salt(0), payloadOf(0, 1024)), "put failed");
  require(store.put(Kind::KvBlock, salt(1), payloadOf(1, 1024)), "put failed");
  require(store.get(Kind::KvBlock, salt(0)).has_value(), "a hit was not served");
  require(store.put(Kind::KvBlock, salt(2), payloadOf(2, 1024)), "put failed");
  require(store.get(Kind::KvBlock, salt(0)).has_value(),
          "a recently used entry was evicted before an older one");
  require(!store.get(Kind::KvBlock, salt(1)).has_value(),
          "the least recently used entry survived");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testSetQuotaShrinks() {
  const auto root = temporaryRoot("shrink");
  Store store(options(root, 1024 * 1024));
  for (uint64_t index = 0; index < 8; ++index)
    require(store.put(Kind::KvBlock, salt(index), payloadOf(index, 8192)), "put failed");
  store.setQuota(3 * (8192 + 64));
  require(store.bytes() <= 3 * (8192 + 64), "shrinking the quota kept too much");
  require(store.keys(Kind::KvBlock).size() == 3, "shrinking the quota kept too many entries");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testSpaceLimitMath() {
  // A volume with 100 GiB free, nothing cached, and the default 32 GiB quota:
  // a quarter of what is free above the floor.
  require(spaceLimit(kAutoQuota, 100 * kGiB, 0, 500 * kGiB) ==
              (100 * kGiB - kFloor) / 4,
          "spaceLimit ignored the free-space rule");
  // The cache's own files do not shrink its own limit.
  require(spaceLimit(kAutoQuota, 5 * kGiB, 20 * kGiB, 500 * kGiB) ==
              (25 * kGiB - kFloor) / 4,
          "spaceLimit counted the cache's own files");
  // A volume with almost nothing free yields nothing.
  require(spaceLimit(kAutoQuota, kGiB, 0, 500 * kGiB) == 0,
          "spaceLimit did not clamp at zero");
  // The quota caps the result.
  require(spaceLimit(2 * kGiB, 400 * kGiB, 0, 1000 * kGiB) == 2 * kGiB,
          "spaceLimit exceeded the quota");
  // Small volumes use a twentieth as their floor.
  require(spaceLimit(kAutoQuota, 20 * kGiB, 0, 100 * kGiB) == (20 * kGiB - 5 * kGiB) / 4,
          "spaceLimit did not scale the floor to the volume");
}

void testSizeParsing() {
  uint64_t bytes = 0;
  require(parseSize("16G", &bytes) && bytes == 16 * kGiB, "16G");
  require(parseSize("512M", &bytes) && bytes == 512 * (kGiB / 1024), "512M");
  require(parseSize("4GiB", &bytes) && bytes == 4 * kGiB, "4GiB");
  require(parseSize("1024", &bytes) && bytes == 1024, "bare bytes");
  require(parseSize("1.5T", &bytes) && bytes == kGiB * 1024 + kGiB * 512, "1.5T");
  require(parseSize("0", &bytes) && bytes == 0, "zero");
  require(!parseSize("", &bytes), "an empty size must not parse");
  require(!parseSize("sixteen", &bytes), "a word must not parse");
  require(!parseSize("16G ", &bytes), "a trailing space must not parse");
  require(!parseSize("G", &bytes), "a bare suffix must not parse");
  require(!parseSize("-1G", &bytes), "a negative size must not parse");
  require(gib(kGiB / 2) == "512.0 MiB", "gib() formatting of MiB");
  require(gib(4 * kGiB) == "4.0 GiB", "gib() formatting of GiB");
}

// -------------------------------------------------------------- concurrency

void testConcurrentUse() {
  const auto root = temporaryRoot("threads");
  Store store(options(root, 1 << 20));
  std::atomic<uint32_t> failures{0};
  std::vector<std::thread> threads;
  for (uint64_t thread = 0; thread < 4; ++thread) {
    threads.emplace_back([&store, &failures, thread] {
      for (uint64_t index = 0; index < 32; ++index) {
        const Key key = salt(thread * 100 + index);
        const auto payload = payloadOf(thread * 100 + index, 256);
        if (!store.put(Kind::KvBlock, key, payload)) {
          ++failures;
          continue;
        }
        const auto loaded = store.get(Kind::KvBlock, key);
        if (!loaded || *loaded != payload)
          ++failures;
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  require(failures == 0, "concurrent use lost or corrupted an entry");
  std::error_code error;
  std::filesystem::remove_all(root, error);
}

void testDisabledStoreDoesNothing() {
  const auto root = temporaryRoot("disabled");
  auto configuration = options(root);
  configuration.enabled = false;
  Store store(configuration);
  require(!store.enabled(), "a disabled store reported itself enabled");
  require(!store.put(Kind::KvBlock, salt(1), payloadOf(1, 32)),
          "a disabled store wrote an entry");
  require(!store.get(Kind::KvBlock, salt(1)).has_value(),
          "a disabled store served a read");
  require(store.bytes() == 0, "a disabled store accounted bytes");
  require(!std::filesystem::exists(root / configuration.ns.hex()),
          "a disabled store created directories");
}

} // namespace

int main() {
  try {
    testKeysAreStableAndDistinct();
    testHexRoundTrip();
    testRoundTrip();
    testMissesAndErase();
    testOverwriteKeepsAccounting();
    testEmptyPayloadAndLimits();
    testCorruptionIsAMiss();
    testForeignNamespaceAndFormat();
    testReopenFindsEntries();
    testNoTemporaryFilesRemain();
    testCrashLeftoversAreIgnored();
    testLargePayloadAndAccounting();
    testSecondServerGetsADisabledStore();
    testQuotaEvictsOldestFirst();
    testReadingRefreshesEvictionOrder();
    testSetQuotaShrinks();
    testSpaceLimitMath();
    testSizeParsing();
    testConcurrentUse();
    testDisabledStoreDoesNothing();
    std::cout << "Prefix cache tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Prefix cache test failed: " << error.what() << '\n';
    return 1;
  }
}
