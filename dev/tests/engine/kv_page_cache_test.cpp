#include "engine/KvCache.hpp"
#include "TestKvPool.hpp"

#include <array>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <random>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
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

CacheNamespace cacheNamespace(uint8_t salt = 0x5a) {
  CacheNamespace result;
  result.digest.fill(salt);
  return result;
}

std::array<uint32_t, KvCache::pageTokens> page(uint32_t token) {
  std::array<uint32_t, KvCache::pageTokens> result{};
  result.fill(token);
  return result;
}

void testImageIdentityKeysBlocks() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(2, false);
  require(acquired.granted() && acquired.pages.size() == 2,
          "test pages were not acquired");

  // Two images with the same grid and placeholder tokens differ only by
  // content digest; text-only lookups carry the zero identity.
  const ImageSpan red{8, 16, 8, 8, 0x1111, 0x2222};
  const ImageSpan blue{8, 16, 8, 8, 0x3333, 0x4444};
  const std::array<ImageSpan, 1> redSpans{red};
  const std::array<ImageSpan, 1> blueSpans{blue};
  require(blockImageIdentity(0, 32, {}) == ImageIdentity{},
          "text-only block identity must be zero");
  require(blockImageIdentity(32, 32, redSpans) == ImageIdentity{},
          "a block after the image must not carry image identity");
  const ImageIdentity redIdentity = blockImageIdentity(0, 32, redSpans);
  const ImageIdentity blueIdentity = blockImageIdentity(0, 32, blueSpans);
  require(redIdentity != ImageIdentity{} && redIdentity != blueIdentity,
          "image identity does not depend on the content digest");
  const ImageSpan redLater{40, 16, 8, 8, 0x1111, 0x2222};
  const std::array<ImageSpan, 1> laterSpans{redLater};
  require(blockImageIdentity(32, 32, laterSpans) != redIdentity,
          "image identity ignores the block's alignment inside the image");

  const auto tokens = page(248056);
  auto redBlock = cache.insert(0, tokens, acquired.pages[0], redIdentity);
  require(redBlock.inserted, "image block was not inserted");
  require(!cache.find(0, tokens), "text-only lookup matched an image block");
  require(!cache.find(0, tokens, blueIdentity),
          "a different image matched a token-identical block");
  auto found = cache.find(0, tokens, redIdentity);
  require(found && found->id == redBlock.id,
          "identical image content did not match its block");
  auto blueBlock = cache.insert(0, tokens, acquired.pages[1], blueIdentity);
  require(blueBlock.inserted && blueBlock.id != redBlock.id,
          "token-identical blocks with different images were merged");
  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
}

void testExactChainedBlocksAndPhysicalOwnership() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(4, false);
  require(acquired.granted() && acquired.pages.size() == 4,
          "test pages were not acquired");

  const auto rootTokens = page(11);
  const auto leftTokens = page(12);
  const auto rightTokens = page(13);
  const auto otherTokens = page(14);
  auto root = cache.insert(0, rootTokens, acquired.pages[0]);
  auto left = cache.insert(root.id, leftTokens, acquired.pages[1]);
  auto right = cache.insert(root.id, rightTokens, acquired.pages[2]);
  auto other = cache.insert(0, otherTokens, acquired.pages[3]);
  require(root.inserted && left.inserted && right.inserted && other.inserted &&
              cache.snapshot().blocks == 4 &&
              cache.snapshot().bytes == 400,
          "page cache did not retain four physical blocks");

  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  require(pool.snapshot().pagesPrefix == 4 &&
              pool.snapshot().pagesActive == 0,
          "page cache ownership was not isolated from active requests");

  auto foundLeft = cache.find(root.id, leftTokens);
  auto wrongTokens = leftTokens;
  wrongTokens.back() ^= 1;
  require(foundLeft && foundLeft->id == left.id &&
              foundLeft->physicalPage == acquired.pages[1] &&
              !cache.find(root.id, wrongTokens),
          "page lookup trusted a hash collision");
  const KvCache::Chain chain = cache.chain(left.id);
  require(chain.blocks == std::vector<uint64_t>{root.id, left.id} &&
              chain.pages == std::vector<uint32_t>{acquired.pages[0],
                                                   acquired.pages[1]} &&
              cache.chainLength(left.id) == 2,
          "page cache did not reconstruct the exact parent chain");

  auto duplicate = cache.insert(root.id, leftTokens, acquired.pages[1]);
  require(!duplicate.inserted && duplicate.id == left.id &&
              duplicate.physicalPage == acquired.pages[1] &&
              cache.snapshot().blocks == 4,
          "exact duplicate created a second KV block");

  requireThrows<std::logic_error>([&] { cache.erase(root.id); },
                                  "parent block was evicted before children");
  cache.retainActive(left.id);
  require(cache.evictionCandidate().value().id == right.id &&
              cache.evictionCandidate(right.id).value().id == other.id &&
              !cache.evictionCandidate(other.id),
          "active page block remained evictable");
  cache.releaseActive(left.id);
  require(cache.evictionCandidate().value().id == right.id &&
              cache.evictionCandidate(right.id).value().id == other.id &&
              cache.evictionCandidate(other.id).value().id == left.id &&
              !cache.evictionCandidate(left.id),
          "leaf eviction did not follow deterministic LRU order");
  cache.touch(right.id);
  require(cache.evictionCandidate().value().id == other.id &&
              cache.evictionCandidate(other.id).value().id == left.id &&
              cache.evictionCandidate(left.id).value().id == right.id &&
              !cache.evictionCandidate(right.id),
          "touch did not refresh the matched leaf without walking its chain");

  cache.erase(right.id);
  cache.erase(left.id);
  cache.erase(root.id);
  cache.erase(other.id);
  require(cache.snapshot().blocks == 0 &&
              pool.snapshot().pagesPrefix == 0 &&
              pool.freePageCount() == pool.pageCount(),
          "page cache erase leaked physical references");
}

void testErasedLeafParentInheritsRecency() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(4, false);
  require(acquired.granted() && acquired.pages.size() == 4,
          "test pages were not acquired");

  // Chain A is inserted first and never touched again; chain B is newer.
  const auto a0Tokens = page(21);
  const auto a1Tokens = page(22);
  const auto b0Tokens = page(23);
  const auto b1Tokens = page(24);
  auto a0 = cache.insert(0, a0Tokens, acquired.pages[0]);
  auto a1 = cache.insert(a0.id, a1Tokens, acquired.pages[1]);
  auto b0 = cache.insert(0, b0Tokens, acquired.pages[2]);
  auto b1 = cache.insert(b0.id, b1Tokens, acquired.pages[3]);
  for (uint32_t physical : acquired.pages) pool.releasePage(physical, false);
  require(cache.evictionCandidate().value().id == a1.id &&
              cache.evictionCandidate(a1.id).value().id == b1.id,
          "leaf order did not start with the cold chain");

  // Erasing A's leaf exposes A's root, which is still colder than B's leaf.
  cache.erase(a1.id);
  require(cache.evictionCandidate().value().id == a0.id &&
              cache.evictionCandidate(a0.id).value().id == b1.id &&
              !cache.evictionCandidate(b1.id),
          "exposed parent jumped ahead of a warmer chain");
  cache.erase(a0.id);
  cache.erase(b1.id);
  require(cache.evictionCandidate().value().id == b0.id &&
              !cache.evictionCandidate(b0.id),
          "last exposed parent was not the sole candidate");
  cache.erase(b0.id);
  require(cache.snapshot().blocks == 0 &&
              pool.freePageCount() == pool.pageCount(),
          "recency test leaked physical references");
}

void testInputValidation() {
  test::TestKvBacking backing(2, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(1, false);
  require(acquired.granted(), "validation page was not acquired");
  const std::array<uint32_t, 1> shortTokens{1};
  requireThrows<std::invalid_argument>(
      [&] {
        static_cast<void>(
            cache.insert(0, shortTokens, acquired.pages[0]));
      },
      "partial token page was accepted");
  requireThrows<std::invalid_argument>(
      [&] {
        static_cast<void>(
            cache.insert(999, page(2), acquired.pages[0]));
      },
      "nonresident parent was accepted");
  pool.releasePage(acquired.pages[0], false);
}

void testCandidateOrderThroughChurn() {
  constexpr uint32_t count = 256;
  test::TestKvBacking backing(count, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);
  auto acquired = pool.acquirePages(count, false);
  require(acquired.granted(), "churn test pages were not acquired");
  struct Reference {
    uint64_t parent = 0;
    uint64_t lastUsed = 0;
    uint32_t children = 0;
    uint32_t activeUsers = 0;
    bool resident = true;
  };
  std::array<Reference, count + 1> entries{};
  uint64_t clock = 0;
  std::mt19937 random(571);
  for (uint32_t index = 0; index < count; ++index) {
    const uint64_t parent = index % 3 ? 1 + random() % index : 0;
    const auto inserted = cache.insert(parent, page(index + 1),
                                       acquired.pages[index]);
    require(inserted.id == index + 1, "unexpected test block identity");
    entries[inserted.id] = {parent, ++clock};
    if (parent) ++entries[parent].children;
    pool.releasePage(acquired.pages[index], false);
  }
  auto checkOrder = [&] {
    std::vector<std::pair<uint64_t, uint64_t>> expected;
    for (uint64_t id = 1; id <= count; ++id) {
      const auto &entry = entries[id];
      if (entry.resident && !entry.activeUsers && !entry.children)
        expected.emplace_back(entry.lastUsed, id);
    }
    std::sort(expected.begin(), expected.end());
    auto candidate = cache.evictionCandidate();
    for (const auto &[lastUsed, id] : expected) {
      require(candidate && candidate->id == id &&
                  candidate->lastUsed == lastUsed,
              "candidate index disagrees with exact reference recency");
      candidate = cache.evictionCandidate(id);
    }
    require(!candidate, "candidate index retained a pinned/non-leaf block");
  };
  for (uint32_t step = 0; step < 8192; ++step) {
    const uint64_t id = 1 + random() % count;
    auto &entry = entries[id];
    if (!entry.resident) continue;
    switch (random() % 3) {
    case 0:
      cache.touch(id);
      entry.lastUsed = ++clock;
      break;
    case 1:
      if (!entry.activeUsers) {
        cache.retainActive(id);
        ++entry.activeUsers;
        entry.lastUsed = ++clock;
      } else {
        cache.releaseActive(id);
        --entry.activeUsers;
        if (!entry.children) entry.lastUsed = ++clock;
      }
      break;
    case 2:
      if (entry.activeUsers || entry.children) break;
      cache.erase(id);
      entry.resident = false;
      if (entry.parent) {
        auto &parent = entries[entry.parent];
        --parent.children;
        if (!parent.children && !parent.activeUsers)
          parent.lastUsed = std::max(parent.lastUsed, entry.lastUsed);
      }
      break;
    }
    checkOrder();
  }
  for (uint64_t id = 1; id <= count; ++id) {
    if (entries[id].resident && entries[id].activeUsers)
      cache.releaseActive(id);
  }
  while (auto candidate = cache.evictionCandidate()) cache.erase(candidate->id);
  require(cache.snapshot().blocks == 0 && pool.freePageCount() == count,
          "candidate churn leaked a block or physical reference");
}

void testHashCollisionStillRequiresExactTokens() {
  const auto left = page(7);
  auto right = left;
  right.back() = 8;
  constexpr uint64_t forcedCollision = 0x12345678;
  const KvBlockKeyView stored{11, forcedCollision, left, {}};
  const KvBlockKeyView colliding{11, forcedCollision, right, {}};
  const KvBlockKeyView exact{11, forcedCollision, left, {}};
  require(!exactKvBlockKeyMatch(stored, colliding) &&
              exactKvBlockKeyMatch(stored, exact),
          "KV block matching trusted a colliding index hash");
}

void testSpliceChainPreservesPrefixAndSuffixPages() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);

  auto acquired = pool.acquirePages(4, false);
  require(acquired.granted() && acquired.pages.size() == 4, "acquire pages failed");

  // Create chain: Root (Block 1, page 0) -> Turn 1 (Block 2, page 1) -> Turn 2 (Block 3, page 2) -> Turn 3 (Block 4, page 3)
  auto r1 = cache.insert(0, page(101), acquired.pages[0]);
  auto r2 = cache.insert(r1.id, page(102), acquired.pages[1]);
  auto r3 = cache.insert(r2.id, page(103), acquired.pages[2]);
  auto r4 = cache.insert(r3.id, page(104), acquired.pages[3]);

  // Splice: Skip middle turn 1 & 2 (blocks 2 & 3), keeping root (block 1) and turn 3 (block 4)
  auto spliced = cache.spliceChain(r1.id, r4.id);
  // Root has 1 block, r4 chain has 4 blocks -> total 5 blocks in spliced chain
  require(spliced.pages.size() == 5, "splice chain size mismatch");
  require(spliced.pages[0] == acquired.pages[0], "root page must match");
  require(spliced.pages.back() == acquired.pages[3], "suffix page must match");
}

void testVaultChainExport() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);

  auto acquired = pool.acquirePages(3, false);
  require(acquired.granted() && acquired.pages.size() == 3, "acquire pages failed");

  auto b1 = cache.insert(0, page(201), acquired.pages[0]);
  auto b2 = cache.insert(b1.id, page(202), acquired.pages[1]);
  auto b3 = cache.insert(b2.id, page(203), acquired.pages[2]);

  auto exported = cache.vaultChain(b3.id);
  require(exported.size() == 3, "vault chain size must be 3");
  require(exported[0].id == b1.id, "first vaulted block must be b1");
  require(exported[1].id == b2.id, "second vaulted block must be b2");
  require(exported[2].id == b3.id, "third vaulted block must be b3");
  require(exported[0].tokens[0] == 201, "tokens must match exported block");
  require(exported[2].tokens[0] == 203, "tokens must match exported block");
}

void testForkBranchCoW() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);

  auto acquired = pool.acquirePages(3, false);
  require(acquired.granted() && acquired.pages.size() == 3, "acquire pages failed");

  auto root = cache.insert(0, page(301), acquired.pages[0]);
  auto b1 = cache.insert(root.id, page(302), acquired.pages[1]);
  auto b2 = cache.insert(b1.id, page(303), acquired.pages[2]);

  // Before fork: b2 is evictable leaf
  require(cache.evictionCandidate().value().id == b2.id, "b2 must be eviction candidate before fork");

  // Fork CoW branch from b2
  uint64_t forkId = cache.forkBranch(b2.id);
  require(forkId == b2.id, "forkId must match blockId");

  // While branch is active, the chain is pinned and not evictable
  require(!cache.evictionCandidate().has_value(), "forked chain must be protected from eviction");

  // Release branch
  cache.releaseChain(forkId);
  require(cache.evictionCandidate().value().id == b2.id, "b2 must be eviction candidate after release");
}

void testAttentionSaliencyTracking() {
  test::TestKvBacking backing(8, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);

  auto acquired = pool.acquirePages(3, false);
  require(acquired.granted() && acquired.pages.size() == 3, "acquire pages failed");

  auto root = cache.insert(0, page(401), acquired.pages[0]);
  auto toolOutput = cache.insert(root.id, page(402), acquired.pages[1]);
  auto finalAnswer = cache.insert(toolOutput.id, page(403), acquired.pages[2]);

  // Default saliency is 1.0f
  require(cache.blockSaliency(toolOutput.id) == 1.0f, "initial saliency must be 1.0f");

  // Record low attention weight on tool output multiple times
  cache.recordSaliency(toolOutput.id, 0.05f);
  cache.recordSaliency(toolOutput.id, 0.05f);
  cache.recordSaliency(toolOutput.id, 0.05f);

  // Record high attention weight on final answer
  cache.recordSaliency(finalAnswer.id, 0.95f);

  require(cache.blockSaliency(toolOutput.id) < 0.6f, "tool output saliency must have decayed");
  require(cache.blockSaliency(finalAnswer.id) > 0.9f, "final answer saliency must remain high");

  // Query low-saliency blocks below threshold 0.7f
  auto lowBlocks = cache.lowSaliencyBlocks(finalAnswer.id, 0.7f);
  require(lowBlocks.size() == 1, "exactly one low-saliency block expected");
  require(lowBlocks[0] == toolOutput.id, "low saliency block must be toolOutput");
}

void testPermanentRootPrefixPinning() {
  test::TestKvBacking backing(5, 100);
  KvPool pool(backing);
  CacheRecency recency;
  KvCache cache(pool, cacheNamespace(), recency);

  auto acquired = pool.acquirePages(3, false);
  require(acquired.granted() && acquired.pages.size() == 3, "acquire pages failed");

  auto root = cache.insert(0, page(501), acquired.pages[0]);
  auto turn1 = cache.insert(root.id, page(502), acquired.pages[1]);

  require(!cache.isRootPinned(root.id), "root should not be pinned initially");

  // Pin root chain: permanently locks root in RAM
  cache.pinRootChain(root.id);
  require(cache.isRootPinned(root.id), "root must now be pinned");

  // Turn 1 leaf is evictable, but pinned root must NEVER be evictable
  cache.erase(turn1.id);

  // After turn1 is erased, root becomes a leaf. But because it is pinned,
  // evictionCandidate() must NOT return it!
  auto candidate = cache.evictionCandidate();
  require(!candidate.has_value(), "pinned root must not be an eviction candidate");

  // Direct erase of pinned root must be rejected
  requireThrows<std::logic_error>(
      [&] { cache.erase(root.id); },
      "cannot evict a pinned KV cache root block");

  // Unpin root chain restores normal eviction eligibility
  cache.unpinRootChain(root.id);
  require(!cache.isRootPinned(root.id), "root must be unpinned");

  auto unpinnedCandidate = cache.evictionCandidate();
  require(unpinnedCandidate.has_value() && unpinnedCandidate->id == root.id,
          "unpinned root must now be an eviction candidate");

  cache.erase(root.id);
  require(cache.snapshot().blocks == 0, "cache must be empty after erasing unpinned root");
}

} // namespace

int main() {
  try {
    testExactChainedBlocksAndPhysicalOwnership();
    testImageIdentityKeysBlocks();
    testErasedLeafParentInheritsRecency();
    testInputValidation();
    testCandidateOrderThroughChurn();
    testHashCollisionStillRequiresExactTokens();
    testSpliceChainPreservesPrefixAndSuffixPages();
    testVaultChainExport();
    testForkBranchCoW();
    testAttentionSaliencyTracking();
    testPermanentRootPrefixPinning();
    std::cout << "KV page cache tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "KV page cache test failed: " << error.what() << '\n';
    return 1;
  }
}


