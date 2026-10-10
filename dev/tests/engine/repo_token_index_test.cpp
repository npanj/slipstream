#include "runtime/ops/RepoTokenIndex.hpp"
#include "runtime/ops/PromptLookup.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace splash::ops {

namespace {

void testRepoTokenIndexWriteAndRead() {
  std::cout << "[testRepoTokenIndexWriteAndRead] Starting..." << std::endl;

  const std::string testPath = "/tmp/test_slipstream_repo_tokens.idx";

  // Synthesize repository tokens representing codebase symbols
  // e.g. "namespace splash { class KvCache final { void pinRootChain() ... "
  std::vector<uint32_t> testTokens;
  for (uint32_t i = 1000; i < 6000; ++i) {
    testTokens.push_back(i);
  }
  // Inject some repeated structural identifiers
  std::vector<uint32_t> pattern = {4201, 4202, 4203, 4204, 4205};
  testTokens.insert(testTokens.end(), pattern.begin(), pattern.end());
  testTokens.push_back(9999);

  // Write index file
  {
    RepoTokenIndexHeader header;
    header.magic = RepoTokenIndex::kMagic;
    header.version = RepoTokenIndex::kCurrentVersion;
    header.totalTokens = static_cast<uint32_t>(testTokens.size());
    header.fileCount = 12;
    header.timestamp = 1728390000ULL;

    std::ofstream out(testPath, std::ios::binary);
    assert(out.is_open() && "Failed to open test index file for write");
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(testTokens.data()),
              testTokens.size() * sizeof(uint32_t));
  }

  // Open via mmap
  RepoTokenIndex index;
  bool ok = index.open(testPath);
  assert(ok && "RepoTokenIndex::open must succeed");
  assert(index.isOpen() && "Index must report open");
  assert(index.totalTokens() == testTokens.size() && "Token count mismatch");
  assert(index.fileCount() == 12 && "File count mismatch");
  assert(index.timestamp() == 1728390000ULL && "Timestamp mismatch");

  auto span = index.tokens();
  assert(span.size() == testTokens.size() && "Token span size mismatch");
  for (size_t i = 0; i < testTokens.size(); ++i) {
    assert(span[i] == testTokens[i] && "Token value mismatch");
  }

  // Ingest mmap span into PromptLookup
  PromptLookup pld;
  auto t0 = std::chrono::high_resolution_clock::now();
  pld.ingestReferenceTokens(span);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ingestMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "  Ingested " << span.size() << " repo tokens in " << ingestMs << " ms" << std::endl;
  assert(pld.size() == testTokens.size() && "PLD size must match repo tokens");

  // Propose speculative tokens based on query pattern
  std::array<uint32_t, 3> query = {4201, 4202, 4203};
  std::array<uint32_t, 3> outDrafts{};
  uint32_t proposed = pld.propose(query, outDrafts, 3);
  std::cout << "  Proposed " << proposed << " drafts: [" << outDrafts[0] << ", "
            << outDrafts[1] << "]" << std::endl;
  assert(proposed >= 2 && "Must propose at least 2 tokens");
  assert(outDrafts[0] == 4204 && "First draft must be 4204");
  assert(outDrafts[1] == 4205 && "Second draft must be 4205");

  index.close();
  ::unlink(testPath.c_str());

  std::cout << "[testRepoTokenIndexWriteAndRead] PASS" << std::endl;
}

void testPromptLookupMultiBranch() {
  std::cout << "[testPromptLookupMultiBranch] Starting..." << std::endl;

  PromptLookup pld;
  // Synthesize token stream with branching continuations:
  // Prefix: [100, 200, 300] -> branch A: [501, 502, 503]
  // Prefix: [100, 200, 300] -> branch B: [601, 602, 603]
  // Prefix: [100, 200, 300] -> branch C: [701, 702]
  std::vector<uint32_t> stream = {
    10, 20,
    100, 200, 300, 501, 502, 503,
    30, 40,
    100, 200, 300, 601, 602, 603,
    50, 60,
    100, 200, 300, 701, 702,
    80
  };

  pld.indexPrompt(stream);

  std::array<uint32_t, 3> query = {100, 200, 300};
  auto branches = pld.proposeBranches(query, 3, 3, 3);

  std::cout << "  Found " << branches.size() << " branches:" << std::endl;
  for (size_t b = 0; b < branches.size(); ++b) {
    std::cout << "    Branch " << b << " (matchLen=" << branches[b].matchLength << "): [";
    for (size_t i = 0; i < branches[b].tokens.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << branches[b].tokens[i];
    }
    std::cout << "]" << std::endl;
  }

  assert(branches.size() == 3 && "Must find exactly 3 distinct candidate branches");
  assert(branches[0].tokens.size() == 3 || branches[0].tokens.size() == 2);
  // All branches should start with distinct follow-ons: 701, 601, 501
  assert(branches[0].tokens[0] == 701 || branches[0].tokens[0] == 601 || branches[0].tokens[0] == 501);
  assert(branches[1].tokens[0] != branches[0].tokens[0]);
  assert(branches[2].tokens[0] != branches[0].tokens[0] && branches[2].tokens[0] != branches[1].tokens[0]);

  std::cout << "[testPromptLookupMultiBranch] PASS" << std::endl;
}

} // namespace

} // namespace splash::ops

int main() {
  std::cout << "=== Running RepoTokenIndex Tests ===" << std::endl;
  splash::ops::testRepoTokenIndexWriteAndRead();
  splash::ops::testPromptLookupMultiBranch();
  std::cout << "=== All RepoTokenIndex Tests Passed! ===" << std::endl;
  return 0;
}
