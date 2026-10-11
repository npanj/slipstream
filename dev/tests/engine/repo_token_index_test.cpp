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

void testPromptLookupSingleAnchorVsTail() {
  std::cout << "[testPromptLookupSingleAnchorVsTail] Starting..." << std::endl;

  PromptLookup pld;
  // Prompt holds a distinctive passage; decode then emits it verbatim and
  // commits each token (as the engine's commit path does via appendToken).
  std::vector<uint32_t> prompt = {10, 20, 30, 100, 200, 300, 400, 500, 600, 40, 50};
  pld.indexPrompt(prompt);
  for (uint32_t t : {100u, 200u, 300u, 400u, 500u}) pld.appendToken(t);

  std::array<uint32_t, 4> out{};
  // Query with an already-appended anchor: qLen==1 assembles
  // (tokens[-2], tokens[-1], anchor) = (400, 500, 500), a doubled token
  // that never occurs, so nothing is proposed. Callers must pass the
  // pending (not yet appended) anchor, or the trailing tokens explicitly.
  std::array<uint32_t, 1> single{500};
  uint32_t n1 = pld.propose(single, out, 4, 4, false);
  std::cout << "  single-appended-anchor query proposed " << n1 << std::endl;
  assert(n1 == 0 && "doubled-anchor query must find nothing");

  // The last 3 tokens. (300,400,500) occurs in the prompt with follow-on
  // 600, and the tail extends the match to length 5.
  std::array<uint32_t, 3> tail{300, 400, 500};
  uint32_t n3 = pld.propose(tail, out, 4, 4, false);
  std::cout << "  three-token query proposed " << n3 << " drafts, first=" << out[0] << std::endl;
  assert(n3 > 0 && "tail query must fire on verbatim repetition");
  assert(out[0] == 600 && "first draft must continue the passage with 600");
  std::cout << "[testPromptLookupSingleAnchorVsTail] PASS" << std::endl;
}

void testPromptLookupPendingAnchorExtends() {
  std::cout << "[testPromptLookupPendingAnchorExtends] Starting..." << std::endl;

  // Echo in progress: prompt passage [50,7,8,9,200,201] committed as
  // [50,7,8,9], anchor 200 pending (not appended). The backward extension
  // must align with the query end (pending), not the history end, or
  // minMatch=4 is unreachable with a 1-token query.
  PromptLookup pld;
  const std::vector<uint32_t> prompt = {1, 2, 3, 50, 7, 8, 9, 200, 201, 4, 5, 6};
  pld.indexPrompt(prompt);
  for (uint32_t t : {50u, 7u, 8u, 9u}) pld.appendToken(t);

  std::array<uint32_t, 1> query{200};
  std::array<uint32_t, 4> out{0, 0, 0, 0};
  uint32_t n = pld.propose(query, out, 4, 4, false);
  std::cout << "  pending-anchor query proposed " << n << std::endl;
  assert(n > 0 && "pending anchor on echo must extend to minMatch=4");
  assert(out[0] == 201 && "first draft must continue the passage");
  std::cout << "[testPromptLookupPendingAnchorExtends] PASS" << std::endl;
}

void testPromptLookupDeepChain() {
  std::cout << "[testPromptLookupDeepChain] Starting..." << std::endl;

  // Trigram (7,8,9) with 41 indexed occurrences: one buried passage
  // [50,7,8,9,200] under 40 newer repetitions [7,8,9,1000+i], as with
  // common trigrams in repetitive code. Tail [50,7,8,9] + pending anchor
  // 9 extends the buried match to 4; the distractors' prefixes differ
  // from 50, so minMatch=4 separates them. Buried sits ~42 deep in its
  // chain: reachable at depth 256, invisible at the default 32.
  PromptLookup pld;
  std::vector<uint32_t> stream = {1, 2, 50, 7, 8, 9, 200};
  for (uint32_t i = 0; i < 40; ++i) {
    stream.insert(stream.end(), {7, 8, 9, static_cast<uint32_t>(1000 + i)});
  }
  stream.insert(stream.end(), {50, 7, 8});  // tail, anchor 9 pending
  pld.indexPrompt(stream);

  std::array<uint32_t, 1> query{9};
  std::array<uint32_t, 2> out{0, 0};
  uint32_t n = pld.propose(query, out, 2, 4, false);
  const uint32_t depth = PromptLookup::scanDepth();
  std::cout << "  depth=" << depth << " proposed " << n << std::endl;
  if (depth > 42) {
    assert(n > 0 && "deep scan must reach the buried occurrence");
    assert(out[0] == 200 && "first draft must be the buried follow-on");
  } else {
    assert(n == 0 && "shallow scan cannot reach the buried occurrence");
  }
  std::cout << "[testPromptLookupDeepChain] PASS" << std::endl;
}

} // namespace

} // namespace splash::ops

int main() {
  std::cout << "=== Running RepoTokenIndex Tests ===" << std::endl;
  splash::ops::testRepoTokenIndexWriteAndRead();
  splash::ops::testPromptLookupMultiBranch();
  splash::ops::testPromptLookupSingleAnchorVsTail();
  splash::ops::testPromptLookupDeepChain();
  splash::ops::testPromptLookupPendingAnchorExtends();
  splash::ops::testPromptLookupSingleAnchorVsTail();
  std::cout << "=== All RepoTokenIndex Tests Passed! ===" << std::endl;
  return 0;
}
