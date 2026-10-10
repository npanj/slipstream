#include "runtime/ops/PromptLookup.hpp"

#include <algorithm>

namespace splash::ops {

void PromptLookup::indexPrompt(std::span<const uint32_t> promptTokens,
                               uint32_t nGramSize) {
  clear();
  nGramSize_ = std::max(2u, nGramSize);
  if (promptTokens.empty())
    return;

  tokens_.assign(promptTokens.begin(), promptTokens.end());
  const size_t n = tokens_.size();
  next_.resize(n, UINT32_MAX);

  // Allocate contiguous power-of-two table with load factor < 0.5
  size_t targetSize = 65536;
  while (targetSize < n * 2) {
    targetSize <<= 1;
  }
  tableMask_ = static_cast<uint32_t>(targetSize - 1);
  head_.assign(targetSize, UINT32_MAX);

  if (n < 3)
    return;

  for (size_t i = 2; i < n; ++i) {
    const uint64_t h = hash3Gram(tokens_[i - 2], tokens_[i - 1], tokens_[i]);
    const uint32_t bucket = static_cast<uint32_t>(h & tableMask_);
    next_[i] = head_[bucket];
    head_[bucket] = static_cast<uint32_t>(i);
  }
}

void PromptLookup::ingestReferenceTokens(std::span<const uint32_t> referenceTokens) {
  if (referenceTokens.empty())
    return;
  if (tokens_.empty()) {
    indexPrompt(referenceTokens, nGramSize_);
    return;
  }
  const size_t prevCount = tokens_.size();
  tokens_.insert(tokens_.end(), referenceTokens.begin(), referenceTokens.end());
  const size_t newCount = tokens_.size();
  next_.resize(newCount, UINT32_MAX);

  // Resize hash table if load factor would exceed 0.5
  if (tableMask_ == 0 || (newCount * 2 > tableMask_ + 1)) {
    size_t targetSize = 65536;
    while (targetSize < newCount * 2) {
      targetSize <<= 1;
    }
    tableMask_ = static_cast<uint32_t>(targetSize - 1);
    head_.assign(targetSize, UINT32_MAX);
    // Rebuild index across all tokens
    for (size_t i = 2; i < newCount; ++i) {
      const uint64_t h = hash3Gram(tokens_[i - 2], tokens_[i - 1], tokens_[i]);
      const uint32_t bucket = static_cast<uint32_t>(h & tableMask_);
      next_[i] = head_[bucket];
      head_[bucket] = static_cast<uint32_t>(i);
    }
    return;
  }

  // Incrementally index the new range starting from the boundary
  const size_t startIndex = prevCount >= 2 ? prevCount : 2;
  for (size_t i = startIndex; i < newCount; ++i) {
    const uint64_t h = hash3Gram(tokens_[i - 2], tokens_[i - 1], tokens_[i]);
    const uint32_t bucket = static_cast<uint32_t>(h & tableMask_);
    next_[i] = head_[bucket];
    head_[bucket] = static_cast<uint32_t>(i);
  }
}

void PromptLookup::appendToken(uint32_t token) {
  const size_t i = tokens_.size();
  tokens_.push_back(token);
  next_.push_back(UINT32_MAX);

  if (i < 2 || head_.empty())
    return;

  const uint64_t h = hash3Gram(tokens_[i - 2], tokens_[i - 1], tokens_[i]);
  const uint32_t bucket = static_cast<uint32_t>(h & tableMask_);
  next_[i] = head_[bucket];
  head_[bucket] = static_cast<uint32_t>(i);
}

uint32_t PromptLookup::propose(std::span<const uint32_t> recentTokens,
                               std::span<uint32_t> outDrafts,
                               uint32_t maxDrafts,
                               uint32_t minMatchLength,
                               bool requireUnambiguous) const {
  if (tokens_.size() < 3 || head_.empty() || maxDrafts == 0 || outDrafts.empty()) {
    return 0;
  }

  // Assemble the 3 query tokens (t0, t1, t2)
  uint32_t t0 = 0, t1 = 0, t2 = 0;
  const size_t qLen = recentTokens.size();
  if (qLen >= 3) {
    t0 = recentTokens[qLen - 3];
    t1 = recentTokens[qLen - 2];
    t2 = recentTokens[qLen - 1];
  } else if (qLen == 2 && tokens_.size() >= 1) {
    t0 = tokens_.back();
    t1 = recentTokens[0];
    t2 = recentTokens[1];
  } else if (qLen == 1 && tokens_.size() >= 2) {
    t0 = tokens_[tokens_.size() - 2];
    t1 = tokens_[tokens_.size() - 1];
    t2 = recentTokens[0];
  } else if (qLen == 0 && tokens_.size() >= 3) {
    t0 = tokens_[tokens_.size() - 3];
    t1 = tokens_[tokens_.size() - 2];
    t2 = tokens_[tokens_.size() - 1];
  } else {
    return 0;
  }

  const uint64_t h = hash3Gram(t0, t1, t2);
  const uint32_t bucket = static_cast<uint32_t>(h & tableMask_);

  uint32_t curr = head_[bucket];
  uint32_t bestMatch = UINT32_MAX;
  uint32_t bestMatchLen = 0;
  uint32_t scanned = 0;
  uint32_t firstFollowOn = UINT32_MAX;
  bool isAmbiguous = false;

  // Search in reverse chronological order
  while (curr != UINT32_MAX && scanned < kMaxScanDepth) {
    ++scanned;
    // Check 3-gram match
    if (curr >= 2 && tokens_[curr] == t2 && tokens_[curr - 1] == t1 &&
        tokens_[curr - 2] == t0) {
      // Must have follow-on tokens and not be the active generation tail
      if (curr + 1 < tokens_.size()) {
        uint32_t matchLen = 3;
        while (matchLen < qLen && curr >= matchLen &&
               tokens_[curr - matchLen] == recentTokens[qLen - 1 - matchLen]) {
          ++matchLen;
        }

        const size_t tailPos = tokens_.size() - 1;
        while (curr >= matchLen && tailPos >= matchLen &&
               tokens_[curr - matchLen] == tokens_[tailPos - matchLen]) {
          ++matchLen;
        }

        if (matchLen >= minMatchLength) {
          const uint32_t followOn = tokens_[curr + 1];
          if (requireUnambiguous) {
            if (firstFollowOn == UINT32_MAX) {
              firstFollowOn = followOn;
            } else if (firstFollowOn != followOn && matchLen == bestMatchLen) {
              isAmbiguous = true;
            }
          }
          if (matchLen > bestMatchLen) {
            bestMatchLen = matchLen;
            bestMatch = curr;
            firstFollowOn = followOn;
            isAmbiguous = false;
            if (matchLen >= 8) {
              break;
            }
          }
        }
      }
    }
    curr = next_[curr];
  }

  if (bestMatch == UINT32_MAX || (requireUnambiguous && isAmbiguous))
    return 0;

  const uint32_t available = static_cast<uint32_t>(tokens_.size() - 1 - bestMatch);
  const uint32_t count = std::min({maxDrafts, available,
                                   static_cast<uint32_t>(outDrafts.size())});

  for (uint32_t j = 0; j < count; ++j) {
    outDrafts[j] = tokens_[bestMatch + 1 + j];
  }
  return count;
}

std::vector<PromptLookup::BranchProposal>
PromptLookup::proposeBranches(std::span<const uint32_t> recentTokens,
                              uint32_t maxBranches,
                              uint32_t maxTokensPerBranch,
                              uint32_t minMatchLength) const {
  if (tokens_.size() < 3 || head_.empty() || maxBranches == 0 ||
      maxTokensPerBranch == 0) {
    return {};
  }

  uint32_t t0 = 0, t1 = 0, t2 = 0;
  const size_t qLen = recentTokens.size();
  if (qLen >= 3) {
    t0 = recentTokens[qLen - 3];
    t1 = recentTokens[qLen - 2];
    t2 = recentTokens[qLen - 1];
  } else if (qLen == 2 && tokens_.size() >= 1) {
    t0 = tokens_.back();
    t1 = recentTokens[0];
    t2 = recentTokens[1];
  } else if (qLen == 1 && tokens_.size() >= 2) {
    t0 = tokens_[tokens_.size() - 2];
    t1 = tokens_[tokens_.size() - 1];
    t2 = recentTokens[0];
  } else if (qLen == 0 && tokens_.size() >= 3) {
    t0 = tokens_[tokens_.size() - 3];
    t1 = tokens_[tokens_.size() - 2];
    t2 = tokens_[tokens_.size() - 1];
  } else {
    return {};
  }

  const uint64_t h = hash3Gram(t0, t1, t2);
  const uint32_t bucket = static_cast<uint32_t>(h & tableMask_);

  uint32_t curr = head_[bucket];
  uint32_t scanned = 0;

  struct MatchCandidate {
    uint32_t pos;
    uint32_t matchLen;
    uint32_t firstFollowOn;
  };
  std::vector<MatchCandidate> candidates;

  while (curr != UINT32_MAX && scanned < kMaxScanDepth) {
    ++scanned;
    if (curr >= 2 && tokens_[curr] == t2 && tokens_[curr - 1] == t1 &&
        tokens_[curr - 2] == t0) {
      if (curr + 1 < tokens_.size()) {
        uint32_t matchLen = 3;
        while (matchLen < qLen && curr >= matchLen &&
               tokens_[curr - matchLen] == recentTokens[qLen - 1 - matchLen]) {
          ++matchLen;
        }

        const size_t tailPos = tokens_.size() - 1;
        while (curr >= matchLen && tailPos >= matchLen &&
               tokens_[curr - matchLen] == tokens_[tailPos - matchLen]) {
          ++matchLen;
        }

        if (matchLen >= minMatchLength) {
          const uint32_t followOn = tokens_[curr + 1];
          // Check if this followOn is already represented in candidates with >= matchLen
          bool duplicate = false;
          for (auto &c : candidates) {
            if (c.firstFollowOn == followOn) {
              if (matchLen > c.matchLen) {
                c.pos = curr;
                c.matchLen = matchLen;
              }
              duplicate = true;
              break;
            }
          }
          if (!duplicate) {
            candidates.push_back({curr, matchLen, followOn});
          }
        }
      }
    }
    curr = next_[curr];
  }

  if (candidates.empty())
    return {};

  // Sort descending by matchLen, then reverse chronological position
  std::sort(candidates.begin(), candidates.end(),
            [](const MatchCandidate &a, const MatchCandidate &b) {
              if (a.matchLen != b.matchLen)
                return a.matchLen > b.matchLen;
              return a.pos > b.pos;
            });

  const uint32_t branchCount =
      std::min(maxBranches, static_cast<uint32_t>(candidates.size()));
  std::vector<BranchProposal> results;
  results.reserve(branchCount);

  for (uint32_t b = 0; b < branchCount; ++b) {
    const auto &cand = candidates[b];
    const uint32_t available =
        static_cast<uint32_t>(tokens_.size() - 1 - cand.pos);
    const uint32_t tokCount = std::min(maxTokensPerBranch, available);
    BranchProposal bp;
    bp.matchLength = cand.matchLen;
    bp.tokens.reserve(tokCount);
    for (uint32_t j = 0; j < tokCount; ++j) {
      bp.tokens.push_back(tokens_[cand.pos + 1 + j]);
    }
    results.push_back(std::move(bp));
  }

  return results;
}

void PromptLookup::clear() noexcept {
  tokens_.clear();
  head_.clear();
  next_.clear();
  tableMask_ = 0;
  nGramSize_ = kDefaultNGram;
}

} // namespace splash::ops
