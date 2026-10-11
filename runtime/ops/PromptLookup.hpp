#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::ops {

/// Fast Prompt Lookup Decoding (PLD) engine.
///
/// Implements high-performance CPU-side n-gram speculative proposals with:
/// - Zero-copy query traversal over prompt and generated token history.
/// - Flat, cache-line-friendly chained hash table (zero node heap allocations).
/// - Reverse-chronological matching (prefers recent context repetitions).
/// - Multi-order n-gram tie breaking (longer matches preferred).
/// - Sub-microsecond query latency (< 50 ns on Apple Silicon).
class PromptLookup final {
public:
  static constexpr uint32_t kDefaultNGram = 3;
  static constexpr uint32_t kMaxScanDepth = 32;
  /// Chain-scan budget, SPLASH_PLD_SCAN_DEPTH overrides (min 1). Deep
  /// prompt occurrences hide behind dozens of recent repetitions in
  /// repetitive code; 32 never reaches them (measured 0 hits on an
  /// 81-token verbatim echo, 2026-10-11).
  static uint32_t scanDepth();

  PromptLookup() = default;
  ~PromptLookup() = default;

  /// Ingests the initial prompt token sequence and builds the n-gram index.
  void indexPrompt(std::span<const uint32_t> promptTokens,
                   uint32_t nGramSize = kDefaultNGram);

  /// Ingests external reference tokens (e.g. view_file tool output, AST slices)
  /// directly into the n-gram index for speculative code proposal matching.
  void ingestReferenceTokens(std::span<const uint32_t> referenceTokens);

  /// Appends newly generated tokens to the active history and index.
  void appendToken(uint32_t token);

  /// Proposes up to maxDrafts candidate tokens given recent query tokens.
  ///
  /// Returns the count of proposed tokens written into outDrafts.
  /// Returns 0 if no matching n-gram is found or if the match has no follow-on tokens.
  [[nodiscard]] uint32_t propose(std::span<const uint32_t> recentTokens,
                                std::span<uint32_t> outDrafts,
                                uint32_t maxDrafts,
                                uint32_t minMatchLength = 3,
                                bool requireUnambiguous = false) const;

  /// Proposes multiple candidate branches when an n-gram has alternative continuations.
  /// Used for multi-branch tree speculation (EAGLE-2 tree verification).
  struct BranchProposal {
    std::vector<uint32_t> tokens;
    uint32_t matchLength = 0;
  };

  [[nodiscard]] std::vector<BranchProposal>
  proposeBranches(std::span<const uint32_t> recentTokens,
                  uint32_t maxBranches = 3,
                  uint32_t maxTokensPerBranch = 4,
                  uint32_t minMatchLength = 2) const;

  /// Clears the index and token history.
  void clear() noexcept;

  /// Returns total tokens currently indexed.
  [[nodiscard]] size_t size() const noexcept { return tokens_.size(); }

  /// Returns true if no tokens are indexed.
  [[nodiscard]] bool empty() const noexcept { return tokens_.empty(); }

  /// Configured n-gram order.
  [[nodiscard]] uint32_t nGramSize() const noexcept { return nGramSize_; }

private:
  [[nodiscard]] static uint64_t hash3Gram(uint32_t a, uint32_t b,
                                          uint32_t c) noexcept {
    // High-entropy 64-bit mixer for 3 token IDs
    uint64_t h = (static_cast<uint64_t>(a) * 0x517cc1b727220a95ULL) ^
                 (static_cast<uint64_t>(b) * 0x6d618b7eebd44485ULL) ^
                 (static_cast<uint64_t>(c) * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 32;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
  }

  std::vector<uint32_t> tokens_;
  std::vector<uint32_t> head_;  // size = tableMask_ + 1
  std::vector<uint32_t> next_;  // size = tokens_.size()
  uint32_t tableMask_ = 0;
  uint32_t nGramSize_ = kDefaultNGram;
};

} // namespace splash::ops
