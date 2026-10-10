#include "model/Model.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace splash;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void clearBit(std::vector<uint32_t> &mask, uint32_t wordsPerRow,
              uint32_t row, uint32_t token) {
  mask[row * wordsPerRow + token / 32] &= ~(1U << (token % 32));
}

} // namespace

int main() {
  constexpr uint32_t wordsPerRow = 4; // 128-token vocabulary coverage
  // Anchor row plus three draft rows, every token allowed.
  const std::vector<uint32_t> full(4 * wordsPerRow, 0xFFFFFFFFu);

  // All drafts accepted.
  {
    const uint32_t proposed[] = {5, 40, 71};
    require(countValidDrafts(proposed, 3, full.data(), 16, wordsPerRow) == 3,
            "all-accepted drafts should count 3");
  }

  // Grammar rejects the second draft; the third row is never examined.
  {
    const uint32_t proposed[] = {5, 40, 71};
    std::vector<uint32_t> mask = full;
    clearBit(mask, wordsPerRow, /*row=*/2, /*token=*/40);
    require(countValidDrafts(proposed, 3, mask.data(), 16, wordsPerRow) == 1,
            "grammar rejection should stop counting at 1");
  }

  // Prompt lookup pads unfilled rows with kEmptyDraftToken. Its id falls
  // outside every mask row, so counting must stop instead of reading far
  // out of bounds (the 27b tool-request crash).
  {
    const uint32_t proposed[] = {kEmptyDraftToken, kEmptyDraftToken,
                                 kEmptyDraftToken};
    require(countValidDrafts(proposed, 3, full.data(), 16, wordsPerRow) == 0,
            "leading padding must count 0 without touching the mask");
  }
  {
    const uint32_t proposed[] = {5, 40, kEmptyDraftToken};
    require(countValidDrafts(proposed, 3, full.data(), 16, wordsPerRow) == 2,
            "trailing padding must stop counting at 2");
  }

  // A mask shorter than the draft list stops at the rows it contains.
  {
    const uint32_t proposed[] = {5, 40, 71};
    const std::vector<uint32_t> shortMask(2 * wordsPerRow, 0xFFFFFFFFu);
    require(countValidDrafts(proposed, 3, shortMask.data(), 8, wordsPerRow) ==
                1,
            "short mask should stop counting at 1");
  }

  // The draft budget bounds the scan even when more rows are proposed.
  {
    const uint32_t proposed[] = {5, 40, 71};
    require(countValidDrafts(proposed, 2, full.data(), 16, wordsPerRow) == 2,
            "draft budget should bound counting at 2");
  }

  // An empty mask counts nothing and touches no memory.
  {
    const uint32_t proposed[] = {5};
    require(countValidDrafts(proposed, 1, nullptr, 0, wordsPerRow) == 0,
            "empty mask should count 0");
  }

  std::cout << "draft-mask-test: ok\n";
  return 0;
}
