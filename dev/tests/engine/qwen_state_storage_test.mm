#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"
#include "ops/Q8PageStorage.hpp"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr model::GdnStateLayout kTargetState{48, 3, 10'240, 48, 128, 128};
constexpr model::CompositeStateLayout kStateLayout{kTargetState, {}};
constexpr model::DraftStateLayout kDraftState{2, 2, 2048, 8};
constexpr model::CompositeStateLayout kDraftLayout{kTargetState, kDraftState};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Exception = std::exception>
void requireThrows(const std::function<void()> &operation,
                   const char *message) {
  try {
    operation();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

uint32_t &word(const metal::MetalBuffer &buffer, uint64_t byteOffset = 0) {
  require(byteOffset + sizeof(uint32_t) <= buffer.sizeBytes(),
          "test marker is outside buffer");
  auto *bytes = static_cast<uint8_t *>(buffer.contents());
  require(bytes != nullptr, "test buffer is not CPU-visible");
  return *reinterpret_cast<uint32_t *>(bytes + byteOffset);
}

void testLayoutFormulas() {
  require(kTargetState.convolutionLayerBytes() == 65'536,
          "GDN convolution layer formula is wrong");
  require(kTargetState.convolutionBytes() == 3'145'728,
          "GDN convolution parity formula is wrong");
  require(kTargetState.recurrentLayerBytes() == 3'145'728,
          "GDN recurrent layer formula is wrong");
  require(kTargetState.recurrentBytes() == 150'994'944,
          "GDN recurrent parity formula is wrong");
  require(kTargetState.cellBytes() == 154'140'672,
          "GDN cell formula is wrong");
  require(kStateLayout.activeCellBytes() == 308'281'344,
          "per-slot byte formula is wrong");
  require(uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                  kStateLayout.activeCellBytes() ==
              1'233'125'376,
          "four-slot byte formula is wrong");
  require(kStateLayout.cachedBytes() == 154'140'672,
          "prefix byte formula is wrong");
}

void run(const std::string &metallib) {
  using model::QwenCompositeState;
  using model::QwenLogicalLengths;
  using model::QwenStateStorage;

  testLayoutFormulas();
  metal::MetalBackend backend(metallib);
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  // Switched off to prove that a pooled cache slot needs no new admission.
  bool admitNewAllocations = true;
  auto admitState = [&governor, &admitNewAllocations](
                        uint64_t bytes,
                        const std::function<void()> &allocate) {
    if (!admitNewAllocations)
      return false;
    auto reservation = governor.tryReserve(bytes);
    if (!reservation)
      return false;
    allocate();
    reservation->commit();
    return true;
  };
  constexpr kv::Q8Layout q8Layout{16, 4, 256};
  kv::Q8PageStorage pageStorage(backend, governor.allocationAdmission(),
                                q8Layout,
                                q8Layout.sparseMappingBatchPages());
  uint64_t beforeStorage = backend.memoryStats().allocatedBytes;
  uint64_t observedStorageActual = 0;
  uint64_t observedSlotActual = 0;
  uint64_t observedPrefixActual = 0;

  {
    QwenStateStorage storage(backend, admitState, kStateLayout);
    observedStorageActual = storage.actualAllocatedBytes();
    observedSlotActual = storage.actualSlotBytes(0);
    require(storage.actualAllocatedBytes() == 0 &&
                backend.memoryStats().allocatedBytes == beforeStorage,
            "state cells were allocated eagerly");
    for (uint32_t slot = 0;
         slot < model::ExecutionLimits::maximumBatchWidth;
         ++slot) {
      require(!storage.metadata(slot).assigned, "slot was assigned eagerly");
      require(storage.actualSlotBytes(slot) == 0,
              "idle slot has physical backing before activation");
    }
    requireThrows<std::out_of_range>(
        [&] { static_cast<void>(storage.buffers(4)); },
        "storage exposed more than four slots");

    require(storage.tryActivateSlot(0, 101) && storage.tryActivateSlot(1, 202),
            "state cell activation failed");
    observedStorageActual = storage.actualAllocatedBytes();
    observedSlotActual = storage.actualSlotBytes(0);
    require(observedSlotActual >= kStateLayout.activeCellBytes(),
            "activated slot allocation is below declared bytes");
    require(storage.actualSlotBytes(1) == observedSlotActual &&
                storage.actualAllocatedBytes() == 2 * observedSlotActual,
            "activated slot accounting is not incremental");

    const auto &slot0 = storage.buffers(0);
    const auto &slot1 = storage.buffers(1);
    void *stableGdnBase = slot0.gdn[0].stateBase.contents();
    void *stableConvBase = slot0.gdn[0].convolutionBase.contents();
    require(stableGdnBase && stableConvBase,
            "stable slot buffers are not CPU-visible");
    require(stableGdnBase != slot1.gdn[0].stateBase.contents(),
            "two slots alias one GDN allocation");
    require(slot0.gdn[0].convolutionLayers[1].contents() ==
                static_cast<uint8_t *>(stableConvBase) +
                    kTargetState.convolutionLayerBytes(),
            "GDN layer view offset is wrong");
    require(slot0.gdn[0].stateBase.sizeBytes() ==
                    kTargetState.cellBytes(),
            "split state-buffer sizes are wrong");

    require(storage.metadata(0).assigned &&
                storage.metadata(0).requestId == 101 &&
                storage.metadata(0).activeParity == 0,
            "slot activation metadata is wrong");
    requireThrows<std::logic_error>(
        [&] { static_cast<void>(storage.tryActivateSlot(0, 303)); },
        "double slot activation was accepted");

    // Hot metadata updates must leave every buffer and marker untouched.
    word(slot0.gdn[0].convolutionBase) = 0x10101010;
    word(slot0.gdn[1].convolutionBase) = 0x21212121;
    word(slot0.gdn[1].recurrentBase) = 0x31313131;
    QwenLogicalLengths lengths{2'048};
    storage.updateLengths(0, lengths);
    require(storage.metadata(0).lengths.targetTokens == 2'048,
            "logical length is wrong");
    require(word(slot0.gdn[1].convolutionBase) == 0x21212121,
            "length update copied or cleared hot state");
    storage.swapParity(0);
    require(storage.metadata(0).activeParity == 1,
            "parity swap did not select parity one");
    require(word(slot0.gdn[0].convolutionBase) == 0x10101010 &&
                word(slot0.gdn[1].convolutionBase) == 0x21212121,
            "parity swap copied hot state");

    // Publication copies the lane's active-parity GDN cell into a cache slot
    // the governor admits. The lane keeps its own
    // cells: no address changes, no aliasing, no parity handoff.
    const uint64_t beforePrefix = backend.memoryStats().allocatedBytes;
    const uint64_t storageBeforePrefix = storage.actualAllocatedBytes();
    void *laneActiveGdnBase = slot0.gdn[1].stateBase.contents();
    std::shared_ptr<const QwenCompositeState> prefix = storage.snapshot(0);
    require(prefix != nullptr, "snapshot could not obtain a cache slot");
    observedPrefixActual = backend.memoryStats().allocatedBytes - beforePrefix;
    require(prefix->bytes() == kStateLayout.cachedBytes(),
            "cached state footprint is not the declared cached bytes");
    require(observedPrefixActual >= kStateLayout.cachedBytes() &&
                storage.actualAllocatedBytes() ==
                    storageBeforePrefix + observedPrefixActual,
            "cache slot allocation is below declared bytes or unaccounted");
    require(slot0.gdn[1].stateBase.contents() == laneActiveGdnBase &&
                slot0.gdn[0].stateBase.contents() == stableGdnBase,
            "snapshot moved or aliased the lane's own cells");
    require(storage.metadata(0).activeParity == 1 &&
                storage.metadata(0).lengths == lengths,
            "snapshot changed the lane's metadata");

    // The cached copy is independent of the lane: writes to the lane's
    // active cell after publication never reach a restore.
    word(slot0.gdn[1].convolutionBase) = 0xa1a1a1a1;
    word(slot0.gdn[1].recurrentBase) = 0xa2a2a2a2;
    word(slot1.gdn[0].convolutionBase) = 0xb0b0b0b0;
    word(slot1.gdn[1].convolutionBase) = 0xb1b1b1b1;
    void *destinationGdnBase = slot1.gdn[0].stateBase.contents();
    storage.restore(1, *prefix);

    require(storage.metadata(1).requestId == 202 &&
                storage.metadata(1).activeParity == 0 &&
                storage.metadata(1).lengths == lengths,
            "restore lost owner, changed parity, or lengths");
    require(slot1.gdn[0].stateBase.contents() == destinationGdnBase,
            "restore replaced the destination's buffers");
    require(word(slot1.gdn[0].convolutionBase) == 0x21212121 &&
                word(slot1.gdn[0].recurrentBase) == 0x31313131,
            "restore did not deliver the pre-mutation GDN snapshot");
    require(word(slot1.gdn[1].convolutionBase) == 0xb1b1b1b1,
            "restore overwrote inactive parity");
    require(word(slot0.gdn[1].convolutionBase) == 0xa1a1a1a1 &&
                word(slot0.gdn[1].recurrentBase) == 0xa2a2a2a2,
            "restore wrote back into the source lane");

    // Later restores into lane 1 land in its other parity.
    storage.swapParity(1);
    // Cancellation releases ownership and returns the lane's buffers to the
    // pool. Reactivation takes them back in the same order and initializes
    // every state that can be read at logical length zero.
    void *reusableGdnBase = slot0.gdn[0].stateBase.contents();
    word(slot0.gdn[0].convolutionBase) = 0xc1c1c1c1;
    word(slot0.gdn[0].recurrentBase) = 0xc2c2c2c2;
    storage.releaseSlot(0, 101);
    require(!storage.metadata(0).assigned && storage.metadata(0).requestId == 0,
            "cancellation did not release metadata");
    require(storage.idleCells() == 2 && storage.actualSlotBytes(0) == 0,
            "released lane buffers did not return to the pool");
    requireThrows<std::logic_error>([&] { storage.swapParity(0); },
                                    "unassigned slot accepted a parity update");
    require(static_cast<bool>(storage.tryActivateSlot(0, 303)), "state cell reuse failed");
    require(storage.idleCells() == 0,
            "reactivation left pooled buffers behind");
    require(slot0.gdn[0].stateBase.contents() == reusableGdnBase,
            "slot reuse changed stable buffer addresses");
    require(storage.metadata(0).requestId == 303 &&
                storage.metadata(0).activeParity == 0 &&
                storage.metadata(0).lengths == QwenLogicalLengths{},
            "slot reuse did not reset logical state");
    require(word(slot0.gdn[0].convolutionBase) == 0 &&
                word(slot0.gdn[0].recurrentBase) == 0,
            "slot reuse did not initialize readable state");

    // Rejected publications fail before any cache slot is taken or admitted.
    const uint64_t beforeRejected = storage.actualAllocatedBytes();
    requireThrows<std::invalid_argument>(
        [&] {
          storage.updateLengths(0, {129});
          static_cast<void>(storage.snapshot(0));
        },
        "unaligned prefix snapshot was accepted");
    require(storage.actualAllocatedBytes() == beforeRejected,
            "rejected snapshot allocated or consumed a cache slot");
    requireThrows<std::logic_error>([&] { storage.releaseSlot(0, 404); },
                                    "slot release accepted the wrong owner");

    const uint64_t beforeSuspend = storage.actualAllocatedBytes();
    const uint64_t releasedSlotBytes = storage.actualSlotBytes(0);
    storage.releaseSlot(0, 303);
    require(storage.releaseIdle(0) == releasedSlotBytes,
            "recomputation preemption retained active backing");
    require(!storage.metadata(0).assigned && storage.actualSlotBytes(0) == 0 &&
                storage.actualAllocatedBytes() == beforeSuspend - releasedSlotBytes,
            "preempted GDN bytes remain outside the cache");
    require(storage.tryActivateSlot(0, 303) &&
                storage.metadata(0).assigned &&
                storage.metadata(0).lengths == QwenLogicalLengths{} &&
                word(slot0.gdn[0].convolutionBase) == 0 &&
                word(slot0.gdn[0].recurrentBase) == 0,
            "recomputation did not start from a fresh empty state");

    // Dropping a cached state returns its buffers to the storage's pool rather
    // than freeing them: accounting stays flat, and the next publication takes
    // the pooled buffers without a governor admission. Only releaseIdle
    // returns pooled bytes to macOS.
    const uint64_t beforeDrop = storage.actualAllocatedBytes();
    const uint64_t backendBeforeDrop = backend.memoryStats().allocatedBytes;
    prefix.reset();
    require(storage.actualAllocatedBytes() == beforeDrop &&
                backend.memoryStats().allocatedBytes == backendBeforeDrop,
            "dropped cached state freed its slot instead of pooling it");
    storage.updateLengths(0, lengths);
    word(slot0.gdn[0].convolutionBase) = 0xe1e1e1e1;
    word(slot0.gdn[0].recurrentBase) = 0xe2e2e2e2;
    admitNewAllocations = false;
    std::shared_ptr<const QwenCompositeState> pooled = storage.snapshot(0);
    require(pooled != nullptr, "snapshot did not reuse the pooled cache slot");
    require(pooled->bytes() == kStateLayout.cachedBytes() &&
                storage.actualAllocatedBytes() == beforeDrop &&
                backend.memoryStats().allocatedBytes == backendBeforeDrop,
            "pooled cache slot reuse allocated new buffers");
    require(storage.snapshot(0) == nullptr,
            "snapshot with an empty pool bypassed the governor");
    require(storage.actualAllocatedBytes() == beforeDrop,
            "denied snapshot leaked cache slot bytes");
    admitNewAllocations = true;
    storage.restore(1, *pooled);
    require(storage.metadata(1).activeParity == 1 &&
                storage.metadata(1).lengths == lengths &&
                word(slot1.gdn[1].convolutionBase) == 0xe1e1e1e1 &&
                word(slot1.gdn[1].recurrentBase) == 0xe2e2e2e2,
            "reused cache slot served stale contents");
    require(word(slot1.gdn[0].convolutionBase) == 0x21212121,
            "restore from the reused slot overwrote inactive parity");
    pooled.reset();
    require(storage.actualAllocatedBytes() == beforeDrop,
            "second dropped cached state was freed instead of pooled");
    require(storage.releaseIdle(0) == observedPrefixActual &&
                storage.actualAllocatedBytes() ==
                    beforeDrop - observedPrefixActual,
            "releaseIdle did not free the pooled cache slot");
    require(storage.metadata(0).assigned && storage.metadata(1).assigned &&
                storage.actualSlotBytes(0) == observedSlotActual &&
                storage.actualSlotBytes(1) == observedSlotActual,
            "pool reclaim touched active lane cells");

    // A live cached state survives its lane's release and reclaim; only its
    // drop plus a later reclaim frees the slot together with idle cells.
    std::shared_ptr<const QwenCompositeState> retained = storage.snapshot(1);
    require(retained != nullptr, "retained snapshot could not admit a slot");
    require(storage.actualAllocatedBytes() ==
                2 * observedSlotActual + observedPrefixActual,
            "retained snapshot accounting is wrong");
    storage.releaseSlot(0, 303);
    storage.releaseSlot(1, 202);
    require(storage.idleCells() == 4,
            "released lane buffers are missing from the pool");
    retained.reset();
    require(storage.idleCells() == 5,
            "dropped cached state did not return its buffers to the pool");
    // Releasing down to one lane's worth keeps two cells warm.
    require(storage.releaseIdle(2) ==
                observedSlotActual + observedPrefixActual &&
                storage.idleCells() == 2,
            "partial idle release did not keep the requested buffers");
    require(storage.releaseIdle(0) == observedSlotActual,
            "idle lane buffers were not reclaimed");
    require(storage.idleCells() == 0, "reclaimed buffers remain pooled");
    require(storage.actualAllocatedBytes() == 0,
            "reclaimed state cells remain accounted");
  }

  // Draft rings allocate, snapshot, and restore alongside GDN cells only
  // when the layout carries a draft model; the draft-less layout above
  // keeps byte-identical behavior.
  {
    require(kDraftLayout.hasDraft(), "draft layout should report a draft");
    require(!kStateLayout.hasDraft(), "plain layout should report no draft");
    constexpr uint64_t kRingTensorBytes = 2ULL * 2048 * 8 * 2;
    constexpr uint64_t kRingBytes = 2ULL * 2 * kRingTensorBytes;
    require(kDraftState.tensorBytes() == kRingTensorBytes,
            "draft tensor formula is wrong");
    require(kDraftState.ringBytes() == kRingBytes, "draft ring formula is wrong");
    require(kDraftLayout.activeCellBytes() ==
                kStateLayout.activeCellBytes() + kRingBytes,
            "draft slot formula is wrong");
    require(kDraftLayout.cachedBytes() == kStateLayout.cachedBytes() + kRingBytes,
            "draft cache formula is wrong");
    QwenStateStorage storage(backend, admitState, kDraftLayout);
    require(storage.tryActivateSlot(0, 501) && storage.tryActivateSlot(1, 502),
            "draft slot activation failed");
    require(storage.idleRings() == 0, "active rings leaked into the pool");
    const auto &slot0 = storage.buffers(0);
    require(slot0.draft.size() == 2, "slot has no draft ring views");
    require(slot0.draft[0].keys.sizeBytes() == kRingTensorBytes &&
                slot0.draft[0].values.sizeBytes() == kRingTensorBytes,
            "draft ring view sizes are wrong");
    require(slot0.draft[0].keys.contents() != nullptr,
            "draft ring is not CPU-visible");
    require(storage.actualSlotBytes(0) >=
                kStateLayout.activeCellBytes() + kRingBytes,
            "draft slot allocation is below declared bytes");

    // A complete 2048-token window snapshots; overfull draft metadata is
    // rejected before any copy.
    const QwenLogicalLengths window{2'048, 0, 2'048, 0};
    requireThrows<std::invalid_argument>(
        [&] { storage.updateLengths(0, {2'048, 0, 2'049, 0}); },
        "overfull draft lengths were accepted");
    storage.updateLengths(0, window);
    storage.updateLengths(1, window);
    word(slot0.draft[0].keys) = 0xd1d1d1d1;
    std::shared_ptr<const QwenCompositeState> prefix = storage.snapshot(0);
    require(prefix != nullptr, "draft snapshot could not obtain a slot");
    require(prefix->bytes() == kDraftLayout.cachedBytes(),
            "draft cache footprint is wrong");
    word(slot0.draft[0].keys) = 0xd2d2d2d2;
    storage.restore(1, *prefix);
    require(storage.metadata(1).lengths == window,
            "draft restore lost lengths");
    require(word(storage.buffers(1).draft[0].keys) == 0xd1d1d1d1,
            "draft restore did not deliver snapshot keys");
    const uint64_t activeBytes = storage.actualSlotBytes(1);
    storage.releaseSlot(0, 501);
    require(storage.idleRings() == 1, "released ring missed the pool");
    prefix.reset();
    require(storage.idleRings() == 2, "dropped draft cache missed the pool");
    static_cast<void>(storage.releaseIdle(0));
    require(storage.idleCells() == 0 && storage.idleRings() == 0,
            "reclaimed buffers remain pooled");
    require(storage.actualAllocatedBytes() == activeBytes,
            "reclaim touched the active lane");
  }
  require(backend.memoryStats().allocatedBytes == beforeStorage,
          "destroyed state slots remained in actual allocation count");

  std::cout << "qwen state storage tests passed: slot_declared="
            << kStateLayout.activeCellBytes()
            << " slot_actual=" << observedSlotActual << " four_slots_declared="
            << uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                   kStateLayout.activeCellBytes()
            << " four_slots_actual=" << observedStorageActual
            << " composite_declared=" << kStateLayout.cachedBytes()
            << " prefix_actual=" << observedPrefixActual << '\n';
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 2) {
    std::cerr << "usage: qwen_state_storage_test METALLIB\n";
    return EXIT_FAILURE;
  }
  @autoreleasepool {
    try {
      run(argv[1]);
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "qwen state storage test failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
