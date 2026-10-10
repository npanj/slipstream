#include "model/QwenState.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

using metal::MetalBuffer;

void *writableContents(const MetalBuffer &buffer, const char *name) {
  void *contents = buffer.contents();
  if (!contents) {
    throw std::logic_error(std::string(name) + " is not CPU-visible");
  }
  return contents;
}

void copyExact(const MetalBuffer &destination, const MetalBuffer &source,
               const char *name) {
  if (destination.sizeBytes() != source.sizeBytes()) {
    throw std::logic_error(std::string(name) + " shape mismatch");
  }
  std::memcpy(writableContents(destination, name),
              writableContents(source, name), destination.sizeBytes());
}

void clear(const MetalBuffer &buffer, const char *name) {
  std::memset(writableContents(buffer, name), 0, buffer.sizeBytes());
}

} // namespace

QwenGdnCell::QwenGdnCell(metal::MetalBackend &backend,
                         std::shared_ptr<StateAllocationTracker> tracker,
                         GdnStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)) {
  if (!tracker_)
    throw std::invalid_argument("Qwen state allocation tracker is empty");
  if (!layout.valid())
    throw std::invalid_argument("Qwen GDN state layout is invalid");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  buffers_.stateBase = backend.allocateBuffer(
      layout.cellBytes(), metal::BufferStorage::Shared, label);
  buffers_.convolutionBase = backend.view(
      buffers_.stateBase, 0, layout.convolutionBytes());
  buffers_.convolutionLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.convolutionLayers[layer] =
        backend.view(buffers_.convolutionBase,
                     uint64_t{layer} * layout.convolutionLayerBytes(),
                     layout.convolutionLayerBytes());
  }
  buffers_.recurrentBase =
      backend.view(buffers_.stateBase, layout.convolutionBytes(),
                   layout.recurrentBytes());
  buffers_.recurrentLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.recurrentLayers[layer] = backend.view(
        buffers_.recurrentBase,
        uint64_t{layer} * layout.recurrentLayerBytes(),
        layout.recurrentLayerBytes());
  }
  if (layout.auxiliaryBytes) {
    buffers_.auxiliary = backend.view(buffers_.stateBase,
                                      layout.auxiliaryOffset(),
                                      layout.auxiliaryBytes);
  }
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.cellBytes()) {
    throw std::logic_error("Qwen GDN allocation is below declared bytes");
  }
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

QwenGdnCell::~QwenGdnCell() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

QwenCompositeState::QwenCompositeState(std::shared_ptr<QwenBufferPool> pool,
                                       QwenCacheSlot slot,
                                       CompositeStateLayout layout,
                                       QwenLogicalLengths lengths)
    : pool_(std::move(pool)), slot_(std::move(slot)), layout_(layout),
      lengths_(lengths) {
  if (!pool_ || !slot_.gdn ||
      (layout_.hasDraft() && !slot_.draft)) {
    throw std::invalid_argument("composite state buffers are empty");
  }
}

QwenCompositeState::~QwenCompositeState() {
  if (!pool_->open)
    return;
  pool_->cells.push_back(std::move(slot_.gdn));
  if (slot_.draft)
    pool_->rings.push_back(std::move(slot_.draft));
}

QwenStateStorage::QwenStateStorage(metal::MetalBackend &backend,
                                   metal::AllocationAdmission admitAllocation,
                                   CompositeStateLayout layout)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout),
      allocations_(std::make_shared<StateAllocationTracker>()),
      pool_(std::make_shared<QwenBufferPool>()) {
  if (!admitAllocation_)
    throw std::invalid_argument("Qwen state allocation admission is required");
  if (!layout_.valid() ||
      (layout_.hasDraft() &&
       layout_.draft.tokens != ExecutionLimits::draftContextTokens)) {
    throw std::invalid_argument("Qwen composite state layout is invalid");
  }
}

QwenStateStorage::~QwenStateStorage() {
  pool_->open = false;
  pool_->cells.clear();
  pool_->rings.clear();
}

const QwenSlotBuffers &QwenStateStorage::buffers(uint32_t index) const {
  return slot(index).buffers;
}

const QwenSlotMetadata &QwenStateStorage::metadata(uint32_t index) const {
  return slot(index).metadata;
}

metal::AllocationResult QwenStateStorage::tryActivateSlot(uint32_t index, uint64_t requestId) {
  if (!requestId)
    throw std::invalid_argument("request id must be non-zero");
  Slot &current = slot(index);
  if (current.metadata.assigned) {
    throw std::logic_error("Qwen state slot is already assigned");
  }
  if (auto admission = allocateSlot(index); !admission)
    return admission;

  // A fresh recurrent sequence reads parity zero immediately. Parity one is
  // fully overwritten by the first transition.
  clear(current.buffers.gdn[0].convolutionBase, "slot convolution state");
  clear(current.buffers.gdn[0].recurrentBase, "slot recurrent state");
  // Zeroed auxiliary state reads as "no history yet"; see the model's layout.
  if (current.buffers.gdn[0].auxiliary)
    clear(current.buffers.gdn[0].auxiliary, "slot auxiliary state");
  current.metadata = {true, requestId, 0, {}};
  return true;
}

void QwenStateStorage::releaseSlot(uint32_t index, uint64_t requestId) {
  Slot &current = slot(index);
  requireAssigned(current);
  if (!requestId || current.metadata.requestId != requestId) {
    throw std::logic_error("Qwen state slot owner mismatch");
  }
  // Parity one first, so the next activation pops parity zero first and a
  // reactivated lane gets its previous buffers back in the same order.
  for (uint32_t parity = current.gdn.size(); parity > 0;) {
    --parity;
    if (current.gdn[parity])
      pool_->cells.push_back(std::move(current.gdn[parity]));
  }
  if (current.draft)
    pool_->rings.push_back(std::move(current.draft));
  refreshViews(current);
  current.metadata = {};
}

uint64_t QwenStateStorage::releaseIdle(uint32_t keepCells) noexcept {
  const uint64_t before = backend_.memoryStats().allocatedBytes;
  while (pool_->cells.size() > keepCells)
    pool_->cells.pop_back();
  while (pool_->rings.size() > keepCells)
    pool_->rings.pop_back();
  const uint64_t after = backend_.memoryStats().allocatedBytes;
  return before >= after ? before - after : 0;
}

uint32_t QwenStateStorage::idleCells() const noexcept {
  return static_cast<uint32_t>(pool_->cells.size());
}

uint32_t QwenStateStorage::idleRings() const noexcept {
  return static_cast<uint32_t>(pool_->rings.size());
}

void QwenStateStorage::updateLengths(uint32_t index,
                                     QwenLogicalLengths lengths) {
  validateLengths(lengths, false);
  Slot &current = slot(index);
  requireAssigned(current);
  current.metadata.lengths = lengths;
}

void QwenStateStorage::swapParity(uint32_t index) {
  Slot &current = slot(index);
  requireAssigned(current);
  current.metadata.activeParity ^= 1;
}

std::shared_ptr<const QwenCompositeState>
QwenStateStorage::snapshot(uint32_t index) {
  Slot &source = slot(index);
  requireAssigned(source);
  validateLengths(source.metadata.lengths, true);
  // Pooled buffers first; a denied admission puts a pooled cell back and
  // drops a fresh one, leaving no trace.
  QwenCacheSlot cacheSlot;
  cacheSlot.gdn = acquireCell("qwen-state-cache-gdn");
  if (!cacheSlot.gdn)
    return nullptr;
  if (layout_.hasDraft()) {
    cacheSlot.draft = acquireRing("qwen-state-cache-draft");
    if (!cacheSlot.draft)
      return nullptr;
  }
  const uint32_t active = source.metadata.activeParity;
  copyExact(cacheSlot.gdn->buffers().stateBase,
            source.gdn[active]->buffers().stateBase, "cached GDN state");
  if (layout_.hasDraft()) {
    for (uint32_t layer = 0; layer < source.buffers.draft.size(); ++layer) {
      copyExact(cacheSlot.draft->layers()[layer].keys,
                source.buffers.draft[layer].keys, "cached draft keys");
      copyExact(cacheSlot.draft->layers()[layer].values,
                source.buffers.draft[layer].values, "cached draft values");
    }
  }
  return std::shared_ptr<const QwenCompositeState>(new QwenCompositeState(
      pool_, std::move(cacheSlot), layout_, source.metadata.lengths));
}

std::shared_ptr<QwenGdnCell>
QwenStateStorage::acquireCell(std::string_view label) {
  if (pool_->cells.empty())
    return allocateGdnCell(label);
  std::shared_ptr<QwenGdnCell> cell = std::move(pool_->cells.back());
  pool_->cells.pop_back();
  return cell;
}

std::shared_ptr<DFlashDraftRing>
QwenStateStorage::acquireRing(std::string_view label) {
  if (pool_->rings.empty())
    return allocateDraftRing(label);
  std::shared_ptr<DFlashDraftRing> ring = std::move(pool_->rings.back());
  pool_->rings.pop_back();
  return ring;
}

void QwenStateStorage::restore(uint32_t index, const CompositeState &state) {
  const auto *typed = dynamic_cast<const QwenCompositeState *>(&state);
  if (!typed) {
    throw std::invalid_argument("composite state is not Qwen state");
  }
  if (typed->layout_ != layout_) {
    throw std::invalid_argument("composite state layout does not match model");
  }
  validateLengths(typed->lengths_, true);
  Slot &destination = slot(index);
  requireAssigned(destination);
  if (!typed->slot_.gdn) {
    throw std::invalid_argument("incompatible Qwen composite state");
  }

  const uint32_t active = destination.metadata.activeParity;
  copyExact(destination.gdn[active]->buffers().stateBase,
            typed->slot_.gdn->buffers().stateBase, "restored GDN state");
  if (layout_.hasDraft()) {
    if (!typed->slot_.draft || !destination.draft) {
      throw std::invalid_argument("incompatible Qwen composite draft state");
    }
    for (uint32_t layer = 0; layer < destination.buffers.draft.size();
         ++layer) {
      copyExact(destination.buffers.draft[layer].keys,
                typed->slot_.draft->layers()[layer].keys,
                "restored draft keys");
      copyExact(destination.buffers.draft[layer].values,
                typed->slot_.draft->layers()[layer].values,
                "restored draft values");
    }
  }
  const uint64_t requestId = destination.metadata.requestId;
  destination.metadata = {true, requestId, active, typed->lengths_};
}

uint64_t QwenStateStorage::actualSlotBytes(uint32_t index) const {
  const Slot &current = slot(index);
  uint64_t result = 0;
  if (current.gdn[0])
    result += current.gdn[0]->actualAllocatedBytes();
  if (current.gdn[1])
    result += current.gdn[1]->actualAllocatedBytes();
  if (current.draft)
    result += current.draft->actualAllocatedBytes();
  return result;
}

metal::AllocationResult QwenStateStorage::allocateSlot(uint32_t index) {
  Slot &destination = slot(index);
  if (destination.gdn[0] || destination.gdn[1] || destination.draft) {
    throw std::logic_error("idle Qwen state slot still owns buffers");
  }
  // Pooled buffers first, then the governor for what the pool lacks. A denied
  // admission leaves no trace: pooled buffers go back, fresh ones are dropped.
  std::array<std::shared_ptr<QwenGdnCell>, 2> gdn;
  std::shared_ptr<DFlashDraftRing> draft;
  metal::AllocationFailure failure = metal::AllocationFailure::None;
  uint32_t pooledCells = 0;
  while (pooledCells < gdn.size() && !pool_->cells.empty()) {
    gdn[pooledCells++] = std::move(pool_->cells.back());
    pool_->cells.pop_back();
  }
  const bool pooledRing = layout_.hasDraft() && !pool_->rings.empty();
  if (pooledRing) {
    draft = std::move(pool_->rings.back());
    pool_->rings.pop_back();
  }
  const auto giveBack = [&] {
    for (uint32_t parity = pooledCells; parity > 0;)
      pool_->cells.push_back(std::move(gdn[--parity]));
    if (pooledRing)
      pool_->rings.push_back(std::move(draft));
  };
  for (uint32_t parity = pooledCells; parity < gdn.size(); ++parity) {
    gdn[parity] = allocateGdnCell("qwen-state-cell-" + std::to_string(index) +
                                  "-gdn-" + std::to_string(parity), &failure);
    if (!gdn[parity]) {
      giveBack();
      return failure;
    }
  }
  if (layout_.hasDraft() && !pooledRing) {
    draft = allocateDraftRing("qwen-state-cell-" + std::to_string(index) +
                              "-draft", &failure);
    if (!draft) {
      giveBack();
      return failure;
    }
  }
  destination.gdn = std::move(gdn);
  destination.draft = std::move(draft);
  refreshViews(destination);
  return true;
}

std::shared_ptr<QwenGdnCell>
QwenStateStorage::allocateGdnCell(std::string_view label,
                                     metal::AllocationFailure *failure) {
  std::shared_ptr<QwenGdnCell> result;
  const auto admission = admitAllocation_(layout_.target.cellBytes(), [&] {
        result = std::shared_ptr<QwenGdnCell>(
            new QwenGdnCell(backend_, allocations_, layout_.target, label));
      });
  if (!admission) {
    if (failure)
      *failure = admission.failure;
    return {};
  }
  if (!result)
    throw std::logic_error("state admission skipped GDN allocation");
  return result;
}

std::shared_ptr<DFlashDraftRing>
QwenStateStorage::allocateDraftRing(std::string_view label,
                                     metal::AllocationFailure *failure) {
  std::shared_ptr<DFlashDraftRing> result;
  const auto admission = admitAllocation_(layout_.draft.ringBytes(), [&] {
        result = std::shared_ptr<DFlashDraftRing>(
            new DFlashDraftRing(backend_, allocations_, layout_.draft,
                                label));
      });
  if (!admission) {
    if (failure)
      *failure = admission.failure;
    return {};
  }
  if (!result)
    throw std::logic_error("state admission skipped draft allocation");
  return result;
}

void QwenStateStorage::refreshViews(Slot &current) {
  for (uint32_t parity = 0; parity < current.gdn.size(); ++parity) {
    current.buffers.gdn[parity] = current.gdn[parity]
                                      ? current.gdn[parity]->buffers()
                                      : GdnParityBuffers{};
  }
  current.buffers.draft =
      current.draft ? current.draft->layers()
                    : std::vector<DFlashDraftRingLayer>{};
}

QwenStateStorage::Slot &QwenStateStorage::slot(uint32_t index) {
  if (index >= slots_.size()) {
    throw std::out_of_range("invalid Qwen state slot");
  }
  return slots_[index];
}

const QwenStateStorage::Slot &QwenStateStorage::slot(uint32_t index) const {
  if (index >= slots_.size()) {
    throw std::out_of_range("invalid Qwen state slot");
  }
  return slots_[index];
}

void QwenStateStorage::validateLengths(const QwenLogicalLengths &lengths,
                                       bool cacheSnapshot) const {
  if (layout_.hasDraft()) {
    if (lengths.draftLength > layout_.draft.tokens ||
        lengths.draftCommitCursor >= layout_.draft.tokens ||
        lengths.draftEnd() > lengths.targetTokens) {
      throw std::invalid_argument("invalid draft ring metadata");
    }
    if (cacheSnapshot &&
        !lengths.hasCompleteDraftWindow(layout_.draft.tokens)) {
      throw std::invalid_argument(
          "composite snapshot requires a complete draft window");
    }
  }
  if (cacheSnapshot &&
      (!lengths.targetTokens || lengths.targetTokens % kv::kPageTokens)) {
    throw std::invalid_argument(
        "composite snapshot requires equal page-aligned committed lengths");
  }
}

void QwenStateStorage::requireAssigned(const Slot &current) {
  if (!current.metadata.assigned || !current.metadata.requestId) {
    throw std::logic_error("Qwen state slot is not assigned");
  }
}

} // namespace splash::model
