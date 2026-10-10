#pragma once

#include "model/ModelFactory.hpp"

#include <cstdint>
#include <memory>

namespace splash::model {

class Runtime final : public RuntimeModel {
public:
  explicit Runtime(RuntimeContext context);
  ~Runtime() override;
  void checkHealth() override;
  [[nodiscard]] bool needsHealthCheck() const noexcept override;

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  // Direct native-oracle entry point. Production admission uses
  // begin() and installs its cache-aware plan explicitly.
  void beginColdRequest(const ModelRequest &request, uint32_t stateSlot);
  [[nodiscard]] StateAdmission
  begin(const ModelRequest &request) override;
  void suspend(uint64_t requestId) override;
  [[nodiscard]] StateAdmission
  resume(const ModelRequest &request) override;
  void restore(uint64_t requestId, uint32_t restoredPrefixLength,
                     std::shared_ptr<const CompositeState> restoredState) override;
  [[nodiscard]] std::vector<ModelStepResult>
  prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion) override;
  [[nodiscard]] std::vector<ModelStepResult>
  decode(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  // Measurement only: run the output head over every row of the prefill
  // chunk that just completed on lane 0, appending bf16 logits
  // [rows x vocabulary] to `path`. It reuses decode scratch, so a request is
  // not safe to continue decoding afterwards; scoring tools end it instead.
  void dumpPrefillLogits(uint32_t rows, const char *path);
  // Measurement only, for guesser training: appends one record for the prefill
  // chunk that just completed on lane 0 - the layers SPLASH_CAPTURE_LAYERS kept,
  // the head's input, and the head's top picks per row (format in Runtime.mm).
  // Same decode-scratch caveat as dumpPrefillLogits.
  void dumpPrefillFeatures(uint32_t rows, const char *path);
  [[nodiscard]] std::shared_ptr<const CompositeState>
  snapshot(uint64_t requestId) override;
  [[nodiscard]] uint64_t reclaimIdleState() noexcept override;
  void provideMask(uint64_t requestId,
                   std::span<const uint32_t> words) override;
  void setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) override;
  [[nodiscard]] bool hasDraft() const noexcept override;
  void end(uint64_t requestId) override;

  [[nodiscard]] WarmupStepResult warmupPrefill(uint32_t rows) override;
  [[nodiscard]] WarmupStepResult
  warmupDecodeBatch(uint32_t width) override;
  [[nodiscard]] WarmupStepResult warmupDraftVerifyCommit() override;
  [[nodiscard]] WarmupStepResult
  warmupCompositeStateRestore() override;
  [[nodiscard]] ModelMemoryActual
  actualRuntimeMemory() const override;

  [[nodiscard]] ModelTelemetry
  telemetry() const noexcept override;

private:
  void prepareWarmupDecode(uint64_t requestId, uint32_t anchor);
  [[nodiscard]] metal::AllocationResult beginAt(const ModelRequest &request,
                                       uint32_t stateSlot);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  prefillAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
               std::function<void()> completion);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  decodeAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
