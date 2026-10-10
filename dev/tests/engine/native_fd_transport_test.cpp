#include "TestImmediateTicket.hpp"
#include "engine/Cache.hpp"
#include "engine/FdTransport.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace splash;
using namespace splash::engine;

namespace {

class Backing final : public KvBacking {
public:
  Backing() : resident_(8, true) {}
  uint32_t pageCount() const noexcept override { return resident_.size(); }
  uint64_t bytesPerPage() const noexcept override { return 4096; }
  bool isResident(uint32_t page) const override { return resident_.at(page); }
  splash::metal::AllocationResult ensureResident(uint32_t page) override {
    resident_.at(page) = true;
    return true;
  }
  bool releaseBackingForPage(uint32_t page) override {
    resident_.at(page) = false;
    return true;
  }
  uint32_t extentFirstPage(uint32_t page) const override { return page; }
  uint32_t extentPageCount(uint32_t) const override { return 1; }
private:
  std::vector<bool> resident_;
};

class Executor final : public model::Model {
public:
  StateAdmission begin(const ModelRequest &) override {
    return {0, StateFailure::None};
  }
  void suspend(uint64_t) override {}
  StateAdmission resume(const ModelRequest &) override {
    return {0, StateFailure::None};
  }
  void restore(uint64_t, uint32_t, std::shared_ptr<const CompositeState>) override {}
  void setDraftContextPlan(uint64_t, DraftContextPlan) override {}
  std::vector<ModelStepResult> prefill(const BatchPlan &,
                                          std::span<const ModelBatchItem>) {
    return {};
  }
  std::vector<ModelStepResult> decode(const BatchPlan &,
                                         std::span<const ModelBatchItem>) {
    return {};
  }
  std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion) override {
    return test::immediateTicket(plan.kind == WorkKind::Prefill
                                     ? prefill(plan, items)
                                     : decode(plan, items),
                                 completion);
  }
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override {
    return {};
  }
  uint64_t reclaimIdleState() noexcept override { return 0; }
  void provideMask(uint64_t, std::span<const uint32_t>) override {}
  void end(uint64_t) override {}
};

struct Pipes final {
  std::array<int, 2> input{};
  std::array<int, 2> output{};
  Pipes() {
    if (pipe(input.data()) || pipe(output.data())) {
      throw std::runtime_error("pipe creation failed");
    }
  }
  ~Pipes() {
    for (int fd : input)
      if (fd >= 0)
        close(fd);
    for (int fd : output)
      if (fd >= 0)
        close(fd);
  }
  void closeInputWriter() {
    close(input[1]);
    input[1] = -1;
  }
};

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

struct Harness final {
  Pipes pipes;
  Backing backing;
  KvPool pool{backing};
  engine::Cache resources{pool, CacheNamespace{}};
  Executor executor;
  engine::FdTransport transport{pipes.input[0], pipes.output[1]};
  engine::NativeRuntime loop{
      {}, resources, executor, transport.outputSink(),
      [] { return std::string("{\"schema_version\":5}"); }};
};

engine::NativeProcessExit run(std::span<const uint8_t> input) {
  Harness harness;
  if (!input.empty()) {
    ssize_t count = write(harness.pipes.input[1], input.data(), input.size());
    require(count == static_cast<ssize_t>(input.size()),
            "failed to seed transport input");
  }
  harness.pipes.closeInputWriter();
  return harness.transport.run(harness.loop);
}

// A shutdown request ends run() with a clean exit while the input is still
// open, and a control handler that reports pending work is run again without
// another notification until it reports none.
void testShutdownRequestAndControlContinuation() {
  {
    Harness harness;
    harness.transport.requestShutdown();
    require(harness.transport.run(harness.loop) == engine::NativeProcessExit::CleanEof,
            "shutdown request did not end the loop cleanly");
  }
  {
    Harness harness;
    int invocations = 0;
    harness.transport.setControlHandler([&] {
      if (++invocations < 3) return true;
      harness.transport.requestShutdown();
      return false;
    });
    harness.transport.controlNotifier()();
    require(harness.transport.run(harness.loop) == engine::NativeProcessExit::CleanEof,
            "control-driven shutdown did not end the loop cleanly");
    require(invocations == 3,
            "control handler was not continued until it reported no pending work");
  }
}

void testCleanEofAndProtocolFailure() {
  require(run({}) == engine::NativeProcessExit::CleanEof,
          "empty clean input did not return clean EOF");
  const std::array<uint8_t, protocol::kFrameHeaderBytes> malformed{};
  require(run(malformed) == engine::NativeProcessExit::ProtocolFailure,
          "malformed input did not return protocol failure");
}

} // namespace

int main() {
  try {
    testCleanEofAndProtocolFailure();
    testShutdownRequestAndControlContinuation();
    std::cout << "native fd transport tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native fd transport tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
