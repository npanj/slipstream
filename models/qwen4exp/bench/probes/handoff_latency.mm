// How fast can the GPU and the host hand control back and forth, and is the
// data the GPU wrote visible to the host at the hand-off?
//
// Decode hands off once per layer: the GPU finishes a layer's router, the
// host picks experts, the GPU continues. Two ways, 48 stages each:
//
//   events  one command buffer per stage; the stage ends with a shared-event
//           signal and the next waits on the event (what the engine does).
//   flags   one command buffer; after a stage a one-thread kernel stores a
//           flag in shared memory, and before the next a one-thread kernel
//           spins on a flag the host writes.
//
// Each stage writes `stage` into 1M floats; at every hand-off the host checks
// a spread of them, so stale data would show up as a count of bad reads.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include "AwakeClock.hpp"

static const char *kSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void work(device float *data [[buffer(0)]], constant uint &stage [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
  data[id] = float(stage);
}
kernel void raise_flag(device atomic_uint *flag [[buffer(0)]], constant uint &stage [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
  if (id == 0) atomic_store_explicit(flag, stage, memory_order_relaxed);
}
kernel void await_flag(device atomic_uint *flag [[buffer(0)]], constant uint &stage [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
  if (id == 0)
    while (atomic_load_explicit(flag, memory_order_relaxed) < stage) {}
}
)";

static constexpr uint32_t kStages = 48;
static constexpr uint32_t kFloats = 1u << 20;

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(kSource) options:nil error:&error];
    if (!library) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
    auto pipeline = [&](const char *name) {
      return [device newComputePipelineStateWithFunction:[library newFunctionWithName:@(name)] error:nil];
    };
    id<MTLComputePipelineState> work = pipeline("work"), raise = pipeline("raise_flag"),
                                await = pipeline("await_flag");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> data = [device newBufferWithLength:kFloats * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> flags = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    auto *gpuFlag = reinterpret_cast<std::atomic<uint32_t> *>(flags.contents);
    auto *hostFlag = reinterpret_cast<std::atomic<uint32_t> *>(static_cast<char *>(flags.contents) + 128);
    const float *values = static_cast<const float *>(data.contents);
    auto check = [&](uint32_t stage) {
      uint32_t bad = 0;
      for (uint32_t i = 0; i < kFloats; i += 4099) bad += values[i] != float(stage);
      return bad;
    };
    auto encodeWork = [&](id<MTLComputeCommandEncoder> encoder, uint32_t stage) {
      [encoder setComputePipelineState:work];
      [encoder setBuffer:data offset:0 atIndex:0];
      [encoder setBytes:&stage length:4 atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(kFloats, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    for (int round = 0; round < 4; ++round) {
      // events
      {
        id<MTLSharedEvent> event = [device newSharedEvent];
        uint32_t bad = 0;
        const auto start = AwakeClock::now();
        std::vector<id<MTLCommandBuffer>> commands;
        for (uint32_t k = 1; k <= kStages; ++k) {
          id<MTLCommandBuffer> command = [queue commandBuffer];
          if (k > 1) [command encodeWaitForEvent:event value:2 * k];
          id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
          encodeWork(encoder, k);
          [encoder endEncoding];
          [command encodeSignalEvent:event value:2 * k + 1];
          [command commit];
          commands.push_back(command);
        }
        for (uint32_t k = 1; k <= kStages; ++k) {
          while (event.signaledValue < 2 * k + 1) {}
          bad += check(k);
          if (k < kStages) event.signaledValue = 2 * (k + 1);
        }
        [commands.back() waitUntilCompleted];
        const double us = std::chrono::duration<double, std::micro>(AwakeClock::now() - start).count();
        printf("events: %6.1f us per stage, %u stale reads\n", us / kStages, bad);
      }
      // flags
      {
        gpuFlag->store(0);
        hostFlag->store(0);
        uint32_t bad = 0;
        const auto start = AwakeClock::now();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        for (uint32_t k = 1; k <= kStages; ++k) {
          if (k > 1) {
            [encoder setComputePipelineState:await];
            [encoder setBuffer:flags offset:128 atIndex:0];
            [encoder setBytes:&k length:4 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
          }
          encodeWork(encoder, k);
          [encoder setComputePipelineState:raise];
          [encoder setBuffer:flags offset:0 atIndex:0];
          [encoder setBytes:&k length:4 atIndex:1];
          [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
        [encoder endEncoding];
        [command commit];
        for (uint32_t k = 1; k <= kStages; ++k) {
          const auto deadline = AwakeClock::now() + std::chrono::seconds(2);
          while (gpuFlag->load(std::memory_order_acquire) < k) {
            if (AwakeClock::now() > deadline) {
              printf("flags: stage %u flag never became visible (reads %u)\n", k, gpuFlag->load());
              hostFlag->store(1000000);  // release the GPU so the buffer can finish
              [command waitUntilCompleted];
              return 2;
            }
          }
          bad += check(k);
          if (k < kStages) hostFlag->store(k + 1, std::memory_order_release);
        }
        [command waitUntilCompleted];
        const double us = std::chrono::duration<double, std::micro>(AwakeClock::now() - start).count();
        printf("flags:  %6.1f us per stage, %u stale reads, status %ld\n", us / kStages, bad, (long)command.status);
      }
    }
  }
  return 0;
}
