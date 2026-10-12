#pragma once

#include "AwakeClock.hpp"
#include "MetalBackend.hpp"

#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <chrono>
#include <cstdint>
#include <mutex>

namespace splash::metal {

// Holds the buffers of one residency set wired between the commands of a
// command queue. Metal wires them while a command runs and lets them go a few
// seconds later, and one request holds them only about two seconds, so a
// heartbeat requests residency every 500 ms. After keepAlive without a command
// it ends residency, which Metal applies at its next GPU operation on any
// queue: a one-byte blit on a queue of its own, as the runtime keeps exactly
// one command in flight on the command queue. The set is used only on the
// heartbeat's serial queue.
class Residency final {
public:
  Residency(id<MTLDevice> device, id<MTLCommandQueue> commands,
            double keepAliveSeconds)
      : commands_(commands), keepAlive_(keepAliveSeconds) {
    set_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new]
                                           error:nil];
    releaseQueue_ = [device newCommandQueue];
    releaseTarget_ = [device newBufferWithLength:1
                                         options:MTLResourceStorageModePrivate];
    if (!set_ || !releaseQueue_ || !releaseTarget_)
      throw MetalBackendError("unable to create the Metal residency set");
    [commands_ addResidencySet:set_];
    queue_ = dispatch_queue_create(
        "splash.metal.residency",
        dispatch_queue_attr_make_with_autorelease_frequency(
            DISPATCH_QUEUE_SERIAL, DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM));
    heartbeat_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue_);
    dispatch_source_set_event_handler(heartbeat_, ^{ beat(); });
    dispatch_activate(heartbeat_);
  }

  ~Residency() {
    dispatch_source_cancel(heartbeat_);
    // Runs after any beat or request already queued: they use this object.
    dispatch_sync(queue_, ^{
      @autoreleasepool {
        [commands_ removeResidencySet:set_];
        std::lock_guard lock(mutex_);
        if (held_) release();
      }
    });
  }

  Residency(const Residency &) = delete;
  Residency &operator=(const Residency &) = delete;

  // Wires the buffer before returning and restarts the keep-alive.
  void add(id<MTLBuffer> buffer) {
    dispatch_sync(queue_, ^{
      [set_ addAllocation:buffer];
      [set_ commit];
      [set_ requestResidency];
    });
    {
      std::lock_guard lock(mutex_);
      bytes_ += buffer.allocatedSize;
    }
    use();
  }

  void remove(id<MTLBuffer> buffer) {
    dispatch_sync(queue_, ^{
      [set_ removeAllocation:buffer];
      [set_ commit];
    });
    std::lock_guard lock(mutex_);
    bytes_ -= buffer.allocatedSize;
  }

  // Marks a command. A lapsed set is requested again at once, off the
  // caller's thread, and the heartbeat resumes.
  void use() {
    {
      std::lock_guard lock(mutex_);
      lastUse_ = AwakeClock::now();
      if (held_ || !bytes_) return;
      held_ = true;
    }
    dispatch_async(queue_, ^{
      [set_ requestResidency];
      dispatch_source_set_timer(heartbeat_, dispatch_time(DISPATCH_TIME_NOW, kBeat),
                                kBeat, kBeat / 10);
    });
  }

  // The bytes of the set while it is not held.
  [[nodiscard]] uint64_t lapsedBytes() const {
    std::lock_guard lock(mutex_);
    return held_ ? 0 : bytes_;
  }

private:
  static constexpr uint64_t kBeat = 500 * NSEC_PER_MSEC;

  void beat() {
    bool lapsed;
    {
      std::lock_guard lock(mutex_);
      lapsed = AwakeClock::now() - lastUse_ >= keepAlive_;
      held_ = !lapsed;
    }
    if (!lapsed) {
      [set_ requestResidency];
      return;
    }
    dispatch_source_set_timer(heartbeat_, DISPATCH_TIME_FOREVER, 0, 0);
    release();
  }

  void release() {
    [set_ endResidency];
    id<MTLCommandBuffer> command = [releaseQueue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit fillBuffer:releaseTarget_ range:NSMakeRange(0, 1) value:0];
    [blit endEncoding];
    [command commit];
  }

  __strong id<MTLCommandQueue> commands_ = nil;
  __strong id<MTLResidencySet> set_ = nil;
  __strong id<MTLCommandQueue> releaseQueue_ = nil;
  __strong id<MTLBuffer> releaseTarget_ = nil;
  __strong dispatch_queue_t queue_ = nil;
  __strong dispatch_source_t heartbeat_ = nil;
  const std::chrono::duration<double> keepAlive_;
  mutable std::mutex mutex_;
  AwakeClock::time_point lastUse_;
  bool held_ = false;
  uint64_t bytes_ = 0;
};

} // namespace splash::metal
