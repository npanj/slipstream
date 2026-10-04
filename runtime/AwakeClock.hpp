#pragma once

#include <time.h>

#include <chrono>

namespace splash {

// The clock the runtime measures every timeout, keep-alive and duration on:
// the time the Mac has been awake. The standard library's steady clock reads
// CLOCK_MONOTONIC_RAW, which keeps counting while the Mac sleeps, so a limit
// measured on it runs out the moment the Mac wakes from a longer sleep, and a
// command in flight across the sleep reads as long as the sleep. The server's
// time.monotonic() reads this same clock, so both sides count alike.
struct AwakeClock final {
  using duration = std::chrono::nanoseconds;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = std::chrono::time_point<AwakeClock>;
  static constexpr bool is_steady = true;

  [[nodiscard]] static time_point now() noexcept {
    return time_point(
        duration(static_cast<rep>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW))));
  }
};

} // namespace splash
