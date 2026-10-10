#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_map.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <span>

namespace splash::ipc {

/**
 * Mach VM Remap Zero-Copy Memory Sharing
 *
 * Enables Darwin Mach kernel vm_remap() page table sharing between processes,
 * allowing the Python agent or external tooling to pass arbitrary page-aligned
 * memory buffers directly into the C++ inference engine virtual memory address
 * space in sub-microsecond time with zero byte copies.
 */
class MachVmRemapBuffer final {
public:
  MachVmRemapBuffer() = default;

  ~MachVmRemapBuffer() {
    release();
  }

  MachVmRemapBuffer(const MachVmRemapBuffer&) = delete;
  MachVmRemapBuffer& operator=(const MachVmRemapBuffer&) = delete;

  MachVmRemapBuffer(MachVmRemapBuffer&& other) noexcept {
    *this = std::move(other);
  }

  MachVmRemapBuffer& operator=(MachVmRemapBuffer&& other) noexcept {
    if (this != &other) {
      release();
      address_ = other.address_;
      size_ = other.size_;
      other.address_ = 0;
      other.size_ = 0;
    }
    return *this;
  }

  /// Remap a memory region from targetTask (or current task if mach_task_self())
  /// into the current process address space.
  bool remapFrom(mach_port_t sourceTask,
                 mach_vm_address_t sourceAddress,
                 mach_vm_size_t size,
                 bool copyOnWrite = false) {
    release();

    if (size == 0) return false;

    mach_port_t targetTask = mach_task_self();
    mach_vm_address_t targetAddress = 0;
    vm_prot_t curProtection = VM_PROT_NONE;
    vm_prot_t maxProtection = VM_PROT_NONE;

    kern_return_t kr = mach_vm_remap(
        targetTask,
        &targetAddress,
        size,
        0, // mask (any alignment)
        VM_FLAGS_ANYWHERE,
        sourceTask,
        sourceAddress,
        copyOnWrite ? TRUE : FALSE, // copy
        &curProtection,
        &maxProtection,
        VM_INHERIT_NONE
    );

    if (kr != KERN_SUCCESS) {
      return false;
    }

    address_ = targetAddress;
    size_ = size;
    return true;
  }

  void release() noexcept {
    if (address_ != 0 && size_ != 0) {
      mach_vm_deallocate(mach_task_self(), address_, size_);
      address_ = 0;
      size_ = 0;
    }
  }

  [[nodiscard]] const void* data() const noexcept {
    return reinterpret_cast<const void*>(address_);
  }

  [[nodiscard]] void* data() noexcept {
    return reinterpret_cast<void*>(address_);
  }

  [[nodiscard]] size_t size() const noexcept {
    return static_cast<size_t>(size_);
  }

  [[nodiscard]] bool valid() const noexcept {
    return address_ != 0 && size_ != 0;
  }

  template <typename T>
  [[nodiscard]] std::span<const T> asSpan() const noexcept {
    if (!valid()) return {};
    return std::span<const T>(reinterpret_cast<const T*>(data()), size_ / sizeof(T));
  }

private:
  mach_vm_address_t address_ = 0;
  mach_vm_size_t size_ = 0;
};

} // namespace splash::ipc
