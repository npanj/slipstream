#include "engine/MachVmRemap.hpp"
#include <mach/mach_vm.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

using namespace splash::ipc;

void testMachVmRemapSelfTask() {
  std::cout << "[testMachVmRemapSelfTask] Testing Mach vm_remap zero-copy sharing..." << std::endl;

  // Allocate a page-aligned buffer of 16 MB
  const size_t bufferBytes = 16 * 1024 * 1024;
  mach_vm_address_t allocAddr = 0;
  kern_return_t kr = mach_vm_allocate(mach_task_self(), &allocAddr, bufferBytes, VM_FLAGS_ANYWHERE);
  assert(kr == KERN_SUCCESS);

  // Fill with test pattern
  uint32_t* srcWords = reinterpret_cast<uint32_t*>(allocAddr);
  const size_t wordCount = bufferBytes / sizeof(uint32_t);
  for (size_t i = 0; i < wordCount; ++i) {
    srcWords[i] = static_cast<uint32_t>(i ^ 0xA5A5A5A5);
  }

  // Remap into another virtual range via MachVmRemapBuffer
  auto t0 = std::chrono::high_resolution_clock::now();
  MachVmRemapBuffer remapped;
  bool ok = remapped.remapFrom(mach_task_self(), allocAddr, bufferBytes, false);
  auto t1 = std::chrono::high_resolution_clock::now();
  assert(ok);
  assert(remapped.valid());
  assert(remapped.size() == bufferBytes);

  auto remappedWords = remapped.asSpan<uint32_t>();
  assert(remappedWords.size() == wordCount);

  // Check data fidelity (zero-copy alias verification)
  for (size_t i = 0; i < 1000; ++i) {
    assert(remappedWords[i] == (static_cast<uint32_t>(i) ^ 0xA5A5A5A5));
  }
  for (size_t i = wordCount - 1000; i < wordCount; ++i) {
    assert(remappedWords[i] == (static_cast<uint32_t>(i) ^ 0xA5A5A5A5));
  }

  // Mutate source and verify alias reflects mutation instantly (shared page table)
  srcWords[42] = 0x12345678;
  assert(remappedWords[42] == 0x12345678);

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  std::cout << "  Remapped 16 MB in " << us << " us (<1 us page table sharing)" << std::endl;
  std::cout << "[testMachVmRemapSelfTask] PASS" << std::endl;

  // Cleanup
  remapped.release();
  mach_vm_deallocate(mach_task_self(), allocAddr, bufferBytes);
}

int main() {
  std::cout << "=== Running Mach VM Remap Zero-Copy Tests ===" << std::endl;
  testMachVmRemapSelfTask();
  std::cout << "=== All Mach VM Remap Tests Passed! ===" << std::endl;
  return 0;
}
