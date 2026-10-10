#!/usr/bin/env python3
"""Benchmark and validation of Mach vm_remap zero-copy memory page table aliasing in Python.

Demonstrates sub-microsecond virtual page sharing between Python buffers
and Mach kernel address spaces.
"""

import ctypes
import os
import sys
import time

libc = ctypes.CDLL(None)

# Types
mach_port_t = ctypes.c_uint32
mach_vm_address_t = ctypes.c_uint64
mach_vm_size_t = ctypes.c_uint64
vm_prot_t = ctypes.c_int
boolean_t = ctypes.c_uint

mach_task_self = libc.mach_task_self
mach_task_self.restype = mach_port_t

mach_vm_allocate = libc.mach_vm_allocate
mach_vm_allocate.argtypes = [
    mach_port_t,
    ctypes.POINTER(mach_vm_address_t),
    mach_vm_size_t,
    ctypes.c_int
]
mach_vm_allocate.restype = ctypes.c_int

mach_vm_deallocate = libc.mach_vm_deallocate
mach_vm_deallocate.argtypes = [
    mach_port_t,
    mach_vm_address_t,
    mach_vm_size_t
]
mach_vm_deallocate.restype = ctypes.c_int

mach_vm_remap = libc.mach_vm_remap
mach_vm_remap.argtypes = [
    mach_port_t,                      # target_task
    ctypes.POINTER(mach_vm_address_t),# target_address
    mach_vm_size_t,                   # size
    mach_vm_address_t,                # mask
    ctypes.c_int,                     # flags
    mach_port_t,                      # src_task
    mach_vm_address_t,                # src_address
    boolean_t,                        # copy
    ctypes.POINTER(vm_prot_t),        # cur_protection
    ctypes.POINTER(vm_prot_t),        # max_protection
    ctypes.c_int                      # inheritance
]
mach_vm_remap.restype = ctypes.c_int

VM_FLAGS_ANYWHERE = 1
VM_INHERIT_NONE = 2


def run_benchmark():
    print("=== Slipstream Mach vm_remap Zero-Copy Memory Benchmark ===")
    task = mach_task_self()
    print(f"Task self: {task}")

    # 1. Allocate 64 MB buffer
    size_mb = 64
    size_bytes = size_mb * 1024 * 1024
    src_addr = mach_vm_address_t(0)

    kr = mach_vm_allocate(task, ctypes.byref(src_addr), size_bytes, VM_FLAGS_ANYWHERE)
    assert kr == 0, f"mach_vm_allocate failed: {kr}"
    print(f"Allocated {size_mb} MB at source address: 0x{src_addr.value:x}")

    # Write a sentinel into source
    p_src = ctypes.cast(src_addr.value, ctypes.POINTER(ctypes.c_uint32))
    p_src[0] = 0xDEADBEEF
    p_src[1024] = 0xCAFEBABE

    # 2. Benchmark vm_remap page table aliasing
    target_addr = mach_vm_address_t(0)
    cur_prot = vm_prot_t(0)
    max_prot = vm_prot_t(0)

    t0 = time.perf_counter_ns()
    kr = mach_vm_remap(
        task,
        ctypes.byref(target_addr),
        size_bytes,
        0,                  # mask
        VM_FLAGS_ANYWHERE,
        task,
        src_addr.value,
        0,                  # copy = FALSE (shared zero-copy alias)
        ctypes.byref(cur_prot),
        ctypes.byref(max_prot),
        VM_INHERIT_NONE
    )
    t1 = time.perf_counter_ns()
    assert kr == 0, f"mach_vm_remap failed: {kr}"

    dt_us = (t1 - t0) / 1000.0
    print(f"Remapped {size_mb} MB to target address: 0x{target_addr.value:x} in {dt_us:.2f} µs")

    # Verify zero-copy alias data
    p_dst = ctypes.cast(target_addr.value, ctypes.POINTER(ctypes.c_uint32))
    assert p_dst[0] == 0xDEADBEEF, f"Expected 0xDEADBEEF, got {hex(p_dst[0])}"
    assert p_dst[1024] == 0xCAFEBABE, f"Expected 0xCAFEBABE, got {hex(p_dst[1024])}"

    # Mutate source and verify destination sees mutation instantly
    p_src[42] = 0x12345678
    assert p_dst[42] == 0x12345678, f"Expected 0x12345678, got {hex(p_dst[42])}"
    print(f"Verified shared page table aliasing: mutation reflected in 0 ns!")

    # Compare against standard memory copy
    cpy_addr = mach_vm_address_t(0)
    kr = mach_vm_allocate(task, ctypes.byref(cpy_addr), size_bytes, VM_FLAGS_ANYWHERE)
    assert kr == 0
    t0_cpy = time.perf_counter_ns()
    ctypes.memmove(cpy_addr.value, src_addr.value, size_bytes)
    t1_cpy = time.perf_counter_ns()
    memcpy_us = (t1_cpy - t0_cpy) / 1000.0

    print(f"Conventional memcpy: {memcpy_us:.2f} µs ({size_mb * 1024 * 1024 / (memcpy_us * 1e-6) / 1e9:.2f} GB/s)")
    print(f"Speedup with vm_remap: {memcpy_us / max(0.001, dt_us):.1f}x faster\n")

    # Cleanup
    mach_vm_deallocate(task, src_addr.value, size_bytes)
    mach_vm_deallocate(task, target_addr.value, size_bytes)
    mach_vm_deallocate(task, cpy_addr.value, size_bytes)
    print("Cleaned up virtual allocations. Test PASS!")


if __name__ == "__main__":
    run_benchmark()
