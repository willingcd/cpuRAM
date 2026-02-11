// SPDX-License-Identifier: Apache-2.0
// RAM memory tracking header for C++

#ifndef VLLM_CPU_RAM_MEMORY_TRACKER_H
#define VLLM_CPU_RAM_MEMORY_TRACKER_H

#include <Python.h>
#include <cstddef>

namespace vllm_cpu {

// Initialize the RAM memory tracker with a Python callback
// The callback should have signature: callback(component_name: str, size_bytes: int) -> None
void init_ram_memory_tracker(PyObject* callback);

// Log RAM memory usage from C++ code
// component_name: Name of the component allocating memory
// size_bytes: Size of memory allocated in bytes
void log_ram_memory_from_cpp(const char* component_name, size_t size_bytes);

// Cleanup the RAM memory tracker
void cleanup_ram_memory_tracker();

}  // namespace vllm_cpu

#endif  // VLLM_CPU_RAM_MEMORY_TRACKER_H
