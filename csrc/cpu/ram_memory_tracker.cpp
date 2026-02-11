// SPDX-License-Identifier: Apache-2.0
// RAM memory tracking from C++ to Python

#include "cpu/ram_memory_tracker.h"

#include <Python.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

namespace {

// Global reference to Python memory tracking callback
static PyObject* g_ram_memory_tracker_callback = nullptr;

// Get current process RSS memory in MB
double get_current_rss_mb() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    // ru_maxrss is in KB on Linux, in bytes on macOS
    // Convert to MB
#ifdef __APPLE__
    return usage.ru_maxrss / (1024.0 * 1024.0);
#else
    return usage.ru_maxrss / 1024.0;
#endif
  }
  return 0.0;
}

}  // namespace

namespace vllm_cpu {

void init_ram_memory_tracker(PyObject* callback) {
  if (callback != nullptr && PyCallable_Check(callback)) {
    Py_INCREF(callback);  // Keep a reference
    g_ram_memory_tracker_callback = callback;
  }
}

void log_ram_memory_from_cpp(const char* component_name, size_t size_bytes) {
  if (g_ram_memory_tracker_callback == nullptr) {
    return;
  }

  // Acquire GIL
  PyGILState_STATE gstate = PyGILState_Ensure();

  try {
    // Create Python string for component name
    PyObject* py_component_name = PyUnicode_FromString(component_name);
    if (py_component_name == nullptr) {
      PyErr_Print();
      PyGILState_Release(gstate);
      return;
    }

    // Create Python int for size in bytes
    PyObject* py_size_bytes = PyLong_FromSize_t(size_bytes);
    if (py_size_bytes == nullptr) {
      Py_DECREF(py_component_name);
      PyErr_Print();
      PyGILState_Release(gstate);
      return;
    }

    // Call Python callback: callback(component_name, size_bytes)
    PyObject* result = PyObject_CallFunctionObjArgs(
        g_ram_memory_tracker_callback, py_component_name, py_size_bytes, NULL);

    Py_DECREF(py_component_name);
    Py_DECREF(py_size_bytes);

    if (result == nullptr) {
      PyErr_Print();
    } else {
      Py_DECREF(result);
    }
  } catch (...) {
    // Catch any C++ exceptions
    std::cerr << "Exception in log_ram_memory_from_cpp" << std::endl;
  }

  PyGILState_Release(gstate);
}

void cleanup_ram_memory_tracker() {
  if (g_ram_memory_tracker_callback != nullptr) {
    Py_DECREF(g_ram_memory_tracker_callback);
    g_ram_memory_tracker_callback = nullptr;
  }
}

}  // namespace vllm_cpu
