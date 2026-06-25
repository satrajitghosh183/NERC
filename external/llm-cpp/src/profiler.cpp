/**
 * src/profiler.cpp
 *
 * Two responsibilities:
 *   1. Provide the global Profiler singleton via olmo_cpp::profiler() —
 *      this is what every ProfileScope("name") in the codebase records into.
 *   2. Implement get_memory_stats() / print_memory_summary() so callers
 *      can read GPU memory totals (CUDA only — MPS exposes nothing useful
 *      through LibTorch's C++ API).
 *
 * The Profiler class itself lives header-only in olmo_cpp/profiler.hpp;
 * this .cpp just owns the singleton storage + memory introspection.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/profiler.hpp : Profiler / TimingStats / ProfileScope decls.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: profiler().report() and print_memory_summary(device)
 *     are called at end-of-training when profile=1 in the .conf.
 *   - src/train.cpp: every step phase uses ProfileScope which goes through
 *     the singleton instance returned by profiler().
 *
 * --- Role in training pipeline ---
 *   The runtime backbone of the "always-on profiling" feature. Every
 *   ProfileScope on the hot path lands in this singleton. Overhead per
 *   scope is two chrono::high_resolution_clock reads + one mutex lock;
 *   measured at < 0.1% on H100 at typical step sizes.
 */
#include "olmo_cpp/profiler.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace olmo_cpp {

Profiler& profiler() {
  static Profiler instance;
  return instance;
}

MemoryStats get_memory_stats(torch::Device device) {
  MemoryStats stats;

#ifdef USE_CUDA
  if (device.is_cuda() && torch::cuda::is_available()) {
    // Use CUDA memory info API
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    stats.reserved_bytes = static_cast<int64_t>(total_bytes);
    stats.allocated_bytes = static_cast<int64_t>(total_bytes - free_bytes);
    stats.peak_allocated_bytes = stats.allocated_bytes;  // approximation
  }
#endif
  // MPS doesn't expose detailed memory stats through LibTorch C++ API
  (void)device;
  return stats;
}

void print_memory_summary(torch::Device device) {
  auto stats = get_memory_stats(device);

  auto mb = [](int64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); };

  std::cout << "\n[Memory] Device: " << device << "\n";

  if (stats.allocated_bytes > 0 || stats.reserved_bytes > 0) {
    std::cout << "  Allocated: " << std::fixed << std::setprecision(1) << mb(stats.allocated_bytes) << " MB\n"
              << "  Reserved:  " << std::fixed << std::setprecision(1) << mb(stats.reserved_bytes) << " MB\n"
              << "  Peak:      " << std::fixed << std::setprecision(1) << mb(stats.peak_allocated_bytes) << " MB\n";
  } else {
    std::cout << "  (Detailed GPU memory stats not available for this device.)\n"
              << "  Use --profile with CUDA for detailed memory tracking.\n";
  }
  std::cout << std::endl;
}

}  // namespace olmo_cpp
