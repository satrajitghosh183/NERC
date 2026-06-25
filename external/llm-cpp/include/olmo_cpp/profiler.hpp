#pragma once

/**
 * include/olmo_cpp/profiler.hpp
 *
 * Always-on, header-mostly wall-clock profiler. Provides a global
 * `Profiler` singleton plus an RAII `ProfileScope` that records the
 * lifetime of a named region. Designed for <0.1% overhead so it can stay
 * compiled into the release binary; the framework instruments hot paths
 * (forward, backward, optimizer step, data loading, allreduce) so an
 * end-of-run report immediately surfaces where time is going. Also exposes
 * a small `MemoryStats` helper backed by `cudaMemGetInfo` for GPU runs.
 *
 * Note: this is a CPU-time profiler. It does NOT insert CUDA events, so
 * GPU-only kernels appear instantaneous; combine with the dedicated
 * CUDA-event profiler in zwt or `nvprof`/`nsys` for kernel-level data.
 *
 * --- Includes from this project ---
 *   (none — pure stdlib + torch headers)
 *
 * --- Callers (concrete uses elsewhere in this repo) ---
 *   - src/main.cpp: `olmo_cpp::profiler().report("Training Profile")`
 *     and `olmo_cpp::print_memory_summary(device)` after training ends,
 *     when `[training] profile=1`.
 *   - src/train.cpp: `ProfileScope` is wrapped around `step_total`,
 *     `data_loading`, `forward`, `backward`, `allreduce`, and
 *     `optimizer_step` in `train_epoch()`.
 *
 * --- Role in training pipeline ---
 *   Header-only state lives in a function-local static `Profiler` (defined
 *   in profiler.cpp). `ProfileScope` ctor calls `start(name)`, dtor calls
 *   `stop(name)` — accumulating count/total/min/max/sum_sq per region into
 *   a shared mutex-guarded map. Reporting sorts by total time descending
 *   and prints a nicely formatted box at end-of-run.
 */

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <mutex>
#include <cmath>
#include <atomic>
#include <torch/torch.h>

#if defined(USE_CUDA)
#  include <cuda_runtime.h>
#  include <c10/cuda/CUDAStream.h>
#  define OLMO_PROFILER_CUDA 1
#endif

namespace olmo_cpp {

/// Lightweight profiler for understanding where time is spent in training/inference.
/// Thread-safe. Designed to be always-on with minimal overhead (<0.1%).
///
/// Usage:
///   {
///     ProfileScope scope("attention");
///     // ... attention code ...
///   }
///
///   // Or manually:
///   profiler().start("ffn");
///   // ... ffn code ...
///   profiler().stop("ffn");
///
///   // At end of training:
///   profiler().report();

/// Per-region timing accumulator. One instance per unique name passed to
/// `Profiler::start/stop` or `ProfileScope`. All times in microseconds
/// internally; `_ms` accessors are convenience.
struct TimingStats {
  /// Number of completed (start, stop) pairs recorded.
  int64_t count = 0;
  double total_us = 0.0;     // microseconds
  /// Minimum single-call time seen so far. Initialised to a huge sentinel
  /// so the first sample replaces it via `std::min`.
  double min_us = 1e18;
  /// Maximum single-call time seen so far.
  double max_us = 0.0;
  /// Sum of squares of per-call times, used to compute std-dev online.
  double sum_sq_us = 0.0;    // for variance

  /// Average per-call time in microseconds (0 if no samples).
  double mean_us() const { return count > 0 ? total_us / count : 0.0; }
  /// Population standard deviation in microseconds. Uses the standard
  /// `E[X^2] - E[X]^2` identity over the recorded sums.
  double std_us() const {
    if (count < 2) return 0.0;
    double mean = mean_us();
    return std::sqrt(sum_sq_us / count - mean * mean);
  }
  /// Total accumulated time in milliseconds.
  double total_ms() const { return total_us / 1000.0; }
  /// Average per-call time in milliseconds.
  double mean_ms() const { return mean_us() / 1000.0; }
};

/// Mutex-protected, name-keyed wall-clock profiler. Use the global
/// instance via `profiler()`; do not construct your own.
class Profiler {
 public:
  /// High-resolution monotonic clock — typically backed by
  /// `clock_gettime(CLOCK_MONOTONIC)` on Linux.
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  /// Begin timing a named region. Pairs with `stop(name)`. Re-entrant
  /// `start` on the same name simply overwrites the start time, so users
  /// must not nest `start/stop` pairs for the same name.
  void start(const std::string& name) {
    // Lock guards both the active-regions map and the stats map.
    std::lock_guard<std::mutex> lock(mutex_);
    // Record the entry timestamp. Overwrites any prior unmatched start.
    active_[name] = Clock::now();
  }

  /// End timing for a named region. Computes elapsed time, removes the
  /// active entry, and folds the sample into `stats_[name]`. Silently
  /// no-ops if `start(name)` was not called (defensive).
  void stop(const std::string& name) {
    // Capture end time BEFORE acquiring the lock so contention isn't
    // attributed to the timed region.
    auto end = Clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.find(name);
    if (it == active_.end()) return;

    // Elapsed time as a double in microseconds.
    double us = std::chrono::duration<double, std::micro>(end - it->second).count();
    // Remove from active set so the same name can be re-timed.
    active_.erase(it);

    // Fold sample into running stats: count, sum, min, max, sum-of-squares.
    auto& stats = stats_[name];
    stats.count++;
    stats.total_us += us;
    stats.min_us = std::min(stats.min_us, us);
    stats.max_us = std::max(stats.max_us, us);
    stats.sum_sq_us += us * us;
  }

  /// Record a pre-measured duration (e.g. from a CUDA event range or a
  /// callsite that already measured its own time). Skips the
  /// start/stop bookkeeping.
  void record(const std::string& name, double us) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Same fold as `stop()` above.
    auto& stats = stats_[name];
    stats.count++;
    stats.total_us += us;
    stats.min_us = std::min(stats.min_us, us);
    stats.max_us = std::max(stats.max_us, us);
    stats.sum_sq_us += us * us;
  }

  /// Get stats for a specific region. Returns a default-constructed
  /// (zero-count) `TimingStats` if no samples were recorded.
  TimingStats get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_.find(name);
    if (it != stats_.end()) return it->second;
    return {};
  }

  /// Print a formatted report sorted by total time. Called at end-of-run
  /// from main when `[training] profile=1`. Pure I/O — does not modify state.
  void report(const std::string& title = "Profile Report") const {
    std::lock_guard<std::mutex> lock(mutex_);
    // No samples ever recorded: print and bail out.
    if (stats_.empty()) {
      std::cout << "[Profiler] No data collected.\n";
      return;
    }

    // Sort by total time descending so the worst offenders show up first.
    std::vector<std::pair<std::string, TimingStats>> sorted(stats_.begin(), stats_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.total_us > b.second.total_us; });

    // Compute total wall time across all regions for the % column.
    // Note: regions can overlap (nested scopes) so this can over-count;
    // it's still useful as a relative weighting.
    double total_wall_us = 0.0;
    for (const auto& [name, s] : sorted) total_wall_us += s.total_us;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────────┐\n"
              << "│ " << std::left << std::setw(78) << title << "│\n"
              << "├──────────────────────┬────────┬──────────┬──────────┬──────────┬───────────────┤\n"
              << "│ Region               │ Calls  │ Total ms │ Mean ms  │ Std ms   │ % of Total    │\n"
              << "├──────────────────────┼────────┼──────────┼──────────┼──────────┼───────────────┤\n";

    for (const auto& [name, s] : sorted) {
      // Percentage of cumulative tracked time consumed by this region.
      double pct = total_wall_us > 0 ? (s.total_us / total_wall_us * 100.0) : 0.0;
      // Tiny ASCII bar: one '#' per 5%. Pure cosmetic.
      std::string bar(static_cast<size_t>(pct / 5.0), '#');

      std::cout << "│ " << std::left << std::setw(21) << name.substr(0, 21)
                << "│ " << std::right << std::setw(6) << s.count
                << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(1) << s.total_ms()
                << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << s.mean_ms()
                << " │ " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << (s.std_us() / 1000.0)
                << " │ " << std::right << std::setw(5) << std::fixed << std::setprecision(1) << pct
                << "% " << std::left << std::setw(7) << bar
                << "│\n";
    }

    std::cout << "├──────────────────────┴────────┴──────────┴──────────┴──────────┴───────────────┤\n"
              << "│ Total wall time: " << std::fixed << std::setprecision(1)
              << std::setw(10) << (total_wall_us / 1000.0) << " ms"
              << std::setw(49) << " " << "│\n"
              << "└─────────────────────────────────────────────────────────────────────────────────┘\n"
              << std::endl;
  }

  /// Reset all collected data. Useful between phases (e.g. warmup vs.
  /// steady-state) to isolate measurements.
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.clear();
    active_.clear();
  }

  /// Get all stats (for programmatic access). Returns a copy under lock so
  /// the caller can iterate without holding the mutex.
  std::unordered_map<std::string, TimingStats> all_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  /// Enable CUDA-event timing: ProfileScope then measures true GPU execution
  /// time of each region (via cudaEvent + a sync at scope exit) instead of
  /// CPU wall time. Set once from main when [training] profile=1 and the
  /// device is CUDA. Adds a per-stage sync, so only use it for profiling runs.
  void set_cuda_timing(bool enabled) { cuda_timing_.store(enabled); }
  bool cuda_timing_enabled() const { return cuda_timing_.load(); }

 private:
  /// Mutex protecting both `stats_` and `active_`. Marked `mutable` so
  /// const methods (`get`, `report`, `all_stats`) can lock it.
  mutable std::mutex mutex_;
  /// Completed-region stats keyed by name.
  std::unordered_map<std::string, TimingStats> stats_;
  /// Currently-open regions: name -> start timestamp.
  std::unordered_map<std::string, TimePoint> active_;
  /// When true, ProfileScope uses CUDA events (GPU time) instead of wall clock.
  std::atomic<bool> cuda_timing_{false};
};

/// Global profiler instance. Returns a reference to a function-local
/// static defined in profiler.cpp; safe to call from any thread.
Profiler& profiler();

/// RAII scope timer — measures the lifetime of the object. Standard
/// pattern: `{ ProfileScope s("region_name"); /* code */ }`.
class ProfileScope {
 public:
  /// Start timing the given region. Defaults to the global profiler;
  /// pass a custom one for unit tests.
  explicit ProfileScope(const std::string& name, Profiler& p = profiler())
      : name_(name), profiler_(p) {
#if OLMO_PROFILER_CUDA
    if (profiler_.cuda_timing_enabled()) {
      // Record a CUDA event on the current stream at region entry. The matching
      // event at exit, plus a sync, gives true GPU execution time for the stage.
      cuda_stream_ = c10::cuda::getCurrentCUDAStream().stream();
      cudaEventCreate(&start_ev_);
      cudaEventCreate(&stop_ev_);
      cudaEventRecord(start_ev_, cuda_stream_);
      cuda_active_ = true;
      return;
    }
#endif
    profiler_.start(name_);
  }
  /// Stop timing and fold elapsed time into the profiler.
  ~ProfileScope() {
#if OLMO_PROFILER_CUDA
    if (cuda_active_) {
      cudaEventRecord(stop_ev_, cuda_stream_);
      cudaEventSynchronize(stop_ev_);   // wait for this stage's GPU work to finish
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start_ev_, stop_ev_);
      profiler_.record(name_, static_cast<double>(ms) * 1000.0);  // record() takes µs
      cudaEventDestroy(start_ev_);
      cudaEventDestroy(stop_ev_);
      return;
    }
#endif
    profiler_.stop(name_);
  }
  /// Non-copyable — copying would double-stop the same region.
  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;

 private:
  /// Region name passed at construction; used for both start and stop.
  std::string name_;
  /// Reference to the profiler that holds the running counts.
  Profiler& profiler_;
#if OLMO_PROFILER_CUDA
  bool cuda_active_ = false;
  cudaStream_t cuda_stream_ = nullptr;
  cudaEvent_t start_ev_ = nullptr;
  cudaEvent_t stop_ev_ = nullptr;
#endif
};

/// Memory tracking utilities. Snapshot of GPU memory at a point in time.
/// All values in bytes; zero on devices that don't expose memory stats.
struct MemoryStats {
  /// Live (allocated and not freed) bytes attributed to torch.
  int64_t allocated_bytes = 0;
  /// Total bytes reserved by the caching allocator (>= allocated).
  int64_t reserved_bytes = 0;
  /// Peak high-water mark of allocated bytes since process start.
  int64_t peak_allocated_bytes = 0;
};

/// Get current GPU memory stats (CUDA or MPS). On CUDA this calls
/// `cudaMemGetInfo`; on CPU/MPS it returns zeros.
MemoryStats get_memory_stats(torch::Device device);

/// Print memory summary for the given device — used in main after training.
void print_memory_summary(torch::Device device);

}  // namespace olmo_cpp
