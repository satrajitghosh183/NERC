#pragma once

#include "zwt/core/device.hpp"
#include "zwt/core/stream.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Built-in CUDA-event profiler for zwt's training loop.
//
// Goal: every stage of training — data load, every forward sub-step, every
// backward sub-step, optimizer, allreduce — gets a named time bucket that
// the trainer prints at the end (and streams to a CSV every log step). No
// dependency on Nsight, nsys, or any external profiler.
//
// Two timing modes:
//   * GPU scopes record cudaEvents on a stream and pull elapsed_ms when the
//     scope closes. Cheap when graphs are off and steady-state.
//   * CPU scopes use std::chrono::steady_clock. For host-side concerns
//     (data loading, optimizer host bookkeeping).
//
// Disabled by default (zero overhead). Enable via:
//   * Env var ZWT_PROFILE=1
//   * Or zwt::Profiler::get().set_enabled(true) from code.
//
// Macros (recommended call sites):
//   ZWT_PROFILE_GPU("attention.qkv_proj.fwd", q.device());
//   ZWT_PROFILE_CPU("data.next");
//
// Both expand to a stack-scoped object whose dtor closes the timer.

namespace zwt {

class Profiler {
 public:
  static Profiler& get();  // process-wide singleton

  // Toggle profiling on/off at runtime. When off, scope ctors return early
  // (no event creation, no chrono call) — essentially free.
  void set_enabled(bool on);
  bool enabled() const { return enabled_; }

  // Reset all accumulated stats. Call between bench warmup and the timed
  // window, or between major phases of a run.
  void reset();

  // Drop bucket names that haven't fired yet (so the summary is clean).
  // No-op currently — left as a hook.
  void compact();

  // Print a sorted summary table to stderr. One row per registered bucket:
  //   stage_name | count | total_ms | mean_us | min_us | max_us | %total
  void print_summary(std::FILE* fp) const;

  // Append one row to a CSV file with the current accumulated stats. Header
  // is written when the file is first opened. The harness consumes this CSV
  // directly — no nsys parser needed.
  void dump_csv(const std::string& path, int64_t step) const;

  // ── Internal scope API ──────────────────────────────────────────────
  // Call from the macros below; not meant for direct use by client code.

  struct GpuScope {
    GpuScope(const char* name, Device dev, StreamHandle s);
    ~GpuScope();
   private:
    int    bucket_id_ = -1;
    void*  start_ev_  = nullptr;
    void*  stop_ev_   = nullptr;
    StreamHandle stream_ = nullptr;
    bool   enabled_ = false;
  };

  struct CpuScope {
    CpuScope(const char* name);
    ~CpuScope();
   private:
    int     bucket_id_ = -1;
    std::chrono::steady_clock::time_point t0_;
    bool    enabled_ = false;
  };

 private:
  Profiler();
  ~Profiler();

  // Internal: look up or create a bucket id for `name`. Thread-safe.
  int  bucket_id_(const char* name);

  // Internal: record a sample (microseconds) into bucket `id`.
  void record_(int id, double us);

  // Internal: drain pending CUDA events that have been recorded but whose
  // elapsed time hasn't been pulled yet. Called once per step from
  // dump_csv() / print_summary(). Without this, GPU-scope samples queued
  // mid-step won't appear in the printout.
  void drain_pending_();

  struct Bucket {
    std::string name;
    uint64_t    count    = 0;
    double      total_us = 0.0;
    double      min_us   = 1e30;
    double      max_us   = 0.0;
  };
  struct Pending {
    int    bucket_id;
    void*  start_ev;
    void*  stop_ev;
  };

  bool                 enabled_;
  std::vector<Bucket>  buckets_;
  std::vector<Pending> pending_;

  // Tiny event pool to avoid cudaEventCreate per scope. Pool is drained
  // when the scope completes; pulled events go back to the pool.
  std::vector<void*>   event_pool_;

  // No locking. The trainer is single-threaded for now and the profiler is
  // not meant to be used from worker threads. If that ever changes, swap
  // these for a mutex-guarded view.
};

}  // namespace zwt

// ── Macros ─────────────────────────────────────────────────────────────
// Use these at function scope. They evaluate to a no-op statement when the
// compiler decides to inline through the disabled path; even when enabled,
// the cost is one event record on construction + one on destruction.

#define ZWT_PROFILE_CONCAT2(a, b) a##b
#define ZWT_PROFILE_CONCAT(a, b)  ZWT_PROFILE_CONCAT2(a, b)

#define ZWT_PROFILE_GPU(name_lit, device_expr)                   \
  ::zwt::Profiler::GpuScope ZWT_PROFILE_CONCAT(_zwt_prof_, __LINE__) {  \
      (name_lit), (device_expr),                                  \
      ::zwt::compute_stream(device_expr).handle }

#define ZWT_PROFILE_GPU_STREAM(name_lit, device_expr, stream_expr) \
  ::zwt::Profiler::GpuScope ZWT_PROFILE_CONCAT(_zwt_prof_, __LINE__) {  \
      (name_lit), (device_expr), (stream_expr) }

#define ZWT_PROFILE_CPU(name_lit)                                       \
  ::zwt::Profiler::CpuScope ZWT_PROFILE_CONCAT(_zwt_prof_, __LINE__) {  \
      (name_lit) }
