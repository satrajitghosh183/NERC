#pragma once

#include "zwt/core/device.hpp"
#include "zwt/layers/parameter.hpp"

#include <cstdint>
#include <vector>

namespace zwt::optim {

// Global L2-norm gradient clipping.
//
// Two surfaces:
//
// * GradClipper — stateful, graph-safe. Caches the per-parameter pointer
//   array, sizes, sumsq accumulator, and scale on device at construction;
//   clip() launches three kernels (zero scalar → sumsq reduction → scale-in-
//   place) without any host syncs. Capture-friendly. Use this in trainers.
//
// * clip_grad_norm() — backward-compat free function that wraps a temp
//   GradClipper. Returns the unclipped global L2 norm to the caller (host
//   sync). Existing tests / one-off scripts still use this.
//
// Param assumptions: every grad is FP32 (zwt's master-grad layout) and lives
// on the same device.
class GradClipper {
 public:
  GradClipper() = default;
  // Snapshot the parameter list. The grad pointers must be stable for the
  // clipper's lifetime — zwt's pool allocator never moves param storage.
  explicit GradClipper(const std::vector<Parameter*>& params);
  ~GradClipper();

  GradClipper(const GradClipper&) = delete;
  GradClipper& operator=(const GradClipper&) = delete;
  GradClipper(GradClipper&&) noexcept;
  GradClipper& operator=(GradClipper&&) noexcept;

  // Launch the clip on the compute stream. Capture-safe: no host sync, no
  // cudaMallocAsync, no cudaMemcpyAsync. max_norm <= 0 ⇒ pure norm read,
  // no in-place scaling. The on-device norm is left in `norm_dev_buf()`
  // for callers that want to pull it at log intervals.
  void clip(float max_norm);

  // CPU device pointer to the unclipped L2 norm (one float). Stable across
  // calls. Pull lazily — never on every step in production.
  const float* norm_dev_buf() const { return d_norm_; }

  // Read the last unclipped norm to host. One D2H sync. Use only at log
  // intervals; capture-region must NOT call this.
  float pull_last_norm() const;

  bool empty() const { return n_ == 0; }

 private:
  void release_();

  int      n_ = 0;
  Device   device_;

  // Device-side caches (CUDA only).
  float**  d_ptrs_  = nullptr;
  int64_t* d_sizes_ = nullptr;
  float*   d_sumsq_ = nullptr;  // 1 float
  float*   d_scale_ = nullptr;  // 1 float
  float*   d_norm_  = nullptr;  // 1 float, unclipped

  // Multi-Tensor-Apply chunk descriptors. Built once at construction so the
  // clip kernels' grid is sized to total chunks (not n_tensors). For a 250M
  // model with mixed param sizes (45M embedding ↔ 896-element norms) this
  // gives uniform SM utilization vs the old "one block per tensor" pattern
  // that wasted blocks on small tensors and starved big ones.
  int*     d_chunk_to_tensor_ = nullptr;  // [n_chunks_]
  int64_t* d_chunk_to_offset_ = nullptr;  // [n_chunks_]
  int      n_chunks_          = 0;

  // CPU fallback: keep the param vector alive by raw pointer copy.
  // Trainer owns the params; clipper just reads.
  std::vector<Parameter*> cpu_params_;
  // Last computed norm on CPU; pull_last_norm reads this when device_ is CPU.
  mutable float cpu_norm_ = 0.f;
};

// Backward-compatible free function. Builds a temporary GradClipper, runs the
// clip, pulls the host-side norm, and tears down. One-shot use only — for
// the hot loop, hold a long-lived GradClipper.
float clip_grad_norm(const std::vector<Parameter*>& params, float max_norm);

}  // namespace zwt::optim
