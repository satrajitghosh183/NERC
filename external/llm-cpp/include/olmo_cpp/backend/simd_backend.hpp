#pragma once

/**
 * include/olmo_cpp/backend/simd_backend.hpp
 *
 * Concrete IBackend that uses hand-tuned SIMD kernels (NEON on Apple
 * Silicon, AVX2 on x86, scalar fallback otherwise; see
 * src/backend/kernels/simd_elementwise.cpp). Activated when the model
 * runs on CPU with float32 contiguous tensors. For any tensor that is
 * non-contiguous, non-CPU, or not float32 the backend falls through to
 * IBackend's ATen reference path automatically.
 *
 * Pairs with the thread-local arena: begin_scope / end_scope wrap a
 * region of scratch allocations, so per-block intermediates recycle the
 * same memory range without repeatedly hitting the system allocator.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/backend/backend.hpp: IBackend base class.
 *   - olmo_cpp/backend/arena.hpp:   thread_arena() for scratch tensors.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: calls use_simd_backend() when a CPU device is chosen.
 *   - src/backend/simd_backend.cpp: the implementation file.
 *
 * --- Role in training pipeline ---
 *   On CPU, the dominant cost of an RMSNorm is reading the row twice
 *   (variance pass + scale pass). The SIMD kernel does both in one
 *   pass with vectorized FMAs, then a horizontal reduction. With the
 *   arena, hot scratch buffers stay in L2/L3 across blocks.
 */

#include "olmo_cpp/backend/backend.hpp"
#include "olmo_cpp/backend/arena.hpp"

namespace olmo_cpp {

/// SIMD-accelerated compute backend with arena memory management.
/// Dispatches to fused SIMD kernels for contiguous CPU float32 tensors.
/// Falls through to default ATen implementations for other dtypes/devices.
/// Uses thread-local arena allocator for scratch memory within scopes.
class SIMDBackend : public IBackend {
 public:
  const char* name() const override { return "simd"; }

  torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) override;
  torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up) override;
  torch::Tensor apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) override;
  torch::Tensor residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                   torch::Tensor weight, double eps) override;

  /// Arena scope management: begin_scope saves the arena mark,
  /// end_scope resets to that mark (freeing all scratch within the scope).
  void begin_scope() override;
  void end_scope() override;

  /// Allocate a temporary tensor from the arena (fast, no malloc).
  /// Falls back to torch::empty if arena is full.
  torch::Tensor alloc_scratch(at::IntArrayRef sizes, torch::ScalarType dtype,
                               torch::Device device = torch::kCPU);

 private:
  static bool can_use_simd(const torch::Tensor& t);
  std::vector<size_t> scope_stack_;
};

/// Convenience: install the SIMD backend as the global backend.
void use_simd_backend();

}  // namespace olmo_cpp
