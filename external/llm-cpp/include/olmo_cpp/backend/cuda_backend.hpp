#pragma once

/**
 * include/olmo_cpp/backend/cuda_backend.hpp
 *
 * Concrete IBackend that dispatches to hand-written CUDA kernels in the
 * kernels/ directory (rms_norm.cu, silu_mul.cu, rope.cu). The kernels
 * are exposed to LibTorch via TORCH_LIBRARY(olmo_ops, ...) and looked
 * up at first call by name through c10::Dispatcher. After the first
 * call the OperatorHandle is cached in a static local, so subsequent
 * dispatch is a simple typed call — no string lookup per layer.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/backend/backend.hpp: IBackend base class.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: calls use_cuda_backend() when device == cuda.
 *   - src/backend/cuda_backend.cpp: implementation that resolves the
 *     "olmo_ops::rms_norm", "olmo_ops::silu_mul", "olmo_ops::apply_rope",
 *     "olmo_ops::residual_rms_norm" handles.
 *
 * --- Role in training pipeline ---
 *   When the model runs on a CUDA device, every TransformerBlock
 *   forward eventually calls get_backend().rms_norm(...) which lands
 *   here, jumps into the kernel via the dispatcher, and returns the
 *   result tensor directly with no extra allocations. The default
 *   ATen path is kept as a safety net for fp16 or other unsupported
 *   dtypes; this gives correct fallback behaviour even when the kernel
 *   library wasn't compiled in (OLMO_HAS_CUDA_KERNELS undefined).
 */

#include "olmo_cpp/backend/backend.hpp"

namespace olmo_cpp {

/// CUDA backend that dispatches to fused CUDA kernels.
/// Auto-activated when a CUDA device is selected.
/// Falls back to default ATen ops for any operation without a fused kernel.
class CUDABackend : public IBackend {
 public:
  const char* name() const override { return "cuda_fused"; }

  /// Fused RMSNorm via custom CUDA kernel (vectorized, warp reductions)
  torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) override;

  /// Fused SiLU(gate) * up in single kernel (eliminates intermediate alloc)
  torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up) override;

  /// Fused RoPE via custom CUDA kernel
  torch::Tensor apply_rope(torch::Tensor t, torch::Tensor sin, torch::Tensor cos) override;

  /// Fused residual add + RMSNorm in single kernel (saves full d_model read/write).
  /// Semantics: out = rms_norm(x + residual). Used by add-then-norm patterns.
  torch::Tensor residual_rms_norm(torch::Tensor x, torch::Tensor residual,
                                   torch::Tensor weight, double eps) override;

  /// Fused norm-then-add (item H): out = residual + rms_norm(x) * weight.
  /// Used by the reordered-norm pattern in OLMo-2 / LLaMA-2 blocks where the
  /// sublayer output is normalized before merging back into the residual.
  torch::Tensor rms_norm_add(torch::Tensor x, torch::Tensor residual,
                              torch::Tensor weight, double eps) override;
};

/// Activate the CUDA fused backend globally
void use_cuda_backend();

}  // namespace olmo_cpp
