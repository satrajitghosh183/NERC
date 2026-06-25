#pragma once
/**
 * include/olmo_cpp/optim/grad_clip.hpp
 *
 * GPU-resident global gradient clipping by norm. Equivalent to PyTorch's
 * torch.nn.utils.clip_grad_norm_ but stays entirely on the GPU so we don't
 * pay a CUDA→host sync on every training step.
 *
 * Math: let G = concat of all defined gradient tensors. Compute
 *     total_norm = ||G||_p   (default p = 2)
 *     coef       = min(max_norm / (total_norm + 1e-6), 1)
 * and rescale every gradient in-place by coef. Result: ||G_new||_p ≤ max_norm.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: tensor type and at::_foreach_* fused ops in the .cpp.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp:236,532,537,801,961: invoked between backward() and
 *     optimizer.step() with cfg.max_grad_norm.
 *
 * --- Role in training pipeline ---
 *   Standard deep-learning regularization. Called once per microbatch right
 *   after backward (or after grad accumulation completes) and before the
 *   optimizer step. Returning a device tensor keeps logging async — callers
 *   can .item<float>() later if they want the value for a metric.
 */
#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// GPU-resident gradient clipping. Computes norms and applies clipping
/// entirely on device with no D2H synchronization.
/// Returns total_norm as a device-resident tensor (caller can .item<float>()
/// later for logging, but clipping is already applied).
torch::Tensor clip_grad_norm_gpu(
    const std::vector<torch::Tensor>& parameters,
    double max_norm,
    double norm_type = 2.0);

}  // namespace olmo_cpp
