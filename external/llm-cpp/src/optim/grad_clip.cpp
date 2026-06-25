/**
 * src/optim/grad_clip.cpp
 *
 * Implementation of clip_grad_norm_gpu. The whole operation — gathering
 * grads, computing per-param norms, the global norm, the clip coefficient,
 * and the in-place rescale — runs on the GPU. The returned total_norm is a
 * device tensor so the caller can defer its .item() sync.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/optim/grad_clip.hpp: declares clip_grad_norm_gpu.
 *
 * --- ATen foreach ops ---
 *   - _foreach_norm: tensor-list p-norm; one fused launch.
 *   - _foreach_mul:  tensor-list elementwise multiply by a (possibly tensor)
 *                    scalar; used to rescale every gradient in one launch.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: called once per microbatch between backward() and
 *     optimizer.step(), with the user's cfg.max_grad_norm.
 *
 * --- Role in training pipeline ---
 *   Standard global-norm gradient clipping. Critical for stability of
 *   transformer training under mixed-precision; spikes in a single layer's
 *   grad are damped by the global rescale.
 */
#include "olmo_cpp/optim/grad_clip.hpp"
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/_foreach_mul.h>

namespace olmo_cpp {

/// Clip each gradient g_i in place such that ||concat(g_i)||_p ≤ max_norm.
torch::Tensor clip_grad_norm_gpu(
    const std::vector<torch::Tensor>& parameters,
    double max_norm,
    double norm_type) {

  // Collect defined gradients into a thread-local scratch vector. Reusing
  // the vector's capacity avoids a per-step heap allocation + N push_backs
  // on the hot training loop (called once per optimizer step).
  static thread_local std::vector<torch::Tensor> grads;
  grads.clear();
  grads.reserve(parameters.size());
  for (const auto& p : parameters) {
    if (p.grad().defined()) {
      grads.push_back(p.grad());
    }
  }

  if (grads.empty()) {
    // No grads → nothing to clip; return a CPU 0-scalar so callers logging
    // total_norm get a sensible value.
    return torch::zeros({});
  }

  torch::Tensor total_norm;

  if (norm_type == std::numeric_limits<double>::infinity()) {
    // Infinity norm: max of all absolute values
    // _foreach_norm doesn't support inf norm, compute manually on GPU.
    // Each abs().max() returns a 0-dim tensor; stack+max collapses them
    // to a single global max without any host roundtrip.
    std::vector<torch::Tensor> abs_maxes;
    abs_maxes.reserve(grads.size());
    for (auto& g : grads) {
      abs_maxes.push_back(g.abs().max());
    }
    total_norm = torch::stack(abs_maxes).max();
  } else {
    // p-norm (typically 2): use _foreach_norm for batched per-tensor norms.
    auto per_param_norms = at::_foreach_norm(grads, norm_type);

    // Reduce the per-param norms to a single global p-norm. For p=2 this
    // is sqrt(sum(n_i^2)), i.e. the same value you would get from norming
    // the concatenated grad vector — but without the concat allocation.
    auto norms_stacked = torch::stack(per_param_norms);
    total_norm = norms_stacked.norm(norm_type);
  }

  // Clip coefficient: max_norm / (||g|| + 1e-6), capped at 1 so we never
  // SCALE UP a small norm. The 1e-6 floor avoids inf/nan when grads are 0.
  auto clip_coef = max_norm / (total_norm + 1e-6);
  clip_coef = torch::clamp_max(clip_coef, 1.0);

  // Apply clipping to all grads in one fused launch. _foreach_mul_ accepts
  // a 0-dim Tensor scalar; broadcasting handles the per-tensor multiply.
  at::_foreach_mul_(grads, clip_coef);

  return total_norm;
}

}  // namespace olmo_cpp
