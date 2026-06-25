#pragma once

/**
 * include/olmo_cpp/model/moe/loss.hpp
 *
 * Declares the auxiliary-loss helpers used by Mixture-of-Experts (MoE) layers.
 * In an MoE layer a small "router" decides which experts each token is sent to.
 * Without regularization the router collapses (all tokens go to one expert);
 * to prevent that, we add two scalar losses to the main cross-entropy loss:
 *
 *   1) router z-loss   — penalizes large logsumexp values so the softmax over
 *                        experts stays numerically well-behaved (Switch-Transformer-style).
 *   2) load-balancing loss — encourages uniform expert utilization across the batch.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: torch::Tensor for batched logit/index inputs.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/moe/loss.cpp: implementation of every method declared here.
 *   - src/model/moe/moe.cpp: MoELayerImpl::forward calls MoELoss::auxiliary_loss
 *     once per layer per microbatch and returns it in MoEOutput::aux_loss.
 *
 * --- Role in training pipeline ---
 *   The training loop sums per-layer aux_loss tensors into the global loss
 *   before .backward(); the gradient flows back through the router's gating
 *   linear layer, shaping the router's behavior over training steps. Without
 *   these losses MoE training reliably diverges or collapses.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// Static helper struct grouping the MoE auxiliary losses. No state — every
/// member is a free function in disguise; the struct is just a namespace.
struct MoELoss {
  /// Router z-loss (Switch Transformer, ST-MoE).
  /// Computed as mean(logsumexp(logits, dim=-1)^2). Discourages router logits
  /// from drifting to extreme magnitudes, which would saturate the top-k softmax
  /// and make routing decisions effectively non-differentiable.
  static torch::Tensor z_loss(const torch::Tensor& router_logits);

  /// Load-balancing loss (Shazeer et al., GShard).
  /// loss = num_experts * sum_i (f_i * p_i)
  ///   f_i = fraction of routed slots assigned to expert i (hard, from top-k)
  ///   p_i = mean softmax probability of expert i across tokens (soft)
  /// f_i is non-differentiable (uses arg-top-k), but p_i is — gradient flows
  /// through p_i and pushes the router toward a uniform allocation.
  static torch::Tensor load_balancing_loss(
      const torch::Tensor& router_logits,
      const torch::Tensor& expert_indices,
      int64_t num_experts);

  /// Convenience: returns zloss_weight * z_loss + lb_loss_weight * load_balancing_loss.
  /// Defaults follow ST-MoE: 1e-3 for z-loss, 1e-2 for load balancing.
  static torch::Tensor auxiliary_loss(
      const torch::Tensor& router_logits,
      const torch::Tensor& expert_indices,
      int64_t num_experts,
      double zloss_weight = 1e-3,
      double lb_loss_weight = 1e-2);
};

}  // namespace olmo_cpp
