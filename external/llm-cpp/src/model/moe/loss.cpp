/**
 * src/model/moe/loss.cpp
 *
 * Implements the static helpers declared in olmo_cpp/model/moe/loss.hpp:
 * router z-loss, load-balancing loss, and a weighted sum of the two. These
 * are tiny tensor-program kernels — every line is a single ATen op.
 *
 * --- Includes from this project ---
 *   - "olmo_cpp/model/moe/loss.hpp": declarations of MoELoss::z_loss,
 *     load_balancing_loss, auxiliary_loss.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/moe/moe.cpp: MoELayerImpl::forward calls auxiliary_loss once
 *     per layer per microbatch and stores the result in MoEOutput::aux_loss.
 *
 * --- Role in training pipeline ---
 *   These auxiliary losses are summed into the global training loss before
 *   backward(). They shape the router toward stable, balanced expert usage;
 *   without them, routing collapses (all tokens hit a single expert).
 */

#include "olmo_cpp/model/moe/loss.hpp"

namespace olmo_cpp {

/// Z-loss: penalize the squared logsumexp of the router logits.
/// Keeps gate magnitudes small so the top-k softmax stays numerically sane.
torch::Tensor MoELoss::z_loss(const torch::Tensor& router_logits) {
  // router_logits: [B*S, num_experts] — raw gate logits, one row per token.
  // logsumexp(dim=-1) is the log-partition function of the softmax over
  // experts; squaring it and averaging over tokens gives a scalar penalty.
  auto lse = torch::logsumexp(router_logits, /*dim=*/-1);  // [B*S]
  // Element-wise square then mean -> scalar. Differentiable through `lse`.
  return (lse * lse).mean();
}

/// Load-balancing loss: encourages uniform expert utilization.
/// The classic formula from Shazeer/GShard/Switch:
///   loss = E * sum_i (f_i * p_i)
/// where f_i is the (hard, non-differentiable) fraction of routing slots
/// assigned to expert i, and p_i is the (soft, differentiable) mean
/// probability of expert i across the batch. The product (f * p) is what
/// actually carries gradient — gradient pushes p toward uniform.
torch::Tensor MoELoss::load_balancing_loss(
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_indices,
    int64_t num_experts) {
  // Shapes contract:
  //   router_logits  [B*S, num_experts]   raw logits
  //   expert_indices [B*S, top_k]         which experts each token chose
  auto num_tokens = router_logits.size(0);
  auto top_k = expert_indices.size(1);

  // ---- f_i: fraction of routing slots that landed on expert i ----
  // Treat each (token, k) pair as one routing slot; total slots = B*S*top_k.
  auto flat_indices = expert_indices.reshape({-1});  // [B*S * top_k]
  // One-hot encode and cast to the logits' dtype so subsequent arithmetic
  // matches the autograd graph (avoids implicit promotions on bf16/fp16).
  auto one_hot = torch::one_hot(flat_indices, num_experts).to(router_logits.dtype());
  // Sum across rows: how many slots picked each expert.
  auto expert_counts = one_hot.sum(/*dim=*/0);  // [num_experts]
  // Normalize by total slots -> empirical routing fractions.
  auto f = expert_counts / static_cast<double>(num_tokens * top_k);  // [num_experts]

  // ---- p_i: mean softmax probability per expert across the batch ----
  // This is the *differentiable* part of the loss: gradients flow through
  // softmax back into router_logits and into the gate Linear's weights.
  auto probs = torch::softmax(router_logits, /*dim=*/-1);  // [B*S, num_experts]
  auto p = probs.mean(/*dim=*/0);  // [num_experts]

  // Final scalar: E * <f, p>. The factor of E makes the optimum 1 (perfectly
  // balanced) instead of 1/E.
  return static_cast<double>(num_experts) * (f * p).sum();
}

/// Convenience: weighted sum of the two regularizers. Returns a scalar tensor
/// suitable for adding directly to the cross-entropy loss before backward().
torch::Tensor MoELoss::auxiliary_loss(
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_indices,
    int64_t num_experts,
    double zloss_weight,
    double lb_loss_weight) {
  // Compute each piece independently — they share inputs but not intermediates.
  auto zl = z_loss(router_logits);
  auto lbl = load_balancing_loss(router_logits, expert_indices, num_experts);
  // Linear combination keeps autograd happy and mirrors ST-MoE conventions.
  return zloss_weight * zl + lb_loss_weight * lbl;
}

}  // namespace olmo_cpp
