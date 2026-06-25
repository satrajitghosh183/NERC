/**
 * src/model/moe/router.cpp
 *
 * Implements the top-k MoE router declared in router.hpp. This is the
 * smallest learnable component of an MoE layer — a single bias-free Linear —
 * but it makes every routing decision in the model.
 *
 * --- Includes from this project ---
 *   - "olmo_cpp/model/moe/router.hpp": TopKRouterImpl declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/moe/moe.cpp: MoELayerImpl::router_ is a TopKRouter and is
 *     called inside MoELayerImpl::forward before dispatching to the expert
 *     wrapper.
 *
 * --- Role in training pipeline ---
 *   Runs before the experts in every MoE block. Its outputs feed both the
 *   expert wrapper (for actual computation) and MoELoss::auxiliary_loss (for
 *   load-balancing/z-loss regularization).
 */

#include "olmo_cpp/model/moe/router.hpp"

namespace olmo_cpp {

/// Construct the gate Linear and capture sizing/normalization flags.
/// .bias(false) follows Switch/Llama convention — the gate has no bias term.
TopKRouterImpl::TopKRouterImpl(int64_t d_model, int64_t num_experts,
                               int64_t top_k, bool normalize_weights)
    : gate_(register_module(
          "gate",
          torch::nn::Linear(
              torch::nn::LinearOptions(d_model, num_experts).bias(false)))),
      num_experts_(num_experts),
      top_k_(top_k),
      normalize_weights_(normalize_weights) {}

/// Forward: returns {weights, indices, logits}. See header for shapes.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
TopKRouterImpl::forward(torch::Tensor x) {
  // Accept either [B, S, D] or [B*S, D]: flatten to a single row-per-token matrix.
  auto d_model = x.size(-1);
  x = x.reshape({-1, d_model});

  // The full router GEMM. Output is one logit per (token, expert) pair.
  auto logits = gate_(x);  // [B*S, num_experts]

  // Pick the K largest logits per row. top_values/top_indices are both [B*S, K].
  // (LibTorch returns a std::tuple; structured bindings unpack.)
  auto [top_values, top_indices] = logits.topk(top_k_, /*dim=*/-1);

  // Softmax over the selected K logits — these become the routing weights
  // applied to each expert's contribution downstream.
  auto weights = torch::softmax(top_values, /*dim=*/-1);

  // After softmax over a finite vector the sum is mathematically 1 already;
  // this re-normalization is a safety net for future variants that might
  // mask out experts (e.g., expert-dropout, expert-parallel zero shards).
  if (normalize_weights_) {
    weights = weights / weights.sum(/*dim=*/-1, /*keepdim=*/true);
  }

  // Returning the *full* logits (not just top-k values) — load-balancing
  // and z-loss both need probabilities/logsumexp over all experts.
  return {weights, top_indices, logits};
}

}  // namespace olmo_cpp
