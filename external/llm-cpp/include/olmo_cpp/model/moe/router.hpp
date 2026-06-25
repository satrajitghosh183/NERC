#pragma once

/**
 * include/olmo_cpp/model/moe/router.hpp
 *
 * Declares the top-k token router for MoE. The router is a single small
 * Linear layer (the "gate") that maps each token's hidden state to a
 * num_experts-dimensional logit vector; the top-k entries (with their
 * softmaxed weights) become the routing decision for that token.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: torch::nn::Module, Linear, TORCH_MODULE.
 *   - <tuple>: forward returns a 3-tuple of tensors.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/moe/router.cpp: implementation.
 *   - src/model/moe/moe.cpp: MoELayerImpl owns a TopKRouter as `router_` and
 *     calls it on the flattened [B*S, D] input before the experts run.
 *
 * --- Role in training pipeline ---
 *   First step inside every MoE layer's forward(). Its outputs are consumed
 *   by the expert MLP module (to know which experts to dispatch each token to)
 *   and by MoELoss::auxiliary_loss (for z-loss and load-balancing regularization).
 */

#include <torch/torch.h>
#include <tuple>

namespace olmo_cpp {

/// Top-K router. Per token, picks the K experts with the highest gate logits
/// and returns the softmaxed top-k weights. Logits over all experts are also
/// returned so that the load-balancing / z-loss can be computed downstream.
class TopKRouterImpl : public torch::nn::Module {
 public:
  /// d_model           — input hidden size (must match transformer width).
  /// num_experts       — total number of experts E in this MoE layer.
  /// top_k             — how many experts to route each token to (typ. 1 or 2).
  /// normalize_weights — if true, re-normalize the top-k softmax weights so
  ///                     they sum to 1 (a no-op mathematically, kept explicit
  ///                     to remain robust against future masking).
  TopKRouterImpl(int64_t d_model, int64_t num_experts, int64_t top_k,
                 bool normalize_weights = true);

  /// Forward returns the tuple {weights, indices, logits}:
  ///   weights: [B*S, top_k] softmax weights over the chosen experts
  ///   indices: [B*S, top_k] integer expert IDs in [0, num_experts)
  ///   logits:  [B*S, E]     raw gate logits — needed for aux losses.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);

 private:
  /// gate_ is a bias-free Linear(d_model, num_experts). This single matmul is
  /// the entire router; the rest of forward() is just topk + softmax.
  torch::nn::Linear gate_;
  int64_t num_experts_, top_k_;  ///< sizing constants captured from ctor
  bool normalize_weights_;       ///< controls the explicit re-normalization step
};

TORCH_MODULE(TopKRouter);

}  // namespace olmo_cpp
