#pragma once

/**
 * include/olmo_cpp/model/moe/moe.hpp
 *
 * Declares MoELayer: the user-facing module that bundles a TopKRouter, one of
 * {MoEMLP, DroplessMoEMLP}, and the auxiliary-loss computation into a single
 * drop-in replacement for a dense FFN block. From the outside it looks like
 * an FFN — input [B, S, D], output {hidden, router_logits, aux_loss}.
 *
 * --- Includes from this project ---
 *   - "olmo_cpp/model/moe/router.hpp": TopKRouter (the gating Linear).
 *   - "olmo_cpp/model/moe/mlp.hpp":    MoEMLP / DroplessMoEMLP expert wrappers.
 *   - "olmo_cpp/model/moe/loss.hpp":   MoELoss::auxiliary_loss helper.
 *   - <torch/torch.h>:                 base nn::Module, AnyModule, TORCH_MODULE.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/moe/moe.cpp: definition of every member.
 *   - src/model/block_variants.cpp / include/olmo_cpp/model/block_variants.hpp:
 *     plumbing that swaps a dense FFN for an MoELayer in transformer blocks.
 *
 * --- Role in training pipeline ---
 *   This is the layer the transformer block actually instantiates when MoE is
 *   enabled. It is called once per block per forward pass. The aux_loss field
 *   of the returned struct is summed into the global training loss before
 *   backward, ensuring the router gets a well-conditioned gradient.
 */

#include "olmo_cpp/model/moe/router.hpp"
#include "olmo_cpp/model/moe/mlp.hpp"
#include "olmo_cpp/model/moe/loss.hpp"
#include <torch/torch.h>

namespace olmo_cpp {

/// Full MoE layer: router + expert MLPs + auxiliary loss in one module.
class MoELayerImpl : public torch::nn::Module {
 public:
  /// Construct a MoE layer.
  /// d_model         — transformer hidden size.
  /// hidden_size     — per-expert FFN inner dim.
  /// num_experts     — total experts E in this layer.
  /// top_k           — experts each token routes to (default 2).
  /// dropless        — if true, use DroplessMoEMLP; else use capacity-bounded MoEMLP.
  /// capacity_factor — only used when dropless=false (Switch default 1.25).
  /// bias            — whether expert Linears carry biases.
  MoELayerImpl(int64_t d_model, int64_t hidden_size, int64_t num_experts,
               int64_t top_k = 2, bool dropless = true,
               double capacity_factor = 1.25, bool bias = false);

  /// Bundle of tensors returned by forward(). All three live on the same device
  /// as the input.
  struct MoEOutput {
    torch::Tensor hidden_states;  ///< output activations, same shape as input
    torch::Tensor router_logits;  ///< raw gate logits [B*S, E] (exposed for ext. losses)
    torch::Tensor aux_loss;       ///< scalar zloss+lb loss already weighted by ctor args
  };

  /// Forward: x is [B, S, D]; returns {hidden, logits, aux_loss}. The two
  /// double arguments scale the aux losses (defaults match ST-MoE).
  MoEOutput forward(torch::Tensor x, double zloss_weight = 1e-3,
                    double lb_loss_weight = 1e-2);

 private:
  /// Router child module. Owned (and registered) by this layer.
  TopKRouter router_;
  /// Expert wrapper. AnyModule erases the static type so we can hold either
  /// MoEMLP or DroplessMoEMLP through a single member; the concrete type is
  /// fixed at construction by the `dropless` flag.
  torch::nn::AnyModule mlp_;
  int64_t num_experts_, top_k_;  ///< sizing copied from ctor for forward()
  bool dropless_;                ///< chooses which AnyModule branch to call
};

TORCH_MODULE(MoELayer);

}  // namespace olmo_cpp
