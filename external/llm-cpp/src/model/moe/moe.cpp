/**
 * src/model/moe/moe.cpp
 *
 * Implements MoELayerImpl: the assembly of router + experts + auxiliary loss
 * into a single transformer-block-friendly module. Most of the work is just
 * shape juggling around the router and dispatch into the AnyModule-typed
 * expert wrapper; the actual heavy lifting lives in router.cpp / mlp.cpp.
 *
 * --- Includes from this project ---
 *   - "olmo_cpp/model/moe/moe.hpp": class declaration plus transitive includes
 *     of router.hpp, mlp.hpp, loss.hpp.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/block_variants.cpp / include/olmo_cpp/model/block_variants.hpp:
 *     a transformer block uses MoELayer in place of the dense FFN whenever the
 *     config asks for MoE.
 *
 * --- Role in training pipeline ---
 *   Called once per MoE-enabled transformer block per microbatch in both
 *   forward and backward. The returned aux_loss is summed into the global
 *   training loss outside of this file.
 */

#include "olmo_cpp/model/moe/moe.hpp"

namespace olmo_cpp {

/// Construct: register the router, then construct + register either the
/// dropless or capacity-bounded expert wrapper and store it in mlp_ via AnyModule.
MoELayerImpl::MoELayerImpl(int64_t d_model, int64_t hidden_size,
                           int64_t num_experts, int64_t top_k, bool dropless,
                           double capacity_factor, bool bias)
    : router_(register_module(
          "router", TopKRouter(d_model, num_experts, top_k))),
      num_experts_(num_experts),
      top_k_(top_k),
      dropless_(dropless) {
  // The two branches construct different concrete types; AnyModule erases the
  // type so we can hold either through a single `mlp_` member.
  if (dropless) {
    // Default branch: full-fidelity dropless MoE.
    auto m = DroplessMoEMLP(d_model, hidden_size, num_experts, bias);
    // register_module ensures parameters are visible to optimizer/state_dict.
    register_module("mlp", m);
    // Move into AnyModule for type-erased forward dispatch.
    mlp_ = torch::nn::AnyModule(std::move(m));
  } else {
    // Capacity-factor branch: bounded memory at the cost of token dropping.
    auto m = MoEMLP(d_model, hidden_size, num_experts, capacity_factor, bias);
    register_module("mlp", m);
    mlp_ = torch::nn::AnyModule(std::move(m));
  }
}

/// Forward pass: route, dispatch to experts, restore shape, compute aux loss.
MoELayerImpl::MoEOutput MoELayerImpl::forward(torch::Tensor x,
                                               double zloss_weight,
                                               double lb_loss_weight) {
  // Save the input's original shape so we can put hidden_states back into it.
  // sizes() returns an IntArrayRef view; .vec() copies into an owning vector.
  auto orig_shape = x.sizes().vec();
  auto d_model = x.size(-1);

  // Flatten leading dims so router/experts see a single [N, D] matrix.
  // Works regardless of whether the caller passes [B, S, D] or [B*S, D].
  x = x.reshape({-1, d_model});

  // Router forward: per-token softmaxed top-k weights, indices, and full logits.
  auto [expert_weights, expert_indices, router_logits] = router_(x);

  // Run the chosen expert wrapper. AnyModule.get<T>() recovers the typed
  // module so we can call its .forward (the inherited Module::forward
  // signature would not type-check with these arguments).
  torch::Tensor hidden_states;
  if (dropless_) {
    hidden_states = mlp_.get<DroplessMoEMLPImpl>().forward(
        x, expert_weights, expert_indices);
  } else {
    hidden_states = mlp_.get<MoEMLPImpl>().forward(
        x, expert_weights, expert_indices);
  }

  // Restore the rank/shape the caller gave us.
  hidden_states = hidden_states.reshape(orig_shape);

  // Compute the aux losses now while logits + indices are still in scope.
  // The result is a scalar that the training loop will add to the main loss.
  auto aux_loss = MoELoss::auxiliary_loss(
      router_logits, expert_indices, num_experts_, zloss_weight, lb_loss_weight);

  // Return all three; router_logits is exposed so external code can compute
  // additional regularizers / diagnostics if desired.
  return {hidden_states, router_logits, aux_loss};
}

}  // namespace olmo_cpp
