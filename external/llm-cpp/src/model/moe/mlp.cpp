/**
 * src/model/moe/mlp.cpp
 *
 * Implements the three expert-MLP modules declared in mlp.hpp:
 *   - ExpertMLPImpl       — single SwiGLU expert (identical to a dense FFN)
 *   - MoEMLPImpl          — N experts with capacity-factor token dropping
 *   - DroplessMoEMLPImpl  — N experts with no token dropping
 *
 * The forward kernels here are reference implementations using ATen index
 * gather/scatter primitives. They are correct but not maximally efficient;
 * a fused CUDA kernel would replace the per-expert Python-style loop with
 * a single dispatched grouped-matmul. For training runs the cost is
 * dominated by the underlying GEMMs anyway.
 *
 * --- Includes from this project ---
 *   - "olmo_cpp/model/moe/mlp.hpp": class declarations.
 *   - <algorithm>: std::min for capacity clipping.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/moe/moe.cpp: MoELayerImpl owns one of these via AnyModule and
 *     dispatches to the matching forward() each step.
 *
 * --- Role in training pipeline ---
 *   Heaviest compute inside an MoE block: expert-FFN GEMMs are the dominant
 *   FLOP cost. These functions are called once per MoE layer per microbatch.
 */

#include "olmo_cpp/model/moe/mlp.hpp"

#include <algorithm>

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// ExpertMLP (SwiGLU, identical to FeedForward)
// ---------------------------------------------------------------------------

/// Construct one SwiGLU expert with three Linear layers.
/// register_module both stores the child and exposes its parameters/buffers
/// to the parent's parameters() / state_dict().
ExpertMLPImpl::ExpertMLPImpl(int64_t d_model, int64_t hidden_size, bool bias)
    : w1_(register_module(
          "w1",
          // Gate projection: d_model -> hidden. .bias(bias) is normally false.
          torch::nn::Linear(
              torch::nn::LinearOptions(d_model, hidden_size).bias(bias)))),
      w2_(register_module(
          "w2",
          // Down projection: hidden -> d_model.
          torch::nn::Linear(
              torch::nn::LinearOptions(hidden_size, d_model).bias(bias)))),
      w3_(register_module(
          "w3",
          // Up projection: d_model -> hidden. (Yes, same shape as w1_; SwiGLU
          // uses two parallel projections multiplied element-wise.)
          torch::nn::Linear(
              torch::nn::LinearOptions(d_model, hidden_size).bias(bias)))) {}

/// SwiGLU forward: y = W2 ( silu(W1 x) ⊙ (W3 x) ).
/// silu(z) = z * sigmoid(z); '⊙' denotes element-wise multiply.
torch::Tensor ExpertMLPImpl::forward(torch::Tensor x) {
  return w2_(torch::silu(w1_(x)) * w3_(x));
}

// ---------------------------------------------------------------------------
// MoEMLP (with capacity factor — Switch Transformer / GShard style)
// ---------------------------------------------------------------------------

/// Construct N experts and stash them in a ModuleList. Note that ExpertMLP is
/// itself a TORCH_MODULE shared_ptr-wrapper, so push_back stores a handle.
MoEMLPImpl::MoEMLPImpl(int64_t d_model, int64_t hidden_size,
                        int64_t num_experts, double capacity_factor, bool bias)
    : experts_(register_module("experts", torch::nn::ModuleList())),
      num_experts_(num_experts),
      capacity_factor_(capacity_factor) {
  // Loop-construct experts. Each expert has its own independent weights.
  for (int64_t i = 0; i < num_experts; ++i) {
    experts_->push_back(ExpertMLP(d_model, hidden_size, bias));
  }
}

/// Capacity-bounded MoE forward. For each expert in turn we (a) find which
/// tokens were routed to it, (b) clip to capacity, (c) run the expert FFN on
/// the gathered subset, and (d) scatter-add the weighted result into output.
/// O(num_experts) Python-style loop — this is the simple correct reference;
/// a production kernel would batch all experts with a grouped GEMM.
torch::Tensor MoEMLPImpl::forward(torch::Tensor x,
                                   torch::Tensor expert_weights,
                                   torch::Tensor expert_indices) {
  // Shape contract:
  //   x              [B*S, D]   per-token hidden states (already flattened)
  //   expert_weights [B*S, K]   softmax weights from the router
  //   expert_indices [B*S, K]   integer expert IDs in [0, num_experts)
  auto num_tokens = x.size(0);
  auto d_model = x.size(1);
  auto top_k = expert_indices.size(1);

  // Compute the per-expert capacity cap. The "ideal" load if perfectly balanced
  // is num_tokens*top_k / num_experts; capacity_factor_ scales that up to
  // absorb routing noise. Tokens beyond cap get silently dropped.
  auto capacity = static_cast<int64_t>(
      std::ceil(static_cast<double>(num_tokens * top_k) /
                static_cast<double>(num_experts_) * capacity_factor_));

  auto output = torch::zeros_like(x);  // [B*S, D] accumulator on same dev/dtype.

  for (int64_t e = 0; e < num_experts_; ++e) {
    // Boolean mask: which (token, slot) pairs picked expert e?
    auto mask = expert_indices.eq(e);  // [B*S, K]

    // nonzero() returns the [n_assigned, 2] index pairs (row=token, col=slot).
    auto positions = mask.nonzero();  // [N_assigned, 2]

    // Skip experts with no work (common in early training).
    if (positions.size(0) == 0) {
      continue;
    }

    // Capacity limit: keep only the first `capacity` pairs. This is the
    // "token dropping" step characteristic of Switch Transformers.
    auto n_assigned = positions.size(0);
    auto n_process = std::min(n_assigned, capacity);
    positions = positions.slice(/*dim=*/0, /*start=*/0, /*end=*/n_process);

    // Split the position-pair tensor into separate row/col index vectors.
    auto token_ids = positions.select(/*dim=*/1, /*index=*/0);  // [N]
    auto slot_ids = positions.select(/*dim=*/1, /*index=*/1);   // [N]

    // Gather just the rows of x that go to this expert.
    auto expert_input = x.index_select(/*dim=*/0, token_ids);  // [N, D]

    // ModuleList holds Module shared_ptrs erased to base; recover the impl.
    auto expert_output =
        experts_[e]->as<ExpertMLPImpl>()->forward(expert_input);  // [N, D]

    // Pull the matching softmax weight for each (token, slot) pair, broadcast-ready.
    auto w = expert_weights.index({token_ids, slot_ids}).unsqueeze(1);  // [N, 1]

    // Scatter-add the weighted expert output back to its row in `output`.
    // index_add_ handles duplicates correctly when top_k>1 and a token has
    // multiple slots assigned to the same expert (rare but possible).
    output.index_add_(/*dim=*/0, token_ids, expert_output * w);
  }

  return output;
}

// ---------------------------------------------------------------------------
// DroplessMoEMLP (no token dropping — full fidelity)
// ---------------------------------------------------------------------------

/// Same construction as MoEMLPImpl minus the capacity factor: no clipping
/// happens at runtime, so we don't need the multiplier here.
DroplessMoEMLPImpl::DroplessMoEMLPImpl(int64_t d_model, int64_t hidden_size,
                                       int64_t num_experts, bool bias)
    : experts_(register_module("experts", torch::nn::ModuleList())),
      num_experts_(num_experts) {
  for (int64_t i = 0; i < num_experts; ++i) {
    experts_->push_back(ExpertMLP(d_model, hidden_size, bias));
  }
}

/// Dropless forward: same gather/run/scatter pattern as the capacity variant
/// but skipping the n_process = min(n_assigned, capacity) clipping step.
/// Every routed (token, slot) pair contributes to the output.
torch::Tensor DroplessMoEMLPImpl::forward(torch::Tensor x,
                                           torch::Tensor expert_weights,
                                           torch::Tensor expert_indices) {
  // Shape contract identical to MoEMLPImpl::forward.
  auto output = torch::zeros_like(x);  // [B*S, D] zero accumulator.

  for (int64_t e = 0; e < num_experts_; ++e) {
    // Boolean mask of routing slots assigned to expert e.
    auto mask = expert_indices.eq(e);  // [B*S, K]
    auto positions = mask.nonzero();   // [N_assigned, 2]

    if (positions.size(0) == 0) {
      continue;  // expert e got no work this microbatch
    }

    // Decompose the [N, 2] index pairs into separate vectors.
    auto token_ids = positions.select(/*dim=*/1, /*index=*/0);  // [N]
    auto slot_ids = positions.select(/*dim=*/1, /*index=*/1);   // [N]

    // Gather the assigned token rows.
    auto expert_input = x.index_select(/*dim=*/0, token_ids);  // [N, D]

    // Run the expert FFN on the entire assigned chunk (no capacity clamp).
    auto expert_output =
        experts_[e]->as<ExpertMLPImpl>()->forward(expert_input);  // [N, D]

    // Per-pair router weight, reshaped for broadcasting against [N, D].
    auto w = expert_weights.index({token_ids, slot_ids}).unsqueeze(1);  // [N, 1]

    // Scatter-add weighted expert output. With top_k>1 a token can appear
    // multiple times in token_ids; index_add_ accumulates correctly.
    output.index_add_(/*dim=*/0, token_ids, expert_output * w);
  }

  return output;
}

}  // namespace olmo_cpp
