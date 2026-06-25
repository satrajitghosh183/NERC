#pragma once

/**
 * include/olmo_cpp/model/moe/mlp.hpp
 *
 * Declares the expert feed-forward modules used by an MoE layer: a single
 * SwiGLU expert plus two flavors of the multi-expert wrapper (with capacity
 * limit / "dropping", and the "dropless" version where every token is processed).
 * One expert here is functionally identical to a regular dense FFN block in a
 * transformer — the MoE wrapper just owns N of them and dispatches tokens.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: torch::nn::Module, Linear, ModuleList, TORCH_MODULE.
 *   - <vector>: included for transitive use by callers (no direct vector here).
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/moe/mlp.cpp: implementations of every Impl class.
 *   - include/olmo_cpp/model/moe/moe.hpp + src/model/moe/moe.cpp:
 *       MoELayerImpl owns one of {MoEMLP, DroplessMoEMLP} via torch::nn::AnyModule.
 *
 * --- Role in training pipeline ---
 *   These modules sit downstream of the router. Given the router's per-token
 *   top-k expert assignments, they gather/scatter tokens to the right experts,
 *   run the SwiGLU FFN, and combine the per-expert outputs with router weights.
 *   The capacity-factor variant trades correctness for predictable memory; the
 *   dropless variant is what we use by default for full-fidelity training.
 */

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// Single SwiGLU expert. Identical algebra to the dense FFN: w2(silu(w1(x)) * w3(x)).
/// w1 is the gate projection, w3 is the up projection, w2 is the down projection.
/// `bias=false` matches Llama / OLMo-2 style FFNs.
class ExpertMLPImpl : public torch::nn::Module {
 public:
  /// d_model    — model hidden size (input/output of the expert)
  /// hidden_size — expert FFN inner dimension (typically 4*d_model or 8/3*d_model)
  /// bias       — whether the three linear layers carry bias terms
  ExpertMLPImpl(int64_t d_model, int64_t hidden_size, bool bias = false);
  /// Forward: x is any shape [..., d_model], output is the same shape.
  torch::Tensor forward(torch::Tensor x);

 private:
  /// w1_: gate proj (d_model -> hidden), w2_: down proj (hidden -> d_model),
  /// w3_: up proj (d_model -> hidden). Names match the Llama/OLMo HF layout.
  torch::nn::Linear w1_, w2_, w3_;
};

/// TORCH_MODULE expands to the shared_ptr-wrapper class `ExpertMLP` so callers
/// can write `ExpertMLP e(...)` instead of `std::make_shared<ExpertMLPImpl>(...)`.
TORCH_MODULE(ExpertMLP);

/// MoE MLP with capacity factor: each expert may process at most
/// ceil(num_tokens * top_k / num_experts * capacity_factor) tokens; tokens
/// beyond that quota are dropped (their slot contributes zero to the output).
/// Used for memory-bounded training; matches GShard/Switch behavior.
class MoEMLPImpl : public torch::nn::Module {
 public:
  /// capacity_factor=1.25 is the typical Switch-Transformer setting.
  MoEMLPImpl(int64_t d_model, int64_t hidden_size, int64_t num_experts,
             double capacity_factor = 1.25, bool bias = false);

  /// Forward inputs are flattened to per-token rows by the caller (MoELayer):
  ///   x:               [B*S, D]
  ///   expert_weights:  [B*S, K]   softmax weights over the K chosen experts
  ///   expert_indices:  [B*S, K]   integer indices in [0, num_experts)
  torch::Tensor forward(torch::Tensor x, torch::Tensor expert_weights,
                        torch::Tensor expert_indices);

 private:
  /// ModuleList holding `num_experts` ExpertMLP children. Indexed at runtime.
  torch::nn::ModuleList experts_;
  int64_t num_experts_;       ///< total number of experts E
  double capacity_factor_;    ///< multiplier applied to the ideal even split
};

TORCH_MODULE(MoEMLP);

/// Dropless MoE: no per-expert capacity limit. Every (token, slot) assignment
/// is processed. Higher fidelity but worst-case memory grows with imbalance.
/// This is the default for our training runs.
class DroplessMoEMLPImpl : public torch::nn::Module {
 public:
  DroplessMoEMLPImpl(int64_t d_model, int64_t hidden_size, int64_t num_experts,
                     bool bias = false);

  /// Same input contract as MoEMLPImpl::forward; difference is purely runtime
  /// (no `n_process = min(n_assigned, capacity)` clipping).
  torch::Tensor forward(torch::Tensor x, torch::Tensor expert_weights,
                        torch::Tensor expert_indices);

 private:
  torch::nn::ModuleList experts_;  ///< `num_experts` ExpertMLP children
  int64_t num_experts_;            ///< total number of experts E
};

TORCH_MODULE(DroplessMoEMLP);

}  // namespace olmo_cpp
