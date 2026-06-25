#pragma once

/**
 * include/olmo_cpp/model/gated_attention.hpp
 *
 * Two small learnable gating modules that scale the attention output by a
 * sigmoid-bounded learned coefficient. They are intended as drop-in
 * post-attention rescaling — multiplying the attention output by
 * sigmoid(gate). Distinguished by parameter granularity: HeadwiseGate has
 * one scalar per head, ElementwiseGate has one scalar per (head, dim).
 *
 * --- Includes from this project ---
 *   (No project headers — only torch/torch.h.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep. The gates live next to the
 *   attention modules as opt-in components for future architecture
 *   experiments; they are not currently wired into ReorderedNormBlock or
 *   FusedTransformerBlock on the bench path.
 *
 * --- Role in training pipeline ---
 *   Off the bench path. Reusable building blocks for future gated-
 *   attention experiments where post-attention output is scaled per-head
 *   or per-element.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// Gated attention: multiplies attention output by a learned gate
/// Headwise: one scalar gate per head [n_heads]
/// Elementwise: one gate per element [n_heads, head_dim]
class HeadwiseGateImpl : public torch::nn::Module {
 public:
  /// Constructs a per-head learned gate initialized to ones (so the initial
  /// effective scaling is sigmoid(1) ~ 0.73).
  HeadwiseGateImpl(int64_t n_heads);
  /// x: [B, n_heads, S, head_dim] -> same shape
  torch::Tensor forward(torch::Tensor x);
 private:
  /// Learned gate parameter, broadcast over batch / sequence / head_dim.
  torch::Tensor gate_;  // [1, n_heads, 1, 1]
};
/// Module holder for HeadwiseGateImpl.
TORCH_MODULE(HeadwiseGate);

/// Per-(head, head_dim) learned gate. Initialized to zeros (so the initial
/// effective scaling is sigmoid(0) = 0.5).
class ElementwiseGateImpl : public torch::nn::Module {
 public:
  /// Build the gate parameter for given head count and per-head dim.
  ElementwiseGateImpl(int64_t n_heads, int64_t head_dim);
  /// Multiply input by sigmoid(gate_) elementwise; preserves shape.
  torch::Tensor forward(torch::Tensor x);
 private:
  /// Learned gate parameter, broadcast over batch / sequence.
  torch::Tensor gate_;  // [1, n_heads, 1, head_dim]
};
/// Module holder for ElementwiseGateImpl.
TORCH_MODULE(ElementwiseGate);

}  // namespace olmo_cpp
