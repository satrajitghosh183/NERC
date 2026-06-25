/**
 * src/model/gated_attention.cpp
 *
 * ─── What "gated attention" is ──────────────────────────────────────
 *
 * Gated attention adds a learnable gate to the attention output. After
 * computing the usual attn(Q, K, V), we multiply the result by a
 * sigmoid-of-something to scale how much of the attention output is
 * blended into the residual stream. Two flavours are offered:
 *
 *   - HeadwiseGate: one scalar gate per attention head — coarse but
 *     few extra parameters. The gate is initialised to 1.0 so
 *     sigmoid(1) ≈ 0.73 — slightly biased to "let the head through"
 *     so training starts close to ungated behaviour.
 *
 *   - ElementwiseGate: per-element gate of shape [head_dim] —
 *     finer-grained, more parameters.
 *
 * Both are off by default (gated_attention=none in the .conf) and
 * exist for ablation experiments / paper benchmarks rather than
 * production training.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/gated_attention.hpp : module declarations.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/attention.cpp / fused_attention.cpp: when
 *     cfg.gated_attention != None, the attention module wraps its
 *     output with the chosen gate before the output projection.
 *
 * --- Role in training pipeline ---
 *   Optional. Off in the quickstart conf.
 */
#include "olmo_cpp/model/gated_attention.hpp"

namespace olmo_cpp {

// --- HeadwiseGate ---

HeadwiseGateImpl::HeadwiseGateImpl(int64_t n_heads) {
  // Initialize to ones so sigmoid(gate_) starts near sigmoid(1) ~ 0.73
  gate_ = register_parameter("gate",
      torch::ones({1, n_heads, 1, 1}));
}

torch::Tensor HeadwiseGateImpl::forward(torch::Tensor x) {
  return torch::sigmoid(gate_) * x;
}

// --- ElementwiseGate ---

ElementwiseGateImpl::ElementwiseGateImpl(int64_t n_heads, int64_t head_dim) {
  // Initialize to zeros so sigmoid(gate_) starts at 0.5
  gate_ = register_parameter("gate",
      torch::zeros({1, n_heads, 1, head_dim}));
}

torch::Tensor ElementwiseGateImpl::forward(torch::Tensor x) {
  return torch::sigmoid(gate_) * x;
}

}  // namespace olmo_cpp
