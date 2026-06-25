#pragma once

/**
 * include/olmo_cpp/backend/fused_ops.hpp
 *
 * Free-function helpers that compose multiple ATen ops into a smaller
 * number of GEMMs / kernel launches:
 *   - fused_qkv_projection: 1 GEMM instead of 3 for the Q,K,V heads.
 *   - fused_attention:       reference SDPA (currently the textbook
 *                            softmax(QK^T/sqrt(d))V).
 *   - fused_swiglu_ffn:      gate+up share a single GEMM, then SiLU*mul
 *                            via the active backend, then down GEMM.
 *   - fused_residual_norm:   residual add + RMSNorm (+ optional dropout).
 *   - fused_gate_up_projection: just the gate+up GEMM-then-split half
 *                            of fused_swiglu_ffn for callers that want
 *                            it standalone.
 *
 * These are *not* IBackend methods — they are recipes that *call into*
 * the active backend for the elementwise pieces and let LibTorch handle
 * the matmuls (which are already fast cuBLAS/MKL).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/backend/backend.hpp (in .cpp): get_backend() for the
 *     elementwise SiLU*mul and rms_norm steps.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep — these are utility
 *   recipes intended for FusedTransformer / FusedBlock and tests.
 *
 * --- Role in training pipeline ---
 *   These compositions live one level above the IBackend ops. They
 *   exist so the FusedTransformer can write `auto qkv = fused_qkv...`
 *   and have a single hook to optimize, rather than three Linears.
 */

#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// Fused compound operations for transformer blocks.
/// These combine multiple ATen ops into fewer kernel launches,
/// reducing overhead from tensor allocation, dispatch, and memory traffic.

namespace fused {

/// Fused QKV projection: single GEMM for Q, K, V instead of three.
/// Input: x [B, S, D], weight_qkv [3*D_out, D], bias_qkv (optional)
/// Returns: {Q, K, V} each [B, S, D_out]
struct QKVResult {
  torch::Tensor q, k, v;
};

QKVResult fused_qkv_projection(torch::Tensor x,
                                torch::Tensor weight_qkv,
                                std::optional<torch::Tensor> bias_qkv = std::nullopt);

/// Fused attention: softmax(Q*K^T / sqrt(d)) * V in one call.
/// Uses FlashAttention-style tiling for memory efficiency when available.
/// Falls back to standard scaled dot product attention.
/// q: [B, H, S, D], k: [B, H, S, D], v: [B, H, S, D]
/// Returns: [B, H, S, D]
torch::Tensor fused_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                               std::optional<torch::Tensor> mask = std::nullopt,
                               double scale = -1.0,
                               bool causal = true);

/// Fused SwiGLU FFN: combines gate projection, up projection, SiLU activation,
/// elementwise multiply, and down projection into minimal kernel launches.
/// x: [B, S, D]
/// w_gate: [H, D], w_up: [H, D], w_down: [D, H]
/// Returns: [B, S, D]
torch::Tensor fused_swiglu_ffn(torch::Tensor x,
                                torch::Tensor w_gate,
                                torch::Tensor w_up,
                                torch::Tensor w_down,
                                std::optional<torch::Tensor> bias_gate = std::nullopt,
                                std::optional<torch::Tensor> bias_up = std::nullopt,
                                std::optional<torch::Tensor> bias_down = std::nullopt);

/// Fused residual + layer norm + dropout pipeline.
/// Combines: dropout(x) + residual, then norm(result).
/// Avoids 3 separate kernel launches and 2 extra full-tensor reads.
/// Returns: {normed, residual_out} where residual_out = dropout(x) + residual
struct ResidualNormResult {
  torch::Tensor normed;       // After norm
  torch::Tensor residual_out; // The residual stream (for next block)
};

ResidualNormResult fused_residual_norm(torch::Tensor x,
                                        torch::Tensor residual,
                                        torch::Tensor norm_weight,
                                        double eps,
                                        double dropout_p = 0.0,
                                        bool training = false);

/// Fused gate+up projection: single GEMM computing [gate; up] = x @ [W_gate; W_up]^T
/// then splits the result. Halves the GEMM launch overhead.
/// x: [B, S, D], weight_gate_up: [2*H, D]
/// Returns: {gate, up} each [B, S, H]
struct GateUpResult {
  torch::Tensor gate, up;
};

GateUpResult fused_gate_up_projection(torch::Tensor x,
                                       torch::Tensor weight_gate_up,
                                       std::optional<torch::Tensor> bias = std::nullopt);

}  // namespace fused
}  // namespace olmo_cpp
