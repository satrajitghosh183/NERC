/**
 * src/model/feed_forward.cpp
 *
 * Implements the SwiGLU feed-forward sublayer used inside every transformer
 * block. SwiGLU is the activation pattern popularized by PaLM/LLaMA: an
 * elementwise SiLU-gated multiplication of two parallel projections, followed
 * by a down-projection back to model dimension. This file supports both the
 * "split" form (separate w1 gate / w3 up matrices) and the "fused" form
 * (a single 2H wide weight that we slice into gate and up halves) — the
 * latter halves the number of CUDA GEMM launches and improves throughput on
 * H100/A100 where launch overhead dominates for small batches.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/feed_forward.hpp: FeedForwardImpl declaration this file
 *     implements (constructor, forward, weight members).
 *   - olmo_cpp/backend/backend.hpp: provides get_backend().silu_mul(), the
 *     fused SiLU(gate)*up kernel that has CPU/SIMD/CUDA implementations.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/block.cpp: instantiates FeedForward(d_model, hidden_size,
 *     bias=false) inside TransformerBlock; called via residual path
 *     feed_forward_norm_->forward_add(feed_forward_(h), h).
 *   - src/model/fused_block.cpp: instantiates FeedForward with
 *     use_fused_gate_up=true for the FusedTransformer variant.
 *   - src/model/block_variants.cpp: used by alternative block topologies
 *     (post-norm, ReZero, parallel attention+FFN, hybrid).
 *
 * --- Role in training pipeline ---
 *   This is the second sublayer of every transformer block (after attention).
 *   On every forward pass per token: project D->2H, gate via SiLU, multiply,
 *   project 2H->H? No - gate (H) * up (H) elementwise, then w2 maps H->D.
 *   It is by far the most parameter-heavy and FLOP-heavy module per layer,
 *   so the fused gate+up path matters for end-to-end training throughput.
 */
#include "olmo_cpp/model/feed_forward.hpp"
#include "olmo_cpp/backend/backend.hpp"
#include "olmo_cpp/backend/fused_ffn.hpp"
#include "olmo_cpp/backend/cublas_direct.hpp"

namespace olmo_cpp {

/// Construct a SwiGLU FFN.
/// d_model: input/output feature dimension (the residual stream width).
/// hidden_size: inner FFN width (typically ~8/3 * d_model for SwiGLU so that
/// total parameter count matches a 4*d_model GeLU FFN).
/// bias: whether Linear layers carry a bias term (OLMo: false).
/// use_fused_gate_up: when true, allocate one [D, 2H] weight instead of two
/// [D, H] weights so a single GEMM produces the concatenated [gate||up].
FeedForwardImpl::FeedForwardImpl(int64_t d_model, int64_t hidden_size, bool bias,
                                  bool use_fused_gate_up)
    : fused_(use_fused_gate_up) {
  if (fused_) {
    // Fused path: one Linear of out_features = 2 * hidden_size. We will slice
    // its output along the last dim into the gate and up tensors at runtime.
    w_gate_up_ = register_module("w_gate_up",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, 2 * hidden_size).bias(bias)));
  } else {
    // Split path: w1 produces the gate stream, w3 produces the up stream.
    // Naming follows LLaMA's convention (w1=gate, w2=down, w3=up).
    w1_ = register_module("w1",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, hidden_size).bias(bias)));
    w3_ = register_module("w3",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, hidden_size).bias(bias)));
  }
  // Down-projection w2: maps the hidden width back to d_model. Always kept
  // separate (no benefit from fusing since it has different shape direction).
  w2_ = register_module("w2",
      torch::nn::Linear(torch::nn::LinearOptions(hidden_size, d_model).bias(bias)));
}

/// Forward pass.
/// Input  x : [B, S, D] (or arbitrary leading shape) of activations.
/// Output    : [B, S, D] (same leading shape) — the FFN contribution.
/// Math: y = w2( silu(gate) * up ).
torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
  // Hot path: when fused_gate_up is on, no FP8 STE, and the inputs are CUDA,
  // call the fused FFN macro kernel (item I). One launch instead of three.
  // CPU and FP8-active paths fall through to the explicit-op variants below.
  if (fused_ && !use_float8_ && x.is_cuda()) {
    return fused_ffn_autograd(x, w_gate_up_->weight, w2_->weight);
  }

  // FP8 STE / non-fused fallback: route each Linear through float8_linear_emulated
  // (when FP8 is on) or fast_linear (item L: direct cuBLASLt bypass of the
  // ATen dispatcher on the hot path).
  auto lin = [&](torch::nn::Linear& m,
                 Float8ScaleState* sx, Float8ScaleState* sw,
                 const torch::Tensor& in) -> torch::Tensor {
    if (use_float8_) {
      return float8_linear_emulated(in, m->weight,
                                    m->bias.defined() ? m->bias : torch::Tensor(),
                                    *sx, *sw);
    }
    return fast_linear(in, m->weight,
                       m->bias.defined() ? m->bias : torch::Tensor());
  };

  if (fused_) {
    auto gate_up = lin(w_gate_up_, fp8_gux_.get(), fp8_guw_.get(), x);
    int64_t h = gate_up.size(-1) / 2;
    auto gate = gate_up.narrow(-1, 0, h);
    auto up = gate_up.narrow(-1, h, h);
    auto act = get_backend().silu_mul(gate, up);
    return lin(w2_, fp8_w2x_.get(), fp8_w2w_.get(), act);
  }
  auto act = get_backend().silu_mul(
      lin(w1_, fp8_w1x_.get(), fp8_w1w_.get(), x),
      lin(w3_, fp8_w3x_.get(), fp8_w3w_.get(), x));
  return lin(w2_, fp8_w2x_.get(), fp8_w2w_.get(), act);
}

void FeedForwardImpl::enable_float8(bool on) {
  use_float8_ = on;
  if (!on) {
    fp8_w1x_.reset(); fp8_w3x_.reset(); fp8_gux_.reset(); fp8_w2x_.reset();
    fp8_w1w_.reset(); fp8_w3w_.reset(); fp8_guw_.reset(); fp8_w2w_.reset();
    return;
  }
  auto make = [] { return std::make_unique<Float8ScaleState>(16); };
  fp8_w2x_ = make(); fp8_w2w_ = make();
  if (fused_) { fp8_gux_ = make(); fp8_guw_ = make(); }
  else        { fp8_w1x_ = make(); fp8_w1w_ = make();
                fp8_w3x_ = make(); fp8_w3w_ = make(); }
}

}  // namespace olmo_cpp
