#pragma once
/**
 * include/olmo_cpp/model/feed_forward.hpp
 *
 * Declaration of the SwiGLU FeedForward sublayer used inside every
 * transformer block. SwiGLU forward:
 *
 *     gate = W1 x        (D -> H)
 *     up   = W3 x        (D -> H, separate matrix)
 *     y    = silu(gate) ⊙ up        (elementwise gated activation)
 *     out  = W2 y        (H -> D)
 *
 * silu(z) = z · sigmoid(z). See kernels/silu_mul.cu for a full
 * pedagogical writeup.
 *
 * The "fused" path concatenates W1 and W3 into a single 2H-wide
 * weight matrix so there's a single matmul launch instead of two —
 * ~constant launch-overhead win.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/feed_forward.cpp : implementation.
 *   - src/model/block.cpp / fused_block.cpp / block_variants.cpp :
 *     instantiates a FeedForward as one of the two sublayers in
 *     every transformer block.
 *
 * --- Role in training pipeline ---
 *   Foundational. Together with attention, this is most of the FLOPs
 *   in every transformer forward pass.
 */

#include "olmo_cpp/float8/float8.hpp"
#include <torch/torch.h>
#include <memory>

namespace olmo_cpp {

/// SwiGLU feed-forward: out = w2(silu(w1(x)) * w3(x))
/// When use_fused_gate_up is true, w1 and w3 are combined into a single
/// [2*H, D] weight matrix for a single GEMM (halves launch overhead).
class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(int64_t d_model, int64_t hidden_size, bool bias = false,
                  bool use_fused_gate_up = false);

  torch::Tensor forward(torch::Tensor x);

  /// Opt into FP8 (E4M3) STE emulation around each Linear's matmul. Off
  /// by default; the block's constructor flips it on when
  /// cfg.use_float8 is true. Allocates one Float8ScaleState pair (input
  /// + weight) per Linear; the disabled path pays zero extra memory.
  void enable_float8(bool on);

 private:
  // Standard path: separate gate (w1) and up (w3)
  torch::nn::Linear w1_{nullptr};
  torch::nn::Linear w3_{nullptr};

  // Fused path: combined gate+up weight [2*H, D]
  torch::nn::Linear w_gate_up_{nullptr};

  // Down projection (always separate)
  torch::nn::Linear w2_{nullptr};

  bool fused_ = false;

  // FP8 emulation state (I-5 / T-6). One amax tracker per Linear's input
  // and weight; populated by enable_float8(true).
  bool use_float8_ = false;
  std::unique_ptr<Float8ScaleState> fp8_w1x_, fp8_w3x_, fp8_gux_, fp8_w2x_;
  std::unique_ptr<Float8ScaleState> fp8_w1w_, fp8_w3w_, fp8_guw_, fp8_w2w_;
};

TORCH_MODULE(FeedForward);

}  // namespace olmo_cpp
