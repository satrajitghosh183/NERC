#pragma once
/**
 * include/olmo_cpp/model/convolution.hpp
 *
 * Declaration for CausalConv1d — a depthwise causal 1-D convolution
 * used as a pre-attention "shift register" in some research configs
 * (Mamba-like hybrid architectures).
 *
 * "Causal" means a token only sees past tokens (we left-pad). "Depth-
 * wise" means each channel has its own filter and channels don't mix.
 * See src/model/convolution.cpp for the longer explanation.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/convolution.cpp : implementation.
 *   - src/model/block_variants.cpp : some block variants insert a
 *     CausalConv1d before attention.
 *
 * --- Role in training pipeline ---
 *   Optional. Inactive in the TinyStories quickstart flow.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// 1D causal convolution for hybrid architectures (Mamba-like)
class CausalConv1dImpl : public torch::nn::Module {
 public:
  CausalConv1dImpl(int64_t channels, int64_t kernel_size);

  /// x: [B, S, D] -> [B, S, D] (causal: only looks at past)
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Conv1d conv_;
  int64_t padding_;
};
TORCH_MODULE(CausalConv1d);

}  // namespace olmo_cpp
