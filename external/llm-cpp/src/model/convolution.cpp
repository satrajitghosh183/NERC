/**
 * src/model/convolution.cpp
 *
 * Causal depthwise 1-D convolution. Used as a pre-attention
 * "shift register" in some research configurations to give the
 * attention layer a bit of explicit local structure.
 *
 * "Causal" means a token can only see itself and earlier tokens —
 * we left-pad by (kernel_size − 1) so the convolution doesn't peek
 * at the future. "Depthwise" means each channel is convolved with
 * its own filter and channels don't mix — this is much cheaper than
 * a full Conv1d (groups=1). The intended use is purely local context
 * mixing; the heavy cross-token mixing still happens in attention.
 *
 * Off by default; enabled with cfg.use_conv=1.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/convolution.hpp : CausalConv1d declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/block_variants.cpp: some block topologies insert a
 *     CausalConv1d before the attention sublayer.
 *
 * --- Role in training pipeline ---
 *   Optional. Off in the quickstart conf.
 */
#include "olmo_cpp/model/convolution.hpp"

namespace olmo_cpp {

CausalConv1dImpl::CausalConv1dImpl(int64_t channels, int64_t kernel_size)
    : conv_(register_module("conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(channels, channels, kernel_size)
            .groups(channels)  // depthwise
            .bias(false)))),
      padding_(kernel_size - 1) {}

torch::Tensor CausalConv1dImpl::forward(torch::Tensor x) {
  // x: [B, S, D] -> transpose to [B, D, S] for Conv1d
  auto out = x.transpose(1, 2);

  // Left-pad to make convolution causal
  out = torch::nn::functional::pad(out,
      torch::nn::functional::PadFuncOptions({padding_, 0}));

  out = conv_(out);

  // Slice to original length and transpose back
  out = out.narrow(2, 0, x.size(1));
  return out.transpose(1, 2);
}

}  // namespace olmo_cpp
