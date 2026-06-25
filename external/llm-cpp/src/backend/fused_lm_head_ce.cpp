/**
 * src/backend/fused_lm_head_ce.cpp
 *
 * CPU reference + host dispatcher for fused LM-head + softmax-CE (A3).
 *
 * The CPU path materializes logits via torch::nn::functional::linear and
 * delegates to torch::nn::functional::cross_entropy — slow but a
 * numerics-correct baseline the CUDA kernel can be validated against.
 *
 * Both paths return a SCALAR tensor on the same device as `h`. The
 * autograd-aware wrapper lives in fused_lm_head_ce_autograd.cpp.
 */

#include "olmo_cpp/backend/fused_lm_head_ce.hpp"

#include <torch/torch.h>

#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
#  include <cuda_runtime.h>
#endif

namespace olmo_cpp {

torch::Tensor fused_lm_head_ce_cpu(torch::Tensor h,
                                     torch::Tensor weight,
                                     torch::Tensor labels,
                                     int64_t ignore_index) {
  TORCH_CHECK(h.dim() == 2, "fused_lm_head_ce_cpu: h must be [N, d]");
  TORCH_CHECK(weight.dim() == 2, "fused_lm_head_ce_cpu: weight must be [V, d]");
  TORCH_CHECK(labels.dim() == 1, "fused_lm_head_ce_cpu: labels must be [N]");
  TORCH_CHECK(labels.size(0) == h.size(0),
              "fused_lm_head_ce_cpu: labels and h disagree on N");

  // Reference: full materialization. Numerics-stable via PyTorch's
  // cross_entropy which itself uses log_softmax + nll_loss.
  auto logits = torch::nn::functional::linear(h, weight);  // [N, V]
  return torch::nn::functional::cross_entropy(
      logits, labels,
      torch::nn::functional::CrossEntropyFuncOptions()
          .ignore_index(ignore_index)
          .reduction(torch::kMean));
}

torch::Tensor fused_lm_head_ce(torch::Tensor h,
                                 torch::Tensor weight,
                                 torch::Tensor labels,
                                 int64_t ignore_index) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (h.is_cuda() && weight.is_cuda() && labels.is_cuda()) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, h.device().index());
    if (props.major >= 12) {
      return fused_lm_head_ce_cpu(h, weight, labels, ignore_index);
    }
    return fused_lm_head_ce_cuda(h, weight, labels, ignore_index);
  }
#endif
  return fused_lm_head_ce_cpu(h, weight, labels, ignore_index);
}

}  // namespace olmo_cpp
