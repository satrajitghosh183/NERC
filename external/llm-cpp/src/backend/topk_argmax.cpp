/**
 * src/backend/topk_argmax.cpp
 *
 * CPU reference + host dispatcher for the fused GEMV+argmax (item K).
 */

#include "olmo_cpp/backend/topk_argmax.hpp"

#include <torch/torch.h>

namespace olmo_cpp {

torch::Tensor lmhead_argmax_cpu(torch::Tensor hidden, torch::Tensor weight) {
  TORCH_CHECK(hidden.is_cpu() && weight.is_cpu(),
              "lmhead_argmax_cpu: tensors must be CPU");
  TORCH_CHECK(hidden.dim() == 3 && weight.dim() == 2,
              "shapes: hidden [B,S,d], weight [V,d]");
  auto h = hidden.to(torch::kFloat32).contiguous();
  auto w = weight.to(torch::kFloat32).contiguous();
  auto logits = torch::matmul(h, w.t());          // [B, S, V]
  return logits.argmax(/*dim=*/-1);                // [B, S] int64
}

torch::Tensor lmhead_argmax(torch::Tensor hidden, torch::Tensor weight) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (hidden.is_cuda()) {
    return lmhead_argmax_cuda(hidden, weight);
  }
#endif
  return lmhead_argmax_cpu(hidden, weight);
}

}  // namespace olmo_cpp
