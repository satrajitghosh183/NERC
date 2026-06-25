/**
 * src/backend/lm_head_gemv.cpp
 *
 * CPU reference + dispatch for the LM-head GEMV (fast-inference [11]).
 */

#include "olmo_cpp/backend/lm_head_gemv.hpp"

#include <torch/torch.h>

namespace olmo_cpp {

torch::Tensor lm_head_gemv_cpu(torch::Tensor hidden, torch::Tensor W_U) {
  // CPU fallback is just a matmul. cuBLAS-equivalent on host.
  auto h = hidden.contiguous().to(torch::kFloat32);
  auto W = W_U.contiguous().to(torch::kFloat32);
  return torch::matmul(W, h);  // [V, H] @ [H] = [V]
}

torch::Tensor lm_head_gemv(torch::Tensor hidden, torch::Tensor W_U) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (hidden.is_cuda()) {
    return lm_head_gemv_cuda(hidden, W_U);
  }
#endif
  return lm_head_gemv_cpu(hidden, W_U);
}

}  // namespace olmo_cpp
