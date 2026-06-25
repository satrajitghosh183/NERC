/**
 * src/backend/flash_decode.cpp
 *
 * CPU reference + dispatch for FlashAttention-2 decode (fast-inference [12]).
 */

#include "olmo_cpp/backend/flash_decode.hpp"

#include <torch/torch.h>

namespace olmo_cpp {

torch::Tensor flash_decode_cpu(torch::Tensor q, torch::Tensor k, torch::Tensor v, float sm_scale) {
  // Reference attention (no online tricks; just for correctness).
  auto q_c = q.contiguous().to(torch::kFloat32);
  auto k_c = k.contiguous().to(torch::kFloat32);
  auto v_c = v.contiguous().to(torch::kFloat32);

  const int64_t n_q_heads  = q_c.size(0);
  const int64_t head_dim   = q_c.size(1);
  const int64_t n_kv_heads = k_c.size(1);

  if (n_kv_heads != n_q_heads) {
    int64_t group = n_q_heads / n_kv_heads;
    k_c = k_c.repeat_interleave(group, /*dim=*/1);
    v_c = v_c.repeat_interleave(group, /*dim=*/1);
  }

  // q [Hq, D] → [Hq, 1, D]; k [T, Hq, D] → [Hq, T, D]
  auto q_b = q_c.unsqueeze(1);
  auto k_b = k_c.transpose(0, 1);
  auto v_b = v_c.transpose(0, 1);

  auto scores = torch::matmul(q_b, k_b.transpose(-1, -2)) * sm_scale;
  auto attn   = torch::softmax(scores, -1);
  auto out    = torch::matmul(attn, v_b).squeeze(1);
  return out.to(q.dtype());
}

torch::Tensor flash_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, float sm_scale) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (q.is_cuda()) return flash_decode_cuda(q, k, v, sm_scale);
#endif
  return flash_decode_cpu(q, k, v, sm_scale);
}

}  // namespace olmo_cpp
