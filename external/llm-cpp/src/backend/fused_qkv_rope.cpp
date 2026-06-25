/**
 * src/backend/fused_qkv_rope.cpp
 *
 * CPU reference + host dispatcher for fused QKV+RoPE (item G).
 *
 * The CPU path lays out the work as ATen sequential ops (3 splits,
 * 3 reshapes, RoPE apply) — slow but a numerics-correct baseline the
 * CUDA kernel can be validated against bitwise.
 */

#include "olmo_cpp/backend/fused_qkv_rope.hpp"

#include <torch/torch.h>

#if defined(USE_CUDA) || defined(OLMO_HAS_CUDA_KERNELS)
#  include <cuda_runtime.h>
#endif

namespace olmo_cpp {

namespace {

// Half-rotation RoPE matching the model's RotaryEmbedding (LLaMA / OLMo
// convention): rotate_half = [-second_half; first_half]. cos/sin tables
// have shape [S, head_dim/2] (the first half; full-dim form repeats).
//   new_first  = first  * cos - second * sin
//   new_second = first  * sin + second * cos
torch::Tensor apply_rope_ref(torch::Tensor t,    // [B, n_heads, S, head_dim]
                              torch::Tensor cos, // [S, head_dim/2]
                              torch::Tensor sin) {
  const int64_t head_dim = t.size(3);
  const int64_t half = head_dim / 2;
  auto first  = t.narrow(-1, 0,    half);   // [B, H, S, D/2]
  auto second = t.narrow(-1, half, half);
  auto cos_b = cos.view({1, 1, cos.size(0), cos.size(1)});
  auto sin_b = sin.view({1, 1, sin.size(0), sin.size(1)});
  auto y_first  = first * cos_b - second * sin_b;
  auto y_second = first * sin_b + second * cos_b;
  return torch::cat({y_first, y_second}, /*dim=*/-1);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_cpu(torch::Tensor x,
                   torch::Tensor w_qkv,
                   torch::Tensor cos,
                   torch::Tensor sin,
                   int64_t n_q_heads,
                   int64_t n_kv_heads,
                   int64_t head_dim) {
  TORCH_CHECK(x.dim() == 3, "x must be [B, S, d]");
  TORCH_CHECK(w_qkv.dim() == 2, "w_qkv must be 2D");
  const int64_t B = x.size(0);
  const int64_t S = x.size(1);
  const int64_t q_dim  = n_q_heads * head_dim;
  const int64_t kv_dim = n_kv_heads * head_dim;
  TORCH_CHECK(w_qkv.size(0) == q_dim + 2 * kv_dim,
              "w_qkv row count must equal (n_q + 2*n_kv) * head_dim");

  auto qkv = torch::nn::functional::linear(x, w_qkv);       // [B, S, q+2kv]
  auto q = qkv.narrow(-1, 0, q_dim);                         // [B, S, q]
  auto k = qkv.narrow(-1, q_dim, kv_dim);                    // [B, S, kv]
  auto v = qkv.narrow(-1, q_dim + kv_dim, kv_dim);

  q = q.view({B, S, n_q_heads,  head_dim}).transpose(1, 2).contiguous();
  k = k.view({B, S, n_kv_heads, head_dim}).transpose(1, 2).contiguous();
  v = v.view({B, S, n_kv_heads, head_dim}).transpose(1, 2).contiguous();

  q = apply_rope_ref(q, cos, sin);
  k = apply_rope_ref(k, cos, sin);
  return {q, k, v};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope(torch::Tensor x,
               torch::Tensor w_qkv,
               torch::Tensor cos,
               torch::Tensor sin,
               int64_t n_q_heads,
               int64_t n_kv_heads,
               int64_t head_dim) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (x.is_cuda()) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, x.device().index());
    // CUDA fused kernels not yet parity-validated on Blackwell; ATen path.
    if (props.major >= 12) {
      return fused_qkv_rope_cpu(x, w_qkv, cos, sin,
                                 n_q_heads, n_kv_heads, head_dim);
    }
    // Prefer the WMMA tensor-core variant on bf16 + aligned shapes +
    // when the [16, F] tile fits in shared memory. F = (n_q+2*n_kv)*hd.
    // Limit: 96 KB (sm_80 conservative); Blackwell sm_120 has 228 KB
    // but we keep the gate conservative so it works across archs.
    if (x.scalar_type() == torch::kBFloat16 &&
        w_qkv.scalar_type() == torch::kBFloat16 &&
        cos.scalar_type()   == torch::kBFloat16 &&
        sin.scalar_type()   == torch::kBFloat16) {
      const int64_t B = x.size(0);
      const int64_t S = x.size(1);
      const int64_t d = x.size(2);
      const int64_t N = B * S;
      const int64_t F = (n_q_heads + 2 * n_kv_heads) * head_dim;
      // Actual kernel shmem = [16, F] bf16 output tile + per-warp fp32
      // scratch (4 warps × 16×16). Compare against the device's opt-in
      // max (the launcher raises the cap via cudaFuncSetAttribute).
      const size_t shmem_bytes = (size_t)16 * F * sizeof(uint16_t)
                               + (size_t)4 * 16 * 16 * sizeof(float);
      int max_optin = 0;
      cudaDeviceGetAttribute(&max_optin,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             x.device().index());
      const bool aligned = (N % 16 == 0) && (d % 16 == 0) && (F % 16 == 0);
      const bool shmem_ok = (max_optin > 0) && (shmem_bytes <= (size_t)max_optin);
      const bool wmma_ok = props.major < 12;
      if (aligned && shmem_ok && wmma_ok) {
        return fused_qkv_rope_wmma_cuda(x, w_qkv, cos, sin,
                                          n_q_heads, n_kv_heads, head_dim);
      }
    }
    return fused_qkv_rope_cuda(x, w_qkv, cos, sin,
                                n_q_heads, n_kv_heads, head_dim);
  }
#endif
  return fused_qkv_rope_cpu(x, w_qkv, cos, sin,
                             n_q_heads, n_kv_heads, head_dim);
}

}  // namespace olmo_cpp
