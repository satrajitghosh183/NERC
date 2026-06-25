/**
 * kernels/fused_qkv_rope.cu
 *
 * Fused QKV projection + reshape + RoPE (item G), CUDA forward kernel.
 *
 * Launch geometry: grid = (B, S, n_q_heads + 2*n_kv_heads); block = head_dim.
 * Each block produces ONE head's row of output (q/k/v depending on its
 * head index range), with the GEMV against the relevant slab of w_qkv
 * done by the threads in that block. For Q and K heads, RoPE is applied
 * inline before the write.
 *
 * This is a "small-batch decode" optimized kernel — at B=1 and S small,
 * one block per (head, position) keeps every block doing useful work
 * even with d_model around 768. For large-batch training, a more tiled
 * matmul layout would be faster; that's a follow-on.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#include "olmo_cpp/backend/fused_qkv_rope.hpp"
#include "mma_sync.cuh"  // item 2: tensor-core building blocks for the QKV matmul

namespace olmo_cpp {

namespace {

// Half-rotation RoPE matching the model's RotaryEmbedding convention:
//   y = x * cos + rotate_half(x) * sin
//   rotate_half(x) = [-x[D/2:], x[:D/2]]
// cos/sin tables here are [S, head_dim/2] (the first half; second half
// is identical by construction). For position i in [0, head_dim/2):
//   new_first[i]  = first[i]  * cos[i] - second[i] * sin[i]
//   new_second[i] = first[i]  * sin[i] + second[i] * cos[i]

template <typename T_in, typename T_cos>
__global__ void fused_qkv_rope_kernel(
    const T_in* __restrict__ x,        // [B, S, d]
    const T_in* __restrict__ w_qkv,    // [(q+2kv)*head_dim, d]
    const T_cos* __restrict__ cos,     // [S, head_dim/2]
    const T_cos* __restrict__ sin,     // [S, head_dim/2]
    T_in* __restrict__ q_out,           // [B, n_q,  S, head_dim]
    T_in* __restrict__ k_out,           // [B, n_kv, S, head_dim]
    T_in* __restrict__ v_out,           // [B, n_kv, S, head_dim]
    int B, int S, int d,
    int n_q, int n_kv, int head_dim) {
  const int b = blockIdx.x;
  const int s = blockIdx.y;
  const int h = blockIdx.z;                                  // 0..(n_q+2*n_kv)
  const int total = n_q + 2 * n_kv;
  if (b >= B || s >= S || h >= total) return;

  // Per-pair RoPE table for this position.
  const int half = head_dim / 2;
  const T_cos* cos_s = cos + static_cast<int64_t>(s) * half;
  const T_cos* sin_s = sin + static_cast<int64_t>(s) * half;

  // Identify destination tensor + head index within that tensor.
  T_in* dst;
  int head_idx;
  bool needs_rope;
  if (h < n_q) {
    dst = q_out; head_idx = h;            needs_rope = true;  // Q
  } else if (h < n_q + n_kv) {
    dst = k_out; head_idx = h - n_q;       needs_rope = true;  // K
  } else {
    dst = v_out; head_idx = h - n_q - n_kv; needs_rope = false; // V
  }

  // Source row of x: [B, S, d] -> offset (b*S + s)*d.
  const T_in* x_row = x + (static_cast<int64_t>(b) * S + s) * d;

  // Source weights: rows [h*head_dim .. (h+1)*head_dim) of w_qkv.
  const T_in* w_head = w_qkv + static_cast<int64_t>(h) * head_dim * d;

  // Compute the head_dim outputs by GEMV: out[i] = dot(x_row, w_head[i,:]).
  // Each thread handles one element of the head output.
  extern __shared__ float smem_out[];   // [head_dim]
  const int i = threadIdx.x;
  if (i < head_dim) {
    float acc = 0.0f;
    const T_in* w_row = w_head + static_cast<int64_t>(i) * d;
    if constexpr (std::is_same<T_in, __nv_bfloat16>::value) {
      for (int j = 0; j < d; ++j) {
        acc += __bfloat162float(x_row[j]) * __bfloat162float(w_row[j]);
      }
    } else {
      for (int j = 0; j < d; ++j) {
        acc += static_cast<float>(x_row[j]) * static_cast<float>(w_row[j]);
      }
    }
    smem_out[i] = acc;
  }
  __syncthreads();

  // Half-rotation RoPE on the head's smem_out for Q/K heads. Each thread
  // i in [0, head_dim/2) operates on (smem_out[i], smem_out[i + head_dim/2]).
  if (needs_rope && i < half) {
    float c, s;
    if constexpr (std::is_same<T_cos, __nv_bfloat16>::value) {
      c = __bfloat162float(cos_s[i]);
      s = __bfloat162float(sin_s[i]);
    } else {
      c = cos_s[i];
      s = sin_s[i];
    }
    const float a = smem_out[i];
    const float b = smem_out[i + half];
    smem_out[i]        = a * c - b * s;
    smem_out[i + half] = a * s + b * c;
  }
  __syncthreads();

  // Write into the destination tensor in head-major layout
  // [B, n_heads_dst, S, head_dim].
  if (i < head_dim) {
    const int n_heads_dst = (dst == q_out) ? n_q : n_kv;
    const int64_t off =
        (((static_cast<int64_t>(b) * n_heads_dst + head_idx) * S) + s) * head_dim
        + i;
    if constexpr (std::is_same<T_in, __nv_bfloat16>::value) {
      dst[off] = __float2bfloat16(smem_out[i]);
    } else {
      dst[off] = static_cast<T_in>(smem_out[i]);
    }
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkv_rope_cuda(torch::Tensor x,
                    torch::Tensor w_qkv,
                    torch::Tensor cos,
                    torch::Tensor sin,
                    int64_t n_q_heads,
                    int64_t n_kv_heads,
                    int64_t head_dim) {
  TORCH_CHECK(x.is_cuda() && w_qkv.is_cuda() && cos.is_cuda() && sin.is_cuda(),
              "fused_qkv_rope_cuda: all tensors must be CUDA");
  c10::cuda::CUDAGuard guard(x.device());
  auto x_c   = x.contiguous();
  auto w_c   = w_qkv.contiguous();
  auto cos_c = cos.contiguous();
  auto sin_c = sin.contiguous();

  const int B = static_cast<int>(x_c.size(0));
  const int S = static_cast<int>(x_c.size(1));
  const int d = static_cast<int>(x_c.size(2));
  const int n_q  = static_cast<int>(n_q_heads);
  const int n_kv = static_cast<int>(n_kv_heads);
  const int hd   = static_cast<int>(head_dim);
  const int total_heads = n_q + 2 * n_kv;

  auto opts = x_c.options();
  auto q = torch::empty({B, n_q,  S, hd}, opts);
  auto k = torch::empty({B, n_kv, S, hd}, opts);
  auto v = torch::empty({B, n_kv, S, hd}, opts);

  dim3 grid(B, S, total_heads);
  const int threads = hd;
  const size_t shmem = hd * sizeof(float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x_c.scalar_type() == torch::kBFloat16) {
    TORCH_CHECK(cos_c.scalar_type() == torch::kBFloat16,
                "fused_qkv_rope_cuda: cos/sin must be bf16 when x is bf16");
    fused_qkv_rope_kernel<__nv_bfloat16, __nv_bfloat16><<<grid, threads, shmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        B, S, d, n_q, n_kv, hd);
  } else {
    fused_qkv_rope_kernel<float, float><<<grid, threads, shmem, stream>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        cos_c.to(torch::kFloat32).contiguous().data_ptr<float>(),
        sin_c.to(torch::kFloat32).contiguous().data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        B, S, d, n_q, n_kv, hd);
  }
  return {q, k, v};
}

}  // namespace olmo_cpp
