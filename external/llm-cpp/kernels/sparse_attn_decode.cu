/**
 * kernels/sparse_attn_decode.cu
 *
 * Fused CUDA kernel for NSA-style content-selected sparse attention,
 * single-query decode. Replaces the ATen reference (sparse_attn.cpp) on
 * CUDA: one kernel launch instead of 4 (block-mean → topk → gather → SDPA).
 *
 * Algorithm (matches CPU reference in src/backend/sparse_attn.cpp):
 *   1. Group-average q across the GQA group (each (B, kv_head) gets one
 *      averaged q used for selection across all its query heads).
 *   2. Score each KV block: dot(q_grouped, mean(K_block)).
 *   3. Maintain a top-k list online (no global score buffer needed even
 *      for n_blocks = 16K at T=1M, block_size=64).
 *   4. Online-softmax SDPA over the gathered top-k * block_size positions
 *      using the per-head q (NOT the group-averaged q — the SDPA itself
 *      is per-head).
 *
 * Grid: (B * H,) blocks. Each block handles one (batch, query_head) pair.
 * Threads: kThreads. Shared mem layout (FP32):
 *
 *   sh_q_self      [D]            this head's q (for SDPA)
 *   sh_q_grp       [D]            group-averaged q (for selection scores)
 *   sh_topk_score  [top_k]        top-k scores, ascending — slot 0 is the
 *                                 worst kept score (insertion threshold)
 *   sh_topk_idx    [top_k]        block indices for the top-k slots
 *   sh_accum       [D]            online-softmax accumulator
 *   plus three scalars (m, l, partial)
 *
 * Numerical: q/k/v cast to FP32 on read; output cast back to q.dtype()
 * after the inv_l divide.
 *
 * Shape contract:
 *   q [B, H,   1, D]  k,v [B, Hkv, T, D]  out [B, H, 1, D]
 *   T must be a multiple of block_size (caller pads).
 *   H must be divisible by Hkv (GQA contract).
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <math_constants.h>
#include <cstdint>
#include <algorithm>

#include "olmo_cpp/backend/sparse_attn.hpp"
#include "cuda_reduce.cuh"

namespace olmo_cpp {

namespace {

constexpr int kThreads = 128;
constexpr int kMaxWarps = (kThreads + 31) >> 5;

__global__ void sparse_attn_decode_kernel(
    const float* __restrict__ q,    // [B, H, 1, D] flattened
    const float* __restrict__ k,    // [B, Hkv, T, D] flattened
    const float* __restrict__ v,    // [B, Hkv, T, D] flattened
    int B, int H, int Hkv, int T, int D,
    int block_size, int top_k,
    float sm_scale,
    float* __restrict__ out         // [B, H, 1, D] flattened
) {
  const int bh    = blockIdx.x;
  if (bh >= B * H) return;
  const int b     = bh / H;
  const int qh    = bh % H;
  const int group = H / Hkv;
  const int kvh   = qh / group;            // GQA mapping
  const int group_start_qh = kvh * group;  // first query head sharing this kvh
  const int tid   = threadIdx.x;
  const int n_blocks = T / block_size;

  extern __shared__ float smem[];
  float* sh_q_self     = smem;                                  // [D]
  float* sh_q_grp      = sh_q_self + D;                          // [D]
  float* sh_topk_score = sh_q_grp + D;                           // [top_k]
  int*   sh_topk_idx   = reinterpret_cast<int*>(sh_topk_score + top_k); // [top_k]
  float* sh_accum      = reinterpret_cast<float*>(sh_topk_idx + top_k); // [D]
  // Warp scratch for block_reduce_sum. One float per warp.
  float* sh_warpbuf    = sh_accum + D;                           // [kMaxWarps]
  __shared__ float sh_m;
  __shared__ float sh_l;

  // ── Phase 0: load q for self and the group-average for selection. ──
  // Self: just q[b, qh, 0, :].
  const float* q_self_ptr = q + (b * H + qh) * D;
  for (int i = tid; i < D; i += blockDim.x) {
    sh_q_self[i] = q_self_ptr[i];
    // Group-average q across the GQA siblings (group consecutive heads).
    float acc = 0.0f;
    for (int g = 0; g < group; ++g) {
      acc += q[(b * H + group_start_qh + g) * D + i];
    }
    sh_q_grp[i] = acc / static_cast<float>(group);
  }

  // Init top-k slots to (-inf, -1).
  for (int i = tid; i < top_k; i += blockDim.x) {
    sh_topk_score[i] = -CUDART_INF_F;
    sh_topk_idx[i]   = -1;
  }
  __syncthreads();

  const float inv_block_size = 1.0f / static_cast<float>(block_size);

  // ── Phase 1: score blocks + online top-k maintenance. ──
  // The scoring formula: score(blk) = mean over t∈blk of (q_grp · K[t]).
  // Equivalent to dot(q_grp, mean(K[blk])). We compute the per-block sum
  // of dot products via thread-cooperative reduction over D, then sum across
  // tokens, then divide by block_size.
  for (int blk = 0; blk < n_blocks; ++blk) {
    const float* k_blk = k + ((b * Hkv + kvh) * T + blk * block_size) * D;
    float local = 0.0f;
    // Thread `tid` walks every (t, d) where d % blockDim.x == tid.
    for (int t = 0; t < block_size; ++t) {
      for (int d = tid; d < D; d += blockDim.x) {
        local += sh_q_grp[d] * k_blk[t * D + d];
      }
    }
    // Warp-tree reduction; replaces the atomicAdd-on-shared pattern.
    const float score = block_reduce_sum(local, tid, blockDim.x, sh_warpbuf)
                          * inv_block_size;

    // Thread 0 maintains the top-k list (sorted ascending; slot 0 is worst).
    if (tid == 0 && score > sh_topk_score[0]) {
      // Find insertion position: largest pos in [0, top_k) with
      // score > sh_topk_score[pos+1]. Bubble the smaller scores left.
      int pos = 0;
      while (pos + 1 < top_k && score > sh_topk_score[pos + 1]) ++pos;
      for (int j = 0; j < pos; ++j) {
        sh_topk_score[j] = sh_topk_score[j + 1];
        sh_topk_idx[j]   = sh_topk_idx[j + 1];
      }
      sh_topk_score[pos] = score;
      sh_topk_idx[pos]   = blk;
    }
    __syncthreads();
  }

  // ── Phase 2: online-softmax SDPA over selected blocks. ──
  if (tid == 0) {
    sh_m = -CUDART_INF_F;
    sh_l = 0.0f;
  }
  for (int i = tid; i < D; i += blockDim.x) sh_accum[i] = 0.0f;
  __syncthreads();

  for (int s = 0; s < top_k; ++s) {
    const int blk = sh_topk_idx[s];
    if (blk < 0) continue;  // no valid block (top_k > n_blocks edge case)
    const int t_start = blk * block_size;
    const float* k_base = k + ((b * Hkv + kvh) * T + t_start) * D;
    const float* v_base = v + ((b * Hkv + kvh) * T + t_start) * D;

    for (int t = 0; t < block_size; ++t) {
      // Score = sm_scale * dot(q_self, k[t]).
      float local = 0.0f;
      for (int d = tid; d < D; d += blockDim.x) {
        local += sh_q_self[d] * k_base[t * D + d];
      }
      const float score =
          block_reduce_sum(local, tid, blockDim.x, sh_warpbuf) * sm_scale;

      // Online-softmax update.
      const float m_new    = fmaxf(sh_m, score);
      const float exp_diff = __expf(sh_m - m_new);
      const float w        = __expf(score - m_new);
      for (int d = tid; d < D; d += blockDim.x) {
        sh_accum[d] = sh_accum[d] * exp_diff + v_base[t * D + d] * w;
      }
      if (tid == 0) {
        sh_l = sh_l * exp_diff + w;
        sh_m = m_new;
      }
      __syncthreads();
    }
  }

  // ── Phase 3: divide by l, write out. ──
  const float inv_l = (sh_l > 0.0f) ? (1.0f / sh_l) : 0.0f;
  float* out_ptr = out + (b * H + qh) * D;
  for (int d = tid; d < D; d += blockDim.x) {
    out_ptr[d] = sh_accum[d] * inv_l;
  }
}

}  // namespace

torch::Tensor sparse_attn_decode_cuda(const torch::Tensor& q,
                                      const torch::Tensor& k,
                                      const torch::Tensor& v,
                                      float sm_scale,
                                      int64_t block_size,
                                      int64_t top_k) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "sparse_attn_decode_cuda: all tensors must be CUDA");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "expect q [B,H,1,D] and k/v [B,Hkv,T,D]");

  c10::cuda::CUDAGuard guard(q.device());

  const auto orig_dtype = q.dtype();

  // Promote to FP32 for the kernel; the kernel does FP32 accumulation.
  // Caller can pass bf16 / fp16 — we cast to fp32 here.
  auto q_c = q.contiguous().to(torch::kFloat32);
  auto k_c = k.contiguous().to(torch::kFloat32);
  auto v_c = v.contiguous().to(torch::kFloat32);

  const int64_t B   = q_c.size(0);
  const int64_t H   = q_c.size(1);
  const int64_t Sq  = q_c.size(2);
  const int64_t D   = q_c.size(3);
  const int64_t Hkv = k_c.size(1);
  const int64_t T   = k_c.size(2);

  TORCH_CHECK(Sq == 1, "decode kernel expects q seq-len 1, got ", Sq);
  TORCH_CHECK(H % Hkv == 0, "n_heads must be divisible by n_kv_heads");
  TORCH_CHECK(T % block_size == 0,
              "T (", T, ") must be divisible by block_size (", block_size,
              "); caller is responsible for padding");
  TORCH_CHECK(top_k > 0 && block_size > 0, "top_k and block_size must be positive");

  const int64_t n_blocks = T / block_size;
  const int64_t k_top    = std::min<int64_t>(top_k, n_blocks);

  auto out = torch::empty({B, H, Sq, D}, q_c.options());

  // Shared memory layout (in floats / ints, packed):
  //   sh_q_self[D] + sh_q_grp[D] + sh_topk_score[k_top] + sh_topk_idx[k_top]
  //   + sh_accum[D] + sh_warpbuf[kMaxWarps]
  // Plus sh_m, sh_l are __shared__ statically.
  size_t shmem_bytes =
      static_cast<size_t>(2 * D + 2 * k_top + D + kMaxWarps) * sizeof(float);

  const dim3 grid(static_cast<unsigned>(B * H));
  const dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  sparse_attn_decode_kernel<<<grid, block, shmem_bytes, stream>>>(
      q_c.data_ptr<float>(),
      k_c.data_ptr<float>(),
      v_c.data_ptr<float>(),
      static_cast<int>(B), static_cast<int>(H), static_cast<int>(Hkv),
      static_cast<int>(T), static_cast<int>(D),
      static_cast<int>(block_size), static_cast<int>(k_top),
      sm_scale,
      out.data_ptr<float>());

  return out.to(orig_dtype);
}

}  // namespace olmo_cpp
