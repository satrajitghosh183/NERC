/**
 * kernels/paged_attention.cu
 *
 * Paged single-query attention kernel — fast-inference [9].
 *
 * DRAFT. Single-query (decode) only. One block per query head; threads
 * within the block walk the page table, compute Q·K for each cached
 * token, online softmax, accumulate weighted V.
 *
 * Algorithm: standard "FlashDecoding" style online softmax over the
 * gathered KV. Per-query reduction across cached tokens.
 *
 * NOT YET WIRED into the Transformer attention forward. That refactor
 * lives elsewhere.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <math_constants.h>
#include <cstdint>
#include <cmath>

#include "olmo_cpp/backend/paged_attention.hpp"
#include "cuda_reduce.cuh"

namespace olmo_cpp {

namespace {

constexpr int kThreads = 128;
constexpr int kMaxWarps = (kThreads + 31) >> 5;

// One block per (q_head, batch=1). Each block streams over n_tokens
// cached positions, gathering K/V from pages via the page table, and
// online-softmaxes the result. Output one row of [n_q_heads, head_dim].
//
// The internal kernel body is shared between the static-n_tokens launcher
// and the graph-capture-friendly launcher; the only difference is whether
// n_tokens is a kernel arg or read from device memory at kernel entry.
__device__ __forceinline__ void paged_attention_decode_body(
    const float* __restrict__ q,           // [n_q_heads, head_dim]
    const float* __restrict__ k_pool,      // [max_pages, page_size, n_kv_heads, head_dim]
    const float* __restrict__ v_pool,      // same
    const int32_t* __restrict__ page_table,// [n_blocks]
    int64_t n_tokens,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int page_size,
    int max_pages,
    float sm_scale,
    float* __restrict__ out                // [n_q_heads, head_dim]
) {
  int q_head = blockIdx.x;
  if (q_head >= n_q_heads) return;

  // GQA: map q_head → kv_head.
  int kv_head = (n_kv_heads == n_q_heads)
                    ? q_head
                    : (q_head * n_kv_heads / n_q_heads);

  extern __shared__ float shmem[];
  float* sh_q       = shmem;                         // [head_dim]
  float* sh_accum   = shmem + head_dim;              // [head_dim]
  // Warp scratch for the per-token Q·K reduction below.
  float* sh_warpbuf = sh_accum + head_dim;           // [kMaxWarps]

  // Load Q row into shared memory.
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    sh_q[i] = q[q_head * head_dim + i];
    sh_accum[i] = 0.0f;
  }
  __syncthreads();

  // Online softmax stats.
  float m = -CUDART_INF_F;   // running max
  float l = 0.0f;            // running sum

  // Walk every cached token sequentially. Each thread block owns one
  // q_head, so contention across blocks is on KV-pool reads only.
  for (int64_t t = 0; t < n_tokens; ++t) {
    int64_t block_idx = t / page_size;
    int64_t in_block  = t % page_size;
    int32_t pg = page_table[block_idx];
    if (pg < 0 || pg >= max_pages) continue;

    // Compute attention score: dot(Q, K[pg, in_block, kv_head]).
    const float* k_row = k_pool +
        (((static_cast<int64_t>(pg) * page_size) + in_block) * n_kv_heads + kv_head) * head_dim;

    // Vectorize K read via float4 (head_dim is always a multiple of 4 in
    // practice — 64, 128, 256). Halves the load instruction count and
    // doubles HBM bandwidth utilization.
    float score = 0.0f;
    if ((head_dim & 3) == 0) {
      const auto* k4 = reinterpret_cast<const float4*>(k_row);
      const auto* q4 = reinterpret_cast<const float4*>(sh_q);
      const int hd4 = head_dim >> 2;
      for (int i = threadIdx.x; i < hd4; i += blockDim.x) {
        float4 kv = k4[i];
        float4 qv = q4[i];
        score += kv.x * qv.x + kv.y * qv.y + kv.z * qv.z + kv.w * qv.w;
      }
    } else {
      for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        score += sh_q[i] * k_row[i];
      }
    }
    // Warp-tree reduction.
    score = block_reduce_sum(score, threadIdx.x, blockDim.x, sh_warpbuf);
    float s = score * sm_scale;

    // Online softmax: re-scale running accumulators by exp(m_old - m_new).
    float m_new = fmaxf(m, s);
    float exp_diff = __expf(m - m_new);
    float w        = __expf(s - m_new);

    const float* v_row = v_pool +
        (((static_cast<int64_t>(pg) * page_size) + in_block) * n_kv_heads + kv_head) * head_dim;

    // Vectorized V update: pair-load (sh_accum, v_row) via float4. sh_accum
    // is only written/read by the same lane each iteration, so no cross-
    // thread visibility is needed between iterations — that's why the
    // trailing __syncthreads() previously here is now elided. block_reduce_sum
    // already ends with a sync before the next iteration starts.
    if ((head_dim & 3) == 0) {
      auto* a4 = reinterpret_cast<float4*>(sh_accum);
      const auto* v4 = reinterpret_cast<const float4*>(v_row);
      const int hd4 = head_dim >> 2;
      for (int i = threadIdx.x; i < hd4; i += blockDim.x) {
        float4 acc = a4[i];
        float4 vv  = v4[i];
        acc.x = acc.x * exp_diff + vv.x * w;
        acc.y = acc.y * exp_diff + vv.y * w;
        acc.z = acc.z * exp_diff + vv.z * w;
        acc.w = acc.w * exp_diff + vv.w * w;
        a4[i] = acc;
      }
    } else {
      for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sh_accum[i] = sh_accum[i] * exp_diff + v_row[i] * w;
      }
    }
    l = l * exp_diff + w;
    m = m_new;
  }

  // Final normalization: divide by total weight.
  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    out[q_head * head_dim + i] = sh_accum[i] * inv_l;
  }
}

// Static-n_tokens launcher (legacy API; n_tokens captured at launch time).
__global__ void paged_attention_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    const int32_t* __restrict__ page_table,
    int64_t n_tokens,
    int n_q_heads, int n_kv_heads, int head_dim,
    int page_size, int max_pages, float sm_scale,
    float* __restrict__ out
) {
  paged_attention_decode_body(q, k_pool, v_pool, page_table,
                              n_tokens, n_q_heads, n_kv_heads, head_dim,
                              page_size, max_pages, sm_scale, out);
}

// Graph-capture-friendly launcher: reads n_tokens from device memory at
// kernel entry. The captured graph's launch parameters reference the
// device pointer, not the value, so each replay sees whatever the caller
// wrote into `*n_tokens_ptr` between replays — no re-capture needed as
// the cache grows.
__global__ void paged_attention_decode_kernel_dyn(
    const float* __restrict__ q,
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ n_tokens_ptr,
    int n_q_heads, int n_kv_heads, int head_dim,
    int page_size, int max_pages, float sm_scale,
    float* __restrict__ out
) {
  // Single read per block; all threads see the same value.
  __shared__ int n_tokens_shared;
  if (threadIdx.x == 0) {
    n_tokens_shared = *n_tokens_ptr;
  }
  __syncthreads();
  paged_attention_decode_body(q, k_pool, v_pool, page_table,
                              static_cast<int64_t>(n_tokens_shared),
                              n_q_heads, n_kv_heads, head_dim,
                              page_size, max_pages, sm_scale, out);
}

}  // namespace

torch::Tensor paged_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    int64_t n_tokens,
    float sm_scale) {
  TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && page_table.is_cuda(),
              "paged_attention_decode_cuda: all tensors must be CUDA");
  TORCH_CHECK(q.dim() == 2, "q must be [n_q_heads, head_dim]");
  TORCH_CHECK(k_pool.dim() == 4, "k_pool must be [max_pages, page_size, n_kv_heads, head_dim]");
  TORCH_CHECK(v_pool.dim() == 4, "v_pool must be [max_pages, page_size, n_kv_heads, head_dim]");

  c10::cuda::CUDAGuard guard(q.device());

  auto q_c = q.contiguous().to(torch::kFloat32);
  auto k_c = k_pool.contiguous().to(torch::kFloat32);
  auto v_c = v_pool.contiguous().to(torch::kFloat32);
  auto pt  = page_table.contiguous().to(torch::kInt32);

  const int n_q_heads = static_cast<int>(q_c.size(0));
  const int head_dim  = static_cast<int>(q_c.size(1));
  const int max_pages = static_cast<int>(k_c.size(0));
  const int page_size = static_cast<int>(k_c.size(1));
  const int n_kv_heads = static_cast<int>(k_c.size(2));

  auto out = torch::empty({n_q_heads, head_dim}, q_c.options());
  // Layout: q[D] + accum[D] + warp scratch[kMaxWarps].
  const size_t shmem = static_cast<size_t>(2 * head_dim + kMaxWarps) * sizeof(float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  paged_attention_decode_kernel<<<n_q_heads, kThreads, shmem, stream>>>(
      q_c.data_ptr<float>(),
      k_c.data_ptr<float>(),
      v_c.data_ptr<float>(),
      pt.data_ptr<int32_t>(),
      n_tokens, n_q_heads, n_kv_heads, head_dim,
      page_size, max_pages, sm_scale,
      out.data_ptr<float>());

  return out.to(q.dtype());
}

// ── Graph-safe paged K/V write kernel ─────────────────────────────────────
//
// One CUDA thread block per (i, head_block) pair. Each thread copies one
// element of the per-head [head_dim] slice from k_src/v_src to the right
// k_pool/v_pool slot. `start` is computed from a device-side read of
// n_tokens at kernel entry, so the captured launch keeps writing to the
// correct slot as the cache grows on replay.
__global__ void paged_kv_write_kernel_dyn(
    const float* __restrict__ k_src,         // [S, n_kv_heads, head_dim]
    const float* __restrict__ v_src,
    float* __restrict__ k_pool,              // [max_pages, page_size, n_kv_heads, head_dim]
    float* __restrict__ v_pool,
    const int32_t* __restrict__ page_table,  // [max_pages]
    const int32_t* __restrict__ n_tokens_ptr,
    int S,
    int n_kv_heads,
    int head_dim,
    int page_size,
    int max_pages
) {
  const int i      = blockIdx.x;        // source row in [0, S)
  const int kv_h   = blockIdx.y;        // kv head in [0, n_kv_heads)
  if (i >= S || kv_h >= n_kv_heads) return;

  // start = n_tokens - S. All threads in this block read the same scalar.
  __shared__ int sh_start;
  if (threadIdx.x == 0) {
    sh_start = (*n_tokens_ptr) - S;
  }
  __syncthreads();

  const int global_pos = sh_start + i;
  const int blk        = global_pos / page_size;
  const int slot       = global_pos % page_size;
  const int pg         = page_table[blk];
  // Defensive: skip if pg out of range. Should never trigger.
  if (pg < 0 || pg >= max_pages) return;

  const int64_t src_base =
      (static_cast<int64_t>(i) * n_kv_heads + kv_h) * head_dim;
  const int64_t dst_base =
      (((static_cast<int64_t>(pg) * page_size) + slot) * n_kv_heads + kv_h) * head_dim;

  // Each thread copies head_dim/blockDim.x consecutive elements. Vectorize
  // through float4 when head_dim is a multiple of 4 (it always is in
  // practice: 64, 128, 256).
  if ((head_dim & 3) == 0) {
    const auto* k4_src = reinterpret_cast<const float4*>(k_src + src_base);
    const auto* v4_src = reinterpret_cast<const float4*>(v_src + src_base);
    auto*       k4_dst = reinterpret_cast<float4*>(k_pool + dst_base);
    auto*       v4_dst = reinterpret_cast<float4*>(v_pool + dst_base);
    const int hd4 = head_dim >> 2;
    for (int j = threadIdx.x; j < hd4; j += blockDim.x) {
      k4_dst[j] = k4_src[j];
      v4_dst[j] = v4_src[j];
    }
  } else {
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
      k_pool[dst_base + j] = k_src[src_base + j];
      v_pool[dst_base + j] = v_src[src_base + j];
    }
  }
}

void paged_kv_write_dyn_cuda(
    torch::Tensor k_src,
    torch::Tensor v_src,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens) {
  TORCH_CHECK(k_src.is_cuda() && v_src.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda()
              && page_table.is_cuda() && n_tokens.is_cuda(),
              "paged_kv_write_dyn_cuda: all tensors must be on the same CUDA device");
  TORCH_CHECK(k_src.dim() == 3 && v_src.dim() == 3,
              "k_src / v_src must be [S, n_kv_heads, head_dim]");
  TORCH_CHECK(k_pool.dim() == 4 && v_pool.dim() == 4,
              "k_pool / v_pool must be [max_pages, page_size, n_kv_heads, head_dim]");
  TORCH_CHECK(n_tokens.scalar_type() == torch::kInt32 && n_tokens.numel() == 1,
              "n_tokens must be int32 scalar");

  c10::cuda::CUDAGuard guard(k_pool.device());

  auto ks = k_src.contiguous().to(torch::kFloat32);
  auto vs = v_src.contiguous().to(torch::kFloat32);
  // The pools may be fp16/bf16 in the future; for now require float32 in
  // the dyn write path to keep the kernel single-dtype. AttentionImpl casts
  // on input as needed.
  TORCH_CHECK(k_pool.scalar_type() == torch::kFloat32 &&
              v_pool.scalar_type() == torch::kFloat32,
              "paged_kv_write_dyn_cuda: pools must be float32 in this build");
  auto pt = page_table.contiguous().to(torch::kInt32);
  auto nt = n_tokens.contiguous();

  const int S          = static_cast<int>(ks.size(0));
  const int n_kv_heads = static_cast<int>(ks.size(1));
  const int head_dim   = static_cast<int>(ks.size(2));
  const int max_pages  = static_cast<int>(k_pool.size(0));
  const int page_size  = static_cast<int>(k_pool.size(1));

  if (S == 0) return;

  dim3 grid(S, n_kv_heads);
  const int kThreads = 32;  // small per-block work; head_dim is the inner loop bound

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  paged_kv_write_kernel_dyn<<<grid, kThreads, 0, stream>>>(
      ks.data_ptr<float>(),
      vs.data_ptr<float>(),
      k_pool.data_ptr<float>(),
      v_pool.data_ptr<float>(),
      pt.data_ptr<int32_t>(),
      nt.data_ptr<int32_t>(),
      S, n_kv_heads, head_dim, page_size, max_pages);
}

// Graph-capture-friendly launcher. Mirrors paged_attention_decode_cuda
// exactly except for the dynamic n_tokens read.
torch::Tensor paged_attention_decode_dyn_cuda(
    torch::Tensor q,
    torch::Tensor k_pool,
    torch::Tensor v_pool,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale) {
  TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && page_table.is_cuda(),
              "paged_attention_decode_dyn_cuda: all tensors must be CUDA");
  TORCH_CHECK(n_tokens.is_cuda(), "n_tokens must live on the same CUDA device as q");
  TORCH_CHECK(n_tokens.scalar_type() == torch::kInt32, "n_tokens must be int32");
  TORCH_CHECK(n_tokens.numel() == 1, "n_tokens must be a scalar");
  TORCH_CHECK(q.dim() == 2, "q must be [n_q_heads, head_dim]");
  TORCH_CHECK(k_pool.dim() == 4, "k_pool must be [max_pages, page_size, n_kv_heads, head_dim]");
  TORCH_CHECK(v_pool.dim() == 4, "v_pool must be [max_pages, page_size, n_kv_heads, head_dim]");

  c10::cuda::CUDAGuard guard(q.device());

  auto q_c  = q.contiguous().to(torch::kFloat32);
  auto k_c  = k_pool.contiguous().to(torch::kFloat32);
  auto v_c  = v_pool.contiguous().to(torch::kFloat32);
  auto pt   = page_table.contiguous().to(torch::kInt32);
  auto nt   = n_tokens.contiguous();

  const int n_q_heads  = static_cast<int>(q_c.size(0));
  const int head_dim   = static_cast<int>(q_c.size(1));
  const int max_pages  = static_cast<int>(k_c.size(0));
  const int page_size  = static_cast<int>(k_c.size(1));
  const int n_kv_heads = static_cast<int>(k_c.size(2));

  auto out = torch::empty({n_q_heads, head_dim}, q_c.options());
  const size_t shmem = static_cast<size_t>(2 * head_dim + kMaxWarps) * sizeof(float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  paged_attention_decode_kernel_dyn<<<n_q_heads, kThreads, shmem, stream>>>(
      q_c.data_ptr<float>(),
      k_c.data_ptr<float>(),
      v_c.data_ptr<float>(),
      pt.data_ptr<int32_t>(),
      nt.data_ptr<int32_t>(),
      n_q_heads, n_kv_heads, head_dim,
      page_size, max_pages, sm_scale,
      out.data_ptr<float>());

  return out.to(q.dtype());
}

// ── INT4-KV paged decode kernel (item U) ──────────────────────────────────
//
// Same online-softmax pattern as paged_attention_decode_kernel_dyn but each
// K/V load dequantizes an int4 nibble × per-vector fp16 scale on the fly.
// Saves 4× HBM read bandwidth on the cache vs bf16.
__global__ void paged_attention_decode_int4_kernel(
    const float* __restrict__ q,           // [n_q_heads, head_dim]
    const uint8_t* __restrict__ k_pool,    // [max_pages, page_size, n_kv_heads, head_dim/2]
    const __nv_bfloat16* __restrict__ k_scales, // [max_pages, page_size, n_kv_heads]
    const uint8_t* __restrict__ v_pool,
    const __nv_bfloat16* __restrict__ v_scales,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ n_tokens_ptr,
    int n_q_heads, int n_kv_heads, int head_dim,
    int page_size, int max_pages, float sm_scale,
    float* __restrict__ out) {
  const int q_head = blockIdx.x;
  if (q_head >= n_q_heads) return;
  const int kv_head = (n_kv_heads == n_q_heads)
                      ? q_head : (q_head * n_kv_heads / n_q_heads);

  extern __shared__ float shmem[];
  float* sh_q     = shmem;
  float* sh_accum = shmem + head_dim;

  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    sh_q[i] = q[q_head * head_dim + i];
    sh_accum[i] = 0.0f;
  }
  __syncthreads();

  __shared__ int n_tokens_sh;
  if (threadIdx.x == 0) n_tokens_sh = *n_tokens_ptr;
  __syncthreads();
  const int n_tokens = n_tokens_sh;

  float m = -CUDART_INF_F;
  float l = 0.0f;

  for (int t = 0; t < n_tokens; ++t) {
    const int blk  = t / page_size;
    const int off  = t % page_size;
    const int pg   = page_table[blk];
    if (pg < 0 || pg >= max_pages) continue;

    const int64_t kv_byte_stride = (((int64_t)pg * page_size + off) * n_kv_heads + kv_head)
                                    * (head_dim / 2);
    const int64_t kv_scale_idx   = ((int64_t)pg * page_size + off) * n_kv_heads + kv_head;
    const uint8_t* k_row = k_pool + kv_byte_stride;
    const uint8_t* v_row = v_pool + kv_byte_stride;
    const float k_scale = __bfloat162float(k_scales[kv_scale_idx]);
    const float v_scale = __bfloat162float(v_scales[kv_scale_idx]);

    // Compute Q·K with inline dequant of K nibbles.
    float score = 0.0f;
    for (int j = threadIdx.x; j < head_dim / 2; j += blockDim.x) {
      const uint8_t byte = k_row[j];
      const int lo = (int)(byte & 0xF) - 8;
      const int hi = (int)((byte >> 4) & 0xF) - 8;
      score += sh_q[2*j]   * (float)lo * k_scale;
      score += sh_q[2*j+1] * (float)hi * k_scale;
    }
    score = block_reduce_sum(score, threadIdx.x, blockDim.x,
                              shmem + 2 * head_dim);
    float s = score * sm_scale;
    float m_new = fmaxf(m, s);
    float exp_diff = __expf(m - m_new);
    float w = __expf(s - m_new);

    for (int j = threadIdx.x; j < head_dim / 2; j += blockDim.x) {
      const uint8_t byte = v_row[j];
      const int lo = (int)(byte & 0xF) - 8;
      const int hi = (int)((byte >> 4) & 0xF) - 8;
      sh_accum[2*j]     = sh_accum[2*j]     * exp_diff + (float)lo * v_scale * w;
      sh_accum[2*j + 1] = sh_accum[2*j + 1] * exp_diff + (float)hi * v_scale * w;
    }
    l = l * exp_diff + w;
    m = m_new;
  }

  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    out[q_head * head_dim + i] = sh_accum[i] * inv_l;
  }
}

torch::Tensor paged_attention_decode_int4_cuda(
    torch::Tensor q,
    torch::Tensor k_pool, torch::Tensor k_scales,
    torch::Tensor v_pool, torch::Tensor v_scales,
    torch::Tensor page_table,
    torch::Tensor n_tokens,
    float sm_scale) {
  TORCH_CHECK(q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda()
              && page_table.is_cuda() && n_tokens.is_cuda(),
              "paged_attention_decode_int4_cuda: tensors must be CUDA");

  c10::cuda::CUDAGuard guard(q.device());
  auto q_c = q.contiguous().to(torch::kFloat32);
  auto kp  = k_pool.contiguous();
  auto vp  = v_pool.contiguous();
  auto ks  = k_scales.contiguous().to(torch::kBFloat16);
  auto vs  = v_scales.contiguous().to(torch::kBFloat16);
  auto pt  = page_table.contiguous().to(torch::kInt32);
  auto nt  = n_tokens.contiguous().to(torch::kInt32);

  const int n_q_heads  = q_c.size(0);
  const int head_dim   = q_c.size(1);
  const int max_pages  = kp.size(0);
  const int page_size  = kp.size(1);
  const int n_kv_heads = kp.size(2);

  auto out = torch::empty({n_q_heads, head_dim}, q_c.options());
  const size_t shmem = (2 * head_dim + kMaxWarps) * sizeof(float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  paged_attention_decode_int4_kernel<<<n_q_heads, kThreads, shmem, stream>>>(
      q_c.data_ptr<float>(),
      kp.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
      vp.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
      pt.data_ptr<int32_t>(),
      nt.data_ptr<int32_t>(),
      n_q_heads, n_kv_heads, head_dim, page_size, max_pages, sm_scale,
      out.data_ptr<float>());

  return out.to(q.dtype());
}

}  // namespace olmo_cpp
