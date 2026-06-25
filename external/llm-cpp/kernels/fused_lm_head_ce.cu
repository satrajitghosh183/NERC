/**
 * kernels/fused_lm_head_ce.cu
 *
 * Fused LM-head + softmax-cross-entropy CUDA kernel — item A3.
 *
 * Per-row work: one block processes one row n.
 *   1. Load h[n, :] into shared memory.
 *   2. Stream V in chunks of `blockDim.x`. For each chunk, every thread
 *      computes one logit = dot(h_row, W[v, :]). Each thread maintains a
 *      running (max, sum) state via the online-softmax merge:
 *          merge((m1, s1), (m2, s2)):
 *              m = max(m1, m2)
 *              s = s1 * exp(m1 - m) + s2 * exp(m2 - m)
 *   3. Block-reduce the per-thread (max, sum) pairs to a single (row_max,
 *      row_sum) via warp-shuffle + shmem.
 *   4. Compute label_logit = dot(h_row, W[label, :]) in a separate pass.
 *   5. row_loss = (row_max + log(row_sum)) - label_logit.
 *   6. atomicAdd row_loss into the scalar `loss_sum_out` and increment
 *      `valid_count_out`. Host divides loss_sum by valid_count to get
 *      the mean (matches PyTorch's reduction=Mean semantics).
 *
 * Numerics: online softmax is exact wrt naive log-sum-exp up to the
 * fp32 atomic adds. The full V-pass uses naive (non-WMMA) per-element
 * GEMV inside each thread; tensor-core variants are a follow-on.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <math_constants.h>   // CUDART_INF_F
#include <cmath>
#include <cstdint>

#include "olmo_cpp/backend/fused_lm_head_ce.hpp"

namespace olmo_cpp {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxD = 4096;   // h_row shmem cap; covers d_model ≤ 4096 (7B-class)

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

// Warp-level online softmax merge. Reduces (m, s) across the warp.
__device__ __forceinline__ void warp_reduce_softmax(float& m, float& s) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float other_m = __shfl_xor_sync(0xffffffff, m, offset);
    float other_s = __shfl_xor_sync(0xffffffff, s, offset);
    float new_m = fmaxf(m, other_m);
    s = s * __expf(m - new_m) + other_s * __expf(other_m - new_m);
    m = new_m;
  }
}

template <typename T_in>
__global__ void fused_lm_head_ce_kernel(
    const T_in* __restrict__ h,         // [N, d]
    const T_in* __restrict__ W,         // [V, d]
    const int64_t* __restrict__ labels, // [N]
    int64_t ignore_index,
    float* __restrict__ loss_sum_out,   // scalar (fp32 atomic)
    int* __restrict__ count_out,        // scalar (int atomic)
    int N, int V, int d) {
  const int row = blockIdx.x;
  if (row >= N) return;
  const int64_t label = labels[row];
  if (label == ignore_index) return;

  // Out-of-vocab labels (treated as ignore by PyTorch's CE for safety in
  // ragged/structural batches). Defensive — keeps the atomicAdd domain
  // valid even when callers haven't pre-filtered.
  if (label < 0 || label >= V) return;

  extern __shared__ float smem[];
  float* h_row       = smem;                       // [d]
  float* warp_maxes  = smem + kMaxD;               // [warps_per_block]
  float* warp_sums   = warp_maxes + (kThreadsPerBlock / kWarpSize);

  // Load h_row.
  const T_in* h_in = h + (int64_t)row * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    h_row[i] = to_float<T_in>(h_in[i]);
  }
  __syncthreads();

  // Pass 1 — single-pass online softmax over V.
  float t_max = -CUDART_INF_F;
  float t_sum = 0.0f;
  for (int v = threadIdx.x; v < V; v += blockDim.x) {
    const T_in* w_row = W + (int64_t)v * d;
    float logit = 0.0f;
    for (int i = 0; i < d; ++i) {
      logit += h_row[i] * to_float<T_in>(w_row[i]);
    }
    // Online merge of (t_max, t_sum) with (logit, 1).
    if (logit > t_max) {
      t_sum = t_sum * __expf(t_max - logit) + 1.0f;
      t_max = logit;
    } else {
      t_sum += __expf(logit - t_max);
    }
  }

  // Warp-level online softmax reduction.
  warp_reduce_softmax(t_max, t_sum);

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int wid  = threadIdx.x >> 5;
  const int num_warps = blockDim.x / kWarpSize;
  if (lane == 0) {
    warp_maxes[wid] = t_max;
    warp_sums[wid]  = t_sum;
  }
  __syncthreads();

  float row_max = -CUDART_INF_F;
  float row_sum = 0.0f;
  if (wid == 0) {
    float m = (lane < num_warps) ? warp_maxes[lane] : -CUDART_INF_F;
    float s = (lane < num_warps) ? warp_sums[lane]  : 0.0f;
    warp_reduce_softmax(m, s);
    if (lane == 0) {
      warp_maxes[0] = m;
      warp_sums[0]  = s;
    }
  }
  __syncthreads();
  row_max = warp_maxes[0];
  row_sum = warp_sums[0];

  // Pass 2 — label_logit = h_row . W[label, :].
  const T_in* w_label = W + (int64_t)label * d;
  float lp = 0.0f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    lp += h_row[i] * to_float<T_in>(w_label[i]);
  }
  lp = warp_reduce_sum(lp);
  if (lane == 0) warp_sums[wid] = lp;
  __syncthreads();
  float label_logit = 0.0f;
  if (wid == 0) {
    float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[0] = v;
  }
  __syncthreads();
  label_logit = warp_sums[0];

  // Accumulate row loss.
  if (threadIdx.x == 0) {
    const float lse = row_max + __logf(row_sum);
    const float row_loss = lse - label_logit;
    atomicAdd(loss_sum_out, row_loss);
    atomicAdd(count_out, 1);
  }
}

}  // namespace

torch::Tensor fused_lm_head_ce_cuda(torch::Tensor h,
                                      torch::Tensor weight,
                                      torch::Tensor labels,
                                      int64_t ignore_index) {
  TORCH_CHECK(h.is_cuda() && weight.is_cuda() && labels.is_cuda(),
              "fused_lm_head_ce_cuda: all tensors must be on CUDA");
  TORCH_CHECK(h.dim() == 2, "h must be [N, d]");
  TORCH_CHECK(weight.dim() == 2, "weight must be [V, d]");
  TORCH_CHECK(labels.dim() == 1, "labels must be [N]");
  TORCH_CHECK(labels.scalar_type() == torch::kInt64, "labels must be int64");
  TORCH_CHECK(h.scalar_type() == weight.scalar_type(),
              "h and weight must share dtype");

  c10::cuda::CUDAGuard guard(h.device());
  auto h_c = h.contiguous();
  auto w_c = weight.contiguous();
  auto l_c = labels.contiguous();

  const int N = static_cast<int>(h_c.size(0));
  const int d = static_cast<int>(h_c.size(1));
  const int V = static_cast<int>(w_c.size(0));
  TORCH_CHECK(w_c.size(1) == d, "weight inner dim mismatch");
  TORCH_CHECK(d <= kMaxD,
              "fused_lm_head_ce: d > ", kMaxD, " not supported");

  auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(h.device());
  auto int_opts  = torch::TensorOptions().dtype(torch::kInt32).device(h.device());
  auto loss_sum = torch::zeros({}, fp32_opts);
  auto count    = torch::zeros({}, int_opts);

  const int warps_per_block = kThreadsPerBlock / kWarpSize;
  const size_t shmem_bytes =
        (size_t)kMaxD * sizeof(float)       // h_row (capacity)
      + (size_t)warps_per_block * sizeof(float) * 2;  // warp_maxes + warp_sums

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (h.scalar_type() == torch::kBFloat16) {
    fused_lm_head_ce_kernel<__nv_bfloat16><<<N, kThreadsPerBlock, shmem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr<at::BFloat16>()),
        l_c.data_ptr<int64_t>(),
        ignore_index,
        loss_sum.data_ptr<float>(),
        count.data_ptr<int>(),
        N, V, d);
  } else if (h.scalar_type() == torch::kFloat32) {
    fused_lm_head_ce_kernel<float><<<N, kThreadsPerBlock, shmem_bytes, stream>>>(
        h_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        l_c.data_ptr<int64_t>(),
        ignore_index,
        loss_sum.data_ptr<float>(),
        count.data_ptr<int>(),
        N, V, d);
  } else {
    TORCH_CHECK(false, "fused_lm_head_ce_cuda: only bf16 / fp32 supported");
  }

  // Mean over non-ignored rows. clamp_min(1) avoids div-by-zero on
  // pathological all-ignored batches (matches PyTorch's behavior of
  // returning 0 in that case, except PyTorch returns NaN; we prefer 0).
  auto count_f = count.to(torch::kFloat32).clamp_min(1.0f);
  auto mean = loss_sum / count_f;
  return mean.to(h.dtype());
}

}  // namespace olmo_cpp
