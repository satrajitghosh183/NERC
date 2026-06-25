/**
 * kernels/topk_argmax.cu
 *
 * Fused GEMV + argmax (item K). One block per (b,s) row; threads in
 * the block stride over the vocabulary chunkwise, tracking a running
 * (max_value, max_index). Block reduce -> one int64 argmax per row.
 *
 * Memory: no [V] logits row ever lands in HBM. For V=50257, d=768,
 * bf16, that's 98 KB / row saved. For a speculative verify of S=3
 * positions, ~300 KB / step.
 *
 * Numerics: identical to torch::matmul(h, w.t()).argmax(-1) on
 * fp32 inputs; bf16 inputs are promoted internally before the dot
 * product, so the comparison ordering matches.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cfloat>

#include "olmo_cpp/backend/topk_argmax.hpp"

namespace olmo_cpp {

namespace {

constexpr int kThreads = 128;

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// Block-reduce max with argmax. Threads have (max_val, max_idx) each;
// after this, thread 0 holds the global max.
__device__ __forceinline__ void block_reduce_argmax(float& v, int& i) {
  __shared__ float smax[kThreads / 32];
  __shared__ int   sidx[kThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  for (int off = 16; off > 0; off >>= 1) {
    float ov = __shfl_xor_sync(0xffffffff, v, off);
    int   oi = __shfl_xor_sync(0xffffffff, i, off);
    if (ov > v) { v = ov; i = oi; }
  }
  if (lane == 0) { smax[warp] = v; sidx[warp] = i; }
  __syncthreads();
  if (warp == 0) {
    const int n_warps = kThreads >> 5;
    v = (lane < n_warps) ? smax[lane] : -FLT_MAX;
    i = (lane < n_warps) ? sidx[lane] : -1;
    for (int off = 16; off > 0; off >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, v, off);
      int   oi = __shfl_xor_sync(0xffffffff, i, off);
      if (ov > v) { v = ov; i = oi; }
    }
  }
}

template <typename T>
__global__ void lmhead_argmax_kernel(
    const T* __restrict__ hidden,    // [N, d]
    const T* __restrict__ weight,    // [V, d]
    int64_t* __restrict__ out,       // [N]
    int N, int V, int d) {
  const int row = blockIdx.x;
  if (row >= N) return;
  const T* h_row = hidden + static_cast<int64_t>(row) * d;

  float best_v = -FLT_MAX;
  int   best_i = -1;

  for (int v = threadIdx.x; v < V; v += blockDim.x) {
    const T* w_row = weight + static_cast<int64_t>(v) * d;
    float dot = 0.0f;
    for (int j = 0; j < d; ++j) {
      dot += to_f32<T>(h_row[j]) * to_f32<T>(w_row[j]);
    }
    if (dot > best_v) { best_v = dot; best_i = v; }
  }

  block_reduce_argmax(best_v, best_i);
  if (threadIdx.x == 0) out[row] = static_cast<int64_t>(best_i);
}

}  // namespace

torch::Tensor lmhead_argmax_cuda(torch::Tensor hidden, torch::Tensor weight) {
  TORCH_CHECK(hidden.is_cuda() && weight.is_cuda(),
              "lmhead_argmax_cuda: tensors must be CUDA");
  TORCH_CHECK(hidden.dim() == 3 && weight.dim() == 2,
              "shapes: hidden [B,S,d], weight [V,d]");
  c10::cuda::CUDAGuard guard(hidden.device());
  auto h = hidden.contiguous();
  auto w = weight.contiguous();
  const int64_t B = h.size(0);
  const int64_t S = h.size(1);
  const int64_t d = h.size(2);
  const int64_t V = w.size(0);
  TORCH_CHECK(w.size(1) == d, "weight inner dim must match hidden");

  const int64_t N = B * S;
  auto out = torch::empty({B, S},
      torch::TensorOptions().dtype(torch::kInt64).device(h.device()));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (h.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16) {
    lmhead_argmax_kernel<__nv_bfloat16><<<static_cast<int>(N), kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
        out.data_ptr<int64_t>(),
        static_cast<int>(N), static_cast<int>(V), static_cast<int>(d));
  } else {
    auto hf = h.to(torch::kFloat32).contiguous();
    auto wf = w.to(torch::kFloat32).contiguous();
    lmhead_argmax_kernel<float><<<static_cast<int>(N), kThreads, 0, stream>>>(
        hf.data_ptr<float>(), wf.data_ptr<float>(),
        out.data_ptr<int64_t>(),
        static_cast<int>(N), static_cast<int>(V), static_cast<int>(d));
  }
  return out;
}

}  // namespace olmo_cpp
