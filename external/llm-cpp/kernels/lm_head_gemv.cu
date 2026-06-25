/**
 * kernels/lm_head_gemv.cu
 *
 * Custom LM-head GEMV — fast-inference [11].
 *
 * One thread per row of W_U. Each thread streams W_U[i, :] and dots
 * with hidden (staged in shared memory). Vectorized 4-wide on the FP32
 * path. BF16 path is scalar (TODO: vectorize via __nv_bfloat162).
 *
 * Same template as fused_lm_head_sample.cu's GEMV portion, but stops
 * at writing logits[V] to HBM (no Gumbel/argmax epilogue). Use this
 * when the caller wants raw logits (e.g. speculative verification).
 *
 * DRAFT — not benchmarked vs cuBLAS yet.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

#include "olmo_cpp/backend/lm_head_gemv.hpp"

namespace olmo_cpp {

namespace {

constexpr int kThreads = 256;

template <typename WT>
__global__ void lm_head_gemv_kernel(
    const float* __restrict__ hidden,
    const WT*    __restrict__ W_U,
    int64_t V,
    int H,
    float* __restrict__ logits) {
  extern __shared__ float sh_hidden[];
  for (int i = threadIdx.x; i < H; i += blockDim.x) sh_hidden[i] = hidden[i];
  __syncthreads();

  int64_t row    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (; row < V; row += stride) {
    const WT* w_row = W_U + row * H;
    float l = 0.0f;
    if constexpr (std::is_same<WT, float>::value) {
      int h = 0;
      for (; h + 4 <= H; h += 4) {
        float4 w = *reinterpret_cast<const float4*>(w_row + h);
        float4 v = *reinterpret_cast<const float4*>(sh_hidden + h);
        l += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
      }
      for (; h < H; ++h) l += w_row[h] * sh_hidden[h];
    } else {
      // BF16 path: pair-load via __nv_bfloat162 (one 32-bit load per pair
      // instead of two 16-bit loads). __low2float / __high2float convert
      // the two halves to FP32 for the dot accumulation in FP32.
      int h = 0;
      const auto* w_row2 = reinterpret_cast<const __nv_bfloat162*>(w_row);
      for (; h + 2 <= H; h += 2) {
        __nv_bfloat162 w2 = w_row2[h >> 1];
        l += __low2float(w2)  * sh_hidden[h]
           + __high2float(w2) * sh_hidden[h + 1];
      }
      for (; h < H; ++h) l += __bfloat162float(w_row[h]) * sh_hidden[h];
    }
    logits[row] = l;
  }
}

}  // namespace

torch::Tensor lm_head_gemv_cuda(torch::Tensor hidden, torch::Tensor W_U) {
  TORCH_CHECK(hidden.is_cuda() && W_U.is_cuda(), "lm_head_gemv_cuda: must be CUDA");
  TORCH_CHECK(hidden.dim() == 1 && W_U.dim() == 2);
  TORCH_CHECK(W_U.size(1) == hidden.size(0));

  c10::cuda::CUDAGuard guard(hidden.device());
  auto h_c = hidden.contiguous().to(torch::kFloat32);
  auto W_c = W_U.contiguous();

  const int64_t V = W_c.size(0);
  const int     H = static_cast<int>(W_c.size(1));

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(hidden.device());
  auto out  = torch::empty({V}, opts);

  int blocks = static_cast<int>(std::min<int64_t>((V + kThreads - 1) / kThreads, 1024));
  size_t shmem = static_cast<size_t>(H) * sizeof(float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (W_c.scalar_type() == torch::kFloat32) {
    lm_head_gemv_kernel<float><<<blocks, kThreads, shmem, stream>>>(
        h_c.data_ptr<float>(), W_c.data_ptr<float>(),
        V, H, out.data_ptr<float>());
  } else if (W_c.scalar_type() == torch::kBFloat16) {
    lm_head_gemv_kernel<__nv_bfloat16><<<blocks, kThreads, shmem, stream>>>(
        h_c.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(W_c.data_ptr<at::BFloat16>()),
        V, H, out.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "lm_head_gemv_cuda: W_U dtype must be FP32 or BF16");
  }

  return out;
}

}  // namespace olmo_cpp
