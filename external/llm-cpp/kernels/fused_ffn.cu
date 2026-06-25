/**
 * kernels/fused_ffn.cu
 *
 * Fused SwiGLU FFN (item I), CUDA forward kernel (per-row, FMA loops).
 *
 * One block per row of output (= B*S total). Shared memory holds the
 * [2H] gate_up intermediate AND the [H] post-silu_mul intermediate.
 * The two matmuls run as FMA loops in this kernel — NOT tensor cores.
 * That makes this kernel correct but slower than cuBLAS at large
 * B*S; it wins only when launch overhead dominates (small-batch
 * decode) OR when the HBM bandwidth saved exceeds the FLOP slowdown.
 *
 * The tensor-core variant (mma.sync / wgmma) lands in a follow-on
 * — that's where the real speedup for training-shape B*S lives.
 * This commit ships the correctness scaffolding.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#include "olmo_cpp/backend/fused_ffn.hpp"
#include "mma_sync.cuh"  // item 2: tensor-core mma helpers for gate_up + down matmuls

namespace olmo_cpp {

namespace {

__device__ __forceinline__ float silu(float v) {
  return v / (1.0f + __expf(-v));
}

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__global__ void fused_ffn_kernel(
    const T* __restrict__ x,         // [N, d]
    const T* __restrict__ w_gate_up, // [2H, d]
    const T* __restrict__ w_down,    // [d, H]
    T* __restrict__ y,                // [N, d]
    int N, int d, int H) {
  const int row = blockIdx.x;
  if (row >= N) return;
  const T* x_row = x + static_cast<int64_t>(row) * d;
  T* y_row = y + static_cast<int64_t>(row) * d;

  extern __shared__ float smem[];
  float* gate_up = smem;          // [2H]
  float* act     = smem + 2 * H;  // [H]

  // 1) gate_up matmul: gate_up[h] = sum_j x_row[j] * w_gate_up[h, j]
  for (int h = threadIdx.x; h < 2 * H; h += blockDim.x) {
    float acc = 0.0f;
    const T* w_row = w_gate_up + static_cast<int64_t>(h) * d;
    for (int j = 0; j < d; ++j) {
      acc += to_f32<T>(x_row[j]) * to_f32<T>(w_row[j]);
    }
    gate_up[h] = acc;
  }
  __syncthreads();

  // 2) silu_mul: act[h] = silu(gate_up[h]) * gate_up[h+H]
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    act[h] = silu(gate_up[h]) * gate_up[h + H];
  }
  __syncthreads();

  // 3) down matmul: y_row[i] = sum_h act[h] * w_down[i, h]
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float acc = 0.0f;
    const T* w_row = w_down + static_cast<int64_t>(i) * H;
    for (int h = 0; h < H; ++h) {
      acc += act[h] * to_f32<T>(w_row[h]);
    }
    y_row[i] = from_f32<T>(acc);
  }
}

}  // namespace

torch::Tensor fused_ffn_cuda(torch::Tensor x,
                              torch::Tensor w_gate_up,
                              torch::Tensor w_down) {
  TORCH_CHECK(x.is_cuda() && w_gate_up.is_cuda() && w_down.is_cuda(),
              "fused_ffn_cuda: all tensors must be CUDA");
  c10::cuda::CUDAGuard guard(x.device());
  auto x_c = x.contiguous();
  auto wg = w_gate_up.contiguous();
  auto wd = w_down.contiguous();

  const int64_t B = x_c.size(0);
  const int64_t S = x_c.size(1);
  const int64_t d = x_c.size(2);
  const int64_t H = wg.size(0) / 2;
  TORCH_CHECK(wg.size(1) == d, "w_gate_up inner dim must match x last dim");
  TORCH_CHECK(wd.size(0) == d && wd.size(1) == H,
              "w_down must be [d, H]");

  auto y = torch::empty_like(x_c);
  const int N = static_cast<int>(B * S);
  const int threads = 128;
  const size_t shmem = (2 * H + H) * sizeof(float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x_c.scalar_type() == torch::kBFloat16) {
    fused_ffn_kernel<__nv_bfloat16><<<N, threads, shmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wg.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wd.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        N, static_cast<int>(d), static_cast<int>(H));
  } else {
    auto xf = x_c.to(torch::kFloat32).contiguous();
    auto wgf = wg.to(torch::kFloat32).contiguous();
    auto wdf = wd.to(torch::kFloat32).contiguous();
    fused_ffn_kernel<float><<<N, threads, shmem, stream>>>(
        xf.data_ptr<float>(),
        wgf.data_ptr<float>(),
        wdf.data_ptr<float>(),
        y.data_ptr<float>(),
        N, static_cast<int>(d), static_cast<int>(H));
  }
  return y;
}

}  // namespace olmo_cpp
