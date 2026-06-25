/**
 * kernels/rms_norm_backward.cu
 *
 * Fused RMSNorm backward (item B3).
 *
 * Forward:  y = x * rsqrt(mean(x^2) + eps) * w
 * Backward (per row r):
 *   r          = rsqrt(mean_i(x[r,i]^2) + eps)
 *   x_hat[i]   = x[r,i] * r
 *   gxh[i]     = grad_y[r,i] * (w defined ? w[i] : 1)
 *   mean_term  = mean_i(gxh[i] * x_hat[i])
 *   grad_x[r,i] = r * (gxh[i] - x_hat[i] * mean_term)
 *   grad_w[i] += grad_y[r,i] * x_hat[i]            (sum over r)
 *
 * Strategy:
 *   - One block per row of the flattened [N, D] view (N = B*S*...).
 *   - Each block loads x_row and grad_y_row into shared memory (fp32).
 *   - Two block-wide reductions:
 *       (a) sum(x^2)  → rms scalar
 *       (b) sum(gxh * x_hat) → mean_term scalar
 *   - Single elementwise pass writes grad_x and atomicAdds into grad_w.
 *
 * grad_w is allocated in fp32 (we cast back to compute dtype host-side
 * after the kernel), to keep accumulation precise across N rows of
 * bf16 source. Atomic contention on D entries × N rows is acceptable
 * on Ampere+ for the shapes we care about (D ≤ 4096).
 *
 * The matching schema-registered backward op is dispatched from
 * cuda_ops_autograd.cpp's RmsNormFn::backward — the fused kernel
 * replaces ~10 ATen ops there.
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "olmo_cpp/backend/rms_norm_backward.hpp"

namespace olmo_cpp {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxD = 4096;

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* shmem) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int wid  = threadIdx.x >> 5;
  const int num_warps = blockDim.x / kWarpSize;
  v = warp_reduce_sum(v);
  if (lane == 0) shmem[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float w = (lane < num_warps) ? shmem[lane] : 0.0f;
    w = warp_reduce_sum(w);
    if (lane == 0) shmem[0] = w;
  }
  __syncthreads();
  return shmem[0];
}

template <typename T>
__global__ void rms_norm_backward_kernel(
    const T* __restrict__ grad_y,    // [N, D]
    const T* __restrict__ x,         // [N, D]
    const T* __restrict__ weight,    // [D] or nullptr
    T* __restrict__ grad_x,           // [N, D]
    float* __restrict__ grad_w_fp32,  // [D] or nullptr
    int N, int D, float eps) {
  const int row = blockIdx.x;
  if (row >= N) return;

  extern __shared__ float smem[];
  float* sh_x  = smem;                // [D]
  float* sh_gy = smem + D;            // [D]
  float* sh_red = smem + 2 * D;       // [warps_per_block]

  // Load both rows into shared memory in fp32. Coalesced reads.
  const T* x_row  = x      + (int64_t)row * D;
  const T* gy_row = grad_y + (int64_t)row * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    sh_x[i]  = to_float<T>(x_row[i]);
    sh_gy[i] = to_float<T>(gy_row[i]);
  }
  __syncthreads();

  // Pass 1 — sum(x^2) → rms.
  float local_sum_x2 = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = sh_x[i];
    local_sum_x2 += v * v;
  }
  const float sum_x2 = block_reduce_sum(local_sum_x2, sh_red);
  const float rms = rsqrtf(sum_x2 / static_cast<float>(D) + eps);

  // Pass 2 — sum(grad_x_hat * x_hat) → mean_term.
  // grad_x_hat[i] = grad_y[i] * weight[i] (weight broadcast over rows).
  // x_hat[i]      = x[i] * rms.
  float local_dot = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float wi  = (weight != nullptr) ? to_float<T>(weight[i]) : 1.0f;
    const float gxh = sh_gy[i] * wi;
    const float xh  = sh_x[i]  * rms;
    local_dot += gxh * xh;
  }
  const float dot_sum = block_reduce_sum(local_dot, sh_red);
  const float mean_term = dot_sum / static_cast<float>(D);

  // Pass 3 — write grad_x; atomic-accumulate grad_w.
  T* gx_row = grad_x + (int64_t)row * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float wi  = (weight != nullptr) ? to_float<T>(weight[i]) : 1.0f;
    const float gxh = sh_gy[i] * wi;
    const float xh  = sh_x[i]  * rms;
    const float gx  = rms * (gxh - xh * mean_term);
    gx_row[i] = from_float<T>(gx);
    if (grad_w_fp32 != nullptr) {
      atomicAdd(&grad_w_fp32[i], sh_gy[i] * xh);
    }
  }
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor>
rms_norm_backward_cuda(torch::Tensor grad_y,
                         torch::Tensor x,
                         c10::optional<torch::Tensor> weight,
                         double eps) {
  TORCH_CHECK(grad_y.is_cuda() && x.is_cuda(),
              "rms_norm_backward_cuda: grad_y, x must be CUDA");
  TORCH_CHECK(grad_y.sizes() == x.sizes(),
              "rms_norm_backward_cuda: grad_y, x shape mismatch");
  TORCH_CHECK(grad_y.scalar_type() == x.scalar_type(),
              "rms_norm_backward_cuda: grad_y, x dtype mismatch");
  c10::cuda::CUDAGuard guard(x.device());

  auto x_c  = x.contiguous();
  auto gy_c = grad_y.contiguous();
  const int D = static_cast<int>(x_c.size(-1));
  const int64_t total = x_c.numel();
  const int N = static_cast<int>(total / D);
  TORCH_CHECK(D > 0 && D <= kMaxD,
              "rms_norm_backward_cuda: D must be in (0, ", kMaxD, "]");

  auto grad_x = torch::empty_like(x_c);

  torch::Tensor weight_c;
  bool has_weight = weight.has_value() && weight->defined();
  if (has_weight) weight_c = weight->contiguous();

  // grad_w accumulator in fp32 (atomic-add precision).
  auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
  torch::Tensor grad_w_fp32;
  if (has_weight) grad_w_fp32 = torch::zeros({D}, fp32_opts);

  const int warps_per_block = kThreadsPerBlock / kWarpSize;
  const size_t shmem_bytes =
        (size_t)(2 * D) * sizeof(float)             // sh_x + sh_gy
      + (size_t)warps_per_block * sizeof(float);    // block-reduce scratch

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == torch::kBFloat16) {
    rms_norm_backward_kernel<__nv_bfloat16>
        <<<N, kThreadsPerBlock, shmem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gy_c.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr<at::BFloat16>()),
        has_weight ? reinterpret_cast<const __nv_bfloat16*>(
                          weight_c.data_ptr<at::BFloat16>())
                   : nullptr,
        reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr<at::BFloat16>()),
        has_weight ? grad_w_fp32.data_ptr<float>() : nullptr,
        N, D, static_cast<float>(eps));
  } else if (x.scalar_type() == torch::kFloat32) {
    rms_norm_backward_kernel<float>
        <<<N, kThreadsPerBlock, shmem_bytes, stream>>>(
        gy_c.data_ptr<float>(),
        x_c.data_ptr<float>(),
        has_weight ? weight_c.data_ptr<float>() : nullptr,
        grad_x.data_ptr<float>(),
        has_weight ? grad_w_fp32.data_ptr<float>() : nullptr,
        N, D, static_cast<float>(eps));
  } else {
    TORCH_CHECK(false, "rms_norm_backward_cuda: only bf16 / fp32 supported");
  }

  torch::Tensor grad_w;
  if (has_weight) grad_w = grad_w_fp32.to(weight_c.dtype());
  return {grad_x, grad_w};
}

}  // namespace olmo_cpp
