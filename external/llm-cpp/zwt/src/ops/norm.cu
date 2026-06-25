#include "zwt/src/ops/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace zwt::ops::k {

namespace {

__device__ __forceinline__ float bf_to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f_to_bf(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.f;
  if (wid == 0) v = warp_sum(v);
  return v;
}

// One row per block. Reduce sum(x^2), then write y = x * rstd * w.
__global__ void k_rmsnorm_fwd_bf16(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* y, float* rstd,
                                   int64_t cols, float eps) {
  int64_t r = blockIdx.x;
  const __nv_bfloat16* xr = x + r * cols;
  __nv_bfloat16* yr = y + r * cols;

  float ss = 0.f;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = bf_to_f(xr[c]);
    ss += v * v;
  }
  ss = block_sum(ss);

  __shared__ float s_rs;
  if (threadIdx.x == 0) s_rs = rsqrtf(ss / float(cols) + eps);
  __syncthreads();

  if (threadIdx.x == 0 && rstd) rstd[r] = s_rs;

  float rs = s_rs;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    yr[c] = f_to_bf(bf_to_f(xr[c]) * rs * bf_to_f(weight[c]));
  }
}

// Fused residual: sum = x + res, then rmsnorm(sum) * w, writing y and sum.
__global__ void k_rmsnorm_residual_fwd_bf16(const __nv_bfloat16* x,
                                            const __nv_bfloat16* res,
                                            const __nv_bfloat16* weight,
                                            __nv_bfloat16* y,
                                            __nv_bfloat16* sum_out,
                                            float* rstd,
                                            int64_t cols, float eps) {
  int64_t r = blockIdx.x;
  const __nv_bfloat16* xr = x + r * cols;
  const __nv_bfloat16* rr = res + r * cols;
  __nv_bfloat16* sr = sum_out + r * cols;
  __nv_bfloat16* yr = y + r * cols;

  float ss = 0.f;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    float s = bf_to_f(xr[c]) + bf_to_f(rr[c]);
    sr[c] = f_to_bf(s);
    ss += s * s;
  }
  ss = block_sum(ss);

  __shared__ float s_rs;
  if (threadIdx.x == 0) s_rs = rsqrtf(ss / float(cols) + eps);
  __syncthreads();
  if (threadIdx.x == 0 && rstd) rstd[r] = s_rs;

  float rs = s_rs;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    yr[c] = f_to_bf(bf_to_f(sr[c]) * rs * bf_to_f(weight[c]));
  }
}

// Backward: produces grad_x, accumulates into grad_weight (fp32).
// Two-pass: reduce dot, then emit. Uses per-block atomic add to grad_weight.
__global__ void k_rmsnorm_bwd_bf16(const __nv_bfloat16* grad_y,
                                   const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   const float* rstd,
                                   __nv_bfloat16* grad_x,
                                   float* grad_weight,
                                   int64_t cols) {
  int64_t r = blockIdx.x;
  float rs = rstd[r];

  const __nv_bfloat16* gr = grad_y + r * cols;
  const __nv_bfloat16* xr = x + r * cols;
  __nv_bfloat16* gxr = grad_x + r * cols;

  float dot = 0.f;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    dot += bf_to_f(xr[c]) * bf_to_f(weight[c]) * bf_to_f(gr[c]);
  }
  dot = block_sum(dot);
  __shared__ float s_dot;
  if (threadIdx.x == 0) s_dot = dot;
  __syncthreads();
  float dotv = s_dot;

  float coef = (rs * rs * rs) / float(cols);
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    float xv = bf_to_f(xr[c]);
    float gyv = bf_to_f(gr[c]);
    float wv = bf_to_f(weight[c]);
    gxr[c] = f_to_bf(wv * rs * gyv - coef * xv * dotv);
    atomicAdd(grad_weight + c, xv * rs * gyv);
  }
}

}  // namespace

void rmsnorm_fwd_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                      __nv_bfloat16* y, float* rstd,
                      int64_t rows, int64_t cols, float eps, cudaStream_t s) {
  int threads = cols >= 1024 ? 1024 : (cols >= 512 ? 512 : 256);
  k_rmsnorm_fwd_bf16<<<static_cast<unsigned>(rows), threads, 0, s>>>(
      x, weight, y, rstd, cols, eps);
}

void rmsnorm_residual_fwd_bf16(const __nv_bfloat16* x, const __nv_bfloat16* res,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* y, __nv_bfloat16* sum_out,
                               float* rstd, int64_t rows, int64_t cols,
                               float eps, cudaStream_t s) {
  int threads = cols >= 1024 ? 1024 : (cols >= 512 ? 512 : 256);
  k_rmsnorm_residual_fwd_bf16<<<static_cast<unsigned>(rows), threads, 0, s>>>(
      x, res, weight, y, sum_out, rstd, cols, eps);
}

void rmsnorm_bwd_bf16(const __nv_bfloat16* grad_y, const __nv_bfloat16* x,
                      const __nv_bfloat16* weight, const float* rstd,
                      __nv_bfloat16* grad_x, float* grad_weight,
                      int64_t rows, int64_t cols, float /*eps*/,
                      cudaStream_t s) {
  int threads = cols >= 1024 ? 1024 : (cols >= 512 ? 512 : 256);
  k_rmsnorm_bwd_bf16<<<static_cast<unsigned>(rows), threads, 0, s>>>(
      grad_y, x, weight, rstd, grad_x, grad_weight, cols);
}

}  // namespace zwt::ops::k
