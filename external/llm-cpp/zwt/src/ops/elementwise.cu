#include "zwt/src/ops/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace zwt::ops::k {

namespace {

__device__ __forceinline__ float bf_to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f_to_bf(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__global__ void k_scale_bf16(__nv_bfloat16* y, float a, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = f_to_bf(bf_to_f(y[i]) * a);
}

__global__ void k_axpy_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, float a, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = f_to_bf(bf_to_f(y[i]) + a * bf_to_f(x[i]));
}

__global__ void k_add_bf16(__nv_bfloat16* out, const __nv_bfloat16* a,
                           const __nv_bfloat16* b, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = f_to_bf(bf_to_f(a[i]) + bf_to_f(b[i]));
}

// 1-D grid-stride add_bias. The 2-D version that lived here before launched
// rows*ceil(cols/256) blocks — for batch*seq=16K, cols=896 that's 65K
// blocks each doing one elementwise op, which exceeds practical scheduling
// width and burns launch overhead. Vectorize via __nv_bfloat162 to halve
// the load/store transactions.
__global__ void k_add_bias_bf162(__nv_bfloat162* __restrict__ y,
                                 const __nv_bfloat162* __restrict__ bias,
                                 int64_t total, int64_t cols2) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t c = i % cols2;
    y[i] = __hadd2(y[i], bias[c]);
  }
}

// Scalar fallback for cols that aren't even (shouldn't happen on this
// codebase — every dim we hit is multiple of 64 — but keep it correct).
__global__ void k_add_bias_bf16(__nv_bfloat16* __restrict__ y,
                                const __nv_bfloat16* __restrict__ bias,
                                int64_t total, int64_t cols) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t c = i % cols;
    y[i] = f_to_bf(bf_to_f(y[i]) + bf_to_f(bias[c]));
  }
}

__global__ void k_silu_mul_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate,
                                const __nv_bfloat16* up, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = bf_to_f(gate[i]);
  float u = bf_to_f(up[i]);
  out[i] = f_to_bf(silu(g) * u);
}

__global__ void k_silu_mul_bwd_bf16(const __nv_bfloat16* grad_out,
                                    const __nv_bfloat16* gate,
                                    const __nv_bfloat16* up,
                                    __nv_bfloat16* grad_gate,
                                    __nv_bfloat16* grad_up, int64_t n) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = bf_to_f(gate[i]);
  float u = bf_to_f(up[i]);
  float go = bf_to_f(grad_out[i]);
  float sig = 1.0f / (1.0f + __expf(-g));
  float s = g * sig;
  float dsilu = sig * (1.0f + g * (1.0f - sig));
  grad_gate[i] = f_to_bf(go * u * dsilu);
  grad_up[i]   = f_to_bf(go * s);
}

// grad_bias[c] += sum_r grad_y[r, c].
// One block per column, threads reduce across rows, write accumulates into
// a fp32 grad_bias (master gradient for a bf16 param).
__global__ void k_bias_backward_bf16(const __nv_bfloat16* grad_y,
                                     float* grad_bias,
                                     int64_t rows, int64_t cols) {
  int64_t c = blockIdx.x;
  if (c >= cols) return;
  float acc = 0.f;
  for (int64_t r = threadIdx.x; r < rows; r += blockDim.x) {
    acc += bf_to_f(grad_y[r * cols + c]);
  }
  // Block reduce into one value. 32-lane warp shuffle then shared.
  __shared__ float sh[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) sh[wid] = acc;
  __syncthreads();
  float v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : 0.f;
  if (wid == 0) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) grad_bias[c] += v;
  }
}

inline dim3 grid_1d(int64_t n, int block = 256) {
  return dim3(static_cast<unsigned>((n + block - 1) / block));
}

}  // namespace

void scale_bf16(__nv_bfloat16* y, float a, int64_t n, cudaStream_t s) {
  k_scale_bf16<<<grid_1d(n), 256, 0, s>>>(y, a, n);
}

void axpy_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, float a, int64_t n, cudaStream_t s) {
  k_axpy_bf16<<<grid_1d(n), 256, 0, s>>>(y, x, a, n);
}

void add_bf16(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
              int64_t n, cudaStream_t s) {
  k_add_bf16<<<grid_1d(n), 256, 0, s>>>(out, a, b, n);
}

void add_bias_bf16(__nv_bfloat16* y, const __nv_bfloat16* bias, int64_t rows,
                   int64_t cols, cudaStream_t s) {
  const int64_t total = rows * cols;
  // Cap the grid at ~enough blocks to saturate H100's 132 SMs at occupancy 4
  // (132*4=528). The grid-stride loop handles tensors larger than that.
  const int64_t max_blocks = 1024;
  if ((cols & 1) == 0) {
    // Vectorize: pair BF16 → __nv_bfloat162 (2 BF16 per load/store).
    const int64_t total2 = total >> 1;
    const int64_t cols2  = cols  >> 1;
    const int64_t blocks = std::min<int64_t>(max_blocks, (total2 + 255) / 256);
    auto* y2    = reinterpret_cast<__nv_bfloat162*>(y);
    auto* bias2 = reinterpret_cast<const __nv_bfloat162*>(bias);
    k_add_bias_bf162<<<static_cast<unsigned>(blocks), 256, 0, s>>>(
        y2, bias2, total2, cols2);
  } else {
    const int64_t blocks = std::min<int64_t>(max_blocks, (total + 255) / 256);
    k_add_bias_bf16<<<static_cast<unsigned>(blocks), 256, 0, s>>>(
        y, bias, total, cols);
  }
}

void silu_mul_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate,
                   const __nv_bfloat16* up, int64_t n, cudaStream_t s) {
  k_silu_mul_bf16<<<grid_1d(n), 256, 0, s>>>(out, gate, up, n);
}

void silu_mul_backward_bf16(const __nv_bfloat16* grad_out,
                            const __nv_bfloat16* gate, const __nv_bfloat16* up,
                            __nv_bfloat16* grad_gate, __nv_bfloat16* grad_up,
                            int64_t n, cudaStream_t s) {
  k_silu_mul_bwd_bf16<<<grid_1d(n), 256, 0, s>>>(grad_out, gate, up,
                                                  grad_gate, grad_up, n);
}

void bias_backward_bf16(const __nv_bfloat16* grad_y, float* grad_bias,
                        int64_t rows, int64_t cols, cudaStream_t s) {
  unsigned blocks = static_cast<unsigned>(cols);
  k_bias_backward_bf16<<<blocks, 256, 0, s>>>(grad_y, grad_bias, rows, cols);
}

// BSHD -> BHSD transpose. out[b,h,s,d] = in[b,s,h,d]
// Data is contiguous in both layouts, only the interior two dims swap order.
__global__ void k_transpose_bshd_bhsd(const __nv_bfloat16* in,
                                      __nv_bfloat16* out,
                                      int64_t B, int64_t S, int64_t H, int64_t D) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = B * S * H * D;
  if (tid >= total) return;
  int64_t d = tid % D;
  int64_t h = (tid / D) % H;
  int64_t s = (tid / (D * H)) % S;
  int64_t b = tid / (D * H * S);
  int64_t dst = ((b * H + h) * S + s) * D + d;
  out[dst] = in[tid];
}

__global__ void k_transpose_bhsd_bshd(const __nv_bfloat16* in,
                                      __nv_bfloat16* out,
                                      int64_t B, int64_t S, int64_t H, int64_t D) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = B * H * S * D;
  if (tid >= total) return;
  int64_t d = tid % D;
  int64_t s = (tid / D) % S;
  int64_t h = (tid / (D * S)) % H;
  int64_t b = tid / (D * S * H);
  int64_t dst = ((b * S + s) * H + h) * D + d;
  out[dst] = in[tid];
}

void transpose_bshd_bhsd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int64_t B, int64_t S, int64_t H, int64_t D,
                              cudaStream_t s) {
  int64_t total = B * S * H * D;
  int block = 256;
  unsigned blocks = static_cast<unsigned>((total + block - 1) / block);
  k_transpose_bshd_bhsd<<<blocks, block, 0, s>>>(in, out, B, S, H, D);
}

void transpose_bhsd_bshd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int64_t B, int64_t S, int64_t H, int64_t D,
                              cudaStream_t s) {
  int64_t total = B * H * S * D;
  int block = 256;
  unsigned blocks = static_cast<unsigned>((total + block - 1) / block);
  k_transpose_bhsd_bshd<<<blocks, block, 0, s>>>(in, out, B, S, H, D);
}

// GQA broadcast: write out[b, kv*group + g, s, d] = in[b, kv, s, d].
// One thread per output element. Input is read `group` times, a pure HBM
// replication — safe to run on a non-Hopper stream.
__global__ void k_repeat_kv_heads(const __nv_bfloat16* in, __nv_bfloat16* out,
                                  int64_t B, int64_t Hkv, int64_t S, int64_t D,
                                  int64_t group) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t H = Hkv * group;
  int64_t total = B * H * S * D;
  if (tid >= total) return;
  int64_t d = tid % D;
  int64_t s = (tid / D) % S;
  int64_t h = (tid / (D * S)) % H;
  int64_t b = tid / (D * S * H);
  int64_t kv = h / group;
  int64_t src = (((b * Hkv) + kv) * S + s) * D + d;
  out[tid] = in[src];
}

// GQA reduce: out[b, kv, s, d] = sum_{g=0..group-1} in[b, kv*group + g, s, d].
// One thread per output element. Accumulator is fp32 to preserve precision
// before the bf16 writeback — matches the rest of zwt's "bf16 storage,
// fp32 compute" discipline.
__global__ void k_reduce_kv_heads_sum(const __nv_bfloat16* in, __nv_bfloat16* out,
                                      int64_t B, int64_t Hkv, int64_t S, int64_t D,
                                      int64_t group) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = B * Hkv * S * D;
  if (tid >= total) return;
  int64_t d = tid % D;
  int64_t s = (tid / D) % S;
  int64_t kv = (tid / (D * S)) % Hkv;
  int64_t b = tid / (D * S * Hkv);
  int64_t H = Hkv * group;
  float acc = 0.f;
  int64_t base = ((b * H) + kv * group) * S * D + s * D + d;
  const int64_t stride = S * D;
  #pragma unroll 4
  for (int64_t g = 0; g < group; ++g) acc += __bfloat162float(in[base + g * stride]);
  out[tid] = __float2bfloat16(acc);
}

void repeat_kv_heads_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                          int64_t B, int64_t Hkv, int64_t S, int64_t D,
                          int64_t group, cudaStream_t s) {
  int64_t total = B * Hkv * group * S * D;
  int block = 256;
  unsigned blocks = static_cast<unsigned>((total + block - 1) / block);
  k_repeat_kv_heads<<<blocks, block, 0, s>>>(in, out, B, Hkv, S, D, group);
}

void reduce_kv_heads_sum_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int64_t B, int64_t Hkv, int64_t S, int64_t D,
                              int64_t group, cudaStream_t s) {
  int64_t total = B * Hkv * S * D;
  int block = 256;
  unsigned blocks = static_cast<unsigned>((total + block - 1) / block);
  k_reduce_kv_heads_sum<<<blocks, block, 0, s>>>(in, out, B, Hkv, S, D, group);
}

// Fused SwiGLU on a combined [N, 2H] GEMM output. One thread per (n, i).
// gate = combined[n, i], up = combined[n, H + i]; out[n, i] = silu(g) * u.
// Saves one HBM read per element vs. running two separate silu_mul passes,
// and keeps the gate/up lanes within the same cache line so the load is
// coalesced across threads in a warp.
__global__ void k_silu_mul_gated(__nv_bfloat16* out,
                                 const __nv_bfloat16* combined,
                                 int64_t N, int64_t H) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = N * H;
  if (tid >= total) return;
  int64_t n = tid / H;
  int64_t i = tid % H;
  const __nv_bfloat16* row = combined + n * 2 * H;
  float g = __bfloat162float(row[i]);
  float u = __bfloat162float(row[H + i]);
  float silu = g / (1.0f + __expf(-g));
  out[tid] = __float2bfloat16(silu * u);
}

// Backward. One thread per (n, i) output element; each thread writes to the
// two halves of grad_combined at positions (n, i) and (n, H + i).
__global__ void k_silu_mul_gated_backward(const __nv_bfloat16* grad_out,
                                          const __nv_bfloat16* combined,
                                          __nv_bfloat16* grad_combined,
                                          int64_t N, int64_t H) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = N * H;
  if (tid >= total) return;
  int64_t n = tid / H;
  int64_t i = tid % H;
  const __nv_bfloat16* row = combined + n * 2 * H;
  float g = __bfloat162float(row[i]);
  float u = __bfloat162float(row[H + i]);
  float go = __bfloat162float(grad_out[tid]);
  float sig  = 1.0f / (1.0f + __expf(-g));
  float silu = g * sig;
  float dsilu = sig * (1.0f + g * (1.0f - sig));
  __nv_bfloat16* grow = grad_combined + n * 2 * H;
  grow[i]     = __float2bfloat16(go * u * dsilu);    // grad_gate
  grow[H + i] = __float2bfloat16(go * silu);          // grad_up
}

void silu_mul_gated_bf16(__nv_bfloat16* out, const __nv_bfloat16* combined,
                         int64_t N, int64_t H, cudaStream_t s) {
  int64_t total = N * H;
  int block = 256;
  unsigned blocks = static_cast<unsigned>((total + block - 1) / block);
  k_silu_mul_gated<<<blocks, block, 0, s>>>(out, combined, N, H);
}

void silu_mul_gated_backward_bf16(const __nv_bfloat16* grad_out,
                                  const __nv_bfloat16* combined,
                                  __nv_bfloat16* grad_combined,
                                  int64_t N, int64_t H, cudaStream_t s) {
  int64_t total = N * H;
  int block = 256;
  unsigned blocks = static_cast<unsigned>((total + block - 1) / block);
  k_silu_mul_gated_backward<<<blocks, block, 0, s>>>(grad_out, combined,
                                                     grad_combined, N, H);
}

}  // namespace zwt::ops::k
