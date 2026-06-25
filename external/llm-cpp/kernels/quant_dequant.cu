/**
 * kernels/quant_dequant.cu
 *
 * CUDA dequantization kernels for FP8 E4M3 and INT4 AWQ — fast-inference
 * [14]+[15]. Replaces the CPU scalar loops in src/nn/quant.cpp with
 * device-resident dequant so:
 *   1. Quantized weights can stay on the GPU (no D->H roundtrip).
 *   2. Dequant happens in parallel across the whole [V, H] grid instead
 *      of one element at a time on the host.
 *   3. Future quantized GEMV / GEMM kernels can fuse dequant inline.
 *
 * Dequant rules match src/nn/quant.cpp:
 *   FP8 E4M3: per-tensor scale; single FP32 multiplier.
 *     out[i] = e4m3_to_f32(weight[i]) * scale
 *   INT4 AWQ: per-group scale (group_size along H), two INT4 packed per byte.
 *     out[v, h] = signed_q(byte) * scales[v, h / group_size]
 *     where signed_q = (low or high nibble) - 8.
 *
 * E4M3 layout (1 sign + 4 exp + 3 mantissa):
 *   bits 7        : sign
 *   bits 6..3     : exponent (bias 7)
 *   bits 2..0     : mantissa (3 bits)
 * Special: zero (0x00, 0x80), NaN (S 1111 111 — exp=15 mant=7).
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "olmo_cpp/nn/quant.hpp"

namespace olmo_cpp {

namespace {

// Mirrors the host-side e4m3_to_f32 in src/nn/quant.cpp.
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
  uint32_t sign = (b >> 7) & 0x1;
  int      exp4 = (b >> 3) & 0xF;
  uint32_t mant = b & 0x7;
  // Zero
  if (exp4 == 0 && mant == 0) {
    uint32_t bits = sign << 31;
    return __int_as_float(static_cast<int>(bits));
  }
  // NaN: exponent all-1 + mantissa all-1 in E4M3 is canonical NaN.
  if (exp4 == 0xF && mant == 0x7) {
    return __int_as_float(0x7fc00000);  // canonical FP32 NaN
  }
  // Subnormals (exp4 == 0, mant != 0): represent as FP32 of value
  //   sign * 2^(-6) * (mant / 8)   since E4M3 bias=7, smallest normal exp=1
  if (exp4 == 0) {
    float v = static_cast<float>(mant) / 8.0f;  // mant/8 in [1/8, 7/8]
    v *= 1.0f / 64.0f;  // 2^-6
    return sign ? -v : v;
  }
  // Normal: 2^(exp4 - 7) * (1 + mant/8)
  int      exp32 = exp4 - 7 + 127;
  uint32_t m32   = mant << 20;
  uint32_t bits  = (sign << 31) | (static_cast<uint32_t>(exp32) << 23) | m32;
  return __int_as_float(static_cast<int>(bits));
}

// FP8 dequant: out[i] = e4m3_to_f32(packed[i]) * scale.
// Grid-strided so the kernel scales to any V*H without launch tuning.
__global__ void dequant_fp8_kernel(const uint8_t* __restrict__ packed,
                                   float scale,
                                   int64_t N,
                                   float* __restrict__ out) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < N;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[i] = e4m3_to_f32(packed[i]) * scale;
  }
}

// INT4 AWQ dequant: each output is one nibble * its group's FP16 scale.
//   group_size : along H (last dim).
//   packed     : [V, H/2] uint8, low nibble = even h, high nibble = odd h.
//   scales     : [V, H/group_size] FP16.
__global__ void dequant_int4_awq_kernel(const uint8_t* __restrict__ packed,
                                        const __nv_bfloat16* __restrict__ scales_fp16, // (we use bf16 for storage; really fp16 in source)
                                        int64_t V,
                                        int64_t H,
                                        int64_t group_size,
                                        float* __restrict__ out) {
  const int64_t total = V * H;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t v  = i / H;
    const int64_t h  = i - v * H;
    const int64_t bi = v * (H / 2) + (h >> 1);
    const uint8_t byte = packed[bi];
    const int     q   = (h & 1) ? (byte >> 4) : (byte & 0x0F);
    const int     sq  = q - 8;  // unpack to [-8, 7]
    const int64_t si = v * (H / group_size) + (h / group_size);
    out[i] = static_cast<float>(sq) * __bfloat162float(scales_fp16[si]);
  }
}

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

inline int grid_for(int64_t n) {
  int64_t blocks = (n + kThreads - 1) / kThreads;
  if (blocks < 1) blocks = 1;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  return static_cast<int>(blocks);
}

}  // namespace

torch::Tensor dequantize_fp8_cuda(const Fp8Quantized& q) {
  TORCH_CHECK(q.weight.is_cuda(), "dequantize_fp8_cuda: weight must be CUDA");
  c10::cuda::CUDAGuard guard(q.weight.device());

  auto packed = q.weight.contiguous();
  const float scale = q.scale.item<float>();
  const int64_t N = packed.numel();

  auto out = torch::empty(packed.sizes(),
      torch::TensorOptions().dtype(torch::kFloat32).device(q.weight.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dequant_fp8_kernel<<<grid_for(N), kThreads, 0, stream>>>(
      packed.data_ptr<uint8_t>(), scale, N, out.data_ptr<float>());
  return out;
}

torch::Tensor dequantize_int4_awq_cuda(const Int4Quantized& q) {
  TORCH_CHECK(q.weight.is_cuda(), "dequantize_int4_awq_cuda: weight must be CUDA");
  c10::cuda::CUDAGuard guard(q.weight.device());

  auto packed = q.weight.contiguous();
  // Promote scales to BF16 for kernel-friendly read regardless of stored
  // dtype. Using BF16 instead of FP16 inside the kernel because the
  // conversion intrinsic (__bfloat162float) is faster than __half2float on
  // SM_80+. The 2-bit difference in mantissa precision is irrelevant for a
  // weight scale.
  auto scales_bf = q.scales.contiguous().to(torch::kBFloat16);

  const int64_t V = packed.size(0);
  const int64_t H = packed.size(1) * 2;
  const int64_t total = V * H;

  auto out = torch::empty({V, H},
      torch::TensorOptions().dtype(torch::kFloat32).device(q.weight.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dequant_int4_awq_kernel<<<grid_for(total), kThreads, 0, stream>>>(
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(scales_bf.data_ptr<at::BFloat16>()),
      V, H, q.group_size,
      out.data_ptr<float>());
  return out;
}

// ─── Fused dequant-in-GEMV kernels ────────────────────────────────────────
//
// Both kernels compute y[i] = sum_j dequant(W[i, j]) * x[j] WITHOUT
// materializing the dequantized weight in HBM. One block per output
// element; threads in a block stride over the input dim, dequantize on
// the fly, accumulate, and do a block-reduce at the end.
//
// Memory bandwidth analysis (the main reason fused GEMV wins): naive
// path moves 4*in bytes (dequantized fp32 weight) per output element;
// fused FP8 path moves 1*in bytes (packed E4M3); fused INT4 path moves
// 0.5*in bytes (packed nibbles). So FP8 GEMV is 4x faster, INT4 GEMV
// is 8x faster than naive when bandwidth-bound.

namespace {

// Block-reduce sum of one float per thread into thread 0.
__device__ __forceinline__ float block_reduce_sum_f(float v) {
  __shared__ float scratch[32];  // one slot per warp; up to 1024 threads = 32 warps
  const int lane = threadIdx.x & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    const int n_warps = (blockDim.x + 31) >> 5;
    v = (lane < n_warps) ? scratch[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_xor_sync(0xffffffff, v, off);
    }
  }
  return v;  // valid in (warp=0, lane=0)
}

}  // namespace

// FP8 GEMV: y = W_fp8 * x   (per-tensor scale).
//
// Inputs:
//   W_packed : [out, in] uint8 (E4M3 bits)
//   x        : [in]      fp32
//   scale    : scalar fp32 (per-tensor)
// Output:
//   y        : [out] fp32
__global__ void fp8_gemv_kernel(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ x,
    float scale,
    int in_features,
    float*         __restrict__ y) {
  const int row = blockIdx.x;
  const uint8_t* W_row = W + static_cast<int64_t>(row) * in_features;
  float acc = 0.0f;
  for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
    acc += e4m3_to_f32(W_row[j]) * x[j];
  }
  acc *= scale;
  acc = block_reduce_sum_f(acc);
  if (threadIdx.x == 0) y[row] = acc;
}

// INT4 AWQ GEMV: y = W_int4 * x  (per-group scale along in_features).
//
// Inputs:
//   W_packed   : [out, in/2] uint8 (two int4 nibbles per byte)
//   x          : [in]        fp32
//   scales     : [out, in/group_size] bf16 (or fp16 — we cast to bf16 in
//                the wrapper for fast __bfloat162float intrinsic)
//   group_size : along in_features
// Output:
//   y          : [out] fp32
__global__ void int4_awq_gemv_kernel(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ x,
    const __nv_bfloat16* __restrict__ scales,
    int in_features,
    int group_size,
    float*         __restrict__ y) {
  const int row = blockIdx.x;
  const int in_half = in_features >> 1;            // bytes per row
  const int groups_per_row = in_features / group_size;

  const uint8_t* W_row     = W + static_cast<int64_t>(row) * in_half;
  const __nv_bfloat16* S_row = scales + static_cast<int64_t>(row) * groups_per_row;

  float acc = 0.0f;
  // Each thread strides 2-nibbles-per-byte. j indexes BYTES.
  for (int b = threadIdx.x; b < in_half; b += blockDim.x) {
    const uint8_t byte = W_row[b];
    const int q_lo = static_cast<int>(byte & 0x0F) - 8;   // even input index
    const int q_hi = static_cast<int>(byte >> 4)   - 8;   // odd input index
    const int j_lo = b * 2;
    const int j_hi = j_lo + 1;
    // Same group for both nibbles in this byte (since group_size is a
    // multiple of 2 in practice).
    const int g_lo = j_lo / group_size;
    const int g_hi = j_hi / group_size;
    const float s_lo = __bfloat162float(S_row[g_lo]);
    const float s_hi = (g_hi == g_lo) ? s_lo : __bfloat162float(S_row[g_hi]);
    acc += static_cast<float>(q_lo) * s_lo * x[j_lo];
    acc += static_cast<float>(q_hi) * s_hi * x[j_hi];
  }
  acc = block_reduce_sum_f(acc);
  if (threadIdx.x == 0) y[row] = acc;
}

torch::Tensor fp8_gemv_cuda(const Fp8Quantized& w, torch::Tensor x) {
  TORCH_CHECK(w.weight.is_cuda() && x.is_cuda(),
              "fp8_gemv_cuda: weight and x must be CUDA");
  TORCH_CHECK(w.weight.dim() == 2 && x.dim() == 1,
              "fp8_gemv_cuda: W must be 2D, x must be 1D");
  c10::cuda::CUDAGuard guard(w.weight.device());

  const auto W = w.weight.contiguous();
  const auto xc = x.contiguous().to(torch::kFloat32);
  const int64_t out_features = W.size(0);
  const int64_t in_features  = W.size(1);
  TORCH_CHECK(xc.size(0) == in_features, "fp8_gemv_cuda: x/W shape mismatch");

  auto y = torch::empty({out_features},
      torch::TensorOptions().dtype(torch::kFloat32).device(W.device()));
  const float scale = w.scale.item<float>();
  const int threads = 128;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fp8_gemv_kernel<<<static_cast<int>(out_features), threads, 0, stream>>>(
      W.data_ptr<uint8_t>(),
      xc.data_ptr<float>(),
      scale,
      static_cast<int>(in_features),
      y.data_ptr<float>());
  return y;
}

torch::Tensor int4_gemv_cuda(const Int4Quantized& w, torch::Tensor x) {
  TORCH_CHECK(w.weight.is_cuda() && x.is_cuda(),
              "int4_gemv_cuda: weight and x must be CUDA");
  TORCH_CHECK(w.weight.dim() == 2 && x.dim() == 1,
              "int4_gemv_cuda: W must be 2D, x must be 1D");
  c10::cuda::CUDAGuard guard(w.weight.device());

  const auto W = w.weight.contiguous();
  const auto xc = x.contiguous().to(torch::kFloat32);
  const auto S  = w.scales.contiguous().to(torch::kBFloat16);
  const int64_t out_features = W.size(0);
  const int64_t in_features  = W.size(1) * 2;
  TORCH_CHECK(xc.size(0) == in_features, "int4_gemv_cuda: x/W shape mismatch");

  auto y = torch::empty({out_features},
      torch::TensorOptions().dtype(torch::kFloat32).device(W.device()));
  const int threads = 128;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int4_awq_gemv_kernel<<<static_cast<int>(out_features), threads, 0, stream>>>(
      W.data_ptr<uint8_t>(),
      xc.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(S.data_ptr<at::BFloat16>()),
      static_cast<int>(in_features),
      static_cast<int>(w.group_size),
      y.data_ptr<float>());
  return y;
}

}  // namespace olmo_cpp
