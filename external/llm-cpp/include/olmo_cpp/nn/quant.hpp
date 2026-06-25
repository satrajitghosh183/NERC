/**
 * include/olmo_cpp/nn/quant.hpp
 *
 * Weight-only quantization — fast-inference [14] (FP8) + [15] (INT4).
 *
 * "Weight-only" means: weights stored in low precision (FP8 or INT4),
 * activations stay in BF16/FP16. Decode is bandwidth-bound on weight
 * reads, so halving (FP8) or quartering (INT4) weight bytes nearly
 * halves/quarters decode latency at minor quality cost.
 *
 * Two formats:
 *   - FP8 E4M3 — H100-native (Hopper Tensor Core support); ~0.5% PPL hit.
 *   - INT4 group-quantized (AWQ-style) — group size 128, FP16 scales,
 *     no zero-points. ~1-2% PPL hit; gainable back via group-aware
 *     fine-tune.
 *
 * DRAFT. Both quantize functions take an FP32/BF16 weight tensor and
 * return a packed quantized tensor + scales. Dequantization happens
 * inside a custom GEMV kernel (kernels/quant_gemv.cu, also draft).
 */

#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <utility>

namespace olmo_cpp {

/// FP8 E4M3 weight-only quantization.
/// Per-tensor scale (single FP32 value).
struct Fp8Quantized {
  torch::Tensor weight;   // [V, H] dtype=uint8 (storing E4M3 bits)
  torch::Tensor scale;    // [1] FP32
};

/// AWQ-style INT4 weight-only quantization.
/// Group size = 128 along the H dimension. Two INT4 values packed per byte.
struct Int4Quantized {
  torch::Tensor weight;   // [V, H/2] dtype=uint8 (two INT4 values per byte)
  torch::Tensor scales;   // [V, H/group_size] FP16
  int64_t group_size;     // typically 128
};

/// Quantize FP32/BF16 [V, H] weight to FP8 E4M3.
Fp8Quantized quantize_fp8(torch::Tensor w);

/// Dequantize FP8 [V, H] back to FP32 (mostly for testing/correctness).
torch::Tensor dequantize_fp8(const Fp8Quantized& q);

/// Quantize FP32/BF16 [V, H] weight to INT4 with AWQ-style scaling.
Int4Quantized quantize_int4_awq(torch::Tensor w, int64_t group_size = 128);

/// Dequantize INT4 back to FP32 (for testing).
torch::Tensor dequantize_int4_awq(const Int4Quantized& q);

/// GEMV against an FP8-quantized weight: out = q.dequant() @ x.
/// Activations stay in FP32; weights are dequantized in the GEMV epilogue.
torch::Tensor fp8_gemv(const Fp8Quantized& w, torch::Tensor x);

/// GEMV against an INT4-quantized weight.
torch::Tensor int4_gemv(const Int4Quantized& w, torch::Tensor x);

#ifdef OLMO_HAS_CUDA_KERNELS
/// CUDA dequant kernels — used internally by dequantize_fp8 /
/// dequantize_int4_awq when the input is on a CUDA device. Replace the
/// CPU scalar loops + roundtrip with one device-resident dispatch.
torch::Tensor dequantize_fp8_cuda(const Fp8Quantized& q);
torch::Tensor dequantize_int4_awq_cuda(const Int4Quantized& q);

/// Fused dequant-in-GEMV: y = W * x where W is FP8 / INT4 packed.
/// The CUDA kernels never materialize a dequantized W in HBM. One CUDA
/// block per output row; threads stride over the input dim, dequant on
/// the fly, accumulate, and block-reduce the partial sum.
torch::Tensor fp8_gemv_cuda(const Fp8Quantized& w, torch::Tensor x);
torch::Tensor int4_gemv_cuda(const Int4Quantized& w, torch::Tensor x);
#endif

}  // namespace olmo_cpp
