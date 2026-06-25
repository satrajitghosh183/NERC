/**
 * src/nn/quant.cpp
 *
 * CPU implementations of weight-only quantization (fast-inference [14][15]).
 *
 * The CPU paths exist as a reference and to make the interface usable
 * without a GPU. The custom dequant-fused GEMV kernels (kernels/quant_gemv.cu)
 * are where the real perf wins live; they're separate drafts.
 *
 * DRAFT. Calibration is per-tensor for FP8 (one scale for the whole
 * matrix) and per-group for INT4 (group size 128 along H). Production
 * AWQ uses an activation-aware per-channel rescale; this draft uses the
 * simpler max-abs scaling.
 */

#include "olmo_cpp/nn/quant.hpp"

#include <torch/torch.h>
#include <cmath>

namespace olmo_cpp {

namespace {

// FP8 E4M3 representable max ≈ 448. Caller scales to fit in this range.
constexpr float kFp8E4M3Max = 448.0f;

// Quantize a float to E4M3 bits (8-bit unsigned packing the FP8 pattern).
// Reference implementation — accuracy not bit-identical to NVIDIA hardware.
uint8_t f32_to_e4m3(float x) {
  if (x == 0.0f) return 0;
  uint32_t bits = *reinterpret_cast<uint32_t*>(&x);
  uint32_t sign = (bits >> 31) & 1;
  int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
  uint32_t mant = bits & 0x7FFFFF;
  // E4M3 bias = 7. Clamp exponent to [-7, 8].
  int e4 = exp + 7;
  if (e4 <= 0)  return static_cast<uint8_t>(sign << 7);  // underflow → ±0
  if (e4 >= 15) return static_cast<uint8_t>((sign << 7) | 0x7E);  // saturate just below ±inf
  uint32_t m3 = mant >> 20;  // top 3 mantissa bits
  return static_cast<uint8_t>((sign << 7) | (e4 << 3) | m3);
}

float e4m3_to_f32(uint8_t b) {
  uint32_t sign = (b >> 7) & 1;
  uint32_t e4   = (b >> 3) & 0xF;
  uint32_t m3   = b & 0x7;
  if (e4 == 0 && m3 == 0) return sign ? -0.0f : 0.0f;
  int exp = static_cast<int>(e4) - 7;
  uint32_t mant = m3 << 20;
  uint32_t bits = (sign << 31) | (static_cast<uint32_t>(exp + 127) << 23) | mant;
  return *reinterpret_cast<float*>(&bits);
}

}  // namespace

Fp8Quantized quantize_fp8(torch::Tensor w) {
  auto wf = w.contiguous().to(torch::kFloat32);
  float max_abs = wf.abs().max().item<float>();
  if (max_abs == 0.0f) max_abs = 1.0f;
  float scale = max_abs / kFp8E4M3Max;

  auto packed = torch::empty(wf.sizes(), torch::TensorOptions().dtype(torch::kUInt8).device(w.device()));
  auto* w_ptr = wf.data_ptr<float>();
  auto* p_ptr = packed.data_ptr<uint8_t>();
  int64_t N = wf.numel();
  float inv_s = 1.0f / scale;
  for (int64_t i = 0; i < N; ++i) p_ptr[i] = f32_to_e4m3(w_ptr[i] * inv_s);

  Fp8Quantized out;
  out.weight = packed;
  out.scale  = torch::full({1}, scale, torch::TensorOptions().dtype(torch::kFloat32).device(w.device()));
  return out;
}

torch::Tensor dequantize_fp8(const Fp8Quantized& q) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (q.weight.is_cuda()) {
    return dequantize_fp8_cuda(q);
  }
#endif
  // CPU fallback (correctness reference). Stays scalar — used only by
  // unit tests / non-CUDA builds where decoded weights are tiny.
  auto packed = q.weight.contiguous().cpu();
  float scale = q.scale.item<float>();
  auto out = torch::empty(packed.sizes(), torch::TensorOptions().dtype(torch::kFloat32));
  auto* p_ptr = packed.data_ptr<uint8_t>();
  auto* o_ptr = out.data_ptr<float>();
  int64_t N = packed.numel();
  for (int64_t i = 0; i < N; ++i) o_ptr[i] = e4m3_to_f32(p_ptr[i]) * scale;
  return out.to(q.weight.device());
}

Int4Quantized quantize_int4_awq(torch::Tensor w, int64_t group_size) {
  auto wf = w.contiguous().to(torch::kFloat32);
  TORCH_CHECK(wf.dim() == 2, "INT4 quant: weight must be 2-D [V, H]");
  const int64_t V = wf.size(0);
  const int64_t H = wf.size(1);
  TORCH_CHECK(H % group_size == 0, "INT4 quant: H must be divisible by group_size");

  // Per-group max-abs scale (real AWQ uses activation-aware rescale; not done here).
  // scales: [V, H/group_size] FP32 → cast to FP16 at end.
  auto wf_grp = wf.view({V, H / group_size, group_size});
  auto max_abs = std::get<0>(wf_grp.abs().max(-1));   // [V, H/group_size]
  // INT4 signed range: [-8, 7]; scale = max_abs / 7.
  auto scales = (max_abs / 7.0f).clamp_min(1e-8f);

  auto wf_q = (wf_grp / scales.unsqueeze(-1)).round().clamp(-8.0f, 7.0f);  // [V, H/g, g]
  // Pack two INT4s per byte. Convert to int8, then nibble-pack.
  auto wf_q_i8 = wf_q.to(torch::kInt8) + 8;  // shift to [0, 15]
  auto packed = torch::empty({V, H / 2}, torch::TensorOptions().dtype(torch::kUInt8).device(w.device()));
  auto src = wf_q_i8.flatten().contiguous();
  auto* sp = src.data_ptr<int8_t>();
  auto* dp = packed.data_ptr<uint8_t>();
  int64_t N = V * H;
  for (int64_t i = 0; i < N; i += 2) {
    uint8_t lo = static_cast<uint8_t>(sp[i]) & 0xF;
    uint8_t hi = (i + 1 < N) ? static_cast<uint8_t>(sp[i + 1]) & 0xF : 0;
    dp[i / 2] = static_cast<uint8_t>(lo | (hi << 4));
  }

  Int4Quantized out;
  out.weight     = packed;
  out.scales     = scales.to(torch::kFloat16);
  out.group_size = group_size;
  return out;
}

torch::Tensor dequantize_int4_awq(const Int4Quantized& q) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (q.weight.is_cuda()) {
    return dequantize_int4_awq_cuda(q);
  }
#endif
  auto packed = q.weight.contiguous().cpu();
  auto scales = q.scales.contiguous().cpu().to(torch::kFloat32);
  const int64_t V = packed.size(0);
  const int64_t H = packed.size(1) * 2;
  const int64_t g = q.group_size;

  auto out = torch::empty({V, H}, torch::TensorOptions().dtype(torch::kFloat32));
  auto* dp = packed.data_ptr<uint8_t>();
  auto* op = out.data_ptr<float>();
  auto sa  = scales.accessor<float, 2>();

  for (int64_t v = 0; v < V; ++v) {
    for (int64_t h = 0; h < H; ++h) {
      int64_t bi = v * (H / 2) + (h / 2);
      uint8_t byte = dp[bi];
      int q_val = (h % 2 == 0) ? (byte & 0xF) : (byte >> 4);
      int signed_q = q_val - 8;  // back to [-8, 7]
      float s = sa[v][h / g];
      op[v * H + h] = static_cast<float>(signed_q) * s;
    }
  }
  return out.to(q.weight.device());
}

torch::Tensor fp8_gemv(const Fp8Quantized& w, torch::Tensor x) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (w.weight.is_cuda()) {
    return fp8_gemv_cuda(w, x);
  }
#endif
  // CPU reference: dequant the entire weight + matmul. Slow but the
  // baseline against which the fused kernel is validated. The "fused"
  // win is on the CUDA path — there, the weight is never materialized
  // in HBM; the kernel reads the packed E4M3 bytes, dequantizes in
  // shared/registers, and accumulates into the GEMV output directly.
  auto W = dequantize_fp8(w);
  return torch::matmul(W, x.contiguous().to(torch::kFloat32));
}

torch::Tensor int4_gemv(const Int4Quantized& w, torch::Tensor x) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (w.weight.is_cuda()) {
    return int4_gemv_cuda(w, x);
  }
#endif
  auto W = dequantize_int4_awq(w);
  return torch::matmul(W, x.contiguous().to(torch::kFloat32));
}

}  // namespace olmo_cpp
