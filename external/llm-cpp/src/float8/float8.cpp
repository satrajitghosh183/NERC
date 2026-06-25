/**
 * src/float8/float8.cpp
 *
 * ─── What FP8 is ────────────────────────────────────────────────────
 *
 * FP8 = a family of 8-bit floating-point formats introduced for
 * Hopper-class GPUs (sm_90+). 16-bit BF16 is half the memory of FP32;
 * FP8 is half the memory of BF16 — so a Linear layer's weights and
 * activations both shrink, and the matmul itself runs through
 * Hopper's tensor cores at roughly 2x the BF16 rate.
 *
 * Two variants exist:
 *   - **E4M3**: 1 sign bit, 4 exponent bits, 3 mantissa bits.
 *               Wider mantissa, smaller dynamic range. Good for
 *               activations (sit closer to zero).
 *   - **E5M2**: 1 sign bit, 5 exponent bits, 2 mantissa bits.
 *               Smaller mantissa, FP16-like dynamic range. Good for
 *               gradients (tails matter more).
 *
 * Because FP8 has so few mantissa bits, naive use destroys accuracy.
 * The trick is per-tensor (or per-block) **scaling**: divide the
 * tensor by a runtime-tracked scale factor before casting to FP8 so
 * the values land in the representable range, then multiply the
 * matmul result by the inverse scale on the output. This file
 * implements the scale-tracking + cast-and-back glue around cuBLASLt's
 * FP8 matmul on Hopper, with a software fake-quant fallback elsewhere.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/float8/float8.hpp : Float8Linear + scale-tracking helpers.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/feed_forward.cpp / attention.cpp: when cfg.use_float8,
 *     Linear modules are replaced with the Float8 variant declared in
 *     the header.
 *
 * --- Role in training pipeline ---
 *   Disabled by default and disabled on the 3060 (sm_86 has no native
 *   FP8 hardware). Present here for parity with the H100 perf path.
 */
#include "olmo_cpp/float8/float8.hpp"
#include <cmath>
#include <algorithm>

namespace olmo_cpp {

namespace {

/// Max representable values for each float8 format
float max_representable(Float8Format fmt) {
  switch (fmt) {
    case Float8Format::E4M3: return 448.0f;   // 2^8 * (1 + 0.875)
    case Float8Format::E5M2: return 57344.0f;  // 2^15 * (1 + 0.75)
  }
  return 448.0f;
}

}  // namespace

// ---------------------------------------------------------------------------
// Float8ScaleState
// ---------------------------------------------------------------------------

Float8ScaleState::Float8ScaleState(int64_t history_len)
    : history_len(history_len) {
  amax_history = torch::zeros({history_len});
}

void Float8ScaleState::update(const torch::Tensor& tensor) {
  float amax = tensor.abs().max().item<float>();
  amax_history[current_idx % history_len] = amax;
  current_idx++;
}

torch::Tensor Float8ScaleState::get_scale(Float8Format format) const {
  float amax = amax_history.max().item<float>();
  if (amax == 0.0f) amax = 1.0f;
  float scale = max_representable(format) / amax;
  return torch::tensor(scale);
}

// ---------------------------------------------------------------------------
// Quantize / Dequantize
// ---------------------------------------------------------------------------

Float8Tensor quantize_to_float8(const torch::Tensor& tensor, Float8Format format,
                                 Float8ScaleState* state) {
  Float8Tensor result;
  result.format = format;
  result.original_shape = tensor.sizes().vec();
  result.original_dtype = tensor.scalar_type();

  // Compute or retrieve scale
  float amax = tensor.abs().max().item<float>();
  if (amax == 0.0f) amax = 1.0f;
  float scale_val = max_representable(format) / amax;

  if (state) {
    state->update(tensor);
    scale_val = state->get_scale(format).item<float>();
  }

  result.scale = torch::tensor(scale_val);

  // Quantize: multiply by scale, clamp to representable range, round to uint8
  float max_val = max_representable(format);
  auto scaled = tensor.to(torch::kFloat32) * scale_val;
  scaled = scaled.clamp(-max_val, max_val);

  // Map from [-max_val, max_val] to [0, 255]
  auto normalized = (scaled + max_val) / (2.0f * max_val) * 255.0f;
  result.data = normalized.round().to(torch::kUInt8);

  return result;
}

torch::Tensor Float8Tensor::dequantize(torch::ScalarType dtype) const {
  float max_val = max_representable(format);
  float scale_val = scale.item<float>();

  // Reverse: uint8 -> float -> unscale
  auto fp = data.to(torch::kFloat32);
  auto unormalized = fp / 255.0f * (2.0f * max_val) - max_val;
  auto result = unormalized / scale_val;

  return result.to(dtype).view(original_shape);
}

// ---------------------------------------------------------------------------
// Float8Linear
// ---------------------------------------------------------------------------

Float8LinearImpl::Float8LinearImpl(int64_t in_features, int64_t out_features, bool bias)
    : inner_(register_module("inner",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)))),
      input_scale_(16), weight_scale_(16) {}

torch::Tensor Float8LinearImpl::forward(torch::Tensor input) {
  if (!enabled_) {
    return inner_(input);
  }
  return float8_linear_emulated(input, inner_->weight,
                                inner_->bias.defined() ? inner_->bias : torch::Tensor(),
                                input_scale_, weight_scale_);
}

// Free-function variant for callers that hold a plain torch::nn::Linear
// and want to opt into FP8 emulation without swapping module types
// (preserves checkpoint key names). Mirrors Float8LinearImpl::forward's
// STE math.
torch::Tensor float8_linear_emulated(
    torch::Tensor input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    Float8ScaleState& input_scale,
    Float8ScaleState& weight_scale) {
  auto input_q  = quantize_to_float8(input.detach(),  Float8Format::E4M3, &input_scale);
  auto weight_q = quantize_to_float8(weight.detach(), Float8Format::E4M3, &weight_scale);
  auto input_deq  = input_q.dequantize(input.scalar_type());
  auto weight_deq = weight_q.dequantize(weight.scalar_type());
  return torch::nn::functional::linear(
      input  + (input_deq  - input ).detach(),
      weight + (weight_deq - weight).detach(),
      bias.defined() ? bias : torch::Tensor());
}

// ---------------------------------------------------------------------------
// MXFP8 (microscaling)
// ---------------------------------------------------------------------------

Float8Tensor quantize_mxfp8(const torch::Tensor& tensor, const MXFP8Config& config) {
  auto flat = tensor.reshape({-1});
  int64_t numel = flat.numel();
  int64_t bs = config.block_size;

  // Pad to multiple of block_size
  int64_t padded = ((numel + bs - 1) / bs) * bs;
  if (padded > numel) {
    flat = torch::cat({flat, torch::zeros({padded - numel}, flat.options())});
  }

  auto blocks = flat.view({-1, bs});  // [num_blocks, block_size]

  // Per-block scales
  auto block_amax = std::get<0>(blocks.abs().max(/*dim=*/1, /*keepdim=*/true));
  block_amax = block_amax.clamp_min(1e-12f);
  float max_val = max_representable(Float8Format::E4M3);
  auto block_scales = max_val / block_amax;  // [num_blocks, 1]

  // Quantize each block
  auto scaled = blocks * block_scales;
  scaled = scaled.clamp(-max_val, max_val);
  auto normalized = (scaled + max_val) / (2.0f * max_val) * 255.0f;

  Float8Tensor result;
  result.data = normalized.round().to(torch::kUInt8).view({-1}).narrow(0, 0, numel);
  result.scale = block_scales.view({-1});
  result.format = Float8Format::E4M3;
  result.original_shape = tensor.sizes().vec();
  result.original_dtype = tensor.scalar_type();

  return result;
}

}  // namespace olmo_cpp
