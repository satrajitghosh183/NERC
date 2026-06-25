#pragma once

/**
 * include/olmo_cpp/float8/float8.hpp
 *
 * Float8 (FP8) emulation utilities for memory-efficient training. Two formats
 * are supported: E4M3 (used for forward activations / weights) and E5M2 (used
 * for backward gradients which have wider dynamic range). The path here is
 * an *emulation* — quantize then dequantize before matmul — because real
 * FP8 matmul requires hardware support (Hopper or newer) plus cuBLASLt FP8
 * kernels. Even so, training with FP8 quantization noise is a useful proxy
 * for the systems-research question "how robust is OLMo to low-precision
 * arithmetic?", and the API mirrors what a hardware-FP8 path would expose.
 *
 * --- Includes from this project ---
 *   - none beyond LibTorch.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/float8/float8.cpp: full implementation.
 *   (No direct trainer callers in src/main or src/train located via quick
 *   grep; the module is opt-in.)
 *
 * --- Role in training pipeline ---
 *   Provides drop-in `Float8Linear` modules and standalone tensor
 *   quantize/dequantize helpers. When `enabled_=false`, `Float8Linear`
 *   degrades to a plain `torch::nn::Linear`, so the layer can stay wired up
 *   even when FP8 is not in use.
 */

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// Two IEEE-style FP8 layouts. E4M3 has lower max (~448) but more mantissa
/// bits — better for the forward pass. E5M2 has wider range (~57344) — used
/// for gradients that may otherwise overflow.
enum class Float8Format {
  E4M3,  // 4 exponent, 3 mantissa (forward pass)
  E5M2   // 5 exponent, 2 mantissa (backward pass)
};

/// Rolling-window amax tracker for "delayed scaling" FP8. Each iteration
/// records the abs-max of the input; the scale factor used at quantization
/// time is the historical max divided into the format's max representable
/// value. This avoids a per-iter sync on a single scalar amax.
struct Float8ScaleState {
  torch::Tensor amax_history;   ///< circular buffer of |x|.max() values
  int64_t history_len;          ///< window length (default 16)
  int64_t current_idx = 0;      ///< monotonically increasing write index

  explicit Float8ScaleState(int64_t history_len = 16);

  /// Push the current tensor's absolute max into the history buffer.
  /// `current_idx % history_len` chooses the slot (circular overwrite).
  void update(const torch::Tensor& tensor);

  /// Return scale = max_repr(format) / max(amax_history). Falls back to
  /// scale=1 when the history is all zeros (typical at step 0).
  torch::Tensor get_scale(Float8Format format) const;
};

/// Container for an FP8-quantized tensor. Real storage is uint8 (the bit
/// pattern), with a separate fp32 scale and metadata needed to reconstruct
/// the original shape and dtype on dequantize.
struct Float8Tensor {
  torch::Tensor data;                    ///< uint8 codes, packed
  torch::Tensor scale;                   ///< fp32 scale (scalar or per-block)
  Float8Format format;                   ///< E4M3 or E5M2
  std::vector<int64_t> original_shape;   ///< shape pre-flatten/pre-pad
  torch::ScalarType original_dtype;      ///< original dtype for round-trip

  /// Reverse the quantization: uint8 -> fp32 -> /scale -> cast to `dtype`,
  /// reshaping back to `original_shape`.
  torch::Tensor dequantize(torch::ScalarType dtype = torch::kBFloat16) const;
};

/// One-shot tensor->FP8 quantizer. If `state` is provided, scaling uses the
/// rolling amax (delayed scaling); otherwise the per-call abs-max is used.
Float8Tensor quantize_to_float8(const torch::Tensor& tensor, Float8Format format,
                                 Float8ScaleState* state = nullptr);

/// Drop-in FP8 linear layer. Wraps a plain `torch::nn::Linear` and, when
/// `enabled_=true`, quantizes both inputs and weights to E4M3 around the
/// matmul to emulate FP8 accuracy. Uses a straight-through estimator so
/// gradients still flow through the unquantized path.
class Float8LinearImpl : public torch::nn::Module {
 public:
  /// `bias=false` matches OLMo's default (no biases on linear layers).
  Float8LinearImpl(int64_t in_features, int64_t out_features, bool bias = false);

  /// Forward returning logits/projection. STE is used internally so the
  /// backward pass sees the original (un-quantized) inputs and weights.
  torch::Tensor forward(torch::Tensor input);

  /// Toggle FP8 emulation at runtime (e.g. disable for the first warm-up
  /// step or for a sensitivity sweep).
  void set_float8_enabled(bool enabled) { enabled_ = enabled; }
  /// Read back current quantization state.
  bool float8_enabled() const { return enabled_; }

 private:
  torch::nn::Linear inner_;          ///< the actual fp32/bf16 weights
  bool enabled_ = true;              ///< false => behave as plain Linear
  Float8ScaleState input_scale_;     ///< rolling amax for activations
  Float8ScaleState weight_scale_;    ///< rolling amax for the weight matrix
};
TORCH_MODULE(Float8Linear);

/// Configuration for the microscaling FP8 (MXFP8) variant: each contiguous
/// `block_size` chunk of the flattened tensor gets its own scale.
struct MXFP8Config {
  int64_t block_size = 32;  ///< OCP MXFP8 spec uses 32; smaller = finer scales
};

/// MXFP8 quantizer: flattens, pads to a multiple of `block_size`, computes a
/// per-block amax-derived scale, and stores per-block scales alongside data.
Float8Tensor quantize_mxfp8(const torch::Tensor& tensor, const MXFP8Config& config);

/// Drop-in replacement for `torch::nn::functional::linear` that emulates
/// FP8 (E4M3) at the inputs AND the weight via STE quantize-then-
/// dequantize-then-matmul. Forward sees the quantized activations and
/// weights; backward passes the gradient through unchanged (the standard
/// straight-through estimator trick: `x + (x_dq - x).detach()` matches
/// `x` in value if we re-arrange but propagates gradient as identity).
///
/// `input_scale` and `weight_scale` are rolling-window amax trackers
/// owned by the caller (one pair per Linear layer). They get updated
/// in-place each call, supplying the delayed-scaling scale factor that
/// would be plumbed to hardware FP8 on Hopper.
///
/// Use this free function rather than swapping `torch::nn::Linear` for
/// `Float8Linear` so checkpoint state_dict keys remain unchanged — the
/// caller still holds a plain Linear and just routes its weight + bias
/// through here when `cfg.use_float8 == true`.
torch::Tensor float8_linear_emulated(
    torch::Tensor input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    Float8ScaleState& input_scale,
    Float8ScaleState& weight_scale);

}  // namespace olmo_cpp
