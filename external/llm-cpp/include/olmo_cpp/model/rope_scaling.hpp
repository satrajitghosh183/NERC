#pragma once
/**
 * include/olmo_cpp/model/rope_scaling.hpp
 *
 * ─── What "RoPE scaling" is ─────────────────────────────────────────
 *
 * RoPE (rotary position embeddings, see kernels/rope.cu) was designed
 * for a fixed maximum context length. If you trained on 4k tokens
 * and try to do inference on 32k, naive RoPE breaks because the
 * angles for positions > 4k are frequencies it has never seen.
 *
 * Several scaling techniques rescale the per-frequency angles so
 * that long-context inference / fine-tuning works:
 *
 *   - **PositionInterpolation**: linear scale of position indices.
 *     Simple, slightly degrades short-context quality.
 *   - **ABF** (Adjusted Base Frequency): retunes the base frequency.
 *   - **YaRN**: piecewise re-scaling that preserves high-frequency
 *     pairs and only rescales the low-frequency ones — best
 *     long-context preservation.
 *   - **Stepwise**: piecewise constant rescaling between regimes.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/rope.hpp : RoPEBuffers struct that the scalers
 *     populate / mutate.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/rope_scaling.cpp : implementations.
 *   - src/model/rope.cpp : applies the chosen scaling at buffer
 *     construction time, before any forward call.
 *
 * --- Role in training pipeline ---
 *   One-shot at model init when cfg.rope_scaling_type != None.
 *   Off in the quickstart conf (rope_scaling_type=none).
 */

#include "olmo_cpp/model/rope.hpp"

#include <torch/torch.h>
#include <optional>
#include <utility>

namespace olmo_cpp {

/// Absolute Base Frequency scaling: changes theta based on seq length ratio
class ABFScaledRoPEImpl : public torch::nn::Module {
 public:
  ABFScaledRoPEImpl(int64_t head_size, int64_t theta, double scaling_factor,
                    int64_t original_max_len = 2048);
  RoPEBuffers get_buffers(int64_t seq_len, torch::Device device);
  std::pair<torch::Tensor, torch::Tensor> apply(
      torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
      std::optional<int64_t> start_pos = std::nullopt);

 private:
  int64_t dim_, theta_;
  double scaling_factor_;
  int64_t original_max_len_;
  torch::Tensor rotate_half(torch::Tensor x);
  torch::Tensor apply_rotary(torch::Tensor t, torch::Tensor sin,
                              torch::Tensor cos);
};
TORCH_MODULE(ABFScaledRoPE);

/// Position Interpolation: linearly interpolate positions
class PIScaledRoPEImpl : public torch::nn::Module {
 public:
  PIScaledRoPEImpl(int64_t head_size, int64_t theta, double scaling_factor);
  RoPEBuffers get_buffers(int64_t seq_len, torch::Device device);
  std::pair<torch::Tensor, torch::Tensor> apply(
      torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
      std::optional<int64_t> start_pos = std::nullopt);

 private:
  int64_t dim_, theta_;
  double scaling_factor_;
  torch::Tensor rotate_half(torch::Tensor x);
  torch::Tensor apply_rotary(torch::Tensor t, torch::Tensor sin,
                              torch::Tensor cos);
};
TORCH_MODULE(PIScaledRoPE);

/// Stepwise RoPE scaling: discrete frequency adjustments at intervals
class StepwiseScaledRoPEImpl : public torch::nn::Module {
 public:
  StepwiseScaledRoPEImpl(int64_t head_size, int64_t theta,
                          double scaling_factor, int64_t step_size = 256);
  RoPEBuffers get_buffers(int64_t seq_len, torch::Device device);
  std::pair<torch::Tensor, torch::Tensor> apply(
      torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
      std::optional<int64_t> start_pos = std::nullopt);

 private:
  int64_t dim_, theta_, step_size_;
  double scaling_factor_;
  torch::Tensor rotate_half(torch::Tensor x);
  torch::Tensor apply_rotary(torch::Tensor t, torch::Tensor sin,
                              torch::Tensor cos);
};
TORCH_MODULE(StepwiseScaledRoPE);

/// YaRN: Yet another RoPE eXtensioN - combines NTK-aware interpolation with
/// attention scaling
class YaRNScaledRoPEImpl : public torch::nn::Module {
 public:
  YaRNScaledRoPEImpl(int64_t head_size, int64_t theta, double scaling_factor,
                      double beta_fast = 32.0, double beta_slow = 1.0,
                      int64_t original_max_len = 2048);
  RoPEBuffers get_buffers(int64_t seq_len, torch::Device device);
  std::pair<torch::Tensor, torch::Tensor> apply(
      torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
      std::optional<int64_t> start_pos = std::nullopt);

 private:
  int64_t dim_, theta_, original_max_len_;
  double scaling_factor_, beta_fast_, beta_slow_;
  double attn_factor_;
  torch::Tensor compute_yarn_inv_freqs(torch::Device device);
  torch::Tensor rotate_half(torch::Tensor x);
  torch::Tensor apply_rotary(torch::Tensor t, torch::Tensor sin,
                              torch::Tensor cos);
  double find_correction_dim(double num_rotations, int64_t dim, double base,
                              int64_t max_position_embeddings);
  std::pair<double, double> find_correction_range(
      double beta_fast, double beta_slow, int64_t dim, double base,
      int64_t max_position_embeddings);
  torch::Tensor linear_ramp_mask(double min_val, double max_val, int64_t dim,
                                  torch::Device device);
};
TORCH_MODULE(YaRNScaledRoPE);

}  // namespace olmo_cpp
