/**
 * src/model/rope_scaling.cpp
 *
 * Implements four RoPE *scaling* variants used to extend a model's effective
 * context length beyond what it saw during pre-training:
 *   - ABFScaledRoPE      : Absolute Base Frequency — scale theta by a factor.
 *   - PIScaledRoPE       : Position Interpolation (Chen et al., 2023).
 *   - StepwiseScaledRoPE : NTK-aware stepwise interpolation (per-dim quant).
 *   - YaRNScaledRoPE     : Peng et al. 2023, NTK-by-parts + attention factor.
 * Each variant overrides only the inv_freq construction (or position scaling)
 * and reuses the same rotate_half + apply_rotary mechanics as the standard
 * RoPE in src/model/rope.cpp. Selection is config-driven for context-window
 * extension experiments.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/rope_scaling.hpp: declares the four *Impl classes
 *     wrapped by their TORCH_MODULE holders, plus RoPEBuffers reuse.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - Direct callers not located via quick grep. Wired in by builder code
 *     that selects between RotaryEmbedding and one of these scaled variants
 *     based on cfg.rope_scaling_type during model construction (see
 *     src/model/transformer.cpp / fused_transformer.cpp RotaryEmbedding
 *     instantiation; the scaled variants substitute in there for long-ctx
 *     experiments).
 *
 * --- Role in training pipeline ---
 *   When enabled, replaces the standard RoPE inside attention. Generates
 *   sin/cos buffers once per (seq_len, device) and applies them to (q, k)
 *   on every step. Used both for fine-tuning longer-context models from a
 *   shorter-context checkpoint and for inference-only context extension.
 */
#include "olmo_cpp/model/rope_scaling.hpp"

#include <cmath>

namespace olmo_cpp {

// ===========================================================================
// Shared helpers (rotate_half / apply_rotary are identical across variants)
// ===========================================================================

// Each class has its own private copies to avoid cross-class coupling.
// The implementations are identical to RotaryEmbeddingImpl.

// ===========================================================================
// Helper: build sin/cos buffers from inv_freq and position sequence
// ===========================================================================

/// Build sin/cos buffers from a per-dim inv_freq and a per-position seq.
/// Used by every variant; only the inv_freq formula differs across variants.
static RoPEBuffers buffers_from_inv_freq(const torch::Tensor& inv_freq,
                                          const torch::Tensor& seq) {
  // inv_freq: (half_dim,)   seq: (seq_len,)
  auto freqs = seq.unsqueeze(1) * inv_freq.unsqueeze(0);  // (seq_len, half_dim)
  // Duplicate halves so the result has the full head_dim layout that
  // rotate_half + (cos, sin) decomposition expects.
  auto positions = torch::cat({freqs, freqs}, -1);         // (seq_len, dim)
  RoPEBuffers bufs;
  bufs.pos_sin = positions.sin();
  bufs.pos_cos = positions.cos();
  return bufs;
}

/// Shared apply() body. Each variant passes its own rotate_half /
/// apply_rotary as lambdas so the helper can stay variant-agnostic.
static std::pair<torch::Tensor, torch::Tensor> apply_rope_to_qk(
    torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
    std::optional<int64_t> start_pos,
    // function pointers for rotate_half / apply_rotary kept as lambdas
    std::function<torch::Tensor(torch::Tensor)> rotate_half_fn,
    std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor)>
        apply_rotary_fn) {
  // Same prefill / decode position math as RotaryEmbeddingImpl::apply.
  auto q_len = q.size(2);
  auto k_len = k.size(2);
  int64_t q_abs_start = start_pos ? *start_pos : (k_len - q_len);
  int64_t k_abs_start = start_pos ? *start_pos : 0;

  // Pick the source buffer in the right dtype. If q/k are already in the
  // master compute dtype (typically FP32), use pos_sin/pos_cos directly.
  // Otherwise lazily cast the *whole* buffer once and slice from the
  // cached copy on every subsequent call. The cast cache is shared across
  // all layer-views via shared_ptr so the first layer to hit a new dtype
  // fills it for the rest of the model.
  const auto q_st = q.scalar_type();
  const auto& src_sin = (q_st == bufs.pos_sin.scalar_type())
                            ? bufs.pos_sin
                            : ([&]() -> const torch::Tensor& {
                                auto& c = *bufs.cast;
                                if (c.dtype != q_st || !c.pos_sin_cast.defined()) {
                                  c.pos_sin_cast = bufs.pos_sin.to(q_st);
                                  c.pos_cos_cast = bufs.pos_cos.to(q_st);
                                  c.dtype        = q_st;
                                }
                                return c.pos_sin_cast;
                              })();
  const auto& src_cos = (q_st == bufs.pos_cos.scalar_type())
                            ? bufs.pos_cos
                            : bufs.cast->pos_cos_cast;

  // Slice the buffer rows for our position window and unsqueeze leading
  // dims so they broadcast across batch and head dimensions.
  auto sin_q = src_sin.slice(0, q_abs_start, q_abs_start + q_len)
                   .unsqueeze(0).unsqueeze(0);
  auto cos_q = src_cos.slice(0, q_abs_start, q_abs_start + q_len)
                   .unsqueeze(0).unsqueeze(0);
  auto sin_k = src_sin.slice(0, k_abs_start, k_abs_start + k_len)
                   .unsqueeze(0).unsqueeze(0);
  auto cos_k = src_cos.slice(0, k_abs_start, k_abs_start + k_len)
                   .unsqueeze(0).unsqueeze(0);

  auto q_rot = apply_rotary_fn(q, sin_q, cos_q);
  auto k_rot = apply_rotary_fn(k, sin_k, cos_k);
  return {q_rot, k_rot};
}

// ===========================================================================
// ABFScaledRoPE — Absolute Base Frequency scaling
// Trick: replace theta by theta * scaling_factor everywhere, which lengthens
// every rotational wavelength uniformly. Equivalent to a "linear" rescale of
// frequencies; simple and works well when only mildly extending context.
// ===========================================================================

/// Construct an ABF-scaled RoPE.
/// scaling_factor   : multiplier on theta. Empirical values are 4-32x.
/// original_max_len : max position the underlying model was pre-trained for
///                     (informational only here).
ABFScaledRoPEImpl::ABFScaledRoPEImpl(int64_t head_size, int64_t theta,
                                      double scaling_factor,
                                      int64_t original_max_len)
    : dim_(head_size),
      theta_(theta),
      scaling_factor_(scaling_factor),
      original_max_len_(original_max_len) {}

/// Same rotate_half as standard RoPE; duplicated to keep classes decoupled.
torch::Tensor ABFScaledRoPEImpl::rotate_half(torch::Tensor x) {
  auto chunks = x.chunk(2, -1);
  return torch::cat({-chunks[1], chunks[0]}, -1);
}

/// Apply rotation: y = t * cos + rotate_half(t) * sin. ATen path (no backend
/// dispatch), to_dtype keeps the result in the input dtype after broadcasting.
torch::Tensor ABFScaledRoPEImpl::apply_rotary(torch::Tensor t,
                                                torch::Tensor sin,
                                                torch::Tensor cos) {
  return (t * cos + rotate_half(t) * sin).to(t.dtype());
}

RoPEBuffers ABFScaledRoPEImpl::get_buffers(int64_t seq_len,
                                            torch::Device device) {
  // Scale the base frequency: theta_scaled = theta * scaling_factor
  auto half_dim = dim_ / 2;
  double scaled_theta = static_cast<double>(theta_) * scaling_factor_;
  auto indices = torch::arange(
      half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  indices = indices * 2.0 / static_cast<double>(dim_);
  auto inv_freq = 1.0 / torch::pow(scaled_theta, indices);

  auto seq = torch::arange(
      seq_len, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  return buffers_from_inv_freq(inv_freq, seq);
}

std::pair<torch::Tensor, torch::Tensor> ABFScaledRoPEImpl::apply(
    torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
    std::optional<int64_t> start_pos) {
  return apply_rope_to_qk(
      q, k, bufs, start_pos,
      [this](torch::Tensor x) { return rotate_half(x); },
      [this](torch::Tensor t, torch::Tensor s, torch::Tensor c) {
        return apply_rotary(t, s, c);
      });
}

// ===========================================================================
// PIScaledRoPE — Position Interpolation
// ===========================================================================

PIScaledRoPEImpl::PIScaledRoPEImpl(int64_t head_size, int64_t theta,
                                    double scaling_factor)
    : dim_(head_size), theta_(theta), scaling_factor_(scaling_factor) {}

torch::Tensor PIScaledRoPEImpl::rotate_half(torch::Tensor x) {
  auto chunks = x.chunk(2, -1);
  return torch::cat({-chunks[1], chunks[0]}, -1);
}

torch::Tensor PIScaledRoPEImpl::apply_rotary(torch::Tensor t,
                                              torch::Tensor sin,
                                              torch::Tensor cos) {
  return (t * cos + rotate_half(t) * sin).to(t.dtype());
}

RoPEBuffers PIScaledRoPEImpl::get_buffers(int64_t seq_len,
                                           torch::Device device) {
  // Standard inv_freq, but positions are scaled down by factor.
  auto half_dim = dim_ / 2;
  auto indices = torch::arange(
      half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  indices = indices * 2.0 / static_cast<double>(dim_);
  auto inv_freq = 1.0 / torch::pow(static_cast<double>(theta_), indices);

  // Scale positions: positions = arange(seq_len) / scaling_factor
  auto seq = torch::arange(
      seq_len, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  seq = seq / scaling_factor_;

  return buffers_from_inv_freq(inv_freq, seq);
}

std::pair<torch::Tensor, torch::Tensor> PIScaledRoPEImpl::apply(
    torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
    std::optional<int64_t> start_pos) {
  return apply_rope_to_qk(
      q, k, bufs, start_pos,
      [this](torch::Tensor x) { return rotate_half(x); },
      [this](torch::Tensor t, torch::Tensor s, torch::Tensor c) {
        return apply_rotary(t, s, c);
      });
}

// ===========================================================================
// StepwiseScaledRoPE — Discrete frequency adjustments at intervals
// ===========================================================================

StepwiseScaledRoPEImpl::StepwiseScaledRoPEImpl(int64_t head_size,
                                                int64_t theta,
                                                double scaling_factor,
                                                int64_t step_size)
    : dim_(head_size),
      theta_(theta),
      step_size_(step_size),
      scaling_factor_(scaling_factor) {}

torch::Tensor StepwiseScaledRoPEImpl::rotate_half(torch::Tensor x) {
  auto chunks = x.chunk(2, -1);
  return torch::cat({-chunks[1], chunks[0]}, -1);
}

torch::Tensor StepwiseScaledRoPEImpl::apply_rotary(torch::Tensor t,
                                                     torch::Tensor sin,
                                                     torch::Tensor cos) {
  return (t * cos + rotate_half(t) * sin).to(t.dtype());
}

RoPEBuffers StepwiseScaledRoPEImpl::get_buffers(int64_t seq_len,
                                                  torch::Device device) {
  // For each dimension i: adj = floor(2*i/d * step_size) / step_size
  // inv_freq[i] = 1.0 / (theta^adj * factor)
  auto half_dim = dim_ / 2;
  auto indices = torch::arange(
      half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  // Compute continuous exponent: 2*i / d
  auto exponent = indices * 2.0 / static_cast<double>(dim_);
  // Quantize to step_size intervals
  auto adj = torch::floor(exponent * static_cast<double>(step_size_)) /
             static_cast<double>(step_size_);
  auto inv_freq =
      1.0 / (torch::pow(static_cast<double>(theta_), adj) * scaling_factor_);

  auto seq = torch::arange(
      seq_len, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  return buffers_from_inv_freq(inv_freq, seq);
}

std::pair<torch::Tensor, torch::Tensor> StepwiseScaledRoPEImpl::apply(
    torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
    std::optional<int64_t> start_pos) {
  return apply_rope_to_qk(
      q, k, bufs, start_pos,
      [this](torch::Tensor x) { return rotate_half(x); },
      [this](torch::Tensor t, torch::Tensor s, torch::Tensor c) {
        return apply_rotary(t, s, c);
      });
}

// ===========================================================================
// YaRNScaledRoPE — Yet another RoPE eXtensioN
// ===========================================================================

YaRNScaledRoPEImpl::YaRNScaledRoPEImpl(int64_t head_size, int64_t theta,
                                        double scaling_factor,
                                        double beta_fast, double beta_slow,
                                        int64_t original_max_len)
    : dim_(head_size),
      theta_(theta),
      original_max_len_(original_max_len),
      scaling_factor_(scaling_factor),
      beta_fast_(beta_fast),
      beta_slow_(beta_slow) {
  // Attention scaling factor: 0.1 * ln(factor) + 1
  attn_factor_ = 0.1 * std::log(scaling_factor_) + 1.0;
}

torch::Tensor YaRNScaledRoPEImpl::rotate_half(torch::Tensor x) {
  auto chunks = x.chunk(2, -1);
  return torch::cat({-chunks[1], chunks[0]}, -1);
}

torch::Tensor YaRNScaledRoPEImpl::apply_rotary(torch::Tensor t,
                                                 torch::Tensor sin,
                                                 torch::Tensor cos) {
  return (t * cos + rotate_half(t) * sin).to(t.dtype());
}

double YaRNScaledRoPEImpl::find_correction_dim(
    double num_rotations, int64_t dim, double base,
    int64_t max_position_embeddings) {
  // Find the dimension d such that the wavelength at that dimension equals
  // num_rotations * 2 * pi * max_position_embeddings.
  // wavelength(d) = 2*pi * base^(2d/dim)
  // Solving: d = dim * log(max_position_embeddings / (num_rotations * 2*pi))
  //              / (2 * log(base))
  return static_cast<double>(dim) *
         std::log(static_cast<double>(max_position_embeddings) /
                  (num_rotations * 2.0 * M_PI)) /
         (2.0 * std::log(base));
}

std::pair<double, double> YaRNScaledRoPEImpl::find_correction_range(
    double beta_fast, double beta_slow, int64_t dim, double base,
    int64_t max_position_embeddings) {
  double low = find_correction_dim(beta_fast, dim, base,
                                    max_position_embeddings);
  double high = find_correction_dim(beta_slow, dim, base,
                                     max_position_embeddings);
  // Clamp to valid range [0, dim/2 - 1]
  low = std::max(std::floor(low), 0.0);
  high = std::min(std::ceil(high), static_cast<double>(dim / 2 - 1));
  return {low, high};
}

torch::Tensor YaRNScaledRoPEImpl::linear_ramp_mask(double min_val,
                                                     double max_val,
                                                     int64_t dim,
                                                     torch::Device device) {
  // Linearly ramp from 0 to 1 over [min_val, max_val].
  // Values below min_val -> 0 (full interpolation)
  // Values above max_val -> 1 (no interpolation)
  auto half_dim = dim / 2;
  auto indices = torch::arange(
      half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  if (max_val == min_val) {
    // Avoid division by zero; all dimensions get mask = 1
    return torch::ones(
        half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }
  auto mask = (indices - min_val) / (max_val - min_val);
  return torch::clamp(mask, 0.0, 1.0);
}

torch::Tensor YaRNScaledRoPEImpl::compute_yarn_inv_freqs(
    torch::Device device) {
  auto half_dim = dim_ / 2;
  auto base = static_cast<double>(theta_);

  // Standard inv_freq
  auto indices = torch::arange(
      half_dim, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  indices = indices * 2.0 / static_cast<double>(dim_);
  auto inv_freq = 1.0 / torch::pow(base, indices);

  // Compute correction range
  auto [low, high] =
      find_correction_range(beta_fast_, beta_slow_, dim_, base,
                            original_max_len_);

  // Build interpolation mask: 0 = full interpolation, 1 = no interpolation
  auto mask = linear_ramp_mask(low, high, dim_, device);

  // Interpolated frequencies:
  // inv_freq_yarn = (1 - mask) * (inv_freq / factor) + mask * inv_freq
  auto inv_freq_interpolated = inv_freq / scaling_factor_;
  auto inv_freq_yarn =
      (1.0 - mask) * inv_freq_interpolated + mask * inv_freq;

  return inv_freq_yarn;
}

RoPEBuffers YaRNScaledRoPEImpl::get_buffers(int64_t seq_len,
                                             torch::Device device) {
  auto inv_freq = compute_yarn_inv_freqs(device);
  auto seq = torch::arange(
      seq_len, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto bufs = buffers_from_inv_freq(inv_freq, seq);

  // Apply attention scaling factor to sin/cos buffers.
  // This is equivalent to scaling the attention logits by attn_factor_.
  // In practice YaRN scales the sin/cos so that the dot products are
  // appropriately adjusted.
  // Note: some implementations apply attn_factor_ directly to attention
  // scores; here we bake it into the embeddings for simplicity.
  // Scaling sin and cos by sqrt(attn_factor_) so that the dot product
  // q^T k gets multiplied by attn_factor_.
  // However, the canonical YaRN paper applies it as a temperature to
  // attention logits. We store it and let callers decide. For the default
  // path we leave sin/cos unscaled and expose attn_factor_ for external use.
  return bufs;
}

std::pair<torch::Tensor, torch::Tensor> YaRNScaledRoPEImpl::apply(
    torch::Tensor q, torch::Tensor k, const RoPEBuffers& bufs,
    std::optional<int64_t> start_pos) {
  return apply_rope_to_qk(
      q, k, bufs, start_pos,
      [this](torch::Tensor x) { return rotate_half(x); },
      [this](torch::Tensor t, torch::Tensor s, torch::Tensor c) {
        return apply_rotary(t, s, c);
      });
}

}  // namespace olmo_cpp
