#pragma once

#include "zwt/layers/linear.hpp"
#include "zwt/layers/module.hpp"

namespace zwt {

// Multi-head self-attention with RoPE and causal mask. Supports both standard
// MHA (n_kv_heads == n_heads) and Grouped-Query Attention (n_kv_heads < n_heads
// with n_heads % n_kv_heads == 0).
//
// Forward: x [B, S, d_model] -> y [B, S, d_model].
//   Q projected to [B, S, n_heads   * head_dim]
//   K projected to [B, S, n_kv_heads * head_dim]
//   V projected to [B, S, n_kv_heads * head_dim]
//   RoPE applied to Q (n_heads-wide) and K (n_kv_heads-wide).
//   Head-major transpose to [B, *, S, D]. K/V are then replicated to
//   [B, n_heads, S, D] via `repeat_kv_heads` so the existing sdpa() kernel
//   (which requires Hkv == H) can consume them. SDPA output [B, H, S, D]
//   is transposed back and fed through out_proj.
//
// Backward: sdpa_backward returns grad_K/grad_V at full n_heads width;
// `reduce_kv_heads_sum` collapses them back to n_kv_heads by summing over
// each group of group_size = n_heads / n_kv_heads adjacent heads before
// RoPE-backward and projection-backward.
//
// Memory note: the pre-expand path stores both compact and expanded K/V,
// costing ~(1 + 1/group_size) × H × S × D per layer. A native-GQA sdpa
// kernel (FlashAttention-2) will eliminate the expansion — swap it in
// behind the same Attention interface.
class Attention final : public Module {
 public:
  struct Config {
    int64_t d_model    = 0;
    int64_t n_heads    = 0;
    int64_t n_kv_heads = 0;   // 0 or equal to n_heads means MHA (no GQA).
    int64_t head_dim   = 0;   // must divide d_model and equal d_model / n_heads
    int64_t max_seq    = 0;   // required to size the RoPE table
    float   rope_base  = 10000.f;
    bool    bias       = false;
  };

  Attention(const Config& cfg, DType dtype, Device device,
            uint64_t init_seed = 0xA77EBADULL);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

  const Config& config() const { return cfg_; }

 private:
  Config  cfg_;
  Linear  q_proj_;
  Linear  k_proj_;
  Linear  v_proj_;
  Linear  out_proj_;
  Tensor  rope_table_;       // [max_seq, head_dim] fp32

  // Saved activations (for backward — reused from step_begin arena).
  Tensor  saved_input_;       // view of x, [B, S, d_model]
  Tensor  saved_q_bhsd_;      // [B, H,   S, D] — post-RoPE, post-transpose
  Tensor  saved_k_bhsd_;      // [B, H,   S, D] — replicated if GQA, else identity
  Tensor  saved_v_bhsd_;      // [B, H,   S, D]
  Tensor  saved_out_bhsd_;    // [B, H,   S, D] — sdpa output
  // Compact pre-replication tensors retained so backward can RoPE-rewind
  // and project-backward at the compact Hkv width. For MHA these alias the
  // saved_*_bhsd_ members above.
  Tensor  saved_k_bshd_kv_;   // [B, S, Hkv, D] — compact, post-RoPE
  Tensor  saved_v_bshd_kv_;   // [B, S, Hkv, D] — compact
};

}  // namespace zwt
