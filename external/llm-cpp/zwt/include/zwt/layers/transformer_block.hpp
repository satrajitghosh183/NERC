#pragma once

#include "zwt/layers/attention.hpp"
#include "zwt/layers/ffn.hpp"
#include "zwt/layers/rmsnorm.hpp"

namespace zwt {

// Pre-norm transformer block (Llama style):
//
//   y = x + attn(norm1(x))
//   z = y + ffn(norm2(y))
//
// The residual add uses the fp32-accumulated axpy kernel on bf16 weights.
// Backward unrolls the residual by accumulating gradients into the skip path.
class TransformerBlock final : public Module {
 public:
  struct Config {
    int64_t d_model    = 0;
    int64_t n_heads    = 0;
    int64_t n_kv_heads = 0;   // 0 or equal to n_heads means MHA (no GQA).
    int64_t head_dim   = 0;
    int64_t d_ffn      = 0;
    int64_t max_seq    = 0;
    float   rope_base  = 10000.f;
    float   norm_eps   = 1e-5f;
    bool    bias       = false;
  };

  TransformerBlock(const Config& cfg, DType dtype, Device device,
                   uint64_t init_seed = 0);

  Tensor forward(const Tensor& x) override;
  Tensor backward(const Tensor& grad_y) override;
  void   collect_params(std::vector<Parameter*>& out) override;

  const Config& config() const { return cfg_; }

 private:
  Config    cfg_;
  RMSNorm   norm1_;
  Attention attn_;
  RMSNorm   norm2_;
  FFN       ffn_;

  // Saved activations for residual backward.
  Tensor    saved_input_;     // x at block entry (for residual 1)
  Tensor    saved_after_attn_; // x + attn(norm1(x)) (input to norm2)
};

}  // namespace zwt
