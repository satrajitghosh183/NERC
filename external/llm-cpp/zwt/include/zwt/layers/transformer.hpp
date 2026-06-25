#pragma once

#include "zwt/core/stream.hpp"
#include "zwt/layers/embedding.hpp"
#include "zwt/layers/linear.hpp"
#include "zwt/layers/rmsnorm.hpp"
#include "zwt/layers/transformer_block.hpp"

#include <memory>
#include <vector>

namespace zwt::dist { class BucketManager; }

namespace zwt {

// Decoder-only causal language model.
//
// Forward: tokens [B, S] (i64) -> logits [B, S, vocab] (bf16/fp32).
// Backward: grad_logits [B, S, vocab] -> (no upstream gradient; integer input).
//
// The LM head is a Linear projection from d_model -> vocab. Tied embeddings
// are supported but default off; enable via `tie_embeddings = true` and the
// embedding matrix is shared with lm_head weight.
class Transformer final : public Module {
 public:
  struct Config {
    int64_t vocab_size     = 0;
    int64_t d_model        = 0;
    int64_t n_heads        = 0;
    int64_t n_kv_heads     = 0;   // 0 or equal to n_heads means MHA (no GQA).
    int64_t head_dim       = 0;
    int64_t d_ffn          = 0;
    int64_t n_layers       = 0;
    int64_t max_seq        = 0;
    float   rope_base      = 10000.f;
    float   norm_eps       = 1e-5f;
    bool    bias           = false;
    bool    tie_embeddings = false;
  };

  Transformer(const Config& cfg, DType dtype, Device device,
              uint64_t init_seed = 0xC0DE'BA5EULL);

  // tokens: [B, S] i64. Returns logits [B, S, vocab].
  Tensor forward(const Tensor& tokens) override;
  // grad_logits: [B, S, vocab]. Returns an empty tensor (no input grad).
  Tensor backward(const Tensor& grad_logits) override;

  // DDP-aware backward. After each layer's backward returns, walks that
  // layer's parameters and calls mgr.mark_ready(...) — letting the bucket
  // manager fire allreduces on the side stream as soon as a bucket fills.
  // Pass the compute stream's handle so the gather copies (D2D) and the
  // allreduce see the right ordering. Used only when world_size > 1; the
  // single-GPU path uses the override above.
  Tensor backward(const Tensor& grad_logits,
                  dist::BucketManager& mgr,
                  StreamHandle compute_s);
  void   collect_params(std::vector<Parameter*>& out) override;

  const Config& config() const { return cfg_; }

 private:
  Config                                      cfg_;
  Embedding                                   tok_emb_;
  std::vector<std::unique_ptr<TransformerBlock>> blocks_;
  RMSNorm                                     final_norm_;
  Linear                                      lm_head_;
};

}  // namespace zwt
