/**
 * src/model/transformer.cpp
 *
 * ─── What a "transformer" is ────────────────────────────────────────
 *
 * A transformer is the neural network at the heart of every modern
 * LLM. It maps a sequence of token ids into a sequence of probability
 * distributions over the vocabulary (the next-token predictions).
 * Internally:
 *
 *   ids: [B, S]                         (B = batch, S = sequence length)
 *     │
 *     ├─ Embedding lookup                  ──> h: [B, S, D]
 *     ├─ N transformer blocks (each:
 *     │    RMSNorm → Attention → residual
 *     │    RMSNorm → FFN(SwiGLU) → residual)
 *     ├─ Final RMSNorm
 *     └─ LM head (Linear D → vocab_size)   ──> logits: [B, S, vocab]
 *
 * This file (`Transformer`) is the standard reference variant. It uses
 * a plain Embedding + N "ReorderedNorm" blocks (defined in block.cpp).
 *
 * The "fused" sister (FusedTransformer in fused_transformer.cpp) is a
 * faster variant that combines the QKV projections and the FFN
 * gate+up projections into single Linear layers — same math, fewer
 * kernel launches.
 *
 * MTP heads ("multi-token prediction"): an optional auxiliary that
 * predicts the next K tokens at once, like medusa. Empty by default.
 *
 * Multi-res embedding (DC-MRE): swapped in via cfg.use_multi_res — see
 * src/nn/multi_res_embedding.cpp for the longer explanation.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/transformer.hpp       : own class declaration.
 *   - olmo_cpp/train/activation_checkpoint.hpp : per-block recompute
 *                                                hook used inside forward
 *                                                when activation_checkpoint
 *                                                is enabled.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: instantiated when use_fused=0; forward() called
 *     each microbatch from inside src/train.cpp.
 *   - tools/dump_embeddings.cpp: instantiated read-only to extract
 *     the embedding matrix from a checkpoint.
 *
 * --- Role in training pipeline ---
 *   THE model. Every microbatch's forward() pass goes through the
 *   constructor's submodules in order: embeddings → blocks → final
 *   norm → lm_head. Loss is computed in src/train.cpp on the returned
 *   logits.
 */
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/train/activation_checkpoint.hpp"
#include "olmo_cpp/backend/fused_lm_head_ce.hpp"
#include <torch/nn/init.h>

namespace {
void trunc_normal_(torch::Tensor& t, double mean, double std, double a, double b, torch::Generator gen) {
  t.normal_(mean, std, gen);
  t.clamp_(a, b);
}
}  // namespace

namespace olmo_cpp {

TransformerImpl::TransformerImpl(const TransformerConfig& cfg)
    : blocks_(register_module("blocks", torch::nn::ModuleList())),
      lm_head_(register_module("lm_head", LMHead(cfg.d_model, cfg.vocab_size, true, cfg.layer_norm_eps))),
      embed_scale_(cfg.embed_scale),
      config_(cfg),
      use_multi_res_(cfg.use_multi_res),
      // mtp_heads_ default-constructs an empty ModuleList. We only
      // register it as a submodule when there are actually heads — that
      // keeps the serialized archive free of an empty "mtp_heads" entry
      // and lets old checkpoints (pre-MTP) load cleanly under new code.
      mtp_heads_(torch::nn::ModuleList()) {
  if (cfg.num_mtp_heads > 0) {
    register_module("mtp_heads", mtp_heads_);
  }

  // Choose embedding: multi-resolution or plain
  if (cfg.use_multi_res) {
    MultiResConfig mr_cfg;
    mr_cfg.char_trigram_buckets = cfg.multi_res_char_buckets;
    mr_cfg.phrase_buckets = cfg.multi_res_phrase_buckets;
    mr_cfg.role_embed_dim = cfg.multi_res_inner_dim;
    mr_cfg.char_embed_dim = cfg.multi_res_inner_dim;
    mr_cfg.phrase_embed_dim = cfg.multi_res_inner_dim;
    multi_res_embed_ = register_module("multi_res_embed",
        MultiResEmbedding(cfg.vocab_size, cfg.d_model, mr_cfg, cfg.bpe_vocab_path));
  } else {
    embeddings_ = register_module("embeddings", torch::nn::Embedding(cfg.vocab_size, cfg.d_model));
  }

  embedding_norm_ = RMSNorm(cfg.d_model, cfg.layer_norm_eps);
  register_module("embedding_norm", embedding_norm_.value());

  for (int64_t i = 0; i < cfg.n_layers; ++i) {
    blocks_->push_back(ReorderedNormTransformerBlock(cfg, i));
  }

  // Create MTP prediction heads
  for (int64_t k = 0; k < cfg.num_mtp_heads; ++k) {
    mtp_heads_->push_back(MTPHead(cfg.d_model, cfg.layer_norm_eps));
  }
}

const std::vector<RoPEBuffers>& TransformerImpl::get_rope_buffers(
    int64_t seq_len, torch::Device device, torch::Dtype dtype) {
  // Reuse cached buffers if they cover the needed sequence length and dtype matches
  if (seq_len <= cached_rope_len_ && !cached_rope_bufs_.empty() && dtype == cached_rope_dtype_) {
    return cached_rope_bufs_;
  }

  // Recompute with some headroom to avoid frequent reallocation.
  // All layers currently share RoPE parameters, so compute the buffers once
  // and broadcast (tensor refcount copy) to per-layer slots.
  int64_t alloc_len = std::max(seq_len, cached_rope_len_ * 2);
  RotaryEmbedding rope(config_.get_head_dim(), config_.rope_theta);
  auto bufs = rope->get_buffers(alloc_len, device, dtype);
  cached_rope_bufs_.assign(static_cast<size_t>(config_.n_layers), bufs);
  cached_rope_len_ = alloc_len;
  cached_rope_dtype_ = dtype;
  return cached_rope_bufs_;
}

void TransformerImpl::init_weights(torch::optional<torch::Generator> gen) {
  torch::NoGradGuard no_grad;
  auto g = gen.value_or(torch::Generator());
  double emb_std = config_.embedding_init_std.value_or(config_.init_std);

  // Initialize embedding weights (semantic stream in multi-res, or plain)
  auto& emb_weight = use_multi_res_
      ? multi_res_embed_->semantic_weight()
      : embeddings_->weight;
  trunc_normal_(emb_weight, 0.0, emb_std, -3 * emb_std, 3 * emb_std, g);
  if (embed_scale_) {
    emb_weight.mul_(*embed_scale_);
  }

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    auto block = blocks_->ptr<ReorderedNormTransformerBlockImpl>(i);
    for (auto& p : block->parameters()) {
      // ONLY initialize 2-D weight matrices (Linear weights). 1-D params are
      // RMSNorm/QK-norm gains (must stay 1.0) and biases (must stay 0.0);
      // overwriting a norm gain with N(0, init_std) noise collapses that
      // sublayer's residual contribution and destroys the tuned init. This
      // loop previously trunc_normal_'d *every* param, zeroing all 24 block
      // norm gains — the root of the garbage/NaN training.
      if (p.defined() && p.dim() >= 2) {
        trunc_normal_(p, 0.0, config_.init_std, -3 * config_.init_std, 3 * config_.init_std, g);
      }
    }
  }

  double lm_std = 1.0 / std::sqrt(static_cast<double>(config_.d_model));
  trunc_normal_(lm_head_->w_out()->weight, 0.0, lm_std, -3 * lm_std, 3 * lm_std, g);

  // Initialize MTP head projections
  for (int64_t k = 0; k < config_.num_mtp_heads; ++k) {
    auto head = mtp_heads_->ptr<MTPHeadImpl>(k);
    for (auto& p : head->parameters()) {
      // Only 2-D weight matrices; keep norm gains at 1.0 and biases at 0.0.
      if (p.defined() && p.dim() >= 2) {
        trunc_normal_(p, 0.0, config_.init_std, -3 * config_.init_std, 3 * config_.init_std, g);
      }
    }
  }
}

torch::Tensor TransformerImpl::forward_backbone(
    torch::Tensor input_ids,
    KVCache* kv_cache) {
  // Embed: multi-resolution (DC-MRE) or plain lookup
  auto h = use_multi_res_
      ? multi_res_embed_->forward(input_ids)
      : embeddings_(input_ids);
  if (embed_scale_ && !use_multi_res_) {
    // Scale only applies to plain embeddings; multi-res has its own projections
    h = h * *embed_scale_;
  }
  h = (*embedding_norm_)(h);

  auto new_seq_len = input_ids.size(1);
  auto device = input_ids.device();

  int64_t cached_len = kv_cache ? kv_cache->seq_len() : 0;
  int64_t total_len = cached_len + new_seq_len;
  const auto& rope_bufs = get_rope_buffers(total_len, device, h.dtype().toScalarType());

  std::optional<int64_t> start_pos =
      kv_cache ? std::optional<int64_t>(cached_len) : std::nullopt;

  bool use_act_ckpt = config_.activation_checkpoint_mode != TransformerConfig::ActivationCheckpointMode::None
                      && is_training();
  int64_t ckpt_interval = config_.activation_checkpoint_interval;

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    auto block = blocks_->ptr<ReorderedNormTransformerBlockImpl>(i);
    LayerKVCache* layer_cache = kv_cache ? &kv_cache->layers[static_cast<size_t>(i)] : nullptr;

    bool do_ckpt = use_act_ckpt &&
        (config_.activation_checkpoint_mode == TransformerConfig::ActivationCheckpointMode::Full ||
         ActivationCheckpoint::should_checkpoint(i, ckpt_interval));

    if (do_ckpt && !layer_cache) {
      // Capture by VALUE — the lambda outlives forward_backbone() (used in
      // backward for recomputation), so raw pointers into locals would dangle.
      auto rope_buf = rope_bufs[i];
      auto sp = start_pos;
      h = ActivationCheckpoint::checkpoint(
          [block, rope_buf, sp](torch::Tensor x) {
            return block->forward(x, &rope_buf, sp, nullptr);
          }, h);
    } else {
      h = block->forward(h, &rope_bufs[i], start_pos, layer_cache);
    }
  }

  return h;
}

torch::Tensor TransformerImpl::forward_backbone_paged(
    torch::Tensor input_ids,
    IPagedKVCache* paged) {
  TORCH_CHECK(paged != nullptr, "forward_backbone_paged: paged is null");

  auto h = use_multi_res_
      ? multi_res_embed_->forward(input_ids)
      : embeddings_(input_ids);
  if (embed_scale_ && !use_multi_res_) {
    h = h * *embed_scale_;
  }
  h = (*embedding_norm_)(h);

  const auto new_seq_len = input_ids.size(1);
  const auto device      = input_ids.device();
  const int64_t cached_len = paged->seq_len();
  const int64_t total_len  = cached_len + new_seq_len;
  const auto& rope_bufs = get_rope_buffers(total_len, device, h.dtype().toScalarType());

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    auto block = blocks_->ptr<ReorderedNormTransformerBlockImpl>(i);
    h = block->forward_paged(h, &rope_bufs[i], cached_len, paged, i);
  }
  return h;
}

torch::Tensor TransformerImpl::forward_paged(
    torch::Tensor input_ids,
    IPagedKVCache* paged) {
  auto h = forward_backbone_paged(input_ids, paged);
  return lm_head_(h);
}

torch::Tensor TransformerImpl::forward_tree(torch::Tensor input_ids,
                                              torch::Tensor attn_mask) {
  // Item 8.1 wiring. input_ids: [1, N] flat tree; attn_mask: [N, N]
  // bool ancestor matrix from DraftTree::flatten. Each block uses the
  // mask in place of the built-in causal mask. No KV cache.
  TORCH_CHECK(input_ids.dim() == 2 && input_ids.size(0) == 1,
              "forward_tree: input_ids must be [1, N]");
  TORCH_CHECK(attn_mask.dim() == 2 && attn_mask.size(0) == attn_mask.size(1)
              && attn_mask.size(0) == input_ids.size(1),
              "forward_tree: attn_mask must be [N, N]");

  auto h = use_multi_res_
      ? multi_res_embed_->forward(input_ids)
      : embeddings_(input_ids);
  if (embed_scale_ && !use_multi_res_) h = h * *embed_scale_;
  h = (*embedding_norm_)(h);

  const int64_t N = input_ids.size(1);
  auto device = input_ids.device();
  const auto& rope_bufs = get_rope_buffers(N, device, h.dtype().toScalarType());

  auto mask_opts = torch::TensorOptions().dtype(h.dtype()).device(device);
  auto add_mask = torch::where(
      attn_mask,
      torch::zeros({}, mask_opts),
      torch::full({}, -std::numeric_limits<float>::infinity(), mask_opts));

  for (int64_t i = 0; i < config_.n_layers; ++i) {
    auto block = blocks_->ptr<ReorderedNormTransformerBlockImpl>(i);
    h = block->forward_with_mask(h, &rope_bufs[i], add_mask);
  }
  return lm_head_(h);
}

torch::Tensor TransformerImpl::forward(
    torch::Tensor input_ids,
    c10::optional<torch::Tensor> labels,
    int64_t ignore_index,
    KVCache* kv_cache) {

  auto h = forward_backbone(input_ids, kv_cache);

  if (!labels.has_value()) {
    // Inference path: just main logits, no MTP loss.
    return lm_head_(h);
  }

  const int64_t seq_len_full = h.size(1);
  const int64_t B = h.size(0);
  const int64_t d = h.size(2);
  const int64_t K = config_.num_mtp_heads;

  // A3 — fused LM-head + softmax-CE. The old path stacked all (K+1)
  // heads' inputs, ran the LM head ONCE to materialize a (K+1)*B*S × V
  // logits tensor, then computed cross_entropy per head. That tensor
  // is ~3.3 GB at V=50k/B*S=16k and forces an extra read in CE; the
  // fused kernel never writes it. We give up the "ONE GEMM" combine
  // (each head now does its own GEMM internally inside the fused
  // kernel) but each remaining GEMM has only the row count it actually
  // needs, and the bandwidth saving dwarfs the small launch overhead.
  auto w_lm = lm_head_->weight();

  // ── Main head ────────────────────────────────────────────────────
  auto h_norm_main = lm_head_->apply_norm(h);
  auto main_h_flat = h_norm_main.reshape({-1, d});
  auto main_labels_flat = labels->reshape(-1);
  auto main_loss = fused_lm_head_ce_autograd(
      main_h_flat, w_lm, main_labels_flat, ignore_index);

  // ── MTP heads ────────────────────────────────────────────────────
  if (K > 0) {
    auto mtp_loss_sum = torch::zeros({}, main_loss.options());
    int64_t valid_heads = 0;
    for (int64_t k = 0; k < K; ++k) {
      const int64_t shift = k + 1;
      if (shift >= seq_len_full) continue;
      auto head = mtp_heads_->ptr<MTPHeadImpl>(k);
      auto mtp_h = head->forward(h).narrow(1, 0, seq_len_full - shift);   // [B, S-shift, d]
      auto labels_shifted = labels->narrow(1, shift, seq_len_full - shift);
      auto mtp_h_normed = lm_head_->apply_norm(mtp_h);
      auto mtp_loss = fused_lm_head_ce_autograd(
          mtp_h_normed.reshape({-1, d}),
          w_lm,
          labels_shifted.reshape(-1),
          ignore_index);
      mtp_loss_sum = mtp_loss_sum + mtp_loss;
      ++valid_heads;
    }
    if (valid_heads > 0) {
      main_loss = main_loss + config_.mtp_loss_weight * mtp_loss_sum / static_cast<double>(valid_heads);
    }
  }
  return main_loss;
}

std::vector<torch::Tensor> TransformerImpl::forward_mtp_draft(torch::Tensor hidden_state) {
  // hidden_state: [1, 1, d_model] or [d_model] — the last position's hidden state
  if (hidden_state.dim() == 1) {
    hidden_state = hidden_state.unsqueeze(0).unsqueeze(0);
  } else if (hidden_state.dim() == 2) {
    hidden_state = hidden_state.unsqueeze(0);
  }

  std::vector<torch::Tensor> draft_logits;
  draft_logits.reserve(static_cast<size_t>(config_.num_mtp_heads));
  if (config_.num_mtp_heads == 0) return draft_logits;

  // B (item from optimization roadmap): stack all MTP head outputs along a
  // new leading dim and project through lm_head in ONE GEMM instead of
  // k separate calls. At S=1 the stacked GEMM is [K, 1, 1, d] → [K, 1, 1, V],
  // a tiny tensor; cuBLAS prefers the bigger M dim. Numerically identical
  // to running them in a loop.
  std::vector<torch::Tensor> inputs;
  inputs.reserve(static_cast<size_t>(config_.num_mtp_heads));
  for (int64_t k = 0; k < config_.num_mtp_heads; ++k) {
    auto head = mtp_heads_->ptr<MTPHeadImpl>(k);
    inputs.push_back(head->forward(hidden_state));    // [B=1, S=1, d]
  }
  auto stacked = torch::stack(inputs, /*dim=*/0);     // [K, 1, 1, d]
  const int64_t K = stacked.size(0);
  const int64_t B = stacked.size(1);
  const int64_t S = stacked.size(2);
  const int64_t d = stacked.size(3);
  auto stacked_flat = stacked.view({K * B * S, d});
  auto all_logits_flat = lm_head_(stacked_flat);                       // ONE GEMM
  auto all_logits = all_logits_flat.view({K, B, S, config_.vocab_size}); // [K, 1, 1, V]

  for (int64_t k = 0; k < K; ++k) {
    // logits[k]: [1, 1, V] → squeeze to [V]
    draft_logits.push_back(all_logits.select(0, k).squeeze(0).squeeze(0));
  }
  return draft_logits;
}

}  // namespace olmo_cpp
