#pragma once

/**
 * include/olmo_cpp/config.hpp
 *
 * Architectural configuration for the transformer model. `TransformerConfig`
 * is the single struct that fully describes a model instance — width,
 * depth, head layout, normalization variant, attention backend, RoPE
 * scaling, MoE settings, activation checkpointing, MTP heads, multi-res
 * embedding, and so on. Both `Transformer` and `FusedTransformer` consume
 * this struct in their constructors, so everything that can vary across
 * runs (and across the various olmo2/Llama-style variants we support) is
 * funneled through here.
 *
 * Also exposes loaders/presets: `load_config_from_json` reads a JSON
 * file (when nlohmann/json is available) and `olmo2_7b_config()` returns
 * the canonical 7B preset for benchmarking.
 *
 * --- Includes from this project ---
 *   (none — pure stdlib)
 *
 * --- Callers (concrete uses elsewhere in this repo) ---
 *   - src/main.cpp: builds a `TransformerConfig` from the `.conf` file
 *     and calls `cfg.validate()` before instantiating a model.
 *   - src/config.cpp: implements `validate()`, `load_config_from_json`,
 *     and the 7B preset.
 *   - include/olmo_cpp/model/transformer.hpp /
 *     include/olmo_cpp/model/fused_transformer.hpp: `Transformer(cfg)` and
 *     `FusedTransformer(cfg)` constructors take a const reference to
 *     this struct.
 *
 * --- Role in training pipeline ---
 *   Plain-old-data carrier with `validate()` for sanity checks. Never
 *   mutated after construction by the model — read-only blueprint. The
 *   `get_*` accessors hide "implicit defaults" (e.g. `n_kv_heads = -1`
 *   meaning "same as n_heads") so callers don't have to repeat the
 *   defaulting logic.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <cmath>

namespace olmo_cpp {

/// Ensure x is rounded up to the nearest multiple of 'of'
/// Used to size FFN hidden dim to a tensor-core-friendly multiple.
/// Example: ensure_multiple_of(11008, 256) -> 11008; (10999, 256) -> 11008.
inline int64_t ensure_multiple_of(int64_t x, int64_t of) {
  // Float division + ceil keeps the math obvious; for non-negative inputs
  // `(x + of - 1) / of * of` would also work without floating point.
  return of * static_cast<int64_t>(std::ceil(static_cast<double>(x) / of));
}

/// Full architectural description of a transformer. Defaults match the
/// olmo2-7B configuration; `main.cpp` overrides every field that appears
/// in `[model]` of the .conf file. Most "-1" defaults mean "derive from
/// other fields" — see the `get_*()` accessors below for the actual rule.
struct TransformerConfig {
  /// Hidden size (a.k.a. embedding dim, d_model).
  int64_t d_model = 4096;
  /// Vocabulary size of the input/output embedding (50257 = GPT-2 BPE).
  int64_t vocab_size = 50257;
  /// Number of transformer blocks (depth).
  int64_t n_layers = 32;
  /// Number of attention heads in the Q projection.
  int64_t n_heads = 32;
  /// Number of KV heads (Grouped-Query Attention). -1 == use full MHA.
  int64_t n_kv_heads = -1;  // -1 means n_kv_heads = n_heads
  /// Per-head dimension. -1 means derive as d_model / n_heads.
  int64_t head_dim = -1;    // -1 means head_dim = d_model / n_heads
  /// Base frequency theta for rotary positional embeddings.
  int64_t rope_theta = 500000;
  /// Epsilon added inside RMSNorm/LayerNorm to avoid div-by-zero.
  double layer_norm_eps = 1e-6;
  /// Standard deviation for the default normal weight init.
  double init_std = 0.02;
  /// Optional override for the embedding-output scale (otherwise sqrt(d_model)
  /// or 1.0 depending on block variant).
  std::optional<double> embed_scale;
  /// Optional separate init std specifically for the embedding matrix.
  std::optional<double> embedding_init_std;
  /// Apply RMSNorm to Q and K before computing attention scores (post-RoPE).
  /// Borrowed from olmo2 — stabilises low-precision attention.
  bool use_qk_norm = true;
  /// Apply per-head QK norm (one scale parameter per head) instead of one
  /// shared scale. Off by default.
  bool use_head_qk_norm = false;
  /// FFN hidden size is rounded up to a multiple of this (matches the GPU
  /// matmul tile size for tensor cores). 256 keeps cuBLAS happy.
  int64_t hidden_size_multiple_of = 256;
  /// SwiGLU expansion factor before rounding. 1.5 yields ~5.33x d_model.
  double hidden_size_multiplier = 1.5;

  // === Block variant ===
  /// Selects which transformer-block recipe to instantiate. Each variant
  /// differs in the placement of LayerNorm, residual connections, and
  /// MoE wiring. ReorderedNorm is the olmo2 default.
  enum class BlockType { ReorderedNorm, PeriNorm, NormalizedNGPT, LayerNormScaled, MoEReorderedNorm, MoEHybridReorderedNorm };
  /// Active block recipe — see the enum above.
  BlockType block_type = BlockType::ReorderedNorm;

  // === Attention backend ===
  /// Which attention kernel implementation to use. SDPA delegates to
  /// PyTorch's `scaled_dot_product_attention` (which itself dispatches
  /// to flash/cudnn under the hood); flash2/3 use FlashAttention sources
  /// directly when linked; transformer_engine uses NVIDIA's TE.
  enum class AttentionBackend { SDPA, FlashAttention2, FlashAttention3, TransformerEngine };
  AttentionBackend attention_backend = AttentionBackend::SDPA;

  // === Sliding window attention ===
  /// Local-attention window in tokens. -1 disables (full causal). Useful
  /// for long-context models like Mistral.
  int64_t sliding_window_size = -1;  // -1 = disabled

  // === Gated attention ===
  /// Optional per-output gate applied to attention output. Headwise =
  /// one gate scalar per head; Elementwise = per-channel gate vector.
  enum class GatedAttentionType { None, Headwise, Elementwise };
  GatedAttentionType gated_attention = GatedAttentionType::None;

  // === RoPE scaling ===
  /// Long-context RoPE rescaling strategies (used when training/finetuning
  /// for context lengths longer than the base trained length).
  enum class RoPEScalingType { None, ABF, PositionInterpolation, Stepwise, YaRN };
  RoPEScalingType rope_scaling_type = RoPEScalingType::None;
  /// Multiplier applied to position indices (or freqs) by the chosen scheme.
  double rope_scaling_factor = 1.0;
  /// YaRN-specific: high-frequency rotation cutoff.
  double rope_yarn_beta_fast = 32.0;
  /// YaRN-specific: low-frequency rotation cutoff.
  double rope_yarn_beta_slow = 1.0;

  // === MoE config ===
  /// Enable Mixture-of-Experts FFN in supported block types.
  bool use_moe = false;
  /// Total number of experts per MoE block.
  int64_t moe_num_experts = 8;
  /// Top-k routing: each token picks this many experts.
  int64_t moe_top_k = 2;
  /// Override expert hidden size; -1 means use the dense FFN size.
  int64_t moe_hidden_size = -1;  // -1 = use get_hidden_size()
  /// Capacity factor for token-choice routing (1.0 = exact balance).
  double moe_capacity_factor = 1.25;
  /// Dropless mode (no token dropping) — uses MegaBlocks-style scatter.
  bool moe_dropless = true;
  /// Weight on the z-loss penalty (keeps router logits in a sane range).
  double moe_zloss_weight = 1e-3;
  /// Weight on the load-balancing auxiliary loss.
  double moe_lb_loss_weight = 1e-2;
  /// Interleave dense and MoE blocks (some block types only).
  bool moe_hybrid = false;         // hybrid dense+MoE blocks
  /// In hybrid mode, every Nth block is MoE; the rest are dense.
  int64_t moe_hybrid_interval = 2; // every Nth block is MoE

  // === Activation checkpointing ===
  /// None = keep all activations; Full = recompute every block during
  /// backward; SelectedBlocks = recompute every Nth block per `interval`.
  enum class ActivationCheckpointMode { None, Full, SelectedBlocks };
  ActivationCheckpointMode activation_checkpoint_mode = ActivationCheckpointMode::None;
  /// Stride for SelectedBlocks mode — checkpoint every Nth block.
  int64_t activation_checkpoint_interval = 1; // checkpoint every Nth block

  // === Convolution ===
  /// Optional 1-D depthwise conv inserted before attention (RWKV-style).
  bool use_conv = false;
  /// Kernel width for the conv (causal, left-padded).
  int64_t conv_kernel_size = 4;

  // === Multi-Token Prediction (MTP) ===
  /// Number of look-ahead heads predicting tokens at offset +2, +3, ... .
  /// 0 disables MTP entirely (DeepSeek-V3 style auxiliary head).
  int64_t num_mtp_heads = 0;        // 0 = disabled, 2-4 recommended
  /// Aggregate weight on the auxiliary MTP losses relative to next-token loss.
  double mtp_loss_weight = 0.1;     // weight for auxiliary MTP losses

  // === Float8 ===
  /// Enable FP8 GEMMs on H100+ (Transformer Engine path).
  bool use_float8 = false;

  // === Mixed-precision per layer (item T.2) ===
  // Per-component dtype overrides. When set, AttentionImpl / FeedForwardImpl /
  // LMHead use the specified dtype on their hot path instead of the autocast
  // default. Empty string = use autocast policy. Recognized values: "bf16",
  // "fp16", "fp32", "fp8", "int8", "int4". Sensible defaults: embedding +
  // LM head stay in bf16 (vocab projection is brittle under aggressive quant),
  // attention/FFN can go FP8 or INT4 on hopper/blackwell respectively.
  std::string attn_dtype = "";          // empty = autocast default
  std::string ffn_dtype = "";
  std::string embed_dtype = "";
  std::string lm_head_dtype = "";

  // === Multi-Resolution Embedding (DC-MRE) ===
  // Enhances structural tokenizer with dual-codebook + morphological features.
  /// Master switch for DC-MRE — adds char-trigram + phrase auxiliary streams.
  bool use_multi_res = false;
  /// Path to GPT-2 vocab.json used to bootstrap the char-trigram table.
  std::string bpe_vocab_path;        // path to GPT-2 vocab.json (for char trigrams)
  /// Hash buckets in the char-trigram codebook.
  int64_t multi_res_char_buckets = 4096;   // hash buckets for char trigrams
  /// Hash buckets in the phrase-context codebook.
  int64_t multi_res_phrase_buckets = 8192; // hash buckets for phrase context
  /// Low-rank dimension of the auxiliary aggregation MLP.
  int64_t multi_res_inner_dim = 64;        // low-rank dimension for aux streams

  // === Layer norm type ===
  /// Pick the normalization layer used throughout the block. RMSNorm is
  /// the modern default; FusedRMSNorm dispatches to our fused kernel.
  enum class LayerNormType { RMSNorm, LayerNorm, L2Norm, FusedRMSNorm };
  LayerNormType layer_norm_type = LayerNormType::RMSNorm;

  /// Compute effective MoE hidden size (defaults to get_hidden_size())
  /// Used by the MoE expert FFN's input/output projections.
  int64_t get_moe_hidden_size() const {
    return moe_hidden_size > 0 ? moe_hidden_size : get_hidden_size();
  }

  /// Compute effective n_kv_heads (defaults to n_heads if not set)
  /// When n_kv_heads < n_heads we have GQA; when equal we have full MHA.
  int64_t get_n_kv_heads() const {
    return n_kv_heads > 0 ? n_kv_heads : n_heads;
  }

  /// Compute effective head_dim (defaults to d_model / n_heads)
  /// All Q, K, V projections produce tensors with this last-dim per head.
  int64_t get_head_dim() const {
    return head_dim > 0 ? head_dim : (d_model / n_heads);
  }

  /// Compute hidden size for feed-forward: 1.5 * d_model, rounded to multiple
  /// SwiGLU formula: base = 8/3 * d_model * multiplier (the 8/3 factor
  /// accounts for SwiGLU using two parallel projections, keeping total FFN
  /// param count comparable to a single 4*d_model GeLU FFN). Rounded up
  /// to `hidden_size_multiple_of` for tensor-core alignment.
  int64_t get_hidden_size() const {
    int64_t base = static_cast<int64_t>(8 * d_model / 3.0 * hidden_size_multiplier);
    return ensure_multiple_of(base, hidden_size_multiple_of);
  }

  /// Throw if the config is internally inconsistent (e.g. n_heads not
  /// divisible by n_kv_heads, or moe_top_k > moe_num_experts).
  /// Implemented in src/config.cpp.
  void validate() const;
};

/// Load config from JSON file (requires nlohmann/json if HAS_NLOHMANN_JSON)
/// Throws if the file is missing, malformed, or built without nlohmann/json.
TransformerConfig load_config_from_json(const std::string& path);

/// Create olmo2_7B preset (4096 d_model, 32 layers, 32 heads, MHA, RoPE).
/// Used by benchmarks and as a sanity reference; production runs use the
/// .conf file path.
TransformerConfig olmo2_7b_config(int64_t vocab_size = 50257);

}  // namespace olmo_cpp

