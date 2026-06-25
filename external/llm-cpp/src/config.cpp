/**
 * src/config.cpp
 *
 * Implements the non-trivial parts of TransformerConfig that don't fit
 * cleanly in the header:
 *   - validate(): sanity-check every field combination (GQA divisibility,
 *     MoE expert counts, RoPE scaling positivity, etc.).
 *   - load_config_from_json(): build a TransformerConfig from a JSON
 *     description. Used to interoperate with the upstream OLMo-core
 *     Python repo, where configs are JSON.
 *   - olmo2_7b_config(): canonical 7B preset for benchmarking against
 *     OLMo-2 reference numbers. Used by run_7B.sh and the H100 conf.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp : TransformerConfig struct + free functions.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: cfg.validate() right after parsing the .conf file
 *     (so we fail fast on bad combinations before allocating GPU memory).
 *   - tools/dump_embeddings.cpp: same — validates after loading the .conf.
 *   - any benchmark binary that wants the OLMo-2 7B canonical shape.
 *
 * --- Role in training pipeline ---
 *   Pure POD plumbing. No model state lives here — this file simply
 *   defines the rules for what makes a valid TransformerConfig.
 */
#include "olmo_cpp/config.hpp"
#include <stdexcept>
#include <fstream>
#include <sstream>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace olmo_cpp {

void TransformerConfig::validate() const {
  if (d_model <= 0 || vocab_size <= 0 || n_layers <= 0 || n_heads <= 0) {
    throw std::runtime_error("Invalid config: d_model, vocab_size, n_layers, n_heads must be positive");
  }
  if (n_kv_heads > 0 && n_heads % n_kv_heads != 0) {
    throw std::runtime_error("n_heads must be divisible by n_kv_heads for GQA");
  }
  if (head_dim > 0 && d_model != n_heads * head_dim) {
    throw std::runtime_error("d_model must equal n_heads * head_dim when head_dim is explicit");
  }
  if (use_moe) {
    if (moe_num_experts <= 0) {
      throw std::runtime_error("moe_num_experts must be positive when use_moe is true");
    }
    if (moe_top_k <= 0 || moe_top_k > moe_num_experts) {
      throw std::runtime_error("moe_top_k must be in [1, moe_num_experts]");
    }
    if (moe_hybrid && moe_hybrid_interval <= 0) {
      throw std::runtime_error("moe_hybrid_interval must be positive when moe_hybrid is true");
    }
  }
  if (rope_scaling_factor <= 0.0) {
    throw std::runtime_error("rope_scaling_factor must be positive");
  }
  if (activation_checkpoint_interval <= 0) {
    throw std::runtime_error("activation_checkpoint_interval must be positive");
  }
  if (use_conv && conv_kernel_size <= 0) {
    throw std::runtime_error("conv_kernel_size must be positive when use_conv is true");
  }
}

TransformerConfig load_config_from_json(const std::string& path) {
#ifdef HAS_NLOHMANN_JSON
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open config file: " + path);
  auto j = nlohmann::json::parse(f);
  TransformerConfig cfg;
  cfg.d_model = j.value("d_model", cfg.d_model);
  cfg.vocab_size = j.value("vocab_size", cfg.vocab_size);
  cfg.n_layers = j.value("n_layers", cfg.n_layers);
  cfg.n_heads = j.value("n_heads", cfg.n_heads);
  cfg.n_kv_heads = j.value("n_kv_heads", -1);
  cfg.head_dim = j.value("head_dim", -1);
  cfg.rope_theta = j.value("rope_theta", cfg.rope_theta);
  cfg.layer_norm_eps = j.value("layer_norm_eps", cfg.layer_norm_eps);
  cfg.init_std = j.value("init_std", cfg.init_std);
  cfg.use_qk_norm = j.value("use_qk_norm", cfg.use_qk_norm);
  cfg.use_head_qk_norm = j.value("use_head_qk_norm", cfg.use_head_qk_norm);
  cfg.hidden_size_multiple_of = j.value("hidden_size_multiple_of", cfg.hidden_size_multiple_of);
  cfg.hidden_size_multiplier = j.value("hidden_size_multiplier", cfg.hidden_size_multiplier);
  if (j.contains("embed_scale")) cfg.embed_scale = j["embed_scale"].get<double>();
  if (j.contains("embedding_init_std")) cfg.embedding_init_std = j["embedding_init_std"].get<double>();

  // Block type
  if (j.contains("block_type")) {
    auto s = j["block_type"].get<std::string>();
    if (s == "reordered_norm") cfg.block_type = TransformerConfig::BlockType::ReorderedNorm;
    else if (s == "peri_norm") cfg.block_type = TransformerConfig::BlockType::PeriNorm;
    else if (s == "normalized_ngpt") cfg.block_type = TransformerConfig::BlockType::NormalizedNGPT;
    else if (s == "layer_norm_scaled") cfg.block_type = TransformerConfig::BlockType::LayerNormScaled;
    else if (s == "moe_reordered_norm") cfg.block_type = TransformerConfig::BlockType::MoEReorderedNorm;
    else if (s == "moe_hybrid_reordered_norm") cfg.block_type = TransformerConfig::BlockType::MoEHybridReorderedNorm;
    else throw std::runtime_error("Unknown block_type: " + s);
  }

  // Attention backend
  if (j.contains("attention_backend")) {
    auto s = j["attention_backend"].get<std::string>();
    if (s == "sdpa") cfg.attention_backend = TransformerConfig::AttentionBackend::SDPA;
    else if (s == "flash_attention_2") cfg.attention_backend = TransformerConfig::AttentionBackend::FlashAttention2;
    else if (s == "flash_attention_3") cfg.attention_backend = TransformerConfig::AttentionBackend::FlashAttention3;
    else if (s == "transformer_engine") cfg.attention_backend = TransformerConfig::AttentionBackend::TransformerEngine;
    else throw std::runtime_error("Unknown attention_backend: " + s);
  }

  // Sliding window attention
  cfg.sliding_window_size = j.value("sliding_window_size", cfg.sliding_window_size);

  // Gated attention
  if (j.contains("gated_attention")) {
    auto s = j["gated_attention"].get<std::string>();
    if (s == "none") cfg.gated_attention = TransformerConfig::GatedAttentionType::None;
    else if (s == "headwise") cfg.gated_attention = TransformerConfig::GatedAttentionType::Headwise;
    else if (s == "elementwise") cfg.gated_attention = TransformerConfig::GatedAttentionType::Elementwise;
    else throw std::runtime_error("Unknown gated_attention: " + s);
  }

  // RoPE scaling
  if (j.contains("rope_scaling_type")) {
    auto s = j["rope_scaling_type"].get<std::string>();
    if (s == "none") cfg.rope_scaling_type = TransformerConfig::RoPEScalingType::None;
    else if (s == "abf") cfg.rope_scaling_type = TransformerConfig::RoPEScalingType::ABF;
    else if (s == "position_interpolation") cfg.rope_scaling_type = TransformerConfig::RoPEScalingType::PositionInterpolation;
    else if (s == "stepwise") cfg.rope_scaling_type = TransformerConfig::RoPEScalingType::Stepwise;
    else if (s == "yarn") cfg.rope_scaling_type = TransformerConfig::RoPEScalingType::YaRN;
    else throw std::runtime_error("Unknown rope_scaling_type: " + s);
  }
  cfg.rope_scaling_factor = j.value("rope_scaling_factor", cfg.rope_scaling_factor);
  cfg.rope_yarn_beta_fast = j.value("rope_yarn_beta_fast", cfg.rope_yarn_beta_fast);
  cfg.rope_yarn_beta_slow = j.value("rope_yarn_beta_slow", cfg.rope_yarn_beta_slow);

  // MoE config
  cfg.use_moe = j.value("use_moe", cfg.use_moe);
  cfg.moe_num_experts = j.value("moe_num_experts", cfg.moe_num_experts);
  cfg.moe_top_k = j.value("moe_top_k", cfg.moe_top_k);
  cfg.moe_hidden_size = j.value("moe_hidden_size", cfg.moe_hidden_size);
  cfg.moe_capacity_factor = j.value("moe_capacity_factor", cfg.moe_capacity_factor);
  cfg.moe_dropless = j.value("moe_dropless", cfg.moe_dropless);
  cfg.moe_zloss_weight = j.value("moe_zloss_weight", cfg.moe_zloss_weight);
  cfg.moe_lb_loss_weight = j.value("moe_lb_loss_weight", cfg.moe_lb_loss_weight);
  cfg.moe_hybrid = j.value("moe_hybrid", cfg.moe_hybrid);
  cfg.moe_hybrid_interval = j.value("moe_hybrid_interval", cfg.moe_hybrid_interval);

  // Activation checkpointing
  if (j.contains("activation_checkpoint_mode")) {
    auto s = j["activation_checkpoint_mode"].get<std::string>();
    if (s == "none") cfg.activation_checkpoint_mode = TransformerConfig::ActivationCheckpointMode::None;
    else if (s == "full") cfg.activation_checkpoint_mode = TransformerConfig::ActivationCheckpointMode::Full;
    else if (s == "selected_blocks") cfg.activation_checkpoint_mode = TransformerConfig::ActivationCheckpointMode::SelectedBlocks;
    else throw std::runtime_error("Unknown activation_checkpoint_mode: " + s);
  }
  cfg.activation_checkpoint_interval = j.value("activation_checkpoint_interval", cfg.activation_checkpoint_interval);

  // Convolution
  cfg.use_conv = j.value("use_conv", cfg.use_conv);
  cfg.conv_kernel_size = j.value("conv_kernel_size", cfg.conv_kernel_size);

  // Multi-Token Prediction
  cfg.num_mtp_heads = j.value("num_mtp_heads", cfg.num_mtp_heads);
  cfg.mtp_loss_weight = j.value("mtp_loss_weight", cfg.mtp_loss_weight);

  // Float8
  cfg.use_float8 = j.value("use_float8", cfg.use_float8);

  // Multi-Resolution Embedding (DC-MRE)
  cfg.use_multi_res = j.value("use_multi_res", cfg.use_multi_res);
  cfg.bpe_vocab_path = j.value("bpe_vocab_path", cfg.bpe_vocab_path);
  cfg.multi_res_char_buckets = j.value("multi_res_char_buckets", cfg.multi_res_char_buckets);
  cfg.multi_res_phrase_buckets = j.value("multi_res_phrase_buckets", cfg.multi_res_phrase_buckets);
  cfg.multi_res_inner_dim = j.value("multi_res_inner_dim", cfg.multi_res_inner_dim);

  // Layer norm type
  if (j.contains("layer_norm_type")) {
    auto s = j["layer_norm_type"].get<std::string>();
    if (s == "rms_norm") cfg.layer_norm_type = TransformerConfig::LayerNormType::RMSNorm;
    else if (s == "layer_norm") cfg.layer_norm_type = TransformerConfig::LayerNormType::LayerNorm;
    else if (s == "l2_norm") cfg.layer_norm_type = TransformerConfig::LayerNormType::L2Norm;
    else if (s == "fused_rms_norm") cfg.layer_norm_type = TransformerConfig::LayerNormType::FusedRMSNorm;
    else throw std::runtime_error("Unknown layer_norm_type: " + s);
  }

  cfg.validate();
  return cfg;
#else
  (void)path;
  throw std::runtime_error("JSON config loading requires nlohmann/json. Use preset configs instead.");
#endif
}

TransformerConfig olmo2_7b_config(int64_t vocab_size) {
  TransformerConfig cfg;
  cfg.d_model = 4096;
  cfg.vocab_size = vocab_size;
  cfg.n_layers = 32;
  cfg.n_heads = 32;
  cfg.n_kv_heads = -1;
  cfg.head_dim = -1;
  cfg.rope_theta = 500000;
  cfg.layer_norm_eps = 1e-6;
  cfg.init_std = 0.02;
  cfg.use_qk_norm = true;
  cfg.use_head_qk_norm = false;
  cfg.hidden_size_multiple_of = 256;
  cfg.hidden_size_multiplier = 1.5;
  cfg.validate();
  return cfg;
}

}  // namespace olmo_cpp
