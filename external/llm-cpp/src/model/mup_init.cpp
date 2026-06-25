/**
 * src/model/mup_init.cpp
 *
 * Implements µP (Maximal Update Parameterization) initialization for the
 * transformer model. µP rescales weight initialization and per-tensor
 * learning rates as a function of model width so that hyperparameters tuned
 * at a small width "transfer" without re-tuning at large widths. Concretely
 * we walk every parameter in the model, classify it (norm / embedding /
 * lm_head / output projection / structural projection / hidden weight) by
 * the parameter name, and apply the rule appropriate to its role.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/mup_init.hpp: declares apply_mup_init,
 *     get_mup_lr_multipliers, MuPConfig, MuPLRMultiplier.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: calls olmo_cpp::apply_mup_init(*model, cfg, mup_cfg,
 *     seed_state.torch_gen) at line 63 right after model construction and
 *     before the optimizer is built. The per-group LR multipliers from
 *     get_mup_lr_multipliers feed into AdamW/Muon parameter groups.
 *
 * --- Role in training pipeline ---
 *   Runs exactly once at startup, before optimizer creation and the first
 *   forward pass. It mutates parameters in-place under NoGradGuard. The
 *   companion get_mup_lr_multipliers() is queried later when assembling
 *   parameter groups so that the optimizer applies the correct width-scaled
 *   LR per group throughout training.
 */
#include "olmo_cpp/model/mup_init.hpp"
#include <torch/nn/init.h>
#include <iostream>
#include <cmath>

namespace {
/// Truncated-normal initialization: sample N(mean, std_val^2), then clamp to
/// [a, b] (a hard truncation, not the rejection-sampling form). For µP we
/// always pass a = -3*std, b = +3*std so we cut outliers at three sigmas.
void trunc_normal_(torch::Tensor& t, double mean, double std_val, double a, double b, torch::Generator gen) {
  t.normal_(mean, std_val, gen);
  t.clamp_(a, b);
}
}  // namespace

namespace olmo_cpp {

/// Apply µP initialization to every parameter in `model`, mutating in-place.
/// cfg     : full transformer config (used to read d_model when target_width
///           defaults to 0).
/// mup_cfg : µP knobs (base_width, target_width, zero_init_lm_head, ...).
/// gen     : optional torch::Generator for reproducible sampling. When not
///           provided, a default generator is constructed (non-deterministic).
void apply_mup_init(
    torch::nn::Module& model,
    const TransformerConfig& cfg,
    const MuPConfig& mup_cfg,
    torch::optional<torch::Generator> gen) {

  // We are about to mutate parameter tensors that the autograd engine sees;
  // disable grad tracking to avoid building a graph for these writes.
  torch::NoGradGuard no_grad;
  auto g = gen.value_or(torch::Generator());

  // Target width defaults to d_model; allows ablating µP at a fixed width
  // by overriding to a smaller value.
  double target_width = mup_cfg.target_width > 0 ? mup_cfg.target_width : static_cast<double>(cfg.d_model);
  // Core µP knob: as we widen, hidden/output stds shrink linearly.
  double width_ratio = mup_cfg.base_width / target_width;

  std::cout << "[µP] Initializing with base_width=" << mup_cfg.base_width
            << ", target_width=" << target_width
            << ", width_ratio=" << width_ratio << "\n";

  // Walk every named parameter; classify by name substring matching since
  // we already know the module naming convention (RMSNorm.weight is "...norm.weight",
  // lm_head's projection is "lm_head.w_out.weight", etc.).
  for (auto& named_param : model.named_parameters()) {
    auto& name = named_param.key();
    auto& p = named_param.value();

    // Defensive: skip undefined or empty parameters (can happen for unused
    // optional submodules that nonetheless register a placeholder).
    if (!p.defined() || p.numel() == 0) continue;

    // --- Norm weights must be checked FIRST (they appear inside lm_head, blocks, etc.) ---
    // RMSNorm/LayerNorm gains are 1-D; pin them to 1.0 so the layer starts
    // as identity. Checked first because "norm" can be a substring of more
    // specific names like "feed_forward_norm.weight".
    if (name.find("norm") != std::string::npos && p.dim() == 1) {
      p.fill_(1.0);

    // --- Embeddings: std=1.0 in µP (match both plain and multi-res names) ---
    // µP keeps embedding init width-independent. We list every embedding
    // name used by the model variants (token, role tags, character / phrase
    // for DC-MRE structural variants).
    } else if (name.find("embeddings") != std::string::npos ||
               name.find("token_embed") != std::string::npos ||
               name.find("role_embed") != std::string::npos ||
               name.find("char_embed") != std::string::npos ||
               name.find("phrase_embed") != std::string::npos) {
      double std_val = 1.0;
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

    // --- LM head output weight: zero-init in µP ---
    // Zero-init at the unembedding produces a uniform initial logit
    // distribution, removing a major source of early-step instability.
    } else if (mup_cfg.zero_init_lm_head &&
               name.find("lm_head") != std::string::npos &&
               name.find("w_out") != std::string::npos) {
      p.zero_();

    // --- Output projections (attention w_out, FFN w2): scale by width_ratio ---
    // These are the residual-back-into-stream tensors. µP shrinks their std
    // by an extra factor of width_ratio compared with input projections.
    } else if (name.find("w_out") != std::string::npos ||
               name.find("w2") != std::string::npos) {
      // fan_in = number of inputs to a single output unit; for a 2-D Linear
      // weight that's size(1) (the columns). 1-D weights fall back to size(0).
      double fan_in = p.dim() >= 2 ? static_cast<double>(p.size(1)) : static_cast<double>(p.size(0));
      double std_val = width_ratio / std::sqrt(fan_in);
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

    // --- Projection layers (role_proj, char_proj, phrase_proj): 1/sqrt(fan_in) ---
    // Side projections used by structural variants — treat as standard
    // hidden weights (no extra width_ratio factor).
    } else if (name.find("_proj") != std::string::npos) {
      double fan_in = p.dim() >= 2 ? static_cast<double>(p.size(1)) : static_cast<double>(p.size(0));
      double std_val = 1.0 / std::sqrt(fan_in);
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);

    // --- Input matrices (QKV, gate_up, w1, w3): standard 1/sqrt(fan_in) ---
    // Default bucket: anything that takes a stream-width input. Standard
    // He-like init at 1/sqrt(fan_in).
    } else {
      double fan_in = p.dim() >= 2 ? static_cast<double>(p.size(1)) : static_cast<double>(p.size(0));
      double std_val = 1.0 / std::sqrt(fan_in);
      trunc_normal_(p, 0.0, std_val, -3 * std_val, 3 * std_val, g);
    }
  }

  // Diagnostic: log total parameter count so the user sees µP touched
  // everything they expected.
  int64_t total_params = 0;
  for (auto& p : model.parameters()) total_params += p.numel();
  std::cout << "[µP] Initialized " << total_params << " parameters\n";
}

/// Compute µP per-group learning-rate multipliers for the optimizer.
/// Returned multipliers are applied on top of the base LR when building
/// optimizer parameter groups (e.g. AdamW with one group per param category).
MuPLRMultiplier get_mup_lr_multipliers(
    const TransformerConfig& cfg,
    const MuPConfig& mup_cfg) {

  double target_width = mup_cfg.target_width > 0 ? mup_cfg.target_width : static_cast<double>(cfg.d_model);
  double width_ratio = mup_cfg.base_width / target_width;

  return MuPLRMultiplier{
    .embedding_mult = 1.0,           // Embeddings: standard LR (width-independent).
    .hidden_mult = width_ratio,      // Hidden: scale down LR with width.
    .output_mult = width_ratio,      // Output: also scale down (Adam-friendly variant).
  };
}

}  // namespace olmo_cpp
