#pragma once
/**
 * include/olmo_cpp/model/mup_init.hpp
 *
 * ─── What "muP" is ──────────────────────────────────────────────────
 *
 * µP = **M**aximal **U**pdate **P**arameterization (Yang & Hu 2021).
 * A specific recipe for how to scale init-stds, learning rates, and
 * residual gains as a function of width so that hyperparameters
 * tuned at small width transfer cleanly to much larger widths.
 *
 * Without µP, the optimal lr at d_model=256 is different from the
 * optimal lr at d_model=2048, so people sweep at every model size.
 * With µP, the same lr that worked for 256 will work for 2048,
 * which lets you do hyperparameter search cheaply on a tiny model
 * and then deploy the results.
 *
 * apply_mup_init() walks the model's parameters and applies the
 * width-aware scaling. Selected by mup=1 in the .conf.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp : reads cfg.d_model and other size knobs.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp : if use_mup is requested, calls apply_mup_init()
 *                     instead of model->init_weights() at startup.
 *   - src/model/mup_init.cpp : implementation.
 *
 * --- Role in training pipeline ---
 *   Optional one-shot at model construction. Off in the quickstart
 *   conf (mup=0); the simpler init_weights() is used.
 */

#include "olmo_cpp/config.hpp"
#include <torch/torch.h>
#include <optional>

namespace olmo_cpp {

/// µP (Maximal Update Parameterization) initialization.
///
/// Standard parameterization (SP) uses the same learning rate for all parameters,
/// which means hyperparameters don't transfer across model widths. µP scales
/// initialization and learning rates so that optimal HPs at small width
/// transfer to large width — enabling cheaper HP tuning.
///
/// Key differences from standard init:
///   - Embedding:  init_std = 1.0 (not scaled by width)
///   - Hidden:     init_std = 1/sqrt(d_model) for input, 1/d_model for output
///   - LM head:    init_std = 0 (zero init)
///   - LR scaling: width-dependent multipliers per param group
///
/// Reference: Yang et al., "Tensor Programs V: Tuning Large Neural Networks
///            via Zero-Shot Hyperparameter Transfer" (2022)
///
/// For the paper: µP enables training quality to improve with scale because
/// the same LR/init works across 33M → 7B, unlike standard parameterization
/// where you'd need to re-tune for each size.

struct MuPConfig {
  double base_width = 256.0;    // The "proxy" model width for HP transfer
  double target_width = 0.0;    // Actual model width (d_model). 0 = use config.d_model
  bool zero_init_lm_head = true;
  bool scale_output_logits = true;
  double output_logit_scale = 0.0;  // 0 = auto (base_width / target_width)
};

/// Apply µP initialization to a transformer model.
/// This modifies weights in-place according to µP scaling rules.
void apply_mup_init(
    torch::nn::Module& model,
    const TransformerConfig& cfg,
    const MuPConfig& mup_cfg,
    torch::optional<torch::Generator> gen = c10::nullopt);

/// Get µP learning rate multipliers for different parameter groups.
/// Returns a map of parameter name patterns to LR multipliers.
/// Usage: multiply the base LR by these factors for each param group.
struct MuPLRMultiplier {
  double embedding_mult;     // typically 1.0
  double hidden_mult;        // typically base_width / target_width
  double output_mult;        // typically 1.0 / target_width
};

MuPLRMultiplier get_mup_lr_multipliers(
    const TransformerConfig& cfg,
    const MuPConfig& mup_cfg);

}  // namespace olmo_cpp
