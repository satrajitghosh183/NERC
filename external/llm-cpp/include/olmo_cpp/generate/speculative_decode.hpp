#pragma once

/**
 * include/olmo_cpp/generate/speculative_decode.hpp
 *
 * Speculative decoding driver that uses the model's *own* Multi-Token
 * Prediction (MTP) heads as the draft model, removing the need for a
 * separate small drafter (Medusa/EAGLE-style "self-speculation"). Provides
 * a single-step API and a full sequence-level generate. Templated over the
 * concrete model type so the standard `Transformer` and the optimized
 * `FusedTransformer` can both be plugged in.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/model/transformer.hpp / fused_transformer.hpp: ModelType
 *     concept; speculative_decode_step calls forward_backbone, apply_lm_head,
 *     forward_mtp_draft, n_layers, num_mtp_heads on the model.
 *   - olmo_cpp/model/kv_cache.hpp: KVCache reused across draft & verify.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/generate/speculative_decode.cpp: definitions + explicit template
 *     instantiations for both model types.
 *   (No direct callers in src/main or chat tools located via quick grep —
 *   the API is exposed for the chat / inference benchmarks.)
 *
 * --- Role in training pipeline ---
 *   Strictly inference-side. Demonstrates that the MTP heads we already pay
 *   ~5% training cost for buy us 2-4x decoding throughput, which is one of
 *   the headline numbers for the paper.
 */

#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"
#include "olmo_cpp/model/kv_cache.hpp"
#include <torch/torch.h>
#include <vector>
#include <functional>

namespace olmo_cpp {

/// Speculative decoding using Multi-Token Prediction (MTP) heads.
///
/// The key insight: MTP heads are trained to predict tokens at positions
/// t+1, t+2, ..., t+K alongside the main next-token prediction. During
/// inference, we use these as a "draft" model to propose K candidate tokens,
/// then verify all K in a single forward pass of the main model.
///
/// Speedup: Instead of K sequential forward passes (one per token),
/// we do 1 MTP draft (cheap, reuses cached hidden states) + 1 verification
/// forward pass. If acceptance rate is high (typically 60-80% for well-trained
/// MTP heads), we generate 2-4x faster.
///
/// For the paper: This is "free" speculative decoding — no separate draft model
/// needed. The MTP heads add <5% training cost but enable 2-4x inference speedup.

/// User-facing knobs for sampling and the speculation budget.
struct SpeculativeConfig {
  int64_t max_draft_tokens = 0;   ///< 0 = use all available MTP heads
  double temperature = 1.0;       ///< softmax temperature (1=neutral)
  int64_t top_k = 50;             ///< 0 disables top-k filtering
  double top_p = 0.95;            ///< 1.0 disables nucleus filtering
  bool greedy = false;            ///< true => deterministic argmax sampling
};

/// Stats and tokens produced by one decode step or a full generate.
struct SpeculativeResult {
  std::vector<int64_t> tokens;        ///< accepted (and final) token ids
  int64_t num_draft_tokens = 0;       ///< proposed by the MTP heads
  int64_t num_accepted = 0;           ///< matched the verifier's prediction
  double acceptance_rate = 0.0;       ///< accepted / draft (paper headline)
  int64_t total_forward_passes = 0;   ///< full backbone runs (cost metric)
  double tokens_per_forward = 0.0;    ///< throughput surrogate (>1 = win)
};

/// Standalone sampler: applies temperature, top-k, top-p filters then either
/// argmax (`greedy`) or multinomial-samples a token id.
int64_t sample_token(torch::Tensor logits, double temperature = 1.0,
                     int64_t top_k = 50, double top_p = 0.95, bool greedy = false);

/// Speculative decode one "step" (may produce 1..K+1 tokens).
///
/// 1. Get hidden state from last generated position
/// 2. Use MTP heads to draft K candidates
/// 3. Run verification forward pass on all candidates
/// 4. Accept prefix that matches, resample the first rejection
///
/// Works with both Transformer and FusedTransformer via template.
template <typename ModelType>
SpeculativeResult speculative_decode_step(
    ModelType& model,
    KVCache& kv_cache,
    int64_t last_token,
    torch::Device device,
    const SpeculativeConfig& config = {});

/// Generate a full sequence using speculative decoding.
template <typename ModelType>
SpeculativeResult speculative_generate(
    ModelType& model,
    torch::Tensor prompt_ids,   // [1, prompt_len]
    int64_t max_new_tokens,
    torch::Device device,
    const SpeculativeConfig& config = {},
    std::function<bool(int64_t)> stop_fn = nullptr);  // return true to stop

}  // namespace olmo_cpp
