/**
 * src/generate/speculative_decode.cpp
 *
 * ─── What "speculative decoding" is ─────────────────────────────────
 *
 * At inference time, generating each output token requires a full
 * forward pass through the big model — slow. Speculative decoding
 * (Leviathan, 2022) speeds this up by using a SMALL "draft" model to
 * propose K candidate next tokens cheaply, then having the BIG model
 * **verify** all K in a single batched forward pass.
 *
 *     repeat:
 *       drafts = small_model.generate(K tokens)         // cheap
 *       logits = big_model.forward(prompt + drafts)     // one batched call
 *       for each draft, accept if it matches what big_model would
 *       have produced; on first rejection, sample one fresh token
 *       from big_model and continue.
 *
 * The end result is BIT-FOR-BIT identical to greedy or sampled
 * generation from the big model alone, but typically 2-3x faster
 * because the big-model forward is K times cheaper per generated
 * token whenever the small model gets at least some drafts right.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/generate/speculative_decode.hpp : declaration.
 *   - olmo_cpp/profiler.hpp                    : ProfileScope around
 *                                                 draft / verify.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - tools/chat.cpp: optional fast-path for interactive generation.
 *
 * --- Role in training pipeline ---
 *   Inference-time only. Not used during training.
 */
#include "olmo_cpp/generate/speculative_decode.hpp"
#include "olmo_cpp/backend/gpu_sample.hpp"
#include "olmo_cpp/profiler.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace olmo_cpp {

int64_t sample_token(torch::Tensor logits, double temperature,
                     int64_t top_k, double top_p, bool greedy) {
  // Single device-resident sampler for every decode path: rep-penalty (none
  // here), temperature, top-k, softmax, top-p (O(V) bucket-radix kernel — not
  // the O(V log V) sort), and the draw all stay on-device; only the token id
  // crosses D->H. greedy == temperature 0.
  return gpu_sample(logits, greedy ? 0.0 : temperature, top_k, top_p,
                    /*rep_tokens=*/{}, /*rep_penalty=*/1.0);
}

// Explicit template instantiation for both model types
template <typename ModelType>
SpeculativeResult speculative_decode_step(
    ModelType& model,
    KVCache& kv_cache,
    int64_t last_token,
    torch::Device device,
    const SpeculativeConfig& config) {

  ProfileScope scope("spec_decode_step");
  SpeculativeResult result;

  int64_t num_mtp = model->num_mtp_heads();
  if (num_mtp == 0) {
    // No MTP heads — fall back to standard autoregressive
    auto input = torch::tensor({last_token}, torch::kLong).unsqueeze(0).to(device);
    auto h = model->forward_backbone(input, &kv_cache);
    auto logits = model->apply_lm_head(h.select(1, -1));  // last position
    int64_t token = sample_token(logits, config.temperature, config.top_k,
                                  config.top_p, config.greedy);
    result.tokens.push_back(token);
    result.total_forward_passes = 1;
    result.tokens_per_forward = 1.0;
    return result;
  }

  int64_t K = config.max_draft_tokens > 0
      ? std::min(config.max_draft_tokens, num_mtp)
      : num_mtp;

  // ── Step 1: Get hidden state from last token ──
  auto input = torch::tensor({last_token}, torch::kLong).unsqueeze(0).to(device);

  torch::Tensor hidden;
  {
    ProfileScope fwd_scope("spec_decode_draft_fwd");
    hidden = model->forward_backbone(input, &kv_cache);
  }

  // Main model's next-token prediction
  auto main_logits = model->apply_lm_head(hidden.select(1, -1));
  int64_t main_token = sample_token(main_logits, config.temperature, config.top_k,
                                     config.top_p, config.greedy);
  result.tokens.push_back(main_token);

  // ── Step 2: Use MTP heads to draft K more candidates ──
  std::vector<int64_t> draft_tokens;
  {
    ProfileScope draft_scope("spec_decode_mtp_draft");
    auto last_hidden = hidden.select(1, -1);  // [1, d_model]
    auto draft_logits = model->forward_mtp_draft(last_hidden);

    for (int64_t k = 0; k < K && k < static_cast<int64_t>(draft_logits.size()); ++k) {
      int64_t draft_tok = sample_token(draft_logits[k], config.temperature,
                                        config.top_k, config.top_p, config.greedy);
      draft_tokens.push_back(draft_tok);
    }
  }
  result.num_draft_tokens = static_cast<int64_t>(draft_tokens.size());

  if (draft_tokens.empty()) {
    result.total_forward_passes = 1;
    result.tokens_per_forward = 1.0;
    return result;
  }

  // ── Step 3: Verify draft tokens with one batched forward pass ──
  // Build verification sequence: [main_token, draft_0, draft_1, ..., draft_{K-1}]
  std::vector<int64_t> verify_seq = {main_token};
  verify_seq.insert(verify_seq.end(), draft_tokens.begin(), draft_tokens.end());

  auto verify_input = torch::tensor(verify_seq, torch::kLong).unsqueeze(0).to(device);

  torch::Tensor verify_hidden;
  {
    ProfileScope verify_scope("spec_decode_verify");
    verify_hidden = model->forward_backbone(verify_input, &kv_cache);
  }

  // ── Step 4: Accept matching prefix ──
  // M7: batch the LM-head over ALL verify positions in ONE GEMM (was K separate
  // apply_lm_head calls), and for greedy do ONE argmax over the vocab, bringing
  // back only the [n+1] chosen ids (was K per-position .item() D->H syncs).
  const int64_t n_draft = static_cast<int64_t>(draft_tokens.size());
  auto all_logits = model->apply_lm_head(verify_hidden).select(0, 0);  // [n+1, V]
  int64_t accepted = 0;

  if (config.greedy) {
    auto choices = all_logits.argmax(-1).to(torch::kCPU).contiguous();  // [n+1]
    const int64_t* mc = choices.template data_ptr<int64_t>();
    for (int64_t i = 0; i < n_draft; ++i) {
      result.tokens.push_back(mc[i]);
      if (mc[i] == draft_tokens[i]) accepted++;
      else break;
    }
    if (accepted == n_draft) result.tokens.push_back(mc[n_draft]);  // bonus token
  } else {
    for (int64_t i = 0; i < n_draft; ++i) {
      auto pos_logits = all_logits.select(0, i);  // [V] view — no extra GEMM
      auto probs = torch::softmax(pos_logits / config.temperature, -1);
      double draft_prob = probs[draft_tokens[i]].template item<double>();
      int64_t verified_token = (draft_prob > 0.01)
          ? draft_tokens[i]
          : sample_token(pos_logits, config.temperature, config.top_k, config.top_p, false);
      result.tokens.push_back(verified_token);
      if (verified_token == draft_tokens[i]) accepted++;
      else break;
    }
    if (accepted == n_draft) {
      int64_t bonus_token = sample_token(all_logits.select(0, n_draft),
                                          config.temperature, config.top_k, config.top_p, false);
      result.tokens.push_back(bonus_token);
    }
  }

  result.num_accepted = accepted;
  result.acceptance_rate = draft_tokens.empty() ? 0.0 :
      static_cast<double>(accepted) / static_cast<double>(draft_tokens.size());
  result.total_forward_passes = 2;  // 1 draft + 1 verify
  result.tokens_per_forward = static_cast<double>(result.tokens.size()) / 2.0;

  return result;
}

template <typename ModelType>
SpeculativeResult speculative_generate(
    ModelType& model,
    torch::Tensor prompt_ids,
    int64_t max_new_tokens,
    torch::Device device,
    const SpeculativeConfig& config,
    std::function<bool(int64_t)> stop_fn) {

  ProfileScope scope("spec_generate_total");

  model->eval();
  torch::NoGradGuard no_grad;

  // Initialize KV cache
  KVCache kv_cache(model->n_layers(), device);

  // Prefill: process the prompt
  {
    ProfileScope prefill("spec_generate_prefill");
    auto h = model->forward_backbone(prompt_ids.to(device), &kv_cache);
    auto logits = model->apply_lm_head(h.select(1, -1));
    // Don't sample from prefill — the last prompt token's prediction is the first generated token
  }

  // Get first token from prompt's last position
  auto h = model->forward_backbone(prompt_ids.to(device));
  auto first_logits = model->apply_lm_head(h.select(1, -1));
  int64_t current_token = sample_token(first_logits, config.temperature,
                                         config.top_k, config.top_p, config.greedy);

  SpeculativeResult total_result;
  total_result.tokens.push_back(current_token);

  int64_t generated = 1;
  int64_t total_drafts = 0;
  int64_t total_accepted = 0;
  int64_t total_forward = 1;  // prefill counts as 1

  while (generated < max_new_tokens) {
    if (stop_fn && stop_fn(current_token)) break;

    auto step_result = speculative_decode_step(model, kv_cache, current_token, device, config);

    for (auto tok : step_result.tokens) {
      total_result.tokens.push_back(tok);
      generated++;
      if (generated >= max_new_tokens) break;
      if (stop_fn && stop_fn(tok)) break;
    }

    if (!step_result.tokens.empty()) {
      current_token = step_result.tokens.back();
    }

    total_drafts += step_result.num_draft_tokens;
    total_accepted += step_result.num_accepted;
    total_forward += step_result.total_forward_passes;
  }

  total_result.num_draft_tokens = total_drafts;
  total_result.num_accepted = total_accepted;
  total_result.acceptance_rate = total_drafts > 0 ?
      static_cast<double>(total_accepted) / total_drafts : 0.0;
  total_result.total_forward_passes = total_forward;
  total_result.tokens_per_forward = total_forward > 0 ?
      static_cast<double>(total_result.tokens.size()) / total_forward : 0.0;

  return total_result;
}

// Explicit template instantiations
template SpeculativeResult speculative_decode_step<Transformer>(
    Transformer&, KVCache&, int64_t, torch::Device, const SpeculativeConfig&);
template SpeculativeResult speculative_decode_step<FusedTransformer>(
    FusedTransformer&, KVCache&, int64_t, torch::Device, const SpeculativeConfig&);
template SpeculativeResult speculative_generate<Transformer>(
    Transformer&, torch::Tensor, int64_t, torch::Device, const SpeculativeConfig&,
    std::function<bool(int64_t)>);
template SpeculativeResult speculative_generate<FusedTransformer>(
    FusedTransformer&, torch::Tensor, int64_t, torch::Device, const SpeculativeConfig&,
    std::function<bool(int64_t)>);

}  // namespace olmo_cpp
