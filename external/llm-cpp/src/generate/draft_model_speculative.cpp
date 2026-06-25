/**
 * src/generate/draft_model_speculative.cpp
 *
 * Two-model speculative decoding (fast-inference [17]).
 *
 * Implementation parallels speculative_decode_step in chat.cpp but uses
 * a separate draft model instead of MTP heads:
 *   1. Draft model autoregressively generates k tokens (k forward passes
 *      on the draft model + sample at each).
 *   2. Target model verifies all k positions in ONE batched forward.
 *   3. Accept matching prefix.
 *
 * DRAFT — wiring into chat.cpp's main loop is a separate task. The
 * caller must supply both models already loaded onto the same device
 * with compatible tokenizers.
 */

#include "olmo_cpp/generate/draft_model_speculative.hpp"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace olmo_cpp {

namespace {

int64_t argmax_1d(torch::Tensor logits_1d) {
  return logits_1d.argmax(-1).item<int64_t>();
}

// Apply temperature, top_k, top_p filtering and produce a normalized
// std::vector<double> probability distribution. Used at both draft
// generation time and target-side acceptance time so the two
// distributions are directly comparable for rejection sampling.
//
// Behavior matches sample_logits() in tools/chat.cpp; pulled into this
// translation unit to keep speculative decoding self-contained.
std::vector<double> filtered_probs(torch::Tensor logits_1d,
                                   double temperature,
                                   int64_t top_k,
                                   double top_p) {
  auto x = logits_1d.cpu().contiguous().to(torch::kFloat32);
  const int64_t V = x.size(0);
  if (temperature > 0.0) {
    x = x / temperature;
  }
  if (top_k > 0 && top_k < V) {
    auto [vals, _] = x.topk(top_k);
    auto thresh = vals.index({top_k - 1}).item<float>();
    x = torch::where(x < thresh,
                     torch::full_like(x, -std::numeric_limits<float>::infinity()),
                     x);
  }
  auto probs = torch::softmax(x, -1);
  if (top_p < 1.0) {
    auto [sorted, sorted_idx] = probs.sort(-1, /*descending=*/true);
    auto cumul = sorted.cumsum(-1);
    auto mask = (cumul - sorted) > top_p;
    sorted.index_put_({mask}, 0.0f);
    probs.zero_();
    probs.scatter_(-1, sorted_idx, sorted);
    auto sum = probs.sum().item<float>();
    if (sum > 0) probs = probs / sum;
  }
  std::vector<double> out(static_cast<size_t>(V));
  auto p_ptr = probs.data_ptr<float>();
  for (int64_t i = 0; i < V; ++i) out[static_cast<size_t>(i)] = p_ptr[i];
  return out;
}

int64_t sample_from(const std::vector<double>& probs, std::mt19937& rng) {
  std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
  return dist(rng);
}

}  // namespace

int64_t draft_model_speculative_step(
    DraftModelSpeculativeState& state,
    std::vector<int64_t>& all_tokens,
    torch::Device device,
    double temperature,
    int64_t top_k,
    double top_p,
    double /*repetition_penalty*/,    // applied at sample time by caller; not used here
    std::mt19937& rng,
    BPETokenizer& tokenizer) {

  torch::NoGradGuard no_grad;
  const int64_t eos_id = static_cast<int64_t>(tokenizer.eos_id());
  const bool greedy = (temperature <= 0.0);

  // Step 1: draft model produces draft_len tokens autoregressively. We
  // also keep the full probability distribution at each draft step so
  // rejection sampling at step 3 has the q_i side of min(1, p_i/q_i).
  std::vector<int64_t> drafts;
  std::vector<std::vector<double>> draft_probs;  // q distributions per step
  drafts.reserve(static_cast<size_t>(state.draft_len));
  draft_probs.reserve(static_cast<size_t>(state.draft_len));

  int64_t cur = all_tokens.back();
  // M6: one reused [1,1] device buffer for the draft inputs — fill_ each step
  // instead of a per-draft-token alloc + H->D.
  auto inp = torch::empty({1, 1},
                          torch::TensorOptions().dtype(torch::kInt64).device(device));
  for (int64_t k = 0; k < state.draft_len; ++k) {
    inp.fill_(cur);
    auto logits = (*state.draft_model)->forward(inp, c10::nullopt, -100, &state.draft_kv);
    auto next = logits.select(1, 0).squeeze(0);

    int64_t tok;
    if (greedy) {
      tok = argmax_1d(next);
      draft_probs.emplace_back();  // not used in greedy path
    } else {
      auto probs = filtered_probs(next, temperature, top_k, top_p);
      tok = sample_from(probs, rng);
      draft_probs.push_back(std::move(probs));
    }
    drafts.push_back(tok);
    if (tok == eos_id) break;
    cur = tok;
  }
  state.total_drafted += static_cast<int64_t>(drafts.size());

  // Step 2: target model verifies [last_token, draft_0, ..., draft_{k-1}]
  // in a single batched forward.
  std::vector<int64_t> verify_in;
  verify_in.reserve(1 + drafts.size());
  verify_in.push_back(all_tokens.back());
  for (auto d : drafts) verify_in.push_back(d);

  auto vinp = torch::tensor(at::IntArrayRef(verify_in.data(), verify_in.size()),
                            torch::kInt64).unsqueeze(0).to(device);

  const int64_t target_snap = state.target_kv.snapshot();
  auto vlogits = (*state.target_model)->forward(vinp, c10::nullopt, -100, &state.target_kv);
  // Greedy verification needs only the per-position argmax (k+1 ids), computed
  // on-device; only the rejection-sampling path needs the full [k+1, V]
  // distributions on host. Don't copy the logits when we won't use them.
  torch::Tensor vchoices_host, vlogits_cpu;
  if (greedy) {
    vchoices_host = vlogits.select(0, 0).argmax(-1).to(torch::kCPU).contiguous();  // [k+1]
  } else {
    vlogits_cpu = vlogits.cpu().contiguous();
  }
  const int64_t* vchoices = greedy ? vchoices_host.data_ptr<int64_t>() : nullptr;

  // Step 3: walk verify positions and decide acceptance.
  int64_t accepted = 0;

  // Position 0 is the target's distribution after consuming last_token —
  // unambiguous main prediction, always accepted.
  int64_t main_tok;
  if (greedy) {
    main_tok = vchoices[0];
  } else {
    auto p_main = filtered_probs(vlogits_cpu.select(0, 0).select(0, 0),
                                 temperature, top_k, top_p);
    main_tok = sample_from(p_main, rng);
  }
  all_tokens.push_back(main_tok);
  ++accepted;
  if (main_tok == eos_id) {
    state.total_accepted += 0;
    return accepted;
  }

  // Positions 1..k correspond to the target's distribution after
  // consuming each draft token. Rejection sampling per Chen et al. 2023:
  //   p_i = target prob of drafts[i]
  //   q_i = draft prob of drafts[i]
  //   accept with prob min(1, p_i/q_i)
  //   on reject: sample from normalize(max(0, p - q))
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  int64_t drafts_accepted = 0;
  for (int64_t k = 0; k < static_cast<int64_t>(drafts.size()); ++k) {
    if (greedy) {
      int64_t target_choice = vchoices[k + 1];
      if (target_choice == drafts[k]) {
        all_tokens.push_back(drafts[k]);
        ++accepted;
        ++drafts_accepted;
        if (drafts[k] == eos_id) break;
      } else {
        all_tokens.push_back(target_choice);
        ++accepted;
        break;
      }
    } else {
      auto vrow = vlogits_cpu.select(0, 0).select(0, k + 1);
      auto p = filtered_probs(vrow, temperature, top_k, top_p);
      const auto& q = draft_probs[static_cast<size_t>(k)];
      const double p_i = p[static_cast<size_t>(drafts[k])];
      const double q_i = q.empty() ? 1.0 : q[static_cast<size_t>(drafts[k])];
      const double accept_prob = q_i > 0.0 ? std::min(1.0, p_i / q_i) : 1.0;
      const double u = uni(rng);
      if (u < accept_prob) {
        all_tokens.push_back(drafts[k]);
        ++accepted;
        ++drafts_accepted;
        if (drafts[k] == eos_id) break;
      } else {
        // Reject. Sample from residual distribution max(0, p - q),
        // renormalized. Per Chen et al., this preserves the target
        // distribution under expectation.
        std::vector<double> resid(p.size(), 0.0);
        double sum = 0.0;
        for (size_t v = 0; v < p.size(); ++v) {
          const double qv = (v < q.size()) ? q[v] : 0.0;
          const double r = p[v] - qv;
          if (r > 0.0) { resid[v] = r; sum += r; }
        }
        int64_t new_tok;
        if (sum > 0.0) {
          for (auto& r : resid) r /= sum;
          new_tok = sample_from(resid, rng);
        } else {
          // Degenerate: p ⊆ q. Fall back to sampling from p directly.
          new_tok = sample_from(p, rng);
        }
        all_tokens.push_back(new_tok);
        ++accepted;
        break;
      }
    }
  }
  state.total_accepted += drafts_accepted;

  // KV rollbacks: keep only the accepted positions on both models.
  const int64_t target_keep = target_snap + accepted;
  if (target_keep < state.target_kv.seq_len()) state.target_kv.rollback(target_keep);

  // The draft model wrote drafts.size() positions to its KV; we accepted
  // drafts_accepted of them. Drop the rejected tail.
  const int64_t draft_drop = static_cast<int64_t>(drafts.size()) - drafts_accepted;
  if (draft_drop > 0) {
    state.draft_kv.rollback(state.draft_kv.seq_len() - draft_drop);
  }

  return accepted;
}

}  // namespace olmo_cpp
