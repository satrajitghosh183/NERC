/**
 * src/backend/gpu_sample.cpp
 *
 * Implementation of the device-resident sampler. See gpu_sample.hpp.
 *
 * Every operation runs on the device the logits live on. The id list for the
 * repetition penalty is filtered on the host (it's a small std::vector) and
 * uploaded once; from there it's a gather/scatter on-device. top-p reuses the
 * bucket-radix filter (CUDA kernel on GPU, O(V) host pass on CPU), and the
 * draw is torch::multinomial on the device generator. The single D->H is the
 * returned token id.
 */

#include "olmo_cpp/backend/gpu_sample.hpp"
#include "olmo_cpp/backend/topp_radix.hpp"

#include <limits>

namespace olmo_cpp {

int64_t gpu_sample(torch::Tensor logits,
                   double temperature,
                   int64_t top_k,
                   double top_p,
                   const std::vector<int64_t>& rep_tokens,
                   double rep_penalty) {
  torch::NoGradGuard no_grad;

  // Work in fp32 (safe for topp_radix_filter and multinomial; bf16/fp16 logits
  // upcast cheaply). .flatten() tolerates [V], [1,V], [1,1,V] callers.
  auto lg = logits.flatten().to(torch::kFloat32);
  const int64_t vocab = lg.size(0);

  // (a) Repetition penalty — gather seen-token logits, divide/multiply, scatter
  //     back. Each seen token is penalized once (index_put with the gathered
  //     values). The id list is filtered + uploaded once; no host round-trip of
  //     the [V] logits.
  if (rep_penalty != 1.0 && !rep_tokens.empty()) {
    std::vector<int64_t> valid;
    valid.reserve(rep_tokens.size());
    for (int64_t id : rep_tokens) {
      if (id >= 0 && id < vocab) valid.push_back(id);
    }
    if (!valid.empty()) {
      auto ids = torch::tensor(
          valid, torch::TensorOptions().dtype(torch::kInt64).device(lg.device()));
      auto vals = lg.index_select(0, ids);
      auto adj = torch::where(vals > 0, vals / rep_penalty, vals * rep_penalty);
      lg = lg.index_put({ids}, adj);  // out-of-place; caller's tensor untouched
    }
  }

  // (b) Greedy — rep-penalty-modified argmax. No RNG, no softmax.
  if (temperature <= 0.0) {
    return lg.argmax(-1).item<int64_t>();
  }

  // (c) Temperature.
  if (temperature != 1.0) {
    lg = lg / temperature;
  }

  // (d) Top-k mask.
  if (top_k > 0 && top_k < vocab) {
    auto thr = std::get<0>(lg.topk(top_k)).index({top_k - 1});
    lg = torch::where(
        lg < thr,
        torch::full_like(lg, -std::numeric_limits<float>::infinity()), lg);
  }

  auto probs = torch::softmax(lg, -1);

  // (e) Top-p (nucleus) — bucket-radix filter on the probabilities. CUDA kernel
  //     on GPU, O(V) pass on CPU. In place.
  if (top_p < 1.0) {
    topp_radix_filter(probs, static_cast<float>(top_p), /*min_p=*/1.0e-5f);
    // Degenerate guard (extreme top-p/top-k can zero everything): fall back to
    // the most-probable token. One 4-byte sync, only on the filtered path.
    if (probs.sum().item<float>() <= 0.0f) {
      return probs.argmax(-1).item<int64_t>();
    }
  }

  // (f) Draw on-device. Uses torch's (CUDA) generator — call torch::manual_seed
  //     upstream for reproducibility. The token id is the only value returned.
  return torch::multinomial(probs, /*num_samples=*/1).item<int64_t>();
}

int64_t gpu_argmax(torch::Tensor logits,
                   const std::vector<int64_t>& rep_tokens,
                   double rep_penalty) {
  return gpu_sample(logits, /*temperature=*/0.0, /*top_k=*/0, /*top_p=*/1.0,
                    rep_tokens, rep_penalty);
}

}  // namespace olmo_cpp
