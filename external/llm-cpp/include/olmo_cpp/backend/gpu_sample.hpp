#pragma once

/**
 * include/olmo_cpp/backend/gpu_sample.hpp
 *
 * Device-resident token sampler — the single sampling entry point for every
 * decode path (chat, linear-chain speculative, tree speculative, draft-model
 * speculative, and the serving scheduler).
 *
 * The whole post-process — repetition penalty, temperature, top-k, softmax,
 * top-p (nucleus) — runs ON the device the logits already live on. Only the
 * chosen token id (8 bytes) crosses the D->H boundary. Compare the legacy
 * host path, which copied the full [V] logits (~200 KB/token at V=50304) to
 * the CPU after every forward and did the filtering there.
 *
 * On CUDA: top-p routes through the bucket-radix kernel (topp_radix_filter_cuda)
 * and the draw through torch::multinomial on the device generator. On CPU/MPS
 * the identical ATen ops run in place — same code, no host round-trip added.
 */

#include <torch/torch.h>
#include <cstdint>
#include <vector>

namespace olmo_cpp {

/// Sample one token id from a logits vector, keeping everything on-device.
///
///   logits      : [V] (or any shape flattenable to [V]) on CPU/CUDA/MPS.
///   temperature : <= 0  → greedy (rep-penalty-modified argmax, no RNG).
///   top_k       : > 0   → keep only the top-k logits before softmax.
///   top_p       : < 1.0 → nucleus filter on the probabilities.
///   rep_tokens  : previously-emitted ids; each is penalized once.
///   rep_penalty : 1.0 → no penalty; >1 down-weights seen tokens.
///
/// Returns the sampled token id. The only device->host transfer is the id.
int64_t gpu_sample(torch::Tensor logits,
                   double temperature,
                   int64_t top_k,
                   double top_p,
                   const std::vector<int64_t>& rep_tokens,
                   double rep_penalty);

/// Greedy convenience: rep-penalty-modified argmax with no RNG. Equivalent to
/// gpu_sample(logits, /*temperature=*/0, 0, 1.0, rep_tokens, rep_penalty).
int64_t gpu_argmax(torch::Tensor logits,
                   const std::vector<int64_t>& rep_tokens = {},
                   double rep_penalty = 1.0);

}  // namespace olmo_cpp
