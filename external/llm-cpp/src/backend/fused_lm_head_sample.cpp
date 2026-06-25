/**
 * src/backend/fused_lm_head_sample.cpp
 *
 * CPU reference + dispatch for the fused LM-head + Gumbel-max sampler.
 * fast-inference [6].
 *
 * The CPU path mirrors the CUDA kernel's algorithm exactly so a unit
 * test can validate the GPU output against a slow-but-correct host
 * implementation. Same Philox-4x32-10, same Gumbel transform, same
 * pack/unpack, same argmax tie-break.
 */

#include "olmo_cpp/backend/fused_lm_head_sample.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>
#include <torch/torch.h>

namespace olmo_cpp {

namespace {

// CPU mirror of the CUDA Philox. Bit-identical.
inline uint32_t mulhi32_cpu(uint32_t a, uint32_t b, uint32_t* lo) {
  uint64_t prod = static_cast<uint64_t>(a) * b;
  *lo = static_cast<uint32_t>(prod);
  return static_cast<uint32_t>(prod >> 32);
}

inline void philox_round_cpu(uint32_t c[4], uint32_t k[2]) {
  uint32_t lo0; uint32_t hi0 = mulhi32_cpu(0xD2511F53u, c[0], &lo0);
  uint32_t lo1; uint32_t hi1 = mulhi32_cpu(0xCD9E8D57u, c[2], &lo1);
  uint32_t n0 = hi1 ^ c[1] ^ k[0];
  uint32_t n1 = lo1;
  uint32_t n2 = hi0 ^ c[3] ^ k[1];
  uint32_t n3 = lo0;
  c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
}

inline float philox_uniform_cpu(uint64_t seed, uint32_t position, uint32_t idx) {
  uint32_t c[4] = { idx, position, 0u, 0u };
  uint32_t k[2] = { static_cast<uint32_t>(seed),
                    static_cast<uint32_t>(seed >> 32) };
  for (int r = 0; r < 10; ++r) {
    philox_round_cpu(c, k);
    k[0] += 0x9E3779B9u;
    k[1] += 0xBB67AE85u;
  }
  uint32_t bits = c[0] >> 8;
  if (bits == 0) bits = 1;
  return static_cast<float>(bits) * (1.0f / static_cast<float>(1 << 24));
}

inline float gumbel_cpu(uint64_t seed, uint32_t position, uint32_t idx) {
  float u = philox_uniform_cpu(seed, position, idx);
  return -std::log(-std::log(u));
}

}  // namespace

int64_t fused_lm_head_sample_cpu(
    torch::Tensor hidden,
    torch::Tensor W_U,
    float temperature,
    uint64_t seed,
    uint32_t position,
    const std::vector<int64_t>& rep_tokens,
    double rep_penalty) {

  TORCH_CHECK(hidden.is_cpu() && W_U.is_cpu(),
              "fused_lm_head_sample_cpu: tensors must be on CPU");
  TORCH_CHECK(hidden.dim() == 1, "hidden must be 1-D [H]");
  TORCH_CHECK(W_U.dim() == 2,    "W_U must be 2-D [V, H]");
  TORCH_CHECK(W_U.size(1) == hidden.size(0), "hidden dim mismatch");
  TORCH_CHECK(temperature > 0.0f, "temperature must be > 0");

  auto h_c = hidden.contiguous().to(torch::kFloat32);
  auto W_c = W_U.contiguous().to(torch::kFloat32);
  const int64_t V = W_c.size(0);
  const int64_t H = W_c.size(1);
  const float inv_T = 1.0f / temperature;

  const float* h_ptr = h_c.data_ptr<float>();
  const float* W_ptr = W_c.data_ptr<float>();

  // Repetition-penalty lookup. Applied to the raw logit BEFORE temperature +
  // Gumbel — bit-identical to the CUDA kernel (same float pf, same conditional).
  const bool do_rep = (rep_penalty != 1.0) && !rep_tokens.empty();
  const float pf = static_cast<float>(rep_penalty);
  std::unordered_set<int64_t> seen;
  if (do_rep) {
    for (int64_t t : rep_tokens) if (t >= 0 && t < V) seen.insert(t);
  }

  float best_score = -std::numeric_limits<float>::infinity();
  int64_t best_idx = -1;

  for (int64_t i = 0; i < V; ++i) {
    const float* w_row = W_ptr + i * H;
    float l = 0.0f;
    for (int64_t h = 0; h < H; ++h) l += w_row[h] * h_ptr[h];
    if (do_rep && seen.count(i)) l = (l > 0.0f) ? (l / pf) : (l * pf);
    float g = gumbel_cpu(seed, position, static_cast<uint32_t>(i));
    float s = l * inv_T + g;
    if (s > best_score) {
      best_score = s;
      best_idx   = i;
    }
  }
  return best_idx;
}

int64_t fused_lm_head_sample(
    torch::Tensor hidden,
    torch::Tensor W_U,
    float temperature,
    uint64_t seed,
    uint32_t position,
    const std::vector<int64_t>& rep_tokens,
    double rep_penalty) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (hidden.is_cuda()) {
    return fused_lm_head_sample_cuda(hidden, W_U, temperature, seed, position,
                                     rep_tokens, rep_penalty);
  }
#endif
  return fused_lm_head_sample_cpu(hidden, W_U, temperature, seed, position,
                                  rep_tokens, rep_penalty);
}

}  // namespace olmo_cpp
