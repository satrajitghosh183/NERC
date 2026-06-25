/**
 * src/backend/topp_radix.cpp
 *
 * CPU reference + dispatch for the bucket-radix top-p kernel.
 * fast-inference [7].
 *
 * The CPU reference uses the SAME bucketing semantics as the CUDA kernel
 * so a unit test can compare outputs bit-identically (modulo float-add
 * ordering inside the histogram, which doesn't change the cutoff bucket
 * for any realistic distribution).
 */

#include "olmo_cpp/backend/topp_radix.hpp"

#include <cmath>

namespace olmo_cpp {

namespace {

constexpr int kNumBuckets = 16;

inline int prob_to_bucket(float p, float min_p) {
  if (!(p > min_p)) return kNumBuckets;
  float lp = -std::log2(p);
  int b = static_cast<int>(std::floor(lp));
  if (b < 0) b = 0;
  if (b >= kNumBuckets) b = kNumBuckets - 1;
  return b;
}

}  // namespace

void topp_radix_filter_cpu(torch::Tensor probs, float top_p, float min_p) {
  TORCH_CHECK(probs.is_cpu(), "topp_radix_filter_cpu: probs must be CPU");
  TORCH_CHECK(probs.dim() == 1, "topp_radix_filter_cpu: probs must be 1-D [V]");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32,
              "topp_radix_filter_cpu: probs must be float32");
  TORCH_CHECK(probs.is_contiguous(), "topp_radix_filter_cpu: probs must be contiguous");
  if (top_p >= 1.0f) return;
  if (top_p <= 0.0f) top_p = 1e-6f;

  const int64_t V = probs.numel();
  float* p = probs.data_ptr<float>();

  // Pass 1: histogram.
  int   count[kNumBuckets] = {0};
  float mass[kNumBuckets]  = {0.0f};
  for (int64_t i = 0; i < V; ++i) {
    int b = prob_to_bucket(p[i], min_p);
    if (b < kNumBuckets) {
      count[b] += 1;
      mass[b]  += p[i];
    }
  }

  // Pass 2: cutoff scan.
  int cutoff = kNumBuckets - 1;
  float cum = 0.0f;
  for (int i = 0; i < kNumBuckets; ++i) {
    cum += mass[i];
    if (cum >= top_p) { cutoff = i; break; }
  }

  // Pass 3: mask + sum.
  float surviving = 0.0f;
  for (int64_t i = 0; i < V; ++i) {
    int b = prob_to_bucket(p[i], min_p);
    if (b > cutoff) p[i] = 0.0f;
    else            surviving += p[i];
  }

  // Pass 4: renorm.
  if (surviving > 0.0f) {
    float inv = 1.0f / surviving;
    for (int64_t i = 0; i < V; ++i) p[i] *= inv;
  }
}

void topp_radix_filter(torch::Tensor probs, float top_p, float min_p) {
#ifdef OLMO_HAS_CUDA_KERNELS
  if (probs.is_cuda()) {
    topp_radix_filter_cuda(probs, top_p, min_p);
    return;
  }
#endif
  topp_radix_filter_cpu(probs, top_p, min_p);
}

}  // namespace olmo_cpp
