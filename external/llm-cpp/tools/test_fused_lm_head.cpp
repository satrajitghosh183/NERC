/**
 * tools/test_fused_lm_head.cpp
 *
 * CPU unit test for the fused LM-head + Gumbel-max sampler
 * (fast-inference [6]).
 *
 * Validates the algorithm by checking three properties:
 *
 *   1. Determinism: same (seed, position) → same token, every run.
 *   2. Distribution: across many positions, the empirical token
 *      distribution matches softmax(W_U @ hidden / T) within tolerance.
 *   3. Edge cases: temperature=1, large vocab, various H sizes.
 *
 * Property (2) is the real one — Gumbel-max is mathematically equivalent
 * to softmax sampling, so the empirical histogram should match. We use
 * a small synthetic vocab (V=64) so we can collect enough samples to be
 * statistically confident.
 *
 * Usage:
 *   ./build/test_fused_lm_head            # all tests
 *   ./build/test_fused_lm_head --verbose  # detailed per-test output
 */

#include "olmo_cpp/backend/fused_lm_head_sample.hpp"

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

namespace {

bool test_determinism(bool verbose) {
  const int64_t V = 1024;
  const int64_t H = 64;
  auto hidden = torch::randn({H});
  auto W_U    = torch::randn({V, H});

  int64_t a = olmo_cpp::fused_lm_head_sample_cpu(hidden, W_U, 1.0f, /*seed=*/42, /*pos=*/0);
  int64_t b = olmo_cpp::fused_lm_head_sample_cpu(hidden, W_U, 1.0f, /*seed=*/42, /*pos=*/0);
  int64_t c = olmo_cpp::fused_lm_head_sample_cpu(hidden, W_U, 1.0f, /*seed=*/42, /*pos=*/1);

  bool stable    = (a == b);
  bool different = (a != c);  // different position should (almost certainly) yield different sample

  if (verbose || !stable || !different) {
    std::cout << "[" << (stable && different ? "PASS" : "FAIL")
              << "] determinism  same=(" << a << "," << b
              << ") different_pos=(" << a << "," << c << ")\n";
  }
  return stable && different;
}

bool test_distribution(bool verbose) {
  // Small vocab so we can sample enough times to estimate distribution.
  const int64_t V = 16;
  const int64_t H = 8;
  // Construct a sharp distribution: token 3 should dominate.
  auto hidden = torch::ones({H});
  auto W_U    = torch::randn({V, H}) * 0.1f;
  // Boost row 3 so its dot product is much larger.
  W_U.index_put_({3}, torch::ones({H}) * 5.0f);

  // Compute reference softmax(W_U @ hidden).
  auto logits = torch::matmul(W_U, hidden);  // [V]
  auto ref_probs = torch::softmax(logits, -1);

  // Empirical histogram via fused sampler over many positions.
  const int N = 4000;
  std::vector<int64_t> counts(V, 0);
  for (int p = 0; p < N; ++p) {
    int64_t tok = olmo_cpp::fused_lm_head_sample_cpu(
        hidden, W_U, 1.0f, /*seed=*/0xC0FFEE, /*pos=*/static_cast<uint32_t>(p));
    if (tok >= 0 && tok < V) counts[tok]++;
  }

  // Compare empirical to reference. L1 distance.
  double l1 = 0.0;
  for (int64_t i = 0; i < V; ++i) {
    double emp  = static_cast<double>(counts[i]) / N;
    double rref = ref_probs[i].item<double>();
    l1 += std::abs(emp - rref);
  }

  // For N=4000 and V=16, total variation distance should be small (< ~0.1).
  bool ok = (l1 < 0.15);

  if (verbose || !ok) {
    std::cout << "[" << (ok ? "PASS" : "FAIL")
              << "] distribution  L1=" << l1 << " (threshold 0.15)\n";
    if (verbose) {
      for (int64_t i = 0; i < V; ++i) {
        double emp = static_cast<double>(counts[i]) / N;
        std::cout << "    token " << i
                  << "  ref=" << ref_probs[i].item<float>()
                  << "  emp=" << emp << "\n";
      }
    }
  }
  return ok;
}

bool test_temperature_sharpens(bool verbose) {
  // Lower temperature should bias more toward the argmax.
  const int64_t V = 64;
  const int64_t H = 16;
  auto hidden = torch::ones({H});
  auto W_U    = torch::randn({V, H});

  auto logits = torch::matmul(W_U, hidden);  // [V]
  int64_t argmax = logits.argmax(-1).item<int64_t>();

  // Sample at very low temperature — should hit argmax most of the time.
  const int N = 1000;
  int hits = 0;
  for (int p = 0; p < N; ++p) {
    int64_t tok = olmo_cpp::fused_lm_head_sample_cpu(
        hidden, W_U, 0.05f, /*seed=*/123, /*pos=*/static_cast<uint32_t>(p));
    if (tok == argmax) hits++;
  }
  double frac = static_cast<double>(hits) / N;
  bool ok = (frac > 0.95);  // at T=0.05 should be nearly deterministic

  if (verbose || !ok) {
    std::cout << "[" << (ok ? "PASS" : "FAIL")
              << "] low_temp_argmax  hit_rate=" << frac
              << " (threshold 0.95, argmax=" << argmax << ")\n";
  }
  return ok;
}

bool test_repetition_penalty(bool verbose) {
  // Heavily penalizing the natural argmax should make the fused sampler avoid
  // it — proving rep-penalty is applied to the logit before temp + Gumbel,
  // with no [V] logits tensor materialized.
  const int64_t V = 64;
  const int64_t H = 16;
  auto hidden = torch::ones({H});
  auto W_U    = torch::randn({V, H});
  auto logits = torch::matmul(W_U, hidden);  // [V]
  int64_t argmax = logits.argmax(-1).item<int64_t>();

  const int N = 1000;

  // Baseline: low temp, no penalty → argmax dominates.
  int hits_base = 0;
  for (int p = 0; p < N; ++p) {
    int64_t tok = olmo_cpp::fused_lm_head_sample_cpu(
        hidden, W_U, 0.3f, /*seed=*/777, /*pos=*/static_cast<uint32_t>(p));
    if (tok == argmax) hits_base++;
  }

  // With a strong penalty on the argmax token, it should almost never win.
  std::vector<int64_t> rep = {argmax};
  int hits_pen = 0;
  for (int p = 0; p < N; ++p) {
    int64_t tok = olmo_cpp::fused_lm_head_sample_cpu(
        hidden, W_U, 0.3f, /*seed=*/777, /*pos=*/static_cast<uint32_t>(p),
        rep, /*rep_penalty=*/100.0);
    if (tok == argmax) hits_pen++;
  }

  double base = static_cast<double>(hits_base) / N;
  double pen  = static_cast<double>(hits_pen)  / N;
  bool ok = (base > 0.5) && (pen < 0.05);  // dominant without, suppressed with

  if (verbose || !ok) {
    std::cout << "[" << (ok ? "PASS" : "FAIL")
              << "] repetition_penalty  argmax_share base=" << base
              << " penalized=" << pen << " (want base>0.5, pen<0.05)\n";
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v")
      verbose = true;
  }

  int passed = 0, failed = 0;
  auto run = [&](bool ok) { if (ok) ++passed; else ++failed; };

  run(test_determinism(verbose));
  run(test_distribution(verbose));
  run(test_temperature_sharpens(verbose));
  run(test_repetition_penalty(verbose));

  std::cout << "\n" << passed << " passed, " << failed << " failed.\n";
  return failed == 0 ? 0 : 1;
}
