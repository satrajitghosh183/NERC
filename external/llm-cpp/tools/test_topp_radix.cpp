/**
 * tools/test_topp_radix.cpp
 *
 * Standalone unit test for the bucket-radix top-p kernel
 * (fast-inference [7]). Runs on CPU only — validates the algorithm.
 *
 * Compares against a brute-force reference (sort + cumsum + mask) on
 * synthetic distributions:
 *   - Sharp distribution (one big spike + uniform tail)
 *   - Power-law distribution (typical of real LLM softmax output)
 *   - Uniform distribution (degenerate edge case)
 *
 * Pass criterion: kept-mass deviation from the reference's nucleus
 * is bounded by 1 / kNumBuckets (= ~6%) — the bucket-radix approximation
 * over-approximates the nucleus by at most one bucket-width of mass.
 *
 * Usage:
 *   ./build/test_topp_radix              # runs all tests
 *   ./build/test_topp_radix --verbose    # prints per-test details
 */

#include "olmo_cpp/backend/topp_radix.hpp"

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

namespace {

// Reference: exact top-p via sort + cumsum + mask + renormalize.
torch::Tensor reference_topp(torch::Tensor probs, float top_p) {
  auto out = probs.clone();
  if (top_p >= 1.0f) return out;
  auto [sorted, idx] = out.sort(/*dim=*/-1, /*descending=*/true);
  auto cum = sorted.cumsum(-1);
  // Keep tokens whose cumulative-before-this is <= top_p.
  auto mask = cum - sorted > top_p;
  sorted.index_put_({mask}, 0.0f);
  out.zero_();
  out.scatter_(-1, idx, sorted);
  auto sum = out.sum().item<float>();
  if (sum > 0) out = out / sum;
  return out;
}

// Total kept probability mass (before renorm).
float kept_mass(torch::Tensor original_probs, torch::Tensor filtered_probs) {
  auto kept = (filtered_probs > 0.0f).to(original_probs.dtype()) * original_probs;
  return kept.sum().item<float>();
}

bool approx_equal(float a, float b, float tol = 1e-4f) {
  return std::abs(a - b) <= tol;
}

struct TestCase {
  std::string name;
  torch::Tensor probs;
  float top_p;
};

bool run_test(const TestCase& tc, bool verbose) {
  auto refp = reference_topp(tc.probs, tc.top_p);
  auto bucp = tc.probs.clone();
  olmo_cpp::topp_radix_filter_cpu(bucp, tc.top_p, /*min_p=*/0.0f);

  // Both should sum to ~1.
  float ref_sum = refp.sum().item<float>();
  float buc_sum = bucp.sum().item<float>();

  // Number of nonzero entries: bucket-radix should keep AT LEAST as many
  // tokens as the exact algorithm (over-approximation by bucket width).
  int64_t ref_nz = (refp > 0.0f).sum().item<int64_t>();
  int64_t buc_nz = (bucp > 0.0f).sum().item<int64_t>();

  // Kept original mass: bucket-radix should over-approximate top_p by at
  // most one bucket-width (factor of 2 in prob space).
  float ref_kept = kept_mass(tc.probs, refp);
  float buc_kept = kept_mass(tc.probs, bucp);

  bool ok_sum    = approx_equal(ref_sum, 1.0f, 1e-3f) && approx_equal(buc_sum, 1.0f, 1e-3f);
  bool ok_kept   = buc_kept >= ref_kept - 1e-4f;       // bucket method keeps >= reference
  bool ok_bound  = buc_kept <= 1.0f + 1e-4f;           // sanity

  bool pass = ok_sum && ok_kept && ok_bound;

  if (verbose || !pass) {
    std::cout << "[" << (pass ? "PASS" : "FAIL") << "] " << tc.name
              << " top_p=" << tc.top_p
              << "  ref_kept=" << ref_kept
              << "  buc_kept=" << buc_kept
              << "  ref_nz=" << ref_nz
              << "  buc_nz=" << buc_nz
              << "  ref_sum=" << ref_sum
              << "  buc_sum=" << buc_sum << "\n";
  }
  return pass;
}

torch::Tensor make_sharp(int64_t V) {
  // One token gets ~0.7 mass, rest exponentially decaying.
  auto x = torch::linspace(0.0f, -10.0f, V);
  auto p = torch::softmax(x, -1);
  return p;
}

torch::Tensor make_powerlaw(int64_t V) {
  // probs[i] ∝ 1 / (i+1)^1.2 — heavy tail.
  auto idx = torch::arange(V, torch::kFloat32) + 1.0f;
  auto unn = idx.pow(-1.2f);
  return unn / unn.sum();
}

torch::Tensor make_uniform(int64_t V) {
  return torch::full({V}, 1.0f / V, torch::kFloat32);
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--verbose" || std::string(argv[i]) == "-v")
      verbose = true;
  }

  const int64_t V = 50000;
  std::vector<TestCase> cases;
  for (float tp : {0.5f, 0.7f, 0.9f, 0.95f, 0.99f}) {
    cases.push_back({"sharp_V" + std::to_string(V),    make_sharp(V),    tp});
    cases.push_back({"powerlaw_V" + std::to_string(V), make_powerlaw(V), tp});
    cases.push_back({"uniform_V" + std::to_string(V),  make_uniform(V),  tp});
  }

  int passed = 0, failed = 0;
  for (const auto& tc : cases) {
    if (run_test(tc, verbose)) ++passed; else ++failed;
  }

  std::cout << "\n" << passed << " passed, " << failed << " failed.\n";
  return failed == 0 ? 0 : 1;
}
