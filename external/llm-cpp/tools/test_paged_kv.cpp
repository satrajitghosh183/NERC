/**
 * tools/test_paged_kv.cpp
 *
 * Smoke test for PagedKVCache. Compares it head-to-head against the
 * concat-based LayerKVCache used today by the model's attention forward.
 *
 * For each step (mix of prefill-shape and decode-shape appends across
 * multiple layers), we:
 *   1. Append the same K/V into both caches.
 *   2. Materialize K/V from the paged cache; read the same view from the
 *      concat cache.
 *   3. Assert the two are bit-for-bit equal (FP32 path, no dtype cast).
 *
 * Also exercises snapshot + rollback (speculative-decode rewind path).
 *
 * Build: this file is compiled as the `test_paged_kv` executable; see
 * CMakeLists. Run: `./build/test_paged_kv`. Exit 0 on success, 1 on
 * any mismatch.
 */

#include "olmo_cpp/model/paged_kv_cache.hpp"
#include "olmo_cpp/model/kv_cache.hpp"

#include <torch/torch.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using olmo_cpp::IPagedKVCache;
using olmo_cpp::KVCache;
using olmo_cpp::make_paged_kv_cache;
using olmo_cpp::make_paged_kv_cache_graph_safe;

namespace {

struct Failure {
  std::string what;
};

void require_close(torch::Tensor a, torch::Tensor b, const std::string& label) {
  if (!a.defined() || !b.defined()) {
    throw Failure{label + ": one tensor is undefined"};
  }
  if (a.sizes() != b.sizes()) {
    std::ostringstream os;
    os << label << ": shape mismatch " << a.sizes() << " vs " << b.sizes();
    throw Failure{os.str()};
  }
  // Allow tiny float jitter from index_put_ on some backends; bitwise on CPU.
  if (!torch::allclose(a, b, /*rtol=*/0.0, /*atol=*/0.0)) {
    auto diff = (a - b).abs().max().item<float>();
    std::ostringstream os;
    os << label << ": tensors differ; max abs diff = " << diff;
    throw Failure{os.str()};
  }
}

torch::Tensor rand_kv(int64_t n_kv_heads, int64_t S, int64_t head_dim,
                      torch::Device device) {
  // [B=1, n_kv_heads, S, head_dim]
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  return torch::randn({1, n_kv_heads, S, head_dim}, opts);
}

}  // namespace

// Run the test body against an arbitrary IPagedKVCache. The reference path
// uses olmo_cpp::KVCache directly. Returns 0 on success, throws Failure on
// mismatch.
int run_equivalence_test(std::unique_ptr<IPagedKVCache> paged,
                         int64_t n_layers,
                         int64_t n_kv_heads,
                         int64_t head_dim,
                         torch::Device device,
                         const std::string& label) {
  KVCache reference(n_layers, device);

  // Step sequence: one prefill of 9 tokens (spans 3 pages with leftover),
  // then 12 decode steps of 1 token, then a snapshot+rollback test.
  std::vector<int64_t> step_sizes = {9};
  for (int i = 0; i < 12; ++i) step_sizes.push_back(1);

  {
    int64_t total = 0;
    for (size_t step = 0; step < step_sizes.size(); ++step) {
      const int64_t S = step_sizes[step];
      for (int64_t layer = 0; layer < n_layers; ++layer) {
        auto k = rand_kv(n_kv_heads, S, head_dim, device);
        auto v = rand_kv(n_kv_heads, S, head_dim, device);
        const int64_t new_len_paged = paged->append(layer, k, v);
        auto [_, __] = reference.layers[static_cast<size_t>(layer)].update(k, v);
        (void)_; (void)__;
        if (new_len_paged != reference.layers[static_cast<size_t>(layer)].seq_len()) {
          throw Failure{"seq_len mismatch after append"};
        }
      }
      total += S;

      // Materialize layer 0 and compare against reference's [0..total) view.
      auto [pk, pv] = paged->materialize(0);
      auto rk = reference.layers[0].k.narrow(2, 0, total);
      auto rv = reference.layers[0].v.narrow(2, 0, total);
      require_close(pk, rk, "step " + std::to_string(step) + " layer 0 K");
      require_close(pv, rv, "step " + std::to_string(step) + " layer 0 V");

      // Spot-check a middle layer too.
      auto [pk2, pv2] = paged->materialize(2);
      require_close(pk2, reference.layers[2].k.narrow(2, 0, total),
                    "step " + std::to_string(step) + " layer 2 K");
      require_close(pv2, reference.layers[2].v.narrow(2, 0, total),
                    "step " + std::to_string(step) + " layer 2 V");
    }

    // Snapshot / rollback. Rewind to 12 tokens, append 3 more for layer 0..n-1,
    // verify materialize matches a fresh concat reference replayed to the same
    // state.
    const int64_t rewind_to = 12;
    paged->rollback(rewind_to);
    for (auto& l : reference.layers) l.truncate(rewind_to);
    if (paged->seq_len() != rewind_to) throw Failure{"rollback did not move cursor"};

    for (int64_t layer = 0; layer < n_layers; ++layer) {
      auto k = rand_kv(n_kv_heads, 3, head_dim, device);
      auto v = rand_kv(n_kv_heads, 3, head_dim, device);
      paged->append(layer, k, v);
      reference.layers[static_cast<size_t>(layer)].update(k, v);
    }
    auto [pk_final, pv_final] = paged->materialize(0);
    require_close(pk_final, reference.layers[0].k.narrow(2, 0, rewind_to + 3),
                  "post-rollback layer 0 K");
    require_close(pv_final, reference.layers[0].v.narrow(2, 0, rewind_to + 3),
                  "post-rollback layer 0 V");

    paged->clear();
    if (paged->seq_len() != 0) throw Failure{"clear did not reset cursor"};

    std::cout << label << " OK: " << step_sizes.size() << " steps, "
              << "final seq_len after re-clear = " << paged->seq_len() << "\n";
  }
  return 0;
}

int main() {
  torch::manual_seed(0);

  const int64_t n_layers   = 4;
  const int64_t n_kv_heads = 2;
  const int64_t head_dim   = 8;
  const int64_t page_size  = 4;
  const int64_t max_pages  = 64;
  auto device = torch::kCPU;
  auto dtype  = torch::kFloat32;

  try {
    // Legacy index_put_ write path.
    auto paged_legacy = make_paged_kv_cache(
        n_layers, n_kv_heads, head_dim, page_size, max_pages, device, dtype);
    run_equivalence_test(std::move(paged_legacy), n_layers, n_kv_heads, head_dim,
                         device, "PagedKVCache[index_put_]");

    // Graph-safe paged_kv_write_dyn path. Same seed, same K/V; should
    // produce bitwise-identical materialize() output to the legacy path.
    torch::manual_seed(0);  // reset so rand_kv produces same sequence
    auto paged_dyn = make_paged_kv_cache_graph_safe(
        n_layers, n_kv_heads, head_dim, page_size, max_pages, device, dtype);
    run_equivalence_test(std::move(paged_dyn), n_layers, n_kv_heads, head_dim,
                         device, "PagedKVCache[paged_kv_write_dyn]");

    std::cout << "all paths OK\n";
    return 0;
  } catch (const Failure& f) {
    std::cerr << "FAIL: " << f.what << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << "\n";
    return 1;
  }
}
