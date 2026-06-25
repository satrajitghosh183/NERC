/**
 * src/distributed/zero1.cpp
 *
 * Implementation of OptimizerStateSharder (fast-inference [T-4]).
 *
 * Partition is index-mod-world_size. allgather_params is a per-parameter
 * broadcast from the owner: we loop, picking root = owner_of(i), and
 * issue a broadcast. To pipeline across parameters, we batch broadcasts
 * into groups by owner rank and issue them concurrently — c10d::Work
 * objects from different ranks don't depend on each other, so the
 * dispatcher can interleave on the wire.
 */

#include "olmo_cpp/distributed/zero1.hpp"
#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace olmo_cpp {

std::vector<torch::Tensor> OptimizerStateSharder::partition(
    const std::vector<torch::Tensor>& all_params) const {
  if (!is_active()) {
    // Single-rank build / no backend: caller's "subset" is the full set.
    return all_params;
  }
  std::vector<torch::Tensor> mine;
  mine.reserve(all_params.size() / static_cast<size_t>(world_size_) + 1);
  for (size_t i = 0; i < all_params.size(); ++i) {
    if (owner_of(i) == rank_) mine.push_back(all_params[i]);
  }
  return mine;
}

void OptimizerStateSharder::allgather_params(
    std::vector<torch::Tensor>& all_params) {
  if (!is_active()) return;

  // One broadcast per parameter, grouped concurrent. Order within a rank
  // doesn't matter for correctness; we issue all then wait on all.
  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  works.reserve(all_params.size());
  for (size_t i = 0; i < all_params.size(); ++i) {
    auto& p = all_params[i];
    if (!p.defined()) continue;
    std::vector<at::Tensor> tensors = {p};
    c10d::BroadcastOptions opts;
    opts.rootRank = owner_of(i);
    works.push_back(backend_->broadcast(tensors, opts));
  }
  for (auto& w : works) w->wait();
}

}  // namespace olmo_cpp
