#pragma once

/**
 * include/olmo_cpp/generate/tree_speculative.hpp
 *
 * Tree-shaped speculative decoding (item 8.1) — Medusa / EAGLE style.
 *
 * Standard (linear-chain) speculative decoding from chat.cpp drafts k
 * tokens sequentially and verifies them in one target forward. Tree
 * speculative drafts a TREE of candidates (each MTP head can fan out
 * multiple branches), verifies the entire tree in ONE target forward
 * with a "tree attention mask" that ensures each branch only sees its
 * own ancestors, and accepts the longest matching path.
 *
 * Expected speedup over linear-chain spec: 1.4-1.8× on top of the
 * 2-2.5× linear chain already gives. Comes from a higher expected
 * acceptance-rate per token-position (you have multiple alternatives
 * at each step).
 *
 * Surface here:
 *   - DraftTree builder: stack candidate tokens + their parent indices
 *     into a flat representation suitable for one batched verify pass.
 *   - tree_attention_mask: produces the [n_nodes, n_nodes] additive
 *     mask that the target SDPA call uses.
 *   - tree_speculative_step(...): the orchestration routine. Returns
 *     the accepted path's tokens.
 */

#include "olmo_cpp/model/transformer.hpp"

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

struct DraftTreeNode {
  int32_t token;
  int32_t parent;  // index of parent in the tree node array; -1 = root
  int32_t depth;
};

struct DraftTree {
  std::vector<DraftTreeNode> nodes;
  /// Build a flat [n_nodes] input id sequence and the [n_nodes, n_nodes]
  /// causal attention mask that allows each node to attend only to its
  /// ancestors (including itself).
  std::pair<torch::Tensor, torch::Tensor> flatten(torch::Device device) const;
};

/// Tree-spec step: drafts a width-`fanout` tree of depth `max_depth`
/// using the MTP heads (top-K candidates per head), runs ONE verify
/// forward on the target via forward_tree, and accepts the longest
/// matching path from the root.
///
/// `seed_token` is the last committed token of the prefix — it serves
/// as the tree's root so the verify forward sees the same context the
/// next-token prediction is conditioned on. forward_tree is stateless;
/// the caller is responsible for re-running the accepted prefix
/// through forward_backbone (or equivalent) to update its KV cache.
///
/// Returns the path's accepted tokens (length ≥ 1 on success).
std::vector<int64_t> tree_speculative_step(
    Transformer& target_model,
    torch::Tensor seed_hidden,
    int64_t seed_token,
    int64_t fanout,
    int64_t max_depth,
    torch::Device device);

}  // namespace olmo_cpp
