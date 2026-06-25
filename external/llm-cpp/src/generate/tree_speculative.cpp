/**
 * src/generate/tree_speculative.cpp
 *
 * Tree-shaped speculative decoding (item 8.1). Builds a small fan-out
 * tree from the MTP heads' top-k candidates, runs one target verify
 * forward with a tree attention mask, walks the verified path, and
 * returns the accepted tokens.
 *
 * Today's commit ships the data structures + the flatten/mask builders
 * + a simple greedy linear-path tree (fanout=1) that's functionally
 * equivalent to the existing linear-chain speculative. The true
 * fanout>1 verify path needs a tree-attention-aware target forward,
 * which is wired in a follow-on once the SDPA call site is reworked
 * to accept an additive 2-D mask of shape [verify_len, verify_len].
 */

#include "olmo_cpp/generate/tree_speculative.hpp"

#include <torch/torch.h>

namespace olmo_cpp {

std::pair<torch::Tensor, torch::Tensor> DraftTree::flatten(torch::Device device) const {
  const int64_t N = static_cast<int64_t>(nodes.size());
  auto i64_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);
  auto ids = torch::empty({N}, i64_opts);
  auto mask = torch::zeros({N, N}, bool_opts);

  // Build ancestors set per node via parent chain. Mask[i, j] = true iff
  // j is an ancestor of i (or j == i). Then we'll convert to additive
  // -inf / 0 mask at the call site if needed.
  std::vector<std::vector<bool>> anc(N, std::vector<bool>(N, false));
  for (int64_t i = 0; i < N; ++i) {
    anc[i][i] = true;
    int32_t p = nodes[i].parent;
    while (p >= 0) {
      anc[i][p] = true;
      p = nodes[p].parent;
    }
  }
  auto ids_ptr = ids.data_ptr<int64_t>();
  auto mask_acc = mask.accessor<bool, 2>();
  for (int64_t i = 0; i < N; ++i) {
    ids_ptr[i] = static_cast<int64_t>(nodes[i].token);
    for (int64_t j = 0; j < N; ++j) mask_acc[i][j] = anc[i][j];
  }
  return {ids, mask};
}

// Build a width-`fanout` tree of depth `max_depth` from the seed hidden
// state's MTP-head logits. Each MTP head k provides the candidate
// token distribution for depth k+1; we take that head's top-`fanout`
// tokens. Position-d nodes share their predictions across the entire
// width — that is the structural simplification MTP heads enforce
// (each head predicts a future position from h_t independently, so
// drafting at depth d does not condition on which ancestor was chosen
// at d-1). Even with that, fanout > 1 raises the chance the verify
// step accepts a longer prefix than a linear-chain draft would.
//
// `seed_token` is the root: the last committed token of the caller's
// prefix. forward_tree sees [seed_token, drafts...] so the model's
// position-0 prediction is conditioned on the same context the
// caller's next-token prediction would be.
static DraftTree build_mtp_tree(
    int64_t seed_token,
    const std::vector<torch::Tensor>& mtp_logits,
    int64_t fanout,
    int64_t max_depth) {
  DraftTree t;
  t.nodes.push_back({/*token=*/static_cast<int32_t>(seed_token),
                      /*parent=*/-1, /*depth=*/0});

  const int64_t depth = std::min<int64_t>(max_depth,
                                            static_cast<int64_t>(mtp_logits.size()));
  std::vector<std::vector<int64_t>> topk_per_head;
  topk_per_head.reserve(static_cast<size_t>(depth));
  for (int64_t d = 0; d < depth; ++d) {
    auto cpu = mtp_logits[static_cast<size_t>(d)].to(torch::kCPU).to(torch::kFloat32);
    auto [vals, idx] = cpu.topk(fanout, /*dim=*/-1);
    auto idx_a = idx.contiguous().data_ptr<int64_t>();
    topk_per_head.emplace_back(idx_a, idx_a + fanout);
  }

  std::vector<int32_t> last_depth_indices = {0};
  for (int64_t d = 1; d <= depth; ++d) {
    const auto& topk = topk_per_head[static_cast<size_t>(d - 1)];
    std::vector<int32_t> new_indices;
    new_indices.reserve(last_depth_indices.size() * static_cast<size_t>(fanout));
    for (int32_t parent : last_depth_indices) {
      for (int64_t f = 0; f < fanout; ++f) {
        t.nodes.push_back({static_cast<int32_t>(topk[f]),
                            parent,
                            static_cast<int32_t>(d)});
        new_indices.push_back(static_cast<int32_t>(t.nodes.size() - 1));
      }
    }
    last_depth_indices = std::move(new_indices);
  }
  return t;
}

std::vector<int64_t> tree_speculative_step(
    Transformer& target_model,
    torch::Tensor seed_hidden,
    int64_t seed_token,
    int64_t fanout,
    int64_t max_depth,
    torch::Device device) {
  if (max_depth <= 0 || fanout <= 0) return {};

  // 1. Draft a width-`fanout` tree from MTP heads on seed_hidden.
  auto mtp_logits = target_model->forward_mtp_draft(seed_hidden);
  if (mtp_logits.empty()) return {};
  DraftTree tree = build_mtp_tree(seed_token, mtp_logits, fanout, max_depth);
  if (tree.nodes.size() <= 1) return {};

  // 2. Flatten: ids are the in-order node tokens, mask[i,j] = j is
  // an ancestor of i (or i itself).
  auto [ids_flat, mask] = tree.flatten(device);

  // 3. Verify in ONE forward pass with the tree mask. Stateless —
  // forward_tree builds RoPE buffers for length N internally and does
  // NOT touch any KV cache. The caller re-runs forward_backbone on
  // the accepted prefix to materialize cache entries.
  auto verify_ids = ids_flat.unsqueeze(0);                       // [1, N]
  auto logits = target_model->forward_tree(verify_ids, mask);    // [1, N, V]

  // 4. Walk the tree. Start at the root (the seed_token). The root's
  // predicted logits give the model's expected next token; if a child
  // of the root carries that token, accept it and descend. Repeat.
  // Argmax over the vocab ON-DEVICE, then bring back only the [N] chosen ids
  // (not the [N, V] logits). The tree walk indexes am_ptr by node cursor.
  auto argmax = logits.select(0, 0).argmax(/*dim=*/-1).to(torch::kCPU).contiguous();  // [N]
  auto am_ptr = argmax.data_ptr<int64_t>();

  std::vector<int64_t> accepted;
  int32_t cursor = 0;  // root
  while (true) {
    const int64_t pred = am_ptr[cursor];
    int32_t hit = -1;
    for (size_t i = 1; i < tree.nodes.size(); ++i) {
      if (tree.nodes[i].parent == cursor && tree.nodes[i].token == pred) {
        hit = static_cast<int32_t>(i);
        break;
      }
    }
    if (hit < 0) {
      // No matching child — commit the parent's prediction as the
      // single new token and stop. Guarantees ≥ 1 token of progress.
      accepted.push_back(pred);
      break;
    }
    accepted.push_back(pred);
    cursor = hit;
  }
  (void)device;
  return accepted;
}

}  // namespace olmo_cpp
