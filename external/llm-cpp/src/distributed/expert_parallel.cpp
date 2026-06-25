/**
 * src/distributed/expert_parallel.cpp
 *
 * ─── What "Expert Parallelism" is ──────────────────────────────────
 *
 * MoE = **M**ixture **o**f **E**xperts. Instead of every token going
 * through one big FFN, an MoE layer has N small "expert" FFNs and a
 * lightweight router. The router picks top-k experts per token; only
 * those experts compute. Result: more total parameters at the same
 * per-token compute. (See src/model/moe/* for the math.)
 *
 * With many experts and many GPUs, "Expert Parallelism" assigns each
 * GPU a disjoint subset of the experts. A token's hidden state is
 * shipped to whichever GPU owns the expert it was routed to, the
 * expert FFN runs there, and the result is shipped back.
 *
 * The collective primitive used is **all_to_all**: each rank sends a
 * shard of its tokens to every other rank and simultaneously receives
 * a shard from every other rank. Two all_to_alls per layer (dispatch
 * + combine).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/expert_parallel.hpp : context + EP utilities.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/model/moe/moe.cpp: when EP is configured, MoELayer routes
 *     tokens through ExpertParallelContext::dispatch / combine.
 *
 * --- Role in training pipeline ---
 *   Active only when use_moe=1 and an expert-parallel sub-group exists.
 *   Without Gloo (the OLMO_HAS_DDP guard) the file collapses to no-op
 *   stubs. The quickstart's single-GPU 3060 flow does NOT use this.
 */
#include "olmo_cpp/distributed/expert_parallel.hpp"
#include <stdexcept>

namespace olmo_cpp {

#ifdef OLMO_HAS_DDP

ExpertParallelContext::ExpertParallelContext(
    c10::intrusive_ptr<c10d::Backend> backend,
    int rank, int ep_size, int num_experts)
    : backend_(std::move(backend)),
      rank_(rank), ep_size_(ep_size), num_experts_(num_experts),
      experts_per_rank_(num_experts / ep_size) {
  if (num_experts % ep_size != 0) {
    throw std::runtime_error("num_experts must be divisible by ep_size");
  }
}

std::optional<ExpertParallelContext> ExpertParallelContext::create(
    c10::intrusive_ptr<c10d::Backend> backend, int num_experts, int ep_size) {
  if (!backend || ep_size < 2) return std::nullopt;
  return ExpertParallelContext(backend, backend->getRank() % ep_size, ep_size, num_experts);
}

ExpertParallelContext::DispatchResult ExpertParallelContext::dispatch(
    torch::Tensor tokens, torch::Tensor expert_ids, torch::Tensor weights) {
  // tokens: [N, D], expert_ids: [N, top_k], weights: [N, top_k]
  auto N = tokens.size(0);
  auto D = tokens.size(1);
  auto top_k = expert_ids.size(1);

  // Flatten to per-assignment: [N*top_k] expert assignments
  auto flat_ids = expert_ids.view({-1});          // [N*top_k]
  auto flat_weights = weights.view({-1});         // [N*top_k]
  // Repeat tokens for each top_k assignment
  auto flat_tokens = tokens.unsqueeze(1).expand({N, top_k, D})
                           .reshape({N * top_k, D});  // [N*top_k, D]

  // Determine which rank owns each expert
  auto target_ranks = flat_ids.div(experts_per_rank_, "trunc").to(torch::kLong);

  // Count tokens going to each rank
  auto send_counts = torch::zeros({ep_size_}, torch::kLong);
  for (int r = 0; r < ep_size_; ++r) {
    send_counts[r] = (target_ranks == r).sum();
  }

  // Sort by target rank for all-to-all
  auto sorted_indices = target_ranks.argsort();
  auto sorted_tokens = flat_tokens.index_select(0, sorted_indices);
  auto sorted_ids = flat_ids.index_select(0, sorted_indices);
  auto sorted_weights = flat_weights.index_select(0, sorted_indices);

  // Exchange counts via allgather
  std::vector<at::Tensor> all_counts(ep_size_);
  for (int i = 0; i < ep_size_; ++i) {
    all_counts[i] = torch::empty({ep_size_}, torch::kLong);
  }
  std::vector<std::vector<at::Tensor>> out_counts = {all_counts};
  std::vector<at::Tensor> in_counts = {send_counts};
  backend_->allgather(out_counts, in_counts)->wait();

  // recv_counts[r] = how many tokens rank r sends to us
  auto recv_counts = torch::zeros({ep_size_}, torch::kLong);
  for (int r = 0; r < ep_size_; ++r) {
    recv_counts[r] = all_counts[r][rank_];
  }

  // All-to-all: exchange tokens
  int64_t total_recv = recv_counts.sum().item<int64_t>();
  auto recv_tokens = torch::empty({total_recv, D}, tokens.options());
  auto recv_ids = torch::empty({total_recv}, torch::kLong);
  auto recv_weights = torch::empty({total_recv}, weights.options());

  // Use point-to-point sends/recvs grouped by rank
  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  int64_t send_offset = 0;
  int64_t recv_offset = 0;

  for (int r = 0; r < ep_size_; ++r) {
    int64_t sc = send_counts[r].item<int64_t>();
    int64_t rc = recv_counts[r].item<int64_t>();

    if (r == rank_) {
      // Local copy
      if (sc > 0 && rc > 0) {
        recv_tokens.narrow(0, recv_offset, rc).copy_(
            sorted_tokens.narrow(0, send_offset, sc));
        recv_ids.narrow(0, recv_offset, rc).copy_(
            sorted_ids.narrow(0, send_offset, sc));
        recv_weights.narrow(0, recv_offset, rc).copy_(
            sorted_weights.narrow(0, send_offset, sc));
      }
    } else {
      if (sc > 0) {
        std::vector<at::Tensor> st = {sorted_tokens.narrow(0, send_offset, sc).contiguous()};
        works.push_back(backend_->send(st, r, /*tag=*/0));
        std::vector<at::Tensor> si = {sorted_ids.narrow(0, send_offset, sc).contiguous()};
        works.push_back(backend_->send(si, r, /*tag=*/1));
        std::vector<at::Tensor> sw = {sorted_weights.narrow(0, send_offset, sc).contiguous()};
        works.push_back(backend_->send(sw, r, /*tag=*/2));
      }
      if (rc > 0) {
        std::vector<at::Tensor> rt = {recv_tokens.narrow(0, recv_offset, rc)};
        works.push_back(backend_->recv(rt, r, /*tag=*/0));
        std::vector<at::Tensor> ri = {recv_ids.narrow(0, recv_offset, rc)};
        works.push_back(backend_->recv(ri, r, /*tag=*/1));
        std::vector<at::Tensor> rw = {recv_weights.narrow(0, recv_offset, rc)};
        works.push_back(backend_->recv(rw, r, /*tag=*/2));
      }
    }
    send_offset += sc;
    recv_offset += rc;
  }
  for (auto& w : works) w->wait();

  // Convert global expert ids to local (0..experts_per_rank-1)
  auto local_ids = recv_ids - rank_ * experts_per_rank_;

  return {recv_tokens, local_ids, recv_weights, send_counts, recv_counts};
}

torch::Tensor ExpertParallelContext::combine(
    torch::Tensor expert_outputs,
    const DispatchResult& dispatch_info) {
  // Reverse all-to-all: send results back to originating ranks
  auto D = expert_outputs.size(1);

  // Swap send/recv counts (reverse direction)
  auto send_counts = dispatch_info.recv_counts;  // we now send what we received
  auto recv_counts = dispatch_info.send_counts;  // we receive what we sent

  int64_t total_recv = recv_counts.sum().item<int64_t>();
  auto result = torch::empty({total_recv, D}, expert_outputs.options());

  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  int64_t send_offset = 0;
  int64_t recv_offset = 0;

  for (int r = 0; r < ep_size_; ++r) {
    int64_t sc = send_counts[r].item<int64_t>();
    int64_t rc = recv_counts[r].item<int64_t>();

    if (r == rank_) {
      if (sc > 0) {
        result.narrow(0, recv_offset, rc).copy_(
            expert_outputs.narrow(0, send_offset, sc));
      }
    } else {
      if (sc > 0) {
        std::vector<at::Tensor> st = {expert_outputs.narrow(0, send_offset, sc).contiguous()};
        works.push_back(backend_->send(st, r, /*tag=*/3));
      }
      if (rc > 0) {
        std::vector<at::Tensor> rt = {result.narrow(0, recv_offset, rc)};
        works.push_back(backend_->recv(rt, r, /*tag=*/3));
      }
    }
    send_offset += sc;
    recv_offset += rc;
  }
  for (auto& w : works) w->wait();

  return result;
}

#else  // !OLMO_HAS_DDP

ExpertParallelContext::ExpertParallelContext(int rank, int ep_size, int num_experts)
    : rank_(rank), ep_size_(ep_size), num_experts_(num_experts),
      experts_per_rank_(num_experts / ep_size) {}

ExpertParallelContext::DispatchResult ExpertParallelContext::dispatch(
    torch::Tensor tokens, torch::Tensor expert_ids, torch::Tensor weights) {
  return {tokens, expert_ids.view({-1}), weights.view({-1}),
          torch::ones({1}, torch::kLong) * tokens.size(0),
          torch::ones({1}, torch::kLong) * tokens.size(0)};
}

torch::Tensor ExpertParallelContext::combine(
    torch::Tensor expert_outputs, const DispatchResult&) {
  return expert_outputs;
}

#endif

}  // namespace olmo_cpp
