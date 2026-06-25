/**
 * src/distributed/context_parallel.cpp
 *
 * Implementation of Context Parallelism. The sequence axis S is sharded
 * across cp_size ranks (each rank owns S/cp_size contiguous tokens).
 * Self-attention is computed via *ring attention*: K and V tiles rotate
 * around a logical ring of cp_size ranks (point-to-point send/recv) while
 * each rank accumulates its piece of the output using an online
 * (log-sum-exp) softmax. Sequence reassembly (when needed) uses an
 * allgather along dim=1.
 *
 * Collective primitives: p2p send/recv (ring KV exchange), allgather
 * (gather_sequence). Causal masking uses absolute query/key positions
 * derived from `rank_` and `kv_origin` so the wrap-around steps that
 * would otherwise expose future tokens are masked out.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/distributed/context_parallel.hpp: declares the class.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep (CP is wired into the
 *   attention layer at trainer setup time).
 *
 * --- Role in training pipeline ---
 *   Lets us train with logical sequence length S while each device only
 *   ever holds S/cp_size of activations. Combined with FSDP/TP this gives
 *   long-context training without OOM.
 */

#include "olmo_cpp/distributed/context_parallel.hpp"
#include <cmath>
#include <limits>

namespace olmo_cpp {

#ifdef OLMO_HAS_DDP

/// Real ctor. `rank` is the position within the CP sub-group (0..cp_size-1).
ContextParallelContext::ContextParallelContext(
    c10::intrusive_ptr<c10d::Backend> backend, int rank, int cp_size)
    : backend_(std::move(backend)), rank_(rank), cp_size_(cp_size) {}

/// Factory: refuses cp_size < 2 (no parallelism) or null backend.
/// `backend->getRank() % cp_size` derives the local CP rank from the global
/// rank -- this assumes ranks are arranged so that each consecutive group
/// of cp_size ranks forms a CP sub-group.
std::optional<ContextParallelContext> ContextParallelContext::create(
    c10::intrusive_ptr<c10d::Backend> backend, int cp_size) {
  if (!backend || cp_size < 2) return std::nullopt;
  return ContextParallelContext(backend, backend->getRank() % cp_size, cp_size);
}

/// Local-only scatter: each rank takes its slice of x along the sequence
/// dimension. No collective is needed because the input is assumed to be
/// already replicated on every rank (or produced identically from an
/// embedding lookup that all ranks ran on the same token ids).
torch::Tensor ContextParallelContext::scatter_sequence(torch::Tensor x) {
  // x: [B, S, D] -> [B, S/cp_size, D]
  auto B = x.size(0);
  auto S = x.size(1);
  auto D = x.size(2);
  int64_t chunk_size = S / cp_size_;
  // narrow(dim, start, length): zero-copy view; .contiguous() materialises.
  return x.narrow(1, rank_ * chunk_size, chunk_size).contiguous();
}

/// Inverse of scatter_sequence: allgather all per-rank S/cp_size chunks
/// and concatenate them along dim=1 to recover the full sequence.
torch::Tensor ContextParallelContext::gather_sequence(torch::Tensor x) {
  // x: [B, S/cp, D] -> allgather -> [B, S, D]
  int64_t chunk_len = x.size(1);
  // Pre-allocate one output buffer per peer; allgather will fill them in
  // rank order so gathered[r] receives rank r's contribution.
  std::vector<at::Tensor> gathered(cp_size_);
  for (int i = 0; i < cp_size_; ++i) {
    gathered[i] = torch::empty_like(x);
  }
  // c10d allgather expects nested lists (vector<vector<>>) for the multi-
  // input case; here each rank contributes one tensor so output has size 1.
  std::vector<std::vector<at::Tensor>> output = {gathered};
  std::vector<at::Tensor> input = {x.contiguous()};
  // ALLGATHER collective: every rank ends up with all peers' chunks.
  backend_->allgather(output, input)->wait();

  // Concatenate along the sequence axis to reconstruct [B, S, D].
  return torch::cat(gathered, /*dim=*/1);
}

/// Ring attention: distributed self-attention where Q stays local but K/V
/// rotate around the ring. After cp_size steps, every rank has effectively
/// attended to every key/value tile in the global sequence.
///
/// Math sketch:
///   For each step s, rank r holds K,V from rank (r - s) mod cp_size.
///   We compute scores = Q_local @ K_step^T / sqrt(D), then update an
///   online softmax accumulator (output, lse) using log-sum-exp combine:
///     m'   = max(lse, block_lse)
///     out  = out * exp(lse - m') + block_out * exp(block_lse - m')
///     lse  = m'
///   This is the same recurrence used by FlashAttention.
torch::Tensor ContextParallelContext::ring_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  // q,k,v: [B, H, S_local, D] where S_local = S / cp_size.
  auto B = q.size(0);
  auto H = q.size(1);
  auto S_local = q.size(2);
  auto D = q.size(3);

  // Online softmax accumulators. Output starts at 0 and lse at -inf so the
  // first block's contribution dominates (exp(-inf) == 0).
  auto output = torch::zeros_like(q);                    // [B,H,S_local,D]
  auto lse = torch::full({B, H, S_local, 1},            // log-sum-exp tracker
                          -std::numeric_limits<float>::infinity(),
                          q.options());

  // Current KV being processed (starts as local). We rotate these around
  // the ring at the end of each step.
  auto cur_k = k.contiguous();
  auto cur_v = v.contiguous();

  // Ring topology: next_rank receives from us, prev_rank sends to us.
  int next_rank = (rank_ + 1) % cp_size_;
  int prev_rank = (rank_ - 1 + cp_size_) % cp_size_;

  for (int step = 0; step < cp_size_; ++step) {
    // After `step` rotations, the KV tile we currently hold originated on
    // `(rank_ - step) mod cp_size`. We need this to compute correct
    // absolute key positions for causal masking.
    int kv_origin = (rank_ - step + cp_size_) % cp_size_;

    // Compute local attention scores: [B, H, S_local, S_local].
    // Standard scaled dot-product: Q @ K^T / sqrt(D).
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    auto scores = torch::matmul(q, cur_k.transpose(-2, -1)) * scale;

    // Apply causal mask if needed. Crucially this must use ABSOLUTE
    // positions: a query at global pos i may only attend to keys at
    // global pos <= i. Without this, the wrap-around steps would leak
    // future tokens into the attention.
    if (causal) {
      // Query positions: [rank_ * S_local, (rank_+1) * S_local)
      // Key positions:   [kv_origin * S_local, (kv_origin+1) * S_local)
      auto q_pos = torch::arange(rank_ * S_local, (rank_ + 1) * S_local, q.device());
      auto k_pos = torch::arange(kv_origin * S_local, (kv_origin + 1) * S_local, q.device());
      // mask[i,j] == true iff query i can see key j.
      auto mask = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0);  // [S_local, S_local]
      // Set masked-out scores to -inf so softmax sends them to 0.
      scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0),
                                   -std::numeric_limits<float>::infinity());
    }

    // Online softmax update (numerically stable incremental attention).
    // First compute the per-row max for this block (subtracted before exp
    // to avoid overflow), then exp, sum, log to get the block's lse.
    auto block_max = std::get<0>(scores.max(-1, /*keepdim=*/true));  // [B,H,S_local,1]
    auto block_exp = (scores - block_max).exp();                      // [B,H,S_local,S_local]
    auto block_sumexp = block_exp.sum(-1, /*keepdim=*/true);          // [B,H,S_local,1]
    auto block_lse = block_max + block_sumexp.log();                  // [B,H,S_local,1]
    // Unnormalised weighted-V for this block.
    auto block_out = torch::matmul(block_exp, cur_v);                 // [B,H,S_local,D]

    // Combine with running accumulator using log-sum-exp:
    //   new_lse = log(exp(old_lse) + exp(block_lse))
    //   weights ensure both terms are scaled by exp(-new_lse) implicitly.
    auto new_lse = torch::logaddexp(lse, block_lse);
    auto old_weight = (lse - new_lse).exp();
    auto new_weight = (block_lse - new_lse).exp();

    output = output * old_weight + block_out * new_weight;
    lse = new_lse;

    // Ring exchange: send our current K,V to rank+1 and receive new K,V
    // from rank-1. Skipped on the last step because no further block
    // remains to be processed.
    if (step < cp_size_ - 1) {
      auto recv_k = torch::empty_like(cur_k);
      auto recv_v = torch::empty_like(cur_v);

      // Wrap each tensor in the single-element vector that c10d expects.
      std::vector<at::Tensor> send_k_vec = {cur_k};
      std::vector<at::Tensor> recv_k_vec = {recv_k};
      std::vector<at::Tensor> send_v_vec = {cur_v};
      std::vector<at::Tensor> recv_v_vec = {recv_v};

      // Issue all four p2p ops (async). Tags differ per (step, k-vs-v) so
      // the receiver can match correctly even if multiple steps overlap.
      auto send_k_work = backend_->send(send_k_vec, next_rank, /*tag=*/step * 4);
      auto recv_k_work = backend_->recv(recv_k_vec, prev_rank, /*tag=*/step * 4);
      auto send_v_work = backend_->send(send_v_vec, next_rank, /*tag=*/step * 4 + 1);
      auto recv_v_work = backend_->recv(recv_v_vec, prev_rank, /*tag=*/step * 4 + 1);

      // Block until exchange completes before next iteration uses the data.
      // (A more advanced impl would overlap this with compute on cur_k/cur_v.)
      send_k_work->wait();
      recv_k_work->wait();
      send_v_work->wait();
      recv_v_work->wait();

      // Adopt the freshly-received KV tile as the working tile.
      cur_k = recv_k;
      cur_v = recv_v;
    }
  }

  // The online softmax already produced the normalised attention output
  // (the divide by sum-exp is folded into old_weight/new_weight), so we
  // don't need a separate normalisation step here.
  return output;
}

#else  // !OLMO_HAS_DDP

ContextParallelContext::ContextParallelContext(int rank, int cp_size)
    : rank_(rank), cp_size_(cp_size) {}

torch::Tensor ContextParallelContext::scatter_sequence(torch::Tensor x) { return x; }
torch::Tensor ContextParallelContext::gather_sequence(torch::Tensor x) { return x; }
torch::Tensor ContextParallelContext::ring_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool /*causal*/) {
  float scale = 1.0f / std::sqrt(static_cast<float>(q.size(-1)));
  auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;
  return torch::matmul(torch::softmax(scores, -1), v);
}

#endif

}  // namespace olmo_cpp
