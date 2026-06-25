/**
 * src/backend/sparse_attn.cpp
 *
 * NSA-style content-selected sparse attention. See
 * include/olmo_cpp/backend/sparse_attn.hpp for the algorithm.
 *
 * Device-agnostic: every op below is pure ATen, so this same code runs
 * on CPU (reference) and CUDA (real). The H100 perf comes from ATen's
 * cuBLAS / cuDNN dispatch on matmul/topk/gather/softmax.
 *
 * Future optimization: a fused CUDA kernel that does
 *   block-mean -> topk -> gather -> SDPA
 * in a single launch, avoiding the 4 HBM round-trips this version pays.
 * That kernel goes in kernels/sparse_attn_decode.cu and will be selected
 * automatically by the dispatch in sparse_attn_decode() once it lands;
 * this file remains as the gold reference for kernel validation.
 */

#include "olmo_cpp/backend/sparse_attn.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <vector>

namespace olmo_cpp {

// Forward declaration of the ATen reference (defined further down in the
// same anonymous namespace block).
namespace { torch::Tensor sparse_attn_decode_aten(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    float sm_scale, int64_t block_size, int64_t top_k); }

torch::Tensor sparse_attn_decode(const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& v,
                                 float sm_scale,
                                 int64_t block_size,
                                 int64_t top_k) {
#ifdef OLMO_HAS_CUDA_KERNELS
  // Fused CUDA kernel — single launch, replaces the 4-kernel ATen path.
  // Requires T % block_size == 0; we pad here so the kernel doesn't have
  // to handle the ragged tail.
  if (q.is_cuda()) {
    const int64_t T   = k.size(2);
    const int64_t pad = (block_size - T % block_size) % block_size;
    if (pad == 0) {
      return sparse_attn_decode_cuda(q, k, v, sm_scale, block_size, top_k);
    }
    const int64_t B   = k.size(0);
    const int64_t Hkv = k.size(1);
    const int64_t D   = k.size(3);
    auto pad_shape = std::vector<int64_t>{B, Hkv, pad, D};
    auto k_pad = torch::cat({k, torch::zeros(pad_shape, k.options())}, /*dim=*/2);
    auto v_pad = torch::cat({v, torch::zeros(pad_shape, v.options())}, /*dim=*/2);
    return sparse_attn_decode_cuda(q, k_pad, v_pad, sm_scale, block_size, top_k);
  }
#endif
  return sparse_attn_decode_aten(q, k, v, sm_scale, block_size, top_k);
}

namespace {

torch::Tensor sparse_attn_decode_aten(const torch::Tensor& q,
                                      const torch::Tensor& k,
                                      const torch::Tensor& v,
                                      float sm_scale,
                                      int64_t block_size,
                                      int64_t top_k) {
  TORCH_CHECK(q.dim() == 4, "q must be [B,H,1,D]");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k,v must be [B,Hkv,T,D]");

  const int64_t B    = q.size(0);
  const int64_t H    = q.size(1);
  const int64_t Sq   = q.size(2);
  const int64_t D    = q.size(3);
  const int64_t Hkv  = k.size(1);
  const int64_t T    = k.size(2);

  TORCH_CHECK(Sq == 1, "decode path expects q with seq-len 1, got ", Sq);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D, "head_dim mismatch");
  TORCH_CHECK(v.size(2) == T && v.size(1) == Hkv, "k/v shape mismatch");
  TORCH_CHECK(H % Hkv == 0, "n_heads (", H, ") must be divisible by n_kv_heads (", Hkv, ")");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(top_k > 0, "top_k must be positive");

  // Pad T up to a multiple of block_size with zeros so the block-mean
  // computation is uniform. The pad rows score very low (mean of zeros
  // dotted with a non-tiny query) and almost never get selected — but if
  // they do, attending over zeros is harmless. This avoids an awkward
  // ragged-block special case in the reference path.
  const int64_t pad     = (block_size - T % block_size) % block_size;
  const int64_t T_pad   = T + pad;
  const int64_t n_blocks = T_pad / block_size;
  const int64_t k_top   = std::min(top_k, n_blocks);

  auto k_full = k;
  auto v_full = v;
  if (pad > 0) {
    auto pad_shape_k = std::vector<int64_t>{B, Hkv, pad, D};
    k_full = torch::cat({k, torch::zeros(pad_shape_k, k.options())}, /*dim=*/2);
    v_full = torch::cat({v, torch::zeros(pad_shape_k, v.options())}, /*dim=*/2);
  }

  // Compute in fp32 for numerically stable scoring + softmax. Cast back at
  // the end.
  auto q32 = q.to(torch::kFloat32);
  auto k32 = k_full.to(torch::kFloat32);
  auto v32 = v_full.to(torch::kFloat32);

  // 1) Block-mean K: [B, Hkv, n_blocks, D].
  auto k_blocks = k32
      .reshape({B, Hkv, n_blocks, block_size, D})
      .mean(/*dim=*/3);

  // 2) Score: dot(q_grouped_avg, k_block_mean). To make selection shared
  //    across the GQA group, average q over its group of H/Hkv query heads
  //    before scoring. This collapses the per-head selection to per-kv-head.
  const int64_t group = H / Hkv;
  // q [B, H, 1, D] -> [B, Hkv, group, D] -> mean over group -> [B, Hkv, D]
  auto q_g = q32
      .reshape({B, Hkv, group, D})
      .mean(/*dim=*/2);

  // scores [B, Hkv, n_blocks] = (q_g [B,Hkv,1,D]) @ (k_blocks [B,Hkv,D,n_blocks])
  auto scores = torch::matmul(
      q_g.unsqueeze(/*dim=*/2),                 // [B, Hkv, 1, D]
      k_blocks.transpose(/*d0=*/-1, /*d1=*/-2)  // [B, Hkv, D, n_blocks]
  ).squeeze(/*dim=*/2);                          // [B, Hkv, n_blocks]

  // 3) Top-k blocks per (B, Hkv).
  auto topk = scores.topk(k_top, /*dim=*/-1, /*largest=*/true, /*sorted=*/false);
  auto top_idx = std::get<1>(topk);  // [B, Hkv, k_top]

  // 4) Gather selected KV positions. Build per-(B,Hkv) absolute index list:
  //    for each kept block index b, expand to b*block_size + [0..block_size-1].
  // Result indices: [B, Hkv, k_top * block_size].
  auto block_offsets = torch::arange(block_size, top_idx.options());  // [block_size]
  auto kept = top_idx.unsqueeze(-1) * block_size + block_offsets;     // [B, Hkv, k_top, block_size]
  auto kept_flat = kept.reshape({B, Hkv, k_top * block_size});        // [B, Hkv, K]

  const int64_t K_sel = k_top * block_size;

  // Gather along the T dimension. torch::gather wants index shape that
  // matches the source except along the gathered dim.
  // k_full: [B, Hkv, T_pad, D]; index expanded: [B, Hkv, K_sel, D].
  auto kept_idx_kv = kept_flat.unsqueeze(-1).expand({B, Hkv, K_sel, D});
  auto k_sel = torch::gather(k32, /*dim=*/2, kept_idx_kv);  // [B, Hkv, K_sel, D]
  auto v_sel = torch::gather(v32, /*dim=*/2, kept_idx_kv);  // [B, Hkv, K_sel, D]

  // 5) Expand kv heads to query heads (GQA). Same selection used by every
  //    head in the group — that's the whole point.
  auto k_sel_exp = k_sel.repeat_interleave(group, /*dim=*/1);  // [B, H, K_sel, D]
  auto v_sel_exp = v_sel.repeat_interleave(group, /*dim=*/1);

  // 6) Exact dense attention over the gathered positions. No mask needed —
  //    every selected position is valid and we want all of them in the
  //    softmax (the un-selected positions are implicitly zero-weight).
  // q32 [B, H, 1, D]; k_sel_exp.transpose(-1,-2) [B, H, D, K_sel]
  auto attn_scores = torch::matmul(q32, k_sel_exp.transpose(-1, -2)) * sm_scale;
  auto attn_probs  = torch::softmax(attn_scores, /*dim=*/-1);
  auto out32       = torch::matmul(attn_probs, v_sel_exp);  // [B, H, 1, D]

  return out32.to(q.dtype());
}

}  // namespace (anonymous)

// ---------------------------------------------------------------------------
// sparse_attn_prefill — full-sequence (training) sparse self-attention.
//
// Per query block, select top-k blocks from the causally-visible past KV
// (block-level causal mask), gather, then run dense SDPA with an intra-block
// causal mask. Loop over query blocks is in C++ (n_q_blocks is ~64 at
// S=4096 / block_size=64), each iteration is a few ATen ops.
//
// Differentiable via straight-through: torch::topk's indices are not
// differentiable, but gather/matmul/softmax all are, so gradients flow:
//   grad_out → grad_q (via SDPA), grad_K_selected, grad_V_selected
//             → grad_K, grad_V (via gather)
// The score-to-selection edge has zero gradient (straight-through is the
// standard NSA approach; soft-selection variants live behind a flag we
// don't need yet).
// ---------------------------------------------------------------------------
torch::Tensor sparse_attn_prefill(const torch::Tensor& q,
                                  const torch::Tensor& k,
                                  const torch::Tensor& v,
                                  float sm_scale,
                                  int64_t block_size,
                                  int64_t top_k) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "expect q/k/v as [B,H,S,D] / [B,Hkv,S,D]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
              "batch size mismatch");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2),
              "self-attn requires q/k/v share seq-len");
  const int64_t B    = q.size(0);
  const int64_t H    = q.size(1);
  const int64_t S    = q.size(2);
  const int64_t D    = q.size(3);
  const int64_t Hkv  = k.size(1);
  TORCH_CHECK(H % Hkv == 0, "n_heads must be divisible by n_kv_heads");
  TORCH_CHECK(block_size > 0 && top_k > 0, "block_size and top_k must be positive");
  const int64_t group = H / Hkv;

  const int64_t pad        = (block_size - S % block_size) % block_size;
  const int64_t S_pad      = S + pad;
  const int64_t n_q_blocks = S_pad / block_size;
  // For self-attention K and Q have the same length, so n_kv_blocks == n_q_blocks.
  const int64_t n_kv_blocks = n_q_blocks;

  auto q_full = q;
  auto k_full = k;
  auto v_full = v;
  if (pad > 0) {
    auto pad_q = std::vector<int64_t>{B, H,   pad, D};
    auto pad_k = std::vector<int64_t>{B, Hkv, pad, D};
    q_full = torch::cat({q, torch::zeros(pad_q, q.options())}, /*dim=*/2);
    k_full = torch::cat({k, torch::zeros(pad_k, k.options())}, /*dim=*/2);
    v_full = torch::cat({v, torch::zeros(pad_k, v.options())}, /*dim=*/2);
  }

  auto opts_idx  = torch::TensorOptions().dtype(torch::kLong).device(q.device());

  // Block-major reshapes: [B, H/Hkv, n_blocks, block_size, D].
  auto q_b = q_full.reshape({B, H,   n_q_blocks,  block_size, D});
  auto k_b = k_full.reshape({B, Hkv, n_kv_blocks, block_size, D});
  auto v_b = v_full.reshape({B, Hkv, n_kv_blocks, block_size, D});

  // Block-mean K: [B, Hkv, n_kv_blocks, D]. Used for selection scoring.
  auto k_block_means = k_b.mean(/*dim=*/3);

  std::vector<torch::Tensor> out_blocks;
  out_blocks.reserve(static_cast<size_t>(n_q_blocks));

  for (int64_t qb = 0; qb < n_q_blocks; ++qb) {
    // Queries in this q_block: [B, H, block_size, D].
    auto q_this = q_b.select(/*dim=*/2, qb);

    // Group-averaged q for this block, used for selection scoring.
    // [B, H, block_size, D] -> [B, Hkv, group, block_size, D] -> mean over (group, block_size).
    auto q_for_score = q_this
        .reshape({B, Hkv, group, block_size, D})
        .mean({/*group=*/2, /*block_size=*/3});  // [B, Hkv, D]

    // Visible kv blocks (causal at the block level): indices 0..qb (inclusive).
    const int64_t k_visible = qb + 1;
    auto k_visible_means = k_block_means.slice(/*dim=*/2, /*start=*/0, /*end=*/k_visible);

    // Scores: [B, Hkv, k_visible].
    auto scores = torch::matmul(
        q_for_score.unsqueeze(/*dim=*/2),               // [B, Hkv, 1, D]
        k_visible_means.transpose(-1, -2)               // [B, Hkv, D, k_visible]
    ).squeeze(/*dim=*/2);                                // [B, Hkv, k_visible]

    // Top-k: never exceed k_visible (early q-blocks see fewer kv blocks).
    const int64_t k_top = std::min<int64_t>(top_k, k_visible);
    auto topk = scores.topk(k_top, /*dim=*/-1, /*largest=*/true, /*sorted=*/false);
    auto top_idx = std::get<1>(topk);                    // [B, Hkv, k_top]

    // Gather selected K/V blocks from k_b/v_b.
    // index shape [B, Hkv, k_top, block_size, D].
    auto idx_gather = top_idx
        .unsqueeze(-1).unsqueeze(-1)
        .expand({B, Hkv, k_top, block_size, D});
    auto k_sel = torch::gather(k_b, /*dim=*/2, idx_gather);  // [B, Hkv, k_top, block_size, D]
    auto v_sel = torch::gather(v_b, /*dim=*/2, idx_gather);

    // Flatten (k_top, block_size) → K positions.
    const int64_t K_sel = k_top * block_size;
    auto k_flat = k_sel.reshape({B, Hkv, K_sel, D});
    auto v_flat = v_sel.reshape({B, Hkv, K_sel, D});

    // Expand kv heads to query heads (selection shared across the GQA group).
    auto k_exp = k_flat.repeat_interleave(group, /*dim=*/1);  // [B, H, K_sel, D]
    auto v_exp = v_flat.repeat_interleave(group, /*dim=*/1);

    // Attention scores: [B, H, block_size, K_sel].
    auto attn_scores = torch::matmul(q_this, k_exp.transpose(-1, -2)) * sm_scale;

    // Intra-block causal mask: a query at absolute position (qb*BS + p) may
    // attend to k positions whose absolute index is <= (qb*BS + p). The
    // absolute index of a selected k-slot s is top_idx[s/BS] * BS + (s%BS).
    auto k_slot = torch::arange(K_sel, opts_idx);                     // Long [K_sel]
    // Integer ops: torch::div with "floor" stays Long; "/" would promote to Float.
    auto k_slot_div = torch::div(k_slot, block_size, /*rounding_mode=*/"floor");
    auto k_slot_mod = torch::remainder(k_slot, block_size);
    auto src_blk_in_topk = k_slot_div.unsqueeze(0).unsqueeze(0)
        .expand({B, Hkv, K_sel});                                       // [B, Hkv, K_sel]
    auto src_blk_abs = top_idx.gather(/*dim=*/-1, src_blk_in_topk);    // [B, Hkv, K_sel]
    auto k_offset_in_blk = k_slot_mod.unsqueeze(0).unsqueeze(0)
        .expand({B, Hkv, K_sel});                                       // [B, Hkv, K_sel]
    auto absolute_k_pos = src_blk_abs * block_size + k_offset_in_blk;  // [B, Hkv, K_sel]

    auto q_pos = torch::arange(block_size, opts_idx) + qb * block_size; // [block_size]

    // [B, Hkv, block_size, K_sel] = (q_pos[:,None] >= absolute_k_pos[None,:])
    auto causal = q_pos.unsqueeze(-1).unsqueeze(0).unsqueeze(0)         // [1,1,block_size,1]
                       >= absolute_k_pos.unsqueeze(-2);                 // [B,Hkv,1,K_sel]
    auto causal_h = causal.repeat_interleave(group, /*dim=*/1);          // [B, H, block_size, K_sel]

    attn_scores = attn_scores.masked_fill(causal_h.logical_not(),
        -std::numeric_limits<float>::infinity());

    auto attn_probs = torch::softmax(attn_scores, /*dim=*/-1);
    auto out_this   = torch::matmul(attn_probs, v_exp);  // [B, H, block_size, D]
    out_blocks.push_back(out_this);
  }

  auto out_pad = torch::stack(out_blocks, /*dim=*/2)
      .reshape({B, H, S_pad, D});
  // Slice off any pad we added.
  return out_pad.slice(/*dim=*/2, /*start=*/0, /*end=*/S);
}

}  // namespace olmo_cpp
