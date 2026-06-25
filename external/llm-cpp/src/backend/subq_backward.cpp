/**
 * src/backend/subq_backward.cpp
 *
 * Content-selected attention backward (item HH / 7 follow-on).
 *
 * Forward (already shipped in kernels/sparse_attn_decode.cu):
 *   1. Score each cached K position against the query via a small
 *      content-selection head.
 *   2. Pick top-k positions (selection_mask is a 0/1 indicator over
 *      cached tokens).
 *   3. Run attention over only the selected positions: out = softmax(q @ K_sel) @ V_sel.
 *
 * Backward: gradient flows through the selected positions normally;
 * the discrete top-k selection itself is non-differentiable, so we
 * apply the Straight-Through Estimator (STE): forward selects k items
 * hard, backward passes gradient through the selection scores
 * unchanged ("as if it were the identity").
 *
 * This file wires the autograd::Function so training can run with
 * SubQ in the inner loop and still produce gradients for the model
 * weights AND the selection head's parameters.
 *
 * For inference (no_grad), the existing sparse_attn_decode kernel is
 * the path; this autograd wrapper is the *training* entry point.
 */

#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>

namespace olmo_cpp {

namespace {

struct SubQAttentionFunction
    : public torch::autograd::Function<SubQAttentionFunction> {
  /// q, k, v: standard attention tensors [B, H, S, D].
  /// selection_scores: [B, H, S_q, S_k] — content-similarity scores
  ///   produced by the selection head.
  /// top_k: how many K/V positions per query to keep.
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor q,
                                 torch::Tensor k,
                                 torch::Tensor v,
                                 torch::Tensor selection_scores,
                                 int64_t top_k) {
    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t Sq = q.size(2);
    const int64_t Sk = k.size(2);
    const int64_t D = q.size(3);

    // Pick top_k positions per query position based on selection_scores.
    auto kk = std::min<int64_t>(top_k, Sk);
    auto topk = selection_scores.topk(kk, /*dim=*/-1);
    auto idx = std::get<1>(topk);                              // [B, H, Sq, kk]

    // Gather K and V at the selected indices. Output shape [B, H, Sq, kk, D].
    auto idx_e = idx.unsqueeze(-1).expand({B, H, Sq, kk, D});
    auto k_e   = k.unsqueeze(2).expand({B, H, Sq, Sk, D});
    auto v_e   = v.unsqueeze(2).expand({B, H, Sq, Sk, D});
    auto k_sel = k_e.gather(3, idx_e);                          // [B, H, Sq, kk, D]
    auto v_sel = v_e.gather(3, idx_e);

    // Attention: [B, H, Sq, 1, D] @ [B, H, Sq, D, kk] -> [B, H, Sq, 1, kk]
    auto q_b = q.unsqueeze(-2);                                 // [B, H, Sq, 1, D]
    auto scale = 1.0 / std::sqrt(static_cast<double>(D));
    auto scores = torch::matmul(q_b, k_sel.transpose(-1, -2)) * scale;  // [B, H, Sq, 1, kk]
    auto attn = torch::softmax(scores, -1);
    auto out = torch::matmul(attn, v_sel).squeeze(-2);          // [B, H, Sq, D]

    ctx->save_for_backward({q, k, v, selection_scores, idx, attn});
    ctx->saved_data["top_k"] = top_k;
    return out;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto q   = saved[0];
    auto k   = saved[1];
    auto v   = saved[2];
    auto sel = saved[3];
    auto idx = saved[4];
    auto attn = saved[5];
    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t Sq = q.size(2);
    const int64_t Sk = k.size(2);
    const int64_t D = q.size(3);
    const int64_t kk = idx.size(-1);

    auto grad_out = grad_outputs[0];                            // [B, H, Sq, D]

    // Recompute K_sel, V_sel for backward.
    auto idx_e = idx.unsqueeze(-1).expand({B, H, Sq, kk, D});
    auto k_e = k.unsqueeze(2).expand({B, H, Sq, Sk, D});
    auto v_e = v.unsqueeze(2).expand({B, H, Sq, Sk, D});
    auto k_sel = k_e.gather(3, idx_e);
    auto v_sel = v_e.gather(3, idx_e);

    // grad_attn = grad_out.unsqueeze(-2) @ v_sel.T  →  [B, H, Sq, 1, kk]
    auto go_b = grad_out.unsqueeze(-2);
    auto grad_attn = torch::matmul(go_b, v_sel.transpose(-1, -2));
    // grad_v_sel = attn.T @ go_b  → [B, H, Sq, kk, D]
    auto grad_v_sel = torch::matmul(attn.transpose(-1, -2), go_b);

    // grad_scores via softmax bwd: ds = attn * (grad_attn - sum(grad_attn*attn))
    auto sum_term = (grad_attn * attn).sum(-1, /*keepdim=*/true);
    auto grad_scores = attn * (grad_attn - sum_term);
    auto scale = 1.0 / std::sqrt(static_cast<double>(D));
    grad_scores = grad_scores * scale;
    // grad_q = grad_scores @ k_sel ; grad_k_sel = grad_scores.T @ q_b
    auto grad_q = torch::matmul(grad_scores, k_sel).squeeze(-2);
    auto grad_k_sel = torch::matmul(grad_scores.transpose(-1, -2),
                                      q.unsqueeze(-2));         // [B, H, Sq, kk, D]

    // Scatter grad_k_sel / grad_v_sel back to grad_k / grad_v at the
    // selected indices. Other positions get 0.
    auto grad_k = torch::zeros_like(k);
    auto grad_v = torch::zeros_like(v);
    {
      auto idx_flat = idx.reshape({B, H, Sq * kk});
      auto gk_flat  = grad_k_sel.reshape({B, H, Sq * kk, D});
      auto gv_flat  = grad_v_sel.reshape({B, H, Sq * kk, D});
      // We accumulate into grad_k[b, h, idx_flat[b,h,i], :] += gk_flat[b, h, i, :].
      // Using scatter_add_ over flattened index.
      auto idx_exp = idx_flat.unsqueeze(-1).expand({B, H, Sq * kk, D});
      grad_k.scatter_add_(2, idx_exp, gk_flat);
      grad_v.scatter_add_(2, idx_exp, gv_flat);
    }

    // STE: gradient on selection_scores is the IDENTITY of the
    // selection's downstream gradient. We approximate by gradient =
    // (broadcast of grad_attn into the [Sk] slot at idx, 0 elsewhere).
    auto grad_sel = torch::zeros_like(sel);
    {
      // grad_attn shape: [B, H, Sq, 1, kk]; squeeze to [B, H, Sq, kk].
      auto ga = grad_attn.squeeze(-2);
      grad_sel.scatter_(/*dim=*/-1, idx, ga);
    }

    return {grad_q, grad_k, grad_v, grad_sel, torch::Tensor()};
  }
};

}  // namespace

/// Differentiable SubQ attention with STE-on-selection backward.
torch::Tensor subq_attention_autograd(torch::Tensor q,
                                        torch::Tensor k,
                                        torch::Tensor v,
                                        torch::Tensor selection_scores,
                                        int64_t top_k) {
  return SubQAttentionFunction::apply(q, k, v, selection_scores, top_k);
}

}  // namespace olmo_cpp
