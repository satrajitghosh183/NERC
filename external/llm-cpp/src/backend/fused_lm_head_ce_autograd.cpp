/**
 * src/backend/fused_lm_head_ce_autograd.cpp
 *
 * Chunked cross-entropy for the LM head.  Replaces the earlier approach of
 * materialising the full [N, V] logit tensor (N=B*S=32768, V=50304, bf16 = 3.3 GB).
 *
 * Why chunked?
 * -----------
 * With 4 LM heads (main + 3 MTP) and a CUDA graph, every tensor allocated
 * during capture lives permanently in the graph's private pool.  Four heads,
 * each with 4–5 intermediate [N, V] tensors, needed ~53 GB simultaneously —
 * causing OOM on an 80 GB H100.
 *
 * The chunked approach tiles the vocabulary dimension in Vc=4096 columns.
 * The largest live tensor at any moment is [N, Vc] ≈ 256 MB, so the private
 * pool for CE is n_chunks × [N, Vc] ≈ 13 × 256 MB × 4 heads ≈ 13 GB instead
 * of ~53 GB.  cuda_graph=1 now fits.
 *
 * Algorithm (online log-sum-exp, same as Flash-Attention's online softmax):
 * ----------
 * Invariant after processing chunks 0…c-1:
 *   m = max of all logits seen so far  (per row)
 *   l = sum of exp(logit - m)          (per row)
 *
 * Update for chunk c:
 *   new_m = max(m, chunk_max)
 *   l     = l * exp(m - new_m) + sum(exp(lc - new_m))
 *   m     = new_m
 *
 * At the end: log_Z = log(l) + m = logsumexp over all V logits.
 * CE per row = -logit[label] + log_Z.
 *
 * Autodiff:
 * ---------
 * No custom Function wrapper: all ops (mm, max, exp, sum, gather, etc.) are
 * standard PyTorch ops whose backward is already registered.  PyTorch's
 * autograd handles the backward transparently.  The backward GEMMs are
 * grad_h += grad_lc @ Wc  and  grad_Wc = h.T @ grad_lc (cuBLAS, same speed
 * as our earlier matmul-based custom backward).
 *
 * CUDA-graph compatibility:
 * -------------------------
 * n_chunks = ceil(V / Vc) is a compile-time constant.  All tensor shapes are
 * fixed per-step.  weight.narrow(0, v0, Vc_actual) produces tensors of fixed
 * shape for each loop iteration (last chunk has Vc_actual = V % Vc, also
 * fixed).  No CPU–GPU synchronisation inside the loop.
 */

#include <torch/torch.h>
#include <algorithm>

namespace olmo_cpp {

torch::Tensor fused_lm_head_ce_autograd(torch::Tensor h,
                                          torch::Tensor weight,
                                          torch::Tensor labels,
                                          int64_t ignore_index) {
  const int64_t N  = h.size(0);
  const int64_t V  = weight.size(0);
  const int64_t Vc = 4096;                            // tile width over vocab
  const int64_t n_chunks = (V + Vc - 1) / Vc;

  // All running accumulators in fp32 for numerical stability.
  auto fp32_opts = h.options().dtype(torch::kFloat32);
  // m starts at -∞ so the first chunk's max always wins.
  auto m       = torch::full({N}, -1e38f, fp32_opts);  // running per-row max
  auto l       = torch::zeros({N}, fp32_opts);          // running sum-exp
  auto labeled = torch::zeros({N}, fp32_opts);          // logit at target label

  for (int64_t c = 0; c < n_chunks; ++c) {
    const int64_t v0       = c * Vc;
    const int64_t Vc_act   = std::min(Vc, V - v0);

    // [N, Vc_act] in bf16 via cuBLAS tensor cores, cast to fp32 for stable CE.
    auto Wc = weight.narrow(0, v0, Vc_act);              // view of weight, no copy
    auto lc = torch::mm(h, Wc.t()).to(torch::kFloat32); // [N, Vc_act] fp32

    // Online logsumexp update.
    auto cm    = std::get<0>(lc.max(/*dim=*/1));         // [N]
    auto new_m = torch::max(m, cm);                      // [N]
    l = l * (m - new_m).exp() +
        (lc - new_m.unsqueeze(1)).exp().sum(/*dim=*/1);
    m = new_m;

    // Accumulate the logit at the target label if it lives in this chunk.
    auto in_chunk  = (labels >= v0) & (labels < v0 + Vc_act);          // [N] bool
    auto local_idx = (labels - v0).clamp(0, Vc_act - 1);               // [N] int64
    auto gathered  = lc.gather(1, local_idx.unsqueeze(1)).squeeze(1);  // [N]
    labeled = labeled + gathered * in_chunk.to(torch::kFloat32);
  }

  // CE per row = -logit[label] + log(sum_exp) + max_logit
  auto log_Z  = l.log() + m;                              // [N]
  auto ce_row = -labeled + log_Z;                          // [N]

  // Average over valid rows (label != ignore_index and in [0, V)).
  auto valid  = (labels != ignore_index) & (labels >= 0) & (labels < V);
  auto valid_f = valid.to(torch::kFloat32);
  auto loss   = (ce_row * valid_f).sum() /
                valid_f.sum().clamp_min(1.0f);

  // Keep the loss in fp32. The CE reduction is already fp32; casting the
  // scalar back to bf16 (8-bit mantissa, ULP ~0.06 around a CE of ~11) only
  // throws away precision in the logged value and in the backward seed. The
  // caller's accum_loss_tensor and the main+MTP combine are all fp32, so an
  // fp32 scalar here is strictly cleaner and costs nothing.
  return loss;
}

}  // namespace olmo_cpp
