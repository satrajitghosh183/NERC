#pragma once

/**
 * include/olmo_cpp/backend/fused_lm_head_ce.hpp
 *
 * Fused LM-head + softmax-cross-entropy — item A3.
 *
 *     loss = mean_{n: labels[n] != ignore_index} (
 *                logsumexp(h[n] @ W^T) - (h[n] @ W^T)[labels[n]] )
 *
 * The unfused chain (lm_head + F.cross_entropy) materializes the full
 * `[B*S, V]` logits tensor and the log_softmax intermediate. At V≈50k
 * and B*S≈16k that's ~3.3 GB written + read back. The fused kernel
 * computes per-row online softmax in registers, never writing the
 * full logits tensor; only the scalar loss + per-row reductions touch
 * HBM. Saving the [B*S, V] write costs ≈22 ms / step at 5060-Ti HBM,
 * 1.5-2% step time at 125M shape — more at larger vocabs.
 *
 * Backward (custom autograd Function): recomputes logits via one
 * fast_linear pass, derives softmax probs, subtracts one_hot at the
 * label, then emits grad_h and grad_w via fast_matmul. Net: +1 GEMM
 * vs the unfused backward, -2 large HBM writes/reads.
 */

#include <torch/torch.h>
#include <cstdint>

namespace olmo_cpp {

/// Returns a scalar tensor: mean cross-entropy over non-ignored rows.
/// Numerically identical (within fp32 epsilon) to:
///   F.cross_entropy(h @ weight.T, labels, ignore_index=ignore_index)
///
/// Shapes:
///   h        : [N, d_model]    (already pre-norm'd if the LM head has one)
///   weight   : [V, d_model]
///   labels   : [N]             int64
/// Output:
///   loss     : scalar
torch::Tensor fused_lm_head_ce(torch::Tensor h,
                                 torch::Tensor weight,
                                 torch::Tensor labels,
                                 int64_t ignore_index = -100);

#ifdef OLMO_HAS_CUDA_KERNELS
torch::Tensor fused_lm_head_ce_cuda(torch::Tensor h,
                                      torch::Tensor weight,
                                      torch::Tensor labels,
                                      int64_t ignore_index);
#endif

torch::Tensor fused_lm_head_ce_cpu(torch::Tensor h,
                                     torch::Tensor weight,
                                     torch::Tensor labels,
                                     int64_t ignore_index);

/// Autograd-aware chunked CE.  Tiles the vocabulary in Vc=4096 columns so
/// the full [N,V] logit tensor is never materialised.  Peak live allocation
/// is n_chunks × [N,Vc] ≈ 13 GB (vs ~53 GB) — fits CUDA-graph private pool.
/// Backward is handled by PyTorch autograd (mm + element-wise ops).
torch::Tensor fused_lm_head_ce_autograd(torch::Tensor h,
                                          torch::Tensor weight,
                                          torch::Tensor labels,
                                          int64_t ignore_index = -100);

}  // namespace olmo_cpp
