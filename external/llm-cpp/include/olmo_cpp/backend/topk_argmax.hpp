#pragma once

/**
 * include/olmo_cpp/backend/topk_argmax.hpp
 *
 * Fused GEMV + argmax — item K from the optimization roadmap.
 *
 * The speculative-decode verify step calls lm_head(hidden) and then
 * argmax over the resulting [B*S, V] logits tensor. Materializing the
 * full V=50257 logits row for every position is bandwidth-wasted —
 * speculative greedy verify only needs the argmax index per row.
 *
 * This kernel fuses GEMV with argmax: one CUDA block per (batch, seq)
 * row, threads cooperatively compute partial dot products against
 * chunks of the vocabulary, track the running max, and block-reduce
 * to one int32 argmax per row. No [V] tensor is written to HBM.
 *
 * CPU reference: equivalent torch::linear + argmax, slow but correct.
 *
 * Inputs:
 *   hidden : [B, S, d] (fp32 / bf16)
 *   weight : [V, d]    (fp32 / bf16) — LM head transposed convention
 * Output:
 *   argmax : [B, S] int64
 */

#include <torch/torch.h>

namespace olmo_cpp {

torch::Tensor lmhead_argmax(torch::Tensor hidden, torch::Tensor weight);

#ifdef OLMO_HAS_CUDA_KERNELS
torch::Tensor lmhead_argmax_cuda(torch::Tensor hidden, torch::Tensor weight);
#endif

torch::Tensor lmhead_argmax_cpu(torch::Tensor hidden, torch::Tensor weight);

}  // namespace olmo_cpp
