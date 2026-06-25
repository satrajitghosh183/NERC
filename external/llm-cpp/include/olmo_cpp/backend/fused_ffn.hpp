#pragma once

/**
 * include/olmo_cpp/backend/fused_ffn.hpp
 *
 * Fused SwiGLU FFN — item I.
 *
 *   y = w_down * (silu(gate) * up)        where gate, up = split(w_gate_up * x)
 *
 * Today's path: cuBLAS GEMM(x, w_gate_up) -> [B,S,2H] in HBM -> silu_mul
 * kernel reads/writes [B,S,2H]+[B,S,H] -> cuBLAS GEMM(act, w_down)
 * reads [B,S,H] writes [B,S,d]. Three large HBM roundtrips.
 *
 * Fused: keep the [2H]→[H]→[d] chain per row in shared memory. The
 * actual matmuls stay tensor-core-driven in the long-term variant
 * (mma / wgmma). Current commit ships the CPU reference, the host
 * dispatch wiring, and a naive CUDA kernel (one block per row, FMA
 * loops, no tensor cores) — correct, identical numerics, used as the
 * baseline against which the tensor-core variant lands.
 */

#include <torch/torch.h>

namespace olmo_cpp {

/// Fused FFN forward. Inputs:
///   x          : [B, S, d]
///   w_gate_up  : [2H, d]   (concatenated gate || up rows along the output)
///   w_down     : [d, H]
/// Output:
///   y          : [B, S, d]
torch::Tensor fused_ffn(torch::Tensor x,
                         torch::Tensor w_gate_up,
                         torch::Tensor w_down);

#ifdef OLMO_HAS_CUDA_KERNELS
torch::Tensor fused_ffn_cuda(torch::Tensor x,
                              torch::Tensor w_gate_up,
                              torch::Tensor w_down);
#endif

torch::Tensor fused_ffn_cpu(torch::Tensor x,
                             torch::Tensor w_gate_up,
                             torch::Tensor w_down);

/// Autograd-aware variant for training call sites.
torch::Tensor fused_ffn_autograd(torch::Tensor x,
                                   torch::Tensor w_gate_up,
                                   torch::Tensor w_down);

/// A1 — training-side forward that also produces gate_up so the
/// autograd backward can skip the recompute matmul. Routes through
/// the WMMA or TMA train variants on CUDA; falls back to the standard
/// fused_ffn path + an extra fast_linear on CPU.
std::pair<torch::Tensor, torch::Tensor>
fused_ffn_train(torch::Tensor x,
                 torch::Tensor w_gate_up,
                 torch::Tensor w_down);

#ifdef OLMO_HAS_CUDA_KERNELS
/// Tensor-core (WMMA) FFN kernel. Preferred path on sm_80+ for bf16
/// inputs with d / H multiples of 16. Falls back to the FMA-loop
/// fused_ffn_cuda otherwise.
torch::Tensor fused_ffn_wmma_cuda(torch::Tensor x,
                                    torch::Tensor w_gate_up,
                                    torch::Tensor w_down);

/// TMA (Tensor Memory Accelerator) variant. On sm_90+ async-loads
/// x tiles via cp.async.bulk.tensor + mbarrier; otherwise routes
/// internally to fused_ffn_wmma_cuda. Same numerics, same outputs.
torch::Tensor fused_ffn_tma_cuda(torch::Tensor x,
                                   torch::Tensor w_gate_up,
                                   torch::Tensor w_down);

/// A1 — training entry points that also output the gate_up
/// intermediate so the backward can skip the recompute matmul.
std::pair<torch::Tensor, torch::Tensor>
fused_ffn_wmma_train_cuda(torch::Tensor x,
                            torch::Tensor w_gate_up,
                            torch::Tensor w_down);

std::pair<torch::Tensor, torch::Tensor>
fused_ffn_tma_train_cuda(torch::Tensor x,
                           torch::Tensor w_gate_up,
                           torch::Tensor w_down);
#endif

}  // namespace olmo_cpp
