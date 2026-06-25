#pragma once

// Internal header: raw-pointer entry points for the CUDA kernels. Public
// Tensor-taking ops in elementwise.cpp / norm.cpp / xent.cpp dispatch to
// these when the tensors live on CUDA.
//
// Only include this from .cu files or .cpp files that are wrapped in
// `#ifdef USE_CUDA` — the signatures depend on CUDA headers.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace zwt::ops::k {

// -- Elementwise --------------------------------------------------------------

void axpy_bf16(__nv_bfloat16* y, const __nv_bfloat16* x, float alpha,
               int64_t n, cudaStream_t s);
void scale_bf16(__nv_bfloat16* y, float alpha, int64_t n, cudaStream_t s);
void add_bias_bf16(__nv_bfloat16* y, const __nv_bfloat16* bias, int64_t rows,
                   int64_t cols, cudaStream_t s);
void add_bf16(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
              int64_t n, cudaStream_t s);

void silu_mul_bf16(__nv_bfloat16* out, const __nv_bfloat16* gate,
                   const __nv_bfloat16* up, int64_t n, cudaStream_t s);
void silu_mul_backward_bf16(const __nv_bfloat16* grad_out,
                            const __nv_bfloat16* gate, const __nv_bfloat16* up,
                            __nv_bfloat16* grad_gate, __nv_bfloat16* grad_up,
                            int64_t n, cudaStream_t s);

// Fused gate+up SwiGLU — combined input [N, 2H], output [N, H].
void silu_mul_gated_bf16(__nv_bfloat16* out, const __nv_bfloat16* combined,
                         int64_t N, int64_t H, cudaStream_t s);
void silu_mul_gated_backward_bf16(const __nv_bfloat16* grad_out,
                                  const __nv_bfloat16* combined,
                                  __nv_bfloat16* grad_combined,
                                  int64_t N, int64_t H, cudaStream_t s);

void bias_backward_bf16(const __nv_bfloat16* grad_y, float* grad_bias,
                        int64_t rows, int64_t cols, cudaStream_t s);

// Transpose [B,S,H,D] <-> [B,H,S,D] (head-major layout swap).
void transpose_bshd_bhsd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int64_t B, int64_t S, int64_t H, int64_t D,
                              cudaStream_t s);
void transpose_bhsd_bshd_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int64_t B, int64_t S, int64_t H, int64_t D,
                              cudaStream_t s);

// GQA head broadcast: [B, Hkv, S, D] -> [B, Hkv*group, S, D] bf16.
void repeat_kv_heads_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                          int64_t B, int64_t Hkv, int64_t S, int64_t D,
                          int64_t group, cudaStream_t s);
// GQA backward: [B, Hkv*group, S, D] -> [B, Hkv, S, D] via sum-reduce.
void reduce_kv_heads_sum_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                              int64_t B, int64_t Hkv, int64_t S, int64_t D,
                              int64_t group, cudaStream_t s);

// -- RMSNorm ------------------------------------------------------------------

void rmsnorm_fwd_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                      __nv_bfloat16* y, float* rstd,
                      int64_t rows, int64_t cols, float eps, cudaStream_t s);

void rmsnorm_residual_fwd_bf16(const __nv_bfloat16* x, const __nv_bfloat16* res,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* y, __nv_bfloat16* sum_out,
                               float* rstd,
                               int64_t rows, int64_t cols, float eps,
                               cudaStream_t s);

void rmsnorm_bwd_bf16(const __nv_bfloat16* grad_y, const __nv_bfloat16* x,
                      const __nv_bfloat16* weight, const float* rstd,
                      __nv_bfloat16* grad_x, float* grad_weight,
                      int64_t rows, int64_t cols, float eps, cudaStream_t s);

// -- Cross entropy + softmax (fused) -----------------------------------------

void softmax_xent_fused_bf16(const __nv_bfloat16* logits, const int64_t* targets,
                             float* loss, __nv_bfloat16* grad_logits,
                             int64_t rows, int64_t vocab,
                             int64_t ignore_index, cudaStream_t s);

}  // namespace zwt::ops::k
