#include "zwt/ops/attn.hpp"
#include "zwt/ops/gemm.hpp"
#include "zwt/core/stream.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#endif

// Forward decl for the causal-softmax kernel launcher in attn.cu.
namespace zwt::ops::k {
#ifdef USE_CUDA
void masked_softmax_inplace_bf16(__nv_bfloat16* scores, int64_t bh, int64_t sq,
                                 int64_t sk, float scale, bool causal,
                                 cudaStream_t s);
void softmax_backward_scaled_bf16(__nv_bfloat16* grad_scores_io,
                                  const __nv_bfloat16* probs,
                                  int64_t bh, int64_t sq, int64_t sk,
                                  float scale, cudaStream_t s);
#endif
}

namespace zwt::ops {

// SDPA as three steps: scores = Q @ K^T * scale; softmax(scores, causal);
// out = scores @ V.  The softmax kernel folds the causal mask and the scale.
//
// Layout (head-major): Q/K/V = [B*H, S, D] after flattening the batch-head dim.
// This form lets us call a single strided-batched GEMM per matmul.
//
// Shape inputs: q [B,H,S,D], k [B,Hkv,S,D], v [B,Hkv,S,D]. GQA: Hkv | H.
void sdpa(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out,
          bool is_causal, float softmax_scale) {
  if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4) {
    throw std::runtime_error("sdpa: expected rank-4 q/k/v [B,H,S,D]");
  }
  const int64_t B  = q.dim(0);
  const int64_t H  = q.dim(1);
  const int64_t Sq = q.dim(2);
  const int64_t D  = q.dim(3);
  const int64_t Hkv = k.dim(1);
  const int64_t Sk  = k.dim(2);
  if (Hkv != H) {
    throw std::runtime_error(
        "sdpa: GQA head broadcast not yet wired; pre-expand K/V to H heads");
  }
  if (softmax_scale < 0) softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));

  // Flatten (B, H) -> BH.
  const int64_t BH = B * H;
  Shape qbh{BH, Sq, D};
  Shape kbh{BH, Sk, D};
  Shape vbh{BH, Sk, D};
  Shape sbh{BH, Sq, Sk};
  Tensor qv = q.view(qbh);
  Tensor kv = k.view(kbh);
  Tensor vv = v.view(vbh);
  Tensor ov = out.view(qbh);

  // scores = Q @ K^T  (shape [BH, Sq, Sk])
  Tensor scores = empty_scratch({BH, Sq, Sk}, q.dtype(), q.device());
  gemm_batched(qv, /*transa=*/false, kv, /*transb=*/true, scores, 1.0f, 0.0f);

  // softmax(scores * scale, causal)
  if (q.device().is_cuda()) {
#ifdef USE_CUDA
    k::masked_softmax_inplace_bf16(
        reinterpret_cast<__nv_bfloat16*>(scores.data()),
        BH, Sq, Sk, softmax_scale, is_causal,
        reinterpret_cast<cudaStream_t>(compute_stream(q.device()).handle));
#endif
  } else {
    // CPU f32 ref
    float* s = scores.as<float>();
    for (int64_t b = 0; b < BH; ++b) {
      for (int64_t i = 0; i < Sq; ++i) {
        float* row = s + b * Sq * Sk + i * Sk;
        float m = -std::numeric_limits<float>::infinity();
        int64_t row_lim = is_causal ? std::min(i + 1, Sk) : Sk;
        for (int64_t j = 0; j < Sk; ++j) {
          row[j] *= softmax_scale;
          if (is_causal && j > i) row[j] = -std::numeric_limits<float>::infinity();
          if (j < row_lim) m = std::max(m, row[j]);
        }
        float ss = 0.f;
        for (int64_t j = 0; j < row_lim; ++j) { row[j] = std::exp(row[j] - m); ss += row[j]; }
        for (int64_t j = 0; j < row_lim; ++j) row[j] /= ss;
        for (int64_t j = row_lim; j < Sk; ++j) row[j] = 0.f;
      }
    }
  }

  // out = scores @ V  (shape [BH, Sq, D])
  gemm_batched(scores, /*transa=*/false, vv, /*transb=*/false, ov, 1.0f, 0.0f);
}

// Backward: compute grad_q, grad_k, grad_v from grad_out.
// Reconstructs softmax(scores * scale) then uses the chain rule:
//   dL/dV     = P^T @ dL/dO                    (P = attn probs)
//   dL/dP     = dL/dO @ V^T
//   dL/dS     = (dL/dP - rowsum(dL/dP*P)) * P  (softmax jacobian)
//   dL/dQ     = dL/dS @ K * scale
//   dL/dK     = dL/dS^T @ Q * scale
//
// The current implementation recomputes P from Q,K (same as forward).
// A later iteration can stash logsumexp + P from forward to avoid recompute.
void sdpa_backward(const Tensor& grad_out, const Tensor& q, const Tensor& k,
                   const Tensor& v, const Tensor& /*out*/,
                   Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
                   bool is_causal, float softmax_scale) {
  if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4 || grad_out.rank() != 4) {
    throw std::runtime_error("sdpa_backward: expected rank-4 tensors");
  }
  const int64_t B  = q.dim(0);
  const int64_t H  = q.dim(1);
  const int64_t Sq = q.dim(2);
  const int64_t D  = q.dim(3);
  const int64_t Hkv = k.dim(1);
  const int64_t Sk  = k.dim(2);
  if (Hkv != H) {
    throw std::runtime_error(
        "sdpa_backward: GQA pre-expand K/V to H heads before calling");
  }
  if (softmax_scale < 0) softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));

  const int64_t BH = B * H;
  Shape qbh{BH, Sq, D};
  Shape kbh{BH, Sk, D};
  Shape sbh{BH, Sq, Sk};

  Tensor qv  = q.view(qbh);
  Tensor kv  = k.view(kbh);
  Tensor vv  = v.view(kbh);
  Tensor gov = grad_out.view(qbh);
  Tensor gqv = grad_q.view(qbh);
  Tensor gkv = grad_k.view(kbh);
  Tensor gvv = grad_v.view(kbh);

  // Recompute P = softmax(Q @ K^T * scale, causal).
  Tensor probs = empty_scratch(sbh, q.dtype(), q.device());
  gemm_batched(qv, /*transa=*/false, kv, /*transb=*/true, probs, 1.0f, 0.0f);
  if (q.device().is_cuda()) {
#ifdef USE_CUDA
    k::masked_softmax_inplace_bf16(
        reinterpret_cast<__nv_bfloat16*>(probs.data()),
        BH, Sq, Sk, softmax_scale, is_causal,
        reinterpret_cast<cudaStream_t>(compute_stream(q.device()).handle));
#endif
  } else {
    // CPU f32 ref — mirror sdpa() forward math.
    float* s = probs.as<float>();
    for (int64_t b = 0; b < BH; ++b) {
      for (int64_t i = 0; i < Sq; ++i) {
        float* row = s + b * Sq * Sk + i * Sk;
        float m = -std::numeric_limits<float>::infinity();
        int64_t lim = is_causal ? std::min(i + 1, Sk) : Sk;
        for (int64_t j = 0; j < Sk; ++j) {
          row[j] *= softmax_scale;
          if (is_causal && j > i) row[j] = -std::numeric_limits<float>::infinity();
          if (j < lim) m = std::max(m, row[j]);
        }
        float ss = 0.f;
        for (int64_t j = 0; j < lim; ++j) { row[j] = std::exp(row[j] - m); ss += row[j]; }
        for (int64_t j = 0; j < lim; ++j) row[j] /= ss;
        for (int64_t j = lim; j < Sk; ++j) row[j] = 0.f;
      }
    }
  }

  // grad_V = P^T @ grad_out  (batched).
  gemm_batched(probs, /*transa=*/true, gov, /*transb=*/false, gvv, 1.0f, 0.0f);

  // grad_P = grad_out @ V^T  (into a fresh scratch; same shape as probs).
  Tensor grad_scores = empty_scratch(sbh, q.dtype(), q.device());
  gemm_batched(gov, /*transa=*/false, vv, /*transb=*/true, grad_scores, 1.0f, 0.0f);

  // grad_scores = scale * P * (grad_P - sum_k(P*grad_P))   (softmax jacobian).
  if (q.device().is_cuda()) {
#ifdef USE_CUDA
    k::softmax_backward_scaled_bf16(
        reinterpret_cast<__nv_bfloat16*>(grad_scores.data()),
        reinterpret_cast<const __nv_bfloat16*>(probs.data()),
        BH, Sq, Sk, softmax_scale,
        reinterpret_cast<cudaStream_t>(compute_stream(q.device()).handle));
#endif
  } else {
    float* gs = grad_scores.as<float>();
    const float* p = probs.as<float>();
    for (int64_t b = 0; b < BH; ++b) {
      for (int64_t i = 0; i < Sq; ++i) {
        float* grow = gs + b * Sq * Sk + i * Sk;
        const float* prow = p + b * Sq * Sk + i * Sk;
        float rs = 0.f;
        for (int64_t j = 0; j < Sk; ++j) rs += prow[j] * grow[j];
        for (int64_t j = 0; j < Sk; ++j) grow[j] = softmax_scale * prow[j] * (grow[j] - rs);
      }
    }
  }

  // grad_Q = grad_scores @ K      (the scale is already folded in).
  gemm_batched(grad_scores, /*transa=*/false, kv, /*transb=*/false, gqv, 1.0f, 0.0f);
  // grad_K = grad_scores^T @ Q
  gemm_batched(grad_scores, /*transa=*/true,  qv, /*transb=*/false, gkv, 1.0f, 0.0f);
}

}  // namespace zwt::ops
