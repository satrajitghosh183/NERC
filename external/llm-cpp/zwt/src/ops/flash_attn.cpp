#include "zwt/ops/flash_attn.hpp"
#include "zwt/ops/attn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace zwt::ops {

// CPU tiled online-softmax forward. Per-row state: (m, l, O). For each K/V
// tile we compute the tile's local max, rescale the running state by
// exp(m_old - m_new), then accumulate exp(s - m_new) into l and V into O.
// At the end O /= l and lse = m + log(l).
//
// Memory footprint for intermediates: O(D) per row, no Sq x Sk matrix.
void flash_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                     Tensor& out, Tensor& lse,
                     bool is_causal, float softmax_scale) {
  if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4) {
    throw std::runtime_error("flash_attention: expected rank-4 q/k/v");
  }
  const int64_t B  = q.dim(0);
  const int64_t H  = q.dim(1);
  const int64_t Sq = q.dim(2);
  const int64_t D  = q.dim(3);
  const int64_t Hkv = k.dim(1);
  const int64_t Sk  = k.dim(2);
  if (Hkv != H) {
    throw std::runtime_error(
        "flash_attention: pre-expand K/V to H heads for GQA");
  }
  if (softmax_scale < 0) softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));

  if (q.device().is_cuda()) {
    // CUDA FA-2 kernel lives in flash_attn.cu (not yet written). Until then
    // we satisfy the contract by delegating to sdpa and leaving lse zeroed.
    // Tests that actually need lse (backward) must run on CPU for now.
    sdpa(q, k, v, out, is_causal, softmax_scale);
    lse.zero_();
    return;
  }

  const int64_t BH = B * H;
  const float* Qp = q.as<float>();
  const float* Kp = k.as<float>();
  const float* Vp = v.as<float>();
  float*       Op = out.as<float>();
  float*       Lp = lse.as<float>();

  constexpr int64_t kBc = 64;

  for (int64_t bh = 0; bh < BH; ++bh) {
    const float* Qbh = Qp + bh * Sq * D;
    const float* Kbh = Kp + bh * Sk * D;
    const float* Vbh = Vp + bh * Sk * D;
    float*       Obh = Op + bh * Sq * D;
    float*       Lbh = Lp + bh * Sq;

    for (int64_t i = 0; i < Sq; ++i) {
      const float* qi = Qbh + i * D;
      float*       oi = Obh + i * D;
      float m_i = -std::numeric_limits<float>::infinity();
      float l_i = 0.f;
      for (int64_t d = 0; d < D; ++d) oi[d] = 0.f;

      const int64_t j_end = is_causal ? std::min(i + 1, Sk) : Sk;

      for (int64_t jb = 0; jb < j_end; jb += kBc) {
        const int64_t jb_end = std::min(jb + kBc, j_end);

        float s_tile[kBc];
        float m_tile = -std::numeric_limits<float>::infinity();
        for (int64_t j = jb; j < jb_end; ++j) {
          const float* kj = Kbh + j * D;
          float s = 0.f;
          for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
          s *= softmax_scale;
          s_tile[j - jb] = s;
          if (s > m_tile) m_tile = s;
        }

        const float m_new = std::max(m_i, m_tile);
        const float alpha = (m_i == -std::numeric_limits<float>::infinity())
                                ? 0.f : std::exp(m_i - m_new);
        l_i *= alpha;
        for (int64_t d = 0; d < D; ++d) oi[d] *= alpha;

        for (int64_t j = jb; j < jb_end; ++j) {
          const float p = std::exp(s_tile[j - jb] - m_new);
          l_i += p;
          const float* vj = Vbh + j * D;
          for (int64_t d = 0; d < D; ++d) oi[d] += p * vj[d];
        }
        m_i = m_new;
      }

      if (l_i > 0.f) {
        const float inv = 1.f / l_i;
        for (int64_t d = 0; d < D; ++d) oi[d] *= inv;
        Lbh[i] = m_i + std::log(l_i);
      } else {
        // Empty row (j_end == 0): emit zeros, lse = -inf.
        for (int64_t d = 0; d < D; ++d) oi[d] = 0.f;
        Lbh[i] = -std::numeric_limits<float>::infinity();
      }
    }
  }
}

// Backward. Uses the FA-2 identity rowsum(P * dP) = rowsum(O * dO) so we
// never materialize P*dP:
//   D_i   = rowsum(dO_i * O_i)
//   P_ij  = exp(S_ij * scale - lse_i)
//   dV_j += P_ij * dO_i
//   dP_ij = dO_i . V_j
//   dS_ij = P_ij * (dP_ij - D_i) * scale
//   dQ_i += dS_ij * K_j
//   dK_j += dS_ij * Q_i
//
// Work per row pair is O(D) — no Sq x Sk buffers allocated.
void flash_attention_backward(const Tensor& grad_out,
                              const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& out, const Tensor& lse,
                              Tensor& grad_q, Tensor& grad_k, Tensor& grad_v,
                              bool is_causal, float softmax_scale) {
  if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4 || grad_out.rank() != 4) {
    throw std::runtime_error("flash_attention_backward: expected rank-4 tensors");
  }
  const int64_t B  = q.dim(0);
  const int64_t H  = q.dim(1);
  const int64_t Sq = q.dim(2);
  const int64_t D  = q.dim(3);
  const int64_t Hkv = k.dim(1);
  const int64_t Sk  = k.dim(2);
  if (Hkv != H) {
    throw std::runtime_error(
        "flash_attention_backward: pre-expand K/V to H heads for GQA");
  }
  if (softmax_scale < 0) softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));

  if (q.device().is_cuda()) {
    sdpa_backward(grad_out, q, k, v, out,
                  grad_q, grad_k, grad_v, is_causal, softmax_scale);
    return;
  }

  const int64_t BH = B * H;
  const float* Qp  = q.as<float>();
  const float* Kp  = k.as<float>();
  const float* Vp  = v.as<float>();
  const float* Op  = out.as<float>();
  const float* Lp  = lse.as<float>();
  const float* dOp = grad_out.as<float>();
  float* dQp = grad_q.as<float>();
  float* dKp = grad_k.as<float>();
  float* dVp = grad_v.as<float>();

  std::memset(dQp, 0, static_cast<size_t>(BH * Sq * D) * sizeof(float));
  std::memset(dKp, 0, static_cast<size_t>(BH * Sk * D) * sizeof(float));
  std::memset(dVp, 0, static_cast<size_t>(BH * Sk * D) * sizeof(float));

  for (int64_t bh = 0; bh < BH; ++bh) {
    const float* Qbh  = Qp  + bh * Sq * D;
    const float* Kbh  = Kp  + bh * Sk * D;
    const float* Vbh  = Vp  + bh * Sk * D;
    const float* Obh  = Op  + bh * Sq * D;
    const float* Lbh  = Lp  + bh * Sq;
    const float* dObh = dOp + bh * Sq * D;
    float* dQbh = dQp + bh * Sq * D;
    float* dKbh = dKp + bh * Sk * D;
    float* dVbh = dVp + bh * Sk * D;

    for (int64_t i = 0; i < Sq; ++i) {
      const float* qi  = Qbh  + i * D;
      const float* oi  = Obh  + i * D;
      const float* doi = dObh + i * D;
      float*       dqi = dQbh + i * D;
      const float  lse_i = Lbh[i];

      float D_i = 0.f;
      for (int64_t d = 0; d < D; ++d) D_i += doi[d] * oi[d];

      const int64_t j_end = is_causal ? std::min(i + 1, Sk) : Sk;
      if (j_end <= 0) continue;

      for (int64_t j = 0; j < j_end; ++j) {
        const float* kj  = Kbh  + j * D;
        const float* vj  = Vbh  + j * D;
        float*       dkj = dKbh + j * D;
        float*       dvj = dVbh + j * D;

        float s = 0.f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        s *= softmax_scale;
        const float p = std::exp(s - lse_i);

        float dp = 0.f;
        for (int64_t d = 0; d < D; ++d) dp += doi[d] * vj[d];
        const float ds = p * (dp - D_i) * softmax_scale;

        for (int64_t d = 0; d < D; ++d) dvj[d] += p * doi[d];
        for (int64_t d = 0; d < D; ++d) dqi[d] += ds * kj[d];
        for (int64_t d = 0; d < D; ++d) dkj[d] += ds * qi[d];
      }
    }
  }
}

}  // namespace zwt::ops
