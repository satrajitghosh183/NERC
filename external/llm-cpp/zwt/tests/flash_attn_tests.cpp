// FlashAttention-2 CPU-reference tests.
//
// The CPU path of ops::flash_attention is the gold reference against which
// the forthcoming CUDA FA-2 kernel will be validated. We cross-check it here
// against a local naive attention implementation (plain triple-loop softmax,
// no tiling, no online max/sum). On CPU fp32 they must agree within fp
// roundoff — the tile-accumulation order differs so bit-exact is not
// expected.
//
// Backward is validated by (a) comparison against the naive backward derived
// from the same triple-loop softmax, and (b) a finite-difference gradcheck
// on a small problem.
//
// We avoid ops::sdpa because its CUDA path uses gemm_batched which has no
// CPU implementation — calling it on this machine throws.

#include "zwt/core/allocator.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/ops/flash_attn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace zwt;

namespace {

int g_failed = 0;

void expect(bool cond, const std::string& what) {
  std::printf("[%s] %s\n", cond ? "PASS" : "FAIL", what.c_str());
  if (!cond) ++g_failed;
}

Tensor rand_f32(const Shape& s, uint64_t seed, float lo = -1.f, float hi = 1.f) {
  Tensor t = empty(s, DType::F32, Device::cpu());
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  for (int64_t i = 0; i < t.numel(); ++i) t.as<float>()[i] = d(rng);
  return t;
}

double max_abs(const float* a, const float* b, int64_t n) {
  double m = 0;
  for (int64_t i = 0; i < n; ++i) {
    double d = std::fabs(double(a[i]) - double(b[i]));
    if (d > m) m = d;
  }
  return m;
}

// Naive causal/non-causal attention reference. No tiling.
//   S = Q @ K^T * scale, mask, softmax along last dim, O = P @ V.
// Inputs/outputs are [B,H,S,D] row-major. Also returns probs if non-null.
void naive_attn(const float* Q, const float* K, const float* V,
                float* O, int64_t B, int64_t H, int64_t Sq, int64_t Sk,
                int64_t D, bool causal, float scale,
                std::vector<float>* probs_out = nullptr) {
  const int64_t BH = B * H;
  if (probs_out) probs_out->assign(static_cast<size_t>(BH * Sq * Sk), 0.f);
  for (int64_t bh = 0; bh < BH; ++bh) {
    for (int64_t i = 0; i < Sq; ++i) {
      const float* qi = Q + bh * Sq * D + i * D;
      float* oi = O + bh * Sq * D + i * D;
      std::vector<float> s(Sk, -std::numeric_limits<float>::infinity());
      const int64_t lim = causal ? std::min(i + 1, Sk) : Sk;
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < lim; ++j) {
        const float* kj = K + bh * Sk * D + j * D;
        float dot = 0.f;
        for (int64_t d = 0; d < D; ++d) dot += qi[d] * kj[d];
        s[j] = dot * scale;
        if (s[j] > m) m = s[j];
      }
      float ss = 0.f;
      for (int64_t j = 0; j < lim; ++j) { s[j] = std::exp(s[j] - m); ss += s[j]; }
      for (int64_t j = 0; j < lim; ++j) s[j] /= ss;
      for (int64_t j = lim; j < Sk; ++j) s[j] = 0.f;

      if (probs_out) {
        float* dst = probs_out->data() + bh * Sq * Sk + i * Sk;
        for (int64_t j = 0; j < Sk; ++j) dst[j] = s[j];
      }
      for (int64_t d = 0; d < D; ++d) oi[d] = 0.f;
      for (int64_t j = 0; j < Sk; ++j) {
        if (s[j] == 0.f) continue;
        const float* vj = V + bh * Sk * D + j * D;
        for (int64_t d = 0; d < D; ++d) oi[d] += s[j] * vj[d];
      }
    }
  }
}

// Naive backward. Given probs P and grad_out dO, compute grad_{Q,K,V}.
void naive_attn_backward(const float* Q, const float* K, const float* V,
                         const float* dO, const float* P,
                         float* dQ, float* dK, float* dV,
                         int64_t B, int64_t H, int64_t Sq, int64_t Sk,
                         int64_t D, bool causal, float scale) {
  const int64_t BH = B * H;
  std::fill(dQ, dQ + BH * Sq * D, 0.f);
  std::fill(dK, dK + BH * Sk * D, 0.f);
  std::fill(dV, dV + BH * Sk * D, 0.f);
  for (int64_t bh = 0; bh < BH; ++bh) {
    for (int64_t i = 0; i < Sq; ++i) {
      const float* qi = Q + bh * Sq * D + i * D;
      const float* doi = dO + bh * Sq * D + i * D;
      float* dqi = dQ + bh * Sq * D + i * D;
      const float* pi = P + bh * Sq * Sk + i * Sk;
      // dP_ij = dO_i . V_j, D_i = sum_j(P_ij * dP_ij) = sum_d(dO_id * O_id)
      // Compute dP and rowsum.
      std::vector<float> dp(Sk, 0.f);
      const int64_t lim = causal ? std::min(i + 1, Sk) : Sk;
      for (int64_t j = 0; j < lim; ++j) {
        const float* vj = V + bh * Sk * D + j * D;
        float x = 0.f;
        for (int64_t d = 0; d < D; ++d) x += doi[d] * vj[d];
        dp[j] = x;
      }
      float rs = 0.f;
      for (int64_t j = 0; j < lim; ++j) rs += pi[j] * dp[j];
      // dS_ij = P_ij * (dP_ij - rs) * scale. Accumulate dQ/dK/dV.
      for (int64_t j = 0; j < lim; ++j) {
        const float* kj = K + bh * Sk * D + j * D;
        const float* vj = V + bh * Sk * D + j * D;
        float* dkj = dK + bh * Sk * D + j * D;
        float* dvj = dV + bh * Sk * D + j * D;
        const float ds = pi[j] * (dp[j] - rs) * scale;
        for (int64_t d = 0; d < D; ++d) {
          dvj[d] += pi[j] * doi[d];
          dqi[d] += ds * kj[d];
          dkj[d] += ds * qi[d];
        }
      }
    }
  }
}

}  // namespace

int main() {
  set_activation_arena_capacity(size_t(64) << 20);

  // --------------------------------------------------------------------
  // Forward: flash_attention vs naive_attn, causal & noncausal.
  // --------------------------------------------------------------------
  const int64_t B = 2, H = 4, S = 48, D = 16;
  const Shape qkv{B, H, S, D};
  const Shape lse_shape{B, H, S};
  const float scale = 1.f / std::sqrt(float(D));

  Tensor q = rand_f32(qkv, 0xC0DE);
  Tensor k = rand_f32(qkv, 0xBEEF);
  Tensor v = rand_f32(qkv, 0xFEED);
  const int64_t Nout = B * H * S * D;

  std::vector<float> ref_out(Nout);
  std::vector<float> ref_probs;
  naive_attn(q.as<float>(), k.as<float>(), v.as<float>(), ref_out.data(),
             B, H, S, S, D, true, scale, &ref_probs);

  Tensor out_fa = zeros(qkv, DType::F32, Device::cpu());
  Tensor lse    = zeros(lse_shape, DType::F32, Device::cpu());
  ops::flash_attention(q, k, v, out_fa, lse, true);

  const double fwd_err = max_abs(ref_out.data(), out_fa.as<float>(), Nout);
  std::printf("fwd causal max_abs = %.3e\n", fwd_err);
  expect(fwd_err < 1e-5, "flash_attention fwd matches naive (causal)");

  // lse finite for all rows that have at least one key (i.e. every row when
  // causal and i>=0 with Sk>=1).
  bool lse_finite = true;
  for (int64_t i = 0; i < lse.numel(); ++i) {
    if (!std::isfinite(lse.as<float>()[i])) { lse_finite = false; break; }
  }
  expect(lse_finite, "flash_attention writes finite lse on populated rows");

  // non-causal forward
  std::vector<float> ref_out_nc(Nout);
  std::vector<float> ref_probs_nc;
  naive_attn(q.as<float>(), k.as<float>(), v.as<float>(), ref_out_nc.data(),
             B, H, S, S, D, false, scale, &ref_probs_nc);
  Tensor out_fa_nc = zeros(qkv, DType::F32, Device::cpu());
  Tensor lse_nc    = zeros(lse_shape, DType::F32, Device::cpu());
  ops::flash_attention(q, k, v, out_fa_nc, lse_nc, false);
  const double fwd_err_nc =
      max_abs(ref_out_nc.data(), out_fa_nc.as<float>(), Nout);
  std::printf("fwd noncausal max_abs = %.3e\n", fwd_err_nc);
  expect(fwd_err_nc < 1e-5, "flash_attention fwd matches naive (noncausal)");

  // --------------------------------------------------------------------
  // Backward: flash_attention_backward vs naive_attn_backward.
  // --------------------------------------------------------------------
  Tensor go = rand_f32(qkv, 0xA110CED);
  std::vector<float> ref_dQ(Nout), ref_dK(Nout), ref_dV(Nout);
  naive_attn_backward(q.as<float>(), k.as<float>(), v.as<float>(),
                      go.as<float>(), ref_probs.data(),
                      ref_dQ.data(), ref_dK.data(), ref_dV.data(),
                      B, H, S, S, D, true, scale);

  Tensor dQ = zeros(qkv, DType::F32, Device::cpu());
  Tensor dK = zeros(qkv, DType::F32, Device::cpu());
  Tensor dV = zeros(qkv, DType::F32, Device::cpu());
  ops::flash_attention_backward(go, q, k, v, out_fa, lse,
                                dQ, dK, dV, true);
  const double eq = max_abs(ref_dQ.data(), dQ.as<float>(), Nout);
  const double ek = max_abs(ref_dK.data(), dK.as<float>(), Nout);
  const double ev = max_abs(ref_dV.data(), dV.as<float>(), Nout);
  std::printf("bwd grad_q max_abs = %.3e\n", eq);
  std::printf("bwd grad_k max_abs = %.3e\n", ek);
  std::printf("bwd grad_v max_abs = %.3e\n", ev);
  expect(eq < 1e-4, "flash bwd grad_q matches naive (causal)");
  expect(ek < 1e-4, "flash bwd grad_k matches naive (causal)");
  expect(ev < 1e-4, "flash bwd grad_v matches naive (causal)");

  // --------------------------------------------------------------------
  // Finite-difference gradcheck on a tiny problem. Uses central differences
  // against L = sum(O) so dL/dO = 1 everywhere. We then expect the numerical
  // grad to agree with flash_attention_backward's grad_q for a few random
  // probes.
  // --------------------------------------------------------------------
  {
    const int64_t b = 1, h = 1, s = 6, d = 4;
    const Shape qs{b, h, s, d};
    const float sc = 1.f / std::sqrt(float(d));
    Tensor fq = rand_f32(qs, 0xF1);
    Tensor fk = rand_f32(qs, 0xF2);
    Tensor fv = rand_f32(qs, 0xF3);
    Tensor fout = zeros(qs, DType::F32, Device::cpu());
    Tensor flse = zeros({b, h, s}, DType::F32, Device::cpu());
    ops::flash_attention(fq, fk, fv, fout, flse, true);

    // Analytic gradient of L = sum(O) wrt Q.
    Tensor go_ones = zeros(qs, DType::F32, Device::cpu());
    for (int64_t i = 0; i < go_ones.numel(); ++i) go_ones.as<float>()[i] = 1.f;
    Tensor agQ = zeros(qs, DType::F32, Device::cpu());
    Tensor agK = zeros(qs, DType::F32, Device::cpu());
    Tensor agV = zeros(qs, DType::F32, Device::cpu());
    ops::flash_attention_backward(go_ones, fq, fk, fv, fout, flse,
                                  agQ, agK, agV, true);

    auto scalar_loss = [&](const Tensor& qq, const Tensor& kk,
                           const Tensor& vv) -> double {
      Tensor oo = zeros(qs, DType::F32, Device::cpu());
      Tensor ll = zeros({b, h, s}, DType::F32, Device::cpu());
      ops::flash_attention(qq, kk, vv, oo, ll, true);
      double s2 = 0;
      for (int64_t i = 0; i < oo.numel(); ++i) s2 += oo.as<float>()[i];
      return s2;
    };
    (void)sc;

    const float eps = 1e-3f;
    std::mt19937_64 probe(42);
    std::uniform_int_distribution<int64_t> idx(0, fq.numel() - 1);
    int probed = 0, fd_ok = 0;
    for (int p = 0; p < 8; ++p) {
      const int64_t pi = idx(probe);
      const float orig = fq.as<float>()[pi];
      fq.as<float>()[pi] = orig + eps;
      double lp = scalar_loss(fq, fk, fv);
      fq.as<float>()[pi] = orig - eps;
      double lm = scalar_loss(fq, fk, fv);
      fq.as<float>()[pi] = orig;
      const double num = (lp - lm) / (2.0 * eps);
      const double ana = agQ.as<float>()[pi];
      const double err = std::fabs(num - ana);
      const double denom = std::max(1.0, std::fabs(num) + std::fabs(ana));
      ++probed;
      if (err / denom < 1e-2) ++fd_ok;
    }
    std::printf("fd gradcheck Q: %d/%d within 1%% rel\n", fd_ok, probed);
    expect(fd_ok == probed, "finite-diff gradcheck on Q agrees with analytic");
  }

  std::printf("---\n%d failed\n", g_failed);
  return g_failed;
}
