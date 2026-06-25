// Kernel correctness suite — exercises each zwt op against an inline analytic
// reference. CPU-runnable (fp32 path). When built with CUDA, the same driver
// runs and surfaces any CPU regression; a parallel CUDA-vs-CPU cross-check
// belongs in a separate binary and is not yet wired.
//
// Exit code 0 = all PASS. Non-zero = failure count.
//
// Per-test output:
//   [PASS] <name> max_abs=<x> (n=<k>)
//   [FAIL] <name> max_abs=<x> exceeds tol=<t>

#include "zwt/core/allocator.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/ops/elementwise.hpp"
#include "zwt/ops/gemm.hpp"
#include "zwt/ops/norm.hpp"
#include "zwt/ops/rope.hpp"
#include "zwt/ops/xent.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace zwt;

namespace {

int g_failed = 0;
int g_passed = 0;

struct Check {
  double max_abs = 0.0;
  size_t n       = 0;
};

Check cmp(const float* a, const float* b, size_t n) {
  Check c;
  c.n = n;
  for (size_t i = 0; i < n; ++i) {
    double d = std::fabs(double(a[i]) - double(b[i]));
    if (d > c.max_abs) c.max_abs = d;
  }
  return c;
}

void report(const std::string& name, Check c, double tol) {
  const char* tag = (c.max_abs <= tol) ? "PASS" : "FAIL";
  if (c.max_abs > tol) ++g_failed; else ++g_passed;
  std::printf("[%s] %s max_abs=%.3e tol=%.3e n=%zu\n",
              tag, name.c_str(), c.max_abs, tol, c.n);
}

// Build a CPU-resident fp32 tensor of the given shape, filled from `data`.
Tensor from_host(const Shape& sh, const std::vector<float>& data) {
  Tensor t = empty(sh, DType::F32, Device::cpu());
  std::memcpy(t.data(), data.data(), data.size() * sizeof(float));
  return t;
}

Tensor zeros_cpu(const Shape& sh) {
  Tensor t = empty(sh, DType::F32, Device::cpu());
  std::memset(t.data(), 0, static_cast<size_t>(t.numel()) * sizeof(float));
  return t;
}

std::vector<float> rand_vec(size_t n, uint64_t seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// =============================================================
// Test: rmsnorm forward
// =============================================================
void test_rmsnorm_fwd() {
  const int64_t N = 4, D = 32;
  const float eps = 1e-6f;
  auto x_host = rand_vec(N * D, 0xA1);
  auto w_host = rand_vec(D, 0xA2, 0.5f, 1.5f);

  Tensor x = from_host({N, D}, x_host);
  Tensor w = from_host({D}, w_host);
  Tensor y = zeros_cpu({N, D});
  Tensor rstd = zeros_cpu({N});
  ops::rmsnorm(x, w, y, rstd, eps);

  std::vector<float> ref(N * D);
  for (int64_t r = 0; r < N; ++r) {
    double ss = 0;
    for (int64_t c = 0; c < D; ++c) ss += double(x_host[r*D+c]) * x_host[r*D+c];
    float rs = float(1.0 / std::sqrt(ss / D + eps));
    for (int64_t c = 0; c < D; ++c) ref[r*D+c] = x_host[r*D+c] * rs * w_host[c];
  }
  report("rmsnorm_fwd", cmp(y.as<float>(), ref.data(), N*D), 1e-5);
}

// =============================================================
// Test: rmsnorm backward vs numerical gradient
// =============================================================
void test_rmsnorm_bwd_numerical() {
  const int64_t N = 2, D = 8;
  const float eps = 1e-6f;
  auto x_host = rand_vec(N * D, 0xB1);
  auto w_host = rand_vec(D, 0xB2, 0.5f, 1.5f);
  auto go_host = rand_vec(N * D, 0xB3);

  Tensor x = from_host({N, D}, x_host);
  Tensor w = from_host({D}, w_host);
  Tensor y = zeros_cpu({N, D});
  Tensor rstd = zeros_cpu({N});
  ops::rmsnorm(x, w, y, rstd, eps);

  Tensor grad_y = from_host({N, D}, go_host);
  Tensor grad_x = zeros_cpu({N, D});
  Tensor grad_w = zeros_cpu({D});
  ops::rmsnorm_backward(grad_y, x, w, rstd, grad_x, grad_w, eps);

  // Numerical dL/dx_i via finite differences where L = sum(grad_y * y).
  const float h = 1e-3f;
  std::vector<float> num_gx(N * D);
  for (int64_t i = 0; i < N * D; ++i) {
    auto xp = x_host; xp[i] += h;
    auto xm = x_host; xm[i] -= h;
    Tensor xpt = from_host({N, D}, xp);
    Tensor xmt = from_host({N, D}, xm);
    Tensor yp = zeros_cpu({N, D});
    Tensor ym = zeros_cpu({N, D});
    Tensor rp = zeros_cpu({N});
    Tensor rm = zeros_cpu({N});
    ops::rmsnorm(xpt, w, yp, rp, eps);
    ops::rmsnorm(xmt, w, ym, rm, eps);
    double Lp = 0, Lm = 0;
    for (int64_t j = 0; j < N*D; ++j) {
      Lp += double(go_host[j]) * yp.as<float>()[j];
      Lm += double(go_host[j]) * ym.as<float>()[j];
    }
    num_gx[i] = float((Lp - Lm) / (2.0 * h));
  }
  report("rmsnorm_bwd_x",
         cmp(grad_x.as<float>(), num_gx.data(), N*D), 2e-3);
}

// =============================================================
// Test: silu_mul forward + backward consistency
// =============================================================
void test_silu_mul() {
  const int64_t N = 64;
  auto g_host = rand_vec(N, 0xC1);
  auto u_host = rand_vec(N, 0xC2);
  Tensor gate = from_host({N}, g_host);
  Tensor up   = from_host({N}, u_host);
  Tensor out  = zeros_cpu({N});
  ops::silu_mul(out, gate, up);
  std::vector<float> ref(N);
  for (int64_t i = 0; i < N; ++i) {
    float g = g_host[i];
    float s = g / (1.f + std::exp(-g));
    ref[i] = s * u_host[i];
  }
  report("silu_mul_fwd", cmp(out.as<float>(), ref.data(), N), 1e-5);

  auto go_host = rand_vec(N, 0xC3);
  Tensor go = from_host({N}, go_host);
  Tensor gg = zeros_cpu({N});
  Tensor gu = zeros_cpu({N});
  ops::silu_mul_backward(go, gate, up, gg, gu);

  // Numerical check on grad_gate only.
  const float h = 1e-3f;
  std::vector<float> num_gg(N);
  for (int64_t i = 0; i < N; ++i) {
    float g = g_host[i];
    auto plus  = g_host; plus[i]  = g + h;
    auto minus = g_host; minus[i] = g - h;
    Tensor gp = from_host({N}, plus);
    Tensor gm = from_host({N}, minus);
    Tensor op = zeros_cpu({N});
    Tensor om = zeros_cpu({N});
    ops::silu_mul(op, gp, up);
    ops::silu_mul(om, gm, up);
    double Lp = 0, Lm = 0;
    for (int64_t j = 0; j < N; ++j) {
      Lp += double(go_host[j]) * op.as<float>()[j];
      Lm += double(go_host[j]) * om.as<float>()[j];
    }
    num_gg[i] = float((Lp - Lm) / (2.0 * h));
  }
  report("silu_mul_bwd_gate", cmp(gg.as<float>(), num_gg.data(), N), 1e-3);
}

// =============================================================
// Test: silu_mul_gated equivalence with separate silu_mul
// =============================================================
void test_silu_mul_gated() {
  const int64_t N = 16, H = 32;
  auto combined = rand_vec(N * 2 * H, 0xD1);
  std::vector<float> gate(N * H), up(N * H);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t i = 0; i < H; ++i) {
      gate[n*H + i] = combined[n*2*H + i];
      up[n*H + i]   = combined[n*2*H + H + i];
    }
  }
  Tensor c  = from_host({N, 2*H}, combined);
  Tensor y1 = zeros_cpu({N, H});
  ops::silu_mul_gated(y1, c);

  Tensor gt = from_host({N, H}, gate);
  Tensor ut = from_host({N, H}, up);
  Tensor y2 = zeros_cpu({N, H});
  ops::silu_mul(y2, gt, ut);

  report("silu_mul_gated_vs_split",
         cmp(y1.as<float>(), y2.as<float>(), N*H), 1e-6);

  // Backward path: grad_combined halves must equal grad_gate/grad_up.
  auto go_host = rand_vec(N * H, 0xD2);
  Tensor go = from_host({N, H}, go_host);
  Tensor gc = zeros_cpu({N, 2*H});
  ops::silu_mul_gated_backward(go, c, gc);

  Tensor gg = zeros_cpu({N, H});
  Tensor gu = zeros_cpu({N, H});
  ops::silu_mul_backward(go, gt, ut, gg, gu);

  std::vector<float> repack(N * 2 * H);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t i = 0; i < H; ++i) {
      repack[n*2*H + i]     = gg.as<float>()[n*H + i];
      repack[n*2*H + H + i] = gu.as<float>()[n*H + i];
    }
  }
  report("silu_mul_gated_bwd_vs_split",
         cmp(gc.as<float>(), repack.data(), N*2*H), 1e-6);
}

// =============================================================
// Test: gemm against naive reference
// =============================================================
void test_gemm() {
  const int M = 7, K = 5, N = 3;
  auto A = rand_vec(M * K, 0xE1);
  auto B = rand_vec(K * N, 0xE2);
  Tensor at = from_host({M, K}, A);
  Tensor bt = from_host({K, N}, B);
  Tensor ct = zeros_cpu({M, N});
  ops::gemm(at, false, bt, false, ct, 1.f, 0.f);

  std::vector<float> ref(M * N, 0.f);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += double(A[i*K+k]) * B[k*N+j];
      ref[i*N+j] = float(acc);
    }
  report("gemm_f32", cmp(ct.as<float>(), ref.data(), M*N), 1e-5);
}

// =============================================================
// Test: transpose round-trip
// =============================================================
void test_transpose_roundtrip() {
  const int64_t B = 2, S = 3, H = 4, D = 5;
  auto src = rand_vec(B * S * H * D, 0xF1);
  Tensor a = from_host({B, S, H, D}, src);
  Tensor b = zeros_cpu({B, H, S, D});
  Tensor c = zeros_cpu({B, S, H, D});
  ops::transpose_bshd_to_bhsd(a, b);
  ops::transpose_bhsd_to_bshd(b, c);
  report("transpose_bshd_roundtrip",
         cmp(a.as<float>(), c.as<float>(), B*S*H*D), 0.0);
}

// =============================================================
// Test: repeat_kv_heads / reduce_kv_heads_sum relationship
// =============================================================
void test_repeat_reduce_kv() {
  const int64_t Bd = 2, Hkv = 2, S = 3, D = 4, group = 3;
  const int64_t H = Hkv * group;
  auto in = rand_vec(Bd * Hkv * S * D, 0xF2);
  Tensor a = from_host({Bd, Hkv, S, D}, in);
  Tensor rep = zeros_cpu({Bd, H, S, D});
  ops::repeat_kv_heads(a, rep);
  Tensor red = zeros_cpu({Bd, Hkv, S, D});
  ops::reduce_kv_heads_sum(rep, red);
  // reduce after repeat => each KV head is summed `group` times.
  std::vector<float> ref(in.size());
  for (size_t i = 0; i < in.size(); ++i) ref[i] = in[i] * float(group);
  report("repeat_then_reduce_kv",
         cmp(red.as<float>(), ref.data(), in.size()), 1e-6);
}

// =============================================================
// Test: rope forward then backward must be identity
// =============================================================
void test_rope_roundtrip() {
  const int64_t B = 2, S = 5, H = 2, D = 8;
  Tensor table = ops::rope_build_table(S, D, 10000.f, Device::cpu());
  auto src = rand_vec(B * S * H * D, 0x1011);
  Tensor x0 = from_host({B, S, H, D}, src);
  Tensor x  = from_host({B, S, H, D}, src);
  ops::rope_apply(x, table);
  ops::rope_apply_backward(x, table);
  report("rope_fwd_then_bwd_identity",
         cmp(x.as<float>(), x0.as<float>(), B*S*H*D), 1e-5);
}

// =============================================================
// Test: cross_entropy matches naive softmax-NLL
// =============================================================
void test_cross_entropy() {
  const int64_t Nn = 4, V = 10;
  auto logits = rand_vec(Nn * V, 0x2021, -2.f, 2.f);
  std::vector<int64_t> targets = {0, 3, 7, 9};

  Tensor lt  = from_host({Nn, V}, logits);
  Tensor tt  = empty({Nn}, DType::I64, Device::cpu());
  std::memcpy(tt.data(), targets.data(), targets.size() * sizeof(int64_t));
  Tensor loss = zeros_cpu({1});
  ops::cross_entropy(lt, tt, loss, nullptr, -100);

  double ref = 0;
  for (int64_t n = 0; n < Nn; ++n) {
    double m = -1e30;
    for (int64_t v = 0; v < V; ++v) if (logits[n*V+v] > m) m = logits[n*V+v];
    double se = 0;
    for (int64_t v = 0; v < V; ++v) se += std::exp(logits[n*V+v] - m);
    double lse = m + std::log(se);
    ref += lse - logits[n*V + targets[n]];
  }
  ref /= double(Nn);
  float diff = std::fabs(loss.as<float>()[0] - float(ref));
  Check c; c.max_abs = diff; c.n = 1;
  report("cross_entropy_scalar", c, 1e-4);
}

}  // namespace

int main() {
  set_activation_arena_capacity(size_t(64) << 20);

  test_rmsnorm_fwd();
  test_rmsnorm_bwd_numerical();
  test_silu_mul();
  test_silu_mul_gated();
  test_gemm();
  test_transpose_roundtrip();
  test_repeat_reduce_kv();
  test_rope_roundtrip();
  test_cross_entropy();

  std::printf("---\n%d passed, %d failed\n", g_passed, g_failed);
  return g_failed;
}
