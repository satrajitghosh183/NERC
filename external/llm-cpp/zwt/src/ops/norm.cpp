#include "zwt/ops/norm.hpp"
#include "zwt/core/stream.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef USE_CUDA
#include "zwt/src/ops/kernels.hpp"
#endif

namespace zwt::ops {

namespace {

// Shape expected: x = [..., D]. Flatten leading dims into "rows".
std::pair<int64_t, int64_t> row_cols(const Tensor& x) {
  const int r = x.rank();
  const int64_t D = x.dim(r - 1);
  int64_t N = 1;
  for (int i = 0; i < r - 1; ++i) N *= x.dim(i);
  return {N, D};
}

void rmsnorm_fwd_f32_cpu(const float* x, const float* w, float* y, float* rstd,
                         int64_t rows, int64_t cols, float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float* yr = y + r * cols;
    float ss = 0.f;
    for (int64_t c = 0; c < cols; ++c) ss += xr[c] * xr[c];
    float rs = 1.0f / std::sqrt(ss / cols + eps);
    rstd[r] = rs;
    for (int64_t c = 0; c < cols; ++c) yr[c] = xr[c] * rs * w[c];
  }
}

void rmsnorm_residual_fwd_f32_cpu(const float* x, const float* res, const float* w,
                                  float* y, float* sum_out, float* rstd,
                                  int64_t rows, int64_t cols, float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    const float* rr = res + r * cols;
    float* sr = sum_out + r * cols;
    float* yr = y + r * cols;
    float ss = 0.f;
    for (int64_t c = 0; c < cols; ++c) {
      float s = xr[c] + rr[c];
      sr[c] = s;
      ss += s * s;
    }
    float rs = 1.0f / std::sqrt(ss / cols + eps);
    rstd[r] = rs;
    for (int64_t c = 0; c < cols; ++c) yr[c] = sr[c] * rs * w[c];
  }
}

// Backward: for y = (x * rstd) * w  and rstd = 1/sqrt(mean(x^2)+eps)
//
//   dL/dx_i = w_i * rstd * dL/dy_i
//             - (rstd^3 / D) * x_i * sum_j (x_j * w_j * dL/dy_j)
//   dL/dw_i = sum_rows (x_i * rstd * dL/dy_i)
void rmsnorm_bwd_f32_cpu(const float* grad_y, const float* x, const float* w,
                         const float* rstd, float* grad_x, float* grad_w,
                         int64_t rows, int64_t cols, float /*eps*/) {
  // Zero grad_w (caller owns accumulation semantics; we accumulate, so do NOT zero).
  for (int64_t r = 0; r < rows; ++r) {
    const float* gr = grad_y + r * cols;
    const float* xr = x + r * cols;
    float* gxr = grad_x + r * cols;
    float rs = rstd[r];
    // dot = sum_j x_j * w_j * gy_j
    float dot = 0.f;
    for (int64_t c = 0; c < cols; ++c) dot += xr[c] * w[c] * gr[c];
    float coef = (rs * rs * rs) / cols;
    for (int64_t c = 0; c < cols; ++c) {
      gxr[c] = w[c] * rs * gr[c] - coef * xr[c] * dot;
      grad_w[c] += xr[c] * rs * gr[c];
    }
  }
}

}  // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& y, Tensor& rstd,
             float eps) {
  auto [rows, cols] = row_cols(x);
  if (x.device().is_cuda()) {
#ifdef USE_CUDA
    k::rmsnorm_fwd_bf16(reinterpret_cast<const __nv_bfloat16*>(x.data()),
                        reinterpret_cast<const __nv_bfloat16*>(weight.data()),
                        reinterpret_cast<__nv_bfloat16*>(y.data()),
                        rstd.as<float>(),
                        rows, cols, eps,
                        reinterpret_cast<cudaStream_t>(compute_stream(x.device()).handle));
    return;
#endif
  }
  if (x.dtype() == DType::F32) {
    rmsnorm_fwd_f32_cpu(x.as<float>(), weight.as<float>(), y.as<float>(),
                        rstd.as<float>(), rows, cols, eps);
    return;
  }
  throw std::runtime_error("rmsnorm: unsupported dtype on CPU");
}

void rmsnorm_residual(const Tensor& x, const Tensor& residual,
                      const Tensor& weight, Tensor& y, Tensor& sum_out,
                      Tensor& rstd, float eps) {
  auto [rows, cols] = row_cols(x);
  if (x.device().is_cuda()) {
#ifdef USE_CUDA
    k::rmsnorm_residual_fwd_bf16(
        reinterpret_cast<const __nv_bfloat16*>(x.data()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data()),
        reinterpret_cast<__nv_bfloat16*>(y.data()),
        reinterpret_cast<__nv_bfloat16*>(sum_out.data()),
        rstd.as<float>(),
        rows, cols, eps,
        reinterpret_cast<cudaStream_t>(compute_stream(x.device()).handle));
    return;
#endif
  }
  if (x.dtype() == DType::F32) {
    rmsnorm_residual_fwd_f32_cpu(x.as<float>(), residual.as<float>(),
                                 weight.as<float>(), y.as<float>(),
                                 sum_out.as<float>(), rstd.as<float>(),
                                 rows, cols, eps);
    return;
  }
  throw std::runtime_error("rmsnorm_residual: unsupported dtype on CPU");
}

void rmsnorm_backward(const Tensor& grad_y, const Tensor& x, const Tensor& weight,
                      const Tensor& rstd, Tensor& grad_x, Tensor& grad_weight,
                      float eps) {
  auto [rows, cols] = row_cols(x);
  if (x.device().is_cuda()) {
#ifdef USE_CUDA
    k::rmsnorm_bwd_bf16(reinterpret_cast<const __nv_bfloat16*>(grad_y.data()),
                        reinterpret_cast<const __nv_bfloat16*>(x.data()),
                        reinterpret_cast<const __nv_bfloat16*>(weight.data()),
                        rstd.as<float>(),
                        reinterpret_cast<__nv_bfloat16*>(grad_x.data()),
                        grad_weight.as<float>(),
                        rows, cols, eps,
                        reinterpret_cast<cudaStream_t>(compute_stream(x.device()).handle));
    return;
#endif
  }
  if (x.dtype() == DType::F32) {
    rmsnorm_bwd_f32_cpu(grad_y.as<float>(), x.as<float>(), weight.as<float>(),
                        rstd.as<float>(), grad_x.as<float>(), grad_weight.as<float>(),
                        rows, cols, eps);
    return;
  }
  throw std::runtime_error("rmsnorm_backward: unsupported dtype on CPU");
}

}  // namespace zwt::ops
