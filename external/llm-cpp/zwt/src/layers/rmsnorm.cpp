#include "zwt/layers/rmsnorm.hpp"
#include "zwt/ops/norm.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace zwt {

namespace {

// Initialize weight to ones (standard RMSNorm init).
void init_ones(Tensor& w) {
  std::vector<uint16_t> bf(w.numel());
  std::vector<float>    f32(w.numel());
  if (w.dtype() == DType::BF16) {
    // 1.0f in bf16 = 0x3F80
    for (auto& x : bf) x = 0x3F80;
    Tensor host(bf.data(), w.shape(), w.strides(), DType::BF16, Device::cpu(),
                nullptr, bf.size() * sizeof(uint16_t));
    copy(host, w);
  } else {
    for (auto& x : f32) x = 1.0f;
    Tensor host(f32.data(), w.shape(), w.strides(), DType::F32, Device::cpu(),
                nullptr, f32.size() * sizeof(float));
    copy(host, w);
  }
}

}  // namespace

RMSNorm::RMSNorm(int64_t dim, float eps, DType dtype, Device device)
    : eps_(eps), dim_(dim) {
  weight_.name  = "weight";
  weight_.value = empty({dim}, dtype, device);
  init_ones(weight_.value);
}

Tensor RMSNorm::forward(const Tensor& x) {
  saved_input_ = x.view(x.shape());  // non-owning

  // rstd is [N]. We allocate fp32 always — half the memory traffic isn't
  // worth the numerical risk on the reduction.
  int64_t N = x.numel() / dim_;
  saved_rstd_ = empty_scratch({N}, DType::F32, x.device());

  Tensor y = empty_scratch(x.shape(), x.dtype(), x.device());
  ops::rmsnorm(x, weight_.value, y, saved_rstd_, eps_);
  return y;
}

Tensor RMSNorm::forward_residual(const Tensor& x, const Tensor& residual,
                                 Tensor& sum_out) {
  int64_t N = x.numel() / dim_;
  saved_rstd_ = empty_scratch({N}, DType::F32, x.device());
  saved_input_ = sum_out.view(sum_out.shape());  // backward operates on the sum
  Tensor y = empty_scratch(x.shape(), x.dtype(), x.device());
  ops::rmsnorm_residual(x, residual, weight_.value, y, sum_out, saved_rstd_, eps_);
  return y;
}

Tensor RMSNorm::backward(const Tensor& grad_y) {
  weight_.ensure_grad();
  Tensor grad_x = empty_scratch(grad_y.shape(), grad_y.dtype(), grad_y.device());
  ops::rmsnorm_backward(grad_y, saved_input_, weight_.value, saved_rstd_,
                        grad_x, weight_.grad, eps_);
  return grad_x;
}

void RMSNorm::collect_params(std::vector<Parameter*>& out) {
  out.push_back(&weight_);
}

}  // namespace zwt
