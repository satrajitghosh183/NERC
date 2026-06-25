#include "zwt/layers/linear.hpp"
#include "zwt/ops/gemm.hpp"
#include "zwt/ops/elementwise.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace zwt {

namespace {

// Kaiming-uniform init: U(-k, k), k = sqrt(1/fan_in). Written directly into
// device memory via a temp CPU buffer + H2D copy — this is init-time only,
// not on the hot path.
void kaiming_init(Tensor& w, int64_t fan_in, uint64_t seed) {
  const size_t n = static_cast<size_t>(w.numel());
  // Simple PCG-style LCG for reproducibility. One-time cost.
  uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
  auto next = [&]() -> float {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = static_cast<uint32_t>(s >> 33);
    return (x / float(1u << 31)) - 1.0f;  // [-1, 1)
  };
  float k = std::sqrt(1.0f / float(fan_in));

  // Stage on CPU f32, quantize if dtype is bf16.
  std::vector<float> cpu(n);
  for (size_t i = 0; i < n; ++i) cpu[i] = next() * k;

  if (w.dtype() == DType::F32) {
    Tensor host(cpu.data(), w.shape(), w.strides(), DType::F32,
                Device::cpu(), nullptr, cpu.size() * sizeof(float));
    copy(host, w);
  } else if (w.dtype() == DType::BF16) {
    std::vector<uint16_t> bf(n);
    for (size_t i = 0; i < n; ++i) {
      uint32_t u; std::memcpy(&u, &cpu[i], 4);
      uint32_t bias = 0x7FFF + ((u >> 16) & 1);
      bf[i] = static_cast<uint16_t>((u + bias) >> 16);
    }
    Tensor host(bf.data(), w.shape(), w.strides(), DType::BF16,
                Device::cpu(), nullptr, bf.size() * sizeof(uint16_t));
    copy(host, w);
  } else {
    throw std::runtime_error("Linear::kaiming_init: unsupported dtype");
  }
}

}  // namespace

Linear::Linear(int64_t in_features, int64_t out_features, bool use_bias,
               DType dtype, Device device, uint64_t init_seed)
    : has_bias_(use_bias),
      in_features_(in_features),
      out_features_(out_features) {
  weight_.name  = "weight";
  weight_.value = empty({out_features, in_features}, dtype, device);
  kaiming_init(weight_.value, in_features, init_seed);

  if (use_bias) {
    bias_.name  = "bias";
    bias_.value = zeros({out_features}, dtype, device);
  }
}

Tensor Linear::forward(const Tensor& x) {
  // Flatten x to 2D [N, in]. Output: [N, out].
  int64_t N = x.numel() / in_features_;
  saved_input_shape_ = x.shape();
  saved_input_ = x.view({N, in_features_});

  Shape out_shape = x.shape();
  out_shape.dims[x.rank() - 1] = out_features_;
  Tensor y = empty_scratch(out_shape, x.dtype(), x.device());

  // y2d = x2d @ W^T : [N, in] @ [in, out] = [N, out]
  // W is stored [out, in], so set transb=true.
  Tensor y2d = y.view({N, out_features_});
  ops::gemm(saved_input_, /*transa=*/false, weight_.value, /*transb=*/true,
            y2d, 1.0f, 0.0f);
  if (has_bias_) ops::add_bias(y2d, bias_.value);

  return y;
}

Tensor Linear::backward(const Tensor& grad_y) {
  int64_t N = grad_y.numel() / out_features_;
  Tensor grad_y_2d = grad_y.view({N, out_features_});

  weight_.ensure_grad();
  if (has_bias_) bias_.ensure_grad();

  // grad_W += grad_y2d^T @ x2d        : [out, in]
  // grad_x  = grad_y2d  @ W           : [N, in]
  ops::gemm(grad_y_2d, /*transa=*/true, saved_input_, /*transb=*/false,
            weight_.grad, 1.0f, 1.0f);

  Tensor grad_x = empty_scratch(saved_input_shape_, grad_y.dtype(), grad_y.device());
  Tensor grad_x_2d = grad_x.view({N, in_features_});
  ops::gemm(grad_y_2d, /*transa=*/false, weight_.value, /*transb=*/false,
            grad_x_2d, 1.0f, 0.0f);

  if (has_bias_) {
    ops::bias_backward(grad_y_2d, bias_.grad);
  }

  return grad_x;
}

void Linear::collect_params(std::vector<Parameter*>& out) {
  out.push_back(&weight_);
  if (has_bias_) out.push_back(&bias_);
}

}  // namespace zwt
