#pragma once

/**
 * include/olmo_cpp/backend/rms_norm_backward.hpp
 *
 * Fused RMSNorm backward (item B3).
 *
 * Returns (grad_x, grad_w) for the forward
 *     y = x * rsqrt(mean(x^2) + eps) * w
 * given grad_y. grad_w is undefined-tensor when `weight` is empty.
 *
 * CUDA-only — CPU autograd path goes through ATen ops in
 * cuda_ops_autograd.cpp::RmsNormFn::backward.
 */

#include <torch/torch.h>
#include <utility>

namespace olmo_cpp {

#ifdef OLMO_HAS_CUDA_KERNELS
std::pair<torch::Tensor, torch::Tensor>
rms_norm_backward_cuda(torch::Tensor grad_y,
                         torch::Tensor x,
                         c10::optional<torch::Tensor> weight,
                         double eps);
#endif

}  // namespace olmo_cpp
