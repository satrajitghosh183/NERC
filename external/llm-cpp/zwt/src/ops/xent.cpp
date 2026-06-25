#include "zwt/ops/xent.hpp"
#include "zwt/core/stream.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA
#include "zwt/src/ops/kernels.hpp"
#endif

namespace zwt::ops {

namespace {

void xent_fwd_f32_cpu(const float* logits, const int64_t* targets,
                      float* loss_out, float* grad_logits,
                      int64_t rows, int64_t vocab, int64_t ignore) {
  // Softmax + NLL, optionally emitting gradient = (softmax - one_hot(target)) / N_valid.
  float sum_loss = 0.f;
  int64_t n_valid = 0;
  for (int64_t r = 0; r < rows; ++r) {
    const float* lr = logits + r * vocab;
    int64_t t = targets[r];
    if (t == ignore) {
      if (grad_logits) {
        for (int64_t c = 0; c < vocab; ++c) grad_logits[r * vocab + c] = 0.f;
      }
      continue;
    }
    // max for stability
    float m = lr[0];
    for (int64_t c = 1; c < vocab; ++c) if (lr[c] > m) m = lr[c];
    float ss = 0.f;
    for (int64_t c = 0; c < vocab; ++c) ss += std::exp(lr[c] - m);
    float logZ = m + std::log(ss);
    float l = logZ - lr[t];
    sum_loss += l;
    ++n_valid;
    if (grad_logits) {
      float* gr = grad_logits + r * vocab;
      for (int64_t c = 0; c < vocab; ++c) gr[c] = std::exp(lr[c] - logZ);
      gr[t] -= 1.0f;
    }
  }
  *loss_out = (n_valid > 0) ? (sum_loss / n_valid) : 0.f;
  if (grad_logits && n_valid > 0) {
    float inv = 1.0f / n_valid;
    for (int64_t i = 0; i < rows * vocab; ++i) grad_logits[i] *= inv;
  }
}

}  // namespace

void cross_entropy(const Tensor& logits, const Tensor& targets, Tensor& loss,
                   Tensor* grad_logits_out, int64_t ignore_index) {
  if (logits.rank() != 2) throw std::runtime_error("cross_entropy: expected 2D logits");
  int64_t rows = logits.dim(0);
  int64_t vocab = logits.dim(1);
  if (targets.numel() != rows)
    throw std::runtime_error("cross_entropy: targets/logits row mismatch");

  if (logits.device().is_cuda()) {
#ifdef USE_CUDA
    __nv_bfloat16* gptr = grad_logits_out
        ? reinterpret_cast<__nv_bfloat16*>(grad_logits_out->data()) : nullptr;
    k::softmax_xent_fused_bf16(
        reinterpret_cast<const __nv_bfloat16*>(logits.data()),
        targets.as<int64_t>(),
        loss.as<float>(),
        gptr,
        rows, vocab, ignore_index,
        reinterpret_cast<cudaStream_t>(compute_stream(logits.device()).handle));
    return;
#endif
  }
  if (logits.dtype() == DType::F32) {
    xent_fwd_f32_cpu(logits.as<float>(), targets.as<int64_t>(),
                     loss.as<float>(),
                     grad_logits_out ? grad_logits_out->as<float>() : nullptr,
                     rows, vocab, ignore_index);
    return;
  }
  throw std::runtime_error("cross_entropy: unsupported dtype on CPU");
}

}  // namespace zwt::ops
