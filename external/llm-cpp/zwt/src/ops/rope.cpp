#include "zwt/ops/rope.hpp"
#include "zwt/core/stream.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#endif

namespace zwt::ops::k {
#ifdef USE_CUDA
void rope_apply_bf16(__nv_bfloat16* x, const float* table,
                     int64_t B, int64_t S, int64_t H, int64_t D,
                     bool inverse, cudaStream_t s);
#endif
}

namespace zwt::ops {

Tensor rope_build_table(int64_t max_seq, int64_t head_dim, float base,
                        Device device) {
  if ((head_dim & 1) != 0) throw std::runtime_error("rope_build_table: head_dim must be even");
  const int64_t half = head_dim / 2;

  std::vector<float> tab(static_cast<size_t>(max_seq * head_dim));
  for (int64_t s = 0; s < max_seq; ++s) {
    for (int64_t i = 0; i < half; ++i) {
      float theta_i = std::pow(base, -float(2 * i) / float(head_dim));
      float angle = float(s) * theta_i;
      tab[s * head_dim + i]        = std::cos(angle);
      tab[s * head_dim + i + half] = std::sin(angle);
    }
  }

  Shape sh{max_seq, head_dim};
  Tensor dev = empty(sh, DType::F32, device);
  Tensor host(tab.data(), sh, contiguous_strides(sh), DType::F32,
              Device::cpu(), nullptr, tab.size() * sizeof(float));
  copy(host, dev);
  return dev;
}

namespace {

void rope_cpu_f32(float* x, const float* tab,
                  int64_t B, int64_t S, int64_t H, int64_t D,
                  bool inverse) {
  const int64_t half = D / 2;
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t i = 0; i < half; ++i) {
          float c = tab[s * D + i];
          float sn = tab[s * D + i + half];
          if (inverse) sn = -sn;
          int64_t base = ((b * S + s) * H + h) * D;
          float lo = x[base + i];
          float hi = x[base + i + half];
          x[base + i]        = lo * c - hi * sn;
          x[base + i + half] = hi * c + lo * sn;
        }
      }
    }
  }
}

void rope_dispatch(Tensor& x, const Tensor& table, bool inverse) {
  if (x.rank() != 4) throw std::runtime_error("rope: x must be [B,S,H,D]");
  if (table.rank() != 2) throw std::runtime_error("rope: table must be [max_seq, D]");
  const int64_t B = x.dim(0);
  const int64_t S = x.dim(1);
  const int64_t H = x.dim(2);
  const int64_t D = x.dim(3);
  if (table.dim(1) != D) throw std::runtime_error("rope: table D mismatch");
  if (table.dim(0) < S) throw std::runtime_error("rope: table too short");

  if (x.device().is_cuda()) {
#ifdef USE_CUDA
    k::rope_apply_bf16(reinterpret_cast<__nv_bfloat16*>(x.data()),
                       table.as<float>(),
                       B, S, H, D, inverse,
                       reinterpret_cast<cudaStream_t>(
                           compute_stream(x.device()).handle));
    return;
#endif
  }
  if (x.dtype() == DType::F32) {
    rope_cpu_f32(x.as<float>(), table.as<float>(), B, S, H, D, inverse);
    return;
  }
  throw std::runtime_error("rope: unsupported dtype on CPU");
}

}  // namespace

void rope_apply(Tensor& x, const Tensor& table) {
  rope_dispatch(x, table, /*inverse=*/false);
}

void rope_apply_backward(Tensor& grad_x, const Tensor& table) {
  rope_dispatch(grad_x, table, /*inverse=*/true);
}

}  // namespace zwt::ops
