#include "zwt/optim/grad_clip.hpp"
#include "zwt/core/stream.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef USE_CUDA
#include "zwt/core/cuda_check.hpp"
#include <cuda_runtime.h>
#endif

namespace zwt::optim {

#ifdef USE_CUDA
namespace k {
void zero_scalar_fp32(float* p, cudaStream_t s);
void sumsq_fp32_many(float** ptrs, const int64_t* sizes, int n_tensors,
                     float* out, cudaStream_t s);
void scale_fp32_many(float** ptrs, const int64_t* sizes, int n_tensors,
                     float alpha, cudaStream_t s);
void scale_fp32_many_dev(float** ptrs, const int64_t* sizes, int n_tensors,
                         const float* alpha_dev, cudaStream_t s);
int  chunk_size_fp32();
void sumsq_fp32_mta(float** ptrs, const int64_t* sizes,
                    const int* chunk_to_tensor,
                    const int64_t* chunk_to_offset,
                    int n_chunks, float* out, cudaStream_t s);
void scale_fp32_mta(float** ptrs, const int64_t* sizes,
                    const int* chunk_to_tensor,
                    const int64_t* chunk_to_offset,
                    int n_chunks, float alpha, cudaStream_t s);
void scale_fp32_mta_dev(float** ptrs, const int64_t* sizes,
                        const int* chunk_to_tensor,
                        const int64_t* chunk_to_offset,
                        int n_chunks, const float* alpha_dev,
                        cudaStream_t s);
void compute_clip_scale(const float* sumsq_dev, float max_norm,
                        float* scale_dev, float* norm_out_dev,
                        cudaStream_t s);
}  // namespace k
#endif

// ---------------------------------------------------------------------------
// GradClipper
// ---------------------------------------------------------------------------

GradClipper::GradClipper(const std::vector<Parameter*>& params) {
  if (params.empty()) return;
  device_ = params.front()->value.device();
  n_ = static_cast<int>(params.size());

  if (!device_.is_cuda()) {
    cpu_params_ = params;
    return;
  }

#ifdef USE_CUDA
  std::vector<float*>  p_h(n_);
  std::vector<int64_t> s_h(n_);
  for (int i = 0; i < n_; ++i) {
    p_h[i] = params[i]->grad.as<float>();
    s_h[i] = params[i]->value.numel();
  }

  // One-time setup. The pool allocator never moves param storage, so these
  // pointers are stable for the trainer's lifetime.
  ZWT_CUDA(cudaMalloc(&d_ptrs_,  sizeof(float*)  * n_));
  ZWT_CUDA(cudaMalloc(&d_sizes_, sizeof(int64_t) * n_));
  ZWT_CUDA(cudaMalloc(&d_sumsq_, sizeof(float)));
  ZWT_CUDA(cudaMalloc(&d_scale_, sizeof(float)));
  ZWT_CUDA(cudaMalloc(&d_norm_,  sizeof(float)));
  ZWT_CUDA(cudaMemcpy(d_ptrs_,  p_h.data(), sizeof(float*)  * n_,
                      cudaMemcpyHostToDevice));
  ZWT_CUDA(cudaMemcpy(d_sizes_, s_h.data(), sizeof(int64_t) * n_,
                      cudaMemcpyHostToDevice));

  // Build MTA chunk descriptors. For each tensor of size s, we emit
  // ceil(s / chunk_size) chunks. Each chunk gets a (tensor_id, offset)
  // entry so the kernel's grid is sized to total chunks rather than
  // total tensors — uniform SM utilization regardless of param size mix.
  const int chunk = k::chunk_size_fp32();
  std::vector<int>     chunk_to_tensor;
  std::vector<int64_t> chunk_to_offset;
  for (int t = 0; t < n_; ++t) {
    const int64_t sz = s_h[t];
    int64_t off = 0;
    while (off < sz) {
      chunk_to_tensor.push_back(t);
      chunk_to_offset.push_back(off);
      off += chunk;
    }
  }
  n_chunks_ = static_cast<int>(chunk_to_tensor.size());
  if (n_chunks_ > 0) {
    ZWT_CUDA(cudaMalloc(&d_chunk_to_tensor_, sizeof(int)     * n_chunks_));
    ZWT_CUDA(cudaMalloc(&d_chunk_to_offset_, sizeof(int64_t) * n_chunks_));
    ZWT_CUDA(cudaMemcpy(d_chunk_to_tensor_, chunk_to_tensor.data(),
                        sizeof(int)     * n_chunks_, cudaMemcpyHostToDevice));
    ZWT_CUDA(cudaMemcpy(d_chunk_to_offset_, chunk_to_offset.data(),
                        sizeof(int64_t) * n_chunks_, cudaMemcpyHostToDevice));
  }

  // Initialize scale=1, norm=0 so that an early pull_last_norm before the
  // first clip() returns sensible defaults.
  float one = 1.f;
  float zero = 0.f;
  ZWT_CUDA(cudaMemcpy(d_scale_, &one,  sizeof(float), cudaMemcpyHostToDevice));
  ZWT_CUDA(cudaMemcpy(d_norm_,  &zero, sizeof(float), cudaMemcpyHostToDevice));
#endif
}

GradClipper::GradClipper(GradClipper&& o) noexcept {
  *this = std::move(o);
}

GradClipper& GradClipper::operator=(GradClipper&& o) noexcept {
  if (this != &o) {
    release_();
    n_                 = o.n_;                 o.n_                 = 0;
    device_            = o.device_;
    d_ptrs_            = o.d_ptrs_;            o.d_ptrs_            = nullptr;
    d_sizes_           = o.d_sizes_;           o.d_sizes_           = nullptr;
    d_sumsq_           = o.d_sumsq_;           o.d_sumsq_           = nullptr;
    d_scale_           = o.d_scale_;           o.d_scale_           = nullptr;
    d_norm_            = o.d_norm_;            o.d_norm_            = nullptr;
    d_chunk_to_tensor_ = o.d_chunk_to_tensor_; o.d_chunk_to_tensor_ = nullptr;
    d_chunk_to_offset_ = o.d_chunk_to_offset_; o.d_chunk_to_offset_ = nullptr;
    n_chunks_          = o.n_chunks_;          o.n_chunks_          = 0;
    cpu_params_ = std::move(o.cpu_params_);
    cpu_norm_   = o.cpu_norm_;
  }
  return *this;
}

GradClipper::~GradClipper() { release_(); }

void GradClipper::release_() {
#ifdef USE_CUDA
  if (d_ptrs_)            cudaFree(d_ptrs_);
  if (d_sizes_)           cudaFree(d_sizes_);
  if (d_sumsq_)           cudaFree(d_sumsq_);
  if (d_scale_)           cudaFree(d_scale_);
  if (d_norm_)            cudaFree(d_norm_);
  if (d_chunk_to_tensor_) cudaFree(d_chunk_to_tensor_);
  if (d_chunk_to_offset_) cudaFree(d_chunk_to_offset_);
#endif
  d_ptrs_ = nullptr; d_sizes_ = nullptr;
  d_sumsq_ = nullptr; d_scale_ = nullptr; d_norm_ = nullptr;
  d_chunk_to_tensor_ = nullptr; d_chunk_to_offset_ = nullptr;
  n_ = 0; n_chunks_ = 0;
}

void GradClipper::clip(float max_norm) {
  if (n_ == 0) return;

  if (device_.is_cuda()) {
#ifdef USE_CUDA
    cudaStream_t s = reinterpret_cast<cudaStream_t>(
        compute_stream(device_).handle);
    // Three-kernel sequence, capture-safe (no host syncs, no async malloc).
    // Use the MTA path so the grid scales with total chunks, giving uniform
    // SM utilization even with a wide param-size distribution.
    k::zero_scalar_fp32(d_sumsq_, s);
    k::sumsq_fp32_mta(d_ptrs_, d_sizes_, d_chunk_to_tensor_, d_chunk_to_offset_,
                      n_chunks_, d_sumsq_, s);
    k::compute_clip_scale(d_sumsq_, max_norm, d_scale_, d_norm_, s);
    if (max_norm > 0.f) {
      k::scale_fp32_mta_dev(d_ptrs_, d_sizes_, d_chunk_to_tensor_,
                            d_chunk_to_offset_, n_chunks_, d_scale_, s);
    }
    return;
#endif
  }

  // CPU reference.
  double sumsq = 0.0;
  for (auto* p : cpu_params_) {
    const float* g = p->grad.as<float>();
    const int64_t n = p->value.numel();
    for (int64_t i = 0; i < n; ++i) sumsq += double(g[i]) * double(g[i]);
  }
  float norm = std::sqrt(static_cast<float>(sumsq));
  if (max_norm > 0.f && norm > max_norm) {
    float scale = max_norm / (norm + 1e-6f);
    for (auto* p : cpu_params_) {
      float* g = p->grad.as<float>();
      const int64_t n = p->value.numel();
      for (int64_t i = 0; i < n; ++i) g[i] *= scale;
    }
  }
  cpu_norm_ = norm;
}

float GradClipper::pull_last_norm() const {
  if (n_ == 0) return 0.f;
  if (device_.is_cuda()) {
#ifdef USE_CUDA
    float h = 0.f;
    ZWT_CUDA(cudaMemcpy(&h, d_norm_, sizeof(float), cudaMemcpyDeviceToHost));
    return h;
#else
    return 0.f;
#endif
  }
  return cpu_norm_;
}

// ---------------------------------------------------------------------------
// Free-function wrapper
// ---------------------------------------------------------------------------

float clip_grad_norm(const std::vector<Parameter*>& params, float max_norm) {
  if (params.empty()) return 0.f;
  GradClipper c(params);
  c.clip(max_norm);
  return c.pull_last_norm();
}

}  // namespace zwt::optim
