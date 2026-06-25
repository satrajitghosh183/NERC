#include "zwt/optim/adamw.hpp"
#include "zwt/core/stream.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#endif

namespace zwt::optim {

#ifdef USE_CUDA
constexpr int kChunkSize = 4096;  // must match adamw.cu

namespace k {
void adamw_multi_tensor_bf16(
    void** p_ptrs,
    float** g_ptrs,
    float** m_ptrs,
    float** v_ptrs,
    const int64_t* sizes,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2_sqrt,
    int64_t total_chunks,
    const int64_t* d_chunk_tensor_idx, const int64_t* d_chunk_offsets,
    cudaStream_t s);
}  // namespace k
#endif

struct AdamW::State {
  // Exponential moving averages — always fp32, regardless of param dtype.
  std::vector<Tensor> m;
  std::vector<Tensor> v;

#ifdef USE_CUDA
  // Device-side plan. Set once at construction and never touched again —
  // parameter buffers don't move after model init, so the chunk layout is
  // static. No process-scope state.
  void*    d_p_ptrs = nullptr;
  float**  d_g_ptrs = nullptr;
  float**  d_m_ptrs = nullptr;
  float**  d_v_ptrs = nullptr;
  int64_t* d_sizes  = nullptr;
  int64_t* d_chunk_tensor_idx = nullptr;
  int64_t* d_chunk_offsets    = nullptr;
  int      n = 0;
  int64_t  total_chunks = 0;
#endif
};

AdamW::AdamW(std::vector<Parameter*> params, AdamWConfig cfg)
    : cfg_(cfg), params_(std::move(params)) {
  state_ = new State();
  for (auto* p : params_) {
    state_->m.emplace_back(zeros(p->value.shape(), DType::F32, p->value.device()));
    state_->v.emplace_back(zeros(p->value.shape(), DType::F32, p->value.device()));
    p->ensure_grad();
  }

  if (params_.empty()) return;

#ifdef USE_CUDA
  if (params_.front()->value.device().is_cuda()) {
    const int n = static_cast<int>(params_.size());
    std::vector<void*>    p_h(n);
    std::vector<float*>   g_h(n);
    std::vector<float*>   m_h(n);
    std::vector<float*>   v_h(n);
    std::vector<int64_t>  s_h(n);
    int64_t total_chunks = 0;
    for (int i = 0; i < n; ++i) {
      p_h[i] = params_[i]->value.data();
      g_h[i] = params_[i]->grad.as<float>();
      m_h[i] = state_->m[i].as<float>();
      v_h[i] = state_->v[i].as<float>();
      s_h[i] = params_[i]->value.numel();
      total_chunks += (s_h[i] + kChunkSize - 1) / kChunkSize;
    }

    std::vector<int64_t> idx_h(total_chunks);
    std::vector<int64_t> off_h(total_chunks);
    int64_t cur = 0;
    for (int i = 0; i < n; ++i) {
      int64_t off = 0;
      while (off < s_h[i]) {
        idx_h[cur] = i;
        off_h[cur] = off;
        ++cur;
        off += kChunkSize;
      }
    }

    cudaMalloc(&state_->d_p_ptrs, sizeof(void*)   * n);
    cudaMalloc(&state_->d_g_ptrs, sizeof(float*)  * n);
    cudaMalloc(&state_->d_m_ptrs, sizeof(float*)  * n);
    cudaMalloc(&state_->d_v_ptrs, sizeof(float*)  * n);
    cudaMalloc(&state_->d_sizes,  sizeof(int64_t) * n);
    cudaMalloc(&state_->d_chunk_tensor_idx, sizeof(int64_t) * total_chunks);
    cudaMalloc(&state_->d_chunk_offsets,    sizeof(int64_t) * total_chunks);
    cudaMemcpy(state_->d_p_ptrs, p_h.data(), sizeof(void*)   * n, cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_g_ptrs, g_h.data(), sizeof(float*)  * n, cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_m_ptrs, m_h.data(), sizeof(float*)  * n, cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_v_ptrs, v_h.data(), sizeof(float*)  * n, cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_sizes,  s_h.data(), sizeof(int64_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_chunk_tensor_idx, idx_h.data(),
               sizeof(int64_t) * total_chunks, cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_chunk_offsets,    off_h.data(),
               sizeof(int64_t) * total_chunks, cudaMemcpyHostToDevice);
    state_->n = n;
    state_->total_chunks = total_chunks;
  }
#endif
}

AdamW::~AdamW() {
#ifdef USE_CUDA
  if (state_) {
    if (state_->d_p_ptrs)           cudaFree(state_->d_p_ptrs);
    if (state_->d_g_ptrs)           cudaFree(state_->d_g_ptrs);
    if (state_->d_m_ptrs)           cudaFree(state_->d_m_ptrs);
    if (state_->d_v_ptrs)           cudaFree(state_->d_v_ptrs);
    if (state_->d_sizes)            cudaFree(state_->d_sizes);
    if (state_->d_chunk_tensor_idx) cudaFree(state_->d_chunk_tensor_idx);
    if (state_->d_chunk_offsets)    cudaFree(state_->d_chunk_offsets);
  }
#endif
  delete state_;
}

void AdamW::zero_grad() {
  for (auto* p : params_) p->zero_grad();
}

Tensor& AdamW::moment_m(size_t i) { return state_->m[i]; }
Tensor& AdamW::moment_v(size_t i) { return state_->v[i]; }

void AdamW::step() {
  ++step_count_;
  const float bc1 = 1.0f - std::pow(cfg_.beta1, float(step_count_));
  const float bc2 = 1.0f - std::pow(cfg_.beta2, float(step_count_));
  [[maybe_unused]] const float bc2_sqrt = std::sqrt(bc2);

  if (params_.empty()) return;

  bool on_cuda = params_.front()->value.device().is_cuda();

  if (on_cuda) {
#ifdef USE_CUDA
    cudaStream_t s = reinterpret_cast<cudaStream_t>(
        compute_stream(params_.front()->value.device()).handle);
    k::adamw_multi_tensor_bf16(
        reinterpret_cast<void**>(state_->d_p_ptrs),
        state_->d_g_ptrs, state_->d_m_ptrs, state_->d_v_ptrs,
        state_->d_sizes,
        cfg_.lr, cfg_.beta1, cfg_.beta2, cfg_.eps, cfg_.weight_decay,
        bc1, bc2_sqrt,
        state_->total_chunks,
        state_->d_chunk_tensor_idx, state_->d_chunk_offsets,
        s);
    return;
#endif
  }

  // CPU reference. One loop over all params; fp32 params + fp32 grads only.
  for (size_t i = 0; i < params_.size(); ++i) {
    auto* p = params_[i];
    if (p->value.dtype() != DType::F32) {
      throw std::runtime_error("AdamW CPU path: F32 params only");
    }
    float* pv = p->value.as<float>();
    float* gv = p->grad.as<float>();
    float* m  = state_->m[i].as<float>();
    float* v  = state_->v[i].as<float>();
    const int64_t n = p->value.numel();
    for (int64_t j = 0; j < n; ++j) {
      float g = gv[j];
      m[j] = cfg_.beta1 * m[j] + (1.0f - cfg_.beta1) * g;
      v[j] = cfg_.beta2 * v[j] + (1.0f - cfg_.beta2) * g * g;
      float m_hat = m[j] / bc1;
      float v_hat = v[j] / bc2;
      pv[j] -= cfg_.lr * (m_hat / (std::sqrt(v_hat) + cfg_.eps) + cfg_.weight_decay * pv[j]);
    }
  }
}

}  // namespace zwt::optim
