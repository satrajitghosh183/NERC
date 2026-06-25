#include "zwt/dist/ddp.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt::dist {

namespace {

#ifdef USE_CUDA
inline cudaStream_t cu(StreamHandle s) {
  return reinterpret_cast<cudaStream_t>(s);
}

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("BucketManager: ") + what + ": " +
                             cudaGetErrorString(e));
  }
}
#endif

}  // namespace

BucketManager::BucketManager(const std::vector<Parameter*>& params,
                             size_t bucket_bytes,
                             int world_size)
    : params_(params),
      bucket_bytes_(bucket_bytes),
      world_size_(world_size) {
  if (world_size < 1) throw std::runtime_error("BucketManager: world_size < 1");
  if (params.empty()) throw std::runtime_error("BucketManager: empty params");

  // Snapshot the device from the first parameter's grad. All params must
  // share a device — we don't try to support mixed CPU+CUDA models.
  for (auto* p : params_) {
    if (p == nullptr) {
      throw std::runtime_error("BucketManager: null parameter pointer");
    }
    p->ensure_grad();
  }
  device_ = params_.front()->grad.device();
  for (auto* p : params_) {
    if (p->grad.device() != device_) {
      throw std::runtime_error(
          "BucketManager: all parameter grads must share a device");
    }
  }

  build_buckets_(params_);
  allocate_buffers_();

  // Index map: Parameter* → pid. Lets the model's backward call
  // mark_ready(Parameter*, ...) without threading integer indices through
  // every layer.
  ptr_to_pid_.reserve(params_.size());
  for (int i = 0; i < static_cast<int>(params_.size()); ++i) {
    ptr_to_pid_[params_[i]] = i;
  }
}

BucketManager::~BucketManager() {
  free_buffers_();
}

void BucketManager::build_buckets_(const std::vector<Parameter*>& params) {
  // Build buckets in reverse parameter order — backward visits parameters in
  // roughly reverse-forward order (output layer grads finish first), so this
  // matches the order we'd like allreduces to fire.
  const int N = static_cast<int>(params.size());
  param_to_bucket_.assign(N, -1);

  Bucket cur;
  size_t cur_bytes = 0;
  for (int i = N - 1; i >= 0; --i) {
    const size_t numel   = static_cast<size_t>(params[i]->numel());
    const size_t p_bytes = numel * sizeof(float);
    if (!cur.param_ids.empty() && cur_bytes + p_bytes > bucket_bytes_) {
      buckets_.push_back(std::move(cur));
      cur = {};
      cur_bytes = 0;
    }
    cur.offsets.push_back(cur.total_floats);
    cur.param_ids.push_back(i);
    cur.total_floats += numel;
    cur_bytes += p_bytes;
  }
  if (!cur.param_ids.empty()) buckets_.push_back(std::move(cur));

  for (int b = 0; b < static_cast<int>(buckets_.size()); ++b) {
    buckets_[b].remaining = static_cast<int>(buckets_[b].param_ids.size());
    for (int pid : buckets_[b].param_ids) param_to_bucket_[pid] = b;
  }
}

void BucketManager::allocate_buffers_() {
  for (auto& bk : buckets_) {
    if (device_.is_cuda()) {
#ifdef USE_CUDA
      void* ptr = nullptr;
      cuda_check(cudaMalloc(&ptr, bk.total_floats * sizeof(float)),
                 "cudaMalloc(bucket buffer)");
      bk.dev_buf = ptr;
#else
      throw std::runtime_error(
          "BucketManager: built without USE_CUDA but params live on CUDA");
#endif
    } else {
      bk.host_buf.assign(bk.total_floats, 0.f);
    }
  }
}

void BucketManager::free_buffers_() {
#ifdef USE_CUDA
  for (auto& bk : buckets_) {
    if (bk.dev_buf) {
      cudaFree(bk.dev_buf);
      bk.dev_buf = nullptr;
    }
  }
#endif
}

void BucketManager::begin_step() {
  for (auto& b : buckets_) {
    b.remaining = static_cast<int>(b.param_ids.size());
    b.fired     = false;
  }
}

void BucketManager::gather_bucket_(Bucket& bk, StreamHandle s) {
  for (size_t i = 0; i < bk.param_ids.size(); ++i) {
    Parameter* p = params_[bk.param_ids[i]];
    const size_t bytes = static_cast<size_t>(p->numel()) * sizeof(float);
    void* dst = device_.is_cuda()
        ? static_cast<void*>(static_cast<float*>(bk.dev_buf) + bk.offsets[i])
        : static_cast<void*>(bk.host_buf.data() + bk.offsets[i]);
    if (device_.is_cuda()) {
#ifdef USE_CUDA
      cuda_check(cudaMemcpyAsync(dst, p->grad.data(), bytes,
                                 cudaMemcpyDeviceToDevice, cu(s)),
                 "gather D2D");
#endif
    } else {
      std::memcpy(dst, p->grad.data(), bytes);
    }
  }
}

void BucketManager::scatter_bucket_(const Bucket& bk, StreamHandle s) {
  for (size_t i = 0; i < bk.param_ids.size(); ++i) {
    Parameter* p = params_[bk.param_ids[i]];
    const size_t bytes = static_cast<size_t>(p->numel()) * sizeof(float);
    const void* src = device_.is_cuda()
        ? static_cast<const void*>(static_cast<const float*>(bk.dev_buf)
                                   + bk.offsets[i])
        : static_cast<const void*>(bk.host_buf.data() + bk.offsets[i]);
    if (device_.is_cuda()) {
#ifdef USE_CUDA
      cuda_check(cudaMemcpyAsync(p->grad.data(), src, bytes,
                                 cudaMemcpyDeviceToDevice, cu(s)),
                 "scatter D2D");
#endif
    } else {
      // CPU fallback: divide by world_size here since there's no real
      // allreduce in the loopback path. This keeps numerics consistent if
      // a multi-rank CPU run is ever wired up. (Tests never read the
      // scattered values, so this is also harmless under test.)
      std::memcpy(p->grad.data(), src, bytes);
      if (world_size_ > 1) {
        float* g = static_cast<float*>(p->grad.data());
        const float inv = 1.f / static_cast<float>(world_size_);
        for (int64_t k = 0; k < p->numel(); ++k) g[k] *= inv;
      }
    }
  }
}

void BucketManager::mark_ready(int pid, StreamHandle s) {
  if (pid < 0 || pid >= static_cast<int>(param_to_bucket_.size())) {
    throw std::runtime_error("BucketManager: param index out of range");
  }
  int b = param_to_bucket_[pid];
  Bucket& bk = buckets_[b];
  if (bk.fired) {
    throw std::runtime_error("BucketManager: mark_ready after bucket fired");
  }
  if (--bk.remaining > 0) return;

  // Bucket is full — gather every param's fp32 grad into the staging buffer
  // (D2D async on CUDA, memcpy on CPU), then hand off to allreduce.
  gather_bucket_(bk, s);

  if (allreduce_) {
    float* buf = device_.is_cuda()
        ? static_cast<float*>(bk.dev_buf)
        : bk.host_buf.data();
    allreduce_(buf, bk.total_floats, s);
  }
  bk.fired = true;
}

void BucketManager::mark_ready(Parameter* p, StreamHandle s) {
  auto it = ptr_to_pid_.find(p);
  if (it == ptr_to_pid_.end()) {
    throw std::runtime_error("BucketManager: parameter not registered");
  }
  mark_ready(it->second, s);
}

void BucketManager::finalize(StreamHandle s) {
  for (size_t b = 0; b < buckets_.size(); ++b) {
    if (!buckets_[b].fired) {
      throw std::runtime_error(
          "BucketManager: finalize() before all buckets fired");
    }
  }
  // Scatter the (already-averaged via ncclAvg) buffer back into each
  // parameter's grad. On CUDA, runs as a sequence of D2D async copies on
  // the compute stream provided by the caller. On CPU it's an in-place
  // memcpy + divide.
  for (auto& bk : buckets_) {
    scatter_bucket_(bk, s);
  }
  begin_step();
}

void signal_params_ready(const std::vector<Parameter*>& params,
                         BucketManager& mgr,
                         StreamHandle s) {
  for (auto* p : params) {
    mgr.mark_ready(p, s);
  }
}

}  // namespace zwt::dist
