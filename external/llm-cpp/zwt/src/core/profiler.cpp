#include "zwt/core/profiler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt {

namespace {

// Read ZWT_PROFILE once at startup. Anything truthy enables; anything else
// (or unset) keeps the profiler off so disabled-path callers pay zero.
bool env_enabled() {
  const char* v = std::getenv("ZWT_PROFILE");
  if (!v) return false;
  return std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0
      && std::strcmp(v, "off") != 0;
}

#ifdef USE_CUDA
inline cudaStream_t cu(StreamHandle s) {
  return reinterpret_cast<cudaStream_t>(s);
}
#endif

}  // namespace

Profiler& Profiler::get() {
  static Profiler p;
  return p;
}

Profiler::Profiler() : enabled_(env_enabled()) {
  buckets_.reserve(64);
  pending_.reserve(256);
  event_pool_.reserve(64);
}

Profiler::~Profiler() {
  // Pool cleanup. cudaEventDestroy can be called even after device shutdown
  // if it was created here, but we ignore errors to avoid noisy warnings on
  // process exit.
#ifdef USE_CUDA
  for (void* ev : event_pool_) {
    if (ev) cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
  }
#endif
}

void Profiler::set_enabled(bool on) {
  enabled_ = on;
}

void Profiler::reset() {
  for (auto& b : buckets_) {
    b.count = 0;
    b.total_us = 0.0;
    b.min_us = 1e30;
    b.max_us = 0.0;
  }
  // Pending events are NOT dropped — they'll record into whatever buckets
  // they belong to when drained. Caller can drain_pending_() first if they
  // want a clean reset.
}

void Profiler::compact() {}

int Profiler::bucket_id_(const char* name) {
  // Linear scan: bucket count is small (< 64 in practice).
  for (size_t i = 0; i < buckets_.size(); ++i) {
    if (buckets_[i].name == name) return static_cast<int>(i);
  }
  buckets_.push_back(Bucket{name, 0, 0.0, 1e30, 0.0});
  return static_cast<int>(buckets_.size() - 1);
}

void Profiler::record_(int id, double us) {
  Bucket& b = buckets_[id];
  ++b.count;
  b.total_us += us;
  if (us < b.min_us) b.min_us = us;
  if (us > b.max_us) b.max_us = us;
}

void Profiler::drain_pending_() {
#ifdef USE_CUDA
  for (auto it = pending_.begin(); it != pending_.end();) {
    cudaEvent_t e0 = reinterpret_cast<cudaEvent_t>(it->start_ev);
    cudaEvent_t e1 = reinterpret_cast<cudaEvent_t>(it->stop_ev);
    cudaError_t r = cudaEventQuery(e1);
    if (r == cudaSuccess) {
      float ms = 0.f;
      if (cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess) {
        record_(it->bucket_id, static_cast<double>(ms) * 1000.0);
      }
      // Pool reuse — pushed back as start, then stop, in that pop order.
      event_pool_.push_back(e0);
      event_pool_.push_back(e1);
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }
#else
  pending_.clear();
#endif
}

void Profiler::print_summary(std::FILE* fp) const {
  // Drain isn't possible without a non-const pull; const_cast is acceptable
  // here since draining only mutates internal state.
  const_cast<Profiler*>(this)->drain_pending_();

  if (buckets_.empty()) {
    std::fprintf(fp, "[zwt-profile] no samples recorded "
                     "(set ZWT_PROFILE=1 to enable)\n");
    return;
  }

  std::vector<int> order(buckets_.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return buckets_[a].total_us > buckets_[b].total_us;
  });
  double grand_total = 0.0;
  for (auto& b : buckets_) grand_total += b.total_us;

  std::fprintf(fp,
      "\n[zwt-profile] per-stage timing — total=%.3f ms across %zu buckets\n",
      grand_total / 1000.0, buckets_.size());
  std::fprintf(fp,
      "  %-36s %8s %12s %10s %10s %10s %6s\n",
      "stage", "count", "total_ms", "mean_us", "min_us", "max_us", "%total");
  for (int i : order) {
    const Bucket& b = buckets_[i];
    if (b.count == 0) continue;
    double mean_us = b.total_us / static_cast<double>(b.count);
    double pct = grand_total > 0.0 ? (100.0 * b.total_us / grand_total) : 0.0;
    std::fprintf(fp,
        "  %-36s %8llu %12.3f %10.2f %10.2f %10.2f %6.2f\n",
        b.name.c_str(),
        static_cast<unsigned long long>(b.count),
        b.total_us / 1000.0,
        mean_us, b.min_us, b.max_us, pct);
  }
  std::fflush(fp);
}

void Profiler::dump_csv(const std::string& path, int64_t step) const {
  const_cast<Profiler*>(this)->drain_pending_();

  // Header is written once. We detect "first call" by trying to open
  // append-mode and checking the file size; if zero, write the header.
  bool need_header = false;
  {
    std::ifstream probe(path, std::ios::binary | std::ios::ate);
    if (!probe.good() || static_cast<long>(probe.tellg()) <= 0) {
      need_header = true;
    }
  }
  std::FILE* fp = std::fopen(path.c_str(), "a");
  if (!fp) return;
  if (need_header) {
    std::fprintf(fp,
        "step,stage,count,total_us,mean_us,min_us,max_us\n");
  }
  for (const auto& b : buckets_) {
    if (b.count == 0) continue;
    double mean_us = b.total_us / static_cast<double>(b.count);
    std::fprintf(fp,
        "%lld,%s,%llu,%.3f,%.3f,%.3f,%.3f\n",
        static_cast<long long>(step), b.name.c_str(),
        static_cast<unsigned long long>(b.count),
        b.total_us, mean_us, b.min_us, b.max_us);
  }
  std::fflush(fp);
  std::fclose(fp);
}

// ── GpuScope ─────────────────────────────────────────────────────────────

Profiler::GpuScope::GpuScope(const char* name, Device dev, StreamHandle s) {
  Profiler& p = Profiler::get();
  if (!p.enabled_) return;
  enabled_ = true;
  bucket_id_ = p.bucket_id_(name);
  stream_ = s;
#ifdef USE_CUDA
  if (dev.is_cuda()) {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (!p.event_pool_.empty()) {
      e0 = reinterpret_cast<cudaEvent_t>(p.event_pool_.back()); p.event_pool_.pop_back();
      e1 = reinterpret_cast<cudaEvent_t>(p.event_pool_.back()); p.event_pool_.pop_back();
    } else {
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
    }
    cudaEventRecord(e0, cu(s));
    start_ev_ = e0;
    stop_ev_  = e1;
  }
#else
  (void)dev; (void)s;
#endif
}

Profiler::GpuScope::~GpuScope() {
  if (!enabled_) return;
  Profiler& p = Profiler::get();
#ifdef USE_CUDA
  if (start_ev_ && stop_ev_) {
    cudaEvent_t e1 = reinterpret_cast<cudaEvent_t>(stop_ev_);
    cudaEventRecord(e1, cu(stream_));
    p.pending_.push_back({bucket_id_, start_ev_, stop_ev_});
  }
#else
  (void)p;
#endif
}

// ── CpuScope ─────────────────────────────────────────────────────────────

Profiler::CpuScope::CpuScope(const char* name) {
  Profiler& p = Profiler::get();
  if (!p.enabled_) return;
  enabled_ = true;
  bucket_id_ = p.bucket_id_(name);
  t0_ = std::chrono::steady_clock::now();
}

Profiler::CpuScope::~CpuScope() {
  if (!enabled_) return;
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0_).count();
  Profiler::get().record_(bucket_id_, us);
}

}  // namespace zwt
