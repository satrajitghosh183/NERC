#include "zwt/core/allocator.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <new>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt {

namespace {

constexpr size_t kDefaultCpuArena = 256ULL << 20;   // 256 MiB
constexpr size_t kDefaultGpuArena = 1024ULL << 20;  // 1 GiB
size_t g_arena_cap = 0;                             // 0 = use default per device

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// Generic device allocation helpers. On CUDA, use cudaMalloc/Free. On CPU,
// posix_memalign. No caching layer here — the arena+pool sit on top.
void* raw_device_alloc(Device dev, size_t bytes) {
  if (bytes == 0) return nullptr;
  if (dev.is_cuda()) {
#ifdef USE_CUDA
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "zwt: cudaMalloc(%zu) failed: %s\n", bytes,
                   cudaGetErrorString(e));
      throw std::bad_alloc{};
    }
    return p;
#else
    (void)dev; (void)bytes;
    throw std::runtime_error("zwt: built without USE_CUDA");
#endif
  }
  void* p = nullptr;
  if (posix_memalign(&p, 256, bytes) != 0 || !p) throw std::bad_alloc{};
  return p;
}

void raw_device_free(Device dev, void* p) {
  if (!p) return;
  if (dev.is_cuda()) {
#ifdef USE_CUDA
    cudaFree(p);
#endif
    return;
  }
  std::free(p);
}

int bucket_for(size_t bytes) {
  // 256-byte minimum bucket, then powers of 2.
  if (bytes <= 256) return 0;
  int b = 0;
  size_t s = 256;
  while (s < bytes) { s <<= 1; ++b; }
  return b;
}

size_t size_for_bucket(int b) { return size_t{256} << b; }

}  // namespace

// ---------------------------------------------------------------------------
// ArenaAllocator
// ---------------------------------------------------------------------------

ArenaAllocator::ArenaAllocator(Device dev, size_t capacity_bytes)
    : device_(dev), capacity_(capacity_bytes) {
  base_ = raw_device_alloc(dev, capacity_bytes);
}

ArenaAllocator::~ArenaAllocator() {
  if (base_) raw_device_free(device_, base_);
}

void* ArenaAllocator::alloc(size_t bytes, size_t alignment) {
  if (bytes == 0) return nullptr;
  // Best-fit scan over the free-list. Entries store the raw (offset, size)
  // pair as returned by a prior free(); the caller-requested alignment may
  // raise offset within the slot, so the usable bytes shrink correspondingly.
  int best = -1;
  size_t best_waste = static_cast<size_t>(-1);
  for (size_t i = 0; i < free_list_.size(); ++i) {
    const FreeEntry& e = free_list_[i];
    size_t aligned = align_up(e.offset, alignment);
    size_t head = aligned - e.offset;
    if (head >= e.size) continue;
    size_t usable = e.size - head;
    if (usable < bytes) continue;
    size_t waste = usable - bytes;
    if (waste < best_waste) { best = static_cast<int>(i); best_waste = waste; }
    if (waste == 0) break;
  }
  if (best >= 0) {
    FreeEntry e = free_list_[best];
    free_list_.erase(free_list_.begin() + best);
    free_bytes_ -= e.size;
    size_t aligned = align_up(e.offset, alignment);
    return static_cast<char*>(base_) + aligned;
  }
  // Bump path.
  size_t start = align_up(offset_, alignment);
  size_t end = start + bytes;
  if (end > capacity_) {
    std::fprintf(stderr,
        "zwt: arena OOM on %s (wanted %zu, have %zu/%zu, free-listed %zu). "
        "Raise capacity via set_activation_arena_capacity().\n",
        device_.is_cuda() ? "cuda" : "cpu", bytes, capacity_ - offset_,
        capacity_, free_bytes_);
    throw std::bad_alloc{};
  }
  offset_ = end;
  if (offset_ > peak_) peak_ = offset_;
  return static_cast<char*>(base_) + start;
}

void ArenaAllocator::free(void* p, size_t bytes) {
  if (!p || bytes == 0) return;
  size_t off = static_cast<char*>(p) - static_cast<char*>(base_);
  // LIFO fast path: freeing the topmost allocation retracts the bump pointer.
  if (off + bytes == offset_) { offset_ = off; return; }
  free_list_.push_back({off, bytes});
  free_bytes_ += bytes;
}

// ---------------------------------------------------------------------------
// PoolAllocator
// ---------------------------------------------------------------------------

PoolAllocator::PoolAllocator(Device dev) : device_(dev) {
  buckets_.resize(40);  // up to 256 * 2^39 bytes — far beyond practical limits
}

PoolAllocator::~PoolAllocator() {
  std::lock_guard<std::mutex> g(mu_);
  for (auto& b : buckets_) {
    for (auto& blk : b) raw_free(blk.ptr);
    b.clear();
  }
}

void* PoolAllocator::raw_alloc(size_t bytes) { return raw_device_alloc(device_, bytes); }
void  PoolAllocator::raw_free(void* p)       { raw_device_free(device_, p); }

void* PoolAllocator::alloc(size_t bytes, size_t /*alignment*/) {
  if (bytes == 0) return nullptr;
  int bkt = bucket_for(bytes);
  size_t class_bytes = size_for_bucket(bkt);
  {
    std::lock_guard<std::mutex> g(mu_);
    auto& list = buckets_[bkt];
    if (!list.empty()) {
      void* p = list.back().ptr;
      list.pop_back();
      cached_bytes_ -= class_bytes;
      live_bytes_ += class_bytes;
      return p;
    }
  }
  void* p = raw_alloc(class_bytes);
  {
    std::lock_guard<std::mutex> g(mu_);
    live_bytes_ += class_bytes;
  }
  return p;
}

void PoolAllocator::free(void* ptr, size_t bytes) {
  if (!ptr) return;
  int bkt = bucket_for(bytes);
  size_t class_bytes = size_for_bucket(bkt);
  std::lock_guard<std::mutex> g(mu_);
  buckets_[bkt].push_back({ptr, class_bytes});
  live_bytes_ -= class_bytes;
  cached_bytes_ += class_bytes;
}

// ---------------------------------------------------------------------------
// Singletons
// ---------------------------------------------------------------------------

namespace {

PoolAllocator& cpu_pool() {
  static PoolAllocator p{Device::cpu()};
  return p;
}

PoolAllocator& gpu_pool(int idx) {
  static PoolAllocator p0{Device::cuda(0)};
  // Multi-GPU expansion point; for now only one GPU pool. Extend if needed.
  (void)idx;
  return p0;
}

ArenaAllocator& cpu_arena() {
  static ArenaAllocator a{Device::cpu(), g_arena_cap ? g_arena_cap : kDefaultCpuArena};
  return a;
}

ArenaAllocator& gpu_arena(int idx) {
  static ArenaAllocator a{Device::cuda(0), g_arena_cap ? g_arena_cap : kDefaultGpuArena};
  (void)idx;
  return a;
}

}  // namespace

Allocator& device_pool(Device dev) {
  if (dev.is_cuda()) return gpu_pool(dev.index);
  return cpu_pool();
}

Allocator& activation_arena(Device dev) {
  if (dev.is_cuda()) return gpu_arena(dev.index);
  return cpu_arena();
}

void set_activation_arena_capacity(size_t bytes) { g_arena_cap = bytes; }

}  // namespace zwt
