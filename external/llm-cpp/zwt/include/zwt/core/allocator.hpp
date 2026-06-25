#pragma once

#include "zwt/core/device.hpp"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace zwt {

// Minimal allocator interface. Two concrete impls below:
//  * ArenaAllocator — bump-pointer, per-stream scratch; reset() frees all at once.
//  * PoolAllocator  — size-bucketed freelist for long-lived tensors
//                     (parameters, optimizer state, persistent buffers).
//
// Neither uses refcounts. Tensor owns its storage explicitly; on destruction
// the allocator is called to release. No PyTorch-style caching indirection.
class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void* alloc(size_t bytes, size_t alignment = 256) = 0;
  virtual void  free(void* ptr, size_t bytes) = 0;
  virtual Device device() const = 0;
};

// Bump-pointer arena with an optional in-step free-list. Forward-pass
// activations get allocated by bump; when an activation is dropped mid-step
// (e.g. a saved tensor consumed by its op's backward), its slot returns to
// the free-list and can be reused by later allocations within the same step.
// reset() still clears everything at step end.
//
// Two paths for free():
//   * LIFO rewind fast path — if the freed slot is exactly the top of the
//     bump pointer, retract offset_ in place. No list entry needed.
//   * Best-fit scan — for out-of-order frees, (offset, size) is pushed onto
//     free_list_; alloc() picks the smallest fitting entry. Entries are
//     consumed whole (no split) so accounting stays trivial; the cost is a
//     little overallocation vs. a splitting allocator.
class ArenaAllocator final : public Allocator {
 public:
  ArenaAllocator(Device dev, size_t capacity_bytes);
  ~ArenaAllocator() override;

  void* alloc(size_t bytes, size_t alignment = 256) override;
  void  free(void* ptr, size_t bytes) override;
  Device device() const override { return device_; }

  void   reset()      { offset_ = 0; free_list_.clear(); free_bytes_ = 0; }
  size_t mark() const { return offset_; }
  void   rewind(size_t m) { offset_ = m; }
  size_t used() const { return offset_; }
  size_t capacity() const { return capacity_; }
  size_t free_listed_bytes() const { return free_bytes_; }
  size_t peak() const { return peak_; }

 private:
  struct FreeEntry { size_t offset; size_t size; };

  Device device_;
  void*  base_     = nullptr;
  size_t capacity_ = 0;
  size_t offset_   = 0;
  size_t peak_     = 0;
  std::vector<FreeEntry> free_list_;
  size_t free_bytes_ = 0;
};

// Size-bucketed pool. Parameters, gradients, optimizer state live here —
// they persist across steps but are freed exactly once. We don't need a
// general-purpose allocator; we need the three or four sizes training uses.
class PoolAllocator final : public Allocator {
 public:
  explicit PoolAllocator(Device dev);
  ~PoolAllocator() override;

  void* alloc(size_t bytes, size_t alignment = 256) override;
  void  free(void* ptr, size_t bytes) override;
  Device device() const override { return device_; }

  size_t live_bytes()  const { return live_bytes_; }
  size_t cached_bytes() const { return cached_bytes_; }

 private:
  struct Block { void* ptr; size_t size; };

  void* raw_alloc(size_t bytes);
  void  raw_free(void* p);

  Device device_;
  // One freelist per power-of-two bucket. Index = ceil(log2(size)).
  std::vector<std::vector<Block>> buckets_;
  std::mutex mu_;
  size_t live_bytes_ = 0;
  size_t cached_bytes_ = 0;
};

// Process-wide allocator lookup. You normally touch these two:
//   device_pool(Device)    — long-lived params/grads/optimizer state
//   activation_arena(Device) — per-step scratch; must be reset each step
Allocator& device_pool(Device dev);
Allocator& activation_arena(Device dev);

// Tuning hook: set the activation arena capacity before the first use.
// Default is 1 GiB on CUDA, 256 MiB on CPU.
void set_activation_arena_capacity(size_t bytes);

}  // namespace zwt
