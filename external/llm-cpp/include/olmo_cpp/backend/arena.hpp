#pragma once

/**
 * include/olmo_cpp/backend/arena.hpp
 *
 * Bump-allocator (a.k.a. linear/region/arena allocator) for short-lived
 * scratch tensors during a transformer forward/backward pass. Instead of
 * calling malloc/free for every intermediate, we reserve one large block
 * up front and hand out aligned slices by advancing a single offset
 * pointer. Freeing is O(1): rewind the offset to a saved "mark" — every
 * tensor that was sub-allocated past that mark is logically gone.
 *
 * The arena is thread-local: in multi-threaded data loaders or per-CUDA-
 * stream worker threads, each thread sees its own buffer with no locking.
 *
 * --- Includes from this project ---
 *   - (none — this is a leaf header)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/backend/simd_backend.cpp: SIMDBackend::begin_scope/end_scope/
 *     alloc_scratch use thread_arena() for FFN/attention scratch buffers.
 *   - src/backend/arena.cpp: provides the implementation.
 *
 * --- Role in training pipeline ---
 *   On CPU/SIMD code paths, every TransformerBlock forward allocates
 *   handfuls of intermediate tensors (Q*K^T, softmax output, gate/up
 *   activations). With the arena, these hit a hot region of cached
 *   memory and bypass the system allocator entirely. begin_scope() at
 *   the top of the block and end_scope() at the bottom turn one block
 *   of scratch into a per-block recycled buffer.
 */

#include <torch/torch.h>
#include <cstdint>
#include <mutex>
#include <vector>

namespace olmo_cpp {

/// Arena-style scratch memory allocator for temporary tensors during forward/backward.
///
/// Eliminates repeated malloc/free overhead by pre-allocating large blocks and
/// sub-allocating from them. Works with begin_scope()/end_scope() on IBackend.
///
/// Usage:
///   Arena arena(64 * 1024 * 1024);  // 64 MB
///   {
///     ArenaScope scope(arena);
///     auto tmp1 = arena.allocate_tensor({batch, seq, dim}, torch::kFloat32);
///     auto tmp2 = arena.allocate_tensor({batch, seq, dim}, torch::kFloat32);
///     // ... use tmp1, tmp2 ...
///   }  // scope ends: all allocations freed at once (O(1) reset)
///
class Arena {
 public:
  /// Create arena with given capacity in bytes.
  explicit Arena(size_t capacity_bytes = 64 * 1024 * 1024);

  /// Allocate a tensor from the arena. Falls back to regular alloc if arena is full.
  /// The returned tensor is NOT initialized (contains garbage).
  torch::Tensor allocate_tensor(at::IntArrayRef sizes, torch::ScalarType dtype,
                                 torch::Device device = torch::kCPU);

  /// Reset the arena offset to the given mark (or 0). O(1) operation.
  void reset(size_t mark = 0);

  /// Get current allocation mark (for nested scopes).
  size_t mark() const { return offset_; }

  /// Total capacity in bytes.
  size_t capacity() const { return capacity_; }

  /// Bytes currently allocated.
  size_t used() const { return offset_; }

  /// Number of allocations that fell through to system malloc.
  int64_t fallback_count() const { return fallback_count_; }

 private:
  /// The actual byte storage. std::vector owns the heap allocation; we
  /// only ever read/write into buffer_.data() and never resize it after
  /// construction (so pointers handed out remain valid for the arena's
  /// lifetime).
  std::vector<uint8_t> buffer_;
  /// Total capacity in bytes; matches buffer_.size().
  size_t capacity_;
  /// Bump pointer: next allocation starts at buffer_.data() + offset_.
  /// reset(mark) just rewinds this counter — no destructors run, no
  /// memory is returned to the OS.
  size_t offset_ = 0;
  /// Counts how many allocations spilled to torch::empty() because the
  /// arena was full. Useful for tuning capacity_.
  int64_t fallback_count_ = 0;
};

/// RAII scope guard for arena. Saves the mark on construction, resets on destruction.
class ArenaScope {
 public:
  explicit ArenaScope(Arena& arena) : arena_(arena), saved_mark_(arena.mark()) {}
  ~ArenaScope() { arena_.reset(saved_mark_); }

  // Non-copyable, non-movable
  ArenaScope(const ArenaScope&) = delete;
  ArenaScope& operator=(const ArenaScope&) = delete;

 private:
  Arena& arena_;
  size_t saved_mark_;
};

/// Thread-local arena accessor. Each thread gets its own arena.
Arena& thread_arena();

/// Set the thread-local arena capacity (must be called before first use).
void set_arena_capacity(size_t bytes);

}  // namespace olmo_cpp
