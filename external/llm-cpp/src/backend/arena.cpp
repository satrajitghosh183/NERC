/**
 * src/backend/arena.cpp
 *
 * Implementation of the bump allocator declared in
 * include/olmo_cpp/backend/arena.hpp. Provides Arena::allocate_tensor,
 * Arena::reset, and the thread_local arena accessor pair
 * thread_arena() / set_arena_capacity().
 *
 * --- Includes from this project ---
 *   - olmo_cpp/backend/arena.hpp: declares Arena and helpers.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/backend/simd_backend.cpp: scope_stack_ pushes/pops via
 *     thread_arena().mark()/.reset(); alloc_scratch routes through
 *     thread_arena().allocate_tensor(...).
 *
 * --- Role in training pipeline ---
 *   This file is the entire memory-management primitive for the SIMD
 *   path. It is also dormant on the CUDA path (CUDABackend owns no
 *   arena state). When SIMD is active, every TransformerBlock forward
 *   bumps the arena offset for scratch and rewinds at the end — so the
 *   peak working-set memory is bounded and the system allocator is
 *   never on the hot path.
 */

#include "olmo_cpp/backend/arena.hpp"
#include <algorithm>
#include <cstring>

namespace olmo_cpp {

/// Construct with a fixed byte capacity. The std::vector ctor zero-
/// initializes all bytes (fine for our purposes — we only ever write
/// over the bytes we hand out).
Arena::Arena(size_t capacity_bytes)
    : buffer_(capacity_bytes), capacity_(capacity_bytes) {}

/// Carve out a region of the arena and wrap it as a torch::Tensor that
/// points into the arena buffer. The arena retains ownership: the
/// tensor's deleter is a no-op, so dropping the last reference does not
/// free anything. Memory is reclaimed only by Arena::reset(...).
torch::Tensor Arena::allocate_tensor(at::IntArrayRef sizes, torch::ScalarType dtype,
                                      torch::Device device) {
  // Only arena-allocate CPU tensors. CUDA / MPS allocations would need
  // a device-side arena (cudaMalloc / Metal), which we don't manage here.
  if (!device.is_cpu()) {
    fallback_count_++;
    return torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(device));
  }

  // Compute number of bytes: prod(sizes) * elementSize(dtype).
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  size_t elem_size = torch::elementSize(dtype);
  size_t bytes = static_cast<size_t>(numel) * elem_size;

  // Align to 64 bytes (one cache line). The bit trick (offset+63) & ~63
  // rounds up to the next multiple of 64. Aligned starts let SIMD
  // load/store instructions hit fast paths (e.g. avoid split loads on
  // x86, naturally align AVX2 256-bit lanes).
  size_t aligned_offset = (offset_ + 63) & ~size_t(63);
  size_t new_offset = aligned_offset + bytes;

  if (new_offset > capacity_) {
    // Arena full — degrade gracefully to a system allocation. The
    // returned tensor owns its memory normally and will free on dtor.
    // fallback_count_ surfaces capacity-tuning needs in diagnostics.
    fallback_count_++;
    return torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(device));
  }

  // Compute a raw pointer into the arena buffer and bump the offset.
  void* ptr = buffer_.data() + aligned_offset;
  offset_ = new_offset;

  // torch::from_blob lets us wrap an external pointer as a Tensor.
  // The lambda deleter receives the same pointer when the tensor is
  // dropped, but does nothing — the arena (this object) is the sole
  // owner of the underlying bytes.
  auto tensor = torch::from_blob(
      ptr, sizes,
      /*deleter=*/[](void*) {},  // no-op: arena owns the memory
      torch::TensorOptions().dtype(dtype).device(torch::kCPU));

  return tensor;
}

/// Rewind the bump pointer. Any tensor handed out at offset >= mark is
/// now logically dead — its memory will be reused on the next allocate.
/// Capping at capacity_ guards against an accidentally over-large mark
/// (e.g. from a paired begin_scope that never ran).
void Arena::reset(size_t mark) {
  offset_ = std::min(mark, capacity_);
}

// ---------------------------------------------------------------------------
// Thread-local arena
// ---------------------------------------------------------------------------

/// Per-thread requested arena size. Defaults to 64 MiB which has
/// comfortable headroom for the largest configured model's per-block
/// scratch on CPU. Each new thread first reads this value, then lazily
/// constructs its private Arena.
static thread_local size_t tl_arena_capacity = 64 * 1024 * 1024;  // 64 MB default
/// Lazily-constructed per-thread Arena. unique_ptr<>::reset() lets us
/// rebuild it when the user changes the capacity.
static thread_local std::unique_ptr<Arena> tl_arena;

/// Return (creating on first access) the arena owned by the calling
/// thread. No locking required — thread_local guarantees one Arena per
/// OS thread.
Arena& thread_arena() {
  if (!tl_arena) {
    tl_arena = std::make_unique<Arena>(tl_arena_capacity);
  }
  return *tl_arena;
}

/// Change the arena size for the *current* thread and discard the
/// existing buffer so the next thread_arena() call constructs one of
/// the new size. Other threads are unaffected.
void set_arena_capacity(size_t bytes) {
  tl_arena_capacity = bytes;
  tl_arena.reset();  // force re-creation on next access
}

}  // namespace olmo_cpp
