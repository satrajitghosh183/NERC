#pragma once

#include "zwt/core/allocator.hpp"
#include "zwt/core/device.hpp"
#include "zwt/core/dtype.hpp"
#include "zwt/core/shape.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace zwt {

// Unrefcounted, owning tensor. Key invariants:
//  * Move-only — copy means "copy the data" (explicit) and must go through ops.
//  * Storage is owned by an Allocator pointed to by `alloc_`. On destruction,
//    Tensor calls alloc_->free() once and only once. No atomic refcount.
//  * Views do not own storage; they carry `alloc_ == nullptr`.
//
// This is deliberately *not* a replacement for torch::Tensor in the Python
// sense — we don't mean to support arbitrary composition. We mean to provide
// a shape-aware buffer handle for the layer/op API and nothing more.
class Tensor {
 public:
  Tensor() = default;

  Tensor(void* data, Shape shape, Shape strides, DType dtype,
         Device device, Allocator* alloc, size_t storage_bytes)
      : data_(data), shape_(shape), strides_(strides), dtype_(dtype),
        device_(device), alloc_(alloc), storage_bytes_(storage_bytes) {}

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  Tensor(Tensor&& o) noexcept { *this = std::move(o); }
  Tensor& operator=(Tensor&& o) noexcept {
    if (this != &o) {
      release_();
      data_ = o.data_;        o.data_ = nullptr;
      shape_ = o.shape_;
      strides_ = o.strides_;
      dtype_ = o.dtype_;
      device_ = o.device_;
      alloc_ = o.alloc_;      o.alloc_ = nullptr;
      storage_bytes_ = o.storage_bytes_;  o.storage_bytes_ = 0;
    }
    return *this;
  }

  ~Tensor() { release_(); }

  // Accessors
  void*          data()           { return data_; }
  const void*    data()     const { return data_; }
  template <typename T> T*       as()       { return static_cast<T*>(data_); }
  template <typename T> const T* as() const { return static_cast<const T*>(data_); }

  const Shape&   shape()    const { return shape_; }
  const Shape&   strides()  const { return strides_; }
  int            rank()     const { return shape_.rank; }
  int64_t        dim(int i) const { return shape_.dims[i]; }
  int64_t        numel()    const { return shape_.numel(); }
  size_t         nbytes()   const { return static_cast<size_t>(numel()) * dtype_size(dtype_); }
  DType          dtype()    const { return dtype_; }
  Device         device()   const { return device_; }
  bool           is_owning() const { return alloc_ != nullptr; }

  // A view carries the same storage but no ownership. Use for reshape/reinterpret
  // inside an op; never escape a view past the owner's scope.
  Tensor view(const Shape& new_shape) const {
    Tensor v;
    v.data_ = data_;
    v.shape_ = new_shape;
    v.strides_ = contiguous_strides(new_shape);
    v.dtype_ = dtype_;
    v.device_ = device_;
    v.alloc_ = nullptr;
    v.storage_bytes_ = 0;
    return v;
  }

  // Zero the storage in-place (on the compute stream for the owning device).
  void zero_();

  // Debug.
  std::string describe() const;

 private:
  void release_() {
    if (alloc_ && data_) {
      alloc_->free(data_, storage_bytes_);
    }
    data_ = nullptr;
    alloc_ = nullptr;
    storage_bytes_ = 0;
  }

  void*      data_ = nullptr;
  Shape      shape_;
  Shape      strides_;
  DType      dtype_  = DType::F32;
  Device     device_;
  Allocator* alloc_  = nullptr;
  size_t     storage_bytes_ = 0;
};

// Factory functions. These allocate out of the pool (persistent) or arena
// (scratch). Choose the right one based on lifetime — there is no escape hatch.
Tensor empty(const Shape& shape, DType dtype, Device device);       // pool
Tensor empty_scratch(const Shape& shape, DType dtype, Device device); // arena
Tensor zeros(const Shape& shape, DType dtype, Device device);

// H2D / D2H transfer on the copy stream (or the compute stream if same device).
void copy(const Tensor& src, Tensor& dst);

}  // namespace zwt
