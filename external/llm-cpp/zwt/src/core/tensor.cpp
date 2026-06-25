#include "zwt/core/tensor.hpp"
#include "zwt/core/stream.hpp"

#include <cstring>
#include <sstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt {

namespace {

Tensor alloc_from(Allocator& a, const Shape& shape, DType dtype, Device device) {
  Shape strides = contiguous_strides(shape);
  size_t bytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);
  void* p = a.alloc(bytes);
  return Tensor(p, shape, strides, dtype, device, &a, bytes);
}

}  // namespace

Tensor empty(const Shape& shape, DType dtype, Device device) {
  return alloc_from(device_pool(device), shape, dtype, device);
}

Tensor empty_scratch(const Shape& shape, DType dtype, Device device) {
  return alloc_from(activation_arena(device), shape, dtype, device);
}

Tensor zeros(const Shape& shape, DType dtype, Device device) {
  Tensor t = empty(shape, dtype, device);
  t.zero_();
  return t;
}

void Tensor::zero_() {
  if (!data_ || nbytes() == 0) return;
  if (device_.is_cuda()) {
#ifdef USE_CUDA
    cudaStream_t s = reinterpret_cast<cudaStream_t>(compute_stream(device_).handle);
    cudaMemsetAsync(data_, 0, nbytes(), s);
#endif
    return;
  }
  std::memset(data_, 0, nbytes());
}

void copy(const Tensor& src, Tensor& dst) {
  size_t n = src.nbytes();
  if (n == 0) return;
  if (n != dst.nbytes()) {
    throw std::runtime_error("zwt::copy: byte-size mismatch");
  }
  bool src_cuda = src.device().is_cuda();
  bool dst_cuda = dst.device().is_cuda();

#ifdef USE_CUDA
  cudaMemcpyKind kind =
      (!src_cuda && !dst_cuda) ? cudaMemcpyHostToHost :
      ( src_cuda && !dst_cuda) ? cudaMemcpyDeviceToHost :
      (!src_cuda &&  dst_cuda) ? cudaMemcpyHostToDevice :
                                  cudaMemcpyDeviceToDevice;
  cudaStream_t s = nullptr;
  if (dst_cuda || src_cuda) {
    Device d = dst_cuda ? dst.device() : src.device();
    s = reinterpret_cast<cudaStream_t>(copy_stream(d).handle);
  }
  cudaMemcpyAsync(dst.data(), src.data(), n, kind, s);
#else
  if (src_cuda || dst_cuda) {
    throw std::runtime_error("zwt::copy: CUDA copy requested on CPU-only build");
  }
  std::memcpy(dst.data(), src.data(), n);
#endif
}

std::string Tensor::describe() const {
  std::ostringstream os;
  os << "Tensor(" << dtype_name(dtype_) << ", [";
  for (int i = 0; i < shape_.rank; ++i) {
    if (i) os << ", ";
    os << shape_.dims[i];
  }
  os << "], " << (device_.is_cuda() ? "cuda" : "cpu") << ":" << int(device_.index) << ")";
  return os.str();
}

}  // namespace zwt
