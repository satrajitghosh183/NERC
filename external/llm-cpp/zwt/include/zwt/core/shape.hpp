#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <initializer_list>

namespace zwt {

// Fixed-rank shape/strides. Rank cap at 6 covers every transformer activation
// (up to [B, H, S_q, S_kv, D, x]) without dragging std::vector into hot code.
constexpr int kMaxRank = 6;

struct Shape {
  std::array<int64_t, kMaxRank> dims{};
  int rank = 0;

  constexpr Shape() = default;

  Shape(std::initializer_list<int64_t> ds) {
    rank = static_cast<int>(ds.size());
    int i = 0;
    for (auto d : ds) dims[i++] = d;
  }

  int64_t operator[](int i) const { return dims[i]; }
  int64_t& operator[](int i)      { return dims[i]; }

  int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < rank; ++i) n *= dims[i];
    return n;
  }

  bool operator==(const Shape& o) const {
    if (rank != o.rank) return false;
    for (int i = 0; i < rank; ++i) if (dims[i] != o.dims[i]) return false;
    return true;
  }
  bool operator!=(const Shape& o) const { return !(*this == o); }
};

// Row-major (contiguous) strides from a shape.
inline Shape contiguous_strides(const Shape& s) {
  Shape out;
  out.rank = s.rank;
  int64_t stride = 1;
  for (int i = s.rank - 1; i >= 0; --i) {
    out.dims[i] = stride;
    stride *= s.dims[i];
  }
  return out;
}

}  // namespace zwt
