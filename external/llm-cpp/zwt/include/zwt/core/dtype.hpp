#pragma once

#include <cstddef>
#include <cstdint>

namespace zwt {

enum class DType : uint8_t {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
  I32 = 3,
  I64 = 4,
  U8 = 5,
  Bool = 6,
};

constexpr size_t dtype_size(DType t) {
  switch (t) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::U8:  return 1;
    case DType::Bool: return 1;
  }
  return 0;
}

constexpr const char* dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "f32";
    case DType::F16: return "f16";
    case DType::BF16: return "bf16";
    case DType::I32: return "i32";
    case DType::I64: return "i64";
    case DType::U8:  return "u8";
    case DType::Bool: return "bool";
  }
  return "?";
}

constexpr bool is_float(DType t) {
  return t == DType::F32 || t == DType::F16 || t == DType::BF16;
}

constexpr bool is_int(DType t) {
  return t == DType::I32 || t == DType::I64 || t == DType::U8;
}

}  // namespace zwt
