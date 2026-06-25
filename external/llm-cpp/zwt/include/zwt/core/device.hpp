#pragma once

#include <cstdint>

namespace zwt {

enum class DeviceKind : uint8_t {
  CPU = 0,
  CUDA = 1,
};

struct Device {
  DeviceKind kind = DeviceKind::CPU;
  int8_t index = 0;

  constexpr Device() = default;
  constexpr Device(DeviceKind k, int8_t i = 0) : kind(k), index(i) {}

  static constexpr Device cpu() { return Device{DeviceKind::CPU, 0}; }
  static constexpr Device cuda(int8_t i = 0) { return Device{DeviceKind::CUDA, i}; }

  constexpr bool is_cpu()  const { return kind == DeviceKind::CPU; }
  constexpr bool is_cuda() const { return kind == DeviceKind::CUDA; }

  constexpr bool operator==(const Device& o) const { return kind == o.kind && index == o.index; }
  constexpr bool operator!=(const Device& o) const { return !(*this == o); }
};

}  // namespace zwt
