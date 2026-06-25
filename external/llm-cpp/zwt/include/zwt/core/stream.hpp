#pragma once

#include "zwt/core/device.hpp"
#include <cstdint>

// Opaque CUDA handles. We don't include <cuda_runtime.h> in public headers
// so CPU-only builds don't need the CUDA toolkit to include zwt.
namespace zwt {

using StreamHandle = void*;  // cudaStream_t when CUDA; nullptr on CPU
using EventHandle  = void*;  // cudaEvent_t when CUDA; nullptr on CPU

struct Stream {
  Device device;
  StreamHandle handle = nullptr;

  Stream() = default;
  Stream(Device d, StreamHandle h) : device(d), handle(h) {}

  void synchronize() const;
};

struct Event {
  Device device;
  EventHandle handle = nullptr;

  Event() = default;
  ~Event();
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& o) noexcept : device(o.device), handle(o.handle) { o.handle = nullptr; }
  Event& operator=(Event&& o) noexcept {
    if (this != &o) { reset(); device = o.device; handle = o.handle; o.handle = nullptr; }
    return *this;
  }

  static Event create(Device d);
  void reset();
  void record(Stream s);
  void wait(Stream s) const;   // make s wait until this event fires (device-side)
  void synchronize() const;    // host-side sync
};

// The "compute" stream is where forward+backward+optimizer run.
// The "copy" stream is where the dataloader stages H2D transfers so they
// overlap with the previous step's compute. Separate streams = zero wait.
Stream compute_stream(Device dev);
Stream copy_stream(Device dev);

// Side stream pool for truly parallel work (Muon NS, grad allreduce, etc.).
// Round-robins through a fixed pool to avoid per-op stream creation cost.
Stream side_stream(Device dev);

}  // namespace zwt
