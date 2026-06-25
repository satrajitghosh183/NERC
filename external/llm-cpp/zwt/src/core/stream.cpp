#include "zwt/core/stream.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt {

#ifdef USE_CUDA

namespace {

constexpr int kSideStreamPool = 4;

struct DeviceStreams {
  cudaStream_t compute = nullptr;
  cudaStream_t copy = nullptr;
  std::array<cudaStream_t, kSideStreamPool> side{};
  std::atomic<int> side_rr{0};
  bool inited = false;
};

DeviceStreams& device_streams(int idx) {
  static DeviceStreams s[8];
  static std::once_flag flags[8];
  if (idx < 0 || idx >= 8) throw std::runtime_error("zwt: device index out of range");
  std::call_once(flags[idx], [idx]() {
    cudaSetDevice(idx);
    cudaStreamCreateWithPriority(&s[idx].compute, cudaStreamNonBlocking, 0);
    cudaStreamCreateWithPriority(&s[idx].copy,    cudaStreamNonBlocking, -1);
    for (auto& ss : s[idx].side) {
      cudaStreamCreateWithPriority(&ss, cudaStreamNonBlocking, 0);
    }
    s[idx].inited = true;
  });
  return s[idx];
}

}  // namespace

void Stream::synchronize() const {
  if (!handle) return;
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(handle));
}

Event Event::create(Device d) {
  Event e;
  e.device = d;
  if (d.is_cuda()) {
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    e.handle = ev;
  }
  return e;
}

Event::~Event() { reset(); }

void Event::reset() {
  if (handle && device.is_cuda()) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(handle));
  }
  handle = nullptr;
}

void Event::record(Stream s) {
  if (!handle) return;
  cudaEventRecord(reinterpret_cast<cudaEvent_t>(handle),
                  reinterpret_cast<cudaStream_t>(s.handle));
}

void Event::wait(Stream s) const {
  if (!handle) return;
  cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(s.handle),
                      reinterpret_cast<cudaEvent_t>(handle), 0);
}

void Event::synchronize() const {
  if (!handle) return;
  cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(handle));
}

Stream compute_stream(Device dev) {
  if (dev.is_cpu()) return Stream{dev, nullptr};
  return Stream{dev, device_streams(dev.index).compute};
}

Stream copy_stream(Device dev) {
  if (dev.is_cpu()) return Stream{dev, nullptr};
  return Stream{dev, device_streams(dev.index).copy};
}

Stream side_stream(Device dev) {
  if (dev.is_cpu()) return Stream{dev, nullptr};
  auto& ds = device_streams(dev.index);
  int slot = ds.side_rr.fetch_add(1, std::memory_order_relaxed) % kSideStreamPool;
  return Stream{dev, ds.side[slot]};
}

#else  // CPU-only stubs

void Stream::synchronize() const {}

Event Event::create(Device d) { Event e; e.device = d; return e; }
Event::~Event() = default;
void Event::reset() { handle = nullptr; }
void Event::record(Stream)   {}
void Event::wait(Stream) const {}
void Event::synchronize() const {}

Stream compute_stream(Device dev) { return Stream{dev, nullptr}; }
Stream copy_stream(Device dev)    { return Stream{dev, nullptr}; }
Stream side_stream(Device dev)    { return Stream{dev, nullptr}; }

#endif

}  // namespace zwt
