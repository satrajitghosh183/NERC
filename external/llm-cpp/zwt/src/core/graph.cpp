#include "zwt/core/graph.hpp"

#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt {

GraphRunner::GraphRunner(Stream stream) : stream_(stream) {}

GraphRunner::~GraphRunner() {
#ifdef USE_CUDA
  if (instance_) {
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(instance_));
    instance_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
    graph_ = nullptr;
  }
#endif
}

void GraphRunner::capture(const std::function<void()>& fn) {
#ifdef USE_CUDA
  if (!stream_.device.is_cuda()) {
    cpu_fn_ = fn;
    cpu_fn_();
    return;
  }
  // Tear down any previous capture.
  if (instance_) {
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(instance_));
    instance_ = nullptr;
  }
  if (graph_) {
    cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
    graph_ = nullptr;
  }
  cudaStream_t s = static_cast<cudaStream_t>(stream_.handle);
  // ThreadLocal mode: only captures ops issued from this thread on this stream
  // and streams that fork from it. Safer than Global — NCCL + cuBLAS may
  // launch from their own worker threads which we don't want to capture.
  cudaError_t err = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) {
    throw std::runtime_error("GraphRunner: cudaStreamBeginCapture failed");
  }
  try {
    fn();
  } catch (...) {
    cudaGraph_t g;
    cudaStreamEndCapture(s, &g);
    if (g) cudaGraphDestroy(g);
    throw;
  }
  cudaGraph_t g;
  err = cudaStreamEndCapture(s, &g);
  if (err != cudaSuccess) {
    throw std::runtime_error("GraphRunner: cudaStreamEndCapture failed");
  }
  cudaGraphExec_t inst = nullptr;
  err = cudaGraphInstantiate(&inst, g, nullptr, nullptr, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(g);
    throw std::runtime_error("GraphRunner: cudaGraphInstantiate failed");
  }
  graph_    = g;
  instance_ = inst;
#else
  cpu_fn_ = fn;
  cpu_fn_();  // Run once so capture has observable side effects in CPU mode.
#endif
}

void GraphRunner::launch() {
#ifdef USE_CUDA
  if (!stream_.device.is_cuda()) {
    if (!cpu_fn_) throw std::runtime_error("GraphRunner: launch before capture");
    cpu_fn_();
    return;
  }
  if (!instance_) {
    throw std::runtime_error("GraphRunner: launch before capture");
  }
  cudaStream_t s = static_cast<cudaStream_t>(stream_.handle);
  cudaError_t err = cudaGraphLaunch(static_cast<cudaGraphExec_t>(instance_), s);
  if (err != cudaSuccess) {
    throw std::runtime_error("GraphRunner: cudaGraphLaunch failed");
  }
#else
  if (!cpu_fn_) throw std::runtime_error("GraphRunner: launch before capture");
  cpu_fn_();
#endif
}

}  // namespace zwt
