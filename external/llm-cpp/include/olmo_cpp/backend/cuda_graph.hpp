#pragma once

/**
 * include/olmo_cpp/backend/cuda_graph.hpp
 *
 * Header-only RAII helper around at::cuda::CUDAGraph. CUDA Graphs let
 * you record a sequence of kernel launches once and replay it as a
 * single op, eliminating per-launch CPU overhead (~5-10 us per launch
 * on H100; with ~100 launches per decode step that is meaningful).
 *
 * The runner expects a function fn(input)->output that always launches
 * the same kernels in the same order with the same shapes. On the first
 * call it does:
 *   1) a warmup call (CUDA refuses to capture if kernels haven't been
 *      JIT-compiled / autotuned),
 *   2) clone the input into a static input_buffer_,
 *   3) replay the function inside graph_.capture_begin/end.
 * Subsequent calls copy the new input into the static buffer and call
 * graph_.replay() — no kernel launch overhead, no allocation.
 *
 * --- Includes from this project ---
 *   - (none — pulls only torch headers)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep.
 *
 * --- Role in training pipeline ---
 *   This is wiring for an upcoming inference / static-shape training
 *   optimization. In Python OLMo the equivalent is torch.cuda.graphs;
 *   we keep the same semantics so a forward call can be replayed
 *   instead of relaunched once shapes stabilize.
 */

#include <torch/torch.h>
#include <functional>
#include <vector>

namespace olmo_cpp {

/// CUDA Graph capture for eliminating kernel launch overhead.
///
/// On H100, each CUDA kernel launch costs ~5-10μs. A single transformer
/// decode step launches 100+ kernels. With CUDA graphs, the entire step
/// is replayed from a single graph launch, cutting overhead by 10-30%.
///
/// Usage:
///   CUDAGraphRunner runner;
///   // First call captures the graph
///   auto out = runner.run(input, [&](torch::Tensor x) {
///     return model->forward(x);
///   });
///   // Subsequent calls replay the captured graph (must match input shape)
///   auto out2 = runner.run(input2, ...);  // replays, no launch overhead
class CUDAGraphRunner {
 public:
  CUDAGraphRunner() = default;
  ~CUDAGraphRunner() = default;

  // Non-copyable
  CUDAGraphRunner(const CUDAGraphRunner&) = delete;
  CUDAGraphRunner& operator=(const CUDAGraphRunner&) = delete;

  /// Run a function, capturing it as a CUDA graph on first call.
  /// Subsequent calls with the same input shape replay the graph.
  /// Falls back to direct execution on non-CUDA devices.
  torch::Tensor run(torch::Tensor input,
                    std::function<torch::Tensor(torch::Tensor)> fn) {
    if (!input.is_cuda()) {
      return fn(input);  // No graphs on CPU/MPS
    }

#ifdef TORCH_CUDA_AVAILABLE
    // Check if we need to (re)capture
    bool need_capture = !captured_;
    if (captured_) {
      // Shape changed — need to recapture
      if (input.sizes() != input_buffer_.sizes()) {
        need_capture = true;
      }
    }

    if (need_capture) {
      capture(input, fn);
    }

    // Copy input data into the captured input buffer
    input_buffer_.copy_(input);

    // Replay the graph
    graph_.replay();

    return output_buffer_;
#else
    return fn(input);
#endif
  }

  /// Check if a graph has been captured
  bool is_captured() const { return captured_; }

  /// Reset the captured graph (forces recapture on next run)
  void reset() {
    captured_ = false;
    input_buffer_ = torch::Tensor();
    output_buffer_ = torch::Tensor();
  }

 private:
#ifdef TORCH_CUDA_AVAILABLE
  void capture(torch::Tensor input,
               std::function<torch::Tensor(torch::Tensor)>& fn) {
    // Warmup run (CUDA needs to see the kernels before capturing)
    auto warmup_out = fn(input);

    // Allocate static buffers
    input_buffer_ = input.clone();

    // Capture
    graph_ = at::cuda::CUDAGraph();
    graph_.capture_begin();
    output_buffer_ = fn(input_buffer_);
    graph_.capture_end();

    captured_ = true;
  }

  at::cuda::CUDAGraph graph_;
#endif

  torch::Tensor input_buffer_;
  torch::Tensor output_buffer_;
  bool captured_ = false;
};

}  // namespace olmo_cpp
