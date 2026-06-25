#pragma once

#include "zwt/core/stream.hpp"

#include <cstdint>
#include <functional>

namespace zwt {

// Thin wrapper around CUDA graph capture-and-replay. The target use case is
// capturing one full training step (forward + backward + optimizer step)
// once, then replaying the graph N times per epoch in place of N individual
// kernel launches. On a 1B model at seq=2048, launch overhead dominates at
// small per-layer batch sizes — graph replay collapses tens of thousands of
// cudaLaunchKernel calls per step into a single cudaGraphLaunch.
//
// Usage:
//   GraphRunner gr(compute_stream(dev));
//   gr.capture([&] {
//     model.forward(fixed_input);
//     model.backward(fixed_grad_in);
//     optimizer.step();
//   });
//   for (step in steps) {
//     copy_new_tokens_to(fixed_input);   // outside the graph
//     gr.launch();
//     sync_and_read_loss_if_logging();   // outside the graph
//   }
//
// Contract on the captured functor:
//   * ALL tensor pointers it touches must be stable across replays. Arena
//     allocations qualify as long as the allocation sequence is deterministic;
//     the bump-pointer returns the same base for the same sequence of sizes.
//   * Input/target buffers must be fixed — copy new data into them, don't
//     reassign pointers.
//   * No host syncs inside the captured region. No cudaMemcpyDeviceToHost.
//     No printf'd loss values. Pull those outside via cudaEventRecord.
//   * Optimizer state is fine (it's on-device). AdamW step_ counter is host-
//     side but only increments; as long as each launch does exactly one step,
//     replaying is equivalent to running.
//
// On CPU builds GraphRunner is a pass-through: capture() invokes the functor
// once to allow CPU-build smoke tests; launch() invokes it each call. This
// keeps the same control flow compilable everywhere.
class GraphRunner {
 public:
  explicit GraphRunner(Stream stream);
  ~GraphRunner();

  GraphRunner(const GraphRunner&) = delete;
  GraphRunner& operator=(const GraphRunner&) = delete;

  // Capture a functor. The functor is invoked once under the capture stream.
  // Must be called before the first launch(). Calling twice replaces the
  // captured graph (destroys the previous instance).
  void capture(const std::function<void()>& fn);

  // Replay the captured graph on the stream. Errors if called before capture.
  void launch();

  bool is_captured() const { return instance_ != nullptr || cpu_fn_ != nullptr; }

 private:
  Stream stream_;
  void*  graph_   = nullptr;   // cudaGraph_t
  void*  instance_ = nullptr;  // cudaGraphExec_t
  // CPU fallback: store the functor so launch() can replay it.
  std::function<void()> cpu_fn_;
};

}  // namespace zwt
