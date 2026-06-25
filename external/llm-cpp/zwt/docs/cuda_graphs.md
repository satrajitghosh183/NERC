# CUDA graph capture for training steps

At the scale zwt runs at (1B params, ~32 layers, bf16), one training step
issues on the order of 10⁴ kernel launches. On H100 each launch has a fixed
overhead of a few microseconds; at small per-layer micro-batch sizes that
overhead can rival the actual compute time. `cudaGraph` replay collapses the
whole step into a single `cudaGraphLaunch`.

## API

```cpp
#include "zwt/core/graph.hpp"

GraphRunner gr(compute_stream(dev));
gr.capture([&] {
  model.forward(fixed_input);      // writes to scratch arena
  model.backward(fixed_grad_in);
  opt.step();
  opt.zero_grad();
});

for (...) {
  copy_new_tokens_to(fixed_input);  // OUTSIDE the graph
  gr.launch();                      // one launch for the whole step
  if (step % log == 0) sync_and_read_loss();
}
```

## Contract for the captured closure

The graph is a static DAG of kernel launches with baked-in pointers. For
replay to be correct:

- **All tensor pointers the closure reads or writes must be stable.** The
  activation arena satisfies this trivially — it's a bump allocator, so for a
  fixed allocation sequence (which a fixed forward/backward graph gives you)
  the addresses are the same on every replay.
- **Input and target buffers must be fixed.** Allocate them once outside the
  closure and copy new data into them before each `launch()`. Do NOT
  reassign `input = loader.next()` — that changes the pointer.
- **No host syncs inside the closure.** `cudaMemcpyDeviceToHost`, `pull_scalar`,
  or any other device-to-host read will break capture. Push loss reads to
  outside the graph and gate them on the log interval.
- **No host branches based on device values.** If you need to skip steps
  based on NaN, use a scalar-on-device guard that the kernel itself checks.
- **Optimizer state is on-device — fine.** AdamW's `step_` counter is
  host-side and monotonic; each `launch()` is exactly one step, so the value
  the kernel sees stays consistent with its captured pointer.

## What's captured today

- `GraphRunner` itself — capture + replay wrapper.
- A bench tool `zwt_graph_bench` that builds a small model from a train
  config, fixes the I/O buffers, warms up, times `iters` per-launch steps,
  captures a step, replays `iters` times, and reports speedup.

What's NOT yet wired:

- Integration into `zwt_pretrain`. The training loop currently reads new
  input/target from `loader.next()` which allocates fresh Tensors. To enable
  graph replay there, the loader needs to write into caller-provided fixed
  buffers — a small refactor. When that lands, a `use_cuda_graph=1` config
  flag will gate replay on.

## Known limitations

- First-time capture cost is non-trivial (~100ms for a 1B model). Amortize
  across the run, don't capture per step.
- Graph must be re-captured if any shape changes. Keep batch/seq fixed once
  the graph is taken. Variable-length sequences need bucketing.
- NCCL collectives — CUDA-aware NCCL supports graph capture since NCCL 2.11,
  but only with `NCCL_GRAPH_REGISTER`. When TP/DDP land this needs revisiting.
