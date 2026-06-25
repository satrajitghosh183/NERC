// graph_bench — capture one training step as a CUDA graph and compare the
// replay cost to per-launch execution. The intent is to isolate the cost of
// kernel-launch overhead at the scale zwt runs at (O(10k) launches per step
// on a 1B model) and verify the expected speedup on small per-layer sizes.
//
// On CPU builds the tool still compiles (GraphRunner is a pass-through) but
// the measurement is meaningless — CPU paths don't have launch overhead.
//
// CLI:
//   --config=PATH     train config (model shape is read from it)
//   --warmup=N        warmup steps before timing (default 5)
//   --iters=N         timed steps in each phase (default 50)
//   --seq=N           sequence length override (defaults to cfg.model.max_seq)
//   --batch=N         batch size override (default 1)
//
// Output:
//   baseline_ms_per_step=X  graph_ms_per_step=Y  speedup=Y/X

#include "zwt/core/allocator.hpp"
#include "zwt/core/graph.hpp"
#include "zwt/core/stream.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/layers/module.hpp"
#include "zwt/layers/transformer.hpp"
#include "zwt/ops/elementwise.hpp"
#include "zwt/ops/xent.hpp"
#include "zwt/optim/adamw.hpp"
#include "zwt/train/config.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace zwt;

namespace {

struct CliArgs {
  std::string config_path;
  int64_t warmup = 5;
  int64_t iters  = 50;
  int64_t seq    = 0;
  int64_t batch  = 1;
};

CliArgs parse_cli(int argc, char** argv) {
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto eq = s.find('=');
    std::string k = s.substr(0, eq);
    std::string v = (eq == std::string::npos) ? std::string{}
                                              : s.substr(eq + 1);
    if      (k == "--config") a.config_path = v;
    else if (k == "--warmup") a.warmup = std::stoll(v);
    else if (k == "--iters")  a.iters  = std::stoll(v);
    else if (k == "--seq")    a.seq    = std::stoll(v);
    else if (k == "--batch")  a.batch  = std::stoll(v);
    else throw std::runtime_error("graph_bench: unknown arg: " + k);
  }
  if (a.config_path.empty()) {
    throw std::runtime_error("graph_bench: --config=PATH required");
  }
  return a;
}

double now_ms() {
  auto t = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

void sync(Device dev) {
#ifdef USE_CUDA
  if (dev.is_cuda()) cudaDeviceSynchronize();
#else
  (void)dev;
#endif
}

void fill_tokens(Tensor& t, int64_t vocab, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(0, vocab - 1);
  const size_t n = static_cast<size_t>(t.numel());
  std::vector<int64_t> host(n);
  for (size_t i = 0; i < n; ++i) host[i] = dist(rng);
#ifdef USE_CUDA
  if (t.device().is_cuda()) {
    cudaMemcpy(t.data(), host.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice);
    return;
  }
#endif
  std::memcpy(t.data(), host.data(), n * sizeof(int64_t));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CliArgs cli = parse_cli(argc, argv);
    train::TrainConfig cfg = train::load_train_config(cli.config_path);
    if (cli.seq == 0) cli.seq = cfg.model.max_seq;

#ifdef USE_CUDA
    Device dev = Device::cuda(0);
    DType  dtype = DType::BF16;
#else
    Device dev = Device::cpu();
    DType  dtype = DType::F32;
    std::fprintf(stderr,
        "graph_bench: CPU build — GraphRunner is a pass-through; timings "
        "measure CPU step cost, not launch overhead.\n");
#endif

    set_activation_arena_capacity(size_t(std::max<int64_t>(cfg.arena_mb, 256)) << 20);

    Transformer model(cfg.model, dtype, dev, cfg.init_seed);
    std::vector<Parameter*> params;
    model.collect_params(params);
    optim::AdamW opt(params, cfg.adamw);

    const int64_t B = cli.batch, S = cli.seq, V = cfg.model.vocab_size;

    // Fixed input / target buffers. Graph capture requires stable pointers;
    // each step copies fresh tokens into these same buffers.
    Tensor input  = empty({B, S},     DType::I64, dev);
    Tensor target = empty({B, S},     DType::I64, dev);
    fill_tokens(input,  V, 0xFEED);
    fill_tokens(target, V, 0xBEEF);

    auto run_step = [&]() {
      step_begin();
      Tensor logits = model.forward(input);
      Tensor logits_flat = logits.view({B * S, V});
      Tensor tgt_flat    = target.view({B * S});
      Tensor loss = empty_scratch({1}, DType::F32, dev);
      Tensor grad_logits = empty_scratch({B * S, V}, dtype, dev);
      ops::cross_entropy(logits_flat, tgt_flat, loss, &grad_logits, -100);
      model.backward(grad_logits.view(logits.shape()));
      opt.step();
      opt.zero_grad();
    };

    // Warmup to populate caches, instantiate any lazy state.
    for (int64_t i = 0; i < cli.warmup; ++i) run_step();
    sync(dev);

    // Phase 1: per-launch baseline.
    double t0 = now_ms();
    for (int64_t i = 0; i < cli.iters; ++i) run_step();
    sync(dev);
    double baseline_ms = (now_ms() - t0) / double(cli.iters);

    // Phase 2: captured + replayed. The capture invocation runs run_step once
    // under stream capture; subsequent launch() calls replay the graph.
    GraphRunner gr(compute_stream(dev));
    gr.capture(run_step);
    sync(dev);

    double t1 = now_ms();
    for (int64_t i = 0; i < cli.iters; ++i) gr.launch();
    sync(dev);
    double graph_ms = (now_ms() - t1) / double(cli.iters);

    std::printf("baseline_ms_per_step=%.3f graph_ms_per_step=%.3f speedup=%.2fx\n",
                baseline_ms, graph_ms,
                (graph_ms > 0 ? baseline_ms / graph_ms : 0.0));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "graph_bench: %s\n", e.what());
    return 1;
  }
}
