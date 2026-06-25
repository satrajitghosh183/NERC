// zwt_pretrain — production pretraining entry point for the Zero-Wait Trainer.
//
// Responsibilities:
//   * parse a TrainConfig from an INI file
//   * build the full Transformer on the selected device
//   * wire up TokenLoader (zero-stall) and fused AdamW
//   * run the step loop with gradient accumulation, cosine LR schedule, and
//     global gradient clipping
//   * capture the per-micro-batch fwd+bwd as a CUDA graph and replay it; the
//     optimizer step / clip / zero_grad run uncaptured (rare and host-touching)
//   * log loss / tok/s / LR at configurable intervals
//   * save/resume from binary checkpoints
//
// Usage:
//   zwt_pretrain <config.ini> [--resume <ckpt.bin>] [--metrics-csv <path>]
//                              [--dry-run] [--no-graph]
//
// Set ZWT_DISABLE_GRAPH=1 in the environment to fall back to the eager loop
// (per-launch fwd+bwd) without recompiling. Useful for debugging when the
// graph path masks an error.

#include "zwt/core/allocator.hpp"
#include "zwt/core/determinism.hpp"
#include "zwt/core/graph.hpp"
#include "zwt/core/profiler.hpp"
#include "zwt/core/stream.hpp"
#include "zwt/core/tensor.hpp"
#include "zwt/data/token_loader.hpp"
#include "zwt/dist/comm.hpp"
#include "zwt/dist/ddp.hpp"
#include "zwt/layers/module.hpp"
#include "zwt/layers/transformer.hpp"
#include "zwt/ops/elementwise.hpp"
#include "zwt/ops/xent.hpp"
#include "zwt/optim/adamw.hpp"
#include "zwt/optim/grad_clip.hpp"
#include "zwt/optim/lr_schedule.hpp"
#include "zwt/train/checkpoint.hpp"
#include "zwt/train/config.hpp"

#include <memory>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  std::string resume_override;
  std::string metrics_csv;
  std::string profile_csv;
  bool        dry_run  = false;
  bool        no_graph = false;
};

CliArgs parse_cli(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
        "usage: %s <config.ini> [--resume <ckpt.bin>] "
        "[--metrics-csv <path>] [--profile-csv <path>] "
        "[--dry-run] [--no-graph]\n", argv[0]);
    std::exit(2);
  }
  CliArgs a;
  a.config_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--resume" && i + 1 < argc) {
      a.resume_override = argv[++i];
    } else if (arg == "--metrics-csv" && i + 1 < argc) {
      a.metrics_csv = argv[++i];
    } else if (arg == "--profile-csv" && i + 1 < argc) {
      a.profile_csv = argv[++i];
    } else if (arg == "--dry-run") {
      a.dry_run = true;
    } else if (arg == "--no-graph") {
      a.no_graph = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      std::exit(2);
    }
  }
  return a;
}

// Pull a single float from a device scalar buffer. One host sync per call.
// Used only at logging intervals — never in the hot loop.
float pull_scalar(const Tensor& t) {
  float h = 0.f;
  if (t.device().is_cuda()) {
#ifdef USE_CUDA
    cudaMemcpy(&h, t.data(), sizeof(float), cudaMemcpyDeviceToHost);
#endif
  } else {
    h = *t.as<float>();
  }
  return h;
}

void apply_lr(optim::AdamW& opt, float lr) {
  opt.config().lr = lr;
}

int64_t count_params(const std::vector<Parameter*>& ps) {
  int64_t n = 0;
  for (auto* p : ps) n += p->numel();
  return n;
}

// D2D copy of one batch into a fixed-pointer destination on the compute
// stream. graph_input/graph_target must already be device-resident.
void copy_batch_into_graph(const Tensor& src, Tensor& dst, Device dev) {
#ifdef USE_CUDA
  if (dev.is_cuda()) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(compute_stream(dev).handle);
    cudaMemcpyAsync(dst.data(), src.data(), src.nbytes(),
                    cudaMemcpyDeviceToDevice, s);
    return;
  }
#endif
  std::memcpy(dst.data(), src.data(), src.nbytes());
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs cli = parse_cli(argc, argv);

  train::TrainConfig cfg = train::load_train_config(cli.config_path);
  if (!cli.resume_override.empty()) {
    cfg.resume_from = cli.resume_override;
  }

  // Profiling: --profile-csv <path> turns it on for the run regardless of
  // ZWT_PROFILE env var. Useful from the bench harness.
  if (!cli.profile_csv.empty()) {
    Profiler::get().set_enabled(true);
  }

  // Env-var overrides for DDP launch (set by launch_ddp.sh / torchrun-style
  // wrappers). These win over what's in the .conf so the same conf can
  // drive both single-GPU and 2-GPU runs.
  if (const char* v = std::getenv("RANK"))        cfg.rank        = std::atoi(v);
  if (const char* v = std::getenv("LOCAL_RANK"))  cfg.local_rank  = std::atoi(v);
  if (const char* v = std::getenv("WORLD_SIZE"))  cfg.world_size  = std::atoi(v);
  if (const char* v = std::getenv("MASTER_ADDR")) cfg.master_addr = v;
  if (const char* v = std::getenv("MASTER_PORT")) cfg.master_port = std::atoi(v);
  const bool ddp_on   = cfg.world_size > 1;
  const bool is_rank0 = (cfg.rank == 0);

  // Must run before the first cuBLAS handle is created — the workspace
  // config is consulted only at handle creation.
  if (cfg.deterministic) {
    set_deterministic(true);
    init_determinism_env();
    if (is_rank0) {
      std::fprintf(stderr, "determinism: ON (cuBLAS workspace :4096:8, "
                           "atomic-free reductions)\n");
    }
  }

#ifdef USE_CUDA
  // Pin the CUDA device to local_rank BEFORE any cuBLAS / cuda allocator
  // touch — otherwise rank 1 ends up sharing rank 0's context.
  if (cudaSetDevice(cfg.local_rank) != cudaSuccess) {
    std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
                 cfg.local_rank, cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  Device dev = Device::cuda(static_cast<int8_t>(cfg.local_rank));
  DType  param_dtype = DType::BF16;
#else
  Device dev = Device::cpu();
  DType  param_dtype = DType::F32;
#endif

  // Determinism + graph capture are mutually exclusive: cuBLAS may pick a
  // different algorithm under capture. We honor determinism over graph.
  // DDP also disables graph capture: overlapping NCCL allreduces with a
  // graph-captured backward is brittle in CUDA 12 and not worth the risk
  // for a first cut; eager replay under DDP is what we ship.
  bool graph_disabled = cli.no_graph
      || cfg.deterministic
      || ddp_on
      || (std::getenv("ZWT_DISABLE_GRAPH") != nullptr);

  std::fprintf(stderr, "=== zwt_pretrain ===\n");
  if (is_rank0) std::fprintf(stderr, "config: %s\n", cli.config_path.c_str());
  if (ddp_on) {
    std::fprintf(stderr,
        "DDP: rank=%d/%d local=%d master=%s:%d bucket_mb=%lld graph=off\n",
        cfg.rank, cfg.world_size, cfg.local_rank,
        cfg.master_addr.c_str(), cfg.master_port,
        (long long)cfg.bucket_mb);
  }
  if (is_rank0) std::fprintf(stderr, "device: %s  dtype: %s  graph: %s\n",
               dev.is_cuda() ? "cuda" : "cpu", dtype_name(param_dtype),
               graph_disabled ? "off" : "on");
  std::fprintf(stderr, "model:  vocab=%lld d_model=%lld n_layers=%lld heads=%lld d_ffn=%lld seq=%lld\n",
               (long long)cfg.model.vocab_size,
               (long long)cfg.model.d_model,
               (long long)cfg.model.n_layers,
               (long long)cfg.model.n_heads,
               (long long)cfg.model.d_ffn,
               (long long)cfg.model.max_seq);
  std::fprintf(stderr, "data:   %s  seq=%lld batch=%lld grad_accum=%lld\n",
               cfg.data_path.c_str(),
               (long long)cfg.seq_len,
               (long long)cfg.batch_size,
               (long long)cfg.grad_accum);

  size_t arena_bytes = static_cast<size_t>(std::max<int64_t>(cfg.arena_mb, 256)) << 20;
  set_activation_arena_capacity(arena_bytes);
  std::fprintf(stderr, "arena:  %zu MiB\n", arena_bytes >> 20);

  Transformer model(cfg.model, param_dtype, dev, cfg.init_seed);
  std::vector<Parameter*> params;
  model.collect_params(params);
  std::fprintf(stderr, "params: %lld (%.2f M)\n",
               (long long)count_params(params),
               double(count_params(params)) / 1e6);

  optim::AdamW opt(params, cfg.adamw);

  // ── DDP construction (only when world_size > 1) ─────────────────────────
  // CommContext owns the NCCL handle; BucketManager owns the per-bucket
  // staging buffers; OverlapHookup owns the bucket_done event chain and
  // wires BucketManager::set_allreduce to do the comm-stream allreduce.
  std::unique_ptr<dist::BucketManager> ddp_mgr;
  std::unique_ptr<dist::OverlapHookup> ddp_overlap;
  dist::CommContext                    ddp_ctx{};  // backend == nullptr means inactive
  if (ddp_on) {
    ddp_ctx = dist::make_nccl_ctx(cfg.rank, cfg.world_size,
                                  cfg.master_addr, cfg.master_port,
                                  cfg.local_rank, dev);
    const size_t bucket_bytes =
        static_cast<size_t>(std::max<int64_t>(cfg.bucket_mb, 1)) << 20;
    ddp_mgr = std::make_unique<dist::BucketManager>(params, bucket_bytes,
                                                    cfg.world_size);
    ddp_overlap = std::make_unique<dist::OverlapHookup>(*ddp_mgr, ddp_ctx);
    if (is_rank0) {
      std::fprintf(stderr, "DDP: %d buckets, bucket_mb=%lld, world=%d\n",
                   ddp_mgr->num_buckets(),
                   (long long)cfg.bucket_mb, cfg.world_size);
    }
  }

  int64_t  resume_step = 0;
  int64_t  resume_cursor = 0;
  if (!cfg.resume_from.empty()) {
    std::fprintf(stderr, "resuming from %s\n", cfg.resume_from.c_str());
    train::CheckpointMeta m = train::load_checkpoint(cfg.resume_from, params, opt);
    resume_step   = m.step;
    resume_cursor = m.data_cursor;
    std::fprintf(stderr, "  resumed at step %lld, cursor %lld\n",
                 (long long)resume_step, (long long)resume_cursor);
  }

  // Build the cached gradient clipper. One device-side allocation pair (ptrs +
  // sizes) and a few scalars; reused every step. Capture-safe — no host syncs.
  optim::GradClipper clipper(params);

  data::TokenLoader::Options dopts;
  dopts.path         = cfg.data_path;
  dopts.seq_len      = cfg.seq_len;
  dopts.batch_size   = cfg.batch_size;
  dopts.shuffle      = cfg.shuffle;
  // Each rank gets a different shuffle seed so no two ranks consume the
  // same micro-batches in a step. Effective global batch is
  // batch_size * grad_accum * world_size.
  dopts.seed         = cfg.data_seed
      ^ (static_cast<uint64_t>(cfg.rank) * 0x9E3779B185EBCA87ULL);
  dopts.device       = dev;
  dopts.start_cursor = resume_cursor;
  data::TokenLoader loader(dopts);
  loader.start();
  std::fprintf(stderr, "loader: %lld chunks, %lld steps/epoch\n",
               (long long)((loader.steps_per_epoch() * cfg.batch_size)),
               (long long)loader.steps_per_epoch());

  if (cli.dry_run) {
    std::fprintf(stderr, "dry-run: exiting before step loop\n");
    return 0;
  }

  std::FILE* metrics_fp = nullptr;
  if (!cli.metrics_csv.empty()) {
    const bool fresh = cli.resume_override.empty();
    metrics_fp = std::fopen(cli.metrics_csv.c_str(), fresh ? "w" : "a");
    if (!metrics_fp) {
      std::fprintf(stderr, "warning: could not open %s for metrics\n",
                   cli.metrics_csv.c_str());
    } else if (fresh) {
      std::fprintf(metrics_fp,
          "step,loss,lr,grad_norm,tokens_seen,wall_secs,tok_per_s\n");
      std::fflush(metrics_fp);
    }
  }

  const int64_t batch = cfg.batch_size;
  const int64_t seq   = cfg.seq_len;
  const int64_t vocab = cfg.model.vocab_size;
  const int64_t max_steps = cfg.max_steps;

  // Fixed input/target buffers consumed by the captured graph. The loader
  // produces tokens into its own ring buffers; we D2D-copy into these for
  // every micro-batch so the captured kernels always read from the same
  // pointer.
  Tensor graph_input  = empty({batch, seq}, DType::I64, dev);
  Tensor graph_target = empty({batch, seq}, DType::I64, dev);

  // Persistent loss buffer. Stable address across replays; pulled to host
  // only on log steps.
  Tensor loss_buf = empty({1}, DType::F32, dev);

  // The captured functor: one micro-batch fwd + bwd. Loss-grad scaling lives
  // here too so 1/grad_accum applies inside the captured kernels, not as a
  // separate uncaptured launch.
  //
  // When DDP is active, the LAST micro-batch of each grad_accum window calls
  // the DDP-aware backward (signals mark_ready, gathers + fires allreduce on
  // the side stream as buckets complete). Earlier micro-batches just
  // accumulate grads locally — equivalent to torch.nn.parallel.DistributedDataParallel
  // wrapped in `model.no_sync()` for the non-final passes.
  StreamHandle compute_handle = compute_stream(dev).handle;
  auto run_micro_batch = [&](bool fire_ddp) {
    step_begin();  // reset arena — deterministic offsets so the bump pointer
                   // hands out the same addresses on every replay.
    Tensor logits;
    {
      ZWT_PROFILE_GPU("step.forward", dev);
      logits = model.forward(graph_input);
    }
    Tensor logits_flat = logits.view({batch * seq, vocab});
    Tensor tgt_flat    = graph_target.view({batch * seq});
    Tensor grad_logits = empty_scratch({batch * seq, vocab}, param_dtype, dev);
    {
      ZWT_PROFILE_GPU("step.cross_entropy", dev);
      ops::cross_entropy(logits_flat, tgt_flat, loss_buf, &grad_logits, /*ignore=*/-100);
      if (cfg.grad_accum > 1) {
        ops::scale(grad_logits, 1.0f / float(cfg.grad_accum));
      }
    }
    {
      ZWT_PROFILE_GPU("step.backward", dev);
      if (fire_ddp && ddp_mgr) {
        model.backward(grad_logits.view(logits.shape()), *ddp_mgr, compute_handle);
      } else {
        model.backward(grad_logits.view(logits.shape()));
      }
    }
  };
  // Existing graph-capture path uses the no-DDP variant. Graph capture is
  // disabled when world_size > 1 (see graph_disabled init above).
  auto run_step = [&]() { run_micro_batch(/*fire_ddp=*/false); };

  // Warmup: run a few uncaptured iterations so cuBLAS handle init, lazy
  // workspace allocation, and one-shot kernel JIT all happen *before* we
  // start a stream-capture region (where allocations are forbidden). We
  // need real tokens, so we drain warmup batches from the loader and
  // discard the gradients afterwards.
  GraphRunner gr(compute_stream(dev));
  if (!graph_disabled) {
    constexpr int kWarmupIters = 3;
    for (int i = 0; i < kWarmupIters; ++i) {
      auto bd = loader.next();
      copy_batch_into_graph(bd.input,  graph_input,  dev);
      copy_batch_into_graph(bd.target, graph_target, dev);
      run_step();
    }
    opt.zero_grad();
#ifdef USE_CUDA
    if (dev.is_cuda()) cudaDeviceSynchronize();
#endif

    try {
      gr.capture(run_step);
    } catch (const std::exception& e) {
      std::fprintf(stderr,
          "graph capture failed (%s) — falling back to eager replay\n",
          e.what());
      graph_disabled = true;
    }
#ifdef USE_CUDA
    if (dev.is_cuda()) cudaDeviceSynchronize();
#endif
    if (!graph_disabled) {
      std::fprintf(stderr,
          "graph: captured fwd+bwd; replaying %lld micro-batch(es) per step\n",
          (long long)cfg.grad_accum);
    }
  }

  auto t_start = std::chrono::steady_clock::now();
  int64_t tokens_seen = 0;
  float last_loss = 0.f;

  for (int64_t step = resume_step + 1; step <= max_steps; ++step) {
    ZWT_PROFILE_GPU("step.total", dev);
    float lr = cfg.schedule.lr_at(step);
    apply_lr(opt, lr);

    {
      ZWT_PROFILE_GPU("step.zero_grad", dev);
      opt.zero_grad();
    }

    const bool log_this_step = (step % cfg.log_interval == 0) || (step == 1);
    float accum_loss = 0.f;
    int   accum_count = 0;

    for (int64_t acc = 0; acc < cfg.grad_accum; ++acc) {
      data::TokenLoader::Batch batch_data;
      {
        ZWT_PROFILE_CPU("step.data_load");
        batch_data = loader.next();
      }
      {
        ZWT_PROFILE_GPU("step.h2d_copy", dev);
        copy_batch_into_graph(batch_data.input,  graph_input,  dev);
        copy_batch_into_graph(batch_data.target, graph_target, dev);
      }

      const bool is_last = (acc == cfg.grad_accum - 1);
      if (graph_disabled) {
        run_micro_batch(/*fire_ddp=*/ddp_on && is_last);
      } else {
        ZWT_PROFILE_GPU("step.graph_replay", dev);
        gr.launch();
      }

      // Loss D2H is gated on log steps only — pulling per step would defeat
      // the point of graph replay.
      if (log_this_step) {
        accum_loss += pull_scalar(loss_buf);
        ++accum_count;
      }
      tokens_seen += batch * seq;
    }
    if (ddp_overlap) {
      // Make compute stream wait on every bucket_done event, then scatter
      // the (already averaged via ncclAvg) bucket buffers back into each
      // Parameter::grad. Optimizer step below sees fully-reduced grads.
      ZWT_PROFILE_GPU("step.ddp_sync", dev);
      ddp_overlap->sync_and_finalize();
    }
    if (log_this_step) {
      last_loss = accum_loss / float(std::max(accum_count, 1));
    }

    // Clip + step run OUTSIDE the captured region. clipper.clip() launches a
    // tiny three-kernel sequence (zero, sumsq, scale-from-device) and writes
    // the on-device norm into clipper.norm_dev_buf(); we only pull it on log
    // steps.
    if (cfg.grad_clip > 0.f) {
      ZWT_PROFILE_GPU("step.grad_clip", dev);
      clipper.clip(cfg.grad_clip);
    }
    {
      ZWT_PROFILE_GPU("step.optim", dev);
      opt.step();
    }

    if (log_this_step && is_rank0) {
      float gnorm = (cfg.grad_clip > 0.f) ? clipper.pull_last_norm() : 0.f;
      auto t_now = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(t_now - t_start).count();
      // tokens_seen is per-rank; multiply by world_size at the print site
      // so the headline matches the bench formula B*S*grad_accum*world*steps.
      double tps = (static_cast<double>(tokens_seen)
                    * static_cast<double>(cfg.world_size))
                   / std::max(secs, 1e-9);
      std::fprintf(stderr,
          "step %6lld  loss %.4f  lr %.2e  |g| %.3f  %.0f tok/s\n",
          (long long)step, last_loss, lr, gnorm, tps);
      if (metrics_fp) {
        std::fprintf(metrics_fp,
            "%lld,%.6f,%.6e,%.6f,%lld,%.3f,%.3f\n",
            (long long)step, last_loss, lr, gnorm,
            (long long)(tokens_seen * cfg.world_size), secs, tps);
        std::fflush(metrics_fp);
      }
      // Profiler CSV dump at every log step on rank 0. The CSV accumulates
      // one row per (step, stage); the bench harness consumes it directly.
      if (Profiler::get().enabled() && !cli.profile_csv.empty()) {
        Profiler::get().dump_csv(cli.profile_csv, step);
        Profiler::get().reset();   // start fresh for the next interval
      }
    }

    if (cfg.ckpt_interval > 0 && step % cfg.ckpt_interval == 0 && is_rank0) {
      train::CheckpointMeta meta;
      meta.step        = step;
      meta.seed        = cfg.init_seed;
      meta.data_cursor = loader.cursor();
      meta.lr          = lr;
      meta.loss        = last_loss;
      std::fprintf(stderr, "  writing ckpt %s (step %lld)\n",
                   cfg.ckpt_path.c_str(), (long long)step);
      train::save_checkpoint(cfg.ckpt_path, params, opt, meta);
    }
  }

#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif

  if (!cfg.ckpt_path.empty() && is_rank0) {
    train::CheckpointMeta meta;
    meta.step        = max_steps;
    meta.seed        = cfg.init_seed;
    meta.data_cursor = loader.cursor();
    meta.lr          = cfg.schedule.lr_at(max_steps);
    meta.loss        = last_loss;
    std::fprintf(stderr, "writing final ckpt %s\n", cfg.ckpt_path.c_str());
    train::save_checkpoint(cfg.ckpt_path, params, opt, meta);
  }
  if (metrics_fp) std::fclose(metrics_fp);

  // End-of-run profiler summary (rank 0 only — every other rank has the
  // same numbers ± noise).
  if (Profiler::get().enabled() && is_rank0) {
    Profiler::get().print_summary(stderr);
  }

  if (ddp_on) {
    // Tear down NCCL before the process exits so peers see a clean abort.
    ddp_overlap.reset();
    ddp_mgr.reset();
    dist::nccl_destroy(ddp_ctx);
  }
  if (is_rank0) std::fprintf(stderr, "done.\n");
  return 0;
}
