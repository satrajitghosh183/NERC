/**
 * src/train.cpp
 *
 * The full training loop. This is THE file to read if you want to know
 * what one training step actually does in this framework. It owns:
 *   - the data loader (TokenDataset / DataLoader),
 *   - the optimizer (AdamW / Muon / Lion / Dion / SGP / SkipStep wrappers),
 *   - the LR schedule (warmup + cosine, or others),
 *   - autocast / pure-BF16 weight casting,
 *   - DDP allreduce (when built with Gloo),
 *   - activation checkpointing,
 *   - the always-on profiler scopes (step_total / data_loading /
 *     forward / backward / allreduce / optimizer_step),
 *   - heartbeat-file emission for long runs,
 *   - eval and periodic checkpointing.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/train.hpp                : TrainConfig + train() entry signature.
 *   - olmo_cpp/data/token_dataset.hpp   : the .npy-backed token tensor.
 *   - olmo_cpp/profiler.hpp             : ProfileScope used to time every stage.
 *   - olmo_cpp/distributed/ddp.hpp      : gradient allreduce.
 *   - olmo_cpp/optim/{adamw,muon,lion,dion,sgp,skip_step}.hpp : every optimizer the loop can pick.
 *   - olmo_cpp/train/grad_scaler.hpp    : FP16 dynamic loss scaling.
 *   - olmo_cpp/train/async_loss_reader.hpp : non-blocking loss readout (AA).
 *   - olmo_cpp/train/activation_checkpoint.hpp : per-block recompute.
 *   - olmo_cpp/train/checkpoint.hpp     : sharded checkpoint manager.
 *   - olmo_cpp/eval/evaluator.hpp       : periodic validation loss.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: the only caller. After parsing the .conf and building
 *     a Transformer or FusedTransformer, main() invokes
 *     olmo_cpp::train(model, cfg, train_cfg, device, callbacks).
 *
 * --- Role in training pipeline ---
 *   This is the orchestrator. It pulls one batch, runs the forward,
 *   computes the loss, runs the backward, optionally allreduces, clips
 *   gradients, and steps the optimizer — wrapping each phase in a
 *   ProfileScope so the user gets a per-stage timing table at the end.
 */
// Full-featured training loop with callbacks, multiple optimizers,
// LR schedulers, activation checkpointing, gradient scaling, and eval
#include "olmo_cpp/train.hpp"
#include "olmo_cpp/train/async_loss_reader.hpp"
#include "olmo_cpp/data/token_dataset.hpp"
#include "olmo_cpp/profiler.hpp"
#include <ATen/autocast_mode.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include "olmo_cpp/distributed/ddp.hpp"
#include "olmo_cpp/distributed/fsdp.hpp"
#include "olmo_cpp/distributed/zero1.hpp"
#include "olmo_cpp/optim/lion.hpp"
#include "olmo_cpp/optim/muon.hpp"
#include "olmo_cpp/optim/dion.hpp"
#include "olmo_cpp/optim/foreach_adamw.hpp"
#include "olmo_cpp/optim/eight_bit_adamw.hpp"
#include "olmo_cpp/optim/grad_clip.hpp"
#include "olmo_cpp/optim/skip_step.hpp"
#include "olmo_cpp/optim/scheduler.hpp"
#include "olmo_cpp/optim/sgp.hpp"
#include "olmo_cpp/optim/sgp_v2.hpp"
#include "olmo_cpp/train/callback.hpp"
#include "olmo_cpp/train/grad_scaler.hpp"
#include "olmo_cpp/train/activation_checkpoint.hpp"
#include "olmo_cpp/train/checkpoint.hpp"
#include "olmo_cpp/eval/evaluator.hpp"
#include "olmo_cpp/io/filesystem.hpp"
#include <torch/nn/init.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <fstream>

namespace olmo_cpp {

// Write current epoch count + timestamp to heartbeat file (atomic via rename)
static void write_heartbeat(const std::string& path, int64_t epoch, int64_t step, int64_t total_steps) {
  if (path.empty()) return;
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::trunc);
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    f << "epoch=" << epoch << "\n"
      << "step=" << step << "/" << total_steps << "\n"
      << "time=" << std::ctime(&t);  // ctime adds newline
  }
  std::rename(tmp.c_str(), path.c_str());
}

// AutocastGuard — shared header, portable across all PyTorch 2.x versions
#include "olmo_cpp/train/autocast_guard.hpp"
using olmo_cpp::AutocastGuard;

namespace {

double cosine_warmup_lr(int64_t step, int64_t warmup_steps, double base_lr, int64_t total_steps) {
  if (step < warmup_steps) {
    return base_lr * static_cast<double>(step + 1) / static_cast<double>(warmup_steps);
  }
  double progress = static_cast<double>(step - warmup_steps) / static_cast<double>(total_steps - warmup_steps);
  return 0.5 * base_lr * (1.0 + std::cos(M_PI * progress));
}

std::string optimizer_display_name(const std::string& name, bool use_foreach) {
  if (name == "adamw" && use_foreach) return "ForeachAdamW (batched _foreach_* ops)";
  if (name == "adamw") return "AdamW (standard per-param)";
  return name;
}

void print_optimization_banner(const std::string& optimizer_name, bool use_foreach,
                               bool gpu_data, bool gpu_data_active,
                               bool use_amp, bool fused_grad_clip,
                               bool cuda_graph = false, bool use_bf16 = false) {
  const char* prec = use_bf16 ? "BF16 (pure, master weights BF16)"
                   : use_amp  ? "BF16 autocast"
                              : "FP32";
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║  OPTIMIZATION STATUS                                    ║\n";
  std::cout << "╠══════════════════════════════════════════════════════════╣\n";
  std::cout << "║  Optimizer:       " << std::left << std::setw(39)
            << optimizer_display_name(optimizer_name, use_foreach) << "║\n";
  std::cout << "║  Grad clipping:   " << std::left << std::setw(39)
            << "GPU-resident (foreach_norm, no D2H)" << "║\n";
  std::cout << "║  Data loading:    " << std::left << std::setw(39)
            << (gpu_data_active ? "GPU-resident (zero per-step H2D)" : "CPU + async prefetch") << "║\n";
  std::cout << "║  Mixed precision: " << std::left << std::setw(39)
            << prec << "║\n";
  std::cout << "║  CUDA graph:      " << std::left << std::setw(39)
            << (cuda_graph ? "ON (fwd+bwd captured)" : "OFF") << "║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";
  std::cout << "\n";
}

}  // namespace

// ---------------------------------------------------------------------------
// Legacy train_epoch (backward compatible)
// ---------------------------------------------------------------------------

void train_epoch(
    Transformer& model,
    const TransformerConfig& cfg,
    int64_t num_steps,
    std::optional<std::string> data_path,
    int64_t batch_size,
    int64_t seq_len,
    double lr,
    int64_t warmup_steps,
    torch::Device device,
    int64_t grad_accum_steps,
    bool use_amp,
    const std::string& optimizer_name) {
  model->train();

  // Create optimizer based on selection
  std::unique_ptr<torch::optim::Optimizer> opt;
  if (optimizer_name == "muon") {
    opt = std::make_unique<Muon>(model->parameters(), MuonOptions(lr));
  } else if (optimizer_name == "lion") {
    opt = std::make_unique<Lion>(model->parameters(), LionOptions(lr).weight_decay(0.01));
  } else if (optimizer_name == "dion") {
    opt = std::make_unique<DION>(model->parameters(), DIONOptions(lr).weight_decay(0.01));
  } else {
    opt = std::make_unique<ForeachAdamW>(
        model->parameters(), ForeachAdamWOptions(lr).weight_decay(0.01));
  }
  auto& optimizer = *opt;

  auto ddp = DDPContext::init_from_env();
  if (ddp && ddp->is_distributed()) {
    auto params = model->parameters();
    ddp->broadcast_parameters(params);
  }

  std::optional<TokenDataset> dataset;
  bool gpu_data_active = false;
  if (data_path && !data_path->empty()) {
    dataset.emplace(*data_path, seq_len, true);
    dataset->reset_epoch();
    dataset->to_device(device, 0);
    gpu_data_active = dataset->is_gpu_resident();
  }

  if (!ddp || ddp->rank() == 0) {
    print_optimization_banner(optimizer_name, true, true, gpu_data_active, use_amp, true);
  }

  // Pre-build DDP parameter list once (avoids per-step allocation)
  std::vector<torch::Tensor> ddp_params;
  if (ddp && ddp->is_distributed()) {
    for (auto& p : model->parameters()) ddp_params.push_back(p);
    // T-1: register autograd hooks so per-bucket allreduce overlaps with
    // backward. allreduce_gradients() at end-of-step becomes a finalize
    // (wait + divide). For grad_accum > 1 the driver controls hook
    // dispatch via set_sync_required() — disabled on non-final accum
    // steps so we only collective the final summed gradient.
    ddp->register_grad_hooks(ddp_params);
  }

  // Epoch tracking
  int64_t tokens_per_step = batch_size * seq_len * grad_accum_steps;
  int64_t dataset_tokens = dataset ? dataset->size() * seq_len : 0;
  int64_t steps_per_epoch = (dataset && dataset_tokens > 0)
      ? std::max(int64_t(1), dataset_tokens / tokens_per_step) : num_steps;
  int64_t current_epoch = 0;

  auto train_start = std::chrono::steady_clock::now();
  int64_t total_tokens = 0;
  double epoch_loss_sum = 0.0;
  int64_t epoch_loss_count = 0;

  // Pre-allocate loss accumulator (avoids per-step allocation)
  torch::Tensor accum_loss_tensor = torch::zeros({}, torch::TensorOptions().device(device));

  // Cache model params once (avoids per-step vector allocation)
  auto model_params = model->parameters();

  for (int64_t step = 0; step < num_steps; ++step) {
    auto step_start = std::chrono::steady_clock::now();
    double step_lr = cosine_warmup_lr(step, warmup_steps, lr, num_steps);
    optimizer.param_groups()[0].options().set_lr(step_lr);

    // Epoch boundary detection
    int64_t new_epoch = dataset ? (step / steps_per_epoch) : 0;
    if (new_epoch > current_epoch && (!ddp || ddp->rank() == 0)) {
      double avg_epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : 0.0;
      std::cout << "--- Epoch " << current_epoch << " complete | avg_loss: "
                << std::fixed << std::setprecision(4) << avg_epoch_loss << " ---" << std::endl;
      epoch_loss_sum = 0.0;
      epoch_loss_count = 0;
      current_epoch = new_epoch;
    }

    {
      ProfileScope step_scope("step_total");
      // set_to_none=true: use nullptr instead of memset (faster)
      optimizer.zero_grad(true);
      accum_loss_tensor.zero_();

      for (int64_t accum = 0; accum < grad_accum_steps; ++accum) {
        torch::Tensor input, labels;
        {
          ProfileScope data_scope("data_loading");
          if (dataset) {
            auto [in, lab] = dataset->get_batch(batch_size, device);
            input = in; labels = lab;
            dataset->prefetch_next(batch_size, device);
          } else {
            input = torch::randint(0, cfg.vocab_size, {batch_size, seq_len},
                                   torch::TensorOptions().dtype(torch::kLong).device(device));
            labels = torch::randint(0, cfg.vocab_size, {batch_size, seq_len},
                                    torch::TensorOptions().dtype(torch::kLong).device(device));
          }
        }

        torch::Tensor loss;
        {
          ProfileScope fwd_scope("forward");
          AutocastGuard ac(use_amp, device);
          loss = model->forward(input, labels, -100) / static_cast<float>(grad_accum_steps);
        }
        {
          ProfileScope bwd_scope("backward");
          loss.backward();
        }
        accum_loss_tensor.add_(loss.detach());
      }

      {
        ProfileScope allreduce_scope("allreduce");
        if (ddp && ddp->is_distributed()) {
          ddp->allreduce_gradients(ddp_params);
        }
      }

      {
        ProfileScope optim_scope("optimizer_step");
        clip_grad_norm_gpu(model_params, 1.0);
        optimizer.step();
      }

      total_tokens += tokens_per_step;

      // AA: non-blocking loss readout. Queue every step (cheap async copy
      // on a side stream); read whatever's ready at log time.
      static thread_local AsyncLossReader _loss_reader_a;
      _loss_reader_a.queue(accum_loss_tensor);
      if (step % 10 == 0 && (!ddp || ddp->rank() == 0)) {
        float accum_loss = _loss_reader_a.poll();
        epoch_loss_sum += accum_loss;
        epoch_loss_count++;
        auto step_end = std::chrono::steady_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        double elapsed_s = std::chrono::duration<double>(step_end - train_start).count();
        double tok_per_s = total_tokens / (elapsed_s > 0 ? elapsed_s : 1);
        double eta_s = (num_steps - step) * (elapsed_s / (step > 0 ? step : 1));

        std::cout << "Epoch " << current_epoch
                  << " | Step " << step << "/" << num_steps
                  << "  loss: " << std::fixed << std::setprecision(4) << accum_loss
                  << "  lr: " << std::scientific << std::setprecision(2) << step_lr
                  << "  step_ms: " << static_cast<int>(step_ms)
                  << "  tok/s: " << static_cast<int>(tok_per_s)
                  << "  ETA: " << static_cast<int>(eta_s / 60) << "m"
                  << std::endl;
      }
    }  // end step_scope
  }

  if (!ddp || ddp->rank() == 0) {
    auto end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(end - train_start).count();
    double avg_step_ms = total_s / num_steps * 1000.0;
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "  Steps: " << num_steps << " (" << current_epoch + 1 << " epochs)" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Wall time: " << std::fixed << std::setprecision(2) << total_s << "s" << std::endl;
    std::cout << "  Avg step: " << std::fixed << std::setprecision(1) << avg_step_ms << "ms" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(total_tokens / total_s) << " tok/s" << std::endl;
    std::cout << "========================" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Full-featured train() with all infrastructure
// ---------------------------------------------------------------------------

void train(
    Transformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks,
    std::shared_ptr<Evaluator> evaluator) {

  model->train();

  // ---- Initialize DDP ----
  auto ddp = DDPContext::init_from_env();
  if (ddp && ddp->is_distributed()) {
    auto params = model->parameters();
    ddp->broadcast_parameters(params);
  }
  int rank = ddp ? ddp->rank() : 0;

  // ---- ZeRO-1: partition optimizer state across DP ranks ----
  // Each rank only constructs its optimizer over its share of the
  // parameters (1/world_size). After every step() we allgather the
  // updated weights so every replica sees identical params again.
  // Single-rank / DDP-off: sharder is null, opt_params == all_params.
  std::unique_ptr<OptimizerStateSharder> zero1;
  if (cfg.use_zero1 && ddp && ddp->is_distributed() && !cfg.use_fsdp) {
    zero1 = std::make_unique<OptimizerStateSharder>(
        ddp->backend(), ddp->rank(), ddp->world_size());
  }

  // ---- FSDP (ZeRO-3): shard params + grads + optimizer state across ranks ----
  // shard_params() must run BEFORE the optimizer is built so the optimizer (and
  // its moment state) are sized to the local 1/world_size shard. The full param
  // is reconstructed (unshard) each step for fwd/bwd, then re-sharded.
  std::optional<FSDPContext> fsdp;
#if defined(OLMO_HAS_NCCL) || defined(OLMO_HAS_DDP)
  if (cfg.use_fsdp && ddp && ddp->is_distributed()) {
    fsdp = FSDPContext::create(ddp->backend(), ShardingStrategy::FULL_SHARD);
    if (rank == 0 && fsdp) {
      std::cout << "FSDP: sharding params across " << ddp->world_size()
                << " ranks (each rank holds 1/" << ddp->world_size() << ")\n";
    }
  }
#else
  if (cfg.use_fsdp) std::cerr << "WARNING: fsdp=1 but binary built without "
                                 "NCCL/DDP — ignoring (build with --nccl).\n";
#endif

  auto all_params = model->parameters();
  if (fsdp) fsdp->shard_params(all_params);  // each param.data() -> local shard
  auto opt_params = (zero1 && !fsdp) ? zero1->partition(all_params) : all_params;

  // ---- Create optimizer ----
  std::unique_ptr<torch::optim::Optimizer> optimizer;
  std::string optim_display;
  if (cfg.optimizer == "lion") {
    optimizer = std::make_unique<Lion>(
        opt_params, LionOptions(cfg.lr).weight_decay(cfg.weight_decay));
    optim_display = "Lion";
  } else if (cfg.optimizer == "muon") {
    optimizer = std::make_unique<Muon>(
        opt_params, MuonOptions(cfg.lr).async_ns(cfg.async_muon));
    optim_display = cfg.async_muon ? "Muon (async NS)" : "Muon";
  } else if (cfg.optimizer == "dion") {
    optimizer = std::make_unique<DION>(
        opt_params, DIONOptions(cfg.lr).weight_decay(cfg.weight_decay));
    optim_display = "DION";
  } else if (cfg.optimizer == "adamw_8bit") {
    // 8-bit block-quantized Adam states. ~4x optimizer-memory reduction
    // vs FP32 Adam; trajectory parity within Adam-noise (bitsandbytes).
    optimizer = std::make_unique<EightBitAdamW>(
        opt_params,
        EightBitAdamWOptions().lr(cfg.lr).weight_decay(cfg.weight_decay));
    optim_display = "EightBitAdamW (block-quantized)";
  } else if (cfg.use_foreach_optimizer) {
    optimizer = std::make_unique<ForeachAdamW>(
        opt_params, ForeachAdamWOptions(cfg.lr).weight_decay(cfg.weight_decay)
                        .master_weights(cfg.use_bf16));
    optim_display = "ForeachAdamW";
  } else {
    optimizer = std::make_unique<torch::optim::AdamW>(
        opt_params,
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
    optim_display = "AdamW (standard)";
  }

  // ---- Create LR scheduler ----
  auto scheduler = create_scheduler(cfg.scheduler, cfg.lr, cfg.warmup_steps);

  // ---- Gradient scaler for mixed precision ----
  std::optional<GradScaler> grad_scaler;
  if (cfg.use_grad_scaler) {
    grad_scaler.emplace();
  }

  // ---- Checkpoint manager ----
  std::optional<CheckpointManager> ckpt_mgr;
  if (!cfg.checkpoint_dir.empty()) {
    ckpt_mgr.emplace(cfg.checkpoint_dir);
  }

  // ---- Dataset ----
  std::optional<TokenDataset> dataset;
  bool gpu_data_active = false;
  if (cfg.data_path && !cfg.data_path->empty()) {
    dataset.emplace(*cfg.data_path, cfg.seq_len, true);
    // Data-parallel sharding: each DDP rank trains on a disjoint 1/world_size
    // stride of the corpus (no-op single-GPU). Must precede reset_epoch().
    if (ddp && ddp->is_distributed()) dataset->set_shard(ddp->rank(), ddp->world_size());
    dataset->reset_epoch();
    if (device.is_cuda()) {
      const int64_t cap = cfg.gpu_resident_data ? cfg.max_gpu_data_tokens : -1;
      dataset->to_device(device, cap);
      gpu_data_active = dataset->is_gpu_resident();
    } else if (cfg.gpu_resident_data) {
      dataset->to_device(device, 0);
      gpu_data_active = dataset->is_gpu_resident();
    }
  }

  if (rank == 0) {
    print_optimization_banner(cfg.optimizer, cfg.use_foreach_optimizer,
                              cfg.gpu_resident_data, gpu_data_active,
                              cfg.use_amp, true, cfg.use_cuda_graph, cfg.use_bf16);
  }

  // Pre-build DDP parameter list once (avoids per-step allocation)
  std::vector<torch::Tensor> ddp_params;
  if (ddp && ddp->is_distributed()) {
    for (auto& p : model->parameters()) ddp_params.push_back(p);
    // T-1: register autograd hooks so per-bucket allreduce overlaps with
    // backward; allreduce_gradients() at end-of-step is then a finalize.
    // Rung 1d: NOT under CUDA graphs — autograd hooks don't re-fire on graph
    // replay (only captured CUDA kernels do), so we'd allreduce once at capture
    // and never again. With graphs we skip hooks and allreduce end-of-step
    // (outside the captured fwd+bwd) via the no-hook path in allreduce_gradients.
    const bool will_graph = cfg.use_cuda_graph && device.is_cuda();
    // FSDP does its own reduce-scatter (not allreduce), so skip DDP hooks there.
    if (!will_graph && !cfg.use_fsdp) {
      ddp->register_grad_hooks(ddp_params);
    }
  }

  // ---- Callback manager ----
  CallbackManager cb_mgr;
  for (auto& cb : callbacks) cb_mgr.add(cb);

  // ---- Epoch tracking ----
  int64_t tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps;
  int64_t dataset_tokens = dataset ? dataset->size() * cfg.seq_len : 0;
  int64_t steps_per_epoch = (dataset && dataset_tokens > 0)
      ? std::max(int64_t(1), dataset_tokens / tokens_per_step) : cfg.num_steps;
  int64_t current_epoch = 0;
  double epoch_loss_sum = 0.0;
  int64_t epoch_loss_count = 0;

  // ---- Heartbeat tracking ----
  auto epoch_start_time = std::chrono::steady_clock::now();
  double avg_epoch_seconds = 0.0;   // measured from second epoch
  int64_t heartbeat_interval = 0;   // epochs between heartbeat writes (0 = not yet calibrated)
  int64_t last_heartbeat_epoch = 0;
  bool heartbeat_enabled = cfg.report_every > 0.0 && !cfg.heartbeat_path.empty();
  if (heartbeat_enabled && rank == 0) {
    write_heartbeat(cfg.heartbeat_path, 0, 0, cfg.num_steps);
    std::cout << "Heartbeat: writing to " << cfg.heartbeat_path
              << " every ~" << cfg.report_every << "s" << std::endl;
  }

  // ---- TrainState ----
  TrainState state;
  state.train_start = std::chrono::steady_clock::now();
  cb_mgr.on_train_start(state);

  int64_t total_tokens = 0;

  // Pre-allocate loss accumulator and cache params (avoid per-step allocation)
  torch::Tensor accum_loss_tensor = torch::zeros({}, torch::TensorOptions().device(device));
  auto model_params = model->parameters();

  // ── CUDA warmup + optional graph capture ────────────────────────────────────
  //
  // WARMUP (always, on CUDA):
  //   3 full forward+backward+step passes on a non-default stream before the
  //   timed training loop. Primes the JIT compiler (PTX→SASS per kernel),
  //   the cuBLAS/cuDNN algorithm selectors, and the CUDA caching allocator's
  //   block pool. Without warmup, step 0 alone takes 20-30s on H100 due to
  //   first-time kernel compilation — masking the true steady-state speed.
  //
  // GRAPH CAPTURE (only when cuda_graph=1):
  //   After warmup, capture one forward+backward as a replayable CUDA graph.
  //   Eliminates per-kernel CPU dispatch overhead. Blocked when:
  //     - DDP (allreduce is cross-device, cannot be captured).
  //     - The fused_lm_head_ce backward materialises [N,V] logit tensors.
  //       With N=32768 and V=50304, 4 LM heads (main+3 MTP) each need ~17 GB
  //       simultaneously in the graph's private pool → OOM on 80 GB H100.
  //       Fix: a fused CE-backward kernel that avoids logit materialisation.
  //       Until then, leave cuda_graph=0 for configs with num_mtp_heads > 0.
  bool graph_active = false;
#ifdef USE_CUDA
  at::cuda::CUDAGraph train_graph;
  torch::Tensor graph_input, graph_labels, graph_loss;
  at::cuda::CUDAStream capture_stream =
      at::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());

  // Rung 1d: graphs ARE allowed under DDP now. We capture fwd+bwd and run the
  // gradient allreduce OUTSIDE the captured region (end-of-step), since NCCL
  // collectives / autograd hooks can't be naively graph-captured. Grad hooks
  // are skipped above when graphing so allreduce_gradients() reduces end-of-step.
  bool want_graph = cfg.use_cuda_graph && device.is_cuda() && !cfg.use_fsdp;

  if (device.is_cuda()) {
    auto int_opts = torch::TensorOptions().dtype(torch::kLong).device(device);
    graph_input  = torch::empty({cfg.batch_size, cfg.seq_len}, int_opts);
    graph_labels = torch::empty({cfg.batch_size, cfg.seq_len}, int_opts);

    // ---- Warmup ----
    // FSDP: params are sharded (1-D) at this point; a warmup forward would see
    // a 1-D embedding weight ('weight must be 2-D'), and a warmup step() would
    // size optimizer state to the FULL param. Skip warmup under FSDP — the
    // first real step unshards, JITs kernels, and creates shard-sized state.
    if (rank == 0) std::cout << (cfg.use_fsdp
        ? "CUDA warmup: SKIPPED under FSDP (first step JITs kernels)\n"
        : "CUDA warmup (3 steps)...\n");
    if (!cfg.use_fsdp) {
      c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
      for (int w = 0; w < 3; ++w) {
        if (dataset) {
          auto [in, lab] = dataset->get_batch(cfg.batch_size, device);
          graph_input.copy_(in);
          graph_labels.copy_(lab);
        } else {
          graph_input.copy_(torch::randint(0, model_cfg.vocab_size,
              {cfg.batch_size, cfg.seq_len}, int_opts));
          graph_labels.copy_(torch::randint(0, model_cfg.vocab_size,
              {cfg.batch_size, cfg.seq_len}, int_opts));
        }
        optimizer->zero_grad();
        {
          AutocastGuard ac(cfg.use_amp, device);
          graph_loss = model->forward(graph_input, graph_labels, -100)
                       / static_cast<float>(cfg.grad_accum_steps);
        }
        graph_loss.backward();
        clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
        optimizer->step();
      }
    }
    if (rank == 0 && !cfg.use_fsdp) std::cout << "CUDA warmup done.\n";

    // ---- Graph capture (only when requested and OOM-safe) ----
    if (want_graph) {
      if (rank == 0) {
        std::cout << "CUDA Graph: capturing forward+backward"
                  << " (will replay " << cfg.grad_accum_steps << "x per step)...\n";
      }
      // Release reserved-but-unused memory from warmup so the graph's private
      // pool can allocate cleanly without competing with the warmup pool.
      c10::cuda::CUDACachingAllocator::emptyCache();

      {
        c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
        // Must use memset zero_grad (not set_to_none) — graph captures the
        // gradient tensor addresses, which must remain valid across replays.
        // zero_grad() defaults to set_to_none=true, which FREES the grad
        // tensors; the captured graph would then write to freed memory and
        // the optimizer would see undefined grads. Pass false to memset.
        optimizer->zero_grad(/*set_to_none=*/false);
        train_graph.capture_begin();
        {
          AutocastGuard ac(cfg.use_amp, device);
          graph_loss = model->forward(graph_input, graph_labels, -100)
                       / static_cast<float>(cfg.grad_accum_steps);
        }
        graph_loss.backward();
        train_graph.capture_end();
      }

      graph_active = true;
      if (rank == 0) std::cout << "CUDA Graph: captured successfully\n";
    }
  }
#endif  // USE_CUDA

  // ---- Resume from latest checkpoint (multi-day runs survive crashes) ----
  // Restores model + optimizer state + step from the newest checkpoint in
  // checkpoint_dir. Relaunch after a crash and it picks up where it stopped.
  int64_t start_step = 0;
  if (ckpt_mgr && cfg.resume) {
    auto tag = ckpt_mgr->latest();
    if (tag) {
      auto meta = ckpt_mgr->load(*tag, *model, *optimizer,
                                 rank, ddp ? ddp->world_size() : 1);
      start_step = meta.step;
      if (rank == 0)
        std::cout << "RESUME: loaded '" << *tag << "' at step " << start_step
                  << " (loss " << meta.loss << ") — continuing.\n";
    } else if (rank == 0) {
      std::cout << "RESUME: no checkpoint in " << cfg.checkpoint_dir
                << " — starting fresh.\n";
    }
  }

  // ---- Training loop ----
  for (int64_t step = start_step; step < cfg.num_steps; ++step) {
    state.step_start = std::chrono::steady_clock::now();
    state.global_step = step;

    // Sequence length scheduling (curriculum)
    int64_t cur_seq_len = cfg.seq_len;
    if (cfg.target_seq_len > 0 && cfg.seq_len_warmup_steps > 0 && step < cfg.seq_len_warmup_steps) {
      double frac = static_cast<double>(step) / cfg.seq_len_warmup_steps;
      cur_seq_len = cfg.seq_len + static_cast<int64_t>(frac * (cfg.target_seq_len - cfg.seq_len));
      cur_seq_len = ((cur_seq_len + 63) / 64) * 64;
    } else if (cfg.target_seq_len > 0) {
      cur_seq_len = cfg.target_seq_len;
    }
    state.seq_len = cur_seq_len;

    // Batch size scheduling
    int64_t cur_batch_size = cfg.batch_size;
    if (cfg.target_batch_size > 0 && cfg.batch_size_ramp_steps > 0) {
      int64_t doubles = 0;
      int64_t bs = cfg.batch_size;
      while (bs < cfg.target_batch_size) { doubles++; bs *= 2; }
      if (doubles > 0) {
        int64_t interval = cfg.batch_size_ramp_steps / doubles;
        int64_t cur_double = std::min(step / std::max(interval, int64_t(1)), doubles);
        cur_batch_size = cfg.batch_size * (1 << cur_double);
        cur_batch_size = std::min(cur_batch_size, cfg.target_batch_size);
      }
    }
    state.batch_size = cur_batch_size;

    // Epoch boundary detection
    int64_t new_epoch = dataset ? (step / steps_per_epoch) : 0;
    if (new_epoch > current_epoch && rank == 0) {
      auto epoch_end_time = std::chrono::steady_clock::now();
      double epoch_seconds = std::chrono::duration<double>(epoch_end_time - epoch_start_time).count();

      double avg_epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : 0.0;
      std::cout << "--- Epoch " << current_epoch << " complete | avg_loss: "
                << std::fixed << std::setprecision(4) << avg_epoch_loss
                << " | time: " << std::setprecision(1) << epoch_seconds << "s ---" << std::endl;

      // Calibrate heartbeat interval from second epoch (first is warmup)
      if (heartbeat_enabled && current_epoch == 1 && avg_epoch_seconds == 0.0) {
        avg_epoch_seconds = epoch_seconds;
        heartbeat_interval = std::max(int64_t(1),
            static_cast<int64_t>(cfg.report_every / avg_epoch_seconds));
        std::cout << "Heartbeat: epoch ~" << std::setprecision(1) << avg_epoch_seconds
                  << "s, writing every " << heartbeat_interval << " epochs" << std::endl;
      }

      // Write heartbeat every heartbeat_interval epochs
      if (heartbeat_enabled && heartbeat_interval > 0 &&
          (new_epoch - last_heartbeat_epoch) >= heartbeat_interval) {
        write_heartbeat(cfg.heartbeat_path, new_epoch, step, cfg.num_steps);
        last_heartbeat_epoch = new_epoch;
      }

      epoch_loss_sum = 0.0;
      epoch_loss_count = 0;
      current_epoch = new_epoch;
      epoch_start_time = std::chrono::steady_clock::now();
    }

    // LR schedule
    double cur_lr = scheduler->get_lr(step, cfg.num_steps);
    scheduler->apply(*optimizer, step, cfg.num_steps);
    state.learning_rate = static_cast<float>(cur_lr);

    cb_mgr.on_step_start(state);

#ifdef USE_CUDA
    if (graph_active) {
      // ---- CUDA Graph path ----
      // Zero gradients once (memset — graph holds fixed gradient tensor addresses).
      // MUST pass set_to_none=false: the default (true) frees the grad tensors
      // the graph captured pointers to, corrupting every replay.
      optimizer->zero_grad(/*set_to_none=*/false);
      accum_loss_tensor.zero_();
      for (int64_t accum = 0; accum < cfg.grad_accum_steps; ++accum) {
        if (dataset) {
          auto [in, lab] = dataset->get_batch(cfg.batch_size, device);
          graph_input.copy_(in);
          graph_labels.copy_(lab);
        }
        train_graph.replay();
        accum_loss_tensor.add_(graph_loss.detach());
      }
    } else
#endif
    {
      // ---- Standard path (non-graph or CPU) ----
      // FSDP: reconstruct full params (allgather) for fwd/bwd; resharded below.
      if (fsdp) fsdp->unshard_params(all_params);
      optimizer->zero_grad(true);
      accum_loss_tensor.zero_();

      for (int64_t accum = 0; accum < cfg.grad_accum_steps; ++accum) {
        torch::Tensor input, labels;
        {
          ProfileScope data_scope("data_loading");
          if (dataset) {
            auto [in, lab] = dataset->get_batch(cur_batch_size, device);
            input = in; labels = lab;
            dataset->prefetch_next(cur_batch_size, device);
          } else {
            input = torch::randint(0, model_cfg.vocab_size, {cur_batch_size, cur_seq_len},
                                   torch::TensorOptions().dtype(torch::kLong).device(device));
            labels = torch::randint(0, model_cfg.vocab_size, {cur_batch_size, cur_seq_len},
                                    torch::TensorOptions().dtype(torch::kLong).device(device));
          }
        }

        torch::Tensor loss;
        {
          ProfileScope fwd_scope("forward");
          AutocastGuard ac(cfg.use_amp, device);
          loss = model->forward(input, labels, -100) / static_cast<float>(cfg.grad_accum_steps);
        }

        {
          ProfileScope bwd_scope("backward");
          if (grad_scaler) {
            grad_scaler->scale(loss).backward();
          } else {
            loss.backward();
          }
        }
        accum_loss_tensor.add_(loss.detach());
      }
    }

    // Defer loss D2H sync: only pull from GPU when actually needed.
    //
    // Previous logic synced whenever ANY callback was registered, even
    // callbacks that never read state.loss (grad_stats, etc.) — that's a
    // free per-step sync that costs ~50µs on H100 and breaks compute/comm
    // overlap. Gate strictly on log/eval/checkpoint boundaries; if a
    // callback genuinely needs loss every step, it should call
    // accum_loss_tensor.item<float>() itself in on_step_start.
    const bool log_step  = (step % cfg.log_interval == 0 && rank == 0);
    const bool eval_step = (evaluator && cfg.eval_interval > 0 &&
                            (step + 1) % cfg.eval_interval == 0);
    const bool ckpt_step = (ckpt_mgr && cfg.checkpoint_interval > 0 &&
                            (step + 1) % cfg.checkpoint_interval == 0);
    const bool need_loss_sync = log_step || eval_step || ckpt_step;
    // AA: queue the loss every step (non-blocking async copy) and poll
    // when we actually want to log/eval/ckpt. Removes the per-log-step
    // stream-drain sync.
    static thread_local AsyncLossReader _loss_reader_b;
    _loss_reader_b.queue(accum_loss_tensor);
    // Poll EVERY step: queue() runs every step, so the read head must advance
    // every step too. Polling only on log steps (with a depth-4 ring) let the
    // writer lap the buffer and the read head stall on cudaErrorNotReady, so
    // the logged value desynced from the actual step. Draining each step keeps
    // read_head in lock-step with write_head; the value is still non-blocking.
    float accum_loss = _loss_reader_b.poll();
    if (need_loss_sync) {
      epoch_loss_sum += accum_loss;
      epoch_loss_count++;
    }
    state.loss = accum_loss;
    cb_mgr.on_after_loss(state);

    // Gradient sync in distributed
    if (fsdp) {
      // FSDP: grads are currently FULL (backward ran on unsharded params).
      // Collect them, re-shard the params (free the full-param memory), then
      // reduce-scatter the grads so each rank keeps only its 1/world_size grad
      // shard — matching its param shard for the optimizer step.
      std::vector<torch::Tensor> fgrads;
      fgrads.reserve(all_params.size());
      for (auto& p : all_params) fgrads.push_back(p.grad());
      fsdp->reshard_params(all_params);
      fsdp->reduce_scatter_grads(fgrads);
      for (size_t i = 0; i < all_params.size(); ++i) {
        if (fgrads[i].defined()) all_params[i].mutable_grad() = fgrads[i];
      }
    } else if (ddp && ddp->is_distributed()) {
      ddp->allreduce_gradients(ddp_params);
    }

    cb_mgr.on_after_backward(state);

    // Unscale + check for inf/nan if using grad scaler
    {
      ProfileScope optim_scope("optimizer_step");  // grad-clip + optimizer update
      if (grad_scaler) {
        bool finite = grad_scaler->unscale_and_check(*optimizer);
        if (finite) {
          clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
          grad_scaler->step(*optimizer);
        }
        grad_scaler->update();
      } else {
        clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
        optimizer->step();
      }
    }
    // ZeRO-1: broadcast each updated parameter from its owning rank.
    if (zero1) zero1->allgather_params(all_params);

    cb_mgr.on_after_optimizer_step(state);

    total_tokens += cur_batch_size * cur_seq_len * cfg.grad_accum_steps;

    // Update metrics
    state.metrics["loss"] = accum_loss;
    state.metrics["lr"] = static_cast<float>(cur_lr);
    state.metrics["seq_len"] = static_cast<float>(cur_seq_len);
    state.metrics["batch_size"] = static_cast<float>(cur_batch_size);
    if (grad_scaler) {
      state.metrics["grad_scale"] = grad_scaler->current_scale();
    }

    cb_mgr.on_step_end(state);

    // Logging
    if (step % cfg.log_interval == 0 && rank == 0) {
      auto now = std::chrono::steady_clock::now();
      double step_ms = std::chrono::duration<double, std::milli>(now - state.step_start).count();
      double elapsed_s = std::chrono::duration<double>(now - state.train_start).count();
      double tok_per_s = total_tokens / (elapsed_s > 0 ? elapsed_s : 1);

      std::cout << "Epoch " << current_epoch
                << " | Step " << step << "/" << cfg.num_steps
                << "  loss: " << std::fixed << std::setprecision(4) << accum_loss
                << "  lr: " << std::scientific << std::setprecision(2) << cur_lr
                << "  step_ms: " << static_cast<int>(step_ms)
                << "  tok/s: " << static_cast<int>(tok_per_s);
      if (grad_scaler) std::cout << "  scale: " << grad_scaler->current_scale();
      std::cout << std::endl;
    }

    // Evaluation
    if (evaluator && cfg.eval_interval > 0 && (step + 1) % cfg.eval_interval == 0) {
      model->eval();
      auto eval_metrics = evaluator->evaluate(*model, device);
      std::unordered_map<std::string, float> eval_float;
      for (const auto& [k, v] : eval_metrics) {
        state.metrics["eval/" + k] = static_cast<float>(v);
        eval_float[k] = static_cast<float>(v);
        if (rank == 0) std::cout << "  eval/" << k << ": " << v << std::endl;
      }
      cb_mgr.on_eval_end(state, eval_float);
      model->train();
    }

    // R.2: async checkpoint save. The CheckpointManager already exposes
    // save_async(); use it so the main training stream doesn't pause to
    // write parameter tensors to disk. The returned future is dropped
    // on the next ckpt iteration (block-on-prior-save semantics).
    if (ckpt_mgr && cfg.checkpoint_interval > 0 && (step + 1) % cfg.checkpoint_interval == 0) {
      std::string tag = "step_" + std::to_string(step + 1);
      CheckpointMetadata meta;
      meta.step = step + 1;
      meta.loss = accum_loss;
      static std::future<void> _last_ckpt;
      if (_last_ckpt.valid()) _last_ckpt.wait();  // ensure previous save finished
      _last_ckpt = ckpt_mgr->save_async(tag, *model, *optimizer, meta, rank,
                                         ddp ? ddp->world_size() : 1);
      ckpt_mgr->prune(cfg.keep_checkpoints);
      // Also drop a single-file <checkpoint_dir>/latest.pt (rank 0, full model)
      // so `chat`/inference can load the in-progress model WITHOUT stopping the
      // run. The sharded save above is for resume; this one is for using it.
      // (DDP: rank 0 holds the full model. Skip under FSDP — rank 0 has shards.)
      if (rank == 0 && !cfg.use_fsdp) {
        try {
          torch::save(model, cfg.checkpoint_dir + "/latest.pt");  // holder, not *model
        } catch (const std::exception& e) {
          std::cerr << "WARN: latest.pt export failed: " << e.what() << "\n";
        }
      }
      cb_mgr.on_checkpoint_save(state, cfg.checkpoint_dir + "/" + tag);
    }
  }

  cb_mgr.on_train_end(state);

  if (rank == 0) {
    if (heartbeat_enabled)
      write_heartbeat(cfg.heartbeat_path, current_epoch, cfg.num_steps, cfg.num_steps);

    auto end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(end - state.train_start).count();
    double avg_step_ms = total_s / cfg.num_steps * 1000.0;
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "  Steps: " << cfg.num_steps << " (" << current_epoch + 1 << " epochs)" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Wall time: " << std::fixed << std::setprecision(2) << total_s << "s" << std::endl;
    std::cout << "  Avg step: " << std::fixed << std::setprecision(1) << avg_step_ms << "ms" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(total_tokens / total_s) << " tok/s" << std::endl;
    std::cout << "========================" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// FusedTransformer overload — delegates to same loop logic
// ---------------------------------------------------------------------------

void train(
    FusedTransformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks,
    std::shared_ptr<Evaluator> evaluator) {

  model->train();

  auto ddp = DDPContext::init_from_env();
  if (ddp && ddp->is_distributed()) {
    auto params = model->parameters();
    ddp->broadcast_parameters(params);
  }
  int rank = ddp ? ddp->rank() : 0;

  // ZeRO-1 (FusedTransformer overload).
  std::unique_ptr<OptimizerStateSharder> zero1;
  if (cfg.use_zero1 && ddp && ddp->is_distributed() && !cfg.use_fsdp) {
    zero1 = std::make_unique<OptimizerStateSharder>(
        ddp->backend(), ddp->rank(), ddp->world_size());
  }

  // ---- FSDP (ZeRO-3): shard params + grads + optimizer state across ranks ----
  // shard_params() must run BEFORE the optimizer is built so the optimizer (and
  // its moment state) are sized to the local 1/world_size shard. The full param
  // is reconstructed (unshard) each step for fwd/bwd, then re-sharded.
  std::optional<FSDPContext> fsdp;
#if defined(OLMO_HAS_NCCL) || defined(OLMO_HAS_DDP)
  if (cfg.use_fsdp && ddp && ddp->is_distributed()) {
    fsdp = FSDPContext::create(ddp->backend(), ShardingStrategy::FULL_SHARD);
    if (rank == 0 && fsdp) {
      std::cout << "FSDP: sharding params across " << ddp->world_size()
                << " ranks (each rank holds 1/" << ddp->world_size() << ")\n";
    }
  }
#else
  if (cfg.use_fsdp) std::cerr << "WARNING: fsdp=1 but binary built without "
                                 "NCCL/DDP — ignoring (build with --nccl).\n";
#endif

  auto all_params = model->parameters();
  if (fsdp) fsdp->shard_params(all_params);  // each param.data() -> local shard
  auto opt_params = (zero1 && !fsdp) ? zero1->partition(all_params) : all_params;

  std::unique_ptr<torch::optim::Optimizer> optimizer;
  if (cfg.optimizer == "lion") {
    optimizer = std::make_unique<Lion>(
        opt_params, LionOptions(cfg.lr).weight_decay(cfg.weight_decay));
  } else if (cfg.optimizer == "muon") {
    optimizer = std::make_unique<Muon>(
        opt_params, MuonOptions(cfg.lr));
  } else if (cfg.optimizer == "dion") {
    optimizer = std::make_unique<DION>(
        opt_params, DIONOptions(cfg.lr).weight_decay(cfg.weight_decay));
  } else if (cfg.optimizer == "adamw_8bit") {
    optimizer = std::make_unique<EightBitAdamW>(
        opt_params,
        EightBitAdamWOptions().lr(cfg.lr).weight_decay(cfg.weight_decay));
  } else if (cfg.use_foreach_optimizer) {
    optimizer = std::make_unique<ForeachAdamW>(
        opt_params, ForeachAdamWOptions(cfg.lr).weight_decay(cfg.weight_decay)
                        .master_weights(cfg.use_bf16));
  } else {
    optimizer = std::make_unique<torch::optim::AdamW>(
        opt_params,
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay));
  }

  auto scheduler = create_scheduler(cfg.scheduler, cfg.lr, cfg.warmup_steps);

  std::optional<TokenDataset> dataset;
  bool gpu_data_active = false;
  if (cfg.data_path && !cfg.data_path->empty()) {
    dataset.emplace(*cfg.data_path, cfg.seq_len, true);
    // Data-parallel sharding: each DDP rank trains on a disjoint 1/world_size
    // stride of the corpus (no-op single-GPU). Must precede reset_epoch().
    if (ddp && ddp->is_distributed()) dataset->set_shard(ddp->rank(), ddp->world_size());
    dataset->reset_epoch();
    if (device.is_cuda()) {
      const int64_t cap = cfg.gpu_resident_data ? cfg.max_gpu_data_tokens : -1;
      dataset->to_device(device, cap);
      gpu_data_active = dataset->is_gpu_resident();
    } else if (cfg.gpu_resident_data) {
      dataset->to_device(device, 0);
      gpu_data_active = dataset->is_gpu_resident();
    }
  }

  if (rank == 0) {
    print_optimization_banner(cfg.optimizer, cfg.use_foreach_optimizer,
                              cfg.gpu_resident_data, gpu_data_active,
                              cfg.use_amp, true, cfg.use_cuda_graph, cfg.use_bf16);
  }

  // Pre-build DDP parameter list once (avoids per-step allocation)
  std::vector<torch::Tensor> ddp_params;
  if (ddp && ddp->is_distributed()) {
    for (auto& p : model->parameters()) ddp_params.push_back(p);
    // T-1: register autograd hooks so per-bucket allreduce overlaps with
    // backward; allreduce_gradients() at end-of-step is then a finalize.
    // Rung 1d: NOT under CUDA graphs — autograd hooks don't re-fire on graph
    // replay (only captured CUDA kernels do), so we'd allreduce once at capture
    // and never again. With graphs we skip hooks and allreduce end-of-step
    // (outside the captured fwd+bwd) via the no-hook path in allreduce_gradients.
    const bool will_graph = cfg.use_cuda_graph && device.is_cuda();
    // FSDP does its own reduce-scatter (not allreduce), so skip DDP hooks there.
    if (!will_graph && !cfg.use_fsdp) {
      ddp->register_grad_hooks(ddp_params);
    }
  }

  // ---- Callback manager ----
  CallbackManager cb_mgr;
  for (auto& cb : callbacks) cb_mgr.add(cb);

  // ---- SGP (Speculative Gradient Prediction) ----
  std::unique_ptr<ISGPPredictor> sgp;
  if (cfg.sgp_enabled) {
    SGPConfig sgp_cfg;
    sgp_cfg.initial_k = cfg.sgp_initial_k;
    sgp_cfg.max_k = cfg.sgp_max_k;
    sgp_cfg.warmup_steps = cfg.sgp_warmup_steps;
    std::vector<torch::Tensor> sgp_params;
    for (auto& p : model->parameters()) sgp_params.push_back(p);
    if (cfg.sgp_version == 2) {
      sgp = std::make_unique<SGPv2Predictor>(sgp_params, sgp_cfg, cfg.sgp_rank);
    } else {
      sgp = std::make_unique<SGPPredictor>(sgp_params, sgp_cfg);
    }
    if (rank == 0) {
      std::cout << "SGP: enabled v" << cfg.sgp_version
                << " (initial_k=" << sgp_cfg.initial_k
                << ", max_k=" << sgp_cfg.max_k
                << ", warmup=" << sgp_cfg.warmup_steps;
      if (cfg.sgp_version == 2) std::cout << ", rank=" << cfg.sgp_rank;
      std::cout << ")\n";
    }
  }

  // Epoch tracking
  int64_t tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps;
  int64_t dataset_tokens = dataset ? dataset->size() * cfg.seq_len : 0;
  int64_t steps_per_epoch = (dataset && dataset_tokens > 0)
      ? std::max(int64_t(1), dataset_tokens / tokens_per_step) : cfg.num_steps;
  int64_t current_epoch = 0;
  double epoch_loss_sum = 0.0;
  int64_t epoch_loss_count = 0;

  // ---- Heartbeat tracking ----
  auto epoch_start_time = std::chrono::steady_clock::now();
  double avg_epoch_seconds = 0.0;
  int64_t heartbeat_interval = 0;
  int64_t last_heartbeat_epoch = 0;
  bool heartbeat_enabled = cfg.report_every > 0.0 && !cfg.heartbeat_path.empty();
  if (heartbeat_enabled && rank == 0) {
    write_heartbeat(cfg.heartbeat_path, 0, 0, cfg.num_steps);
    std::cout << "Heartbeat: writing to " << cfg.heartbeat_path
              << " every ~" << cfg.report_every << "s" << std::endl;
  }

  TrainState state;
  state.batch_size = cfg.batch_size;
  state.seq_len = cfg.seq_len;
  state.train_start = std::chrono::steady_clock::now();
  cb_mgr.on_train_start(state);

  auto train_start = std::chrono::steady_clock::now();
  int64_t total_tokens = 0;

  // Pre-allocate loss accumulator outside loop (avoids per-step allocation)
  torch::Tensor accum_loss_tensor = torch::zeros({}, torch::TensorOptions().device(device));

  // Cache model params once (avoids per-step vector rebuild)
  auto model_params = model->parameters();

  // ---------------------------------------------------------------------------
  // CUDA Graph: capture forward+backward as a replayable graph.
  // Eliminates per-kernel launch overhead (~5μs × 100+ kernels = 0.5-1ms/step)
  // and enables the GPU to pipeline operations without waiting for CPU dispatch.
  //
  // Requirements: CUDA device, fixed shapes (no curriculum), no activation
  // checkpointing, no DDP (allreduce is cross-device). The optimizer step
  // runs OUTSIDE the graph because lr/bias_correction change per step.
  // grad_accum > 1 works: we replay the captured graph N times per step.
  // ---------------------------------------------------------------------------
  bool graph_active = false;
#ifdef USE_CUDA
  at::cuda::CUDAGraph train_graph;
  torch::Tensor graph_input, graph_labels, graph_loss;
  // Capture stream — CUDA graphs require a non-default stream for capture.
  // After capture, replay happens on whatever stream is current (default is fine).
  at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());

  // Rung 1d: graphs ARE allowed under DDP now. We capture fwd+bwd and run the
  // gradient allreduce OUTSIDE the captured region (end-of-step), since NCCL
  // collectives / autograd hooks can't be naively graph-captured. Grad hooks
  // are skipped above when graphing so allreduce_gradients() reduces end-of-step.
  bool want_graph = cfg.use_cuda_graph && device.is_cuda() && !cfg.use_fsdp;
  if (want_graph) {
    auto int_opts = torch::TensorOptions().dtype(torch::kLong).device(device);
    graph_input  = torch::empty({cfg.batch_size, cfg.seq_len}, int_opts);
    graph_labels = torch::empty({cfg.batch_size, cfg.seq_len}, int_opts);

    // ---- Warmup on capture stream ----
    // Must warm up on the SAME stream we'll capture on, so the caching
    // allocator records allocations on that stream's memory pool.
    if (rank == 0) std::cout << "CUDA Graph: warming up (3 steps)...\n";
    {
      c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
      for (int w = 0; w < 3; ++w) {
        if (dataset) {
          auto [in, lab] = dataset->get_batch(cfg.batch_size, device);
          graph_input.copy_(in);
          graph_labels.copy_(lab);
        } else {
          graph_input.copy_(torch::randint(0, model_cfg.vocab_size,
              {cfg.batch_size, cfg.seq_len}, int_opts));
          graph_labels.copy_(torch::randint(0, model_cfg.vocab_size,
              {cfg.batch_size, cfg.seq_len}, int_opts));
        }
        optimizer->zero_grad();
        {
          AutocastGuard ac(cfg.use_amp, device);
          graph_loss = model->forward(graph_input, graph_labels, -100)
                       / static_cast<float>(cfg.grad_accum_steps);
        }
        graph_loss.backward();
        clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
        optimizer->step();
      }
    }

    // ---- Capture on non-default stream ----
    // The graph captures one forward+backward pass on a single micro-batch.
    // For grad_accum > 1, we replay the graph N times per optimizer step —
    // gradients accumulate naturally across replays since we only zero_grad
    // once at the start.
    if (rank == 0) {
      std::cout << "CUDA Graph: capturing forward+backward"
                << " (will replay " << cfg.grad_accum_steps << "x per step)...\n";
    }

    // Release reserved-but-unused memory from warmup so the graph's private
    // pool can allocate. Without this, the caching allocator holds ~50+ GiB
    // reserved for the default pool, starving the capture pool.
    c10::cuda::CUDACachingAllocator::emptyCache();

    {
      c10::cuda::CUDAStreamGuard stream_guard(capture_stream);

      // Must use memset zero_grad (not set_to_none) — graph references
      // specific gradient tensor addresses that must remain valid.
      // The default set_to_none=true frees them; pass false to memset.
      optimizer->zero_grad(/*set_to_none=*/false);

      train_graph.capture_begin();
      {
        AutocastGuard ac(cfg.use_amp, device);
        graph_loss = model->forward(graph_input, graph_labels, -100)
                     / static_cast<float>(cfg.grad_accum_steps);
      }
      graph_loss.backward();
      train_graph.capture_end();
    }

    graph_active = true;
    if (rank == 0) std::cout << "CUDA Graph: captured successfully\n";
  }
#endif  // USE_CUDA

  for (int64_t step = 0; step < cfg.num_steps; ++step) {
    auto step_start = std::chrono::steady_clock::now();
    state.global_step = step;
    state.step_start = step_start;
    state.tokens_seen = total_tokens;
    cb_mgr.on_step_start(state);

    double cur_lr = scheduler->get_lr(step, cfg.num_steps);
    state.learning_rate = static_cast<float>(cur_lr);
    scheduler->apply(*optimizer, step, cfg.num_steps);

    // Epoch boundary detection
    int64_t new_epoch = dataset ? (step / steps_per_epoch) : 0;
    if (new_epoch > current_epoch && rank == 0) {
      auto epoch_end_time = std::chrono::steady_clock::now();
      double epoch_seconds = std::chrono::duration<double>(epoch_end_time - epoch_start_time).count();

      double avg_epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : 0.0;
      std::cout << "--- Epoch " << current_epoch << " complete | avg_loss: "
                << std::fixed << std::setprecision(4) << avg_epoch_loss
                << " | time: " << std::setprecision(1) << epoch_seconds << "s ---" << std::endl;

      // Calibrate heartbeat interval from second epoch (first is warmup)
      if (heartbeat_enabled && current_epoch == 1 && avg_epoch_seconds == 0.0) {
        avg_epoch_seconds = epoch_seconds;
        heartbeat_interval = std::max(int64_t(1),
            static_cast<int64_t>(cfg.report_every / avg_epoch_seconds));
        std::cout << "Heartbeat: epoch ~" << std::setprecision(1) << avg_epoch_seconds
                  << "s, writing every " << heartbeat_interval << " epochs" << std::endl;
      }

      // Write heartbeat every heartbeat_interval epochs
      if (heartbeat_enabled && heartbeat_interval > 0 &&
          (new_epoch - last_heartbeat_epoch) >= heartbeat_interval) {
        write_heartbeat(cfg.heartbeat_path, new_epoch, step, cfg.num_steps);
        last_heartbeat_epoch = new_epoch;
      }

      epoch_loss_sum = 0.0;
      epoch_loss_count = 0;
      current_epoch = new_epoch;
      epoch_start_time = std::chrono::steady_clock::now();
    }

#ifdef USE_CUDA
    if (graph_active) {
      // ---- CUDA Graph path ----
      // Zero gradients once (memset — graph references these tensor addresses).
      // set_to_none=false is REQUIRED: the default frees the captured grads.
      optimizer->zero_grad(/*set_to_none=*/false);
      accum_loss_tensor.zero_();

      // Replay captured micro-step for each grad_accum iteration.
      // Gradients accumulate across replays (we only zero_grad once above).
      for (int64_t accum = 0; accum < cfg.grad_accum_steps; ++accum) {
        // 1. Copy fresh data into static graph buffers
        if (dataset) {
          auto [in, lab] = dataset->get_batch(cfg.batch_size, device);
          graph_input.copy_(in);
          graph_labels.copy_(lab);
        }

        // 2. Replay captured forward + backward (single graph launch)
        train_graph.replay();

        // 3. Accumulate loss
        accum_loss_tensor.add_(graph_loss.detach());
      }
    } else
#endif
    {
      // ---- Standard path (non-graph or CPU) ----
      bool sgp_skip = sgp && sgp->should_skip_backward(step);

      if (sgp_skip) {
        // SGP predicted step — skip backward entirely, fill grads from predictor
        optimizer->zero_grad(true);
        sgp->apply_predicted_gradients();
        accum_loss_tensor.zero_();  // no loss available on predicted steps
      } else {
        // Real backward — compute gradients normally
        optimizer->zero_grad(true);
        accum_loss_tensor.zero_();

        for (int64_t accum = 0; accum < cfg.grad_accum_steps; ++accum) {
          torch::Tensor input, labels;
          if (dataset) {
            auto [in, lab] = dataset->get_batch(cfg.batch_size, device);
            input = in; labels = lab;
            dataset->prefetch_next(cfg.batch_size, device);
          } else {
            input = torch::randint(0, model_cfg.vocab_size, {cfg.batch_size, cfg.seq_len},
                                   torch::TensorOptions().dtype(torch::kLong).device(device));
            labels = torch::randint(0, model_cfg.vocab_size, {cfg.batch_size, cfg.seq_len},
                                    torch::TensorOptions().dtype(torch::kLong).device(device));
          }

          torch::Tensor loss;
          {
            AutocastGuard ac(cfg.use_amp, device);
            loss = model->forward(input, labels, -100) / static_cast<float>(cfg.grad_accum_steps);
          }
          loss.backward();
          accum_loss_tensor.add_(loss.detach());
        }

        // Record real gradients for future prediction
        if (sgp) sgp->observe_real_gradients();
      }
    }

    cb_mgr.on_after_backward(state);

    // Gradient sync + optimizer step always OUTSIDE graph
    // (LR and bias correction change per step)
    if (ddp && ddp->is_distributed()) {
      ddp->allreduce_gradients(ddp_params);
    }

    clip_grad_norm_gpu(model_params, cfg.max_grad_norm);
    optimizer->step();
    if (zero1) zero1->allgather_params(all_params);
    cb_mgr.on_after_optimizer_step(state);

    total_tokens += tokens_per_step;
    state.tokens_seen = total_tokens;

    // AA: queue + poll (non-blocking) instead of synchronous .item().
    static thread_local AsyncLossReader _loss_reader_c;
    _loss_reader_c.queue(accum_loss_tensor);
    // Poll every step so the ring's read head keeps pace with the per-step
    // queue() (otherwise the writer laps the buffer and the logged loss
    // desyncs from the step number). Non-blocking.
    float _polled_loss = _loss_reader_c.poll();
    if (step % cfg.log_interval == 0 && rank == 0) {
      float accum_loss = _polled_loss;
      state.loss = accum_loss;
      epoch_loss_sum += accum_loss;
      epoch_loss_count++;
      auto step_end = std::chrono::steady_clock::now();
      double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
      double elapsed_s = std::chrono::duration<double>(step_end - train_start).count();
      double tok_per_s = total_tokens / (elapsed_s > 0 ? elapsed_s : 1);

      std::cout << "Epoch " << current_epoch
                << " | Step " << step << "/" << cfg.num_steps
                << "  loss: " << std::fixed << std::setprecision(4) << accum_loss
                << "  lr: " << std::scientific << std::setprecision(2) << cur_lr
                << "  step_ms: " << static_cast<int>(step_ms)
                << "  tok/s: " << static_cast<int>(tok_per_s);
      if (sgp) {
        std::cout << "  sgp_k=" << sgp->current_k()
                  << " skip=" << std::fixed << std::setprecision(1) << (sgp->skip_rate() * 100.0) << "%"
                  << " err=" << std::setprecision(3) << sgp->last_prediction_error();
      }
      std::cout << std::endl;
    }

    cb_mgr.on_step_end(state);
  }

  cb_mgr.on_train_end(state);

  if (rank == 0) {
    if (heartbeat_enabled)
      write_heartbeat(cfg.heartbeat_path, current_epoch, cfg.num_steps, cfg.num_steps);

    auto end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(end - train_start).count();
    double avg_step_ms = total_s / cfg.num_steps * 1000.0;
    std::cout << "\n=== Training Summary ===" << std::endl;
    std::cout << "  Steps: " << cfg.num_steps << " (" << current_epoch + 1 << " epochs)" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Wall time: " << std::fixed << std::setprecision(2) << total_s << "s" << std::endl;
    std::cout << "  Avg step: " << std::fixed << std::setprecision(1) << avg_step_ms << "ms" << std::endl;
    std::cout << "  Throughput: " << static_cast<int>(total_tokens / total_s) << " tok/s" << std::endl;
    std::cout << "========================" << std::endl;
  }
}

}  // namespace olmo_cpp
