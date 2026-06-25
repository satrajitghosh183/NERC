#pragma once

/**
 * include/olmo_cpp/train.hpp
 *
 * Public training API. Declares `TrainConfig` (the runtime/hyperparameter
 * counterpart to `TransformerConfig`'s architectural settings) plus the
 * two `train()` overloads — one for `Transformer`, one for
 * `FusedTransformer` — and the legacy `train_epoch()` used by older
 * benchmarks. Everything in `TrainConfig` is populated from the
 * `[training] / [optimization]` sections of the .conf file in main.cpp;
 * `train()` itself owns the optimizer, scheduler, dataset, DDP context,
 * gradient scaler, callbacks, and checkpoint manager for the whole run.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/config.hpp: `TransformerConfig` is passed alongside the
 *     `TrainConfig` so the loop knows the model's vocab/hidden sizes for
 *     things like fake-data generation.
 *   - olmo_cpp/model/transformer.hpp: the `Transformer` overload's first arg.
 *   - olmo_cpp/model/fused_transformer.hpp: the `FusedTransformer` overload.
 *
 * --- Callers (concrete uses elsewhere in this repo) ---
 *   - src/main.cpp: `olmo_cpp::train(model, cfg, train_cfg, device, callbacks)`
 *     after the model has been moved to `device` and weights initialised.
 *   - src/train.cpp: implements all three functions.
 *
 * --- Role in training pipeline ---
 *   The single user-facing entry point for training. Constructed once per
 *   run; lives for the entire training duration. Side effects: optimizer
 *   steps, allreduces in DDP runs, checkpoint files written to disk,
 *   evaluation logs to stdout, callbacks invoked at each lifecycle hook.
 */

#include "olmo_cpp/config.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"
#include <torch/torch.h>
#include <optional>
#include <string>
#include <vector>
#include <memory>

namespace olmo_cpp {

/// Forward decl — defined in train/callback.hpp; implemented by various
/// callback classes (gradient stats, throughput, etc.).
class Callback;
/// Forward decl — defined in eval/evaluator.hpp; runs validation passes.
class Evaluator;

/// Training configuration. All fields are mutable knobs; nothing here
/// affects model architecture (that's in `TransformerConfig`). Defaults
/// are tuned for a small smoke-test run; production .conf files override
/// almost every field.
struct TrainConfig {
  /// Total number of optimizer steps to run.
  int64_t num_steps = 1000;
  /// Per-replica micro-batch size (before gradient accumulation).
  int64_t batch_size = 4;
  /// Sequence length in tokens. Together with batch_size determines
  /// per-step token count.
  int64_t seq_len = 128;
  /// Peak learning rate (after warmup).
  double lr = 1e-4;
  /// Linear warmup steps before the cosine/linear decay kicks in.
  int64_t warmup_steps = 100;
  /// Micro-batches per optimizer step. Effective batch =
  /// batch_size * grad_accum_steps * world_size.
  int64_t grad_accum_steps = 1;
  /// Global gradient clipping norm (0 disables).
  double max_grad_norm = 1.0;
  /// AdamW/Lion weight decay (decoupled L2).
  double weight_decay = 0.01;

  // Mixed precision
  /// Autocast mode: FP32 master weights, per-op BF16. Best for <~1B params.
  bool use_amp = false;           // autocast: FP32 master weights, per-op BF16 (small models)
  /// Pure BF16 mode: weights AND optimizer state in BF16. ~50% memory.
  bool use_bf16 = false;          // pure BF16: convert weights to BF16, no autocast (large models)
  /// Loss scaling for FP16 training (rarely needed with BF16).
  bool use_grad_scaler = false;   // loss scaling for fp16

  // Optimizer selection
  /// Optimizer name dispatched in train(). One of: adamw, lion, muon, dion.
  std::string optimizer = "adamw";  // adamw, lion, muon, dion

  // LR scheduler
  /// LR schedule name. cosine = warmup+cosine decay; wsd = warmup-stable-decay.
  std::string scheduler = "cosine";  // cosine, linear, constant, wsd, etc.

  // Activation checkpointing
  /// Stride between checkpointed blocks. 0 disables. (Mostly redundant
  /// with the model-side knob in TransformerConfig.)
  int64_t activation_checkpoint_interval = 0;  // 0 = disabled

  // Data
  /// Path to a tokenised .npy corpus; nullopt drives the loop with random
  /// data (used in unit/perf tests).
  std::optional<std::string> data_path;
  /// Optional held-out eval corpus path.
  std::optional<std::string> eval_data_path;
  /// Steps between eval passes when an evaluator is wired up.
  int64_t eval_interval = 500;  // steps between evals

  // Checkpointing
  /// Directory to write torch checkpoints to. Empty disables checkpointing.
  std::string checkpoint_dir;
  /// Steps between checkpoint writes.
  int64_t checkpoint_interval = 1000;
  /// Keep at most this many recent checkpoints (older are pruned).
  int keep_checkpoints = 3;
  /// Resume from the latest checkpoint in checkpoint_dir if one exists
  /// (restores model + optimizer + step). Lets a multi-day run survive a
  /// crash: just relaunch. No-op when checkpoint_dir is empty or has none.
  bool resume = false;

  // Sequence/batch scheduling
  /// Target seq_len at end of curriculum ramp; -1 disables (constant seq_len).
  int64_t target_seq_len = -1;      // for curriculum: ramp from seq_len to this
  /// Steps over which to linearly ramp seq_len from start to target.
  int64_t seq_len_warmup_steps = 0;
  /// Target batch_size at end of doubling ramp; -1 disables.
  int64_t target_batch_size = -1;
  /// Steps over which batch size doubles from start to target.
  int64_t batch_size_ramp_steps = 0;

  // Performance optimizations
  /// Use the `_foreach_*` batched primitives inside AdamW (much faster
  /// on GPU since it cuts per-parameter kernel launches).
  bool use_foreach_optimizer = true;  // Use _foreach_ batched ops in AdamW
  /// ZeRO-1: shard optimizer state across DP ranks. Each rank only
  /// owns and updates 1/world_size of the parameters; an allgather
  /// after step() syncs the updated weights. Single-rank: no-op.
  bool use_zero1 = false;
  /// FSDP (ZeRO-3): shard params + grads + optimizer state across DP ranks.
  /// Each rank stores 1/world_size of every weight; the full param is
  /// reconstructed (allgather) only for forward/backward, then re-sharded.
  /// Mutually exclusive with use_cuda_graph (unshard/reshard can't be captured).
  bool use_fsdp = false;
  /// Place the entire tokenised dataset on GPU memory if it fits.
  bool gpu_resident_data = true;      // If false: pinned host + H2D streaming (no full corpus on GPU)
  /// 0 = auto VRAM budget for full GPU residency; >0 = max token count allowed on GPU;
  /// -1 = never put full corpus on GPU (streaming only).
  int64_t max_gpu_data_tokens = 0;
  /// Steps between log-print and loss D2H sync. Larger = less host
  /// blocking, looser monitoring.
  int64_t log_interval = 10;         // Steps between loss D2H sync
  /// Capture fwd+bwd as a CUDA graph and replay each step. Eliminates
  /// per-kernel launch overhead. Requires fixed shapes (no curriculum).
  /// Default ON (item BB): the runtime gate (device.is_cuda() && !ddp) in
  /// train.cpp ensures CPU/MPS/multi-rank runs auto-skip; users who want
  /// to disable on CUDA for ablation can set use_cuda_graph=0 in [optimization].
  bool use_cuda_graph = true;

  // Heartbeat monitoring
  /// Wall-seconds between heartbeat-file writes (calibrated dynamically
  /// against avg epoch time). 0 disables.
  double report_every = 300.0;       // seconds between heartbeat writes (0 = disabled)
  /// Path to a small text file periodically updated with epoch/step info,
  /// for external watchdog scripts. Empty = disabled.
  std::string heartbeat_path;        // file to write heartbeat to (empty = disabled)

  // Async Muon: run Newton-Schulz orthogonalization on a side CUDA stream
  /// Overlap Muon's expensive matrix-iteration step with the next forward
  /// pass on a separate CUDA stream.
  bool async_muon = false;

  // Speculative Gradient Prediction (SGP)
  /// Master switch for SGP: predict gradients for k-1 of every k steps
  /// using a linear (v1) or low-rank (v2) predictor; do real backward
  /// only on the kth step.
  bool sgp_enabled = false;
  /// 1 = linear predictor (v1); 2 = rank-r subspace (v2).
  int64_t sgp_version = 1;         // 1 = linear predictor (v1); 2 = rank-r subspace (v2)
  /// Initial k value (predicts k-1 steps, runs real backward on the kth).
  int64_t sgp_initial_k = 2;
  /// Maximum k SGP will adapt to during training.
  int64_t sgp_max_k = 8;
  /// Number of steps SGP runs in observation-only mode before predicting.
  int64_t sgp_warmup_steps = 100;
  /// Rank for v2's low-rank gradient predictor.
  int64_t sgp_rank = 4;            // rank for v2 subspace predictor

  // Gradient statistics (Phase 0 of SGP research track)
  /// CSV path for periodic gradient-statistic dumps. Empty = disabled.
  std::string grad_stats_path;       // empty = disabled; path to CSV output
  /// Steps between gradient stat measurements.
  int64_t grad_stats_interval = 10;  // steps between measurements
};

/// Train for num_steps (legacy API, kept for backward compat)
/// Older entry point retained for benchmarks that pre-date the
/// `TrainConfig` struct. New code should use `train()` below.
void train_epoch(
    Transformer& model,
    const TransformerConfig& cfg,
    int64_t num_steps,
    std::optional<std::string> data_path = std::nullopt,
    int64_t batch_size = 4,
    int64_t seq_len = 128,
    double lr = 1e-4,
    int64_t warmup_steps = 100,
    torch::Device device = torch::Device(torch::kCPU),
    int64_t grad_accum_steps = 1,
    bool use_amp = false,
    const std::string& optimizer_name = "adamw");

/// Full-featured training with callbacks, schedulers, checkpointing.
/// Modern entry point for the standard (non-fused) Transformer. Owns the
/// optimizer, LR scheduler, dataset iterator, DDP allreduce hooks,
/// gradient scaler, callback manager, and checkpoint manager for the
/// duration of the run. Returns when `train_cfg.num_steps` steps complete.
void train(
    Transformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& train_cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks = {},
    std::shared_ptr<Evaluator> evaluator = nullptr);

/// Full-featured training for FusedTransformer
/// Same loop semantics as the non-fused overload but tied to the fused
/// model type so the static dispatch through `model->forward(...)` calls
/// the fused QKV / fused gate-up FFN path. Also the only `train()`
/// overload that wires in CUDA graph capture and SGP.
void train(
    FusedTransformer& model,
    const TransformerConfig& model_cfg,
    const TrainConfig& train_cfg,
    torch::Device device,
    std::vector<std::shared_ptr<Callback>> callbacks = {},
    std::shared_ptr<Evaluator> evaluator = nullptr);

}  // namespace olmo_cpp
