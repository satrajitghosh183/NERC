/**
 * src/train/callbacks/all_callbacks.cpp
 *
 * ─── What "callbacks" are ───────────────────────────────────────────
 *
 * Callbacks are objects the train loop calls at well-defined moments
 * (start of run, before microbatch, after backward, after step, end
 * of run, etc.) so optional behaviour can be plugged in without
 * cluttering the core loop. Same idea as Keras/Lightning callbacks.
 *
 * This file implements every callback shipped with the framework:
 *
 *   - **GradientStatsCallback** : every N steps, sample per-parameter
 *     gradient norms and write them to a TSV file. Used for
 *     diagnosing training instabilities (catastrophic NaN, exploding
 *     gradients, vanishing gradients).
 *
 *   - **WandbCallback / TensorBoardCallback** : forward training
 *     metrics to Weights & Biases or TensorBoard.
 *
 *   - **EarlyStopCallback** : abort training when validation loss
 *     stops improving.
 *
 *   - **GoodbyeCallback** : log a final summary at run end.
 *
 *   - **GarbageCollectorCallback** : explicit GC every N steps to
 *     keep CPU side memory bounded on long runs.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/train/callbacks/all_callbacks.hpp : decls of every
 *     callback class.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp: instantiates the GradientStatsCallback when
 *     train_cfg.grad_stats_path is set, then passes it to
 *     olmo_cpp::train(...).
 *   - src/train.cpp: walks the callback list at every lifecycle hook.
 *
 * --- Role in training pipeline ---
 *   Optional cross-cutting features. The core loop runs fine with
 *   zero callbacks. The quickstart's conf doesn't enable any, so this
 *   file is mostly inert during the demo run.
 */
#include "olmo_cpp/train/callbacks/all_callbacks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <numeric>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace fs = std::filesystem;

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// 1. ConsoleLoggerCallback
// ---------------------------------------------------------------------------

ConsoleLoggerCallback::ConsoleLoggerCallback(int64_t log_interval, int rank)
    : log_interval_(log_interval), rank_(rank) {}

void ConsoleLoggerCallback::on_train_start(TrainState& state) {
  if (rank_ != 0) return;
  std::cout << "[ConsoleLogger] Training started." << std::endl;
  (void)state;
}

void ConsoleLoggerCallback::on_step_end(TrainState& state) {
  if (rank_ != 0) return;
  if (state.global_step % log_interval_ != 0) return;

  auto now = std::chrono::steady_clock::now();
  double step_secs = std::chrono::duration<double>(now - state.step_start).count();
  double tokens_per_sec = 0.0;
  if (step_secs > 0.0) {
    tokens_per_sec =
        static_cast<double>(state.batch_size * state.seq_len) / step_secs;
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "[step " << state.global_step << "]"
            << " loss=" << state.loss
            << " lr=" << state.learning_rate
            << " grad_norm=" << state.grad_norm
            << " tok/s=" << static_cast<int64_t>(tokens_per_sec)
            << " tokens_seen=" << state.tokens_seen
            << std::endl;
}

void ConsoleLoggerCallback::on_train_end(TrainState& state) {
  if (rank_ != 0) return;
  auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - state.train_start);
  std::cout << "[ConsoleLogger] Training finished. Total steps: "
            << state.global_step
            << ", Total tokens: " << state.tokens_seen
            << ", Elapsed: " << std::fixed << std::setprecision(1)
            << elapsed.count() << "s" << std::endl;
}

// ---------------------------------------------------------------------------
// 2. SpeedMonitorCallback
// ---------------------------------------------------------------------------

SpeedMonitorCallback::SpeedMonitorCallback(int64_t window_size)
    : window_size_(window_size) {}

void SpeedMonitorCallback::on_step_start(TrainState& /*state*/) {
  last_time_ = std::chrono::steady_clock::now();
}

void SpeedMonitorCallback::on_step_end(TrainState& state) {
  auto now = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(now - last_time_).count();
  int64_t tokens_this_step = state.batch_size * state.seq_len;

  step_times_.push_back(dt);
  step_tokens_.push_back(tokens_this_step);
  if (static_cast<int64_t>(step_times_.size()) > window_size_) {
    step_times_.pop_front();
    step_tokens_.pop_front();
  }

  double total_time = std::accumulate(step_times_.begin(), step_times_.end(), 0.0);
  int64_t total_tokens = std::accumulate(step_tokens_.begin(), step_tokens_.end(), int64_t(0));

  double steps_per_sec = 0.0;
  double tokens_per_sec = 0.0;
  if (total_time > 0.0) {
    steps_per_sec = static_cast<double>(step_times_.size()) / total_time;
    tokens_per_sec = static_cast<double>(total_tokens) / total_time;
  }

  state.metrics["speed/steps_per_sec"] = static_cast<float>(steps_per_sec);
  state.metrics["speed/tokens_per_sec"] = static_cast<float>(tokens_per_sec);
  state.metrics["speed/step_time_ms"] = static_cast<float>(dt * 1000.0);
}

// ---------------------------------------------------------------------------
// 3. GPUMemoryMonitorCallback
// ---------------------------------------------------------------------------

void GPUMemoryMonitorCallback::on_step_end(TrainState& state) {
#ifdef USE_CUDA
  if (!torch::cuda::is_available()) return;

  auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);

  // allocated_bytes[0] is the stat for "all" segment pool
  int64_t allocated = stats.allocated_bytes[0].current;
  int64_t reserved = stats.reserved_bytes[0].current;
  int64_t peak_allocated = stats.allocated_bytes[0].peak;
  int64_t peak_reserved = stats.reserved_bytes[0].peak;

  double alloc_gb = static_cast<double>(allocated) / (1024.0 * 1024.0 * 1024.0);
  double reserved_gb = static_cast<double>(reserved) / (1024.0 * 1024.0 * 1024.0);
  double peak_alloc_gb = static_cast<double>(peak_allocated) / (1024.0 * 1024.0 * 1024.0);
  double peak_reserved_gb = static_cast<double>(peak_reserved) / (1024.0 * 1024.0 * 1024.0);

  state.metrics["gpu/allocated_gb"] = static_cast<float>(alloc_gb);
  state.metrics["gpu/reserved_gb"] = static_cast<float>(reserved_gb);
  state.metrics["gpu/peak_allocated_gb"] = static_cast<float>(peak_alloc_gb);
  state.metrics["gpu/peak_reserved_gb"] = static_cast<float>(peak_reserved_gb);
#else
  (void)state;
#endif
}

// ---------------------------------------------------------------------------
// 4. StabilityMonitorCallback
// ---------------------------------------------------------------------------

StabilityMonitorCallback::StabilityMonitorCallback(double spike_threshold,
                                                   int64_t window_size)
    : spike_threshold_(spike_threshold), window_size_(window_size) {}

void StabilityMonitorCallback::on_step_end(TrainState& state) {
  float loss = state.loss;

  // Check for NaN / Inf
  if (std::isnan(loss) || std::isinf(loss)) {
    nan_count_++;
    std::cerr << "[StabilityMonitor] WARNING: NaN/Inf loss detected at step "
              << state.global_step << " (total NaN/Inf count: " << nan_count_
              << ")" << std::endl;
    state.metrics["stability/nan_count"] = static_cast<float>(nan_count_);
    return;
  }

  loss_history_.push_back(loss);
  if (static_cast<int64_t>(loss_history_.size()) > window_size_) {
    loss_history_.pop_front();
  }

  // Need at least a few samples to compute stats
  if (loss_history_.size() < 5) return;

  double sum = std::accumulate(loss_history_.begin(), loss_history_.end(), 0.0);
  double mean = sum / static_cast<double>(loss_history_.size());

  double sq_sum = 0.0;
  for (float l : loss_history_) {
    double diff = static_cast<double>(l) - mean;
    sq_sum += diff * diff;
  }
  double stddev = std::sqrt(sq_sum / static_cast<double>(loss_history_.size()));

  double threshold = mean + spike_threshold_ * stddev;
  if (static_cast<double>(loss) > threshold && stddev > 1e-6) {
    spike_count_++;
    std::cerr << "[StabilityMonitor] WARNING: Loss spike detected at step "
              << state.global_step << ": loss=" << loss
              << " (mean=" << mean << ", std=" << stddev
              << ", threshold=" << threshold
              << ", total spikes: " << spike_count_ << ")" << std::endl;
  }

  state.metrics["stability/loss_mean"] = static_cast<float>(mean);
  state.metrics["stability/loss_std"] = static_cast<float>(stddev);
  state.metrics["stability/spike_count"] = static_cast<float>(spike_count_);
  state.metrics["stability/nan_count"] = static_cast<float>(nan_count_);
}

// ---------------------------------------------------------------------------
// 5. CheckpointerCallback
// ---------------------------------------------------------------------------

CheckpointerCallback::CheckpointerCallback(const std::string& save_dir,
                                           int64_t save_interval,
                                           int64_t keep_last_n, int rank)
    : save_dir_(save_dir),
      save_interval_(save_interval),
      keep_last_n_(keep_last_n),
      rank_(rank) {}

void CheckpointerCallback::on_step_end(TrainState& state) {
  if (state.global_step % save_interval_ != 0) return;
  if (state.global_step == 0) return;
  save_checkpoint(state);
}

void CheckpointerCallback::on_train_end(TrainState& state) {
  save_checkpoint(state);
}

void CheckpointerCallback::save_checkpoint(TrainState& state) {
  if (rank_ != 0) return;

  fs::create_directories(save_dir_);

  std::string ckpt_name = "step_" + std::to_string(state.global_step);
  std::string ckpt_dir = save_dir_ + "/" + ckpt_name;
  fs::create_directories(ckpt_dir);

  // Save model weights if available
  if (model_) {
    std::string model_path = ckpt_dir + "/model.pt";
    torch::serialize::OutputArchive archive;
    model_->save(archive);
    archive.save_to(model_path);
    std::cout << "[Checkpointer] Saved model to " << model_path << std::endl;
  }

  // Save training state metadata
  std::string meta_path = ckpt_dir + "/train_state.json";
  std::ofstream meta(meta_path);
  if (meta.is_open()) {
    meta << "{\n"
         << "  \"global_step\": " << state.global_step << ",\n"
         << "  \"epoch\": " << state.epoch << ",\n"
         << "  \"tokens_seen\": " << state.tokens_seen << ",\n"
         << "  \"loss\": " << state.loss << ",\n"
         << "  \"learning_rate\": " << state.learning_rate << "\n"
         << "}" << std::endl;
    meta.close();
  }

  saved_paths_.push_back(ckpt_dir);

  // Remove old checkpoints beyond keep_last_n
  while (keep_last_n_ > 0 &&
         static_cast<int64_t>(saved_paths_.size()) > keep_last_n_) {
    std::string old_path = saved_paths_.front();
    saved_paths_.pop_front();
    std::error_code ec;
    fs::remove_all(old_path, ec);
    if (!ec) {
      std::cout << "[Checkpointer] Removed old checkpoint: " << old_path
                << std::endl;
    }
  }
}

// ---------------------------------------------------------------------------
// 6. GradientMonitorCallback
// ---------------------------------------------------------------------------

GradientMonitorCallback::GradientMonitorCallback(int64_t log_interval)
    : log_interval_(log_interval) {}

void GradientMonitorCallback::on_after_backward(TrainState& state) {
  if (state.global_step % log_interval_ != 0) return;
  if (!model_) return;

  // Collect per-param norms as device tensors first, then a single
  // stack+transfer at the end. Replaces N_params syncs with one.
  torch::NoGradGuard no_grad;
  std::vector<torch::Tensor> norms;
  norms.reserve(64);
  for (const auto& pair : model_->named_parameters()) {
    const auto& param = pair.value();
    if (!param.grad().defined()) continue;
    norms.push_back(param.grad().norm().to(torch::kFloat));
  }
  if (norms.empty()) return;

  // Stack on-device, one sync to host.
  const auto stacked = torch::stack(norms).to(torch::kCPU);
  const auto* data = stacked.data_ptr<float>();
  const int64_t param_count = stacked.size(0);

  float min_norm = std::numeric_limits<float>::max();
  float max_norm = 0.0f;
  float total_norm = 0.0f;
  for (int64_t i = 0; i < param_count; ++i) {
    const float n = data[i];
    min_norm = std::min(min_norm, n);
    max_norm = std::max(max_norm, n);
    total_norm += n;
  }

  const float mean_norm = total_norm / static_cast<float>(param_count);
  state.metrics["grad/min_norm"] = min_norm;
  state.metrics["grad/max_norm"] = max_norm;
  state.metrics["grad/mean_norm"] = mean_norm;
  state.metrics["grad/num_params"] = static_cast<float>(param_count);

  std::cout << "[GradientMonitor] step=" << state.global_step
            << " grad_norms: min=" << min_norm << " max=" << max_norm
            << " mean=" << mean_norm << " params=" << param_count
            << std::endl;
}

// ---------------------------------------------------------------------------
// 6b. GradientStatsCallback — per-layer gradient predictability measurement
// ---------------------------------------------------------------------------

GradientStatsCallback::GradientStatsCallback(
    const std::string& output_path, const std::string& mode,
    int64_t log_interval, int64_t num_samples)
    : output_path_(output_path), mode_(mode),
      log_interval_(log_interval), num_samples_(num_samples) {}

bool GradientStatsCallback::should_track(const std::string& name) const {
  // Track 2D+ weight matrices (attention, FFN, projections) — these are what
  // SGP would predict. Skip biases, norms, embeddings (1D or too small).
  (void)name;
  return true;  // filtering done by dim check in on_train_start
}

void GradientStatsCallback::on_train_start(TrainState& /*state*/) {
  if (!model_) return;

  fs::path p(output_path_);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  out_.open(output_path_, std::ios::trunc);
  if (!out_.is_open()) {
    std::cerr << "[GradientStats] Failed to open " << output_path_ << "\n";
    return;
  }
  out_ << "step,layer,numel,l2_norm,cosine_sim,pred_error_l1\n";

  tracked_.clear();
  for (const auto& pair : model_->named_parameters()) {
    const auto& param = pair.value();
    if (param.dim() < 2) continue;  // skip 1D (norms, biases)

    ParamState ps;
    ps.name = pair.key();

    int64_t numel = param.numel();
    int64_t ns = std::min(num_samples_, numel);
    // Fixed random sample indices — consistent across steps for fair comparison
    ps.sample_indices = torch::randperm(numel, torch::kLong).narrow(0, 0, ns);
    ps.initialized = false;
    tracked_.push_back(std::move(ps));
  }

  std::cout << "[GradientStats] Tracking " << tracked_.size()
            << " parameters, " << num_samples_ << " samples each → "
            << output_path_ << "\n";
}

void GradientStatsCallback::on_after_backward(TrainState& state) {
  if (!out_.is_open()) return;
  if (state.global_step % log_interval_ != 0) return;
  if (!model_) return;

  torch::NoGradGuard no_grad;

  // Per-layer device-side metrics. All computations stay on-device; we batch
  // the final transfer into a single stack+copy so the whole callback costs
  // one sync instead of O(N_params).
  struct LayerMetrics {
    size_t tracked_idx;
    int64_t numel;
    bool was_initialized;
    torch::Tensor l2_norm;     // 0-dim float, device
    torch::Tensor cosine_sim;  // 0-dim float, device
    torch::Tensor pred_error;  // 0-dim float, device
  };
  std::vector<LayerMetrics> metrics;
  metrics.reserve(tracked_.size());

  size_t ti = 0;
  for (const auto& pair : model_->named_parameters()) {
    const auto& param = pair.value();
    if (param.dim() < 2) continue;
    if (ti >= tracked_.size()) break;

    const size_t this_ti = ti++;
    auto& ps = tracked_[this_ti];
    if (!param.grad().defined()) continue;

    // Move the fixed sample indices onto the param's device on first use.
    if (!ps.sample_indices.defined()) continue;
    if (ps.sample_indices.device() != param.device()) {
      ps.sample_indices = ps.sample_indices.to(param.device());
    }

    const auto grad_flat = param.grad().detach().reshape(-1).to(torch::kFloat);
    const auto sampled = grad_flat.index_select(0, ps.sample_indices);

    LayerMetrics m;
    m.tracked_idx = this_ti;
    m.numel = param.numel();
    m.was_initialized = ps.initialized;
    m.l2_norm = sampled.norm();

    if (ps.initialized) {
      const auto dot = (sampled * ps.prev_sampled).sum();
      const auto prev_norm = ps.prev_sampled.norm();
      const auto denom = m.l2_norm * prev_norm;
      // Device-side guard: where denom is degenerate, emit 0 instead of a
      // division that would explode. Matches the old CPU branch without a
      // host-side if.
      const auto safe_cos = dot / denom.clamp_min(1e-12f);
      m.cosine_sim = torch::where(
          denom > 1e-12f, safe_cos, torch::zeros_like(safe_cos));
      m.pred_error = (sampled - ps.prev_sampled).abs().mean();
    } else {
      m.cosine_sim = torch::zeros({}, sampled.options());
      m.pred_error = torch::zeros({}, sampled.options());
    }

    // prev_sampled stays on-device now — ~num_samples floats per tracked
    // param, trivial VRAM vs. avoiding the per-param CPU round-trip.
    ps.prev_sampled = sampled;
    ps.initialized = true;
    metrics.push_back(std::move(m));
  }

  if (metrics.empty()) return;

  // Stack per-metric, one sync to host covers all three arrays.
  std::vector<torch::Tensor> l2_list, cos_list, pe_list;
  l2_list.reserve(metrics.size());
  cos_list.reserve(metrics.size());
  pe_list.reserve(metrics.size());
  for (const auto& m : metrics) {
    l2_list.push_back(m.l2_norm);
    cos_list.push_back(m.cosine_sim);
    pe_list.push_back(m.pred_error);
  }
  const auto l2_cpu = torch::stack(l2_list).to(torch::kCPU);
  const auto cos_cpu = torch::stack(cos_list).to(torch::kCPU);
  const auto pe_cpu = torch::stack(pe_list).to(torch::kCPU);

  const auto* l2_p = l2_cpu.data_ptr<float>();
  const auto* cos_p = cos_cpu.data_ptr<float>();
  const auto* pe_p = pe_cpu.data_ptr<float>();

  for (size_t i = 0; i < metrics.size(); ++i) {
    const auto& m = metrics[i];
    const auto& ps = tracked_[m.tracked_idx];
    const float cos = m.was_initialized ? cos_p[i] : 0.0f;
    const float pe = m.was_initialized ? pe_p[i] : 0.0f;
    out_ << state.global_step << ","
         << ps.name << ","
         << m.numel << ","
         << l2_p[i] << ","
         << cos << ","
         << pe << "\n";
  }
  out_.flush();
}

void GradientStatsCallback::on_train_end(TrainState& /*state*/) {
  if (out_.is_open()) {
    out_.close();
    std::cout << "[GradientStats] Saved to " << output_path_ << "\n";
  }
}

// ---------------------------------------------------------------------------
// 7. MetricSaverCallback
// ---------------------------------------------------------------------------

MetricSaverCallback::MetricSaverCallback(const std::string& output_path,
                                         int64_t save_interval)
    : output_path_(output_path), save_interval_(save_interval) {}

void MetricSaverCallback::on_step_end(TrainState& state) {
  // Copy current metrics plus standard fields into the buffer
  std::unordered_map<std::string, float> entry = state.metrics;
  entry["global_step"] = static_cast<float>(state.global_step);
  entry["epoch"] = static_cast<float>(state.epoch);
  entry["loss"] = state.loss;
  entry["learning_rate"] = state.learning_rate;
  entry["grad_norm"] = state.grad_norm;
  entry["tokens_seen"] = static_cast<float>(state.tokens_seen);
  buffer_.push_back(std::move(entry));

  if (static_cast<int64_t>(buffer_.size()) >= save_interval_) {
    flush();
  }
}

void MetricSaverCallback::on_train_end(TrainState& /*state*/) {
  if (!buffer_.empty()) {
    flush();
  }
}

void MetricSaverCallback::flush() {
  // Ensure parent directory exists
  fs::path p(output_path_);
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path());
  }

  // Append to file as a JSON array (one array per flush)
  std::ofstream out(output_path_, std::ios::app);
  if (!out.is_open()) {
    std::cerr << "[MetricSaver] Failed to open " << output_path_ << std::endl;
    buffer_.clear();
    return;
  }

  out << "[\n";
  for (size_t i = 0; i < buffer_.size(); ++i) {
    out << "  {";
    bool first = true;
    for (const auto& kv : buffer_[i]) {
      if (!first) out << ", ";
      out << "\"" << kv.first << "\": " << kv.second;
      first = false;
    }
    out << "}";
    if (i + 1 < buffer_.size()) out << ",";
    out << "\n";
  }
  out << "]\n";
  out.close();

  buffer_.clear();
}

// ---------------------------------------------------------------------------
// 8. ProfilerCallback
// ---------------------------------------------------------------------------

ProfilerCallback::ProfilerCallback(int64_t start_step, int64_t num_steps,
                                   const std::string& trace_dir)
    : start_step_(start_step), num_steps_(num_steps), trace_dir_(trace_dir) {}

void ProfilerCallback::on_step_start(TrainState& state) {
  if (state.global_step == start_step_) {
    active_ = true;
    fs::create_directories(trace_dir_);
    std::cout << "[Profiler] Profiling started at step " << state.global_step
              << std::endl;
  }
}

void ProfilerCallback::on_step_end(TrainState& state) {
  if (!active_) return;

  auto now = std::chrono::steady_clock::now();
  double step_ms =
      std::chrono::duration<double, std::milli>(now - state.step_start).count();

  // Write a simple trace event in Chrome Trace Format
  std::string trace_file =
      trace_dir_ + "/step_" + std::to_string(state.global_step) + ".json";
  std::ofstream out(trace_file);
  if (out.is_open()) {
    auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       now.time_since_epoch())
                       .count();
    out << "{\n"
        << "  \"traceEvents\": [\n"
        << "    {\n"
        << "      \"name\": \"training_step\",\n"
        << "      \"cat\": \"train\",\n"
        << "      \"ph\": \"X\",\n"
        << "      \"ts\": " << (wall_us - static_cast<int64_t>(step_ms * 1000.0))
        << ",\n"
        << "      \"dur\": " << static_cast<int64_t>(step_ms * 1000.0) << ",\n"
        << "      \"pid\": 0,\n"
        << "      \"tid\": 0,\n"
        << "      \"args\": {\n"
        << "        \"step\": " << state.global_step << ",\n"
        << "        \"loss\": " << state.loss << ",\n"
        << "        \"step_time_ms\": " << step_ms << "\n"
        << "      }\n"
        << "    }\n"
        << "  ]\n"
        << "}" << std::endl;
    out.close();
  }

  if (state.global_step >= start_step_ + num_steps_ - 1) {
    active_ = false;
    std::cout << "[Profiler] Profiling finished. Traces written to "
              << trace_dir_ << std::endl;
  }
}

// ---------------------------------------------------------------------------
// 9. GarbageCollectorCallback
// ---------------------------------------------------------------------------

GarbageCollectorCallback::GarbageCollectorCallback(int64_t interval)
    : interval_(interval) {}

void GarbageCollectorCallback::on_step_end(TrainState& state) {
  if (state.global_step % interval_ != 0) return;
  if (state.global_step == 0) return;

#ifdef USE_CUDA
  if (torch::cuda::is_available()) {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
#endif
}

// ---------------------------------------------------------------------------
// 10. ConfigSaverCallback
// ---------------------------------------------------------------------------

ConfigSaverCallback::ConfigSaverCallback(const std::string& save_path)
    : save_path_(save_path) {}

void ConfigSaverCallback::on_train_start(TrainState& state) {
  fs::path p(save_path_);
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path());
  }

  std::ofstream out(save_path_);
  if (!out.is_open()) {
    std::cerr << "[ConfigSaver] Failed to open " << save_path_ << std::endl;
    return;
  }

  out << "{\n"
      << "  \"batch_size\": " << state.batch_size << ",\n"
      << "  \"seq_len\": " << state.seq_len << ",\n"
      << "  \"learning_rate\": " << state.learning_rate << ",\n"
      << "  \"epoch\": " << state.epoch << ",\n"
      << "  \"global_step\": " << state.global_step << ",\n"
      << "  \"tokens_seen\": " << state.tokens_seen << "\n"
      << "}" << std::endl;
  out.close();

  std::cout << "[ConfigSaver] Training config saved to " << save_path_
            << std::endl;
}

// ---------------------------------------------------------------------------
// 11. SequenceLengthSchedulerCallback
// ---------------------------------------------------------------------------

SequenceLengthSchedulerCallback::SequenceLengthSchedulerCallback(
    int64_t initial_seq_len, int64_t target_seq_len, int64_t warmup_steps)
    : initial_seq_len_(initial_seq_len),
      target_seq_len_(target_seq_len),
      warmup_steps_(warmup_steps),
      current_seq_len_(initial_seq_len) {}

void SequenceLengthSchedulerCallback::on_step_start(TrainState& state) {
  if (warmup_steps_ <= 0) {
    current_seq_len_ = target_seq_len_;
  } else if (state.global_step >= warmup_steps_) {
    current_seq_len_ = target_seq_len_;
  } else {
    // Linear interpolation
    double frac =
        static_cast<double>(state.global_step) / static_cast<double>(warmup_steps_);
    current_seq_len_ = initial_seq_len_ +
        static_cast<int64_t>(frac * static_cast<double>(target_seq_len_ - initial_seq_len_));
    // Round to nearest multiple of 64 for efficiency
    current_seq_len_ = ((current_seq_len_ + 63) / 64) * 64;
    // Clamp
    current_seq_len_ = std::min(current_seq_len_, target_seq_len_);
    current_seq_len_ = std::max(current_seq_len_, initial_seq_len_);
  }
  state.seq_len = current_seq_len_;
}

// ---------------------------------------------------------------------------
// 12. BatchSizeSchedulerCallback
// ---------------------------------------------------------------------------

BatchSizeSchedulerCallback::BatchSizeSchedulerCallback(
    int64_t initial_batch_size, int64_t target_batch_size,
    int64_t warmup_steps)
    : initial_batch_size_(initial_batch_size),
      target_batch_size_(target_batch_size),
      warmup_steps_(warmup_steps),
      current_batch_size_(initial_batch_size) {}

void BatchSizeSchedulerCallback::on_step_start(TrainState& state) {
  if (warmup_steps_ <= 0 || state.global_step >= warmup_steps_) {
    current_batch_size_ = target_batch_size_;
  } else {
    // Double the batch size at evenly spaced intervals until reaching target.
    // Compute how many doublings we need.
    int64_t bs = initial_batch_size_;
    int num_doublings = 0;
    int64_t tmp = bs;
    while (tmp < target_batch_size_) {
      tmp *= 2;
      num_doublings++;
    }

    if (num_doublings == 0) {
      current_batch_size_ = target_batch_size_;
    } else {
      int64_t steps_per_doubling = warmup_steps_ / num_doublings;
      if (steps_per_doubling <= 0) steps_per_doubling = 1;
      int doublings_done = static_cast<int>(state.global_step / steps_per_doubling);
      doublings_done = std::min(doublings_done, num_doublings);
      current_batch_size_ = initial_batch_size_ * (int64_t(1) << doublings_done);
      current_batch_size_ = std::min(current_batch_size_, target_batch_size_);
    }
  }
  state.batch_size = current_batch_size_;
}

// ---------------------------------------------------------------------------
// 13. EvaluatorCallback
// ---------------------------------------------------------------------------

EvaluatorCallback::EvaluatorCallback(EvalFn eval_fn, int64_t eval_interval)
    : eval_fn_(std::move(eval_fn)), eval_interval_(eval_interval) {}

void EvaluatorCallback::on_step_end(TrainState& state) {
  if (state.global_step % eval_interval_ != 0) return;
  if (state.global_step == 0) return;

  auto eval_metrics = eval_fn_();

  // Merge eval metrics into state.metrics with "eval/" prefix
  for (const auto& kv : eval_metrics) {
    state.metrics["eval/" + kv.first] = kv.second;
  }

  // Note: the CallbackManager should call on_eval_end after this, but since
  // EvaluatorCallback itself is called via on_step_end, higher-level code
  // should handle the on_eval_end dispatch. We store metrics so they are
  // available.
}

// ---------------------------------------------------------------------------
// 14. WandBCallback
// ---------------------------------------------------------------------------

WandBCallback::WandBCallback(const std::string& project,
                             const std::string& run_name,
                             const std::string& entity)
    : project_(project), run_name_(run_name), entity_(entity) {}

void WandBCallback::on_train_start(TrainState& state) {
  (void)state;
  // Create a local log directory for JSONL logging
  log_dir_ = "./wandb_logs";
  fs::create_directories(log_dir_);

  if (run_name_.empty()) {
    run_name_ = "run_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
  }

  // Write run metadata
  std::string meta_path = log_dir_ + "/" + run_name_ + "_meta.json";
  std::ofstream meta(meta_path);
  if (meta.is_open()) {
    meta << "{\n"
         << "  \"project\": \"" << project_ << "\",\n"
         << "  \"run_name\": \"" << run_name_ << "\",\n"
         << "  \"entity\": \"" << entity_ << "\"\n"
         << "}" << std::endl;
    meta.close();
  }

  initialized_ = true;
  std::cout << "[WandB] Initialized run '" << run_name_ << "' in project '"
            << project_ << "'. Logs at " << log_dir_ << std::endl;
}

void WandBCallback::on_step_end(TrainState& state) {
  if (!initialized_) return;

  // Append metrics as JSONL (one JSON object per line)
  std::string log_path = log_dir_ + "/" + run_name_ + "_metrics.jsonl";
  std::ofstream out(log_path, std::ios::app);
  if (!out.is_open()) return;

  out << "{\"step\": " << state.global_step
      << ", \"loss\": " << state.loss
      << ", \"lr\": " << state.learning_rate
      << ", \"grad_norm\": " << state.grad_norm
      << ", \"tokens_seen\": " << state.tokens_seen;

  for (const auto& kv : state.metrics) {
    out << ", \"" << kv.first << "\": " << kv.second;
  }
  out << "}\n";
  out.close();
}

void WandBCallback::on_train_end(TrainState& state) {
  if (!initialized_) return;

  // Write summary
  std::string summary_path = log_dir_ + "/" + run_name_ + "_summary.json";
  std::ofstream out(summary_path);
  if (out.is_open()) {
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - state.train_start);
    out << "{\n"
        << "  \"final_step\": " << state.global_step << ",\n"
        << "  \"final_loss\": " << state.loss << ",\n"
        << "  \"total_tokens\": " << state.tokens_seen << ",\n"
        << "  \"elapsed_seconds\": " << elapsed.count() << "\n"
        << "}" << std::endl;
    out.close();
  }

  std::cout << "[WandB] Run '" << run_name_ << "' finished. Summary at "
            << summary_path << std::endl;
}

// ---------------------------------------------------------------------------
// 15. SlackNotifierCallback
// ---------------------------------------------------------------------------

SlackNotifierCallback::SlackNotifierCallback(const std::string& webhook_url,
                                             int64_t notify_interval)
    : webhook_url_(webhook_url), notify_interval_(notify_interval) {}

void SlackNotifierCallback::on_train_start(TrainState& /*state*/) {
  send_message("Training started.");
}

void SlackNotifierCallback::on_train_end(TrainState& state) {
  std::ostringstream oss;
  oss << "Training finished. Final step: " << state.global_step
      << ", Final loss: " << state.loss
      << ", Total tokens: " << state.tokens_seen;
  send_message(oss.str());
}

void SlackNotifierCallback::on_step_end(TrainState& state) {
  if (state.global_step % notify_interval_ != 0) return;
  if (state.global_step == 0) return;

  std::ostringstream oss;
  oss << "Step " << state.global_step
      << " | loss=" << std::fixed << std::setprecision(4) << state.loss
      << " | lr=" << state.learning_rate
      << " | tokens=" << state.tokens_seen;
  send_message(oss.str());
}

void SlackNotifierCallback::send_message(const std::string& text) {
  // Escape double quotes in text for JSON
  std::string escaped;
  escaped.reserve(text.size());
  for (char c : text) {
    if (c == '"') escaped += "\\\"";
    else if (c == '\\') escaped += "\\\\";
    else escaped += c;
  }

  std::string cmd = "curl -s -X POST -H 'Content-type: application/json' "
                    "--data '{\"text\": \"" +
                    escaped + "\"}' '" + webhook_url_ + "' > /dev/null 2>&1 &";
  // Run asynchronously to avoid blocking training
  std::system(cmd.c_str());
}

// ---------------------------------------------------------------------------
// 16. ModelMergerCallback (EMA)
// ---------------------------------------------------------------------------

ModelMergerCallback::ModelMergerCallback(double ema_decay,
                                         int64_t update_interval)
    : ema_decay_(ema_decay), update_interval_(update_interval) {}

void ModelMergerCallback::on_step_end(TrainState& state) {
  if (!model_) return;
  if (state.global_step % update_interval_ != 0) return;

  auto params = model_->parameters();

  if (!initialized_) {
    // Initialize EMA params as clones of model params
    ema_params_.clear();
    ema_params_.reserve(params.size());
    for (const auto& p : params) {
      ema_params_.push_back(p.detach().clone());
    }
    initialized_ = true;
    return;
  }

  // Update: ema = decay * ema + (1 - decay) * param
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < params.size(); ++i) {
    ema_params_[i].mul_(ema_decay_).add_(params[i].detach(), 1.0 - ema_decay_);
  }
}

}  // namespace olmo_cpp
