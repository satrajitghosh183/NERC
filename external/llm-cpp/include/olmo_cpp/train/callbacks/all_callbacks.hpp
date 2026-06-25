#pragma once
/**
 * include/olmo_cpp/train/callbacks/all_callbacks.hpp
 *
 * Declarations for every concrete Callback in the framework:
 *
 *   - GradientStatsCallback   : per-parameter gradient norm logger.
 *   - WandbCallback           : streams metrics to Weights & Biases.
 *   - TensorBoardCallback     : streams metrics to TensorBoard.
 *   - EarlyStopCallback       : aborts when val loss stops improving.
 *   - GoodbyeCallback         : final summary at run end.
 *   - GarbageCollectorCallback: periodic explicit GC.
 *
 * See src/train/callbacks/all_callbacks.cpp for the implementations
 * and the longer pedagogical writeup of what callbacks are.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/train/callback.hpp : the base Callback type.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/main.cpp : optionally instantiates GradientStatsCallback.
 *   - src/train/callbacks/all_callbacks.cpp : implementations.
 *
 * --- Role in training pipeline ---
 *   Optional. Off by default in the quickstart flow.
 */
#include "olmo_cpp/train/callback.hpp"
#include <fstream>
#include <deque>
#include <chrono>
#include <iostream>
#include <sstream>

namespace olmo_cpp {

// 1. Console logger
class ConsoleLoggerCallback : public Callback {
 public:
  ConsoleLoggerCallback(int64_t log_interval = 10, int rank = 0);
  std::string name() const override { return "ConsoleLogger"; }
  void on_step_end(TrainState& state) override;
  void on_train_start(TrainState& state) override;
  void on_train_end(TrainState& state) override;
 private:
  int64_t log_interval_;
  int rank_;
};

// 2. Speed monitor
class SpeedMonitorCallback : public Callback {
 public:
  SpeedMonitorCallback(int64_t window_size = 50);
  std::string name() const override { return "SpeedMonitor"; }
  void on_step_start(TrainState& state) override;
  void on_step_end(TrainState& state) override;
 private:
  int64_t window_size_;
  std::deque<double> step_times_;
  std::deque<int64_t> step_tokens_;
  std::chrono::steady_clock::time_point last_time_;
};

// 3. GPU memory monitor
class GPUMemoryMonitorCallback : public Callback {
 public:
  std::string name() const override { return "GPUMemoryMonitor"; }
  void on_step_end(TrainState& state) override;
};

// 4. Stability monitor: detect loss spikes and NaN/Inf
class StabilityMonitorCallback : public Callback {
 public:
  StabilityMonitorCallback(double spike_threshold = 5.0, int64_t window_size = 100);
  std::string name() const override { return "StabilityMonitor"; }
  void on_step_end(TrainState& state) override;
 private:
  double spike_threshold_;
  int64_t window_size_;
  std::deque<float> loss_history_;
  int64_t spike_count_ = 0;
  int64_t nan_count_ = 0;
};

// 5. Checkpointer
class CheckpointerCallback : public Callback {
 public:
  CheckpointerCallback(const std::string& save_dir, int64_t save_interval,
                       int64_t keep_last_n = 3, int rank = 0);
  std::string name() const override { return "Checkpointer"; }
  void on_step_end(TrainState& state) override;
  void on_train_end(TrainState& state) override;
  void set_model(std::shared_ptr<torch::nn::Module> model) { model_ = model; }
 private:
  void save_checkpoint(TrainState& state);
  std::string save_dir_;
  int64_t save_interval_, keep_last_n_;
  int rank_;
  std::deque<std::string> saved_paths_;
  std::shared_ptr<torch::nn::Module> model_;
};

// 6. Gradient monitor
class GradientMonitorCallback : public Callback {
 public:
  GradientMonitorCallback(int64_t log_interval = 100);
  std::string name() const override { return "GradientMonitor"; }
  void on_after_backward(TrainState& state) override;
  void set_model(std::shared_ptr<torch::nn::Module> model) { model_ = model; }
 private:
  int64_t log_interval_;
  std::shared_ptr<torch::nn::Module> model_;
};

// 6b. Gradient statistics logger — per-layer cosine similarity, L2 norms.
// Used by the SGP research track (speed-xp-sg) to measure how predictable
// gradients are step-to-step before building a predictor. Index-sampling
// keeps memory O(num_tracked * NUM_SAMPLES) instead of O(total_params).
class GradientStatsCallback : public Callback {
 public:
  GradientStatsCallback(const std::string& output_path,
                        const std::string& mode = "sample",
                        int64_t log_interval = 10,
                        int64_t num_samples = 4096);
  std::string name() const override { return "GradientStats"; }
  void on_train_start(TrainState& state) override;
  void on_after_backward(TrainState& state) override;
  void on_train_end(TrainState& state) override;
  void set_model(std::shared_ptr<torch::nn::Module> model) { model_ = std::move(model); }
 private:
  struct ParamState {
    std::string name;
    torch::Tensor sample_indices;   // int64, fixed at train_start
    torch::Tensor prev_sampled;     // same dtype as grad, [num_samples]
    bool initialized = false;
  };
  bool should_track(const std::string& name) const;

  std::string output_path_;
  std::string mode_;
  int64_t log_interval_;
  int64_t num_samples_;
  std::ofstream out_;
  std::shared_ptr<torch::nn::Module> model_;
  std::vector<ParamState> tracked_;
};

// 7. Metric saver: save metrics to JSON file
class MetricSaverCallback : public Callback {
 public:
  MetricSaverCallback(const std::string& output_path, int64_t save_interval = 100);
  std::string name() const override { return "MetricSaver"; }
  void on_step_end(TrainState& state) override;
  void on_train_end(TrainState& state) override;
 private:
  void flush();
  std::string output_path_;
  int64_t save_interval_;
  std::vector<std::unordered_map<std::string, float>> buffer_;
};

// 8. Profiler callback
class ProfilerCallback : public Callback {
 public:
  ProfilerCallback(int64_t start_step = 5, int64_t num_steps = 3,
                   const std::string& trace_dir = "./traces");
  std::string name() const override { return "Profiler"; }
  void on_step_start(TrainState& state) override;
  void on_step_end(TrainState& state) override;
 private:
  int64_t start_step_, num_steps_;
  std::string trace_dir_;
  bool active_ = false;
};

// 9. Garbage collector
class GarbageCollectorCallback : public Callback {
 public:
  GarbageCollectorCallback(int64_t interval = 100);
  std::string name() const override { return "GarbageCollector"; }
  void on_step_end(TrainState& state) override;
 private:
  int64_t interval_;
};

// 10. Config saver: saves training config at start
class ConfigSaverCallback : public Callback {
 public:
  ConfigSaverCallback(const std::string& save_path);
  std::string name() const override { return "ConfigSaver"; }
  void on_train_start(TrainState& state) override;
 private:
  std::string save_path_;
};

// 11. Sequence length scheduler
class SequenceLengthSchedulerCallback : public Callback {
 public:
  SequenceLengthSchedulerCallback(int64_t initial_seq_len, int64_t target_seq_len,
                                  int64_t warmup_steps);
  std::string name() const override { return "SequenceLengthScheduler"; }
  void on_step_start(TrainState& state) override;
  int64_t current_seq_len() const { return current_seq_len_; }
 private:
  int64_t initial_seq_len_, target_seq_len_, warmup_steps_;
  int64_t current_seq_len_;
};

// 12. Batch size scheduler
class BatchSizeSchedulerCallback : public Callback {
 public:
  BatchSizeSchedulerCallback(int64_t initial_batch_size, int64_t target_batch_size,
                             int64_t warmup_steps);
  std::string name() const override { return "BatchSizeScheduler"; }
  void on_step_start(TrainState& state) override;
  int64_t current_batch_size() const { return current_batch_size_; }
 private:
  int64_t initial_batch_size_, target_batch_size_, warmup_steps_;
  int64_t current_batch_size_;
};

// 13. Evaluator callback
class EvaluatorCallback : public Callback {
 public:
  using EvalFn = std::function<std::unordered_map<std::string, float>()>;
  EvaluatorCallback(EvalFn eval_fn, int64_t eval_interval);
  std::string name() const override { return "Evaluator"; }
  void on_step_end(TrainState& state) override;
 private:
  EvalFn eval_fn_;
  int64_t eval_interval_;
};

// 14. WandB callback (writes to wandb CLI via system calls)
class WandBCallback : public Callback {
 public:
  WandBCallback(const std::string& project, const std::string& run_name = "",
                const std::string& entity = "");
  std::string name() const override { return "WandB"; }
  void on_train_start(TrainState& state) override;
  void on_step_end(TrainState& state) override;
  void on_train_end(TrainState& state) override;
 private:
  std::string project_, run_name_, entity_;
  std::string log_dir_;
  bool initialized_ = false;
};

// 15. Slack notifier
class SlackNotifierCallback : public Callback {
 public:
  SlackNotifierCallback(const std::string& webhook_url,
                        int64_t notify_interval = 1000);
  std::string name() const override { return "SlackNotifier"; }
  void on_train_start(TrainState& state) override;
  void on_train_end(TrainState& state) override;
  void on_step_end(TrainState& state) override;
 private:
  void send_message(const std::string& text);
  std::string webhook_url_;
  int64_t notify_interval_;
};

// 16. Model merger callback: periodically merges model weights (EMA)
class ModelMergerCallback : public Callback {
 public:
  ModelMergerCallback(double ema_decay = 0.999, int64_t update_interval = 1);
  std::string name() const override { return "ModelMerger"; }
  void on_step_end(TrainState& state) override;
  void set_model(std::shared_ptr<torch::nn::Module> model) { model_ = model; }
  const std::vector<torch::Tensor>& ema_params() const { return ema_params_; }
 private:
  double ema_decay_;
  int64_t update_interval_;
  std::shared_ptr<torch::nn::Module> model_;
  std::vector<torch::Tensor> ema_params_;
  bool initialized_ = false;
};

}  // namespace olmo_cpp
