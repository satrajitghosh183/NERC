#pragma once
/**
 * include/olmo_cpp/train/callback.hpp
 *
 * The Callback abstract base class plus the CallbackManager that
 * the train loop walks at every lifecycle hook (begin run, before
 * step, after step, eval, checkpoint, end run). Concrete callbacks
 * live in include/olmo_cpp/train/callbacks/all_callbacks.hpp.
 *
 * Same pattern as Keras / PyTorch Lightning callbacks: a way to add
 * cross-cutting features (metric logging, gradient inspection, early
 * stopping, GC, etc.) without growing the core train loop.
 *
 * --- Includes from this project ---
 *   (none — torch + stdlib only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - include/olmo_cpp/train/callbacks/all_callbacks.hpp : concrete
 *     callbacks subclass Callback.
 *   - src/train.cpp : walks the CallbackManager at each hook.
 *
 * --- Role in training pipeline ---
 *   Optional cross-cutting plumbing.
 */
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <functional>

namespace olmo_cpp {

/// Training state passed to callbacks
struct TrainState {
  int64_t global_step = 0;
  int64_t epoch = 0;
  int64_t tokens_seen = 0;
  float loss = 0.0f;
  float learning_rate = 0.0f;
  float grad_norm = 0.0f;
  int64_t batch_size = 0;
  int64_t seq_len = 0;
  std::chrono::steady_clock::time_point step_start;
  std::chrono::steady_clock::time_point train_start;
  std::unordered_map<std::string, float> metrics;
};

/// Base callback class
class Callback {
 public:
  virtual ~Callback() = default;
  virtual std::string name() const { return "Callback"; }

  /// Called before training begins
  virtual void on_train_start(TrainState& state) { (void)state; }
  /// Called after training ends
  virtual void on_train_end(TrainState& state) { (void)state; }
  /// Called before each training step
  virtual void on_step_start(TrainState& state) { (void)state; }
  /// Called after each training step
  virtual void on_step_end(TrainState& state) { (void)state; }
  /// Called before each epoch
  virtual void on_epoch_start(TrainState& state) { (void)state; }
  /// Called after each epoch
  virtual void on_epoch_end(TrainState& state) { (void)state; }
  /// Called after loss computation (before backward)
  virtual void on_after_loss(TrainState& state) { (void)state; }
  /// Called after backward pass
  virtual void on_after_backward(TrainState& state) { (void)state; }
  /// Called after optimizer step
  virtual void on_after_optimizer_step(TrainState& state) { (void)state; }
  /// Called on checkpoint save
  virtual void on_checkpoint_save(TrainState& state, const std::string& path) { (void)state; (void)path; }
  /// Called on checkpoint load
  virtual void on_checkpoint_load(TrainState& state, const std::string& path) { (void)state; (void)path; }
  /// Called when evaluation completes
  virtual void on_eval_end(TrainState& state, const std::unordered_map<std::string, float>& eval_metrics) { (void)state; (void)eval_metrics; }
};

/// Manages a list of callbacks
class CallbackManager {
 public:
  void add(std::shared_ptr<Callback> cb) { callbacks_.push_back(std::move(cb)); }
  void on_train_start(TrainState& s) { for (auto& c : callbacks_) c->on_train_start(s); }
  void on_train_end(TrainState& s) { for (auto& c : callbacks_) c->on_train_end(s); }
  void on_step_start(TrainState& s) { for (auto& c : callbacks_) c->on_step_start(s); }
  void on_step_end(TrainState& s) { for (auto& c : callbacks_) c->on_step_end(s); }
  void on_epoch_start(TrainState& s) { for (auto& c : callbacks_) c->on_epoch_start(s); }
  void on_epoch_end(TrainState& s) { for (auto& c : callbacks_) c->on_epoch_end(s); }
  void on_after_loss(TrainState& s) { for (auto& c : callbacks_) c->on_after_loss(s); }
  void on_after_backward(TrainState& s) { for (auto& c : callbacks_) c->on_after_backward(s); }
  void on_after_optimizer_step(TrainState& s) { for (auto& c : callbacks_) c->on_after_optimizer_step(s); }
  void on_checkpoint_save(TrainState& s, const std::string& p) { for (auto& c : callbacks_) c->on_checkpoint_save(s, p); }
  void on_checkpoint_load(TrainState& s, const std::string& p) { for (auto& c : callbacks_) c->on_checkpoint_load(s, p); }
  void on_eval_end(TrainState& s, const std::unordered_map<std::string, float>& m) { for (auto& c : callbacks_) c->on_eval_end(s, m); }
  size_t size() const { return callbacks_.size(); }
 private:
  std::vector<std::shared_ptr<Callback>> callbacks_;
};

}  // namespace olmo_cpp
