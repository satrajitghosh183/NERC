#pragma once

/**
 * include/olmo_cpp/eval/evaluator.hpp
 *
 * Generic evaluation framework. Defines a tiny interface (Evaluator) that
 * a training loop can call periodically with `(model, device)` and receive
 * back a string -> double `MetricMap` (perplexity, accuracy, bits-per-byte,
 * etc.). Concrete implementations live in `lm_evaluator.hpp` and elsewhere.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/eval/metrics.hpp: provides MetricMap and the scalar metric
 *     functions used by the multi-task evaluator.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/eval/evaluator.cpp: implements MultiTaskEvaluator.
 *   - include/olmo_cpp/eval/lm_evaluator.hpp: extends Evaluator for LM tasks.
 *   (No direct `make_unique<MultiTaskEvaluator>` callers found in src/train
 *   on quick grep — the framework is wired up via concrete subclasses.)
 *
 * --- Role in training pipeline ---
 *   The trainer can hold one or more `Evaluator` pointers and invoke them at
 *   eval boundaries. Keeping the interface generic over `torch::nn::Module`
 *   lets us reuse it for the standard Transformer and the FusedTransformer.
 */

#include <torch/torch.h>
#include "olmo_cpp/eval/metrics.hpp"
#include <string>
#include <vector>
#include <memory>

namespace olmo_cpp {

/// One evaluation "task": a name (used as metric prefix), a list of
/// (input, label) tensor pairs, and a list of metric names to report.
struct EvalTask {
  std::string name;                                          ///< prefix for metric keys
  std::vector<std::pair<torch::Tensor, torch::Tensor>> data; ///< (input_ids, labels) pairs
  std::vector<std::string> metric_names;                     ///< e.g. {"perplexity","accuracy"}
};

/// Polymorphic base for any evaluator. The trainer only depends on this
/// interface; concrete subclasses (LMEvaluator, DownstreamEvaluator, ...)
/// know how to load their data and interpret model outputs.
class Evaluator {
 public:
  virtual ~Evaluator() = default;
  /// Evaluate `model` on the device, returning a flat metric dictionary.
  virtual MetricMap evaluate(torch::nn::Module& model, torch::Device device) = 0;
  /// Short name used for logging / report headers.
  virtual std::string name() const = 0;
};

/// Concrete evaluator that owns a list of `EvalTask`s and concatenates
/// their metrics into a single MetricMap (with each task's name prefixed).
class MultiTaskEvaluator : public Evaluator {
 public:
  /// Append an eval task; tasks are evaluated in insertion order.
  void add_task(EvalTask task);
  /// Walk every task, every batch, accumulate loss, then emit metrics
  /// requested in `metric_names` for each task.
  MetricMap evaluate(torch::nn::Module& model, torch::Device device) override;
  std::string name() const override { return "multi_task"; }
 private:
  std::vector<EvalTask> tasks_;  ///< owned task list; reused on every call
};

}  // namespace olmo_cpp
