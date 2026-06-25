/**
 * src/eval/evaluator.cpp
 *
 * Generic multi-task evaluation orchestrator. The training loop
 * registers a list of EvalTask objects (each with a name, a dataset,
 * and a metric function) and `MultiTaskEvaluator::run(model)` walks
 * them in order, returning a structured report.
 *
 * Used to keep the training-loop code agnostic of WHICH evaluations
 * the user has configured — adding a new task is just registering
 * one more EvalTask.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/eval/evaluator.hpp : MultiTaskEvaluator + EvalTask.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: evaluator is invoked every `eval_interval` steps.
 *
 * --- Role in training pipeline ---
 *   Periodic monitoring. Inactive between eval points.
 */
#include "olmo_cpp/eval/evaluator.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"
#include <iostream>
#include <optional>

namespace olmo_cpp {

void MultiTaskEvaluator::add_task(EvalTask task) {
  tasks_.push_back(std::move(task));
}

// Run forward + loss on the concrete model type. Returns nullopt when
// `model` is neither Transformer nor FusedTransformer (the only types that
// expose a (input_ids, labels) -> scalar-loss forward). The base
// torch::nn::Module has no virtual `forward` to call into, so without a
// dynamic_cast we cannot compute loss generically.
static std::optional<double> run_forward_loss(torch::nn::Module& model,
                                              const torch::Tensor& input,
                                              const torch::Tensor& labels) {
  if (auto* t = dynamic_cast<TransformerImpl*>(&model)) {
    auto loss = t->forward(input, labels);
    return loss.item<double>();
  }
  if (auto* ft = dynamic_cast<FusedTransformerImpl*>(&model)) {
    auto loss = ft->forward(input, labels);
    return loss.item<double>();
  }
  return std::nullopt;
}

MetricMap MultiTaskEvaluator::evaluate(torch::nn::Module& model, torch::Device device) {
  MetricMap all_metrics;
  torch::NoGradGuard no_grad;

  for (const auto& task : tasks_) {
    double total_loss = 0.0;
    int64_t num_batches = 0;
    bool any_failed = false;

    for (const auto& [input, labels] : task.data) {
      auto inp = input.to(device);
      auto lab = labels.to(device);
      auto loss_opt = run_forward_loss(model, inp, lab);
      if (!loss_opt) { any_failed = true; break; }
      total_loss += *loss_opt;
      num_batches++;
    }

    const std::string prefix = task.name + "/";
    if (any_failed || num_batches == 0) {
      // Honest failure signal — no fake zero loss. Caller can detect this
      // by the absence of usual loss/perplexity keys.
      all_metrics[prefix + "evaluator_status"] =
          any_failed ? -1.0 /*forward unsupported*/ : 0.0 /*no batches*/;
      continue;
    }

    const double avg_loss = total_loss / num_batches;

    for (const auto& metric : task.metric_names) {
      if (metric == "perplexity")        all_metrics[prefix + "perplexity"]    = perplexity(avg_loss);
      else if (metric == "loss")         all_metrics[prefix + "loss"]          = avg_loss;
      else if (metric == "bits_per_byte") all_metrics[prefix + "bits_per_byte"] = bits_per_byte(avg_loss);
      // "accuracy" requires per-token argmax — out of scope for this generic
      // evaluator. Use a model-specific evaluator (e.g. LMEvaluator) if the
      // task needs token-level accuracy.
    }
  }

  return all_metrics;
}

}  // namespace olmo_cpp
