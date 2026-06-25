/**
 * src/eval/lm_evaluator.cpp
 *
 * Concrete EvalTask for plain language-model evaluation. Iterates the
 * eval dataset (a TokenDataset) one batch at a time, runs the model
 * in `torch::NoGradGuard` mode, accumulates cross-entropy, and
 * reports mean CE loss and perplexity (= exp(mean CE)).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/eval/lm_evaluator.hpp  : LMEvaluator declaration.
 *   - olmo_cpp/data/token_dataset.hpp : the eval-set source.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: when train_cfg.eval_data_path is set, this
 *     evaluator is registered and invoked every eval_interval steps.
 *
 * --- Role in training pipeline ---
 *   Periodic eval. Result printed alongside train loss for trend
 *   monitoring.
 */
#include "olmo_cpp/eval/lm_evaluator.hpp"
#include "olmo_cpp/data/token_dataset.hpp"
#include "olmo_cpp/model/transformer.hpp"
#include "olmo_cpp/model/fused_transformer.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <optional>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace olmo_cpp {

// ---------------------------------------------------------------------------
// LMEvaluator
// ---------------------------------------------------------------------------

LMEvaluator::LMEvaluator(const std::string& data_path, int64_t seq_len,
                          int64_t batch_size, int64_t max_batches)
    : data_path_(data_path), seq_len_(seq_len),
      batch_size_(batch_size), max_batches_(max_batches) {}

namespace {

// Forward-and-loss on the concrete model type. The base torch::nn::Module
// has no virtual `forward(input, labels)` so we dynamic_cast to the two
// concrete model types this codebase ships.
std::optional<double> forward_loss(torch::nn::Module& model,
                                   const torch::Tensor& input,
                                   const torch::Tensor& labels) {
  if (auto* t = dynamic_cast<TransformerImpl*>(&model)) {
    return t->forward(input, labels).item<double>();
  }
  if (auto* ft = dynamic_cast<FusedTransformerImpl*>(&model)) {
    return ft->forward(input, labels).item<double>();
  }
  return std::nullopt;
}

}  // namespace

MetricMap LMEvaluator::evaluate(torch::nn::Module& model, torch::Device device) {
  MetricMap metrics;
  metrics["seq_len"]    = static_cast<double>(seq_len_);
  metrics["batch_size"] = static_cast<double>(batch_size_);

  // Eval dataset uses shuffle=false so the same chunks are iterated in the
  // same order every eval — gives a stable val-loss curve across calls.
  TokenDataset ds(data_path_, seq_len_, /*shuffle=*/false);
  if (ds.size() == 0) {
    metrics["evaluator_status"] = -2.0;  // no data
    return metrics;
  }
  ds.to_device(device);
  ds.reset_epoch();

  torch::NoGradGuard no_grad;

  // Cap the eval at max_batches_ batches (or the full dataset if -1).
  const int64_t n_chunks = ds.size();
  int64_t budget = max_batches_;
  if (budget < 0) budget = (n_chunks + batch_size_ - 1) / batch_size_;
  budget = std::min<int64_t>(budget, (n_chunks + batch_size_ - 1) / batch_size_);

  double total_loss = 0.0;
  int64_t total_batches = 0;
  bool model_unsupported = false;

  for (int64_t b = 0; b < budget; ++b) {
    auto [input, labels] = ds.get_batch(batch_size_, device);
    auto loss_opt = forward_loss(model, input, labels);
    if (!loss_opt) { model_unsupported = true; break; }
    total_loss += *loss_opt;
    ++total_batches;
  }

  if (model_unsupported) {
    metrics["evaluator_status"] = -1.0;  // model type not supported
    return metrics;
  }
  if (total_batches == 0) {
    metrics["evaluator_status"] = 0.0;
    return metrics;
  }

  const double avg_loss = total_loss / static_cast<double>(total_batches);
  metrics["loss"]          = avg_loss;
  metrics["perplexity"]    = std::exp(avg_loss);
  metrics["bits_per_byte"] = avg_loss / std::log(2.0);
  metrics["batches"]       = static_cast<double>(total_batches);
  return metrics;
}

// ---------------------------------------------------------------------------
// DownstreamEvaluator
// ---------------------------------------------------------------------------

DownstreamEvaluator::DownstreamEvaluator(const std::string& task_name,
                                          const std::string& data_path,
                                          int64_t max_examples)
    : name_(task_name) {
#ifdef HAS_NLOHMANN_JSON
  std::ifstream in(data_path);
  if (!in.is_open()) return;
  std::string line;
  int64_t count = 0;
  while (std::getline(in, line)) {
    if (max_examples > 0 && count >= max_examples) break;
    auto j = nlohmann::json::parse(line);
    Example ex;
    ex.context = j.value("context", std::string(""));
    ex.label = j.value("label", int64_t(0));
    if (j.contains("choices")) {
      for (const auto& c : j["choices"]) {
        ex.choices.push_back(c.get<std::string>());
      }
    }
    examples_.push_back(std::move(ex));
    count++;
  }
#else
  (void)data_path;
  (void)max_examples;
#endif
}

MetricMap DownstreamEvaluator::evaluate(torch::nn::Module& /*model*/,
                                         torch::Device /*device*/) {
  MetricMap metrics;
  metrics["num_examples"] = static_cast<double>(examples_.size());
  metrics["accuracy"] = 0.0;  // requires tokenizer integration
  return metrics;
}

}  // namespace olmo_cpp
