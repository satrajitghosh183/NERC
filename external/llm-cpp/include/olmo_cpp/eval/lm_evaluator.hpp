#pragma once

/**
 * include/olmo_cpp/eval/lm_evaluator.hpp
 *
 * Two domain-specific evaluators: `LMEvaluator` for autoregressive language
 * modeling (perplexity / accuracy on a held-out token stream) and
 * `DownstreamEvaluator` for multiple-choice tasks loaded from JSONL.
 * Both implement the polymorphic `Evaluator` interface so the trainer can
 * fan out and collect metrics uniformly.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/eval/evaluator.hpp: base class + MetricMap typedef.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/eval/lm_evaluator.cpp: implementations.
 *   (Direct callers in src/train not located via quick grep — concrete
 *   eval-loop integration is currently in train.cpp using the Transformer
 *   type directly, per the comment in lm_evaluator.cpp.)
 *
 * --- Role in training pipeline ---
 *   Provide the structured "eval task" objects that the trainer instantiates
 *   from `[evaluation]` config. Concrete data loading (numpy token files,
 *   JSONL example files) happens here so the trainer is data-agnostic.
 */

#include "olmo_cpp/eval/evaluator.hpp"
#include <string>
#include <vector>

namespace olmo_cpp {

/// Plain LM evaluator: read a tokenized dataset from `data_path`, slice it
/// into `(seq_len, batch_size)` chunks, run forward, compute cross-entropy
/// over the next-token labels, then emit perplexity and accuracy.
class LMEvaluator : public Evaluator {
 public:
  /// `data_path` should point at a tokenized .npy file produced by
  /// `prepare_data`; `max_batches=-1` means evaluate the entire dataset.
  LMEvaluator(const std::string& data_path, int64_t seq_len,
              int64_t batch_size = 8, int64_t max_batches = -1);

  /// Forward through the model in `NoGradGuard` mode; aggregate loss and
  /// accuracy across batches; return them keyed by metric name.
  MetricMap evaluate(torch::nn::Module& model, torch::Device device) override;
  std::string name() const override { return "lm_eval"; }

 private:
  std::string data_path_;                          ///< tokenized .npy path
  int64_t seq_len_, batch_size_, max_batches_;     ///< slicing parameters
};

/// Multiple-choice eval (HellaSwag/PIQA-style). Each line of the JSONL is one
/// `Example`; the evaluator scores each candidate continuation and reports
/// accuracy of the model's argmax over the candidates.
class DownstreamEvaluator : public Evaluator {
 public:
  /// `task_name` becomes the metric prefix; `data_path` is a JSONL file with
  /// `{"context", "choices": [...], "label": int}` records.
  DownstreamEvaluator(const std::string& task_name, const std::string& data_path,
                      int64_t max_examples = -1);

  /// Score each example (currently a stub returning 0.0 until tokenizer
  /// integration lands) and report `num_examples` and `accuracy`.
  MetricMap evaluate(torch::nn::Module& model, torch::Device device) override;
  std::string name() const override { return name_; }

 private:
  std::string name_;                ///< user-supplied task identifier
  /// One multiple-choice item; `label` indexes into `choices`.
  struct Example {
    std::string context;
    std::vector<std::string> choices;
    int64_t label;
  };
  std::vector<Example> examples_;   ///< parsed from the JSONL on construction
};

}  // namespace olmo_cpp
