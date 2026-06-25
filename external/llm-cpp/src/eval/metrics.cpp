/**
 * src/eval/metrics.cpp
 *
 * Standard language-model evaluation metrics.
 *
 *   - perplexity(ce_loss) = exp(ce_loss). Cross-entropy is the
 *     average negative log-likelihood per token; perplexity is its
 *     exponential and equals "the effective number of equally-likely
 *     next-token choices the model is uncertain among". Lower is
 *     better. A bigram baseline on English text gives perplexity
 *     ~100; a well-trained 7B LM gets to ~5 on natural-language data.
 *
 *   - other metrics: accuracy, top-k accuracy, etc., depending on
 *     the task. Computed from raw logits + labels.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/eval/metrics.hpp : metric function declarations.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/eval/lm_evaluator.cpp / evaluator.cpp: report perplexity
 *     and accuracy after each eval pass.
 *   - src/train.cpp: mid-training eval calls these to log progress.
 *
 * --- Role in training pipeline ---
 *   Pure functions. Run during the periodic eval pass when
 *   eval_data_path is set in the .conf.
 */
#include "olmo_cpp/eval/metrics.hpp"

// Bring in the full LibTorch headers for the .cpp implementation. The
// header was trimmed to a Tensor forward-decl to keep its include cost
// down; that means every translation unit that *implements* metrics
// needs the full include here.
#include <torch/torch.h>
#include <cmath>

namespace olmo_cpp {

double perplexity(double ce_loss) {
  return std::exp(ce_loss);
}

double bits_per_byte(double ce_loss) {
  // bits_per_byte = ce_loss / ln(2) / chars_per_token
  // Using standard ~3.5 chars/token for English text
  return ce_loss / std::log(2.0);
}

double accuracy(const torch::Tensor& logits, const torch::Tensor& labels,
                int64_t ignore_index) {
  // logits: [B, S, V], labels: [B, S]
  auto preds = logits.argmax(-1);  // [B, S]
  auto mask = labels != ignore_index;
  auto correct = (preds == labels) & mask;
  auto total = mask.sum().item<double>();
  if (total == 0.0) return 0.0;
  return correct.sum().item<double>() / total;
}

double top_k_accuracy(const torch::Tensor& logits, const torch::Tensor& labels,
                      int64_t k, int64_t ignore_index) {
  // logits: [B, S, V], labels: [B, S]
  auto topk = std::get<1>(logits.topk(k, -1));  // [B, S, k]
  auto labels_expanded = labels.unsqueeze(-1).expand_as(topk);
  auto mask = labels != ignore_index;
  auto correct = (topk == labels_expanded).any(-1) & mask;  // [B, S]
  auto total = mask.sum().item<double>();
  if (total == 0.0) return 0.0;
  return correct.sum().item<double>() / total;
}

double f1_score(const torch::Tensor& predictions, const torch::Tensor& labels,
                int64_t num_classes) {
  double total_f1 = 0.0;
  int valid_classes = 0;

  for (int64_t c = 0; c < num_classes; ++c) {
    auto pred_c = predictions == c;
    auto label_c = labels == c;
    double tp = (pred_c & label_c).sum().item<double>();
    double fp = (pred_c & ~label_c).sum().item<double>();
    double fn = (~pred_c & label_c).sum().item<double>();

    if (tp + fp + fn > 0) {
      double precision = (tp + fp > 0) ? tp / (tp + fp) : 0.0;
      double recall = (tp + fn > 0) ? tp / (tp + fn) : 0.0;
      double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
      total_f1 += f1;
      valid_classes++;
    }
  }

  return (valid_classes > 0) ? total_f1 / valid_classes : 0.0;
}

}  // namespace olmo_cpp
