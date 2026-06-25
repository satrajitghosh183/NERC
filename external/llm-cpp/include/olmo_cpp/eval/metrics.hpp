#pragma once

/**
 * include/olmo_cpp/eval/metrics.hpp
 *
 * Pure scalar metrics used by the evaluator framework. Each function takes
 * already-computed losses or tensors and returns a `double`; there is no
 * model state in this header. Keeping the metric definitions in one place
 * lets `MultiTaskEvaluator` look up metrics by string name.
 *
 * --- Includes from this project ---
 *   - none beyond LibTorch: this header is intentionally a leaf so it can be
 *     included by any evaluator without pulling in trainer-side types.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/eval/metrics.cpp: implementations.
 *   - src/eval/evaluator.cpp: calls perplexity() and bits_per_byte() inside
 *     MultiTaskEvaluator::evaluate.
 *
 * --- Role in training pipeline ---
 *   Mathematical leaf: turns model outputs / loss values into the numbers
 *   that get logged to console, W&B, or TensorBoard during training.
 */

#include <cstdint>
#include <string>
#include <unordered_map>

// Forward-decl Tensor instead of pulling in <torch/torch.h>. Tensor is only
// referenced by const-ref in the public API below; the .cpp includes the
// full header. Skipping torch/torch.h saves ~5 MB / 200K LOC of transitive
// includes per consumer of metrics.hpp — meaningful when this header is
// pulled into eval orchestrators, training callbacks, and offline tools.
namespace at { class Tensor; }
namespace torch { using at::Tensor; }

namespace olmo_cpp {

/// Conventional metric container — string key (e.g. "val/perplexity") to
/// scalar value. Used as the return type of all `Evaluator::evaluate` calls.
using MetricMap = std::unordered_map<std::string, double>;

/// Convert mean cross-entropy (in nats) to perplexity = exp(loss).
double perplexity(double ce_loss);

/// Convert mean cross-entropy (in nats) to bits/byte. NB: the current
/// implementation divides by ln(2) only — caller must supply chars/token
/// rescaling externally if needed (see comment in metrics.cpp).
double bits_per_byte(double ce_loss);

/// Token-level next-token accuracy. `logits` is [B,S,V], `labels` is [B,S].
/// `ignore_index` (default -100, PyTorch convention) is excluded from both
/// numerator and denominator.
double accuracy(const torch::Tensor& logits, const torch::Tensor& labels,
                int64_t ignore_index = -100);

/// Top-k variant: counts a position correct if the gold token appears
/// anywhere in the top-k logits at that position.
double top_k_accuracy(const torch::Tensor& logits, const torch::Tensor& labels,
                      int64_t k, int64_t ignore_index = -100);

/// Macro-averaged F1 over `num_classes`. Classes that never appear (TP=FP=FN=0)
/// are dropped from the average rather than contributing zero.
double f1_score(const torch::Tensor& predictions, const torch::Tensor& labels,
                int64_t num_classes);

}  // namespace olmo_cpp
