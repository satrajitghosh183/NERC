#pragma once
/**
 * include/olmo_cpp/train/activation_checkpoint.hpp
 *
 * Activation checkpointing — drops intermediate activations during
 * forward and re-runs the segment under enable_grad() during
 * backward. Trades one extra forward for big memory savings. Crucial
 * for fitting big models on limited VRAM (e.g. 125M+ on a 12 GB
 * 3060).
 *
 * Implemented via a torch::autograd::Function. See
 * src/train/activation_checkpoint.cpp for the longer description.
 *
 * --- Includes from this project ---
 *   (none — torch only.)
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train/activation_checkpoint.cpp : implementation.
 *   - src/model/transformer.cpp / fused_transformer.cpp : forward()
 *     wraps each block call when the .conf enables checkpointing.
 *
 * --- Role in training pipeline ---
 *   Memory-saving optimisation, opt-in. Off in the 30M quickstart
 *   conf (30M fits trivially); ON in the 125M conf.
 */

#include <torch/torch.h>
#include <functional>

namespace olmo_cpp {

/// Activation checkpointing: trades compute for memory by recomputing
/// forward activations during backward instead of storing them.
class ActivationCheckpoint {
 public:
  /// Checkpoint a function: saves only inputs during forward,
  /// recomputes during backward.
  static torch::Tensor checkpoint(
      std::function<torch::Tensor(torch::Tensor)> fn,
      torch::Tensor input);

  /// Whether to checkpoint a given layer based on interval
  static bool should_checkpoint(int64_t layer_idx, int64_t interval);
};

}  // namespace olmo_cpp
