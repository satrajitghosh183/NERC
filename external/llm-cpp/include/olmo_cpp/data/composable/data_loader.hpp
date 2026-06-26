#pragma once

/**
 * include/olmo_cpp/data/composable/data_loader.hpp
 *
 * Terminal stage of the composable data pipeline: takes an InstanceSource
 * (already-shaped per-example token sequences) and packs `batch_size` of
 * them into rectangular [B, S] CPU tensors, then transfers to the target
 * device. This is the alternative path to TokenDataset for users who want
 * the multi-source / shuffle-buffer / mixing primitives in composable/.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/composable/instance_source.hpp: pulls Instance objects.
 *   - <torch/torch.h>: assembles the per-step torch::Tensor batch.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep. Used by experimental data
 *   pipelines that don't fit the single-.npy assumption of TokenDataset.
 *
 * --- Role in training pipeline ---
 *   Adapter from the composable per-example sources to the [B, S] tensor
 *   contract expected by Transformer::forward().
 */

#include "olmo_cpp/data/composable/instance_source.hpp"
#include <torch/torch.h>
#include <memory>

namespace olmo_cpp {

/// Composable data loader: wraps an instance source into batched tensors
class ComposableDataLoader {
 public:
  /// source:        upstream Instance producer (assumed to emit fixed-length
  ///                instances; partial batches are accepted at end-of-stream).
  /// batch_size:    number of instances per batch (B in [B, S]).
  /// device:        target device for the returned tensors (host->device
  ///                copy happens at the end of next_batch()).
  /// pad_token_id:  filler used when an Instance is shorter than seq_len.
  ComposableDataLoader(std::unique_ptr<InstanceSource> source,
                       int64_t batch_size, torch::Device device,
                       int64_t pad_token_id = 0);

  /// Get next batch: {input_ids [B, S], labels [B, S]}
  /// labels uses -100 for positions that should be ignored by CE loss.
  std::tuple<torch::Tensor, torch::Tensor> next_batch();

  /// Whether more batches are available (delegates to the InstanceSource).
  bool has_next() const;

  /// Reset for new epoch (resets the underlying source).
  void reset();

 private:
  std::unique_ptr<InstanceSource> source_;  // upstream Instance producer
  int64_t batch_size_;                      // B
  torch::Device device_;                    // target for final tensors
  int64_t pad_token_id_;                    // padding filler for input_ids
};

}  // namespace olmo_cpp
