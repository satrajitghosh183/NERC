#pragma once

/**
 * include/olmo_cpp/data/collator.hpp
 *
 * Variable-length-sequence collator: takes a list of (input_ids, labels)
 * pairs of unequal length and packs them into rectangular [B, T] tensors,
 * filling unused positions with pad_token_id (for inputs/mask) and -100
 * (the standard PyTorch CrossEntropy ignore_index) for labels.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: returns torch::Tensor batches usable by the model.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep. The collator is a utility used
 *   by ad-hoc evaluation / fine-tuning paths; the main pretraining loop uses
 *   TokenDataset (already-rectangular chunks) so it does not invoke this.
 *
 * --- Role in training pipeline ---
 *   Bridges variable-length tokenizer output to fixed-shape model input. Used
 *   when batches contain real sequence-end boundaries (chat / SFT) rather
 *   than the concat-and-chunk style of pretraining.
 */

#include <torch/torch.h>
#include <vector>

namespace olmo_cpp {

/// Where padding lives within each row of the output tensor. Right padding
/// is the GPT-style default; left padding matches HF's "padding_side='left'"
/// used for batched generation so all sequences end at the same column.
enum class PaddingDirection { Left, Right };

/// Data collator: pads and batches sequences
class DataCollator {
 public:
  /// pad_token_id: id used to fill empty slots in input_ids; matches the
  ///   model's pad embedding (often 0 for OLMo / SimpleTokenizer).
  /// padding_dir: see enum above; Right is correct for causal LM training.
  /// max_length: optional hard cap; -1 means "use the longest sequence in
  ///   the batch" (dynamic padding, saves memory for short batches).
  DataCollator(int64_t pad_token_id = 0,
               PaddingDirection padding_dir = PaddingDirection::Right,
               int64_t max_length = -1);

  /// Collate variable-length sequences into a padded batch
  /// Returns {input_ids [B, max_len], attention_mask [B, max_len], labels [B, max_len]}
  /// All three tensors are kInt64. attention_mask is 1 where there is a real
  /// token and 0 where padding lives (model multiplies attention scores by it).
  /// labels uses -100 for padded positions so CrossEntropyLoss skips them.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> collate(
      const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>& batch) const;

 private:
  int64_t pad_token_id_;          // value used to fill input_ids padding
  PaddingDirection padding_dir_;  // Left vs Right
  int64_t max_length_;            // -1 = dynamic, else hard cap on T
};

}  // namespace olmo_cpp
