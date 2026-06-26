/**
 * src/data/collator.cpp
 *
 * ─── What a "collator" does ─────────────────────────────────────────
 *
 * A dataset usually returns one sample at a time. The model wants a
 * batch — that is, several samples stacked into a single tensor of
 * shape [batch_size, seq_len]. The collator is the function that
 * does this packing. Its job is non-trivial because samples can have
 * different lengths and need to be padded, and you have to decide:
 *
 *   - which token id is the "pad" token,
 *   - whether to pad on the left or the right (causal LMs usually
 *     left-pad for generation, right-pad for training),
 *   - whether to truncate over-long samples or refuse them.
 *
 * DataCollator centralises all that.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/collator.hpp : DataCollator declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/composable/data_loader.cpp: ComposableDataLoader uses
 *     a DataCollator to assemble each batch.
 *
 * --- Role in training pipeline ---
 *   Sits between the per-sample data source and the per-batch
 *   training step.
 */
#include "olmo_cpp/data/collator.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace olmo_cpp {

DataCollator::DataCollator(int64_t pad_token_id, PaddingDirection padding_dir,
                           int64_t max_length)
    : pad_token_id_(pad_token_id),
      padding_dir_(padding_dir),
      max_length_(max_length) {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DataCollator::collate(
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>&
        batch) const {
  if (batch.empty()) {
    throw std::runtime_error("DataCollator::collate: empty batch");
  }

  // Find the maximum length in the batch
  int64_t max_len = 0;
  for (const auto& [input_ids, labels] : batch) {
    max_len = std::max(max_len, static_cast<int64_t>(input_ids.size()));
  }

  // Apply max_length cap if set
  if (max_length_ > 0) {
    max_len = std::min(max_len, max_length_);
  }

  const int64_t batch_size = static_cast<int64_t>(batch.size());

  // Initialize tensors
  auto input_ids_tensor = torch::full(
      {batch_size, max_len}, pad_token_id_,
      torch::TensorOptions().dtype(torch::kLong));
  auto attention_mask_tensor = torch::zeros(
      {batch_size, max_len},
      torch::TensorOptions().dtype(torch::kLong));
  auto labels_tensor = torch::full(
      {batch_size, max_len}, static_cast<int64_t>(-100),
      torch::TensorOptions().dtype(torch::kLong));

  auto input_acc = input_ids_tensor.accessor<int64_t, 2>();
  auto mask_acc = attention_mask_tensor.accessor<int64_t, 2>();
  auto label_acc = labels_tensor.accessor<int64_t, 2>();

  for (int64_t b = 0; b < batch_size; ++b) {
    const auto& [input_ids, labels] = batch[b];
    const int64_t seq_len = std::min(static_cast<int64_t>(input_ids.size()),
                                     max_len);
    const int64_t label_len = std::min(static_cast<int64_t>(labels.size()),
                                       max_len);

    // Bulk memcpy beats element-by-element accessor writes — input_ids
    // and labels are std::vector<int64_t>, exactly the dtype/layout of
    // input_acc[b] / label_acc[b]. ComposableDataLoader already does this;
    // bringing the legacy collator in line.
    if (padding_dir_ == PaddingDirection::Right) {
      std::memcpy(&input_acc[b][0], input_ids.data(), seq_len * sizeof(int64_t));
      std::fill_n(&mask_acc[b][0], seq_len, int64_t{1});
      std::memcpy(&label_acc[b][0], labels.data(), label_len * sizeof(int64_t));
    } else {
      const int64_t offset       = max_len - seq_len;
      const int64_t label_offset = max_len - label_len;
      std::memcpy(&input_acc[b][offset], input_ids.data(), seq_len * sizeof(int64_t));
      std::fill_n(&mask_acc[b][offset], seq_len, int64_t{1});
      std::memcpy(&label_acc[b][label_offset], labels.data(), label_len * sizeof(int64_t));
    }
  }

  return {input_ids_tensor, attention_mask_tensor, labels_tensor};
}

}  // namespace olmo_cpp
