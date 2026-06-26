/**
 * src/data/composable/data_loader.cpp
 *
 * Top layer of the composable data pipeline (see
 * document_source.cpp's docblock for the full architecture).
 *
 * ComposableDataLoader takes an InstanceSource and assembles batches
 * for the training loop:
 *
 *   - pulls `batch_size` instances from the underlying InstanceSource,
 *   - delegates to a DataCollator to pack them into a tensor,
 *   - moves the tensor to device (with optional pinned memory for
 *     faster H2D copies).
 *
 * The interface mirrors what src/train.cpp expects from any data
 * loader, so the composable path is a drop-in replacement for the
 * simpler TokenDataset+sampler combo.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/composable/data_loader.hpp : declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: instantiated when the configuration calls for
 *     the composable pipeline.
 *
 * --- Role in training pipeline ---
 *   Active only in the composable data path. The quickstart's flow
 *   uses the simpler TokenDataset path.
 */
#include "olmo_cpp/data/composable/data_loader.hpp"
#include <cstring>
#include <stdexcept>

namespace olmo_cpp {

ComposableDataLoader::ComposableDataLoader(
    std::unique_ptr<InstanceSource> source, int64_t batch_size,
    torch::Device device, int64_t pad_token_id)
    : source_(std::move(source)),
      batch_size_(batch_size),
      device_(device),
      pad_token_id_(pad_token_id) {
  if (batch_size <= 0) {
    throw std::runtime_error(
        "ComposableDataLoader: batch_size must be positive");
  }
}

bool ComposableDataLoader::has_next() const {
  return source_->has_next();
}

std::tuple<torch::Tensor, torch::Tensor> ComposableDataLoader::next_batch() {
  std::vector<Instance> instances;
  instances.reserve(batch_size_);

  for (int64_t i = 0; i < batch_size_ && source_->has_next(); ++i) {
    instances.push_back(source_->next());
  }

  if (instances.empty()) {
    throw std::runtime_error("ComposableDataLoader: no more batches");
  }

  // Determine sequence length from the first instance
  const int64_t seq_len = static_cast<int64_t>(instances[0].input_ids.size());
  const int64_t actual_batch = static_cast<int64_t>(instances.size());

  // Allocate tensors on CPU first, then move to device
  auto input_ids = torch::full({actual_batch, seq_len}, pad_token_id_,
                               torch::TensorOptions().dtype(torch::kLong));
  auto labels = torch::full({actual_batch, seq_len}, static_cast<int64_t>(-100),
                            torch::TensorOptions().dtype(torch::kLong));

  auto* input_ptr = input_ids.data_ptr<int64_t>();
  auto* label_ptr = labels.data_ptr<int64_t>();

  for (int64_t b = 0; b < actual_batch; ++b) {
    const auto& inst = instances[b];
    const int64_t len = std::min(static_cast<int64_t>(inst.input_ids.size()), seq_len);
    std::memcpy(input_ptr + b * seq_len, inst.input_ids.data(),
                static_cast<size_t>(len) * sizeof(int64_t));
    const int64_t label_len = std::min(static_cast<int64_t>(inst.labels.size()), seq_len);
    std::memcpy(label_ptr + b * seq_len, inst.labels.data(),
                static_cast<size_t>(label_len) * sizeof(int64_t));
  }

  return {input_ids.to(device_), labels.to(device_)};
}

void ComposableDataLoader::reset() {
  source_->reset();
}

}  // namespace olmo_cpp
