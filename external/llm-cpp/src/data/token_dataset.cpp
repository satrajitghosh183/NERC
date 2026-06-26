/**
 * src/data/token_dataset.cpp
 *
 * ─── What a "token dataset" is ──────────────────────────────────────
 *
 * A language model's training data is just a giant 1-D array of
 * integer token ids. For TinyStories with a GPT-2 BPE tokenizer
 * that's roughly 600 million ints — about 1.2 GB in uint16. The
 * model wants to see batches of sequences of length S, but those
 * sequences are just contiguous chunks of the giant array.
 *
 * Storing the array on disk as a NumPy `.npy` file (produced by
 * `prepare_data`) and **memory-mapping** it gives us:
 *
 *   1. zero-copy load — the kernel maps the file into virtual memory
 *      so we don't read 1.2 GB up-front;
 *   2. lazy, page-fault-driven loading;
 *   3. shared between processes if multiple ranks point at the same
 *      file.
 *
 * This file (`TokenDataset`) is exactly that thin wrapper. It owns
 * the .npy load (via the vendored cnpy library), exposes batch
 * accessors that the DataLoader iterates over, and optionally pins
 * the entire array on the GPU when train_cfg.gpu_resident_data=1
 * (saves the host->device copy on every batch).
 *
 * Supported dtypes for the .npy file: uint16 / uint32 / int32 / int64.
 * Internally we always upcast to int64 because that's what
 * torch::nn::Embedding's index path expects.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/token_dataset.hpp : the class declaration.
 *   - olmo_cpp/seed.hpp               : RNG for shuffle order.
 *   - third_party/cnpy/cnpy.h         : .npy reader.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: constructed once at the top of train(), then
 *     stepped each microbatch inside the "data_loading" ProfileScope.
 *
 * --- Role in training pipeline ---
 *   The source of every input batch. A misconfigured data_path here
 *   is the most common reason a fresh setup fails to train.
 */
#include "olmo_cpp/data/token_dataset.hpp"
#include "olmo_cpp/seed.hpp"
#include <cnpy.h>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace olmo_cpp {

TokenDataset::TokenDataset(const std::string& path, int64_t seq_len, bool shuffle)
    : seq_len_(seq_len), shuffle_(shuffle), chunk_cursor_(0) {
  cnpy::NpyArray arr = cnpy::npy_load(path);
  size_t num_vals = arr.num_vals;

  // Support uint16, uint32, int32, int64
  if (arr.word_size == 2) {
    const uint16_t* data = arr.data<uint16_t>();
    tokens_.reserve(num_vals);
    for (size_t i = 0; i < num_vals; ++i) {
      tokens_.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (arr.word_size == 4) {
    const uint32_t* data = arr.data<uint32_t>();
    tokens_.reserve(num_vals);
    for (size_t i = 0; i < num_vals; ++i) {
      tokens_.push_back(static_cast<int64_t>(data[i]));
    }
  } else if (arr.word_size == 8) {
    const int64_t* data = arr.data<int64_t>();
    tokens_.reserve(num_vals);
    for (size_t i = 0; i < num_vals; ++i) {
      tokens_.push_back(data[i]);
    }
  } else {
    throw std::runtime_error("TokenDataset: unsupported .npy dtype (word_size=" +
                             std::to_string(arr.word_size) + ")");
  }

  // Each chunk needs seq_len + 1 tokens (seq_len inputs + 1 for the last label)
  num_chunks_ = (static_cast<int64_t>(tokens_.size()) - 1) / seq_len;
  if (num_chunks_ == 0) {
    throw std::runtime_error("TokenDataset: not enough tokens for seq_len=" +
                             std::to_string(seq_len));
  }

  chunk_indices_.resize(static_cast<size_t>(num_chunks_));
  for (int64_t i = 0; i < num_chunks_; ++i) {
    chunk_indices_[i] = i;
  }

  // Pre-create the full token tensor on CPU once
  tokens_tensor_ = torch::from_blob(
      tokens_.data(),
      {static_cast<int64_t>(tokens_.size())},
      torch::TensorOptions().dtype(torch::kInt64)).clone();
}

void TokenDataset::drain_stream_prefetch() {
  if (stream_prefetch_inflight_ && stream_prefetch_future_.valid()) {
    stream_prefetch_future_.wait();
  }
  stream_prefetch_inflight_ = false;
}

void TokenDataset::ensure_stream_buf_capacity(int64_t batch_size) {
  if (stream_cap_b_ >= batch_size) return;
  drain_stream_prefetch();
  auto opts = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
  for (int i = 0; i < 2; ++i) {
    stream_pin_in_[i] = torch::empty({batch_size, seq_len_}, opts);
    stream_pin_la_[i] = torch::empty({batch_size, seq_len_}, opts);
  }
  stream_cap_b_ = batch_size;
}

void TokenDataset::set_shard(int rank, int world) {
  dp_world_ = (world > 1) ? world : 1;
  dp_rank_  = (world > 1) ? rank  : 0;
  chunk_cursor_ = static_cast<size_t>(dp_rank_);  // rank's start offset into the stride
}

void TokenDataset::reset_epoch() {
  if (shuffle_) {
    if (dp_world_ > 1) {
      // Sharded: shuffle DETERMINISTICALLY per epoch so every rank produces the
      // identical order (random_device / global RNG could diverge across the
      // separate rank processes). Each rank then strides into a disjoint slice.
      std::mt19937_64 g(0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(epoch_));
      std::shuffle(chunk_indices_.begin(), chunk_indices_.end(), g);
      ++epoch_;
    } else {
      // Single-GPU: global seeded RNG (mirrors OLMo-core's seed approach).
      // Falls back to random_device if seed_all() hasn't been called yet.
      try {
        auto& state = global_seed_state();
        std::shuffle(chunk_indices_.begin(), chunk_indices_.end(), state.rng);
      } catch (const std::runtime_error&) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(chunk_indices_.begin(), chunk_indices_.end(), g);
      }
    }
  }
  chunk_cursor_ = static_cast<size_t>(dp_rank_);  // 0 when not sharded
  if (gpu_resident_) {
    // Re-upload shuffled indices to GPU (full corpus path)
    auto idx_tensor = torch::from_blob(
        chunk_indices_.data(),
        {static_cast<int64_t>(chunk_indices_.size())},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    gpu_chunk_indices_ = idx_tensor.to(resident_device_);
    gpu_cursor_ = 0;
  }
}

std::tuple<torch::Tensor, torch::Tensor> TokenDataset::prepare_batch_cpu(int64_t batch_size) {
  // Gather chunk offsets (thread-safe cursor advance)
  std::vector<int64_t> offsets(static_cast<size_t>(batch_size));
  {
    std::lock_guard<std::mutex> lock(cursor_mutex_);
    for (int64_t b = 0; b < batch_size; ++b) {
      if (chunk_cursor_ >= static_cast<size_t>(num_chunks_)) {
        reset_epoch();
      }
      offsets[static_cast<size_t>(b)] = chunk_indices_[chunk_cursor_] * seq_len_;
      // Stride by dp_world_ (==1 single-GPU; skips other ranks' chunks when sharded).
      chunk_cursor_ += static_cast<size_t>(dp_world_);
    }
  }

  // Build index tensor for gather. The [0, seq_len) range is shape-constant,
  // so cache it across calls — avoids one CPU allocation per batch.
  auto idx_opts = torch::TensorOptions().dtype(torch::kInt64);
  if (!cpu_range_.defined()) {
    cpu_range_ = torch::arange(seq_len_, idx_opts).unsqueeze(0);  // [1, seq_len]
  }
  auto offset_tensor = torch::from_blob(
      offsets.data(), {batch_size, 1}, idx_opts).clone();

  auto input_indices = offset_tensor + cpu_range_;
  auto label_indices = input_indices + 1;

  auto input = tokens_tensor_.index_select(0, input_indices.reshape(-1)).reshape({batch_size, seq_len_});
  auto labels = tokens_tensor_.index_select(0, label_indices.reshape(-1)).reshape({batch_size, seq_len_});

  // SFT loss masking: ignore (-100) every label position whose mask bit is 0, so
  // the CE loss counts only assistant tokens. Indexed at the LABEL positions to
  // match how labels are gathered.
  if (mask_tensor_.defined()) {
    auto lab_mask = mask_tensor_.index_select(0, label_indices.reshape(-1)).reshape({batch_size, seq_len_});
    labels = labels.masked_fill(lab_mask == 0, -100);
  }

  return {input, labels};
}

void TokenDataset::set_loss_mask(const std::string& mask_npy_path) {
  cnpy::NpyArray arr = cnpy::npy_load(mask_npy_path);
  size_t n = arr.num_vals;
  if (n != tokens_.size()) {
    throw std::runtime_error("TokenDataset::set_loss_mask: mask length " +
        std::to_string(n) + " != token length " + std::to_string(tokens_.size()));
  }
  std::vector<int64_t> mv;
  mv.reserve(n);
  if (arr.word_size == 1) { const uint8_t* d = arr.data<uint8_t>();
    for (size_t i = 0; i < n; ++i) mv.push_back(static_cast<int64_t>(d[i])); }
  else if (arr.word_size == 2) { const uint16_t* d = arr.data<uint16_t>();
    for (size_t i = 0; i < n; ++i) mv.push_back(static_cast<int64_t>(d[i])); }
  else if (arr.word_size == 4) { const uint32_t* d = arr.data<uint32_t>();
    for (size_t i = 0; i < n; ++i) mv.push_back(static_cast<int64_t>(d[i])); }
  else if (arr.word_size == 8) { const int64_t* d = arr.data<int64_t>();
    for (size_t i = 0; i < n; ++i) mv.push_back(d[i]); }
  else throw std::runtime_error("TokenDataset::set_loss_mask: unsupported mask dtype");
  mask_tensor_ = torch::from_blob(mv.data(), {static_cast<int64_t>(n)},
                                  torch::TensorOptions().dtype(torch::kInt64)).clone();
}

std::tuple<torch::Tensor, torch::Tensor> TokenDataset::get_batch(
    int64_t batch_size,
    torch::Device device) {
  if (gpu_resident_ && device == resident_device_) {
    return get_batch_gpu(batch_size);
  }

  // Pinned double-buffer: overlap CPU gather with previous GPU step
  if (streaming_mode_ && device.is_cuda()) {
    ensure_stream_buf_capacity(batch_size);
    if (stream_prefetch_inflight_) {
      stream_prefetch_future_.wait();
      stream_prefetch_inflight_ = false;
      auto in = stream_pin_in_[stream_last_filled_slot_].to(device, /*non_blocking=*/true);
      auto lab = stream_pin_la_[stream_last_filled_slot_].to(device, /*non_blocking=*/true);
      return {in, lab};
    }
    auto [a, b] = prepare_batch_cpu(batch_size);
    return {a.to(device, /*non_blocking=*/true), b.to(device, /*non_blocking=*/true)};
  }

  if (has_prefetch_ && prefetch_future_.valid()) {
    auto [input, labels] = prefetch_future_.get();
    has_prefetch_ = false;
    return {input.to(device, /*non_blocking=*/true), labels.to(device, /*non_blocking=*/true)};
  }

  auto [input, labels] = prepare_batch_cpu(batch_size);
  return {input.to(device, /*non_blocking=*/true), labels.to(device, /*non_blocking=*/true)};
}

std::tuple<torch::Tensor, torch::Tensor> TokenDataset::get_batch_gpu(int64_t batch_size) {
  // Reshuffle on GPU if we've exhausted all chunks
  if (gpu_cursor_ + batch_size > num_chunks_) {
    if (shuffle_) {
      auto perm = torch::randperm(num_chunks_,
          torch::TensorOptions().dtype(torch::kInt64).device(resident_device_));
      gpu_chunk_indices_ = gpu_chunk_indices_.index_select(0, perm);
    }
    gpu_cursor_ = 0;
  }

  // Grab batch_size chunk offsets (all on GPU)
  auto offsets = gpu_chunk_indices_.narrow(0, gpu_cursor_, batch_size) * seq_len_;
  gpu_cursor_ += batch_size;

  // Build gather indices entirely on GPU. The [0, seq_len) range is
  // shape-constant, so cache it — avoids one GPU allocation + kernel
  // launch on every step of the hot data path.
  if (!gpu_range_.defined()) {
    gpu_range_ = torch::arange(seq_len_,
        torch::TensorOptions().dtype(torch::kInt64).device(resident_device_))
        .unsqueeze(0);  // [1, seq_len]
  }
  auto input_indices = offsets.unsqueeze(1) + gpu_range_;   // [B, seq_len]
  auto label_indices = input_indices + 1;

  auto input = gpu_tokens_tensor_.index_select(0, input_indices.reshape(-1))
                                  .reshape({batch_size, seq_len_});
  auto labels = gpu_tokens_tensor_.index_select(0, label_indices.reshape(-1))
                                   .reshape({batch_size, seq_len_});

  return {input, labels};
}

void TokenDataset::to_device(torch::Device device, int64_t max_gpu_tokens) {
  if (!device.is_cuda()) {
    resident_device_ = device;
    return;
  }

  streaming_mode_ = false;
  gpu_resident_ = false;

#ifdef USE_CUDA
  const int64_t num_tokens = tokens_tensor_.size(0);
  const size_t bytes_tokens = static_cast<size_t>(num_tokens) * sizeof(int64_t);
  const size_t bytes_indices = static_cast<size_t>(num_chunks_) * sizeof(int64_t);
  const size_t bytes_needed = bytes_tokens + bytes_indices + (4 << 20);  // +4 MiB slack

  size_t vram_free = 0, vram_total = 0;
  cudaMemGetInfo(&vram_free, &vram_total);
  const size_t budget_default = vram_free / 4;
  size_t budget = budget_default;
  if (max_gpu_tokens > 0) {
    const size_t cap_bytes = static_cast<size_t>(max_gpu_tokens) * sizeof(int64_t) + bytes_indices + (4 << 20);
    budget = std::min(budget_default, cap_bytes);
  }

  const bool force_stream = (max_gpu_tokens == -1);
  const bool over_user_cap = (max_gpu_tokens > 0 && num_tokens > max_gpu_tokens);
  const bool fits_budget = (bytes_needed <= budget * 9 / 10);
  const bool try_full_gpu = !force_stream && !over_user_cap && fits_budget;

  if (!try_full_gpu) {
    try {
      tokens_tensor_ = tokens_tensor_.contiguous().pin_memory();
    } catch (const c10::Error&) {
      // stay unpinned; H2D still works
    }
    streaming_mode_ = true;
    resident_device_ = device;
    std::cerr << "TokenDataset: streaming mode (pinned host + async prefetch), "
              << num_tokens << " tokens; full GPU residency needs ~"
              << (bytes_needed / (1024 * 1024)) << " MiB data + indices\n";
    return;
  }

  try {
    auto pinned = tokens_tensor_.pin_memory();
    gpu_tokens_tensor_ = pinned.to(device);

    auto idx_tensor = torch::from_blob(
        chunk_indices_.data(),
        {static_cast<int64_t>(chunk_indices_.size())},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    gpu_chunk_indices_ = idx_tensor.to(device);

    gpu_resident_ = true;
    resident_device_ = device;
    gpu_cursor_ = 0;

    std::cerr << "TokenDataset: GPU-resident with " << gpu_tokens_tensor_.size(0)
              << " tokens on " << device << "\n";
  } catch (const c10::Error&) {
    std::cerr << "TokenDataset: GPU residency allocation failed; using streaming mode\n";
    try {
      tokens_tensor_ = tokens_tensor_.contiguous().pin_memory();
    } catch (const c10::Error&) {
    }
    streaming_mode_ = true;
    resident_device_ = device;
  }
#else
  (void)max_gpu_tokens;
  resident_device_ = device;
#endif
}

void TokenDataset::prefetch_next(int64_t batch_size, torch::Device device) {
  if (gpu_resident_ && device == resident_device_) return;

  if (streaming_mode_ && device.is_cuda()) {
    ensure_stream_buf_capacity(batch_size);
    stream_last_filled_slot_ = stream_write_slot_;
    stream_write_slot_ = 1 - stream_write_slot_;
    const int slot = stream_last_filled_slot_;
    stream_prefetch_future_ = std::async(std::launch::async, [this, batch_size, slot]() {
      auto [a, b] = prepare_batch_cpu(batch_size);
      stream_pin_in_[slot].copy_(a, /*non_blocking=*/false);
      stream_pin_la_[slot].copy_(b, /*non_blocking=*/false);
    });
    stream_prefetch_inflight_ = true;
    return;
  }

  prefetch_device_ = device;
  prefetch_future_ = std::async(std::launch::async, [this, batch_size]() {
    return prepare_batch_cpu(batch_size);
  });
  has_prefetch_ = true;
}

TokenDataset::~TokenDataset() {
  drain_stream_prefetch();
  if (has_prefetch_ && prefetch_future_.valid()) {
    prefetch_future_.wait();
    has_prefetch_ = false;
  }
}

}  // namespace olmo_cpp
