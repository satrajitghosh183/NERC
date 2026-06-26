#pragma once

/**
 * include/olmo_cpp/data/token_dataset.hpp
 *
 * The hot-path data source used by the OLMo pretraining loop. Memory-maps a
 * single .npy file containing the full tokenized corpus (one big 1-D
 * integer array), partitions it into fixed-length seq_len chunks, and
 * serves shuffled (input, label) batches where the labels are the inputs
 * shifted by one (next-token prediction).
 *
 * Three execution modes (see to_device()):
 *   - GPU-resident: full token tensor + chunk-index permutation live on
 *     CUDA, gather is a single cudaMemcpy/index_select per step. Used when
 *     the corpus fits in ~25% of free VRAM (or under the user cap).
 *   - Streaming (CUDA): tokens stay pinned on the host; a background
 *     thread prepares batch N+1 into a pinned double-buffer while batch N
 *     trains. Bounded host memory; one async H2D per step.
 *   - CPU: simple synchronous gather, used for unit tests and CPU bench.
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: returned tensors and device handling.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/train.cpp: instantiates a TokenDataset at TrainConfig parse time
 *     and pulls a batch per global step.
 *   - tools/dump_params.cpp / scripts/verify_*: consume validation shards.
 *
 * --- Role in training pipeline ---
 *   Owns the memory-mapped token tensor read by the training loop each
 *   step; constructed once at TrainConfig parse time. Provides the only
 *   non-trivial overlap of data prep with compute on the H100/A100 path.
 */

#include <string>
#include <vector>
#include <torch/torch.h>
#include <optional>
#include <future>
#include <mutex>

namespace olmo_cpp {

/// Token dataset backed by .npy file. Loads tokens, chunks into fixed seq_len.
/// Supports uint16 or uint32 token IDs (common for OLMo).
/// Includes async prefetch: while GPU runs forward/backward on batch N,
/// CPU prepares batch N+1 in a background thread.
///
/// Tensors layout per step:
///   input:  [batch_size, seq_len]  int64
///   labels: [batch_size, seq_len]  int64  (= tokens shifted by +1 position)
class TokenDataset {
 public:
  /// Load token array from .npy file. Expects 1D array of token IDs.
  TokenDataset(const std::string& path, int64_t seq_len, bool shuffle = true);
  ~TokenDataset();

  /// Number of chunks (sequences) in the dataset
  int64_t size() const { return num_chunks_; }

  /// Get a batch of sequences. If prefetch was started, returns the
  /// prefetched batch (zero-wait). Otherwise prepares synchronously.
  std::tuple<torch::Tensor, torch::Tensor> get_batch(
      int64_t batch_size,
      torch::Device device);

  /// Start prefetching the next batch in a background thread.
  /// Call this immediately after get_batch() to overlap data prep with compute.
  void prefetch_next(int64_t batch_size, torch::Device device);

  /// Reset shuffle indices for next epoch
  void reset_epoch();

  /// Data-parallel sharding: rank `r` of `world` sees a disjoint 1/world stride
  /// of the (identically-shuffled) chunk list, so DDP ranks train on different
  /// data. Call once after construction with the DDP rank/world_size. world<=1
  /// is single-GPU (no sharding). When sharded, the shuffle is seeded
  /// deterministically per epoch so every rank produces the SAME order.
  void set_shard(int rank, int world);

  /// Prepare for training on `device`.
  /// - CUDA + enough budget: full token array + chunk indices on GPU (fast path).
  /// - CUDA + not enough / max_gpu_tokens==-1: pinned CPU + double-buffered H2D (bounded memory).
  /// - max_gpu_tokens==0: auto (use ~25% of free VRAM as budget).
  /// - max_gpu_tokens>0: refuse full GPU residency if token count exceeds this cap.
  /// - max_gpu_tokens==-1: never upload the full corpus to GPU.
  /// No-op for non-CUDA devices (CPU training uses unpinned host tensors).
  void to_device(torch::Device device, int64_t max_gpu_tokens = 0);

  /// Check if the dataset is GPU-resident
  bool is_gpu_resident() const { return gpu_resident_; }

  /// SFT loss masking. Load a parallel .npy (same length as the token array;
  /// uint8/int) where 1 = train on this position, 0 = ignore. When set,
  /// prepare_batch_cpu() sets labels to -100 (ignore_index) wherever the mask at
  /// the LABEL position is 0 — the trainer's CE then trains ONLY on unmasked
  /// (assistant) tokens. Use the CPU/streaming path (gpu_data=0) for SFT; the
  /// GPU-resident fast path does NOT apply the mask.
  void set_loss_mask(const std::string& mask_npy_path);
  bool has_loss_mask() const { return mask_tensor_.defined(); }

 private:
  /// Prepare a batch on CPU (no device transfer yet)
  std::tuple<torch::Tensor, torch::Tensor> prepare_batch_cpu(int64_t batch_size);

  /// Prepare a batch entirely on GPU (zero CPU involvement)
  std::tuple<torch::Tensor, torch::Tensor> get_batch_gpu(int64_t batch_size);

  void drain_stream_prefetch();
  void ensure_stream_buf_capacity(int64_t batch_size);

  std::vector<int64_t> tokens_;
  torch::Tensor tokens_tensor_;  // Pre-built CPU tensor for fast gather (pinned after to_device streaming)
  torch::Tensor mask_tensor_;    // SFT loss mask [N] int64 (1=train, 0=ignore); empty = no masking
  int64_t seq_len_;
  int64_t num_chunks_;
  bool shuffle_;
  std::vector<int64_t> chunk_indices_;
  size_t chunk_cursor_;
  // Data-parallel sharding (set_shard). dp_world_ > 1 means rank dp_rank_ walks
  // the chunk list with stride dp_world_ starting at dp_rank_. epoch_ drives a
  // deterministic per-epoch shuffle seed so all ranks shuffle identically.
  int dp_rank_ = 0;
  int dp_world_ = 1;
  int64_t epoch_ = 0;

  // Async prefetch state (CPU tensors path when not streaming_mode_)
  std::future<std::tuple<torch::Tensor, torch::Tensor>> prefetch_future_;
  torch::Device prefetch_device_{torch::kCPU};
  bool has_prefetch_ = false;
  std::mutex cursor_mutex_;

  // Pinned double-buffer path (CUDA, !gpu_resident_)
  bool streaming_mode_ = false;
  std::array<torch::Tensor, 2> stream_pin_in_{};
  std::array<torch::Tensor, 2> stream_pin_la_{};
  int64_t stream_cap_b_ = 0;
  std::future<void> stream_prefetch_future_{};
  bool stream_prefetch_inflight_ = false;
  int stream_write_slot_ = 0;       // next slot for background prefetch
  int stream_last_filled_slot_ = 0; // slot to read after wait (set when prefetch is queued)

  // GPU-resident data state
  torch::Tensor gpu_tokens_tensor_;   // Full token array on CUDA
  torch::Tensor gpu_chunk_indices_;   // Shuffled chunk indices on CUDA
  int64_t gpu_cursor_ = 0;
  bool gpu_resident_ = false;
  torch::Device resident_device_{torch::kCPU};

  // Cached [0, seq_len) index ranges used to build gather indices.
  // Built once per device so we don't reallocate a seq_len tensor on every
  // get_batch call (per-step GPU allocation + kernel launch).
  torch::Tensor cpu_range_;   // [seq_len] int64 on host, built lazily in prepare_batch_cpu
  torch::Tensor gpu_range_;   // [seq_len] int64 on GPU, built lazily in get_batch_gpu
};

}  // namespace olmo_cpp
