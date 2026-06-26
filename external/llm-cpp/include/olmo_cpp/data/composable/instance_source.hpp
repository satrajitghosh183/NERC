#pragma once

/**
 * include/olmo_cpp/data/composable/instance_source.hpp
 *
 * Third stage of the composable data pipeline: an InstanceSource yields
 * fixed-length (input_ids, labels) pairs ready for batching. Concrete
 * implementations cover concat-and-chunk (the GPT pretraining default),
 * pack-with-padding, all-random (for benchmarks), uniform mixing across
 * multiple sources, and a shuffle-buffer wrapper.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/composable/token_source.hpp: ConcatAndChunk consumes
 *     a flat TokenSource.
 *   - olmo_cpp/data/composable/document_source.hpp: PackingInstanceSource
 *     prefers document boundaries.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/composable/data_loader.cpp: ComposableDataLoader stacks
 *     batch_size Instances into a torch::Tensor batch.
 *   - src/data/vsl_dataset.cpp: VSLInstanceSource derives from this
 *     interface.
 *
 * --- Role in training pipeline ---
 *   Materialises individual training examples; the level at which seq_len
 *   is fixed and labels are computed (input shifted by 1).
 */

#include "olmo_cpp/data/composable/token_source.hpp"
#include "olmo_cpp/data/composable/document_source.hpp"
#include <vector>
#include <memory>
#include <random>

namespace olmo_cpp {

/// An instance is a fixed-length token sequence ready for training.
/// labels[i] is conceptually the next-token target for input_ids[i] —
/// i.e. labels = tokens[1..N+1] when input_ids = tokens[0..N].
struct Instance {
  std::vector<int64_t> input_ids;
  std::vector<int64_t> labels;  // shifted by 1
};

/// Abstract instance source. Pull-style: has_next/next/reset.
class InstanceSource {
 public:
  virtual ~InstanceSource() = default;
  virtual bool has_next() const = 0;
  virtual Instance next() = 0;
  virtual void reset() = 0;
};

/// Concat-and-chunk: concatenate all tokens, chunk into seq_len pieces.
/// This is the canonical GPT pretraining recipe: ignore document
/// boundaries entirely, slice the giant token stream into seq_len-long
/// windows, derive labels by shifting one position.
class ConcatAndChunkInstanceSource : public InstanceSource {
 public:
  ConcatAndChunkInstanceSource(std::unique_ptr<TokenSource> source, int64_t seq_len);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  std::unique_ptr<TokenSource> source_;
  int64_t seq_len_;
  std::vector<int64_t> buffer_;  // accumulated tokens; need seq_len+1 to emit
};

/// Packing instance source: packs multiple documents into seq_len with padding.
/// Respects document boundaries (optionally inserts eos_token_id between
/// documents) and pads at the end of the stream when there aren't enough
/// tokens left for a full instance.
class PackingInstanceSource : public InstanceSource {
 public:
  PackingInstanceSource(std::unique_ptr<DocumentSource> source, int64_t seq_len,
                        int64_t pad_token_id = 0, int64_t eos_token_id = -1);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  std::unique_ptr<DocumentSource> source_;
  int64_t seq_len_, pad_token_id_, eos_token_id_;
  std::vector<int64_t> buffer_;  // packed token buffer
};

/// Random instance source: generates random token sequences (for testing).
/// Used in benchmarks where you want to pin compute throughput without any
/// real data dependency.
class RandomInstanceSource : public InstanceSource {
 public:
  RandomInstanceSource(int64_t seq_len, int64_t vocab_size, int64_t num_instances,
                       int64_t seed = 42);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  int64_t seq_len_, vocab_size_, num_instances_;
  int64_t cursor_ = 0;          // number of instances emitted so far
  std::mt19937 rng_;
  int64_t seed_;
};

/// Mixing instance source: mixes instances from multiple sources by ratio.
/// Same CDF-sampling pattern as MixingDocumentSource, but at the
/// post-chunking level.
class MixingInstanceSource : public InstanceSource {
 public:
  MixingInstanceSource(std::vector<std::unique_ptr<InstanceSource>> sources,
                       std::vector<double> ratios, int64_t seed = 42);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  std::vector<std::unique_ptr<InstanceSource>> sources_;
  std::vector<double> cumulative_ratios_;
  std::mt19937 rng_;
  int64_t seed_;
};

/// Shuffled instance source: buffers and shuffles instances. Approximates
/// a random shuffle of the underlying stream by maintaining a fixed-size
/// reservoir; adequate for breaking local order without needing the entire
/// dataset in memory.
class ShuffledInstanceSource : public InstanceSource {
 public:
  ShuffledInstanceSource(std::unique_ptr<InstanceSource> source,
                        int64_t buffer_size = 10000, int64_t seed = 42);
  bool has_next() const override;
  Instance next() override;
  void reset() override;
 private:
  std::unique_ptr<InstanceSource> source_;
  std::vector<Instance> buffer_;  // shuffled reservoir
  int64_t buffer_size_;           // capacity of the reservoir
  size_t cursor_ = 0;             // index into the shuffled buffer_
  std::mt19937 rng_;
  int64_t seed_;
  bool filled_ = false;           // true once buffer_ has been populated once
};

}  // namespace olmo_cpp
