#pragma once

/**
 * include/olmo_cpp/data/composable/document_source.hpp
 *
 * First stage of the composable data pipeline: a DocumentSource yields
 * variable-length Documents (sequences of token ids with a source-id
 * provenance tag). Concrete implementations cover in-memory test data,
 * .npy-backed real data with eos-delimited document boundaries, sequential
 * concatenation of multiple sources, with-replacement sampling, and
 * weighted mixing across multiple sources.
 *
 * Pipeline shape:
 *   DocumentSource -> TokenSource -> InstanceSource -> ComposableDataLoader
 *
 * --- Includes from this project ---
 *   - <torch/torch.h>: pulled in transitively for downstream loader types.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/composable/instance_source.cpp (PackingInstanceSource).
 *   - src/data/composable/token_source.cpp (DocumentTokenSource).
 *   - src/data/vsl_dataset.cpp (VSLInstanceSource consumes a DocumentSource).
 *
 * --- Role in training pipeline ---
 *   Source of per-document token streams. Document boundaries matter for
 *   packing strategies that prepend EOS between docs and for mixture
 *   experiments that need provenance.
 */

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <functional>

namespace olmo_cpp {

/// A document is a sequence of token IDs with a boundary marker.
/// `tokens` is variable length; `source_id` records which top-level source
/// (corpus) this document originated from for provenance tracking.
struct Document {
  std::vector<int64_t> tokens;
  int64_t source_id = 0;  // which source this came from
};

/// Abstract document source. All concrete sources are pull-style streams:
/// has_next() peeks, next() consumes, reset() rewinds.
class DocumentSource {
 public:
  virtual ~DocumentSource() = default;
  virtual bool has_next() const = 0;
  virtual Document next() = 0;
  virtual void reset() = 0;
  virtual int64_t num_documents() const = 0;
};

/// In-memory document source. Docs live entirely in process memory; useful
/// for unit tests and small ad-hoc evaluation sets.
class InMemoryDocumentSource : public DocumentSource {
 public:
  explicit InMemoryDocumentSource(std::vector<Document> docs);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<Document> docs_;  // all documents resident
  size_t cursor_ = 0;           // index of next document to emit
};

/// Numpy document source: reads .npy files with document boundary markers.
/// The .npy contains a 1-D array of token ids (uint16/int32/int64 supported).
/// If eos_token_id >= 0, the array is split on every occurrence of that id
/// (the EOS itself is included as the last token of the document). If
/// eos_token_id < 0 the entire file becomes one giant document.
class NumpyDocumentSource : public DocumentSource {
 public:
  NumpyDocumentSource(const std::string& path, int64_t eos_token_id = -1);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<Document> docs_;  // all documents materialised at construction
  size_t cursor_ = 0;
};

/// Concatenated document source: chains multiple sources end-to-end. Sources
/// are visited in order; once one exhausts we move to the next.
class ConcatenatedDocumentSource : public DocumentSource {
 public:
  explicit ConcatenatedDocumentSource(std::vector<std::unique_ptr<DocumentSource>> sources);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<std::unique_ptr<DocumentSource>> sources_;
  size_t current_source_ = 0;  // index of the source we're currently draining
};

/// Sampling document source: randomly samples from a source with replacement.
/// Effectively turns any finite source into an infinite stream by uniform
/// resampling. Pre-loads all documents into memory at construction time.
class SamplingDocumentSource : public DocumentSource {
 public:
  SamplingDocumentSource(std::unique_ptr<DocumentSource> source, int64_t seed = 42);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<Document> all_docs_;
  std::mt19937 rng_;             // Mersenne Twister, seeded from `seed_`
  int64_t seed_;                 // saved for reset()
};

/// Mixing document source: samples from multiple sources with given ratios.
/// Maintains a precomputed cumulative-distribution vector and at each next()
/// call draws U(0,1), inverse-CDF-samples a source index, and pulls one doc.
class MixingDocumentSource : public DocumentSource {
 public:
  MixingDocumentSource(std::vector<std::unique_ptr<DocumentSource>> sources,
                       std::vector<double> ratios, int64_t seed = 42);
  bool has_next() const override;
  Document next() override;
  void reset() override;
  int64_t num_documents() const override;
 private:
  std::vector<std::unique_ptr<DocumentSource>> sources_;
  std::vector<double> cumulative_ratios_;  // CDF of normalized ratios
  std::mt19937 rng_;
  int64_t seed_;
};

}  // namespace olmo_cpp
