#pragma once

/**
 * include/olmo_cpp/data/source_mixture.hpp
 *
 * Lightweight POD configuration describing a *mixture* of pretraining
 * corpora — one or more tokenized .npy shards, each with a sampling weight
 * and optional token cap. The actual mixing logic lives in
 * MixingDocumentSource / MixingTokenSource (see composable/) and reads the
 * normalized weights produced here.
 *
 * --- Includes from this project ---
 *   - (none — STL only).
 *
 * --- Callers (concrete uses elsewhere) ---
 *   Direct callers not located via quick grep. Designed to be parsed from
 *   a "mixture file" listed in TrainConfig and turned into a tree of
 *   MixingDocumentSource / TokenSource at data-pipeline construction time.
 *
 * --- Role in training pipeline ---
 *   Declarative description of "which corpora to mix, in what proportions,
 *   capped at how many tokens each" — read once at startup, before any
 *   tensors are built.
 */

#include <string>
#include <vector>

namespace olmo_cpp {

/// Configuration for a single data source in a mixture
struct SourceConfig {
  std::string path;           // path to .npy file (1-D token id array)
  double weight = 1.0;        // relative weight in mixture (un-normalized)
  std::string name = "";      // optional name (defaults to basename of path)
  int64_t max_tokens = -1;    // max tokens from this source (-1 = all)
};

/// Configuration for a data mixture
struct MixtureConfig {
  std::vector<SourceConfig> sources;  // ordered list of corpora
  int64_t seed = 42;                  // RNG seed for reproducible sampling

  /// Normalize weights to sum to 1. Throws if all weights are <= 0.
  /// Returned vector is parallel to sources_ and consumed by the mixers.
  std::vector<double> normalized_weights() const;

  /// Load from a text file: one source per line, "weight<tab>path[<tab>name]".
  /// Lines starting with '#' and blank lines are skipped. Throws if the file
  /// is missing or contains no valid sources.
  static MixtureConfig load_from_file(const std::string& path);
};

}  // namespace olmo_cpp
