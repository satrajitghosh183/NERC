/**
 * src/data/source_mixture.cpp
 *
 * ─── What "source mixing" is ────────────────────────────────────────
 *
 * Real LLM training rarely uses one corpus; it mixes several at fixed
 * ratios — e.g. "70% web, 20% code, 10% papers". SourceMixture takes a
 * list of (TokenSource, weight) pairs and at each batch decides which
 * source to draw from according to those weights. This is the way
 * data-engineers control what the model "specialises in" without
 * having to physically interleave files on disk.
 *
 * The sampling is **stochastic** with the requested per-source
 * probabilities, but a low-discrepancy scheme keeps the long-run
 * empirical mix close to the target (so a small batch isn't
 * accidentally 95% one source).
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/source_mixture.hpp : SourceMixture declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/composable/token_source.cpp: a SourceMixture is one
 *     possible TokenSource in the composable data pipeline.
 *
 * --- Role in training pipeline ---
 *   Used only when multiple data sources are configured. The
 *   quickstart's single-source TinyStories run does NOT exercise it.
 */
#include "olmo_cpp/data/source_mixture.hpp"
#include <fstream>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <algorithm>

namespace olmo_cpp {

std::vector<double> MixtureConfig::normalized_weights() const {
  std::vector<double> weights;
  weights.reserve(sources.size());
  for (const auto& src : sources) {
    weights.push_back(src.weight);
  }

  double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (sum <= 0.0) {
    throw std::runtime_error(
        "MixtureConfig::normalized_weights: weights must sum to positive value");
  }

  for (auto& w : weights) {
    w /= sum;
  }
  return weights;
}

MixtureConfig MixtureConfig::load_from_file(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(
        "MixtureConfig::load_from_file: cannot open " + path);
  }

  MixtureConfig config;
  std::string line;

  while (std::getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') continue;

    // Trim leading/trailing whitespace
    auto start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) continue;
    line = line.substr(start);

    // Parse: weight<tab>path[<tab>name]
    // Also support space-separated
    std::istringstream iss(line);
    SourceConfig src;

    // Try to parse weight first
    double weight;
    if (!(iss >> weight)) {
      // If first token is not a number, skip line
      continue;
    }
    src.weight = weight;

    // Read the path
    if (!(iss >> src.path)) {
      throw std::runtime_error(
          "MixtureConfig::load_from_file: missing path in line: " + line);
    }

    // Optionally read the name (rest of line)
    std::string remaining;
    if (std::getline(iss >> std::ws, remaining)) {
      // Trim trailing whitespace
      auto end = remaining.find_last_not_of(" \t\r\n");
      if (end != std::string::npos) {
        src.name = remaining.substr(0, end + 1);
      }
    }

    // If name is empty, derive from path
    if (src.name.empty()) {
      auto slash_pos = src.path.find_last_of('/');
      if (slash_pos != std::string::npos) {
        src.name = src.path.substr(slash_pos + 1);
      } else {
        src.name = src.path;
      }
    }

    config.sources.push_back(std::move(src));
  }

  if (config.sources.empty()) {
    throw std::runtime_error(
        "MixtureConfig::load_from_file: no sources found in " + path);
  }

  return config;
}

}  // namespace olmo_cpp
