/**
 * src/data/simple_tokenizer.cpp
 *
 * Tiny whitespace-and-punctuation tokenizer used for unit tests and
 * toy examples. Splits text on whitespace, strips punctuation, looks
 * each word up in a small vocab file. NOT used for real training —
 * the GPT-2 BPE in bpe_tokenizer.cpp is what you want for that.
 *
 * Useful if you want to rerun the model on a custom toy vocabulary
 * without bringing in the BPE machinery.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/simple_tokenizer.hpp : SimpleTokenizer declaration.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - tools/prepare_data.cpp: optional fallback path when no BPE
 *     vocab is supplied.
 *
 * --- Role in training pipeline ---
 *   Test/example only. Not on the TinyStories quickstart path.
 */
#include "olmo_cpp/data/simple_tokenizer.hpp"
#include <fstream>
#include <stdexcept>

namespace olmo_cpp {

bool SimpleTokenizer::load_vocab(const std::string& path) {
  std::ifstream f(path);
  if (!f) return false;
  vocab_.clear();
  next_id_ = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    vocab_[line] = next_id_++;
  }
  return true;
}

bool SimpleTokenizer::save_vocab(const std::string& path) const {
  std::ofstream f(path);
  if (!f) return false;
  std::vector<std::pair<std::string, uint32_t>> sorted(vocab_.begin(),
                                                        vocab_.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  for (const auto& p : sorted) {
    f << p.first << "\n";
  }
  return true;
}

}  // namespace olmo_cpp
