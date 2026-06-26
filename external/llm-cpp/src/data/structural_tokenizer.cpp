/**
 * src/data/structural_tokenizer.cpp
 *
 * ─── What a "structural tokenizer" is ───────────────────────────────
 *
 * A plain BPE tokenizer (bpe_tokenizer.cpp) just splits text into
 * sub-word pieces and assigns each an integer id. The structural
 * tokenizer wraps that with extra **role tags** that encode the
 * syntactic role each token plays — noun, verb, punctuation, etc.
 * Those role tags become an extra "stream" the model can attend to,
 * which is what the DC-MRE multi-resolution embedding consumes (see
 * src/nn/multi_res_embedding.cpp).
 *
 * Roles are derived from a lightweight set of heuristics over the
 * token string (uppercase / digits / common stop-words / punctuation).
 * Cheap to compute, no external dep on a real POS tagger.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/structural_tokenizer.hpp : declarations.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/nn/multi_res_embedding.cpp: queries roles at construction
 *     time to populate the role-stream codebook.
 *   - tools/prepare_data.cpp: optional augmentation when --structural
 *     is passed.
 *
 * --- Role in training pipeline ---
 *   Only used when DC-MRE is on (cfg.use_multi_res=1).
 */
#include "olmo_cpp/data/structural_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace olmo_cpp {

// ─── Loading ────────────────────────────────────────────────────────────────

bool StructuralTokenizer::load(const std::string& config_dir,
                               const std::string& bpe_vocab_path,
                               const std::string& bpe_merges_path) {
  // Load BPE fallback tokenizer
  if (!bpe_.load(bpe_vocab_path, bpe_merges_path)) {
    std::cerr << "StructuralTokenizer: failed to load BPE tokenizer\n";
    return false;
  }

#ifdef HAS_NLOHMANN_JSON
  // Load patterns (try mined first, fall back to seed)
  std::string patterns_path = config_dir + "/mined_patterns.json";
  {
    std::ifstream f(patterns_path);
    if (!f.good()) {
      patterns_path = config_dir + "/seed_patterns.json";
      f = std::ifstream(patterns_path);
    }
    if (f.good()) {
      try {
        nlohmann::json j;
        f >> j;
        for (auto& jp : j["patterns"]) {
          Pattern p;
          p.id = jp.value("id", 0u);
          p.name = jp.value("name", "");
          p.pattern = jp.value("pattern", "");
          p.domain = jp.value("domain", "code");
          if (jp.contains("slots") && jp["slots"].is_array()) {
            for (auto& s : jp["slots"]) {
              if (s.is_string()) p.slots.push_back(s.get<std::string>());
            }
          }
          if (p.id >= PATTERN_BASE && p.id < PATTERN_END && !p.pattern.empty()) {
            compile_pattern(p);
            patterns_.push_back(std::move(p));
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "StructuralTokenizer: error loading patterns from "
                  << patterns_path << ": " << e.what() << "\n";
      }
    }
  }

  // Sort patterns by priority (longer patterns first for greedy matching)
  std::sort(patterns_.begin(), patterns_.end(),
            [](const Pattern& a, const Pattern& b) {
              return a.priority > b.priority;
            });

  // Load identifier atoms
  std::string ident_path = config_dir + "/identifiers.json";
  {
    std::ifstream f(ident_path);
    if (f.good()) {
      try {
        nlohmann::json j;
        f >> j;
        uint32_t base = j.value("base_id", IDENT_BASE);
        uint32_t id = base;
        for (auto& atom : j["atoms"]) {
          std::string name = atom.get<std::string>();
          if (id < IDENT_END) {
            atom_map_[name] = id;
            atom_names_.push_back(name);
            id_to_string_[id] = name;
            id++;
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "StructuralTokenizer: error loading identifiers: " << e.what() << "\n";
      }
    }
  }

  // Build numeric atoms: single digits, teens, powers of 2, common constants
  {
    uint32_t id = NUMERIC_BASE;
    auto add_numeric = [&](const std::string& s) {
      if (id < NUMERIC_END) {
        numeric_map_[s] = id;
        numeric_names_.push_back(s);
        id_to_string_[id] = s;
        id++;
      }
    };

    // Single digits 0-9
    for (int i = 0; i <= 9; ++i) add_numeric(std::to_string(i));
    // Teens and common small numbers
    for (int i = 10; i <= 20; ++i) add_numeric(std::to_string(i));
    // Tens
    for (int i = 30; i <= 100; i += 10) add_numeric(std::to_string(i));
    // Powers of 2
    for (int p = 7; p <= 20; ++p) add_numeric(std::to_string(1 << p)); // 128..1048576
    // Common ML constants
    add_numeric("256"); add_numeric("512"); add_numeric("768");
    add_numeric("1024"); add_numeric("2048"); add_numeric("4096");
    add_numeric("8192"); add_numeric("16384"); add_numeric("32768");
    add_numeric("65536"); add_numeric("50257"); add_numeric("50304");
    // Float constants
    add_numeric("0.0"); add_numeric("1.0"); add_numeric("0.1");
    add_numeric("0.01"); add_numeric("0.001"); add_numeric("0.0001");
    add_numeric("1e-4"); add_numeric("1e-5"); add_numeric("1e-6");
    add_numeric("1e-8"); add_numeric("3e-4"); add_numeric("0.5");
    add_numeric("0.9"); add_numeric("0.99"); add_numeric("0.999");
    add_numeric("2.0"); add_numeric("-1"); add_numeric("-1.0");
    add_numeric("1e6"); add_numeric("1e9");
  }

  // Build decode map for patterns
  for (auto& p : patterns_) {
    id_to_string_[p.id] = "[" + (p.name.empty() ? p.pattern : p.name) + "]";
  }

#else
  std::cerr << "StructuralTokenizer: built without JSON support, "
            << "only BPE fallback available\n";
#endif

  std::cout << "StructuralTokenizer loaded: " << patterns_.size() << " patterns, "
            << atom_map_.size() << " identifier atoms, "
            << numeric_map_.size() << " numeric atoms\n";
  return true;
}

// ─── Pattern compilation ────────────────────────────────────────────────────

void StructuralTokenizer::compile_pattern(Pattern& p) {
  // Split pattern into literal parts (text between {SLOT} placeholders)
  // e.g., "for ({TYPE} {VAR} = {EXPR})" → ["for (", " ", " = ", ")"]
  p.literal_parts.clear();
  std::string current;
  size_t i = 0;
  while (i < p.pattern.size()) {
    if (p.pattern[i] == '{') {
      p.literal_parts.push_back(current);
      current.clear();
      // Skip to closing }
      while (i < p.pattern.size() && p.pattern[i] != '}') ++i;
      if (i < p.pattern.size()) ++i;  // skip '}'
    } else {
      current += p.pattern[i];
      ++i;
    }
  }
  p.literal_parts.push_back(current);

  // Priority = total literal character count (prefer patterns with more fixed structure)
  p.priority = 0;
  for (auto& part : p.literal_parts) {
    p.priority += static_cast<int>(part.size());
  }
}

// ─── Pattern matching ───────────────────────────────────────────────────────

size_t StructuralTokenizer::match_pattern(const Pattern& p, const std::string& text,
                                          size_t pos,
                                          std::vector<std::string>& slot_values) const {
  slot_values.clear();
  size_t cur = pos;

  for (size_t i = 0; i < p.literal_parts.size(); ++i) {
    const std::string& lit = p.literal_parts[i];

    // Match literal part
    if (!lit.empty()) {
      if (cur + lit.size() > text.size()) return std::string::npos;
      if (text.compare(cur, lit.size(), lit) != 0) return std::string::npos;
      cur += lit.size();
    }

    // If not the last literal part, extract slot value up to next literal
    if (i + 1 < p.literal_parts.size()) {
      const std::string& next_lit = p.literal_parts[i + 1];
      if (next_lit.empty()) {
        // Last slot consumes to end of line/statement
        size_t end = text.find_first_of("\n;", cur);
        if (end == std::string::npos) end = text.size();
        slot_values.push_back(text.substr(cur, end - cur));
        cur = end;
      } else {
        // Find next literal
        size_t found = text.find(next_lit, cur);
        if (found == std::string::npos || found - cur > 200) return std::string::npos;
        slot_values.push_back(text.substr(cur, found - cur));
        cur = found;
      }
    }
  }

  return cur;
}

// ─── Domain detection ───────────────────────────────────────────────────────

StructuralTokenizer::Domain StructuralTokenizer::detect_domain(const std::string& text) const {
  // Simple heuristic based on character frequency
  int braces = 0, semicolons = 0, parens = 0;
  int json_start = 0;
  bool has_def = false, has_include = false;

  size_t check_len = std::min(text.size(), size_t(2000));
  for (size_t i = 0; i < check_len; ++i) {
    switch (text[i]) {
      case '{': case '}': braces++; break;
      case ';': semicolons++; break;
      case '(': case ')': parens++; break;
    }
  }

  // Check for JSON: starts with { or [ and has lots of quoted keys
  if (!text.empty() && (text[0] == '{' || text[0] == '[')) {
    int quotes = 0;
    for (size_t i = 0; i < check_len; ++i) {
      if (text[i] == '"') quotes++;
    }
    if (quotes > 4) return Domain::DATA;
  }

  // Check for code markers
  if (text.find("#include") != std::string::npos ||
      text.find("def ") != std::string::npos ||
      text.find("function ") != std::string::npos ||
      text.find("class ") != std::string::npos) {
    return Domain::CODE;
  }

  if (semicolons > 3 || (braces > 2 && parens > 4)) return Domain::CODE;
  return Domain::PROSE;
}

// ─── Identifier splitting ───────────────────────────────────────────────────

std::vector<std::string> StructuralTokenizer::split_identifier(const std::string& ident) const {
  std::vector<std::string> parts;
  std::string current;

  for (size_t i = 0; i < ident.size(); ++i) {
    char c = ident[i];

    if (c == '_') {
      // snake_case boundary
      if (!current.empty()) {
        parts.push_back(current);
        current.clear();
      }
    } else if (std::isupper(c) && !current.empty() && std::islower(current.back())) {
      // camelCase boundary: "camelCase" → "camel", "Case"
      parts.push_back(current);
      current.clear();
      current += static_cast<char>(std::tolower(c));
    } else {
      current += static_cast<char>(std::tolower(c));
    }
  }
  if (!current.empty()) parts.push_back(current);
  return parts;
}

// ─── Atom encoding ──────────────────────────────────────────────────────────

bool StructuralTokenizer::try_atom_encode(const std::string& word, std::vector<uint32_t>& out) {
  // Try exact match first
  auto it = atom_map_.find(word);
  if (it != atom_map_.end()) {
    out.push_back(it->second);
    last_stats_.atom_tokens++;
    return true;
  }

  // Try splitting into sub-atoms (camelCase/snake_case)
  auto parts = split_identifier(word);
  if (parts.size() <= 1) return false;

  // Check that ALL parts have atom mappings
  std::vector<uint32_t> temp;
  for (auto& part : parts) {
    auto pit = atom_map_.find(part);
    if (pit == atom_map_.end()) return false;
    temp.push_back(pit->second);
  }

  // All parts matched — emit them
  for (auto id : temp) {
    out.push_back(id);
    last_stats_.atom_tokens++;
  }
  return true;
}

bool StructuralTokenizer::try_numeric_encode(const std::string& num_str, std::vector<uint32_t>& out) {
  auto it = numeric_map_.find(num_str);
  if (it != numeric_map_.end()) {
    out.push_back(it->second);
    last_stats_.atom_tokens++;
    return true;
  }
  return false;
}

// ─── Pattern matching on a line ─────────────────────────────────────────────

bool StructuralTokenizer::try_pattern_match(const std::string& line, std::vector<uint32_t>& out) {
  std::string trimmed = line;
  // Trim leading whitespace
  size_t start = trimmed.find_first_not_of(" \t");
  if (start == std::string::npos) return false;
  std::string ws_prefix = trimmed.substr(0, start);
  trimmed = trimmed.substr(start);

  // Try each pattern (sorted by priority — longest first)
  for (auto& p : patterns_) {
    std::vector<std::string> slot_values;
    size_t end = match_pattern(p, trimmed, 0, slot_values);

    if (end != std::string::npos && end == trimmed.size()) {
      // Full line match! Emit: pattern token, then encode each slot value
      out.push_back(p.id);
      last_stats_.pattern_tokens++;

      for (auto& sv : slot_values) {
        std::string val = sv;
        // Trim whitespace from slot values
        while (!val.empty() && std::isspace(val.front())) val.erase(val.begin());
        while (!val.empty() && std::isspace(val.back())) val.pop_back();

        if (val.empty()) continue;

        // Try atom encoding for identifiers
        bool is_ident = true;
        for (char c : val) {
          if (!std::isalnum(c) && c != '_') { is_ident = false; break; }
        }
        if (is_ident && try_atom_encode(val, out)) continue;

        // Try numeric encoding
        bool is_num = true;
        for (char c : val) {
          if (!std::isdigit(c) && c != '.' && c != '-' && c != 'e' && c != 'E') {
            is_num = false; break;
          }
        }
        if (is_num && try_numeric_encode(val, out)) continue;

        // BPE fallback for complex slot values
        bpe_fallback(val, out);
      }
      return true;
    }
  }
  return false;
}

// ─── BPE fallback ───────────────────────────────────────────────────────────

void StructuralTokenizer::bpe_fallback(const std::string& text, std::vector<uint32_t>& out) {
  auto ids = bpe_.encode(text);
  // Remove trailing EOS that BPE adds
  if (!ids.empty() && ids.back() == bpe_.eos_id()) {
    ids.pop_back();
  }
  for (auto id : ids) {
    // BPE tokens stay in range [0, 50000)
    if (id < BPE_END) {
      out.push_back(id);
      last_stats_.bpe_tokens++;
    }
  }
}

// ─── Main encode ────────────────────────────────────────────────────────────

std::vector<uint32_t> StructuralTokenizer::encode(const std::string& text) {
  last_stats_ = {};
  last_stats_.input_bytes = static_cast<int>(text.size());

  std::vector<uint32_t> result;
  result.reserve(text.size() / 3);  // rough estimate

  // Process line by line
  std::istringstream stream(text);
  std::string line;
  bool first_line = true;

  while (std::getline(stream, line)) {
    // Add newline token between lines (except first)
    if (!first_line) {
      // Encode newline via BPE
      bpe_fallback("\n", result);
    }
    first_line = false;

    if (line.empty()) continue;

    // Preserve leading whitespace via BPE (indentation matters)
    size_t indent_end = line.find_first_not_of(" \t");
    if (indent_end == std::string::npos) {
      // All whitespace line
      bpe_fallback(line, result);
      continue;
    }

    std::string indent = line.substr(0, indent_end);
    std::string content = line.substr(indent_end);

    // Encode indentation via BPE
    if (!indent.empty()) {
      bpe_fallback(indent, result);
    }

    // Stage 1: Try structural pattern match on content
    if (try_pattern_match(content, result)) {
      continue;
    }

    // Stage 2-3: Word-by-word atom encoding with BPE fallback
    // Tokenize the content into words and punctuation
    size_t i = 0;
    std::string pending_bpe;  // accumulate text for BPE fallback

    auto flush_bpe = [&]() {
      if (!pending_bpe.empty()) {
        bpe_fallback(pending_bpe, result);
        pending_bpe.clear();
      }
    };

    while (i < content.size()) {
      // Skip spaces (add to pending for BPE)
      if (std::isspace(content[i])) {
        pending_bpe += content[i];
        ++i;
        continue;
      }

      // Try to extract an identifier
      if (std::isalpha(content[i]) || content[i] == '_') {
        size_t start = i;
        while (i < content.size() && (std::isalnum(content[i]) || content[i] == '_')) ++i;
        std::string word = content.substr(start, i - start);

        // Try atom encoding
        std::vector<uint32_t> atom_ids;
        if (try_atom_encode(word, atom_ids)) {
          flush_bpe();
          for (auto aid : atom_ids) result.push_back(aid);
        } else {
          pending_bpe += word;
        }
        continue;
      }

      // Try to extract a number
      if (std::isdigit(content[i]) ||
          (content[i] == '-' && i + 1 < content.size() && std::isdigit(content[i + 1]))) {
        size_t start = i;
        if (content[i] == '-') ++i;
        while (i < content.size() && (std::isdigit(content[i]) || content[i] == '.' ||
               content[i] == 'e' || content[i] == 'E' || content[i] == '-' || content[i] == '+'))
          ++i;
        // Trailing type suffixes
        while (i < content.size() && (content[i] == 'f' || content[i] == 'F' ||
               content[i] == 'l' || content[i] == 'L' || content[i] == 'u' || content[i] == 'U'))
          ++i;

        std::string num = content.substr(start, i - start);
        std::vector<uint32_t> num_ids;
        if (try_numeric_encode(num, num_ids)) {
          flush_bpe();
          for (auto nid : num_ids) result.push_back(nid);
        } else {
          pending_bpe += num;
        }
        continue;
      }

      // Punctuation/operators — accumulate for BPE
      pending_bpe += content[i];
      ++i;
    }
    flush_bpe();
  }

  // Append EOS
  result.push_back(eos_id_);
  last_stats_.total_tokens = static_cast<int>(result.size());

  return result;
}

// ─── Decode ─────────────────────────────────────────────────────────────────

std::string StructuralTokenizer::decode(const std::vector<uint32_t>& ids) {
  std::string result;

  for (auto id : ids) {
    if (id == eos_id_) break;

    if (id < BPE_END) {
      // BPE token
      result += bpe_.decode_token(id);
    } else if (id >= PATTERN_BASE && id < PATTERN_END) {
      // Pattern token — decode as pattern name
      auto it = id_to_string_.find(id);
      if (it != id_to_string_.end()) {
        result += it->second;
      } else {
        result += "<pat:" + std::to_string(id) + ">";
      }
    } else if (id >= IDENT_BASE && id < IDENT_END) {
      // Identifier atom
      uint32_t idx = id - IDENT_BASE;
      if (idx < atom_names_.size()) {
        result += atom_names_[idx];
      } else {
        result += "<id:" + std::to_string(id) + ">";
      }
    } else if (id >= NUMERIC_BASE && id < NUMERIC_END) {
      // Numeric atom
      uint32_t idx = id - NUMERIC_BASE;
      if (idx < numeric_names_.size()) {
        result += numeric_names_[idx];
      } else {
        result += "<num:" + std::to_string(id) + ">";
      }
    } else {
      result += "<unk:" + std::to_string(id) + ">";
    }
  }

  return result;
}

std::string StructuralTokenizer::decode_token(uint32_t id) const {
  if (id == eos_id_) return "<eos>";

  if (id < BPE_END) {
    return bpe_.decode_token(id);
  }

  auto it = id_to_string_.find(id);
  if (it != id_to_string_.end()) return it->second;

  if (id >= IDENT_BASE && id < IDENT_END) {
    uint32_t idx = id - IDENT_BASE;
    if (idx < atom_names_.size()) return atom_names_[idx];
  }
  if (id >= NUMERIC_BASE && id < NUMERIC_END) {
    uint32_t idx = id - NUMERIC_BASE;
    if (idx < numeric_names_.size()) return numeric_names_[idx];
  }

  return "<unk:" + std::to_string(id) + ">";
}

}  // namespace olmo_cpp
