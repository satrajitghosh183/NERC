/**
 * tools/mine_patterns.cpp
 *
 * Structural pattern miner used to bootstrap the structural tokenizer.
 * It walks a corpus directory, abstracts every line by replacing
 * identifiers / numeric literals / string literals with typed slot
 * placeholders ({IDENT}, {NUM}, {STR}), counts how often each abstract
 * pattern appears, scores patterns by `frequency * (tokens_saved - 1)`,
 * and writes the top-N patterns to a JSON file. Optionally merges in a
 * hand-curated "seed" pattern set so trusted patterns survive the
 * top-N cut.
 *
 * Examples:
 *   # Mine TinyStories prose patterns
 *   ./build/mine_patterns --dir data/tinystories_raw/ \
 *       --output data/structural_tokenizer/mined_patterns.json
 *
 *   # Mine source code, keep top-5000 patterns seen >=3 times
 *   ./build/mine_patterns --dir src/ --top 5000 --min-freq 3
 *
 *   # Merge mined patterns with a hand-curated seed set
 *   ./build/mine_patterns --dir src/ \
 *       --seed data/structural_tokenizer/seed_patterns.json --merge
 *
 * --- Flags ---
 *   --dir <p>          (REQUIRED) corpus directory to walk
 *   --output <p>       output JSON path
 *   --seed <p>         seed pattern JSON to optionally merge in
 *   --merge            include the seed patterns in the output
 *   --top <n>          keep top-N patterns by score (default 5000)
 *   --min-freq <n>     drop patterns seen fewer than n times (default 2)
 *   --max-files <n>    cap on files scanned (default 10000)
 *   --verbose / -v     print per-file progress
 *
 * --- Build target ---
 *   mine_patterns (CMakeLists.txt:571). Standalone executable — does
 *   not link olmo_cpp or LibTorch (only header-only filesystem/regex
 *   STL). Compiled with -O3 -march=native.
 *
 * --- Includes from this project ---
 *   (none — this tool is intentionally self-contained.)
 *
 * --- Reads / Writes ---
 *   - reads:  every text/code/data file under --dir; optional --seed JSON
 *   - writes: --output JSON file. ID space convention:
 *               * 50000-50999 reserved for seed patterns
 *               * mined patterns start at 51000 (or at the configured
 *                 base_id when --merge is not set)
 *
 * --- Role in workflow ---
 *   First step in building a structural tokenizer. The output JSON is
 *   then consumed by the StructuralTokenizer at load time — see
 *   `benchmark_tokenizer` and `prepare_data --structural-config`.
 */

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// ─── Helpers ────────────────────────────────────────────────────────────────

static std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static std::string trim(const std::string& s) {
  auto a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  auto b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Count how many BPE tokens a string would roughly produce (heuristic: split on
// spaces/punctuation boundaries, similar to GPT-2 pre-tokenization).
static int estimate_tokens(const std::string& s) {
  if (s.empty()) return 0;
  int count = 1;
  for (size_t i = 1; i < s.size(); ++i) {
    char c = s[i];
    char p = s[i - 1];
    // Token boundary heuristics (rough GPT-2 approximation)
    if (c == ' ' || c == '\t' || c == '\n') { count++; continue; }
    if (std::ispunct(c) && !std::ispunct(p)) count++;
    else if (!std::ispunct(c) && std::ispunct(p) && p != ' ') count++;
    else if (std::isupper(c) && std::islower(p)) count++;  // camelCase
  }
  return count;
}

// ─── Pattern abstraction ────────────────────────────────────────────────────

// Detect if a string looks like an identifier
static bool is_identifier(const std::string& s) {
  if (s.empty()) return false;
  if (!std::isalpha(s[0]) && s[0] != '_') return false;
  for (char c : s) {
    if (!std::isalnum(c) && c != '_') return false;
  }
  return true;
}

// Detect if a string looks like a numeric literal
static bool is_numeric(const std::string& s) {
  if (s.empty()) return false;
  size_t start = 0;
  if (s[0] == '-' || s[0] == '+') start = 1;
  if (start >= s.size()) return false;
  // hex
  if (s.size() > start + 2 && s[start] == '0' && (s[start + 1] == 'x' || s[start + 1] == 'X')) {
    for (size_t i = start + 2; i < s.size(); ++i) {
      if (!std::isxdigit(s[i]) && s[i] != '_') return false;
    }
    return true;
  }
  bool has_digit = false;
  for (size_t i = start; i < s.size(); ++i) {
    if (std::isdigit(s[i])) { has_digit = true; continue; }
    if (s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == 'f' ||
        s[i] == 'F' || s[i] == 'l' || s[i] == 'L' || s[i] == 'u' ||
        s[i] == 'U' || s[i] == '_') continue;
    if ((s[i] == '+' || s[i] == '-') && i > 0 && (s[i-1] == 'e' || s[i-1] == 'E')) continue;
    return false;
  }
  return has_digit;
}

// Detect string literals
static bool is_string_literal(const std::string& s) {
  return s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                           (s.front() == '\'' && s.back() == '\''));
}

// Replace identifiers and literals with typed slots, preserving structure.
// Returns the abstracted pattern string.
static std::string abstract_line(const std::string& line) {
  std::string result;
  size_t i = 0;
  int ident_count = 0;
  int num_count = 0;
  int str_count = 0;

  while (i < line.size()) {
    // Skip whitespace — preserve it
    if (std::isspace(line[i])) {
      result += line[i];
      ++i;
      continue;
    }

    // String literal
    if (line[i] == '"' || line[i] == '\'') {
      char quote = line[i];
      size_t start = i;
      ++i;
      while (i < line.size() && line[i] != quote) {
        if (line[i] == '\\') ++i;  // skip escaped char
        ++i;
      }
      if (i < line.size()) ++i;  // closing quote
      result += "{STR}";
      str_count++;
      continue;
    }

    // Identifier or keyword
    if (std::isalpha(line[i]) || line[i] == '_') {
      size_t start = i;
      while (i < line.size() && (std::isalnum(line[i]) || line[i] == '_')) ++i;
      std::string word = line.substr(start, i - start);

      // Keep keywords as-is (they define structure)
      static const std::unordered_set<std::string> keywords = {
        "for", "while", "if", "else", "elif", "switch", "case", "break", "continue",
        "return", "void", "int", "float", "double", "char", "bool", "auto", "const",
        "static", "class", "struct", "enum", "def", "import", "from", "in", "range",
        "function", "var", "let", "const", "new", "delete", "throw", "try", "catch",
        "finally", "public", "private", "protected", "virtual", "override", "template",
        "typename", "namespace", "using", "typedef", "sizeof", "nullptr", "true", "false",
        "null", "None", "True", "False", "self", "this", "super", "yield", "async",
        "await", "lambda", "with", "as", "not", "and", "or", "is", "pass", "raise",
        "except", "include", "define", "ifdef", "ifndef", "endif", "pragma",
        // C++ STL identifiers that are structural
        "std", "vector", "string", "map", "set", "pair", "cout", "endl", "cerr",
        // Python builtins that are structural
        "print", "len", "enumerate", "zip", "type", "isinstance",
        // Torch structural
        "torch", "Tensor", "nn", "Module",
      };

      if (keywords.count(word)) {
        result += word;
      } else {
        result += "{IDENT}";
        ident_count++;
      }
      continue;
    }

    // Numeric literal
    if (std::isdigit(line[i]) || (line[i] == '.' && i + 1 < line.size() && std::isdigit(line[i + 1]))) {
      while (i < line.size() && (std::isalnum(line[i]) || line[i] == '.' || line[i] == '_' ||
             line[i] == 'x' || line[i] == 'X')) ++i;
      result += "{NUM}";
      num_count++;
      continue;
    }

    // Operators and punctuation — keep as-is (structural)
    result += line[i];
    ++i;
  }

  return result;
}

// ─── Pattern storage ────────────────────────────────────────────────────────

struct MinedPattern {
  std::string pattern;       // abstracted pattern text
  int64_t frequency = 0;     // how many times seen
  int estimated_tokens = 0;  // tokens in the raw pattern
  int slots = 0;             // number of {SLOT} placeholders
  std::string example;       // first concrete example seen
  std::string domain;        // "code" or "prose"

  // Score: frequency × (tokens_saved - 1). The -1 accounts for the single
  // structural token that replaces the pattern.
  double score() const {
    int saved = estimated_tokens - 1;  // 1 token for the pattern itself
    if (saved <= 0) return 0;
    return static_cast<double>(frequency) * saved;
  }
};

// ─── Domain detection ───────────────────────────────────────────────────────

static std::string detect_domain(const std::string& path) {
  std::string ext;
  auto pos = path.rfind('.');
  if (pos != std::string::npos) ext = path.substr(pos);

  static const std::unordered_set<std::string> code_exts = {
    ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx", ".py", ".java", ".js", ".ts",
    ".go", ".rs", ".rb", ".swift", ".kt", ".scala", ".cs", ".m", ".mm",
  };
  static const std::unordered_set<std::string> data_exts = {
    ".json", ".jsonl", ".xml", ".yaml", ".yml", ".csv", ".tsv", ".toml",
  };

  if (code_exts.count(ext)) return "code";
  if (data_exts.count(ext)) return "data";
  return "prose";
}

// ─── Main mining logic ──────────────────────────────────────────────────────

static void mine_file(const std::string& path, const std::string& content,
                      std::unordered_map<std::string, MinedPattern>& patterns) {
  std::string domain = detect_domain(path);
  std::istringstream stream(content);
  std::string line;

  while (std::getline(stream, line)) {
    std::string trimmed = trim(line);
    if (trimmed.empty()) continue;
    if (trimmed.size() < 5) continue;   // too short to be interesting
    if (trimmed.size() > 200) continue;  // too long — not a reusable pattern

    std::string pattern = abstract_line(trimmed);
    std::string pattern_trimmed = trim(pattern);
    if (pattern_trimmed.empty()) continue;

    // Skip patterns that are entirely slots (no structural content)
    bool has_structure = false;
    for (size_t i = 0; i < pattern_trimmed.size(); ++i) {
      if (pattern_trimmed[i] != '{' && pattern_trimmed[i] != '}' &&
          !std::isspace(pattern_trimmed[i]) && !std::isalpha(pattern_trimmed[i])) {
        has_structure = true;
        break;
      }
    }
    // Also check for keywords
    if (!has_structure) {
      static const std::vector<std::string> structural_kw = {
        "for", "while", "if", "else", "return", "def", "class", "import", "function"
      };
      for (auto& kw : structural_kw) {
        if (pattern_trimmed.find(kw) != std::string::npos) {
          has_structure = true;
          break;
        }
      }
    }
    if (!has_structure) continue;

    auto& p = patterns[pattern_trimmed];
    if (p.frequency == 0) {
      p.pattern = pattern_trimmed;
      p.example = trimmed;
      p.domain = domain;
      p.estimated_tokens = estimate_tokens(trimmed);
      // Count slots
      for (size_t j = 0; j < pattern_trimmed.size(); ++j) {
        if (pattern_trimmed[j] == '{') p.slots++;
      }
    }
    p.frequency++;
  }
}

// ─── Seed pattern loading ───────────────────────────────────────────────────

struct SeedPattern {
  int id;
  std::string name;
  std::string pattern;
  std::string domain;
};

static std::vector<SeedPattern> load_seed_patterns(const std::string& path) {
  std::vector<SeedPattern> seeds;
  std::string content = read_file(path);
  if (content.empty()) return seeds;

  // Simple JSON parsing for our known format (no full JSON library needed)
  // Look for pattern objects between { "id": ..., "name": ..., "pattern": ... }
  size_t pos = 0;
  while ((pos = content.find("\"id\":", pos)) != std::string::npos) {
    SeedPattern sp;

    // Parse id
    pos += 5;
    while (pos < content.size() && !std::isdigit(content[pos])) pos++;
    sp.id = std::atoi(content.c_str() + pos);

    // Parse name
    auto name_pos = content.find("\"name\":", pos);
    if (name_pos == std::string::npos) break;
    auto q1 = content.find('"', name_pos + 7);
    auto q2 = content.find('"', q1 + 1);
    if (q1 != std::string::npos && q2 != std::string::npos)
      sp.name = content.substr(q1 + 1, q2 - q1 - 1);

    // Parse pattern
    auto pat_pos = content.find("\"pattern\":", pos);
    if (pat_pos == std::string::npos) break;
    q1 = content.find('"', pat_pos + 10);
    q2 = q1 + 1;
    while (q2 < content.size() && !(content[q2] == '"' && content[q2 - 1] != '\\')) q2++;
    if (q1 != std::string::npos && q2 < content.size())
      sp.pattern = content.substr(q1 + 1, q2 - q1 - 1);

    // Parse domain
    auto dom_pos = content.find("\"domain\":", pos);
    if (dom_pos != std::string::npos && dom_pos < content.find("\"id\":", pos + 1)) {
      q1 = content.find('"', dom_pos + 9);
      q2 = content.find('"', q1 + 1);
      if (q1 != std::string::npos && q2 != std::string::npos)
        sp.domain = content.substr(q1 + 1, q2 - q1 - 1);
    }

    seeds.push_back(sp);
    pos = q2 + 1;
  }

  return seeds;
}

// ─── JSON output ────────────────────────────────────────────────────────────

static std::string escape_json(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 10);
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:   out += c;
    }
  }
  return out;
}

static void write_output(const std::string& path,
                         const std::vector<MinedPattern>& patterns,
                         const std::vector<SeedPattern>& seeds,
                         bool merge, int base_id) {
  std::ofstream f(path);
  if (!f) {
    std::cerr << "Error: cannot write to " << path << "\n";
    return;
  }

  f << "{\n";
  f << "  \"version\": \"1.0\",\n";
  f << "  \"description\": \"Mined structural patterns from corpus analysis\",\n";
  f << "  \"total_patterns\": " << patterns.size();
  if (merge && !seeds.empty()) f << ",\n  \"seed_patterns_merged\": " << seeds.size();
  f << ",\n";
  f << "  \"patterns\": [\n";

  int id = base_id;
  bool first = true;

  // Seed patterns first (if merging)
  if (merge) {
    for (auto& sp : seeds) {
      if (!first) f << ",\n";
      first = false;
      f << "    {\n";
      f << "      \"id\": " << sp.id << ",\n";
      f << "      \"name\": \"" << escape_json(sp.name) << "\",\n";
      f << "      \"pattern\": \"" << escape_json(sp.pattern) << "\",\n";
      f << "      \"domain\": \"" << escape_json(sp.domain) << "\",\n";
      f << "      \"source\": \"seed\"\n";
      f << "    }";
    }
    // Mined patterns start after seed range
    id = 51000;  // After seed patterns which use 50000-50999
  }

  // Mined patterns
  for (auto& p : patterns) {
    if (!first) f << ",\n";
    first = false;
    f << "    {\n";
    f << "      \"id\": " << id << ",\n";
    f << "      \"pattern\": \"" << escape_json(p.pattern) << "\",\n";
    f << "      \"domain\": \"" << escape_json(p.domain) << "\",\n";
    f << "      \"frequency\": " << p.frequency << ",\n";
    f << "      \"estimated_tokens\": " << p.estimated_tokens << ",\n";
    f << "      \"slots\": " << p.slots << ",\n";
    f << "      \"score\": " << std::fixed << std::setprecision(1) << p.score() << ",\n";
    f << "      \"example\": \"" << escape_json(p.example) << "\",\n";
    f << "      \"source\": \"mined\"\n";
    f << "    }";
    id++;
  }

  f << "\n  ]\n";
  f << "}\n";
  f.close();

  std::cout << "Wrote " << path << " (" << (merge ? seeds.size() : 0)
            << " seed + " << patterns.size() << " mined = " << id - base_id << " total)\n";
}

// ─── Multi-line pattern mining ──────────────────────────────────────────────

// Mine 2-line and 3-line patterns (e.g., if/else blocks, for loop bodies)
static void mine_multiline(const std::string& content,
                           std::unordered_map<std::string, MinedPattern>& patterns,
                           const std::string& domain) {
  std::istringstream stream(content);
  std::vector<std::string> lines;
  std::string line;

  while (std::getline(stream, line)) {
    std::string trimmed = trim(line);
    if (!trimmed.empty() && trimmed.size() <= 120) {
      lines.push_back(trimmed);
    }
  }

  // 2-line patterns
  for (size_t i = 0; i + 1 < lines.size(); ++i) {
    std::string a1 = trim(abstract_line(lines[i]));
    std::string a2 = trim(abstract_line(lines[i + 1]));
    if (a1.empty() || a2.empty()) continue;
    if (a1.size() + a2.size() > 150) continue;

    std::string combined = a1 + " | " + a2;

    // Only keep if both lines have structure
    bool has_kw = false;
    for (auto& kw : {"for", "while", "if", "else", "return", "def", "class"}) {
      if (combined.find(kw) != std::string::npos) { has_kw = true; break; }
    }
    if (!has_kw) continue;

    auto& p = patterns[combined];
    if (p.frequency == 0) {
      p.pattern = combined;
      p.example = lines[i] + " | " + lines[i + 1];
      p.domain = domain;
      p.estimated_tokens = estimate_tokens(lines[i]) + estimate_tokens(lines[i + 1]);
      for (char c : combined) if (c == '{') p.slots++;
    }
    p.frequency++;
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  std::string dir_path;
  std::string output_path = "data/structural_tokenizer/mined_patterns.json";
  std::string seed_path;
  int top_n = 5000;
  int min_freq = 2;
  int max_files = 10000;
  bool merge = false;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dir" && i + 1 < argc) {
      dir_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      seed_path = argv[++i];
    } else if (arg == "--top" && i + 1 < argc) {
      top_n = std::atoi(argv[++i]);
    } else if (arg == "--min-freq" && i + 1 < argc) {
      min_freq = std::atoi(argv[++i]);
    } else if (arg == "--max-files" && i + 1 < argc) {
      max_files = std::atoi(argv[++i]);
    } else if (arg == "--merge") {
      merge = true;
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: mine_patterns [OPTIONS]\n\n"
                << "  --dir <path>        Directory to analyze\n"
                << "  --output <path>     Output JSON file (default: data/structural_tokenizer/mined_patterns.json)\n"
                << "  --seed <path>       Seed patterns JSON to merge with\n"
                << "  --merge             Merge mined patterns with seed patterns\n"
                << "  --top <n>           Keep top N patterns (default: 5000)\n"
                << "  --min-freq <n>      Minimum frequency threshold (default: 2)\n"
                << "  --max-files <n>     Max files to process (default: 10000)\n"
                << "  --verbose / -v      Show per-file progress\n";
      return 0;
    }
  }

  if (dir_path.empty()) {
    std::cerr << "Error: --dir is required\n";
    return 1;
  }

  // Load seed patterns if merging
  std::vector<SeedPattern> seeds;
  if (!seed_path.empty()) {
    seeds = load_seed_patterns(seed_path);
    std::cout << "Loaded " << seeds.size() << " seed patterns from " << seed_path << "\n";
  }

  // Scan files
  static const std::unordered_set<std::string> valid_exts = {
    ".txt", ".py", ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx",
    ".java", ".js", ".ts", ".go", ".rs", ".rb", ".swift",
    ".json", ".jsonl", ".md", ".rst", ".text", ".csv",
  };

  std::unordered_map<std::string, MinedPattern> pattern_map;
  int file_count = 0;
  int64_t total_bytes = 0;
  int64_t total_lines = 0;

  for (auto& entry : fs::recursive_directory_iterator(dir_path)) {
    if (!entry.is_regular_file()) continue;
    auto ext = entry.path().extension().string();
    if (!valid_exts.count(ext)) continue;
    if (file_count >= max_files) break;

    std::string content = read_file(entry.path().string());
    if (content.empty()) continue;

    if (verbose) {
      std::cout << "[" << (file_count + 1) << "] " << entry.path().string()
                << " (" << content.size() << " bytes)\n";
    }

    // Single-line patterns
    mine_file(entry.path().string(), content, pattern_map);

    // Multi-line patterns
    std::string domain = detect_domain(entry.path().string());
    mine_multiline(content, pattern_map, domain);

    total_bytes += content.size();
    total_lines += std::count(content.begin(), content.end(), '\n');
    file_count++;
  }

  std::cout << "\nScanned " << file_count << " files ("
            << total_bytes << " bytes, " << total_lines << " lines)\n";
  std::cout << "Found " << pattern_map.size() << " unique patterns (before filtering)\n";

  // Filter by minimum frequency
  std::vector<MinedPattern> filtered;
  for (auto& [key, p] : pattern_map) {
    if (p.frequency >= min_freq && p.estimated_tokens >= 2) {
      filtered.push_back(p);
    }
  }
  std::cout << "After min-freq=" << min_freq << " filter: " << filtered.size() << " patterns\n";

  // Sort by score (frequency × tokens_saved)
  std::sort(filtered.begin(), filtered.end(), [](const MinedPattern& a, const MinedPattern& b) {
    return a.score() > b.score();
  });

  // Keep top N
  if (static_cast<int>(filtered.size()) > top_n) {
    filtered.resize(top_n);
  }

  // Print top 30 for visibility
  std::cout << "\nTop 30 mined patterns:\n";
  std::cout << std::string(90, '-') << "\n";
  std::cout << std::left << std::setw(8) << "Freq"
            << std::setw(8) << "Tokens"
            << std::setw(10) << "Score"
            << std::setw(8) << "Domain"
            << "Pattern\n";
  std::cout << std::string(90, '-') << "\n";

  for (size_t i = 0; i < 30 && i < filtered.size(); ++i) {
    auto& p = filtered[i];
    std::string display = p.pattern;
    if (display.size() > 55) display = display.substr(0, 52) + "...";
    std::cout << std::left << std::setw(8) << p.frequency
              << std::setw(8) << p.estimated_tokens
              << std::setw(10) << std::fixed << std::setprecision(0) << p.score()
              << std::setw(8) << p.domain
              << display << "\n";
  }
  std::cout << std::string(90, '-') << "\n";

  // Write output
  int base_id = merge ? 50000 : 51000;
  write_output(output_path, filtered, seeds, merge, base_id);

  return 0;
}
