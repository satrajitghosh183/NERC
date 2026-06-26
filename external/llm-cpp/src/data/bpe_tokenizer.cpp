/**
 * src/data/bpe_tokenizer.cpp
 *
 * Implementation of the GPT-2 byte-level BPE tokenizer declared in
 * include/olmo_cpp/data/bpe_tokenizer.hpp. The encoding pipeline is:
 *
 *   text -> pre_tokenize() -> chunks
 *   for each chunk: byte->unicode mapping -> bpe_encode_chunk() -> ids
 *
 * The decoding pipeline reverses both steps: ids -> unicode token strings
 * -> raw bytes via unicode_to_byte_.
 *
 * --- Includes from this project ---
 *   - olmo_cpp/data/bpe_tokenizer.hpp: class declaration and member layout.
 *
 * --- Callers (concrete uses elsewhere) ---
 *   - src/data/structural_tokenizer.cpp: uses BPETokenizer as fallback.
 *   - tools/prepare_data.cpp, tools/chat.cpp, tools/inspect_tokens.cpp.
 *
 * --- Role in training pipeline ---
 *   Off-line tokenization (prepare_data) and inference-time decoding.
 *   Not on the training-loop critical path: by the time training starts,
 *   text has already been turned into a token .npy file.
 */

#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <list>
#include <queue>
#include <climits>

// nlohmann/json is auto-fetched by CMake (see top-level CMakeLists.txt).
// If unavailable at build time the `load()` path returns false and the
// tokenizer can still be used for in-memory test cases.
#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace olmo_cpp {

namespace {
/// Encode Unicode codepoint to UTF-8.
std::string codepoint_to_utf8(uint32_t cp) {
  if (cp < 128) {
    return std::string(1, static_cast<char>(cp));
  }
  if (cp < 2048) {
    return std::string({
        static_cast<char>(0xC0 | (cp >> 6)),
        static_cast<char>(0x80 | (cp & 0x3F)),
    });
  }
  if (cp < 65536) {
    return std::string({
        static_cast<char>(0xE0 | (cp >> 12)),
        static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
        static_cast<char>(0x80 | (cp & 0x3F)),
    });
  }
  return std::string(1, static_cast<char>(cp));
}
}  // namespace

void BPETokenizer::init_bytes_to_unicode() {
  std::vector<int> bs, cs;
  for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) {
    bs.push_back(i);
    cs.push_back(i);
  }
  for (int i = 0x00A1; i <= 0x00AC; ++i) {
    bs.push_back(i);
    cs.push_back(i);
  }
  for (int i = 0x00AE; i <= 0x00FF; ++i) {
    bs.push_back(i);
    cs.push_back(i);
  }
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    std::string key(1, static_cast<char>(static_cast<unsigned char>(bs[i])));
    std::string val = (cs[i] < 256)
                          ? std::string(1, static_cast<char>(static_cast<unsigned char>(cs[i])))
                          : codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
    bytes_to_unicode_[key] = val;
    unicode_to_byte_[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
  }
}

// ---------------------------------------------------------------------------
// GPT-2 pre-tokenization
//
// The GPT-2 regex pattern splits text into chunks:
//   '(?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}{1,3}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// For ASCII English (TinyStories, etc.), this simplifies to:
//   - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
//   - Words: optional non-alnum char + letters  (captures " word" as one chunk)
//   - Numbers: 1-3 digits
//   - Punctuation: optional space + non-alnum non-space + optional newlines
//   - Whitespace: newlines, trailing spaces
// ---------------------------------------------------------------------------

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) {
  std::vector<std::string> chunks;
  size_t i = 0;
  size_t len = text.size();

  while (i < len) {
    unsigned char c = static_cast<unsigned char>(text[i]);

    // --- Contractions: 's 't 're 've 'm 'll 'd ---
    if (c == '\'' && i + 1 < len) {
      char next = static_cast<char>(std::tolower(text[i + 1]));
      // Check for 'll, 're, 've
      if (i + 2 < len) {
        char next2 = static_cast<char>(std::tolower(text[i + 2]));
        if ((next == 'l' && next2 == 'l') ||
            (next == 'r' && next2 == 'e') ||
            (next == 'v' && next2 == 'e')) {
          chunks.push_back(text.substr(i, 3));
          i += 3;
          continue;
        }
      }
      // Check for 's, 't, 'm, 'd
      if (next == 's' || next == 't' || next == 'm' || next == 'd') {
        chunks.push_back(text.substr(i, 2));
        i += 2;
        continue;
      }
    }

    // --- Word: optional non-alnum + letters ---
    if (std::isalpha(c) || (c == ' ' && i + 1 < len && std::isalpha(static_cast<unsigned char>(text[i + 1])))) {
      size_t start = i;
      // Include leading space if followed by letter
      if (c == ' ') ++i;
      // Consume letters
      while (i < len && std::isalpha(static_cast<unsigned char>(text[i]))) ++i;
      if (i > start) {
        chunks.push_back(text.substr(start, i - start));
        continue;
      }
    }

    // --- Numbers: 1-3 digits ---
    if (std::isdigit(c)) {
      size_t start = i;
      int count = 0;
      while (i < len && std::isdigit(static_cast<unsigned char>(text[i])) && count < 3) {
        ++i;
        ++count;
      }
      chunks.push_back(text.substr(start, i - start));
      continue;
    }

    // --- Newlines ---
    if (c == '\n' || c == '\r') {
      size_t start = i;
      while (i < len && (text[i] == '\n' || text[i] == '\r')) ++i;
      chunks.push_back(text.substr(start, i - start));
      continue;
    }

    // --- Spaces (standalone) ---
    if (c == ' ') {
      size_t start = i;
      // Check if followed by non-alnum non-space (punctuation after space)
      if (i + 1 < len && !std::isalnum(static_cast<unsigned char>(text[i + 1])) &&
          text[i + 1] != ' ' && text[i + 1] != '\n' && text[i + 1] != '\r') {
        // Space + punctuation
        ++i;
        while (i < len && !std::isalnum(static_cast<unsigned char>(text[i])) &&
               !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        // Include trailing newlines
        while (i < len && (text[i] == '\n' || text[i] == '\r')) ++i;
        chunks.push_back(text.substr(start, i - start));
        continue;
      }
      // Standalone spaces
      while (i < len && text[i] == ' ') ++i;
      chunks.push_back(text.substr(start, i - start));
      continue;
    }

    // --- Other (punctuation, symbols) ---
    {
      size_t start = i;
      while (i < len && !std::isalnum(static_cast<unsigned char>(text[i])) &&
             !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
      // Include trailing newlines
      while (i < len && (text[i] == '\n' || text[i] == '\r')) ++i;
      if (i > start) {
        chunks.push_back(text.substr(start, i - start));
        continue;
      }
    }

    // Fallback: single character
    chunks.push_back(text.substr(i, 1));
    ++i;
  }

  return chunks;
}

std::vector<uint32_t> BPETokenizer::bpe_encode_chunk(const std::string& chunk) {
  if (chunk.empty()) return {};

  // Convert each byte to its Unicode representation
  // Use a doubly-linked list for O(1) merge operations
  struct Node {
    std::string token;
    Node* prev = nullptr;
    Node* next = nullptr;
  };

  std::vector<std::unique_ptr<Node>> nodes;
  for (unsigned char c : chunk) {
    std::string key(1, static_cast<char>(c));
    auto it = bytes_to_unicode_.find(key);
    auto node = std::make_unique<Node>();
    node->token = (it != bytes_to_unicode_.end()) ? it->second : key;
    if (!nodes.empty()) {
      nodes.back()->next = node.get();
      node->prev = nodes.back().get();
    }
    nodes.push_back(std::move(node));
  }

  if (nodes.size() <= 1) {
    // Single token or empty — skip BPE
    std::vector<uint32_t> ids;
    for (auto& n : nodes) {
      auto it = vocab_.find(n->token);
      if (it != vocab_.end()) ids.push_back(it->second);
    }
    return ids;
  }

  // Helper: get merge rank for a pair, returns INT_MAX if not a valid merge.
  // Reuse a thread_local key buffer so we don't allocate a new std::string
  // on every call. The unordered_map lookup itself still hashes the bytes
  // but the per-call malloc/free pair is gone — at thousands of get_rank
  // calls per encoded chunk that's measurable.
  thread_local std::string key_buf;
  auto get_rank = [this](const std::string& a, const std::string& b) -> int {
    key_buf.clear();
    key_buf.reserve(a.size() + b.size() + 1);
    key_buf.append(a);
    key_buf.push_back('\0');
    key_buf.append(b);
    auto it = merge_ranks_.find(key_buf);
    return (it != merge_ranks_.end()) ? it->second : INT_MAX;
  };

  // Priority queue: (rank, left_node_ptr)
  // We use the rank as priority (lower rank = higher priority merge)
  using PQEntry = std::pair<int, Node*>;
  auto cmp = [](const PQEntry& a, const PQEntry& b) { return a.first > b.first; };
  std::priority_queue<PQEntry, std::vector<PQEntry>, decltype(cmp)> pq(cmp);

  // Initialize: add all adjacent pairs
  for (auto& n : nodes) {
    if (n->next) {
      int rank = get_rank(n->token, n->next->token);
      if (rank < INT_MAX) {
        pq.push({rank, n.get()});
      }
    }
  }

  // Merge loop: always merge the highest-priority pair
  while (!pq.empty()) {
    auto [rank, left] = pq.top();
    pq.pop();

    // Validate: left and left->next must still be adjacent and form this merge
    if (!left->next) continue;
    int cur_rank = get_rank(left->token, left->next->token);
    if (cur_rank != rank) continue;  // stale entry

    // Merge left and left->next. `append` extends left->token in place;
    // the `+` operator we used previously always allocated a new string
    // big enough for both halves, then move-assigned it back, so for
    // long runs we paid O(merges × token_len) in allocator overhead.
    Node* right = left->next;
    left->token.append(right->token);
    left->next = right->next;
    if (right->next) right->next->prev = left;

    // Invalidate right node (orphan it)
    right->prev = nullptr;
    right->next = nullptr;

    // Re-check new pairs
    if (left->prev) {
      int r = get_rank(left->prev->token, left->token);
      if (r < INT_MAX) pq.push({r, left->prev});
    }
    if (left->next) {
      int r = get_rank(left->token, left->next->token);
      if (r < INT_MAX) pq.push({r, left});
    }
  }

  // Collect final tokens by walking the linked list from the head
  Node* head = nodes[0].get();
  while (head->prev) head = head->prev;  // find actual head

  std::vector<uint32_t> ids;
  for (Node* n = head; n; n = n->next) {
    auto it = vocab_.find(n->token);
    if (it != vocab_.end()) {
      ids.push_back(it->second);
    }
  }
  return ids;
}

bool BPETokenizer::load(const std::string& vocab_path,
                        const std::string& merges_path) {
#ifdef HAS_NLOHMANN_JSON
  init_bytes_to_unicode();
  std::ifstream vf(vocab_path);
  if (!vf) return false;
  nlohmann::json j;
  try {
    vf >> j;
  } catch (...) {
    return false;
  }
  uint32_t max_id = 0;
  for (auto it = j.begin(); it != j.end(); ++it) {
    uint32_t id = it.value().get<uint32_t>();
    vocab_[it.key()] = id;
    if (id > max_id) max_id = id;
    if (it.key() == "<|endoftext|>") eos_id_ = id;
  }
  id_to_token_.resize(max_id + 1);
  for (const auto& [tok, id] : vocab_) {
    if (id < id_to_token_.size()) id_to_token_[id] = tok;
  }

  // Detect special tokens (<|...|>) so encode_append emits them atomically
  // instead of BPE-splitting them. Sort longest-first for greedy matching.
  special_tokens_.clear();
  for (const auto& [tok, id] : vocab_) {
    if (tok.size() >= 4 && tok.compare(0, 2, "<|") == 0 &&
        tok.compare(tok.size() - 2, 2, "|>") == 0) {
      special_tokens_.emplace_back(tok, id);
    }
  }
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  std::ifstream mf(merges_path);
  if (!mf) return false;
  std::string line;
  std::getline(mf, line);  // skip "#version: 0.2"
  while (std::getline(mf, line)) {
    if (line.empty()) continue;
    size_t sp = line.find(' ');
    if (sp == std::string::npos) continue;
    std::string a = line.substr(0, sp);
    std::string b = line.substr(sp + 1);
    int rank = static_cast<int>(merges_.size());
    merges_.emplace_back(a, b);
    merge_ranks_[a + '\0' + b] = rank;
  }
  return true;
#else
  (void)vocab_path;
  (void)merges_path;
  return false;
#endif
}

std::vector<uint32_t> BPETokenizer::encode(const std::string& text) {
  std::vector<uint32_t> out;
  encode_append(text, out);
  out.push_back(eos_id_);
  return out;
}

void BPETokenizer::encode_append(const std::string& text,
                                std::vector<uint32_t>& out) {
  auto bpe_segment = [&](const std::string& seg) {
    for (const auto& chunk : pre_tokenize(seg)) {
      auto ids = bpe_encode_chunk(chunk);
      out.insert(out.end(), ids.begin(), ids.end());
    }
  };

  if (special_tokens_.empty()) {  // plain corpora: original fast path
    bpe_segment(text);
    return;
  }

  // ChatML / instruct text: emit special tokens (<|im_start|> etc.) atomically,
  // BPE-encoding only the spans between them.
  const size_t n = text.size();
  size_t seg_start = 0, i = 0;
  while (i < n) {
    bool matched = false;
    if (text[i] == '<' && i + 1 < n && text[i + 1] == '|') {
      for (const auto& [tok, id] : special_tokens_) {  // longest-first
        if (i + tok.size() <= n && text.compare(i, tok.size(), tok) == 0) {
          if (i > seg_start) bpe_segment(text.substr(seg_start, i - seg_start));
          out.push_back(id);
          i += tok.size();
          seg_start = i;
          matched = true;
          break;
        }
      }
    }
    if (!matched) ++i;
  }
  if (n > seg_start) bpe_segment(text.substr(seg_start));
}

std::string BPETokenizer::decode(const std::vector<uint32_t>& ids) {
  std::string token_text;
  for (uint32_t id : ids) {
    if (id == eos_id_) break;
    if (legacy_decode_ && id == 33) {
      token_text += ' ';
    } else if (id < id_to_token_.size() && !id_to_token_[id].empty()) {
      token_text += id_to_token_[id];
    }
  }

  // Decode Unicode codepoints back to raw bytes
  std::string out;
  out.reserve(token_text.size());
  size_t i = 0;
  while (i < token_text.size()) {
    uint32_t cp = 0;
    size_t char_len = 0;
    auto c0 = static_cast<unsigned char>(token_text[i]);

    if (c0 < 0x80) {
      cp = c0;
      char_len = 1;
    } else if ((c0 & 0xE0) == 0xC0 && i + 1 < token_text.size()) {
      auto c1 = static_cast<unsigned char>(token_text[i + 1]);
      cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
      char_len = 2;
    } else if ((c0 & 0xF0) == 0xE0 && i + 2 < token_text.size()) {
      auto c1 = static_cast<unsigned char>(token_text[i + 1]);
      auto c2 = static_cast<unsigned char>(token_text[i + 2]);
      cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
      char_len = 3;
    } else if ((c0 & 0xF8) == 0xF0 && i + 3 < token_text.size()) {
      auto c1 = static_cast<unsigned char>(token_text[i + 1]);
      auto c2 = static_cast<unsigned char>(token_text[i + 2]);
      auto c3 = static_cast<unsigned char>(token_text[i + 3]);
      cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
      char_len = 4;
    } else {
      ++i;
      continue;
    }

    auto it = unicode_to_byte_.find(cp);
    if (it != unicode_to_byte_.end()) {
      out += static_cast<char>(it->second);
    } else {
      out.append(token_text, i, char_len);
    }
    i += char_len;
  }

  return out;
}

std::string BPETokenizer::decode(const std::vector<int64_t>& ids) {
  std::vector<uint32_t> uids(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) uids[i] = static_cast<uint32_t>(ids[i]);
  return decode(uids);
}

std::vector<std::string> BPETokenizer::get_pre_tokenized_chunks(const std::string& text) {
  return pre_tokenize(text);
}

std::string BPETokenizer::decode_token(uint32_t id) const {
  if (id < id_to_token_.size()) {
    return id_to_token_[id];
  }
  return "<unk:" + std::to_string(id) + ">";
}

}  // namespace olmo_cpp
