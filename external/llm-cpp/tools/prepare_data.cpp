/**
 * C++ data preparation pipeline - all C++, no Python. Fast parallel tokenization.
 *
 * Usage:
 *   prepare_data --input data/texts/ --output data/tokens.npy
 *   prepare_data --input data/texts/ --output data/tokens.npy --vocab-file vocab.json --merges-file merges.txt  # GPT-2 BPE
 *   prepare_data --download https://example.com/data.jsonl --output data/tokens.npy --vocab-file vocab.json --merges-file merges.txt
 *
 * Build: cmake --build build --target prepare_data
 */

#include <cnpy.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "olmo_cpp/data/simple_tokenizer.hpp"
#include "olmo_cpp/data/bpe_tokenizer.hpp"
#include "olmo_cpp/data/structural_tokenizer.hpp"

namespace fs = std::filesystem;

static std::vector<std::string> collect_text_files(const std::string& path) {
  std::vector<std::string> files;
  fs::path p(path);
  if (!fs::exists(p)) return files;
  if (fs::is_regular_file(p)) {
    files.push_back(path);
    return files;
  }
  for (const auto& entry : fs::recursive_directory_iterator(
           p, fs::directory_options::skip_permission_denied)) {
    if (entry.is_regular_file()) {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (ext == ".txt" || ext == ".text" || ext == ".jsonl") {
        files.push_back(entry.path().string());
      }
    }
  }
  return files;
}

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  f.seekg(0, std::ios::end);
  size_t size = f.tellg();
  f.seekg(0);
  std::string buf(size, '\0');
  f.read(&buf[0], static_cast<std::streamsize>(size));
  return buf;
}

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
static std::vector<std::string> extract_text_from_jsonl(const std::string& content) {
  std::vector<std::string> texts;
  std::istringstream iss(content);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty()) continue;
    try {
      auto j = nlohmann::json::parse(line);
      if (j.contains("text")) texts.push_back(j["text"].get<std::string>());
      else if (j.contains("content")) texts.push_back(j["content"].get<std::string>());
      else if (j.contains("story")) texts.push_back(j["story"].get<std::string>());
    } catch (...) {}
  }
  return texts;
}

/// Fetch text rows from HuggingFace datasets server API (max 100/request).
static std::vector<std::string> fetch_hf_rows(const std::string& dataset,
                                              const std::string& config,
                                              const std::string& split,
                                              size_t offset, size_t length) {
  std::string url = "https://datasets-server.huggingface.co/rows?dataset=" + dataset +
      "&config=" + config + "&split=" + split + "&offset=" + std::to_string(offset) +
      "&length=" + std::to_string(std::min(length, size_t(100)));
  std::string cmd = "curl -sL \"" + url + "\"";
  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) return {};
  std::string content;
  char buf[65536];
  while (fgets(buf, sizeof(buf), fp)) content += buf;
  pclose(fp);
  std::vector<std::string> texts;
  try {
    auto j = nlohmann::json::parse(content);
    if (!j.contains("rows")) return {};
    std::string text_key = "text";
    if (j.contains("features") && !j["features"].empty() && j["features"][0].contains("name")) {
      text_key = j["features"][0]["name"].get<std::string>();
    }
    for (const auto& row : j["rows"]) {
      if (row.contains("row") && row["row"].contains(text_key)) {
        texts.push_back(row["row"][text_key].get<std::string>());
      }
    }
  } catch (...) {}
  return texts;
}
#endif

static void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  --input <path>         File or directory with .txt/.jsonl\n"
            << "  --input <f1> <f2>...   Explicit file list\n"
            << "  --download <url>       Download from URL (raw text or JSONL)\n"
            << "  --download-hf <name>   Download from HuggingFace (e.g. roneneldan/TinyStories)\n"
            << "  --output <path>        Output .npy (default: data/tokens.npy)\n"
            << "  --max-tokens <n>       Max tokens (default: 100000000)\n"
            << "  --vocab-file <path>    GPT-2 vocab.json (enables BPE)\n"
            << "  --merges-file <path>   GPT-2 merges.txt (required with --vocab-file)\n"
            << "  --structural-config <dir>  Structural tokenizer config dir (enables structural tokenizer)\n"
            << "  --vocab <path>         Save simple tokenizer vocab\n"
            << "  --load-vocab <path>    Load simple tokenizer vocab\n"
            << "  --threads <n>           Parallel threads (default: hardware concurrency)\n"
            << "  --seed <n>             Random seed (default: 42)\n"
            << "  --no-shuffle           Don't shuffle files\n"
            << "  --random <n>           Generate n random tokens for testing\n"
            << "\n"
            << "Examples:\n"
            << "  " << prog << " --input data/texts/ --output data/tokens.npy\n"
            << "  " << prog << " --input data/ --output data/tokens.npy --vocab-file vocab.json --merges-file merges.txt --threads 8\n"
            << "\n"
            << "Download GPT-2 tokenizer (one-time):\n"
            << "  mkdir -p data/gpt2 && curl -sL https://huggingface.co/gpt2/resolve/main/vocab.json -o data/gpt2/vocab.json\n"
            << "  curl -sL https://huggingface.co/gpt2/resolve/main/merges.txt -o data/gpt2/merges.txt\n";
}

int main(int argc, char* argv[]) {
  std::string output_path = "data/tokens.npy";
  std::string vocab_path, load_vocab_path, vocab_file, merges_file, download_url, download_hf, structural_config;
  std::vector<std::string> input_paths;
  size_t max_tokens = 100'000'000;
  size_t random_tokens = 0;
  int seed = 42;
  bool shuffle = true;
  unsigned threads = std::max(1u, std::thread::hardware_concurrency());

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input") {
      while (i + 1 < argc && argv[i + 1][0] != '-') input_paths.push_back(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = static_cast<size_t>(std::stoull(argv[++i]));
    else if (arg == "--vocab" && i + 1 < argc) vocab_path = argv[++i];
    else if (arg == "--load-vocab" && i + 1 < argc) load_vocab_path = argv[++i];
    else if (arg == "--vocab-file" && i + 1 < argc) vocab_file = argv[++i];
    else if (arg == "--merges-file" && i + 1 < argc) merges_file = argv[++i];
    else if (arg == "--structural-config" && i + 1 < argc) structural_config = argv[++i];
    else if (arg == "--download" && i + 1 < argc) download_url = argv[++i];
    else if (arg == "--download-hf" && i + 1 < argc) download_hf = argv[++i];
    else if (arg == "--threads" && i + 1 < argc) threads = static_cast<unsigned>(std::stoul(argv[++i]));
    else if (arg == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
    else if (arg == "--no-shuffle") shuffle = false;
    else if (arg == "--random" && i + 1 < argc) random_tokens = static_cast<size_t>(std::stoull(argv[++i]));
    else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
  }

  bool use_structural = !structural_config.empty() && !vocab_file.empty() && !merges_file.empty();
  bool use_bpe = !use_structural && !vocab_file.empty() && !merges_file.empty();
  if (!vocab_file.empty() && merges_file.empty()) {
    std::cerr << "Error: --vocab-file and --merges-file required together for BPE\n";
    return 1;
  }

  if (random_tokens > 0) {
    std::cout << "Generating " << random_tokens << " random tokens (vocab_size=50257)\n";
    std::vector<uint16_t> tokens(random_tokens);
    std::mt19937 g(static_cast<unsigned>(seed));
    std::uniform_int_distribution<int> dist(0, 50256);
    for (size_t i = 0; i < random_tokens; ++i) tokens[i] = static_cast<uint16_t>(dist(g));
    fs::create_directories(fs::path(output_path).parent_path());
    cnpy::npy_save(output_path, tokens);
    std::cout << "Wrote " << random_tokens << " tokens to " << output_path << "\n";
    std::cout << "Vocab size: 50257. Use --vocab-size 50257 when training.\n";
    return 0;
  }

  if (input_paths.empty() && download_url.empty() && download_hf.empty()) {
    std::cerr << "Error: --input, --download, --download-hf, or --random required\n";
    print_usage(argv[0]);
    return 1;
  }

  std::vector<std::string> files;
  if (!download_hf.empty()) {
#ifdef HAS_NLOHMANN_JSON
    std::cerr << "Downloading from HuggingFace: " << download_hf << " ...\n";
    std::string config = "default";
    std::string split = "train";
    size_t max_rows = 100000;  // ~100M tokens for TinyStories
    size_t offset = 0;
    std::string tmp_dir = output_path + ".tmp_hf";
    fs::create_directories(tmp_dir);
    size_t file_idx = 0;
    while (offset < max_rows) {
      auto texts = fetch_hf_rows(download_hf, config, split, offset, 100);
      if (texts.empty()) break;
      std::string fpath = tmp_dir + "/part_" + std::to_string(file_idx++) + ".txt";
      std::ofstream f(fpath);
      for (const auto& t : texts) f << t << "\n\n";
      files.push_back(fpath);
      offset += texts.size();
      if (texts.size() < 100) break;
      if (file_idx % 100 == 0) std::cerr << "\r  Fetched " << offset << " rows..." << std::flush;
    }
    std::cerr << "\n  Fetched " << offset << " rows total\n";
    if (files.empty()) {
      std::cerr << "Error: no data from HuggingFace\n";
      return 1;
    }
#else
    std::cerr << "Error: --download-hf requires nlohmann/json\n";
    return 1;
#endif
  } else if (!download_url.empty()) {
#ifdef HAS_NLOHMANN_JSON
    std::cerr << "Downloading from " << download_url << " ...\n";
    std::string cmd = "curl -sL \"" + download_url + "\"";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) {
      std::cerr << "Error: curl not found or failed. Install curl.\n";
      return 1;
    }
    std::string content;
    char buf[65536];
    while (fgets(buf, sizeof(buf), fp)) content += buf;
    pclose(fp);
    if (content.find('{') != std::string::npos) {
      auto texts = extract_text_from_jsonl(content);
      for (size_t i = 0; i < texts.size(); ++i) {
        std::string tmp = output_path + ".tmp_" + std::to_string(i) + ".txt";
        std::ofstream f(tmp);
        f << texts[i];
        files.push_back(tmp);
      }
    } else {
      std::string tmp = output_path + ".tmp_download.txt";
      std::ofstream f(tmp);
      f << content;
      files.push_back(tmp);
    }
    if (files.empty()) {
      std::cerr << "Error: no content from download\n";
      return 1;
    }
#else
    std::cerr << "Error: --download requires nlohmann/json for JSONL. Use --input with local files.\n";
    return 1;
#endif
  } else if (input_paths.empty()) {
    std::cerr << "Error: --input or --download required\n";
    print_usage(argv[0]);
    return 1;
  } else {
    for (const auto& p : input_paths) {
      auto c = collect_text_files(p);
      files.insert(files.end(), c.begin(), c.end());
    }
  }

  if (files.empty()) {
    std::cerr << "Error: no .txt/.jsonl files found\n";
    return 1;
  }

  if (shuffle) {
    std::mt19937 g(static_cast<unsigned>(seed));
    std::shuffle(files.begin(), files.end(), g);
  }

  std::cout << "Found " << files.size() << " file(s), using " << threads << " threads\n";

  std::vector<uint32_t> all_tokens;
  all_tokens.reserve(std::min(max_tokens, static_cast<size_t>(100'000'000)));
  std::mutex out_mutex;

  auto start = std::chrono::steady_clock::now();
  uint32_t vocab_size_out = 0;

  if (use_structural) {
#ifdef HAS_NLOHMANN_JSON
    olmo_cpp::StructuralTokenizer tokenizer;
    if (!tokenizer.load(structural_config, vocab_file, merges_file)) {
      std::cerr << "Error: failed to load structural tokenizer from " << structural_config << "\n";
      return 1;
    }
    std::cout << "Using Structural tokenizer (vocab_size=" << tokenizer.vocab_size() << ")\n";

    std::atomic<size_t> files_done{0};
    std::atomic<size_t> total_tokens_produced{0};
    auto tok_start = std::chrono::steady_clock::now();

    auto process_file = [&](const std::string& path) {
      std::string text = read_file(path);
      if (text.empty()) return;
      auto ids = tokenizer.encode(text);
      {
        std::lock_guard<std::mutex> lock(out_mutex);
        if (all_tokens.size() + ids.size() <= max_tokens) {
          all_tokens.insert(all_tokens.end(), ids.begin(), ids.end());
        }
      }
      total_tokens_produced += ids.size();
      size_t done = ++files_done;
      if (done % 5 == 0 || done == files.size()) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tok_start).count();
        size_t toks = total_tokens_produced.load();
        std::cerr << "\r  Tokenized " << done << "/" << files.size()
                  << " files (" << toks << " tokens, "
                  << static_cast<size_t>(toks / (elapsed > 0 ? elapsed : 1))
                  << " tok/s)" << std::flush;
      }
    };

    std::atomic<size_t> idx{0};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; ++t) {
      workers.emplace_back([&]() {
        while (true) {
          size_t i = idx.fetch_add(1);
          if (i >= files.size()) break;
          process_file(files[i]);
        }
      });
    }
    for (auto& w : workers) w.join();
    std::cerr << "\n";
    vocab_size_out = tokenizer.vocab_size();
#else
    std::cerr << "Error: Structural tokenizer requires nlohmann/json.\n";
    return 1;
#endif
  } else if (use_bpe) {
#ifdef HAS_NLOHMANN_JSON
    olmo_cpp::BPETokenizer tokenizer;
    if (!tokenizer.load(vocab_file, merges_file)) {
      std::cerr << "Error: failed to load BPE tokenizer from " << vocab_file << " / " << merges_file << "\n";
      return 1;
    }
    std::cout << "Using GPT-2 BPE tokenizer (vocab_size=" << tokenizer.vocab_size() << ")\n";

    std::atomic<size_t> files_done{0};
    std::atomic<size_t> total_tokens_produced{0};
    auto tok_start = std::chrono::steady_clock::now();

    auto process_file = [&](const std::string& path) {
      std::string content = read_file(path);
      if (content.empty()) return;
      std::vector<uint32_t> ids;
      // For .jsonl (e.g. C4, FineWeb), tokenize only the document text field —
      // NOT the raw JSON ({"text":...,"url":...,"timestamp":...}), so the model
      // doesn't waste capacity learning JSON syntax/URLs. One EOS per document.
      bool is_jsonl = path.size() >= 6 && path.compare(path.size() - 6, 6, ".jsonl") == 0;
      if (is_jsonl) {
        for (auto& doc : extract_text_from_jsonl(content)) {
          if (doc.empty()) continue;
          tokenizer.encode_append(doc, ids);
          ids.push_back(tokenizer.eos_id());
        }
      } else {
        tokenizer.encode_append(content, ids);
        ids.push_back(tokenizer.eos_id());
      }
      if (ids.empty()) return;
      {
        std::lock_guard<std::mutex> lock(out_mutex);
        if (all_tokens.size() + ids.size() <= max_tokens) {
          all_tokens.insert(all_tokens.end(), ids.begin(), ids.end());
        }
      }
      total_tokens_produced += ids.size();
      size_t done = ++files_done;
      if (done % 5 == 0 || done == files.size()) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tok_start).count();
        size_t toks = total_tokens_produced.load();
        std::cerr << "\r  Tokenized " << done << "/" << files.size()
                  << " files (" << toks << " tokens, "
                  << static_cast<size_t>(toks / (elapsed > 0 ? elapsed : 1))
                  << " tok/s)" << std::flush;
      }
    };

    std::atomic<size_t> idx{0};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; ++t) {
      workers.emplace_back([&]() {
        while (true) {
          size_t i = idx.fetch_add(1);
          if (i >= files.size()) break;
          process_file(files[i]);
        }
      });
    }
    for (auto& w : workers) w.join();
    std::cerr << "\n";
    vocab_size_out = tokenizer.vocab_size();
#else
    std::cerr << "Error: BPE requires nlohmann/json. Build with JSON support.\n";
    return 1;
#endif
  } else {
    olmo_cpp::SimpleTokenizer tokenizer;
    if (!load_vocab_path.empty()) tokenizer.load_vocab(load_vocab_path);

    for (const auto& path : files) {
      if (all_tokens.size() >= max_tokens) break;
      std::string text = read_file(path);
      if (text.empty()) continue;
      std::string ext = fs::path(path).extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
      if (ext == ".jsonl") {
#ifdef HAS_NLOHMANN_JSON
        auto texts = extract_text_from_jsonl(text);
        for (const auto& t : texts) {
          tokenizer.encode_append(t, all_tokens);
          all_tokens.push_back(olmo_cpp::SimpleTokenizer::kEosId);
        }
#else
        tokenizer.encode_append(text, all_tokens);
        all_tokens.push_back(olmo_cpp::SimpleTokenizer::kEosId);
#endif
      } else {
        tokenizer.encode_append(text, all_tokens);
        all_tokens.push_back(olmo_cpp::SimpleTokenizer::kEosId);
      }
    }
    vocab_size_out = tokenizer.vocab_size();
    if (!vocab_path.empty()) tokenizer.save_vocab(vocab_path);
  }

  if (all_tokens.size() > max_tokens) all_tokens.resize(max_tokens);

  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Done: " << all_tokens.size() << " tokens in " << sec << " s ("
            << static_cast<size_t>(all_tokens.size() / (sec > 0 ? sec : 1)) << " tok/s)\n";

  if (all_tokens.empty()) {
    std::cerr << "Error: no tokens produced\n";
    return 1;
  }

  uint32_t max_id = 0;
  for (uint32_t t : all_tokens) if (t > max_id) max_id = t;

  fs::create_directories(fs::path(output_path).parent_path());

  if (max_id < 65536) {
    std::vector<uint16_t> out_u16(all_tokens.size());
    for (size_t i = 0; i < all_tokens.size(); ++i) out_u16[i] = static_cast<uint16_t>(all_tokens[i]);
    cnpy::npy_save(output_path, out_u16);
  } else {
    cnpy::npy_save(output_path, all_tokens);
  }

  std::cout << "Wrote " << all_tokens.size() << " tokens to " << output_path;
  if (fs::exists(output_path)) std::cout << " (" << (fs::file_size(output_path) / 1e6) << " MB)";
  std::cout << "\n";

  std::cout << "Vocab size: " << vocab_size_out << ". Use --vocab-size " << vocab_size_out << " when training.\n";

  if (!download_url.empty() || !download_hf.empty()) {
    std::error_code ec;
    for (const auto& f : files) fs::remove(f, ec);
    if (!download_hf.empty()) fs::remove_all(output_path + ".tmp_hf", ec);
  }

  return 0;
}
