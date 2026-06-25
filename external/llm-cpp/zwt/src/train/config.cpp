#include "zwt/train/config.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace zwt::train {

namespace {

std::string trim(std::string s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

bool parse_bool(const std::string& v) {
  std::string t = v;
  for (auto& c : t) c = static_cast<char>(std::tolower(c));
  if (t == "true" || t == "1" || t == "yes" || t == "on")  return true;
  if (t == "false"|| t == "0" || t == "no"  || t == "off") return false;
  throw std::runtime_error("config: invalid bool value '" + v + "'");
}

int64_t parse_i64(const std::string& v) {
  char* end = nullptr;
  long long x = std::strtoll(v.c_str(), &end, 10);
  if (!end || *end != '\0') throw std::runtime_error("config: bad integer '" + v + "'");
  return static_cast<int64_t>(x);
}

uint64_t parse_u64(const std::string& v) {
  char* end = nullptr;
  unsigned long long x = std::strtoull(v.c_str(), &end, 0);  // 0 = auto base (0x prefix ok)
  if (!end || *end != '\0') throw std::runtime_error("config: bad uint '" + v + "'");
  return static_cast<uint64_t>(x);
}

float parse_f32(const std::string& v) {
  char* end = nullptr;
  double x = std::strtod(v.c_str(), &end);
  if (!end || *end != '\0') throw std::runtime_error("config: bad float '" + v + "'");
  return static_cast<float>(x);
}

using KV = std::unordered_map<std::string, std::string>;
using Sections = std::unordered_map<std::string, KV>;

Sections parse_ini(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("config: cannot open " + path);

  Sections out;
  std::string line;
  std::string cur_section = "global";
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    // Strip comments (# or ;).
    for (size_t i = 0; i < line.size(); ++i) {
      if (line[i] == '#' || line[i] == ';') { line.resize(i); break; }
    }
    std::string t = trim(line);
    if (t.empty()) continue;

    if (t.front() == '[' && t.back() == ']') {
      cur_section = trim(t.substr(1, t.size() - 2));
      continue;
    }
    size_t eq = t.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error("config: line " + std::to_string(lineno)
                               + ": expected 'key = value'");
    }
    std::string key = trim(t.substr(0, eq));
    std::string val = trim(t.substr(eq + 1));
    out[cur_section][key] = val;
  }
  return out;
}

// Apply one section's key/value pairs. Unknown keys throw so typos don't get
// silently ignored — "fail loud" is the right call for a file that drives an
// overnight 3B pretraining run.
template <class F>
void apply_section(const std::string& name, const KV& kv, F&& setter) {
  for (const auto& [k, v] : kv) {
    if (!setter(k, v)) {
      throw std::runtime_error("config: unknown key [" + name + "] " + k);
    }
  }
}

}  // namespace

TrainConfig load_train_config(const std::string& path) {
  TrainConfig c;
  Sections sections = parse_ini(path);

  for (auto& [sect, kv] : sections) {
    if (sect == "model") {
      apply_section(sect, kv, [&](const std::string& k, const std::string& v) {
        if (k == "vocab_size")     { c.model.vocab_size = parse_i64(v); return true; }
        if (k == "d_model")        { c.model.d_model    = parse_i64(v); return true; }
        if (k == "n_heads")        { c.model.n_heads    = parse_i64(v); return true; }
        if (k == "n_kv_heads")     { c.model.n_kv_heads = parse_i64(v); return true; }
        if (k == "head_dim")       { c.model.head_dim   = parse_i64(v); return true; }
        if (k == "d_ffn")          { c.model.d_ffn      = parse_i64(v); return true; }
        if (k == "n_layers")       { c.model.n_layers   = parse_i64(v); return true; }
        if (k == "max_seq")        { c.model.max_seq    = parse_i64(v); return true; }
        if (k == "rope_base")      { c.model.rope_base  = parse_f32(v); return true; }
        if (k == "norm_eps")       { c.model.norm_eps   = parse_f32(v); return true; }
        if (k == "bias")           { c.model.bias       = parse_bool(v); return true; }
        if (k == "tie_embeddings") { c.model.tie_embeddings = parse_bool(v); return true; }
        return false;
      });
    } else if (sect == "data") {
      apply_section(sect, kv, [&](const std::string& k, const std::string& v) {
        if (k == "path")        { c.data_path  = v;             return true; }
        if (k == "seq_len")     { c.seq_len    = parse_i64(v);  return true; }
        if (k == "batch_size")  { c.batch_size = parse_i64(v);  return true; }
        if (k == "grad_accum")  { c.grad_accum = parse_i64(v);  return true; }
        if (k == "seed")        { c.data_seed  = parse_u64(v);  return true; }
        if (k == "shuffle")     { c.shuffle    = parse_bool(v); return true; }
        return false;
      });
    } else if (sect == "optim") {
      apply_section(sect, kv, [&](const std::string& k, const std::string& v) {
        if (k == "lr")            { c.adamw.lr           = parse_f32(v); return true; }
        if (k == "beta1")         { c.adamw.beta1        = parse_f32(v); return true; }
        if (k == "beta2")         { c.adamw.beta2        = parse_f32(v); return true; }
        if (k == "eps")           { c.adamw.eps          = parse_f32(v); return true; }
        if (k == "weight_decay")  { c.adamw.weight_decay = parse_f32(v); return true; }
        if (k == "grad_clip")     { c.grad_clip          = parse_f32(v); return true; }
        if (k == "peak_lr")       { c.schedule.peak_lr   = parse_f32(v); return true; }
        if (k == "min_lr")        { c.schedule.min_lr    = parse_f32(v); return true; }
        if (k == "warmup_steps")  { c.schedule.warmup_steps = parse_i64(v); return true; }
        if (k == "max_steps")     { c.schedule.max_steps    = parse_i64(v); return true; }
        return false;
      });
    } else if (sect == "runtime") {
      apply_section(sect, kv, [&](const std::string& k, const std::string& v) {
        if (k == "max_steps")     { c.max_steps     = parse_i64(v); return true; }
        if (k == "log_interval")  { c.log_interval  = parse_i64(v); return true; }
        if (k == "ckpt_interval") { c.ckpt_interval = parse_i64(v); return true; }
        if (k == "ckpt_path")     { c.ckpt_path     = v;            return true; }
        if (k == "resume_from")   { c.resume_from   = v;            return true; }
        if (k == "init_seed")     { c.init_seed     = parse_u64(v); return true; }
        if (k == "arena_mb")      { c.arena_mb      = parse_i64(v); return true; }
        if (k == "deterministic") { c.deterministic = parse_bool(v); return true; }
        return false;
      });
    } else if (sect == "dist") {
      apply_section(sect, kv, [&](const std::string& k, const std::string& v) {
        if (k == "rank")        { c.rank        = static_cast<int>(parse_i64(v)); return true; }
        if (k == "local_rank")  { c.local_rank  = static_cast<int>(parse_i64(v)); return true; }
        if (k == "world_size")  { c.world_size  = static_cast<int>(parse_i64(v)); return true; }
        if (k == "master_addr") { c.master_addr = v;                              return true; }
        if (k == "master_port") { c.master_port = static_cast<int>(parse_i64(v)); return true; }
        if (k == "bucket_mb")   { c.bucket_mb   = parse_i64(v);                   return true; }
        return false;
      });
    } else if (sect == "global") {
      if (kv.empty()) continue;
      throw std::runtime_error("config: keys outside any section");
    } else {
      throw std::runtime_error("config: unknown section [" + sect + "]");
    }
  }

  // Fill in convenient derived defaults. If the optimizer schedule didn't set
  // max_steps, borrow the runtime value so the cosine decay spans the run.
  if (c.schedule.max_steps == 100000 /* default */ && c.max_steps != 100000) {
    c.schedule.max_steps = c.max_steps;
  }
  if (c.schedule.peak_lr == 3e-4f /* default */ && c.adamw.lr != 3e-4f) {
    c.schedule.peak_lr = c.adamw.lr;
  }
  // Sanity.
  if (c.model.vocab_size <= 0)
    throw std::runtime_error("config: model.vocab_size missing");
  if (c.model.d_model != c.model.n_heads * c.model.head_dim)
    throw std::runtime_error("config: d_model must equal n_heads*head_dim");
  if (c.seq_len > c.model.max_seq)
    throw std::runtime_error("config: seq_len > model.max_seq");
  if (c.data_path.empty())
    throw std::runtime_error("config: data.path missing");
  return c;
}

std::string dump_train_config(const TrainConfig& c) {
  std::ostringstream os;
  os << "[model]\n"
     << "vocab_size    = " << c.model.vocab_size    << "\n"
     << "d_model       = " << c.model.d_model       << "\n"
     << "n_heads       = " << c.model.n_heads       << "\n"
     << "n_kv_heads    = " << c.model.n_kv_heads    << "\n"
     << "head_dim      = " << c.model.head_dim      << "\n"
     << "d_ffn         = " << c.model.d_ffn         << "\n"
     << "n_layers      = " << c.model.n_layers      << "\n"
     << "max_seq       = " << c.model.max_seq       << "\n"
     << "rope_base     = " << c.model.rope_base     << "\n"
     << "norm_eps      = " << c.model.norm_eps      << "\n"
     << "bias          = " << (c.model.bias ? "true" : "false") << "\n"
     << "tie_embeddings= " << (c.model.tie_embeddings ? "true" : "false") << "\n"
     << "\n[data]\n"
     << "path          = " << c.data_path           << "\n"
     << "seq_len       = " << c.seq_len             << "\n"
     << "batch_size    = " << c.batch_size          << "\n"
     << "grad_accum    = " << c.grad_accum          << "\n"
     << "seed          = " << c.data_seed           << "\n"
     << "shuffle       = " << (c.shuffle ? "true" : "false") << "\n"
     << "\n[optim]\n"
     << "lr            = " << c.adamw.lr            << "\n"
     << "beta1         = " << c.adamw.beta1         << "\n"
     << "beta2         = " << c.adamw.beta2         << "\n"
     << "eps           = " << c.adamw.eps           << "\n"
     << "weight_decay  = " << c.adamw.weight_decay  << "\n"
     << "grad_clip     = " << c.grad_clip           << "\n"
     << "peak_lr       = " << c.schedule.peak_lr    << "\n"
     << "min_lr        = " << c.schedule.min_lr     << "\n"
     << "warmup_steps  = " << c.schedule.warmup_steps << "\n"
     << "max_steps     = " << c.schedule.max_steps  << "\n"
     << "\n[runtime]\n"
     << "max_steps     = " << c.max_steps           << "\n"
     << "log_interval  = " << c.log_interval        << "\n"
     << "ckpt_interval = " << c.ckpt_interval       << "\n"
     << "ckpt_path     = " << c.ckpt_path           << "\n"
     << "resume_from   = " << c.resume_from         << "\n"
     << "init_seed     = " << c.init_seed           << "\n"
     << "arena_mb      = " << c.arena_mb            << "\n"
     << "deterministic = " << (c.deterministic ? 1 : 0) << "\n"
     << "\n[dist]\n"
     << "rank          = " << c.rank                << "\n"
     << "local_rank    = " << c.local_rank          << "\n"
     << "world_size    = " << c.world_size          << "\n"
     << "master_addr   = " << c.master_addr         << "\n"
     << "master_port   = " << c.master_port         << "\n"
     << "bucket_mb     = " << c.bucket_mb           << "\n";
  return os.str();
}

}  // namespace zwt::train
