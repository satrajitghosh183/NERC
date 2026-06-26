#include "zwt/data/token_loader.hpp"
#include "zwt/core/stream.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt::data {

namespace {

// Parse a NumPy .npy v1.0/v2.0 header and return (dtype_string, numel, data_offset).
// Only handles the subset we actually produce in prepare_data: 1-D i64 / u32 / u16.
//
// Format reference (npy 1.0):
//   [0..6)   magic '\x93NUMPY'
//   [6]      major = 1
//   [7]      minor = 0
//   [8..10)  header_len (u16 LE)
//   [10..10+header_len) ASCII dict  "{'descr': '<i8', 'fortran_order': False, 'shape': (N,), }"
//   [hdr_end..)  raw data
struct NpyInfo {
  std::string descr;
  int64_t     numel;
  int64_t     data_offset;
};

NpyInfo parse_npy_header(std::ifstream& f) {
  char magic[6];
  if (!f.read(magic, 6) || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("TokenLoader: not an .npy file (bad magic)");
  }
  uint8_t major = 0, minor = 0;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t h16 = 0;
    f.read(reinterpret_cast<char*>(&h16), 2);
    header_len = h16;
  } else if (major == 2 || major == 3) {
    uint32_t h32 = 0;
    f.read(reinterpret_cast<char*>(&h32), 4);
    header_len = h32;
  } else {
    throw std::runtime_error("TokenLoader: unsupported .npy major version");
  }

  std::string header(header_len, '\0');
  if (!f.read(header.data(), header_len)) {
    throw std::runtime_error("TokenLoader: truncated .npy header");
  }

  // Minimal dict parsing — the header is ASCII and always well-formed. We
  // only need `descr` and `shape`.
  auto find_value = [&](const std::string& key) -> std::string {
    std::string needle = "'" + key + "'";
    size_t p = header.find(needle);
    if (p == std::string::npos) throw std::runtime_error("TokenLoader: npy missing " + key);
    p = header.find(':', p);
    if (p == std::string::npos) throw std::runtime_error("TokenLoader: npy bad " + key);
    ++p;
    while (p < header.size() && header[p] == ' ') ++p;
    return header.substr(p);
  };

  NpyInfo info;

  // descr: '<i8' or similar (quoted).
  {
    std::string rest = find_value("descr");
    if (rest.size() < 2 || rest[0] != '\'') {
      throw std::runtime_error("TokenLoader: bad descr");
    }
    size_t end = rest.find('\'', 1);
    if (end == std::string::npos) throw std::runtime_error("TokenLoader: bad descr");
    info.descr = rest.substr(1, end - 1);
  }

  // fortran_order: must be False.
  {
    std::string rest = find_value("fortran_order");
    if (rest.find("True") != std::string::npos) {
      throw std::runtime_error("TokenLoader: fortran_order=True not supported");
    }
  }

  // shape: (N,) or (N, M, ...). Multiply the components to get total numel.
  {
    std::string rest = find_value("shape");
    size_t open = rest.find('(');
    size_t close = rest.find(')', open + 1);
    if (open == std::string::npos || close == std::string::npos) {
      throw std::runtime_error("TokenLoader: bad shape");
    }
    std::string dims = rest.substr(open + 1, close - open - 1);
    int64_t numel = 1;
    bool any = false;
    std::string cur;
    auto flush = [&]() {
      if (cur.empty()) return;
      numel *= std::atoll(cur.c_str());
      any = true;
      cur.clear();
    };
    for (char c : dims) {
      if (c == ',' || c == ' ') flush();
      else if (c >= '0' && c <= '9') cur += c;
    }
    flush();
    if (!any) throw std::runtime_error("TokenLoader: empty shape");
    info.numel = numel;
  }

  info.data_offset = f.tellg();
  return info;
}

// Convert any integer stream into i64. Handles the common token dtypes that
// prepare_data / HF tokenizers produce: i64 native, u32 (rare), u16 (common
// for GPT-2-sized vocabs — cuts disk by 4x).
std::vector<int64_t> load_tokens_from_ifstream(std::ifstream& f,
                                               const std::string& descr,
                                               int64_t numel,
                                               int64_t data_offset) {
  f.seekg(data_offset, std::ios::beg);
  std::vector<int64_t> out(static_cast<size_t>(numel));

  auto read_raw = [&](size_t elem_bytes, auto convert) {
    std::vector<uint8_t> buf(static_cast<size_t>(numel) * elem_bytes);
    if (!f.read(reinterpret_cast<char*>(buf.data()),
                static_cast<std::streamsize>(buf.size()))) {
      throw std::runtime_error("TokenLoader: short read on token stream");
    }
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = convert(buf.data() + i * elem_bytes);
    }
  };

  // Accept both explicit-endian ('<i8') and native-endian ('|i1', 'i8').
  // All supported hosts are little-endian so we don't byteswap.
  if (descr == "<i8" || descr == "i8" || descr == "=i8" || descr == "int64") {
    if (!f.read(reinterpret_cast<char*>(out.data()),
                static_cast<std::streamsize>(numel * 8))) {
      throw std::runtime_error("TokenLoader: short read on i64 stream");
    }
  } else if (descr == "<u4" || descr == "u4" || descr == "uint32") {
    read_raw(4, [](const uint8_t* p) -> int64_t {
      uint32_t v;
      std::memcpy(&v, p, 4);
      return static_cast<int64_t>(v);
    });
  } else if (descr == "<i4" || descr == "i4" || descr == "int32") {
    read_raw(4, [](const uint8_t* p) -> int64_t {
      int32_t v;
      std::memcpy(&v, p, 4);
      return static_cast<int64_t>(v);
    });
  } else if (descr == "<u2" || descr == "u2" || descr == "uint16") {
    read_raw(2, [](const uint8_t* p) -> int64_t {
      uint16_t v;
      std::memcpy(&v, p, 2);
      return static_cast<int64_t>(v);
    });
  } else {
    throw std::runtime_error("TokenLoader: unsupported .npy dtype '" + descr + "'");
  }
  return out;
}

// Load i64 tokens from either a .npy file (auto-detected by magic) or a raw
// i64 stream. Raw streams are accepted for back-compat with tools that dump
// tokens directly.
std::vector<int64_t> load_tokens(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("TokenLoader: cannot open " + path);

  // Peek at magic.
  char peek[6] = {};
  f.read(peek, 6);
  f.seekg(0, std::ios::beg);
  bool is_npy = (f.gcount() == 6 && std::memcmp(peek, "\x93NUMPY", 6) == 0);

  if (is_npy) {
    NpyInfo info = parse_npy_header(f);
    return load_tokens_from_ifstream(f, info.descr, info.numel, info.data_offset);
  }

  // Raw i64 stream fallback.
  f.seekg(0, std::ios::end);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  if (n % 8 != 0) {
    throw std::runtime_error("TokenLoader: raw stream size not a multiple of 8");
  }
  std::vector<int64_t> out(static_cast<size_t>(n) / 8);
  if (!f.read(reinterpret_cast<char*>(out.data()), n)) {
    throw std::runtime_error("TokenLoader: read error");
  }
  return out;
}

}  // namespace

TokenLoader::TokenLoader(Options opts) : opts_(std::move(opts)) {
  if (opts_.seq_len <= 0) throw std::runtime_error("TokenLoader: seq_len must be > 0");
  if (opts_.batch_size <= 0) throw std::runtime_error("TokenLoader: batch_size must be > 0");
  if (opts_.ring_size < 2) opts_.ring_size = 2;

  tokens_ = load_tokens(opts_.path);
  if (static_cast<int64_t>(tokens_.size()) < opts_.seq_len + 1) {
    throw std::runtime_error("TokenLoader: too few tokens for one chunk");
  }
  num_chunks_ = (static_cast<int64_t>(tokens_.size()) - 1) / opts_.seq_len;
  chunk_order_.resize(static_cast<size_t>(num_chunks_));
  for (int64_t i = 0; i < num_chunks_; ++i) chunk_order_[i] = i;
  if (opts_.shuffle) {
    std::mt19937_64 g(opts_.seed);
    std::shuffle(chunk_order_.begin(), chunk_order_.end(), g);
  }

  // Honor resume cursor. Clamped to [0, num_chunks_).
  int64_t c = opts_.start_cursor;
  if (c < 0) c = 0;
  if (c >= num_chunks_) c = 0;
  chunk_cursor_.store(c);

  // Allocate ring. Host buffers use plain malloc (not pinned on CPU-only
  // build); with CUDA we pin them for fast H2D.
  ring_.reserve(opts_.ring_size);
  for (int i = 0; i < opts_.ring_size; ++i) {
    auto s = std::make_unique<Slot>();
#ifdef USE_CUDA
    if (opts_.device.is_cuda()) {
      int64_t n = opts_.batch_size * opts_.seq_len;
      int64_t* hi = nullptr;
      int64_t* ht = nullptr;
      cudaHostAlloc(reinterpret_cast<void**>(&hi), n * sizeof(int64_t), cudaHostAllocDefault);
      cudaHostAlloc(reinterpret_cast<void**>(&ht), n * sizeof(int64_t), cudaHostAllocDefault);
      Shape sh{opts_.batch_size, opts_.seq_len};
      s->host_input  = Tensor(hi, sh, contiguous_strides(sh), DType::I64,
                              Device::cpu(), nullptr, n * sizeof(int64_t));
      s->host_target = Tensor(ht, sh, contiguous_strides(sh), DType::I64,
                              Device::cpu(), nullptr, n * sizeof(int64_t));
      s->dev_input   = empty(sh, DType::I64, opts_.device);
      s->dev_target  = empty(sh, DType::I64, opts_.device);
    } else
#endif
    {
      Shape sh{opts_.batch_size, opts_.seq_len};
      s->host_input  = empty(sh, DType::I64, Device::cpu());
      s->host_target = empty(sh, DType::I64, Device::cpu());
    }
    ring_.push_back(std::move(s));
  }
}

TokenLoader::~TokenLoader() {
  stop_.store(true);
  ring_cv_.notify_all();
  if (producer_thread_.joinable()) producer_thread_.join();
#ifdef USE_CUDA
  if (opts_.device.is_cuda()) {
    for (auto& s : ring_) {
      if (s->host_input.data())  cudaFreeHost(s->host_input.data());
      if (s->host_target.data()) cudaFreeHost(s->host_target.data());
    }
  }
#endif
}

void TokenLoader::start() {
  producer_thread_ = std::thread([this]() { producer_loop_(); });
}

int64_t TokenLoader::steps_per_epoch() const {
  return num_chunks_ / opts_.batch_size;
}

void TokenLoader::fill_slot_(int slot_idx) {
  auto& slot = *ring_[slot_idx];
  int64_t* hi = slot.host_input.as<int64_t>();
  int64_t* ht = slot.host_target.as<int64_t>();
  const int64_t seq = opts_.seq_len;
  for (int64_t b = 0; b < opts_.batch_size; ++b) {
    int64_t cur = chunk_cursor_.fetch_add(1, std::memory_order_relaxed);
    if (cur >= num_chunks_) {
      // Reshuffle and wrap around.
      // NOTE: this writes under no lock; with multiple producers we'd need one.
      // Single producer thread = safe.
      if (opts_.shuffle) {
        std::mt19937_64 g(opts_.seed + produce_cursor_.load());
        std::shuffle(chunk_order_.begin(), chunk_order_.end(), g);
      }
      chunk_cursor_.store(0, std::memory_order_relaxed);
      cur = chunk_cursor_.fetch_add(1, std::memory_order_relaxed);
    }
    int64_t start = chunk_order_[cur] * seq;
    std::memcpy(hi + b * seq, tokens_.data() + start,     seq * sizeof(int64_t));
    std::memcpy(ht + b * seq, tokens_.data() + start + 1, seq * sizeof(int64_t));
  }

#ifdef USE_CUDA
  if (opts_.device.is_cuda()) {
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(copy_stream(opts_.device).handle);
    cudaMemcpyAsync(slot.dev_input.data(),  hi,
                    slot.host_input.nbytes(),  cudaMemcpyHostToDevice, cs);
    cudaMemcpyAsync(slot.dev_target.data(), ht,
                    slot.host_target.nbytes(), cudaMemcpyHostToDevice, cs);
    // Don't synchronize — the consumer will make the compute stream wait on
    // the copy stream via an event.
  }
#endif
}

void TokenLoader::producer_loop_() {
  while (!stop_.load()) {
    int64_t pc = produce_cursor_.load();
    int slot_idx = static_cast<int>(pc % opts_.ring_size);
    auto& slot = *ring_[slot_idx];

    // Wait until this slot has been consumed.
    {
      std::unique_lock<std::mutex> lk(ring_mu_);
      ring_cv_.wait(lk, [&]() { return slot.consumed.load() || stop_.load(); });
      if (stop_.load()) return;
      slot.consumed.store(false);
    }

    fill_slot_(slot_idx);
    slot.ready.store(true);
    produce_cursor_.fetch_add(1);
    ring_cv_.notify_all();
  }
}

TokenLoader::Batch TokenLoader::next() {
  int64_t cc = consume_cursor_.load();
  int slot_idx = static_cast<int>(cc % opts_.ring_size);
  auto& slot = *ring_[slot_idx];

  {
    std::unique_lock<std::mutex> lk(ring_mu_);
    ring_cv_.wait(lk, [&]() { return slot.ready.load() || stop_.load(); });
    if (stop_.load()) return {};
  }

  Batch b;
#ifdef USE_CUDA
  if (opts_.device.is_cuda()) {
    // The consumer (compute stream) needs to wait until the copy stream is
    // done writing into dev_input/dev_target. Insert a copy-stream event +
    // have the compute stream wait on it.
    static thread_local cudaEvent_t copy_done = nullptr;
    if (!copy_done) cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming);
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(copy_stream(opts_.device).handle);
    cudaStream_t xs = reinterpret_cast<cudaStream_t>(compute_stream(opts_.device).handle);
    cudaEventRecord(copy_done, cs);
    cudaStreamWaitEvent(xs, copy_done, 0);

    b.input  = slot.dev_input.view(slot.dev_input.shape());
    b.target = slot.dev_target.view(slot.dev_target.shape());
  } else
#endif
  {
    b.input  = slot.host_input.view(slot.host_input.shape());
    b.target = slot.host_target.view(slot.host_target.shape());
  }

  slot.ready.store(false);
  slot.consumed.store(true);
  consume_cursor_.fetch_add(1);
  ring_cv_.notify_all();
  return b;
}

}  // namespace zwt::data
