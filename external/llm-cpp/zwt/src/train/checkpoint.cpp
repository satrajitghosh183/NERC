#include "zwt/train/checkpoint.hpp"
#include "zwt/core/stream.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace zwt::train {

namespace {

constexpr char     kMagic[8]     = {'Z','W','T','C','K','P','T','1'};
constexpr uint32_t kVersion      = 1;
constexpr size_t   kHeaderBytes  = 64;
constexpr size_t   kTRecBytes    = 80;
constexpr size_t   kStreamBuf    = 2 << 20;  // 2 MiB ofstream buffer

#pragma pack(push, 1)
struct FileHeader {
  char     magic[8];
  uint32_t version;
  uint32_t flags;
  int64_t  step;
  int64_t  data_cursor;
  uint64_t seed;
  int32_t  n_records;
  uint32_t pad0;
  float    lr;
  float    loss;
  uint64_t reserved;
};
static_assert(sizeof(FileHeader) == kHeaderBytes, "FileHeader size");

struct TRec {
  uint32_t name_len;
  uint8_t  dtype;
  uint8_t  rank;
  uint16_t pad0;
  uint64_t nbytes;
  int64_t  dims[6];
  uint64_t reserved0;
  uint64_t reserved1;
};
static_assert(sizeof(TRec) == kTRecBytes, "TRec size");
#pragma pack(pop)

inline size_t pad_to_8(size_t n) { return (n + 7u) & ~size_t{7}; }

void write_bytes(std::ofstream& f, const void* p, size_t n) {
  if (!f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n))) {
    throw std::runtime_error("checkpoint: write failed");
  }
}

void read_bytes(std::ifstream& f, void* p, size_t n) {
  if (!f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n))) {
    throw std::runtime_error("checkpoint: read failed or truncated");
  }
}

void write_pad_to_8(std::ofstream& f, size_t bytes_written) {
  size_t rem = pad_to_8(bytes_written) - bytes_written;
  if (rem == 0) return;
  char zeros[8] = {};
  write_bytes(f, zeros, rem);
}

void skip_pad_to_8(std::ifstream& f, size_t bytes_read) {
  size_t rem = pad_to_8(bytes_read) - bytes_read;
  if (rem == 0) return;
  f.seekg(static_cast<std::streamoff>(rem), std::ios::cur);
}

void fill_record(TRec& r, const Tensor& t, size_t name_len) {
  r.name_len = static_cast<uint32_t>(name_len);
  r.dtype    = static_cast<uint8_t>(t.dtype());
  r.rank     = static_cast<uint8_t>(t.rank());
  r.pad0     = 0;
  r.nbytes   = t.nbytes();
  for (int i = 0; i < 6; ++i) {
    r.dims[i] = (i < t.rank()) ? t.dim(i) : 0;
  }
  r.reserved0 = 0;
  r.reserved1 = 0;
}

void validate_record(const std::string& name, const TRec& r, const Tensor& t) {
  if (r.dtype != static_cast<uint8_t>(t.dtype())) {
    throw std::runtime_error("checkpoint: dtype mismatch for '" + name + "'");
  }
  if (r.rank != static_cast<uint8_t>(t.rank())) {
    throw std::runtime_error("checkpoint: rank mismatch for '" + name + "'");
  }
  for (int i = 0; i < t.rank(); ++i) {
    if (r.dims[i] != t.dim(i)) {
      throw std::runtime_error("checkpoint: shape mismatch for '" + name + "'");
    }
  }
  if (r.nbytes != t.nbytes()) {
    throw std::runtime_error("checkpoint: nbytes mismatch for '" + name + "'");
  }
}

// Pinned host staging. Two buffers are used as a ping-pong so the D2H (or H2D)
// on one can overlap the file write (or file read) on the other. On a CPU-only
// build the buffers are plain malloc'd and the overlap is a no-op — we serve
// tensor bytes straight from the host pointer.
struct Pinned {
  void*  buf[2]  = {nullptr, nullptr};
  size_t cap     = 0;
#ifdef USE_CUDA
  cudaEvent_t evt[2] = {};
  bool        have_events = false;
#endif

  void ensure(size_t n) {
    if (n <= cap) return;
    free_all();
#ifdef USE_CUDA
    cudaMallocHost(&buf[0], n);
    cudaMallocHost(&buf[1], n);
#else
    buf[0] = std::malloc(n);
    buf[1] = std::malloc(n);
#endif
    cap = n;
  }

  void free_all() {
    for (int i = 0; i < 2; ++i) {
      if (!buf[i]) continue;
#ifdef USE_CUDA
      cudaFreeHost(buf[i]);
#else
      std::free(buf[i]);
#endif
      buf[i] = nullptr;
    }
    cap = 0;
  }

  ~Pinned() {
#ifdef USE_CUDA
    if (have_events) {
      cudaEventDestroy(evt[0]);
      cudaEventDestroy(evt[1]);
    }
#endif
    free_all();
  }
};

}  // namespace

void save_checkpoint(const std::string& path,
                     const std::vector<Parameter*>& params,
                     optim::AdamW& opt,
                     const CheckpointMeta& meta) {
  if (params.size() != opt.n_params()) {
    throw std::runtime_error("checkpoint: param count / optimizer size mismatch");
  }

  // Flat list of (name, tensor*) in write order: value, .m, .v per param.
  struct Rec { std::string name; Tensor* t; };
  std::vector<Rec> recs;
  recs.reserve(params.size() * 3);
  for (size_t i = 0; i < params.size(); ++i) {
    recs.push_back({params[i]->name,        &params[i]->value});
    recs.push_back({params[i]->name + ".m", &opt.moment_m(i)});
    recs.push_back({params[i]->name + ".v", &opt.moment_v(i)});
  }
  const size_t n = recs.size();

  size_t max_bytes = 0;
  bool   any_cuda  = false;
  for (const auto& r : recs) {
    if (r.t->nbytes() > max_bytes) max_bytes = r.t->nbytes();
    if (r.t->device().is_cuda())   any_cuda  = true;
  }

  const std::string tmp = path + ".tmp";
  std::ofstream f;
  std::vector<char> file_buf(kStreamBuf);
  f.rdbuf()->pubsetbuf(file_buf.data(), file_buf.size());
  f.open(tmp, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("checkpoint: cannot open " + tmp);

  FileHeader h{};
  std::memcpy(h.magic, kMagic, 8);
  h.version     = kVersion;
  h.step        = meta.step;
  h.data_cursor = meta.data_cursor;
  h.seed        = meta.seed;
  h.n_records   = static_cast<int32_t>(n);
  h.lr          = meta.lr;
  h.loss        = meta.loss;
  write_bytes(f, &h, sizeof(h));

  Pinned pin;
#ifdef USE_CUDA
  cudaStream_t cs = nullptr;
  if (any_cuda) {
    pin.ensure(max_bytes);
    cudaEventCreateWithFlags(&pin.evt[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&pin.evt[1], cudaEventDisableTiming);
    pin.have_events = true;
    cs = reinterpret_cast<cudaStream_t>(
        compute_stream(recs.front().t->device()).handle);
  }
#else
  (void)any_cuda;
#endif

  auto kick_d2h = [&](size_t i) {
    if (i >= n) return;
#ifdef USE_CUDA
    if (recs[i].t->device().is_cuda()) {
      cudaMemcpyAsync(pin.buf[i & 1], recs[i].t->data(),
                      recs[i].t->nbytes(),
                      cudaMemcpyDeviceToHost, cs);
      cudaEventRecord(pin.evt[i & 1], cs);
    }
#else
    (void)i;
#endif
  };

  auto ptr_for = [&](size_t i) -> const void* {
#ifdef USE_CUDA
    if (recs[i].t->device().is_cuda()) return pin.buf[i & 1];
#endif
    return recs[i].t->data();
  };

  auto wait_d2h = [&](size_t i) {
#ifdef USE_CUDA
    if (recs[i].t->device().is_cuda()) {
      cudaEventSynchronize(pin.evt[i & 1]);
    }
#else
    (void)i;
#endif
  };

  if (n > 0) kick_d2h(0);
  for (size_t i = 0; i < n; ++i) {
    if (i + 1 < n) kick_d2h(i + 1);

    const auto& r = recs[i];
    TRec hdr;
    fill_record(hdr, *r.t, r.name.size());
    write_bytes(f, &hdr, sizeof(hdr));
    write_bytes(f, r.name.data(), r.name.size());
    write_pad_to_8(f, r.name.size());

    wait_d2h(i);
    write_bytes(f, ptr_for(i), r.t->nbytes());
    write_pad_to_8(f, r.t->nbytes());
  }

  f.close();
  if (!f) throw std::runtime_error("checkpoint: error closing " + tmp);

  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    throw std::runtime_error("checkpoint: rename " + tmp + " -> " + path + " failed");
  }
}

CheckpointMeta read_checkpoint_meta(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("checkpoint: cannot open " + path);
  FileHeader h{};
  read_bytes(f, &h, sizeof(h));
  if (std::memcmp(h.magic, kMagic, 8) != 0) {
    throw std::runtime_error("checkpoint: bad magic in " + path);
  }
  if (h.version != kVersion) {
    throw std::runtime_error("checkpoint: unsupported version");
  }
  CheckpointMeta m;
  m.step        = h.step;
  m.seed        = h.seed;
  m.data_cursor = h.data_cursor;
  m.lr          = h.lr;
  m.loss        = h.loss;
  return m;
}

CheckpointMeta load_checkpoint(const std::string& path,
                               const std::vector<Parameter*>& params,
                               optim::AdamW& opt) {
  if (params.size() != opt.n_params()) {
    throw std::runtime_error("checkpoint: param count / optimizer size mismatch");
  }

  std::ifstream f;
  std::vector<char> file_buf(kStreamBuf);
  f.rdbuf()->pubsetbuf(file_buf.data(), file_buf.size());
  f.open(path, std::ios::binary);
  if (!f) throw std::runtime_error("checkpoint: cannot open " + path);

  FileHeader h{};
  read_bytes(f, &h, sizeof(h));
  if (std::memcmp(h.magic, kMagic, 8) != 0) {
    throw std::runtime_error("checkpoint: bad magic in " + path);
  }
  if (h.version != kVersion) {
    throw std::runtime_error("checkpoint: unsupported version");
  }

  // Name -> (param index, kind). kind 0 = value, 1 = moment_m, 2 = moment_v.
  struct Slot { size_t idx; int kind; };
  std::unordered_map<std::string, Slot> table;
  table.reserve(params.size() * 3);
  size_t max_bytes = 0;
  bool   any_cuda  = false;
  for (size_t i = 0; i < params.size(); ++i) {
    table[params[i]->name]         = {i, 0};
    table[params[i]->name + ".m"]  = {i, 1};
    table[params[i]->name + ".v"]  = {i, 2};
    auto consider = [&](const Tensor& t) {
      if (t.nbytes() > max_bytes) max_bytes = t.nbytes();
      if (t.device().is_cuda())   any_cuda  = true;
    };
    consider(params[i]->value);
    consider(opt.moment_m(i));
    consider(opt.moment_v(i));
  }

  int32_t n_records_expected = static_cast<int32_t>(params.size() * 3);
  if (h.n_records != n_records_expected) {
    throw std::runtime_error("checkpoint: record count mismatch (file "
                             + std::to_string(h.n_records) + ", expected "
                             + std::to_string(n_records_expected) + ")");
  }

  Pinned pin;
#ifdef USE_CUDA
  cudaStream_t cs = nullptr;
  if (any_cuda) {
    pin.ensure(max_bytes);
    cudaEventCreateWithFlags(&pin.evt[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&pin.evt[1], cudaEventDisableTiming);
    pin.have_events = true;
    cs = reinterpret_cast<cudaStream_t>(
        compute_stream(params.front()->value.device()).handle);
  }
#else
  (void)any_cuda;
#endif

  // Track in-flight H2D so we can wait on the opposite slot before reusing.
#ifdef USE_CUDA
  bool   inflight[2] = {false, false};
  size_t cuda_idx = 0;  // counts only cuda tensors, for ping-pong
#endif
  auto wait_slot = [&](int slot) {
#ifdef USE_CUDA
    if (inflight[slot]) {
      cudaEventSynchronize(pin.evt[slot]);
      inflight[slot] = false;
    }
#else
    (void)slot;
#endif
  };

  std::vector<char> scratch;  // fallback for CPU tensors / unknown records
  std::vector<char> seen(params.size() * 3, 0);

  for (int32_t r = 0; r < h.n_records; ++r) {
    TRec rec;
    read_bytes(f, &rec, sizeof(rec));

    std::string name(rec.name_len, '\0');
    read_bytes(f, name.data(), rec.name_len);
    skip_pad_to_8(f, rec.name_len);

    auto it = table.find(name);
    if (it == table.end()) {
      f.seekg(static_cast<std::streamoff>(pad_to_8(rec.nbytes)), std::ios::cur);
      continue;
    }

    Tensor* target = nullptr;
    switch (it->second.kind) {
      case 0: target = &params[it->second.idx]->value; break;
      case 1: target = &opt.moment_m(it->second.idx); break;
      case 2: target = &opt.moment_v(it->second.idx); break;
    }
    validate_record(name, rec, *target);

#ifdef USE_CUDA
    if (target->device().is_cuda()) {
      int slot = static_cast<int>(cuda_idx & 1);
      wait_slot(slot);
      read_bytes(f, pin.buf[slot], rec.nbytes);
      skip_pad_to_8(f, rec.nbytes);
      cudaMemcpyAsync(target->data(), pin.buf[slot], rec.nbytes,
                      cudaMemcpyHostToDevice, cs);
      cudaEventRecord(pin.evt[slot], cs);
      inflight[slot] = true;
      ++cuda_idx;
    } else
#endif
    {
      if (scratch.size() < rec.nbytes) scratch.resize(rec.nbytes);
      read_bytes(f, scratch.data(), rec.nbytes);
      skip_pad_to_8(f, rec.nbytes);
      std::memcpy(target->data(), scratch.data(), rec.nbytes);
    }

    size_t slot_idx = it->second.idx * 3 + static_cast<size_t>(it->second.kind);
    seen[slot_idx] = 1;
  }

  // Drain remaining H2Ds before returning.
  wait_slot(0);
  wait_slot(1);

  for (size_t i = 0; i < seen.size(); ++i) {
    if (!seen[i]) {
      size_t param_idx = i / 3;
      int    kind      = static_cast<int>(i % 3);
      const char* suffix = (kind == 0) ? "" : (kind == 1 ? ".m" : ".v");
      throw std::runtime_error("checkpoint: missing record for '"
                               + params[param_idx]->name + suffix + "'");
    }
  }

  opt.set_step_count(h.step);

  CheckpointMeta m;
  m.step        = h.step;
  m.seed        = h.seed;
  m.data_cursor = h.data_cursor;
  m.lr          = h.lr;
  m.loss        = h.loss;
  return m;
}

}  // namespace zwt::train
