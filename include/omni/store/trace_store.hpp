// omni/store/trace_store.hpp — persistent columnar trace store.
//
// Many compressed trace columns (one per tap site) are packed into a single file with
// a directory; the reader memory-maps it and decodes any column on demand — the
// substrate the time-travel engine scrubs over (PLAN.md §4.6, §4.8).
//
// File layout (little-endian):
//   ['O','T','R','1'] [u32 num_columns]
//   directory[num_columns]: { u32 site_id, u32 _pad, u64 offset, u64 length, u64 count }
//   blobs: concatenated codec outputs (trace::encode)
#pragma once
#include "omni/trace/codec.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace omni::store {

class TraceWriter {
public:
    void add(uint32_t site_id, const trace::Column& col);   // encodes immediately
    bool write(const std::string& path) const;
    size_t size() const { return entries_.size(); }
private:
    struct Entry { uint32_t site_id; uint64_t count; std::vector<uint8_t> blob; };
    std::vector<Entry> entries_;
};

class TraceReader {
public:
    ~TraceReader() { close(); }
    bool open(const std::string& path);
    void close();
    size_t num_columns() const { return dir_.size(); }

    bool column_by_index(size_t i, trace::Column& out, uint32_t* site_id = nullptr) const;
    bool column_by_site(uint32_t site_id, trace::Column& out) const;

private:
    struct Dir { uint32_t site_id; uint64_t offset, length, count; };
    const uint8_t* base_ = nullptr;
    size_t map_len_ = 0;
    int fd_ = -1;
    std::vector<Dir> dir_;
};

} // namespace omni::store
