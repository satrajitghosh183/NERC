#include "omni/store/trace_store.hpp"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace omni::store {

static void put_u32(std::vector<uint8_t>& o, uint32_t v) { for (int i=0;i<4;++i) o.push_back((uint8_t)(v>>(8*i))); }
static void put_u64(std::vector<uint8_t>& o, uint64_t v) { for (int i=0;i<8;++i) o.push_back((uint8_t)(v>>(8*i))); }
static uint32_t rd_u32(const uint8_t* p) { uint32_t v=0; for (int i=0;i<4;++i) v |= (uint32_t)p[i]<<(8*i); return v; }
static uint64_t rd_u64(const uint8_t* p) { uint64_t v=0; for (int i=0;i<8;++i) v |= (uint64_t)p[i]<<(8*i); return v; }

void TraceWriter::add(uint32_t site_id, const trace::Column& col) {
    entries_.push_back({site_id, (uint64_t)col.size(), trace::encode(col)});
}

bool TraceWriter::write(const std::string& path) const {
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), {'O','T','R','1'});
    put_u32(buf, (uint32_t)entries_.size());

    const uint64_t dir_off = buf.size();
    const uint64_t dir_bytes = (uint64_t)entries_.size() * 32; // 4+4+8+8+8
    uint64_t blob_cursor = dir_off + dir_bytes;

    // directory
    for (const auto& e : entries_) {
        put_u32(buf, e.site_id);
        put_u32(buf, 0);                 // pad
        put_u64(buf, blob_cursor);
        put_u64(buf, (uint64_t)e.blob.size());
        put_u64(buf, e.count);
        blob_cursor += e.blob.size();
    }
    // blobs
    for (const auto& e : entries_) buf.insert(buf.end(), e.blob.begin(), e.blob.end());

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    return w == buf.size();
}

bool TraceReader::open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;
    struct stat st;
    if (fstat(fd_, &st) != 0 || st.st_size < 8) { close(); return false; }
    map_len_ = (size_t)st.st_size;
    void* p = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) { close(); return false; }
    base_ = (const uint8_t*)p;

    if (std::memcmp(base_, "OTR1", 4) != 0) { close(); return false; }
    uint32_t n = rd_u32(base_ + 4);
    const uint8_t* d = base_ + 8;
    dir_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* e = d + (size_t)i * 32;
        dir_[i].site_id = rd_u32(e);
        dir_[i].offset  = rd_u64(e + 8);
        dir_[i].length  = rd_u64(e + 16);
        dir_[i].count   = rd_u64(e + 24);
        if (dir_[i].offset + dir_[i].length > map_len_) { close(); return false; }
    }
    return true;
}

void TraceReader::close() {
    if (base_) { munmap((void*)base_, map_len_); base_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    map_len_ = 0; dir_.clear();
}

bool TraceReader::column_by_index(size_t i, trace::Column& out, uint32_t* site_id) const {
    if (i >= dir_.size()) return false;
    const Dir& e = dir_[i];
    if (site_id) *site_id = e.site_id;
    out = trace::decode(base_ + e.offset, (size_t)e.length);
    return out.size() == e.count;
}

bool TraceReader::column_by_site(uint32_t site_id, trace::Column& out) const {
    for (size_t i = 0; i < dir_.size(); ++i)
        if (dir_[i].site_id == site_id) return column_by_index(i, out);
    return false;
}

} // namespace omni::store
