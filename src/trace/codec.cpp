#include "omni/trace/codec.hpp"
#include <cmath>
#include <unordered_map>
#include <algorithm>

namespace omni::trace {

// ---- LSB-first bit I/O ------------------------------------------------------
namespace {
struct BitWriter {
    std::vector<uint8_t>& out;
    uint64_t acc = 0; int nbits = 0;
    explicit BitWriter(std::vector<uint8_t>& o) : out(o) {}
    void put(uint32_t v, int bits) {                 // bits in [0,32]
        if (bits == 0) return;
        uint32_t mask = (bits >= 32) ? 0xffffffffu : ((1u << bits) - 1u);
        acc |= (uint64_t)(v & mask) << nbits; nbits += bits;
        while (nbits >= 8) { out.push_back((uint8_t)(acc & 0xff)); acc >>= 8; nbits -= 8; }
    }
    void flush() { if (nbits > 0) { out.push_back((uint8_t)(acc & 0xff)); acc = 0; nbits = 0; } }
};
struct BitReader {
    const uint8_t* p; size_t n, pos = 0;
    uint64_t acc = 0; int nbits = 0;
    BitReader(const uint8_t* p_, size_t n_) : p(p_), n(n_) {}
    uint32_t get(int bits) {                          // bits in [0,32]
        if (bits == 0) return 0;
        while (nbits < bits) { acc |= (uint64_t)(pos < n ? p[pos++] : 0) << nbits; nbits += 8; }
        uint32_t mask = (bits >= 32) ? 0xffffffffu : ((1u << bits) - 1u);
        uint32_t v = (uint32_t)(acc & mask); acc >>= bits; nbits -= bits; return v;
    }
};
// LEB128 varint for byte-aligned header fields.
void put_varint(std::vector<uint8_t>& o, uint64_t v) {
    while (v >= 0x80) { o.push_back((uint8_t)(v | 0x80)); v >>= 7; } o.push_back((uint8_t)v);
}
uint64_t get_varint(const uint8_t* p, size_t n, size_t& i) {
    uint64_t v = 0; int s = 0;
    while (i < n) { uint8_t b = p[i++]; v |= (uint64_t)(b & 0x7f) << s; if (!(b & 0x80)) break; s += 7; }
    return v;
}
int bit_width(uint32_t x) { int b = 0; while (x) { ++b; x >>= 1; } return b; }
} // namespace

static constexpr uint8_t MAGIC = 0xC0;

std::vector<uint8_t> encode(const Column& c) {
    std::vector<uint8_t> out;
    out.push_back(MAGIC);
    out.push_back((uint8_t)c.kind);
    put_varint(out, c.warp_size);
    put_varint(out, c.size());

    // Active mask: 1 flag byte + (if not all-active) a bitmask.
    bool all_active = c.active.empty();
    out.push_back(all_active ? 1 : 0);
    if (!all_active) {
        BitWriter mw(out);
        for (size_t i = 0; i < c.size(); ++i) mw.put(c.lane_active(i) ? 1 : 0, 1);
        mw.flush();
    }

    const uint32_t W = c.warp_size ? c.warp_size : 1;
    const bool xorm = (c.kind == ValueKind::F32);

    BitWriter bw(out);
    for (size_t w0 = 0; w0 < c.size(); w0 += W) {
        size_t w1 = std::min(c.size(), w0 + (size_t)W);
        // base over active lanes in this warp.
        bool have = false; uint32_t base = 0;
        for (size_t i = w0; i < w1; ++i) {
            if (!c.lane_active(i)) continue;
            uint32_t v = c.bits[i];
            if (!have) { base = v; have = true; }
            else if (!xorm && v < base) base = v;     // U32: min for non-negative residuals
        }
        // residuals + width.
        uint32_t maxr = 0;
        std::vector<uint32_t> r; r.reserve(W);
        for (size_t i = w0; i < w1; ++i) {
            if (!c.lane_active(i)) continue;
            uint32_t v = c.bits[i];
            uint32_t res = xorm ? (v ^ base) : (v - base);
            r.push_back(res); maxr = std::max(maxr, res);
        }
        int b = bit_width(maxr);                       // 0 when all residuals are zero
        // Hybrid: pick the cheaper of {base+residual} vs {raw}. The 1-bit mode flag
        // bounds the codec at raw + ~1 bit/warp, so incompressible data never blows up.
        size_t cost_packed = 1 + 32 + 6 + r.size() * (size_t)b;
        size_t cost_raw    = 1 + r.size() * 32;
        if (cost_raw < cost_packed) {
            bw.put(1, 1);                              // mode 1 = raw
            for (size_t i = w0; i < w1; ++i) {
                if (!c.lane_active(i)) continue;
                bw.put(c.bits[i], 32);
            }
        } else {
            bw.put(0, 1);                              // mode 0 = base + residual
            bw.put(base, 32);
            bw.put((uint32_t)b, 6);                    // width 0..32 fits in 6 bits
            for (uint32_t res : r) bw.put(res, b);
        }
    }
    bw.flush();
    return out;
}

Column decode(const uint8_t* data, size_t n) {
    Column c;
    if (n < 2 || data[0] != MAGIC) return c;
    size_t i = 1;
    c.kind = (ValueKind)data[i++];
    c.warp_size = (uint32_t)get_varint(data, n, i);
    size_t count = (size_t)get_varint(data, n, i);
    c.bits.assign(count, 0);
    bool all_active = data[i++] != 0;
    if (!all_active) {
        c.active.assign(count, 0);
        BitReader mr(data + i, n - i);
        for (size_t k = 0; k < count; ++k) c.active[k] = (uint8_t)mr.get(1);
        i += (count + 7) / 8;                          // mask consumed this many bytes
    }
    const uint32_t W = c.warp_size ? c.warp_size : 1;
    const bool xorm = (c.kind == ValueKind::F32);
    BitReader br(data + i, n - i);
    for (size_t w0 = 0; w0 < count; w0 += W) {
        size_t w1 = std::min(count, w0 + (size_t)W);
        uint32_t mode = br.get(1);
        if (mode == 1) {                               // raw
            for (size_t k = w0; k < w1; ++k) {
                if (!all_active && !c.active[k]) continue;
                c.bits[k] = br.get(32);
            }
        } else {                                       // base + residual
            uint32_t base = br.get(32);
            int b = (int)br.get(6);
            for (size_t k = w0; k < w1; ++k) {
                if (!all_active && !c.active[k]) continue;
                uint32_t res = br.get(b);
                c.bits[k] = xorm ? (res ^ base) : (res + base);
            }
        }
    }
    return c;
}

// ---- entropy references -----------------------------------------------------
double order0_entropy_bits(const std::vector<uint32_t>& v) {
    if (v.empty()) return 0.0;
    std::unordered_map<uint32_t, uint32_t> hist;
    hist.reserve(v.size() * 2);
    for (uint32_t x : v) ++hist[x];
    const double N = (double)v.size();
    double H = 0.0;
    for (auto& [val, cnt] : hist) { double p = cnt / N; H -= p * std::log2(p); }
    return H;
}

double warp_conditional_entropy_bits(const Column& c) {
    // Residuals under the per-warp base model; their order-0 entropy is the target.
    const uint32_t W = c.warp_size ? c.warp_size : 1;
    const bool xorm = (c.kind == ValueKind::F32);
    std::vector<uint32_t> residuals;
    residuals.reserve(c.active_count());
    for (size_t w0 = 0; w0 < c.size(); w0 += W) {
        size_t w1 = std::min(c.size(), w0 + (size_t)W);
        bool have = false; uint32_t base = 0;
        for (size_t i = w0; i < w1; ++i) {
            if (!c.lane_active(i)) continue;
            uint32_t v = c.bits[i];
            if (!have) { base = v; have = true; }
            else if (!xorm && v < base) base = v;
        }
        for (size_t i = w0; i < w1; ++i) {
            if (!c.lane_active(i)) continue;
            uint32_t v = c.bits[i];
            residuals.push_back(xorm ? (v ^ base) : (v - base));
        }
    }
    return order0_entropy_bits(residuals);
}

} // namespace omni::trace
