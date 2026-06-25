// omni/trace/synth_columns.hpp — synthetic trace columns with controllable
// coherence, modelling real SIMT capture patterns (header-only; used by tests
// and the benchmark so both exercise identical data).
#pragma once
#include "omni/trace/codec.hpp"
#include <cmath>
#include <cstdint>

namespace omni::trace::synth {

// Small deterministic LCG (no <random> dependency; reproducible across platforms).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}
    uint32_t u32() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(s >> 32); }
    float unit() { return (u32() >> 8) * (1.0f / 16777216.0f); } // [0,1)
};

inline Column constant(size_t n, uint32_t v, uint32_t W = 32) {
    Column c; c.warp_size = W; c.kind = ValueKind::U32; c.bits.assign(n, v); return c;
}

inline Column random32(size_t n, uint64_t seed = 1, uint32_t W = 32) {
    Column c; c.warp_size = W; c.kind = ValueKind::U32; c.bits.resize(n);
    Rng r(seed); for (auto& x : c.bits) x = r.u32(); return c;
}

// Coherent gradient: within a warp lanes differ by a small step; warps drift slowly.
// Models a smoothly-varying integer value (e.g., a tiled coordinate or loop index).
inline Column warp_gradient(size_t n, uint32_t W = 32, uint32_t lane_step = 1,
                            uint32_t warp_step = 3, uint32_t noise = 2, uint64_t seed = 7) {
    Column c; c.warp_size = W; c.kind = ValueKind::U32; c.bits.resize(n);
    Rng r(seed);
    for (size_t i = 0; i < n; ++i) {
        uint32_t warp = (uint32_t)(i / W), lane = (uint32_t)(i % W);
        uint32_t nz = noise ? (r.u32() % (noise + 1)) : 0;
        c.bits[i] = 1000u + warp * warp_step + lane * lane_step + nz;
    }
    return c;
}

// Quad coherence: groups of 4 invocations share a value (fragment derivatives /
// 2x2 quads). Within a warp most residuals are zero.
inline Column quad_coherent(size_t n, uint32_t W = 32, uint64_t seed = 11) {
    Column c; c.warp_size = W; c.kind = ValueKind::U32; c.bits.resize(n);
    Rng r(seed); uint32_t cur = 500;
    for (size_t i = 0; i < n; ++i) {
        if (i % 4 == 0) cur = 500 + (r.u32() % 64);
        c.bits[i] = cur;
    }
    return c;
}

// Smoothly varying float value -> stored as float bits, XOR-coded (shared exponent/sign).
inline Column float_smooth(size_t n, uint32_t W = 32, double freq = 0.01, double amp = 8.0) {
    Column c; c.warp_size = W; c.kind = ValueKind::F32; c.bits.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float f = (float)(amp * std::sin(freq * (double)i) + amp);
        uint32_t b; std::memcpy(&b, &f, 4); c.bits[i] = b;
    }
    return c;
}

// Apply control-flow divergence: deactivate lanes with probability (1 - active_prob).
inline Column with_divergence(Column c, double active_prob, uint64_t seed = 99) {
    Rng r(seed);
    c.active.assign(c.bits.size(), 1);
    for (size_t i = 0; i < c.bits.size(); ++i) c.active[i] = (r.unit() < active_prob) ? 1 : 0;
    return c;
}

} // namespace omni::trace::synth
