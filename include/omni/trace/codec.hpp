// omni/trace/codec.hpp — divergence-aware columnar trace codec (THE spearhead).
//
// A captured value-tap produces, for one program point, a *column*: one value per
// shader invocation (lane). In SIMT, lanes are grouped into warps of W; within a
// warp values are usually highly coherent (a smooth fragment value, a quad's shared
// derivative, a loop counter). We exploit that:
//
//   per warp:  base = min(active lane values)        [U32]   or  base = first active [F32-XOR]
//              residual_i = value_i - base           [U32]   or  value_i XOR base    [F32]
//              bit-pack residuals at width b = ceil(log2(max residual + 1))
//
// Inactive lanes (control-flow divergence) are skipped entirely — the mask is stored
// once per column. Perfect coherence -> residuals are 0 -> ~0 bits. The achieved
// bits/value provably approaches the residual entropy H(value | warp base); see
// warp_conditional_entropy_bits(). Lossless and round-trip exact.
#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>

namespace omni::trace {

enum class ValueKind : uint8_t { U32 = 0, F32 = 1 };

struct Column {
    std::vector<uint32_t> bits;   // value bit-patterns (uint value, or float bits)
    std::vector<uint8_t> active;  // 1 == lane active; empty == all active
    uint32_t warp_size = 32;
    ValueKind kind = ValueKind::U32;

    size_t size() const { return bits.size(); }
    bool lane_active(size_t i) const { return active.empty() || active[i]; }
    size_t active_count() const {
        if (active.empty()) return bits.size();
        size_t n = 0; for (uint8_t a : active) n += (a != 0); return n;
    }
};

// Encode/decode (lossless for active lanes).
std::vector<uint8_t> encode(const Column& c);
Column decode(const uint8_t* data, size_t n);

// ---- analysis / information-theoretic references ----------------------------
// Order-0 Shannon entropy (bits/value) of the raw value distribution — a proxy for
// what a generic entropy coder (gzip/zstd-class) could achieve on the column.
double order0_entropy_bits(const std::vector<uint32_t>& values);

// Residual entropy H(value | warp base) under our model — the target our bit-packing
// approaches. This is the quantity the codec is designed to hit.
double warp_conditional_entropy_bits(const Column& c);

} // namespace omni::trace
