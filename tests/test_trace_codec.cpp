#include "omni/test.hpp"
#include "omni/trace/codec.hpp"
#include "omni/trace/synth_columns.hpp"
#include <vector>

using namespace omni::trace;

// Verify decode(encode(c)) reproduces every ACTIVE lane exactly (lossless).
static void check_roundtrip(const Column& c) {
    auto bytes = encode(c);
    Column d = decode(bytes.data(), bytes.size());
    REQUIRE_EQ(d.size(), c.size());
    REQUIRE_EQ(d.warp_size, c.warp_size);
    for (size_t i = 0; i < c.size(); ++i) {
        if (!c.lane_active(i)) continue;
        if (d.bits[i] != c.bits[i]) {
            std::printf("    mismatch at %zu: got %u want %u\n", i, d.bits[i], c.bits[i]);
            REQUIRE(false);
        }
    }
}

TEST(codec, roundtrip_constant)      { check_roundtrip(synth::constant(1000, 42)); }
TEST(codec, roundtrip_random)        { check_roundtrip(synth::random32(1000, 5)); }
TEST(codec, roundtrip_gradient)      { check_roundtrip(synth::warp_gradient(1000)); }
TEST(codec, roundtrip_quad)          { check_roundtrip(synth::quad_coherent(1000)); }
TEST(codec, roundtrip_float_smooth)  { check_roundtrip(synth::float_smooth(1000)); }

TEST(codec, roundtrip_with_divergence) {
    check_roundtrip(synth::with_divergence(synth::warp_gradient(1000), 0.5, 3));
    check_roundtrip(synth::with_divergence(synth::random32(1000), 0.1, 4));
}

TEST(codec, edge_cases) {
    check_roundtrip(synth::constant(0, 0));      // empty
    check_roundtrip(synth::constant(1, 7));       // single
    check_roundtrip(synth::warp_gradient(33));    // non-multiple of warp size
    check_roundtrip(synth::warp_gradient(31));    // partial single warp
    // all lanes inactive
    Column alloff = synth::warp_gradient(64);
    alloff.active.assign(64, 0);
    check_roundtrip(alloff);
}

TEST(codec, compression_beats_raw_on_coherent) {
    Column c = synth::quad_coherent(1u << 16);
    auto bytes = encode(c);
    double achieved_bpv = (double)bytes.size() * 8.0 / (double)c.active_count();
    std::printf("    quad_coherent achieved %.3f bits/value (raw=32)\n", achieved_bpv);
    CHECK(achieved_bpv < 32.0);   // must beat raw
    CHECK(achieved_bpv < 12.0);   // coherent data should compress hard
}

TEST(codec, residual_entropy_below_order0_on_coherent) {
    Column c = synth::warp_gradient(1u << 16);
    double h_res = warp_conditional_entropy_bits(c);
    double h0 = order0_entropy_bits(c.bits);
    std::printf("    gradient: H(res)=%.3f  H0=%.3f bits/value\n", h_res, h0);
    // Conditioning on the warp base removes information -> residual entropy is lower.
    CHECK(h_res <= h0 + 1e-9);
}

TEST(codec, random_is_near_raw) {
    Column c = synth::random32(1u << 16, 123);
    auto bytes = encode(c);
    double achieved_bpv = (double)bytes.size() * 8.0 / (double)c.active_count();
    std::printf("    random achieved %.3f bits/value\n", achieved_bpv);
    // Incompressible: should be close to 32 (plus small per-warp overhead), never far above.
    CHECK(achieved_bpv > 30.0);
    CHECK(achieved_bpv < 36.0);
}
