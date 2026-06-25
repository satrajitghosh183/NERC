#include "omni/test.hpp"
#include "omni/analysis/diff.hpp"
#include "omni/trace/codec.hpp"
#include "omni/trace/synth_columns.hpp"
#include <cmath>
#include <cstring>
#include <vector>

using namespace omni;
namespace synth = omni::trace::synth;

static uint32_t fbits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

TEST(diff, ulp_distance_basics) {
    CHECK_EQ(analysis::ulp_distance(1.0f, 1.0f), 0u);
    CHECK_EQ(analysis::ulp_distance(1.0f, std::nextafter(1.0f, 2.0f)), 1u);
    CHECK_EQ(analysis::ulp_distance(1.0f, std::nextafter(std::nextafter(1.0f, 2.0f), 2.0f)), 2u);
    // crossing zero is symmetric and finite
    CHECK_EQ(analysis::ulp_distance(0.0f, -0.0f), 0u);
    CHECK(analysis::ulp_distance(-1.0f, 1.0f) > 0u);
}

TEST(diff, detects_injected_discrepancy) {
    trace::Column ref; ref.kind = trace::ValueKind::F32; ref.warp_size = 32;
    for (int i = 0; i < 64; ++i) ref.bits.push_back(fbits((float)i * 0.1f));
    trace::Column gpu = ref;
    // Perturb lane 5 by 3 ULPs.
    float v; std::memcpy(&v, &gpu.bits[5], 4);
    for (int k = 0; k < 3; ++k) v = std::nextafter(v, 1e30f);
    gpu.bits[5] = fbits(v);

    auto d = analysis::diff_float_columns(ref, gpu);
    CHECK_EQ(d.mismatches, 1ull);
    CHECK_EQ(d.max_ulp, 3u);
    CHECK_EQ(d.first_mismatch, 5);
    CHECK(d.max_rel > 0.0);
}

TEST(diff, identical_columns_zero) {
    auto c = synth::float_smooth(1000);
    auto d = analysis::diff_float_columns(c, c);
    CHECK_EQ(d.mismatches, 0ull);
    CHECK_EQ(d.max_ulp, 0u);
}

// Real debugging target: GPU fast-math (fused multiply-add, single rounding) vs a strict
// CPU reference (two roundings). The debugger must surface these tiny discrepancies.
TEST(diff, catches_fastmath_fma_vs_strict) {
    trace::Column strict; strict.kind = trace::ValueKind::F32; strict.warp_size = 32;
    trace::Column fast;   fast.kind   = trace::ValueKind::F32; fast.warp_size = 32;
    synth::Rng r(2024);
    const int N = 1 << 16;
    for (int i = 0; i < N; ++i) {
        float a = r.unit() * 4.0f - 2.0f, b = r.unit() * 4.0f - 2.0f, c = r.unit() * 4.0f - 2.0f;
        float s = a * b; s = s + c;          // strict: two roundings
        float f = std::fmaf(a, b, c);        // fused: one rounding
        strict.bits.push_back(fbits(s));
        fast.bits.push_back(fbits(f));
    }
    auto d = analysis::diff_float_columns(strict, fast);
    std::printf("    fma vs strict: %llu/%llu lanes differ, max_ulp=%u mean_ulp=%.4f max_rel=%.2e\n",
                (unsigned long long)d.mismatches, (unsigned long long)d.n, d.max_ulp, d.mean_ulp, d.max_rel);
    // Statistically certain to differ on some lanes; differences are small (a few ULP).
    CHECK(d.mismatches > 0ull);
    CHECK(d.max_ulp >= 1u);
}

TEST(diff, branch_divergence_metric) {
    // All lanes same class -> converged.
    trace::Column conv = synth::constant(64, 1);
    CHECK_NEAR(analysis::branch_divergence(conv), 0.0, 1e-12);

    // Half class 0, half class 1 within each warp -> divergence 0.5.
    trace::Column half; half.warp_size = 32; half.bits.resize(64);
    for (int i = 0; i < 64; ++i) half.bits[i] = (i % 32) < 16 ? 0u : 1u;
    CHECK_NEAR(analysis::branch_divergence(half), 0.5, 1e-9);

    // Every active lane a distinct class -> near-maximal divergence (1 - 1/32).
    trace::Column alld; alld.warp_size = 32; alld.bits.resize(32);
    for (uint32_t i = 0; i < 32; ++i) alld.bits[i] = i;
    CHECK_NEAR(analysis::branch_divergence(alld), 1.0 - 1.0 / 32.0, 1e-9);
}
