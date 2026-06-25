#include "omni/analysis/diff.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace omni::analysis {

uint32_t ulp_distance(float a, float b) {
    if (a == b) return 0;            // equal values incl. +0.0 == -0.0
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    // Map IEEE bits to a monotonic ordering so subtraction counts representable steps.
    auto mono = [](uint32_t u) -> uint32_t { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); };
    uint32_t ma = mono(ua), mb = mono(ub);
    return ma > mb ? ma - mb : mb - ma;
}

double relative_error(float a, float b) {
    double da = a, db = b;
    double denom = std::max({std::fabs(da), std::fabs(db), 1e-30});
    return std::fabs(da - db) / denom;
}

NumDiff diff_float_columns(const trace::Column& cpu, const trace::Column& gpu) {
    NumDiff d;
    size_t n = std::min(cpu.size(), gpu.size());
    double sum_ulp = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!cpu.lane_active(i) || !gpu.lane_active(i)) continue;
        float a, b;
        std::memcpy(&a, &cpu.bits[i], 4);
        std::memcpy(&b, &gpu.bits[i], 4);
        uint32_t u = ulp_distance(a, b);
        double r = relative_error(a, b);
        ++d.n;
        sum_ulp += u;
        d.max_ulp = std::max(d.max_ulp, u);
        d.max_rel = std::max(d.max_rel, r);
        if (u > 0) { ++d.mismatches; if (d.first_mismatch < 0) d.first_mismatch = (int64_t)i; }
    }
    d.mean_ulp = d.n ? sum_ulp / (double)d.n : 0.0;
    return d;
}

double branch_divergence(const trace::Column& cls) {
    const uint32_t W = cls.warp_size ? cls.warp_size : 1;
    double weighted = 0.0; double total_active = 0.0;
    std::unordered_map<uint32_t, uint32_t> hist;
    for (size_t w0 = 0; w0 < cls.size(); w0 += W) {
        size_t w1 = std::min(cls.size(), w0 + (size_t)W);
        hist.clear();
        uint32_t active = 0, majority = 0;
        for (size_t i = w0; i < w1; ++i) {
            if (!cls.lane_active(i)) continue;
            uint32_t c = ++hist[cls.bits[i]];
            majority = std::max(majority, c);
            ++active;
        }
        if (active == 0) continue;
        double div = 1.0 - (double)majority / (double)active;
        weighted += div * active;
        total_active += active;
    }
    return total_active > 0 ? weighted / total_active : 0.0;
}

} // namespace omni::analysis
