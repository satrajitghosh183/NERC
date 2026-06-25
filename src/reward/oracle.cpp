#include "omni/reward/oracle.hpp"
#include <cmath>
#include <algorithm>

namespace omni::reward {

Breakdown score(const Inputs& in, const Weights& w) {
    Breakdown b;
    b.compile = w.compile * (in.compiled ? 1.0 : 0.0);
    if (!in.compiled) { b.total = b.compile; return b; }  // gate: cannot run what won't build
    b.exec      = w.exec      * (in.executed ? 1.0 : 0.0);
    b.divergence= w.divergence* (1.0 - std::clamp(in.divergence, 0.0, 1.0));
    b.numerical = w.numerical * (1.0 - std::clamp(in.numerical_error, 0.0, 1.0));
    b.visual    = w.visual    * std::clamp(in.visual_match, 0.0, 1.0);
    b.perf      = -w.perf     * std::clamp(in.perf_penalty, 0.0, 1.0);
    b.total = b.compile + b.exec + b.divergence + b.numerical + b.visual + b.perf;
    return b;
}

double normalize_ulp(double mean_ulp, double scale) {
    if (mean_ulp <= 0.0) return 0.0;
    return 1.0 - std::exp(-mean_ulp / std::max(scale, 1e-9)); // saturating, monotone
}

double image_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = (double)a[i] - (double)b[i]; mse += d * d; }
    mse /= (double)n;
    return std::exp(-mse);     // identical -> 1.0, diverging -> 0
}

Blame attribute(const std::vector<double>& err, const std::vector<omni::capture::TapSite>& sites) {
    Blame blame;
    double best = -1.0; size_t bi = 0;
    for (size_t i = 0; i < err.size(); ++i) if (err[i] > best) { best = err[i]; bi = i; }
    if (best > 0.0 && bi < sites.size()) {
        blame.found = true; blame.severity = best;
        blame.site_id = sites[bi].id; blame.line = sites[bi].line;
    }
    return blame;
}

} // namespace omni::reward
