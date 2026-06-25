#include "omni/timetravel/timetravel.hpp"
#include <algorithm>

namespace omni::timetravel {

Timeline Timeline::reconstruct(const omni::store::TraceReader& store,
                               const std::vector<omni::capture::TapSite>& sites,
                               uint32_t invocation) {
    Timeline tl;
    // Program order == tap-site id order (taps were inserted in instruction order).
    std::vector<omni::capture::TapSite> ordered = sites;
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) { return a.id < b.id; });

    for (const auto& s : ordered) {
        omni::trace::Column col;
        if (!store.column_by_site(s.id, col)) continue;
        if (invocation >= col.size()) continue;
        if (!col.lane_active(invocation)) continue;   // inactive (diverged) lane: no value
        tl.steps_.push_back(Step{ s.id, s.line, col.bits[invocation], s.def_op });
    }
    tl.cursor_ = 0;
    return tl;
}

} // namespace omni::timetravel
