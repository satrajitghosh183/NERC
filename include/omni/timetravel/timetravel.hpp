// omni/timetravel/timetravel.hpp — reconstruct one invocation's stepped history.
//
// Given the columnar trace store (one column per tap site, indexed by invocation) plus
// the tap-site metadata, rebuild the ordered sequence of values a chosen invocation
// produced, and step forward/backward through it — the Metal-style "shader history",
// but reconstructed from our own compressed capture (PLAN.md §4.8).
#pragma once
#include "omni/store/trace_store.hpp"
#include "omni/capture/instrument.hpp"
#include <cstdint>
#include <vector>

namespace omni::timetravel {

struct Step {
    uint32_t site_id = 0;
    uint32_t line = 0;
    uint32_t value_bits = 0;
    omni::uir::Op def_op{};
    float as_f32() const { float f; __builtin_memcpy(&f, &value_bits, 4); return f; }
};

class Timeline {
public:
    // Reconstruct the history for `invocation` from the store + site metadata.
    static Timeline reconstruct(const omni::store::TraceReader& store,
                                const std::vector<omni::capture::TapSite>& sites,
                                uint32_t invocation);

    size_t size() const { return steps_.size(); }
    const Step& at(size_t i) const { return steps_[i]; }
    const Step& current() const { return steps_[cursor_]; }
    size_t cursor() const { return cursor_; }

    bool step_forward() { if (cursor_ + 1 < steps_.size()) { ++cursor_; return true; } return false; }
    bool step_back()    { if (cursor_ > 0) { --cursor_; return true; } return false; }
    void seek(size_t i) { cursor_ = i < steps_.size() ? i : (steps_.empty() ? 0 : steps_.size() - 1); }

private:
    std::vector<Step> steps_;
    size_t cursor_ = 0;
};

} // namespace omni::timetravel
