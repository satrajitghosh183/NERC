// omni/analysis/diff.hpp — quantified GPU-vs-CPU numerical diff and SIMT divergence.
//
// These are the dense, principled signals the reward oracle consumes (PLAN.md §11):
//   numerical_error  <- ULP / relative-error between captured GPU values and the CPU reference
//   divergence       <- per-warp control-flow divergence from per-lane branch classes
// Both are honest measurements, not heuristics.
#pragma once
#include "omni/trace/codec.hpp"
#include <cstdint>
#include <vector>

namespace omni::analysis {

// Units in the last place between two IEEE-754 floats (monotonic-key method).
// 0 == bit-identical; small == benign rounding; large == a real discrepancy.
uint32_t ulp_distance(float a, float b);

// Relative error |a-b| / max(|a|,|b|,tiny).
double relative_error(float a, float b);

struct NumDiff {
    uint64_t n = 0;
    uint64_t mismatches = 0;     // ulp > 0
    uint32_t max_ulp = 0;
    double mean_ulp = 0.0;
    double max_rel = 0.0;
    int64_t first_mismatch = -1; // lane index, or -1
};

// Compare a CPU-reference float column against a captured GPU float column.
NumDiff diff_float_columns(const trace::Column& cpu_ref, const trace::Column& gpu);

// Per-warp control-flow divergence from a column whose value per lane is a "class"
// (e.g., the taken branch target / path id). Returns the active-lane-weighted mean of
// (1 - majority_fraction): 0.0 == fully converged, →1.0 == maximally divergent.
double branch_divergence(const trace::Column& classes);

} // namespace omni::analysis
