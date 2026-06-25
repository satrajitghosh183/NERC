// omni/reward/oracle.hpp — the debugger-in-the-loop reward oracle (PLAN.md §11).
//
// Turns a candidate shader's *captured execution* into a dense, decomposed, per-line
// reward — the signal that makes debugger feedback beat compile-pass feedback for
// shader synthesis. Every term is a real measurement from the debugger pipeline:
//   compile / exec   <- did it build and run without NaN/Inf
//   divergence       <- analysis::branch_divergence
//   numerical_error  <- normalized analysis::NumDiff (ULP) vs the CPU reference
//   visual_match     <- image similarity to the target render
// plus per-site credit assignment that blames the responsible tap site / source line.
#pragma once
#include "omni/capture/instrument.hpp"
#include <cstdint>
#include <vector>

namespace omni::reward {

struct Weights {
    double compile = 1.0, exec = 1.0, divergence = 0.5,
           numerical = 1.0, visual = 2.0, perf = 0.2;
};

struct Inputs {
    bool compiled = false;
    bool executed = false;       // ran with no NaN/Inf
    double divergence = 0.0;     // [0,1]  (0 = converged)
    double numerical_error = 0.0;// [0,1]  (0 = bit-exact vs reference)
    double visual_match = 0.0;   // [0,1]  (1 = matches target)
    double perf_penalty = 0.0;   // [0,1]  (0 = free)
};

struct Breakdown {
    double total = 0.0;
    double compile = 0, exec = 0, divergence = 0, numerical = 0, visual = 0, perf = 0;
};

// Compose the reward. If !compiled, downstream terms are gated to 0 (you cannot run a
// shader that does not build) — only the compile term contributes.
Breakdown score(const Inputs& in, const Weights& w = {});

// Normalizers (measurement -> [0,1]).
double normalize_ulp(double mean_ulp, double scale = 4.0);          // 0 -> 0, large -> ~1
double image_similarity(const std::vector<float>& a, const std::vector<float>& b); // MSE -> (0,1]

// Per-line credit assignment: blame the tap site with the largest error.
struct Blame { uint32_t site_id = 0; uint32_t line = 0; double severity = 0.0; bool found = false; };
Blame attribute(const std::vector<double>& per_site_error,
                const std::vector<omni::capture::TapSite>& sites);

} // namespace omni::reward
