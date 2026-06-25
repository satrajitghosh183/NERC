#include "omni/test.hpp"
#include "omni/reward/oracle.hpp"
#include "omni/capture/instrument.hpp"
#include <vector>

using namespace omni;

TEST(reward, perfect_beats_broken_and_gates_on_compile) {
    reward::Weights w;
    reward::Inputs perfect{true, true, 0.0, 0.0, 1.0, 0.0};
    reward::Inputs broken{false, false, 0.9, 1.0, 0.0, 1.0};
    auto p = reward::score(perfect, w);
    auto b = reward::score(broken, w);
    CHECK(p.total > b.total);
    // Non-compiling shader scores exactly the (zero) compile term, nothing downstream.
    CHECK_EQ(b.total, 0.0);
    CHECK_EQ(b.exec, 0.0);
    // Perfect shader earns compile+exec+div+num+vis.
    CHECK_NEAR(p.total, w.compile + w.exec + w.divergence + w.numerical + w.visual, 1e-9);
}

TEST(reward, monotonic_in_signals) {
    reward::Weights w;
    auto lo_vis = reward::score({true, true, 0.0, 0.0, 0.2, 0.0}, w).total;
    auto hi_vis = reward::score({true, true, 0.0, 0.0, 0.9, 0.0}, w).total;
    CHECK(hi_vis > lo_vis);                        // more visual match -> more reward
    auto lo_div = reward::score({true, true, 0.1, 0.0, 1.0, 0.0}, w).total;
    auto hi_div = reward::score({true, true, 0.8, 0.0, 1.0, 0.0}, w).total;
    CHECK(lo_div > hi_div);                        // more divergence -> less reward
}

TEST(reward, normalizers) {
    CHECK_EQ(reward::normalize_ulp(0.0), 0.0);
    CHECK(reward::normalize_ulp(1.0) > 0.0);
    CHECK(reward::normalize_ulp(100.0) > reward::normalize_ulp(1.0));
    CHECK(reward::normalize_ulp(1e9) <= 1.0);   // saturates to 1.0 for huge errors

    std::vector<float> a = {1, 2, 3, 4}, b = {1, 2, 3, 4}, c = {2, 3, 4, 5};
    CHECK_NEAR(reward::image_similarity(a, b), 1.0, 1e-9);
    CHECK(reward::image_similarity(a, c) < 1.0);
}

TEST(reward, credit_assignment_blames_worst_site) {
    std::vector<capture::TapSite> sites = {
        {0, 100, uir::Op::FMul, 5},
        {1, 101, uir::Op::FAdd, 6},
        {2, 102, uir::Op::VectorTimesScalar, 7},
    };
    std::vector<double> err = {0.01, 0.5, 0.02};   // site 1 (line 6) is worst
    auto blame = reward::attribute(err, sites);
    CHECK(blame.found);
    CHECK_EQ(blame.site_id, 1u);
    CHECK_EQ(blame.line, 6u);
    CHECK_NEAR(blame.severity, 0.5, 1e-12);

    // No error anywhere -> no blame.
    std::vector<double> zero = {0, 0, 0};
    CHECK(!reward::attribute(zero, sites).found);
}
