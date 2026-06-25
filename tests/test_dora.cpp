#include "omni/test.hpp"
#include "omni/ml/dora.hpp"
#include <cmath>
#include <vector>

using namespace omni::ml;

// Local LCG so the test is deterministic without <random>.
struct R { uint64_t s; R(uint64_t x):s(x){} float n(){ s=s*6364136223846793005ull+1; return ((s>>40)/(float)(1<<24))*2.0f-1.0f; } };

static float colnorm(const std::vector<float>& W, int d, int k, int j) {
    double s = 0; for (int i = 0; i < d; ++i) { double v = W[(size_t)i*k+j]; s += v*v; } return (float)std::sqrt(s);
}

// With A=B=0 and m=||W0[:,j]||, the adapted weight must equal the base exactly.
TEST(dora, identity_when_unadapted) {
    const int d = 8, k = 6, r = 2;
    DoRALinear m(d, k, r);
    R rng(42);
    for (auto& x : m.W0) x = rng.n();
    m.init_from_base();

    auto Wp = m.effective_weight();
    REQUIRE_EQ(Wp.size(), m.W0.size());
    for (size_t i = 0; i < Wp.size(); ++i) CHECK_NEAR(Wp[i], m.W0[i], 1e-5);

    std::vector<float> x(k); for (auto& v : x) v = rng.n();
    auto y = m.forward(x);
    // Compare to a direct W0 * x.
    for (int i = 0; i < d; ++i) {
        double acc = 0; for (int j = 0; j < k; ++j) acc += (double)m.W0[(size_t)i*k+j]*x[j];
        CHECK_NEAR(y[i], (float)acc, 1e-4);
    }
}

// DoRA's defining property: the magnitude vector m controls each output column's norm,
// while the low-rank adapter changes only its direction.
TEST(dora, magnitude_controls_column_norm_direction_free) {
    const int d = 10, k = 7, r = 3;
    DoRALinear m(d, k, r);
    R rng(7);
    for (auto& x : m.W0) x = rng.n();
    m.init_from_base();                       // m[j] = ||W0[:,j]||, A=B=0
    // Now perturb the direction via a non-trivial adapter.
    for (auto& a : m.A) a = 0.3f * rng.n();
    for (auto& b : m.B) b = 0.3f * rng.n();

    auto Wp = m.effective_weight();
    bool any_direction_changed = false;
    for (int j = 0; j < k; ++j) {
        // Each adapted column keeps the magnitude m[j]...
        CHECK_NEAR(colnorm(Wp, d, k, j), m.m[j], 1e-4);
        // ...but its direction generally differs from the base column.
        double dot = 0, n0 = 0, n1 = 0;
        for (int i = 0; i < d; ++i) {
            double a = m.W0[(size_t)i*k+j], b = Wp[(size_t)i*k+j];
            dot += a*b; n0 += a*a; n1 += b*b;
        }
        double cosang = dot / (std::sqrt(n0)*std::sqrt(n1) + 1e-12);
        if (cosang < 0.9999) any_direction_changed = true;
    }
    CHECK(any_direction_changed);
}

TEST(dora, parameter_budget_smaller_than_full) {
    DoRALinear m(4096, 4096, 16);             // a 4096x4096 layer, rank 16
    CHECK(m.trainable_params() < m.full_params());
    double frac = (double)m.trainable_params() / (double)m.full_params();
    std::printf("    DoRA trainable fraction at d=k=4096,r=16: %.4f%%\n", frac * 100.0);
    CHECK(frac < 0.02);                        // <2% of full fine-tune
}
