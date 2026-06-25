#include "omni/ml/dora.hpp"
#include <cmath>

namespace omni::ml {

static float col_norm(const std::vector<float>& W, int d, int k, int j) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) { double v = W[(size_t)i * k + j]; s += v * v; }
    return (float)std::sqrt(s);
}

void DoRALinear::init_from_base() {
    for (auto& a : A) a = 0.0f;
    for (auto& b : B) b = 0.0f;
    for (int j = 0; j < k; ++j) m[j] = col_norm(W0, d, k, j);
}

std::vector<float> DoRALinear::effective_weight() const {
    std::vector<float> V = W0;                         // V = W0 + B*A
    for (int i = 0; i < d; ++i)
        for (int l = 0; l < r; ++l) {
            float bil = B[(size_t)i * r + l];
            if (bil == 0.0f) continue;
            for (int j = 0; j < k; ++j)
                V[(size_t)i * k + j] += bil * A[(size_t)l * k + j];
        }
    // Per-column normalise to unit direction, then scale by magnitude m[j].
    std::vector<float> Wp(V.size());
    for (int j = 0; j < k; ++j) {
        double s = 0.0;
        for (int i = 0; i < d; ++i) { double v = V[(size_t)i * k + j]; s += v * v; }
        double nrm = std::sqrt(s);
        double scale = nrm > 0.0 ? (double)m[j] / nrm : 0.0;
        for (int i = 0; i < d; ++i) Wp[(size_t)i * k + j] = (float)(V[(size_t)i * k + j] * scale);
    }
    return Wp;
}

std::vector<float> DoRALinear::forward(const std::vector<float>& x) const {
    std::vector<float> Wp = effective_weight();
    std::vector<float> y(d, 0.0f);
    for (int i = 0; i < d; ++i) {
        double acc = 0.0;
        for (int j = 0; j < k; ++j) acc += (double)Wp[(size_t)i * k + j] * (double)x[j];
        y[i] = (float)acc;
    }
    return y;
}

} // namespace omni::ml
