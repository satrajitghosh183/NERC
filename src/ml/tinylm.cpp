#include "omni/ml/tinylm.hpp"
#include <cmath>
#include <algorithm>

namespace omni::ml {

namespace { struct Rng { uint64_t s; float n(){ s=s*6364136223846793005ull+1; return ((s>>40)/(float)(1<<24))*2.f-1.f; } }; }

TinyLM::TinyLM(int vocab, int dim, uint64_t seed) : V(vocab), d(dim), E(vocab*dim), U(vocab*dim), b(vocab, 0.f) {
    Rng r{seed ? seed : 1};
    float scale = 0.1f;
    for (auto& x : E) x = r.n() * scale;
    for (auto& x : U) x = r.n() * scale;
}

// logits[k] = b[k] + sum_j U[k,j]*h[j];  softmax; returns -log p[target] and (optionally) writes p.
static double forward(const TinyLM& m, const float* h, int target, std::vector<float>& p) {
    p.assign(m.V, 0.f);
    double maxz = -1e30;
    for (int k = 0; k < m.V; ++k) {
        double z = m.b[k];
        const float* Uk = &m.U[(size_t)k * m.d];
        for (int j = 0; j < m.d; ++j) z += Uk[j] * h[j];
        p[k] = (float)z; if (z > maxz) maxz = z;
    }
    double sum = 0; for (int k = 0; k < m.V; ++k) { double e = std::exp(p[k] - maxz); p[k] = (float)e; sum += e; }
    double inv = 1.0 / sum; for (int k = 0; k < m.V; ++k) p[k] = (float)(p[k] * inv);
    return -std::log(std::max((double)p[target], 1e-12));
}

double TinyLM::loss(const std::vector<uint16_t>& tokens) const {
    if (tokens.size() < 2) return 0.0;
    std::vector<float> p; double total = 0; size_t n = 0;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        uint16_t cur = tokens[i], next = tokens[i+1];
        if (cur >= V || next >= V) continue;
        total += forward(*this, &E[(size_t)cur * d], next, p); ++n;
    }
    return n ? total / n : 0.0;
}

double TinyLM::train_epoch(const std::vector<uint16_t>& tokens, float lr) {
    if (tokens.size() < 2) return 0.0;
    std::vector<float> p, dh(d);
    double total = 0; size_t n = 0;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        uint16_t cur = tokens[i], next = tokens[i+1];
        if (cur >= V || next >= V) continue;
        float* h = &E[(size_t)cur * d];
        total += forward(*this, h, next, p); ++n;

        // dz = p - onehot(next).  Update U,b; accumulate dh; then update E[cur].
        std::fill(dh.begin(), dh.end(), 0.f);
        for (int k = 0; k < V; ++k) {
            float dz = p[k] - (k == next ? 1.f : 0.f);
            float* Uk = &U[(size_t)k * d];
            for (int j = 0; j < d; ++j) { dh[j] += dz * Uk[j]; Uk[j] -= lr * dz * h[j]; }
            b[k] -= lr * dz;
        }
        for (int j = 0; j < d; ++j) h[j] -= lr * dh[j];   // update the embedding row in place
    }
    return n ? total / n : 0.0;
}

uint16_t TinyLM::predict(uint16_t cur) const {
    if (cur >= V) return 0;
    std::vector<float> p; forward(*this, &E[(size_t)cur * d], 0, p);
    int best = 0; for (int k = 1; k < V; ++k) if (p[k] > p[best]) best = k;
    return (uint16_t)best;
}

} // namespace omni::ml
