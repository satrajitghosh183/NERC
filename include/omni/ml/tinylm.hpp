// omni/ml/tinylm.hpp — a from-scratch CPU neural language model + training loop.
//
// SCOPE: this is a *scaled-down, hardware-independent proof* of the SLM-training mechanism
// (PLAN.md §10/§20). The headline run — a from-scratch SLM and a DoRA-32B trained with the
// debugger reward in the loop — needs datacenter GPUs and the CUDA-native llm-cpp trainer,
// which this Apple-silicon machine cannot run. Here we prove the loop is real: a low-rank
// neural bigram LM with hand-derived gradients, trained on the shader corpus, whose loss
// provably decreases. The reward-in-the-loop half is proven separately (reward oracle #19,
// synth driver #15).
#pragma once
#include <cstdint>
#include <vector>

namespace omni::ml {

// Neural bigram LM: h = E[cur] (dim d); logits = U*h + b (vocab V). Cross-entropy next-token.
struct TinyLM {
    int V, d;
    std::vector<float> E;   // V*d  input embeddings
    std::vector<float> U;   // V*d  output projection
    std::vector<float> b;   // V    bias

    TinyLM(int vocab, int dim, uint64_t seed = 1);

    // Average cross-entropy (nats) over consecutive (cur,next) pairs in `tokens`.
    double loss(const std::vector<uint16_t>& tokens) const;

    // One SGD epoch over the token stream; returns the average loss during the epoch.
    double train_epoch(const std::vector<uint16_t>& tokens, float lr);

    // Greedy next-token given current token.
    uint16_t predict(uint16_t cur) const;
};

} // namespace omni::ml
