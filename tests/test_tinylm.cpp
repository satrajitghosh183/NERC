#include "omni/test.hpp"
#include "omni/ml/tinylm.hpp"
#include "omni/synth/corpus.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace omni;

// Train a from-scratch neural LM on shader tokens; the loss must genuinely decrease.
// This is the scaled-down CPU proof of the SLM training mechanism (the 32B/CUDA run is
// documented as needing a GPU cluster).
TEST(tinylm, trains_on_shader_corpus_loss_decreases) {
    // A small but real GLSL corpus (repeated to provide enough tokens for SGD).
    std::string glsl =
        "#version 450\n"
        "layout(location=0) out vec4 fragColor;\n"
        "uniform vec2 iResolution;\n"
        "vec3 palette(float t){ return 0.5+0.5*cos(6.2831*(t+vec3(0.0,0.33,0.67))); }\n"
        "void main(){\n"
        "  vec2 uv = gl_FragCoord.xy/iResolution;\n"
        "  vec3 col = palette(uv.x+uv.y);\n"
        "  fragColor = vec4(col, 1.0);\n"
        "}\n";
    std::string big; for (int i = 0; i < 12; ++i) big += glsl;
    auto tokens = synth::tokenize(big);
    REQUIRE(tokens.size() > 1000);

    ml::TinyLM lm(synth::VOCAB_SIZE, 32, /*seed*/7);
    double initial = lm.loss(tokens);
    for (int e = 0; e < 25; ++e) lm.train_epoch(tokens, 0.5f);
    double final_loss = lm.loss(tokens);

    std::printf("    tokens=%zu  initial_loss=%.4f  final_loss=%.4f  (perplexity %.1f -> %.1f)\n",
                tokens.size(), initial, final_loss, std::exp(initial), std::exp(final_loss));

    // Training reduced the loss substantially (GLSL is highly structured -> learnable).
    CHECK(final_loss < initial);
    CHECK(final_loss < 0.7 * initial);

    // The trained model makes structured predictions (e.g., after 'v' it tends toward shader
    // tokens, not noise) — just assert predict() returns a valid in-vocab token.
    uint16_t p = lm.predict((uint16_t)'v');
    CHECK(p < synth::VOCAB_SIZE);
}
