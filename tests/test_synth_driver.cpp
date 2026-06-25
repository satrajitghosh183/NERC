#include "omni/test.hpp"
#include "omni/synth/driver.hpp"
#include "omni/bench.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace omni;
using namespace omni::synth;

// Deterministic stand-in for the LLM. "hard" prompts fail on the first attempt but get fixed
// once feedback (the compiler error) is provided — exactly the behaviour the loop exploits.
struct StubGenerator : Generator {
    std::string generate(const std::string& prompt, const std::string& feedback) override {
        bool hard = prompt.find("hard") != std::string::npos;
        if (hard && feedback.empty()) {
            return "#version 450\nlayout(location=0) out vec4 c;\n"
                   "void main(){ c = vec4(0.2,0.4,0.6,1.0) }\n";   // missing ';' -> compile error
        }
        return "#version 450\nlayout(location=0) out vec4 c;\n"
               "void main(){ c = vec4(0.2,0.4,0.6,1.0); }\n";       // valid
    }
};

TEST(synth_driver, feedback_loop_fixes_compile_errors) {
    Driver d;
    if (!d.available()) { std::printf("    glslangValidator unavailable; skipping\n"); return; }
    StubGenerator gen;
    DriverConfig cfg; cfg.max_attempts = 3;

    // A hard prompt compiles only after feedback (attempt 2).
    auto hard = d.run(gen, "hard: blue panel", Stage::Fragment, cfg);
    CHECK(hard.compiled);
    CHECK_EQ(hard.attempts, 2);

    // An easy prompt compiles immediately.
    auto easy = d.run(gen, "easy: red panel", Stage::Fragment, cfg);
    CHECK(easy.compiled);
    CHECK_EQ(easy.attempts, 1);
}

TEST(synth_driver, compile_at_k_benchmark) {
    Driver d;
    if (!d.available()) { std::printf("    glslangValidator unavailable; skipping\n"); return; }
    StubGenerator gen;
    DriverConfig cfg; cfg.max_attempts = 3;

    std::vector<std::string> prompts = {
        "easy: red", "easy: green", "easy: gradient", "easy: checker",
        "hard: blue", "hard: noise", "hard: plasma", "hard: fire",
    };
    auto res = benchmark_compile_at_k(d, gen, prompts, Stage::Fragment, cfg);
    std::printf("    n=%d compile@1=%.3f compile@k=%.3f mean_attempts=%.2f\n",
                res.n, res.compile_at_1, res.compile_at_k, res.mean_attempts);

    // The feedback loop lifts compile rate (the core shader_cmake result, reproduced in C++).
    CHECK(res.compile_at_k > res.compile_at_1);
    CHECK_NEAR(res.compile_at_1, 0.5, 1e-9);   // half the prompts are "hard"
    CHECK_NEAR(res.compile_at_k, 1.0, 1e-9);   // all fixed within the loop

    omni::bench::Suite s("synth_compile_at_k");
    s.add("stub_generator", "compile_at_1", res.compile_at_1, "frac");
    s.add("stub_generator", "compile_at_k", res.compile_at_k, "frac");
    s.add("stub_generator", "mean_attempts", res.mean_attempts, "count");
    s.write();
}
