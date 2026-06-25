// omni/synth/driver.hpp — the generate -> validate -> diagnose -> feedback -> retry loop
// (C++ rewrite of shader_cmake's generator.py). The model plugs in via Generator; the loop
// upgrades shader_cmake's compile-error feedback toward OmniTrace's richer diagnostics
// (PLAN.md §13/§14 Stage 1). Measures compile@1 vs compile@k.
#pragma once
#include "omni/synth/validator.hpp"
#include <string>
#include <vector>

namespace omni::synth {

// Pluggable shader generator. A real LLM (via llm-cpp inference) implements this; tests use
// a deterministic stub. `feedback` is empty on the first attempt, else the prior diagnostics.
struct Generator {
    virtual ~Generator() = default;
    virtual std::string generate(const std::string& prompt, const std::string& feedback) = 0;
};

struct DriverConfig { int max_attempts = 3; };

struct DriverResult {
    bool compiled = false;
    int attempts = 0;
    std::string source;
    std::string log;
};

class Driver {
public:
    bool available() const { return validator_.available(); }
    DriverResult run(Generator& gen, const std::string& prompt, Stage stage, const DriverConfig& cfg);
private:
    GlslValidator validator_;
};

struct CompileAtK {
    int n = 0;
    double compile_at_1 = 0.0;   // fraction compiling on the first attempt
    double compile_at_k = 0.0;   // fraction compiling within max_attempts (the loop's lift)
    double mean_attempts = 0.0;
};

CompileAtK benchmark_compile_at_k(Driver& d, Generator& gen,
                                  const std::vector<std::string>& prompts,
                                  Stage stage, const DriverConfig& cfg);

} // namespace omni::synth
