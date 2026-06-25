#include "omni/synth/driver.hpp"

namespace omni::synth {

DriverResult Driver::run(Generator& gen, const std::string& prompt, Stage stage, const DriverConfig& cfg) {
    DriverResult r;
    std::string feedback;                          // empty on first attempt
    for (int a = 1; a <= cfg.max_attempts; ++a) {
        r.attempts = a;
        r.source = gen.generate(prompt, feedback);
        CompileResult c = validator_.validate(r.source, stage);
        r.log = c.log;
        if (c.ok) { r.compiled = true; return r; }
        // Build feedback for the next attempt. shader_cmake fed back only the compiler error;
        // OmniTrace enriches this with execution diagnostics (divergence/NaN/numerical) once
        // the candidate runs — the hook is here.
        feedback = "The shader failed to compile:\n" + c.log;
        if (c.first_error_line >= 0) feedback += "\n(error near line " + std::to_string(c.first_error_line) + ")";
    }
    return r;                                       // exhausted retries
}

CompileAtK benchmark_compile_at_k(Driver& d, Generator& gen,
                                  const std::vector<std::string>& prompts,
                                  Stage stage, const DriverConfig& cfg) {
    CompileAtK out; out.n = (int)prompts.size();
    if (prompts.empty()) return out;
    int ok1 = 0, okk = 0, total_attempts = 0;
    DriverConfig one = cfg; one.max_attempts = 1;
    for (const auto& p : prompts) {
        // compile@1: a single attempt (no feedback).
        if (d.run(gen, p, stage, one).compiled) ++ok1;
        // compile@k: full loop with feedback.
        auto rk = d.run(gen, p, stage, cfg);
        if (rk.compiled) { ++okk; total_attempts += rk.attempts; }
        else total_attempts += cfg.max_attempts;
    }
    out.compile_at_1 = (double)ok1 / out.n;
    out.compile_at_k = (double)okk / out.n;
    out.mean_attempts = (double)total_attempts / out.n;
    return out;
}

} // namespace omni::synth
