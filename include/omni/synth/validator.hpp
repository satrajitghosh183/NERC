// omni/synth/validator.hpp — GLSL compile/link validation (C++ rewrite of shader_cmake's
// Python ModernGL validator). The compile signal is the first feedback term; the driver
// loop upgrades it with OmniTrace diagnostics (PLAN.md §13 synth/validator).
//
// We use the real reference GLSL compiler (glslangValidator from the Vulkan SDK) as the
// compile oracle — the compiler is the *target* we debug, not a contribution shortcut.
#pragma once
#include <string>

namespace omni::synth {

enum class Stage { Vertex, Fragment, Compute };

struct CompileResult {
    bool ok = false;
    std::string log;       // compiler diagnostics (empty on success)
    int first_error_line = -1;
};

class GlslValidator {
public:
    GlslValidator();                       // locates glslangValidator
    bool available() const { return !glslang_.empty(); }
    const std::string& tool_path() const { return glslang_; }

    CompileResult validate(const std::string& source, Stage stage) const;

private:
    std::string glslang_;
};

} // namespace omni::synth
