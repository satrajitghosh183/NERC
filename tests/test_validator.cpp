#include "omni/test.hpp"
#include "omni/synth/validator.hpp"
#include <cstdio>
#include <string>

using namespace omni::synth;

static const char* GOOD_FRAG = R"(#version 450
layout(location=0) out vec4 c;
void main(){ c = vec4(1.0, 0.5, 0.25, 1.0); }
)";

// Missing semicolon / undefined identifier -> compile error.
static const char* BAD_FRAG = R"(#version 450
layout(location=0) out vec4 c;
void main(){ c = vec4(undefined_var, 0.0, 0.0, 1.0) }
)";

TEST(validator, compiles_good_rejects_bad) {
    GlslValidator v;
    if (!v.available()) { std::printf("    glslangValidator unavailable; skipping\n"); return; }
    std::printf("    using %s\n", v.tool_path().c_str());

    auto good = v.validate(GOOD_FRAG, Stage::Fragment);
    if (!good.ok) std::printf("    unexpected error: %s\n", good.log.c_str());
    CHECK(good.ok);

    auto bad = v.validate(BAD_FRAG, Stage::Fragment);
    CHECK(!bad.ok);
    CHECK(!bad.log.empty());
    std::printf("    bad shader rejected, first_error_line=%d\n", bad.first_error_line);
}
