#include "omni/synth/validator.hpp"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <array>

namespace omni::synth {

static bool file_exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

GlslValidator::GlslValidator() {
    // 1) explicit override, 2) known Vulkan SDK location, 3) PATH.
    if (const char* env = std::getenv("OMNI_GLSLANG"); env && file_exists(env)) { glslang_ = env; return; }
    const char* candidates[] = {
        "/Users/satrajitghosh/Software/VulkanSDK/1.4.328.1/macOS/bin/glslangValidator",
    };
    for (const char* c : candidates) if (file_exists(c)) { glslang_ = c; return; }
    // PATH lookup via `command -v`.
    if (FILE* p = popen("command -v glslangValidator 2>/dev/null", "r")) {
        std::array<char, 1024> buf{};
        if (std::fgets(buf.data(), buf.size(), p)) {
            std::string s(buf.data());
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (file_exists(s)) glslang_ = s;
        }
        pclose(p);
    }
}

static const char* ext_for(Stage s) {
    switch (s) { case Stage::Vertex: return "vert"; case Stage::Compute: return "comp"; default: return "frag"; }
}

CompileResult GlslValidator::validate(const std::string& source, Stage stage) const {
    CompileResult r;
    if (glslang_.empty()) { r.log = "glslangValidator not found"; return r; }

    const char* tmpdir = std::getenv("TMPDIR"); if (!tmpdir) tmpdir = "/tmp";
    std::string path = std::string(tmpdir) + "/omni_val_" + std::to_string(getpid()) + "." + ext_for(stage);
    if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(source.data(), 1, source.size(), f); std::fclose(f); }
    else { r.log = "cannot write temp shader"; return r; }

    // -V = compile to SPIR-V for Vulkan (full front-end + validation). Discard the binary.
    std::string cmd = "'" + glslang_ + "' -V '" + path + "' -o /dev/null 2>&1";
    std::string out;
    if (FILE* p = popen(cmd.c_str(), "r")) {
        std::array<char, 4096> buf{};
        size_t got;
        while ((got = std::fread(buf.data(), 1, buf.size(), p)) > 0) out.append(buf.data(), got);
        int rc = pclose(p);
        r.ok = (rc == 0);
    } else {
        r.log = "failed to launch compiler";
        std::remove(path.c_str());
        return r;
    }
    std::remove(path.c_str());

    if (!r.ok) {
        r.log = out;
        // glslang error lines look like: "ERROR: 0:12: '...' : ..." — pull the first line number.
        size_t pos = out.find("ERROR: ");
        if (pos != std::string::npos) {
            size_t colon1 = out.find(':', pos + 7);
            size_t colon2 = (colon1 != std::string::npos) ? out.find(':', colon1 + 1) : std::string::npos;
            if (colon1 != std::string::npos && colon2 != std::string::npos)
                r.first_error_line = std::atoi(out.substr(colon1 + 1, colon2 - colon1 - 1).c_str());
        }
    }
    return r;
}

} // namespace omni::synth
