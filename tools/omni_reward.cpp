// omni/tools/omni_reward.cpp
//
// Debugger-in-the-loop reward CLI — the OmniTrace debugger exposed as a reward signal for
// shader-synthesis RL. Pure C++; links libomni_core.
//
// Pipeline (each stage is a real measurement from the debugger, not a heuristic):
//   1. compile   GLSL -> SPIR-V via glslangValidator (the compile oracle).
//   2. lift      SPIR-V -> UIR via the hand-written frontend (structural validity).
//   3. execute   run the UIR on the CPU SIMT reference over a grid of fragment
//                coordinates; flag NaN/Inf and degenerate (constant) output.
//   4. score     omni::reward::score() composes the decomposed reward.
//
// The point: a shader can *compile* and still be broken (NaN, all-black, constant). Only
// running it catches that — which is exactly what compile@k misses and the debugger provides.
//
// Usage:
//   omni_reward < shader.frag                 # read GLSL from stdin
//   omni_reward --file shader.frag            # or from a file
//   omni_reward --grid 12 --file shader.frag  # NxN execution probe (default 8)
// Output: one line of JSON to stdout. Exit code 0 always (the JSON carries the verdict).

#include "omni/synth/validator.hpp"
#include "omni/frontends/spirv.hpp"
#include "omni/cpuref/interp.hpp"
#include "omni/uir/ir.hpp"
#include "omni/reward/oracle.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>  // close (mkstemps owns the descriptor)

namespace {

using namespace omni;

// ---- RAII temp file: created in the system temp dir, unlinked on scope exit. -------------
class TempFile {
public:
    explicit TempFile(std::string_view suffix) {
        // mkstemps creates the file atomically and keeps the suffix (so glslangValidator
        // infers the shader stage from the .frag / .spv extension).
        std::string tmpl = "/tmp/omni_reward_XXXXXX" + std::string(suffix);
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        const int fd = ::mkstemps(buf.data(), static_cast<int>(suffix.size()));
        if (fd != -1) { ::close(fd); path_.assign(buf.data()); }
        else          { path_ = std::move(tmpl); }  // fall back to a fixed name
    }
    ~TempFile() { std::remove(path_.c_str()); }

    TempFile(const TempFile&) = delete;             // Rule of Five: this owns a filesystem
    TempFile& operator=(const TempFile&) = delete;  // resource, so forbid copies/moves.
    TempFile(TempFile&&) = delete;
    TempFile& operator=(TempFile&&) = delete;

    [[nodiscard]] const std::string& path() const noexcept { return path_; }

    [[nodiscard]] bool write(std::string_view data) const {
        std::ofstream os(path_, std::ios::binary | std::ios::trunc);
        os.write(data.data(), static_cast<std::streamsize>(data.size()));
        return static_cast<bool>(os);
    }

private:
    std::string path_;
};

[[nodiscard]] std::string read_stream(std::istream& in) {
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

[[nodiscard]] std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
                else out += c;
        }
    }
    return out;
}

// glslangValidator: GLSL -> .spv. Returns the spv path written into `spv`, or nullopt.
[[nodiscard]] bool emit_spirv(const std::string& glslang_tool, const std::string& frag_path,
                              const std::string& spv_path) {
    // -V = Vulkan SPIR-V, -g = keep debug names (so the interpreter can resolve globals).
    const std::string cmd = glslang_tool + " -V -g \"" + frag_path + "\" -o \"" + spv_path +
                            "\" >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

[[nodiscard]] uir::FuncId find_function(const uir::Module& m, std::string_view name) {
    for (std::size_t i = 0; i < m.num_functions(); ++i) {
        if (m.function(static_cast<uir::FuncId>(i)).name == name)
            return static_cast<uir::FuncId>(i);
    }
    return uir::INVALID;
}

// Locate the fragment output variable's storage cell by trying the conventional names our
// harness and Shadertoy shaders use. Returns nullptr if none are present.
[[nodiscard]] cpuref::Cell* find_output(cpuref::Interp& interp) {
    static constexpr std::array names{"_O", "outColor", "fragColor", "gl_FragColor", "color", "FragColor"};
    for (std::string_view n : names)
        if (cpuref::Cell* c = interp.global(std::string(n))) return c;
    return nullptr;
}

[[nodiscard]] bool is_finite(const cpuref::Val& v) {
    for (unsigned k = 0; k < v.n; ++k)
        if (!std::isfinite(v.f[k])) return false;
    return true;
}

// Result of running the lifted shader on the CPU reference over a coordinate grid.
struct ExecProbe {
    bool   ran          = false;  // the interpreter executed the function to completion
    bool   all_finite   = true;   // no NaN/Inf in any sampled output
    double output_var   = 0.0;    // luminance variance across the grid (0 == degenerate)
    int    samples      = 0;
};

// Best-effort execution: drive `f` across an NxN grid of fragment coordinates. The v1
// interpreter does not cover every shader (built-ins, unbounded loops); a failure to run is
// reported, never fatal — compile/lift still stand.
[[nodiscard]] ExecProbe probe_execution(const uir::Module& m, uir::FuncId f, int grid, float res) {
    ExecProbe p;
    std::vector<double> lum;
    lum.reserve(static_cast<std::size_t>(grid) * grid);

    for (int gy = 0; gy < grid; ++gy) {
        for (int gx = 0; gx < grid; ++gx) {
            cpuref::Interp interp(m);                 // fresh per-lane state
            // Seed common fragment inputs if the shader exposes them by name.
            const float fx = (static_cast<float>(gx) + 0.5f) / static_cast<float>(grid) * res;
            const float fy = (static_cast<float>(gy) + 0.5f) / static_cast<float>(grid) * res;
            if (cpuref::Cell* fc = interp.global("gl_FragCoord")) fc->val = cpuref::Val::vecf({fx, fy, 0.0f, 1.0f});
            if (cpuref::Cell* fc = interp.global("fragCoord"))    fc->val = cpuref::Val::vecf({fx, fy});

            std::string err;
            if (!interp.run(f, &err)) return p;        // unsupported instruction -> ran stays false
            p.ran = true;

            cpuref::Cell* out = find_output(interp);
            if (out == nullptr || !out->leaf) continue;
            if (!is_finite(out->val)) { p.all_finite = false; }
            const cpuref::Val& c = out->val;
            lum.push_back(0.2126 * c.f[0] + 0.7152 * c.f[1] + 0.0722 * c.f[2]);
            ++p.samples;
        }
    }

    if (lum.size() >= 2) {
        double mean = 0.0;
        for (double x : lum) mean += x;
        mean /= static_cast<double>(lum.size());
        double var = 0.0;
        for (double x : lum) var += (x - mean) * (x - mean);
        p.output_var = var / static_cast<double>(lum.size());
    }
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    std::string file;
    int grid = 8;
    float res = 256.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--file" && i + 1 < argc) file = argv[++i];
        else if (a == "--grid" && i + 1 < argc) grid = std::max(1, std::atoi(argv[++i]));
        else if (a == "--res"  && i + 1 < argc) res  = static_cast<float>(std::atof(argv[++i]));
    }

    std::string src;
    if (!file.empty()) {
        std::ifstream is(file, std::ios::binary);
        if (!is) { std::puts(R"({"error":"cannot open --file"})"); return 0; }
        src = read_stream(is);
    } else {
        src = read_stream(std::cin);
    }

    // ---- stage 1: compile ----------------------------------------------------------------
    const synth::GlslValidator validator;
    const synth::CompileResult cr = validator.validate(src, synth::Stage::Fragment);

    reward::Inputs in;
    in.compiled = cr.ok;
    bool lifted = false;
    ExecProbe probe;

    // ---- stages 2-3: lift + execute (only meaningful once it compiles) --------------------
    if (cr.ok && validator.available()) {
        const TempFile frag(".frag");
        const TempFile spv(".spv");
        if (frag.write(src) && emit_spirv(validator.tool_path(), frag.path(), spv.path())) {
            uir::Module m;
            const frontends::SpirvLiftResult lr = frontends::lift_spirv_file(spv.path(), m);
            lifted = lr.ok;
            if (lr.ok) {
                const uir::FuncId f = find_function(m, "main");
                if (f != uir::INVALID) {
                    probe = probe_execution(m, f, grid, res);
                    // Map debugger measurements onto the reward oracle's inputs.
                    in.executed     = probe.ran && probe.all_finite && probe.samples > 0;
                    // A shader that runs to a constant image is "valid but degenerate": give
                    // partial visual credit that saturates with output variance.
                    in.visual_match = in.executed ? (1.0 - std::exp(-probe.output_var * 32.0)) : 0.0;
                }
            }
        }
    }

    const reward::Breakdown b = reward::score(in);

    // ---- output: one JSON line -----------------------------------------------------------
    std::printf(
        "{\"compiled\":%s,\"lifted\":%s,\"executed\":%s,\"all_finite\":%s,"
        "\"output_variance\":%.6g,\"exec_samples\":%d,\"reward\":%.6f,"
        "\"breakdown\":{\"compile\":%.4f,\"exec\":%.4f,\"visual\":%.4f},"
        "\"compile_log\":\"%s\"}\n",
        in.compiled ? "true" : "false", lifted ? "true" : "false",
        in.executed ? "true" : "false", probe.all_finite ? "true" : "false",
        probe.output_var, probe.samples, b.total,
        b.compile, b.exec, b.visual,
        json_escape(cr.log).c_str());
    return 0;
}
