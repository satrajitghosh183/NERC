#include "omni/synth/renderer.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>

namespace omni::synth {

static bool exists(const std::string& p) { struct stat s; return stat(p.c_str(), &s) == 0; }

static std::string find_glslang() {
    if (const char* e = std::getenv("OMNI_GLSLANG"); e && exists(e)) return e;
    std::string p = "/Users/satrajitghosh/Software/VulkanSDK/1.4.328.1/macOS/bin/glslangValidator";
    return exists(p) ? p : "";
}

// Compile GLSL compute source -> SPIR-V words via the reference compiler.
static bool compile_compute(const std::string& src, std::vector<uint32_t>& out, std::string& err) {
    std::string gl = find_glslang();
    if (gl.empty()) { err = "glslangValidator not found"; return false; }
    const char* td = std::getenv("TMPDIR"); std::string dir = td ? td : "/tmp";
    std::string base = dir + "/omni_render_" + std::to_string(getpid());
    std::string cpath = base + ".comp", spath = base + ".comp.spv";
    if (FILE* f = std::fopen(cpath.c_str(), "wb")) { std::fwrite(src.data(), 1, src.size(), f); std::fclose(f); }
    else { err = "cannot write shader"; return false; }
    std::string cmd = "'" + gl + "' -V '" + cpath + "' -o '" + spath + "' 2>&1";
    std::string log; if (FILE* p = popen(cmd.c_str(), "r")) { char b[512]; size_t g; while ((g = std::fread(b,1,sizeof b,p))>0) log.append(b,g); pclose(p); }
    std::remove(cpath.c_str());
    if (!exists(spath)) { err = "compile failed: " + log; return false; }
    bool ok = false;
    if (FILE* f = std::fopen(spath.c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
        out.resize(n / 4); ok = std::fread(out.data(), 4, out.size(), f) == out.size(); std::fclose(f);
    }
    std::remove(spath.c_str());
    if (!ok) err = "cannot read spirv";
    return ok;
}

bool Renderer::init(std::string* err) { return vk_.init(err); }

RenderResult Renderer::render(const std::string& color_expr, int w, int h) {
    RenderResult r;
    if (!vk_.ready()) { r.error = "renderer not initialised"; return r; }

    char hdr[256];
    std::snprintf(hdr, sizeof hdr,
        "#version 450\nlayout(local_size_x=8, local_size_y=8) in;\n"
        "layout(std430,set=0,binding=0) buffer Img { vec4 px[]; };\n"
        "const int WW=%d, HH=%d;\n", w, h);
    std::string src = hdr;
    src += "void main(){\n"
           "  ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
           "  if (g.x>=WW || g.y>=HH) return;\n"
           "  vec2 iResolution = vec2(float(WW),float(HH));\n"
           "  vec2 fragCoord = vec2(g)+0.5;\n"
           "  vec2 uv = fragCoord/iResolution;\n"
           "  vec3 col = (" + color_expr + ");\n"
           "  px[g.y*WW + g.x] = vec4(col,1.0);\n"
           "}\n";

    std::vector<uint32_t> spirv;
    if (!compile_compute(src, spirv, r.error)) return r;

    auto raw = vk_.run_raw(spirv, (size_t)w * h * 16, (w + 7) / 8, (h + 7) / 8, 1);
    if (!raw.ok) { r.error = raw.error; return r; }

    r.image.w = w; r.image.h = h;
    r.image.rgba.resize((size_t)w * h * 4);
    std::memcpy(r.image.rgba.data(), raw.bytes.data(), (size_t)w * h * 16);
    r.ok = true;
    return r;
}

} // namespace omni::synth
