// omni/tools/omni_render.cpp
//
// Render a Shadertoy `mainImage` shader to an image on the real GPU (Vulkan / MoltenVK),
// the way shader_cmake's ModernGL preview did — but in C++ via OmniTrace's renderer.
// Embeds the shader's mainImage into a compute shader (one invocation per pixel), runs it,
// reads back the RGBA image, and writes a PPM (the CLI converts to PNG and opens it).
//
//   omni_render <shader.frag> <out.ppm> [width] [height] [iTime]
//
// Exit 0 on success; prints "ok WxH" / "err <msg>" to stdout.

#include "omni/gpu/vulkan_capture.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss; ss << in.rdbuf();
    return ss.str();
}

// Strip a wrapping harness if present (keep the mainImage + helper functions, drop a
// trailing `void main(){...}` and Shadertoy-uniform #defines/uniform blocks we redeclare).
[[nodiscard]] std::string shader_body(std::string src) {
    if (const auto p = src.find("void main()"); p != std::string::npos)
        src.erase(p);                                   // drop the harness entry point
    return src;
}

// glslangValidator: compile GLSL compute -> SPIR-V words.
[[nodiscard]] bool compile_compute(const std::string& glsl, std::vector<uint32_t>& out,
                                   std::string& err) {
    const std::string comp = std::string(std::tmpnam(nullptr)) + ".comp";
    const std::string spv  = comp + ".spv";
    { std::ofstream o(comp); o << glsl; }
    const std::string cmd = "glslangValidator -V \"" + comp + "\" -o \"" + spv + "\" 2>/dev/null";
    const bool ran_ok = std::system(cmd.c_str()) == 0;
    std::ifstream in(spv, std::ios::binary);
    std::remove(comp.c_str());
    if (!ran_ok || !in) { err = "compute compile failed"; std::remove(spv.c_str()); return false; }
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
    std::remove(spv.c_str());
    out.resize(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), out.size() * 4);
    return !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::puts("err usage: omni_render <shader> <out.ppm> [w] [h] [iTime]"); return 1; }
    const std::string shader = argv[1], out = argv[2];
    const int   W  = argc > 3 ? std::max(8, std::atoi(argv[3])) : 512;
    const int   H  = argc > 4 ? std::max(8, std::atoi(argv[4])) : 512;
    const float iT = argc > 5 ? static_cast<float>(std::atof(argv[5])) : 0.0f;

    // Build a compute shader that calls the candidate's mainImage for every pixel.
    std::ostringstream src;
    src << "#version 450\nlayout(local_size_x=8, local_size_y=8) in;\n"
        << "layout(std430,set=0,binding=0) buffer Img { vec4 px[]; };\n"
        << "const int WW=" << W << ", HH=" << H << ";\n"
        << "vec3 iResolution = vec3(float(WW),float(HH),1.0);\n"
        << "float iTime = " << iT << ";\nvec4 iMouse = vec4(0.0);\nint iFrame = 0;\n"
        << "float iTimeDelta = 0.016;\n"
        << shader_body(slurp(shader)) << "\n"
        << "void main(){\n"
        << "  ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
        << "  if (g.x>=WW || g.y>=HH) return;\n"
        << "  vec2 fragCoord = vec2(g.x, HH-1-g.y) + 0.5;\n"     // flip Y so it's upright
        << "  vec4 fragColor = vec4(0.0,0.0,0.0,1.0);\n"
        << "  mainImage(fragColor, fragCoord);\n"
        << "  px[g.y*WW + g.x] = vec4(clamp(fragColor.rgb,0.0,1.0), 1.0);\n}\n";

    std::vector<uint32_t> spirv;
    std::string err;
    if (!compile_compute(src.str(), spirv, err)) { std::printf("err %s\n", err.c_str()); return 1; }

    omni::gpu::VulkanCompute vk;
    if (!vk.init(&err)) { std::printf("err vulkan: %s\n", err.c_str()); return 1; }
    const auto raw = vk.run_raw(spirv, static_cast<size_t>(W) * H * 16, (W + 7) / 8, (H + 7) / 8, 1);
    if (!raw.ok) { std::printf("err render: %s\n", raw.error.c_str()); return 1; }

    // RGBA float -> 8-bit RGB -> PPM (P6).
    const float* f = reinterpret_cast<const float*>(raw.bytes.data());
    std::ofstream img(out, std::ios::binary);
    img << "P6\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < W * H; ++i)
        for (int c = 0; c < 3; ++c)
            img.put(static_cast<char>(std::clamp(f[i * 4 + c], 0.0f, 1.0f) * 255.0f + 0.5f));

    std::printf("ok %dx%d on %s\n", W, H, vk.device_name().c_str());
    return 0;
}
