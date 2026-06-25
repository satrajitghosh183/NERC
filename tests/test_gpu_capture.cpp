#include "omni/test.hpp"
#include "omni/gpu/vulkan_capture.hpp"
#include "omni/trace/codec.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/stat.h>

using namespace omni;

#ifdef OMNI_HAVE_VULKAN
static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(), &s) == 0; }
static std::string glslang() {
    if (const char* e = std::getenv("OMNI_GLSLANG"); e && exists(e)) return e;
    std::string p = "/Users/satrajitghosh/Software/VulkanSDK/1.4.328.1/macOS/bin/glslangValidator";
    return exists(p) ? p : "";
}

// Compute shader: each invocation writes a warp-coherent value (groups of 8 share a value)
// so the captured column compresses. This SSBO write IS a lowered TraceTap.
static const char* COMP = R"(#version 450
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) buffer Out { uint data[]; };
void main(){ uint i = gl_GlobalInvocationID.x; data[i] = i / 8u; }
)";

TEST(gpu, real_gpu_capture_then_compress) {
    std::string gl = glslang();
    if (gl.empty()) { std::printf("    glslangValidator unavailable; skipping\n"); return; }

    // Compile the compute shader to SPIR-V.
    const char* td = std::getenv("TMPDIR"); std::string dir = td ? td : "/tmp";
    std::string cpath = dir + "/omni_cap.comp", spath = dir + "/omni_cap.comp.spv";
    if (FILE* f = std::fopen(cpath.c_str(), "wb")) { std::fwrite(COMP, 1, std::strlen(COMP), f); std::fclose(f); }
    std::string cmd = "'" + gl + "' -V '" + cpath + "' -o '" + spath + "' 2>&1";
    if (std::system(cmd.c_str()) != 0 || !exists(spath)) { std::printf("    shader compile failed; skipping\n"); return; }
    std::vector<uint32_t> spirv;
    if (FILE* f = std::fopen(spath.c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
        spirv.resize(n / 4); std::fread(spirv.data(), 4, spirv.size(), f); std::fclose(f);
    }
    REQUIRE(!spirv.empty());

    gpu::VulkanCompute vk;
    std::string err;
    if (!vk.init(&err)) { std::printf("    Vulkan init failed (%s); skipping\n", err.c_str()); return; }
    std::printf("    GPU device: %s\n", vk.device_name().c_str());

    const uint32_t local = 64, groups = 16, count = local * groups; // 1024 invocations
    auto res = vk.run(spirv, count, groups);
    if (!res.ok) std::printf("    run error: %s\n", res.error.c_str());
    REQUIRE(res.ok);
    REQUIRE_EQ(res.values.size(), (size_t)count);

    // The GPU computed exactly what the shader specifies: data[i] == i/8.
    for (uint32_t i = 0; i < count; ++i) REQUIRE_EQ(res.values[i], i / 8u);

    // Feed the REAL GPU capture into the divergence-aware codec.
    trace::Column col; col.kind = trace::ValueKind::U32; col.warp_size = 32; col.bits = res.values;
    auto bytes = trace::encode(col);
    trace::Column dec = trace::decode(bytes.data(), bytes.size());
    REQUIRE_EQ(dec.size(), col.size());
    for (uint32_t i = 0; i < count; ++i) REQUIRE_EQ(dec.bits[i], col.bits[i]);  // lossless

    double bpv = bytes.size() * 8.0 / count;
    std::printf("    captured %u GPU values, compressed to %.3f bits/value (%.1fx), lossless\n",
                count, bpv, 32.0 / bpv);
    CHECK(bpv < 32.0);
    std::remove(cpath.c_str()); std::remove(spath.c_str());
}
#else
TEST(gpu, disabled) { std::printf("    built without Vulkan; GPU capture compiled out\n"); }
#endif
