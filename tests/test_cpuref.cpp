#include "omni/test.hpp"
#include "omni/frontends/spirv.hpp"
#include "omni/cpuref/interp.hpp"
#include "omni/trace/codec.hpp"
#include "omni/uir/ir.hpp"
#include <cstring>
#include <string>

using namespace omni;

static std::string spv_path() {
    return std::string(OMNI_SOURCE_DIR) + "/data/shaders/simple.frag.spv";
}

static uir::FuncId main_fn(const uir::Module& m) {
    for (size_t i = 0; i < m.num_functions(); ++i)
        if (m.function((uir::FuncId)i).name == "main") return (uir::FuncId)i;
    return uir::INVALID;
}

// Run simple.frag for one lane with given uv, k. outColor must equal the closed form
// (uv.x*k, uv.y*k, 0.5*k, 1.0).
TEST(cpuref, executes_real_shader_matches_closed_form) {
    uir::Module m;
    REQUIRE(frontends::lift_spirv_file(spv_path(), m).ok);
    uir::FuncId f = main_fn(m);
    REQUIRE(f != uir::INVALID);

    cpuref::Interp interp(m);
    cpuref::Cell* uv = interp.global("uv");
    cpuref::Cell* pc = interp.global("pc");
    REQUIRE(uv != nullptr);
    REQUIRE(pc != nullptr);
    REQUIRE(!pc->leaf);                 // push-constant struct { float k; }
    REQUIRE(pc->members.size() == 1);

    const float UVX = 0.3f, UVY = 0.7f, K = 2.0f;
    uv->val = cpuref::Val::vecf({UVX, UVY});
    pc->members[0]->val = cpuref::Val::scalar_f(K);

    std::string err;
    bool ok = interp.run(f, &err);
    if (!ok) std::printf("    interp error: %s\n", err.c_str());
    REQUIRE(ok);

    cpuref::Val out = interp.read_global_leaf("outColor");
    REQUIRE_EQ((int)out.n, 4);
    CHECK_NEAR(out.f[0], UVX * K, 1e-6);
    CHECK_NEAR(out.f[1], UVY * K, 1e-6);
    CHECK_NEAR(out.f[2], 0.5f * K, 1e-6);
    CHECK_NEAR(out.f[3], 1.0f, 1e-6);
}

// Drive a full warp of 32 lanes (uv.x = lane/32), capture outColor.x as a float trace
// column, and round-trip it through the codec. End-to-end: execute -> capture -> compress.
TEST(cpuref, warp_execution_feeds_codec_lossless) {
    uir::Module m;
    REQUIRE(frontends::lift_spirv_file(spv_path(), m).ok);
    uir::FuncId f = main_fn(m);

    const int W = 32;
    const float K = 1.5f;
    trace::Column col; col.kind = trace::ValueKind::F32; col.warp_size = W; col.bits.resize(W);
    std::vector<float> expected(W);

    for (int lane = 0; lane < W; ++lane) {
        cpuref::Interp interp(m);            // fresh lane state
        interp.global("uv")->val = cpuref::Val::vecf({(float)lane / W, 0.25f});
        interp.global("pc")->members[0]->val = cpuref::Val::scalar_f(K);
        REQUIRE(interp.run(f));
        cpuref::Val out = interp.read_global_leaf("outColor");
        expected[lane] = out.f[0];           // outColor.x = (lane/32)*K
        std::memcpy(&col.bits[lane], &out.f[0], 4);
    }

    // Sanity: lane 8 -> (8/32)*1.5 = 0.375
    CHECK_NEAR(expected[8], 0.375f, 1e-6);

    auto bytes = trace::encode(col);
    trace::Column dec = trace::decode(bytes.data(), bytes.size());
    REQUIRE_EQ(dec.size(), col.size());
    for (int lane = 0; lane < W; ++lane) {
        float got; std::memcpy(&got, &dec.bits[lane], 4);
        CHECK_NEAR(got, expected[lane], 0.0f); // bit-exact lossless
    }
    std::printf("    warp captured & round-tripped: %zu bytes for %d float values (%.2f bpv)\n",
                bytes.size(), W, bytes.size() * 8.0 / W);
}
