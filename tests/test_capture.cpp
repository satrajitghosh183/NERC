#include "omni/test.hpp"
#include "omni/frontends/spirv.hpp"
#include "omni/capture/instrument.hpp"
#include "omni/cpuref/interp.hpp"
#include "omni/trace/codec.hpp"
#include "omni/uir/verify.hpp"
#include <cstring>
#include <string>
#include <vector>

using namespace omni;

static std::string spv_path() { return std::string(OMNI_SOURCE_DIR) + "/data/shaders/simple.frag.spv"; }
static uir::FuncId main_fn(const uir::Module& m) {
    for (size_t i = 0; i < m.num_functions(); ++i)
        if (m.function((uir::FuncId)i).name == "main") return (uir::FuncId)i;
    return uir::INVALID;
}
static int count_op(const uir::Module& m, uir::Op op) {
    int n = 0; for (size_t i = 0; i < m.num_insts(); ++i) if (m.inst((uir::InstId)i).op == op) ++n; return n;
}

TEST(capture, inserts_taps_and_stays_valid) {
    uir::Module m;
    REQUIRE(frontends::lift_spirv_file(spv_path(), m).ok);
    uir::FuncId f = main_fn(m);

    int taps_before = count_op(m, uir::Op::TraceTap);
    CHECK_EQ(taps_before, 0);

    auto sites = capture::insert_taps(m, f);
    std::printf("    inserted %zu tap sites\n", sites.size());
    CHECK(!sites.empty());

    // One TraceTap instruction per site, and the module is still structurally valid.
    CHECK_EQ((size_t)count_op(m, uir::Op::TraceTap), sites.size());
    std::vector<std::string> errs;
    bool ok = verify(m, errs);
    for (auto& e : errs) std::printf("    verify: %s\n", e.c_str());
    CHECK(ok);

    // Idempotent: re-running the pass adds no new taps (it skips existing TraceTaps).
    auto sites2 = capture::insert_taps(m, f);
    CHECK(sites2.empty());
}

TEST(capture, end_to_end_warp_capture_compress) {
    uir::Module m;
    REQUIRE(frontends::lift_spirv_file(spv_path(), m).ok);
    uir::FuncId f = main_fn(m);
    auto sites = capture::insert_taps(m, f);
    REQUIRE(!sites.empty());

    const int W = 32;
    const float K = 1.25f;
    std::vector<trace::Column> cols(sites.size());
    for (auto& c : cols) { c.kind = trace::ValueKind::F32; c.warp_size = W; c.bits.assign(W, 0); }

    // Drive a warp; each lane records all tapped values via the sink.
    for (int lane = 0; lane < W; ++lane) {
        cpuref::Interp interp(m);
        interp.global("uv")->val = cpuref::Val::vecf({(float)lane / W, 0.5f});
        interp.global("pc")->members[0]->val = cpuref::Val::scalar_f(K);
        std::vector<std::pair<uint32_t, cpuref::Val>> sink;
        interp.tap_sink = &sink;
        REQUIRE(interp.run(f));
        CHECK_EQ(sink.size(), sites.size());          // each straight-line tap fires once
        for (auto& [site, val] : sink) {
            if (site < cols.size()) std::memcpy(&cols[site].bits[lane], &val.f[0], 4);
        }
    }

    // Every per-site column compresses losslessly.
    size_t total_bytes = 0, total_vals = 0;
    for (auto& c : cols) {
        auto bytes = trace::encode(c);
        trace::Column d = trace::decode(bytes.data(), bytes.size());
        REQUIRE_EQ(d.size(), c.size());
        for (size_t i = 0; i < c.size(); ++i) REQUIRE_EQ(d.bits[i], c.bits[i]);
        total_bytes += bytes.size(); total_vals += c.size();
    }
    std::printf("    captured %zu sites x %d lanes = %zu values in %zu bytes (%.2f bpv)\n",
                sites.size(), W, total_vals, total_bytes, total_bytes * 8.0 / total_vals);
}
