#include "omni/test.hpp"
#include "omni/frontends/spirv.hpp"
#include "omni/capture/instrument.hpp"
#include "omni/cpuref/interp.hpp"
#include "omni/store/trace_store.hpp"
#include "omni/timetravel/timetravel.hpp"
#include <cstdio>
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
static std::string store_path() {
    const char* d = std::getenv("OMNI_DATA_DIR"); std::string dir = d ? d : "/tmp";
    return dir + "/omni_timetravel.otr";
}

TEST(timetravel, reconstruct_invocation_history) {
    // Lift + instrument.
    uir::Module m;
    REQUIRE(frontends::lift_spirv_file(spv_path(), m).ok);
    uir::FuncId f = main_fn(m);
    auto sites = capture::insert_taps(m, f);
    REQUIRE(!sites.empty());

    // Capture a warp of 32 invocations into per-site columns.
    const int W = 32; const float K = 2.0f;
    std::vector<trace::Column> cols(sites.size());
    for (auto& c : cols) { c.kind = trace::ValueKind::F32; c.warp_size = W; c.bits.assign(W, 0); }
    for (int lane = 0; lane < W; ++lane) {
        cpuref::Interp interp(m);
        interp.global("uv")->val = cpuref::Val::vecf({(float)lane / W, 0.5f});
        interp.global("pc")->members[0]->val = cpuref::Val::scalar_f(K);
        std::vector<std::pair<uint32_t, cpuref::Val>> sink;
        interp.tap_sink = &sink;
        REQUIRE(interp.run(f));
        for (auto& [site, val] : sink)
            if (site < cols.size()) std::memcpy(&cols[site].bits[lane], &val.f[0], 4);
    }

    // Persist to the columnar store.
    store::TraceWriter w;
    for (size_t i = 0; i < sites.size(); ++i) w.add(sites[i].id, cols[i]);
    std::string path = store_path();
    REQUIRE(w.write(path));
    store::TraceReader r;
    REQUIRE(r.open(path));

    // Reconstruct invocation 8's full history.
    auto tl8 = timetravel::Timeline::reconstruct(r, sites, 8);
    REQUIRE_EQ(tl8.size(), sites.size());

    // Steps are in program order (ascending site id).
    for (size_t i = 1; i < tl8.size(); ++i)
        CHECK(tl8.at(i).site_id > tl8.at(i - 1).site_id);

    // Reconstruction matches the underlying captured columns exactly.
    for (size_t i = 0; i < sites.size(); ++i)
        CHECK_EQ(tl8.at(i).value_bits, cols[i].bits[8]);

    // Forward/backward navigation.
    CHECK_EQ(tl8.cursor(), 0u);
    size_t steps = 0; while (tl8.step_forward()) ++steps;
    CHECK_EQ(steps, sites.size() - 1);
    CHECK_EQ(tl8.cursor(), sites.size() - 1);
    CHECK(tl8.step_back());
    tl8.seek(0);
    CHECK_EQ(tl8.cursor(), 0u);

    // Different invocations have different histories (uv.x varies across lanes).
    auto tl16 = timetravel::Timeline::reconstruct(r, sites, 16);
    bool differs = false;
    for (size_t i = 0; i < tl8.size(); ++i)
        if (tl8.at(i).value_bits != tl16.at(i).value_bits) differs = true;
    CHECK(differs);

    std::printf("    reconstructed %zu-step history for invocations 8 and 16 (differ=%d)\n",
                tl8.size(), (int)differs);
    r.close();
    std::remove(path.c_str());
}
