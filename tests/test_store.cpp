#include "omni/test.hpp"
#include "omni/store/trace_store.hpp"
#include "omni/trace/synth_columns.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace omni;
namespace synth = omni::trace::synth;

static std::string tmp_path() {
    const char* d = std::getenv("OMNI_DATA_DIR");
    std::string dir = d ? d : "/tmp";
    return dir + "/omni_test_store.otr";
}

TEST(store, write_read_roundtrip_by_index_and_site) {
    std::vector<trace::Column> cols = {
        synth::constant(1000, 7),
        synth::warp_gradient(777),
        synth::quad_coherent(2048),
        synth::with_divergence(synth::random32(500, 9), 0.4, 3),
    };
    std::vector<uint32_t> site_ids = {10, 20, 30, 40};

    store::TraceWriter w;
    for (size_t i = 0; i < cols.size(); ++i) w.add(site_ids[i], cols[i]);
    std::string path = tmp_path();
    REQUIRE(w.write(path));

    store::TraceReader r;
    REQUIRE(r.open(path));
    REQUIRE_EQ(r.num_columns(), cols.size());

    // By index.
    for (size_t i = 0; i < cols.size(); ++i) {
        trace::Column got; uint32_t sid = 0;
        REQUIRE(r.column_by_index(i, got, &sid));
        CHECK_EQ(sid, site_ids[i]);
        REQUIRE_EQ(got.size(), cols[i].size());
        for (size_t k = 0; k < got.size(); ++k)
            if (cols[i].lane_active(k)) REQUIRE_EQ(got.bits[k], cols[i].bits[k]);
    }

    // By site id (out of order).
    trace::Column c30;
    REQUIRE(r.column_by_site(30, c30));
    REQUIRE_EQ(c30.size(), cols[2].size());
    for (size_t k = 0; k < c30.size(); ++k) REQUIRE_EQ(c30.bits[k], cols[2].bits[k]);

    // Missing site fails cleanly.
    trace::Column none;
    CHECK(!r.column_by_site(999, none));

    r.close();
    std::remove(path.c_str());
}

TEST(store, empty_and_reopen) {
    store::TraceWriter w;                 // no columns
    std::string path = tmp_path();
    REQUIRE(w.write(path));
    store::TraceReader r;
    REQUIRE(r.open(path));
    CHECK_EQ(r.num_columns(), 0u);
    r.close();
    std::remove(path.c_str());
}
