#include "omni/test.hpp"
#include "omni/version.hpp"
#include <string>

TEST(smoke, version_present) {
    std::string v = omni::version();
    CHECK(!v.empty());
    CHECK(v.find("OmniTrace") != std::string::npos);
}

TEST(smoke, arithmetic_sanity) {
    CHECK_EQ(2 + 2, 4);
    CHECK_NEAR(0.1 + 0.2, 0.3, 1e-9);
}
