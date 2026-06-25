#include "omni/test.hpp"
#include "omni/synth/renderer.hpp"
#include "omni/reward/oracle.hpp"
#include <cstdio>

using namespace omni;

#ifdef OMNI_HAVE_VULKAN
TEST(renderer, renders_on_gpu_and_visual_match) {
    synth::Renderer rr;
    std::string err;
    if (!rr.init(&err)) { std::printf("    renderer init failed (%s); skipping\n", err.c_str()); return; }
    std::printf("    rendering on %s\n", rr.device_name().c_str());

    const int W = 32, H = 32;
    auto grad = rr.render("vec3(uv, 0.5)", W, H);
    if (!grad.ok) std::printf("    render error: %s\n", grad.error.c_str());
    REQUIRE(grad.ok);
    REQUIRE_EQ((int)grad.image.rgba.size(), W * H * 4);

    // Pixel (16,16): uv = (16.5/32, 16.5/32) = 0.515625, blue = 0.5, alpha = 1.
    const float* p = grad.image.px(16, 16);
    CHECK_NEAR(p[0], 16.5f / 32.0f, 1e-3);   // r == uv.x
    CHECK_NEAR(p[1], 16.5f / 32.0f, 1e-3);   // g == uv.y
    CHECK_NEAR(p[2], 0.5f, 1e-3);            // b
    CHECK_NEAR(p[3], 1.0f, 1e-3);            // a

    // Solid red, and the visual_match reward term over real GPU renders.
    auto red = rr.render("vec3(1.0, 0.0, 0.0)", W, H);
    REQUIRE(red.ok);
    double self = reward::image_similarity(red.image.rgba, red.image.rgba);
    double cross = reward::image_similarity(red.image.rgba, grad.image.rgba);
    std::printf("    visual_match: red-vs-red=%.4f red-vs-gradient=%.4f\n", self, cross);
    CHECK_NEAR(self, 1.0, 1e-6);             // identical renders -> perfect match
    CHECK(cross < self);                     // different renders -> lower match
}
#else
TEST(renderer, disabled) { std::printf("    built without Vulkan; renderer compiled out\n"); }
#endif
