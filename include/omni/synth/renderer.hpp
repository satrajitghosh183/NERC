// omni/synth/renderer.hpp — GPU offscreen renderer (C++ replacement for renderer_moderngl).
//
// Renders a Shadertoy-style per-pixel color expression by running it as a compute shader
// (one invocation per pixel) on the real GPU, reading back an RGBA float image. Produces
// the target/candidate renders for the visual_match reward term (PLAN.md §13 synth/renderer).
#pragma once
#include "omni/gpu/vulkan_capture.hpp"
#include <string>
#include <vector>

namespace omni::synth {

struct Image {
    int w = 0, h = 0;
    std::vector<float> rgba;                 // w*h*4, row-major
    const float* px(int x, int y) const { return &rgba[(size_t)(y * w + x) * 4]; }
};

struct RenderResult { bool ok = false; std::string error; Image image; };

class Renderer {
public:
    bool init(std::string* err = nullptr);
    bool ready() const { return vk_.ready(); }
    const std::string& device_name() const { return vk_.device_name(); }

    // `color_expr` is a GLSL vec3 expression in terms of `uv` (0..1), `fragCoord`,
    // and `iResolution`, e.g. "vec3(uv, 0.5)".
    RenderResult render(const std::string& color_expr, int w, int h);

private:
    gpu::VulkanCompute vk_;
};

} // namespace omni::synth
