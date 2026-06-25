// omni/gpu/vulkan_capture.hpp — real on-GPU value capture via Vulkan (MoltenVK on macOS).
//
// Runs a compute shader on the actual GPU; the shader writes one value per invocation to
// a storage buffer (this IS a lowered TraceTap), which we read back and feed to the
// divergence-aware codec. This is the runtime side of the capture spearhead — real
// hardware values, not the CPU reference (PLAN.md §4.7, §11). Vulkan handles are kept
// opaque so vulkan.h does not leak into the rest of the project.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace omni::gpu {

struct CaptureResult {
    bool ok = false;
    std::string error;
    std::vector<uint32_t> values;   // per-invocation captured words read back from the GPU
};

class VulkanCompute {
public:
    ~VulkanCompute();
    bool init(std::string* err = nullptr);
    void shutdown();
    bool ready() const { return inited_; }
    const std::string& device_name() const { return device_name_; }

    // Run `spirv` (a compute shader) with `groups_x` workgroups. The shader must write
    // `count` uint32 to an std430 storage buffer at set=0 binding=0. Values come back in
    // result.values.
    CaptureResult run(const std::vector<uint32_t>& spirv, uint32_t count, uint32_t groups_x);

    // General form: dispatch (gx,gy,gz) groups over an std430 SSBO of `byte_size` bytes
    // at set=0 binding=0; returns the raw buffer bytes (e.g. an RGBA float image).
    struct RawResult { bool ok = false; std::string error; std::vector<uint8_t> bytes; };
    RawResult run_raw(const std::vector<uint32_t>& spirv, size_t byte_size,
                      uint32_t gx, uint32_t gy, uint32_t gz);

private:
    bool inited_ = false;
    std::string device_name_;
    // Opaque Vulkan handles (cast in the .cpp).
    void* instance_ = nullptr;
    void* phys_ = nullptr;
    void* device_ = nullptr;
    void* queue_ = nullptr;
    uint32_t queue_family_ = 0;
};

} // namespace omni::gpu
