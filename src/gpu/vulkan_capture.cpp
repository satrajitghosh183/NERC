#include "omni/gpu/vulkan_capture.hpp"
#include <vulkan/vulkan.h>
#include <cstdlib>
#include <cstring>

namespace omni::gpu {

#define VKH(handle) reinterpret_cast<handle>
static VkInstance       I(void* p) { return reinterpret_cast<VkInstance>(p); }
static VkPhysicalDevice P(void* p) { return reinterpret_cast<VkPhysicalDevice>(p); }
static VkDevice         D(void* p) { return reinterpret_cast<VkDevice>(p); }
static VkQueue          Q(void* p) { return reinterpret_cast<VkQueue>(p); }

VulkanCompute::~VulkanCompute() { shutdown(); }

bool VulkanCompute::init(std::string* err) {
    auto fail = [&](const char* s) { if (err) *err = s; shutdown(); return false; };

    // Point the loader at the MoltenVK ICD (so this works without external env setup).
#ifdef OMNI_VULKAN_SDK
    if (!std::getenv("VK_ICD_FILENAMES"))
        setenv("VK_ICD_FILENAMES", OMNI_VULKAN_SDK "/share/vulkan/icd.d/MoltenVK_icd.json", 1);
#endif

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "OmniTrace"; app.apiVersion = VK_API_VERSION_1_1;

    const char* inst_exts[] = {
        "VK_KHR_portability_enumeration",
        "VK_KHR_get_physical_device_properties2",
    };
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.flags = 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2; ici.ppEnabledExtensionNames = inst_exts;
    VkInstance inst;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return fail("vkCreateInstance failed");
    instance_ = inst;

    uint32_t ndev = 0; vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    if (ndev == 0) return fail("no Vulkan physical devices");
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice pd = devs[0];
    phys_ = pd;
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    device_name_ = props.deviceName;

    // Find a compute-capable queue family.
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < nq; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queue_family_ = i; found = true; break; }
    if (!found) return fail("no compute queue family");

    // Enable VK_KHR_portability_subset if present (required by MoltenVK).
    uint32_t next = 0; vkEnumerateDeviceExtensionProperties(pd, nullptr, &next, nullptr);
    std::vector<VkExtensionProperties> exts(next);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &next, exts.data());
    std::vector<const char*> dev_exts;
    for (auto& e : exts) if (std::strcmp(e.extensionName, "VK_KHR_portability_subset") == 0)
        dev_exts.push_back("VK_KHR_portability_subset");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)dev_exts.size();
    dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) return fail("vkCreateDevice failed");
    device_ = dev;
    VkQueue q; vkGetDeviceQueue(dev, queue_family_, 0, &q); queue_ = q;

    inited_ = true;
    return true;
}

static uint32_t find_mem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

VulkanCompute::RawResult VulkanCompute::run_raw(const std::vector<uint32_t>& spirv, size_t byte_size,
                                                uint32_t gx, uint32_t gy, uint32_t gz) {
    RawResult r;
    if (!inited_) { r.error = "not initialised"; return r; }
    VkDevice dev = D(device_);
    auto fail = [&](const char* s) { r.error = s; return r; };

    // Host-visible storage buffer for the captured values.
    VkDeviceSize bytes = (VkDeviceSize)byte_size;
    VkBufferCreateInfo bci{}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf; if (vkCreateBuffer(dev, &bci, nullptr, &buf) != VK_SUCCESS) return fail("create buffer");
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, buf, &req);
    uint32_t mt = find_mem(P(phys_), req.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) return fail("no host-visible memory");
    VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size; mai.memoryTypeIndex = mt;
    VkDeviceMemory mem; if (vkAllocateMemory(dev, &mai, nullptr, &mem) != VK_SUCCESS) return fail("alloc mem");
    vkBindBufferMemory(dev, buf, mem, 0);

    // Shader module + descriptor set (binding 0 = storage buffer).
    VkShaderModuleCreateInfo smci{}; smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * 4; smci.pCode = spirv.data();
    VkShaderModule sm; if (vkCreateShaderModule(dev, &smci, nullptr, &sm) != VK_SUCCESS) return fail("shader module");

    VkDescriptorSetLayoutBinding b0{}; b0.binding = 0;
    b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b0.descriptorCount = 1;
    b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{}; dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 1; dlci.pBindings = &b0;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &dsl);

    VkPipelineLayoutCreateInfo plci{}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

    VkComputePipelineCreateInfo cpci{}; cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.layout = pl;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
    VkPipeline pipe; if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe) != VK_SUCCESS) return fail("pipeline");

    VkDescriptorPoolSize ps{}; ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{}; dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
    VkDescriptorSetAllocateInfo dsai{}; dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev, &dsai, &ds);
    VkDescriptorBufferInfo dbi{}; dbi.buffer = buf; dbi.range = bytes;
    VkWriteDescriptorSet wds{}; wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = ds; wds.dstBinding = 0; wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

    // Record + submit.
    VkCommandPoolCreateInfo cpc{}; cpc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpc.queueFamilyIndex = queue_family_;
    VkCommandPool cmdpool; vkCreateCommandPool(dev, &cpc, nullptr, &cmdpool);
    VkCommandBufferAllocateInfo cbai{}; cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmdpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{}; cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, gz);
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(Q(queue_), 1, &si, fence) != VK_SUCCESS) return fail("submit");
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);

    // Read back.
    void* mapped = nullptr;
    vkMapMemory(dev, mem, 0, bytes, 0, &mapped);
    r.bytes.resize(byte_size);
    std::memcpy(r.bytes.data(), mapped, byte_size);
    vkUnmapMemory(dev, mem);
    r.ok = true;

    // Cleanup.
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cmdpool, nullptr);
    vkDestroyDescriptorPool(dev, dp, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkFreeMemory(dev, mem, nullptr);
    vkDestroyBuffer(dev, buf, nullptr);
    return r;
}

CaptureResult VulkanCompute::run(const std::vector<uint32_t>& spirv, uint32_t count, uint32_t groups_x) {
    CaptureResult r;
    RawResult raw = run_raw(spirv, (size_t)count * 4, groups_x, 1, 1);
    r.ok = raw.ok; r.error = raw.error;
    if (raw.ok) { r.values.resize(count); std::memcpy(r.values.data(), raw.bytes.data(), (size_t)count * 4); }
    return r;
}

void VulkanCompute::shutdown() {
    if (device_) { vkDeviceWaitIdle(D(device_)); vkDestroyDevice(D(device_), nullptr); device_ = nullptr; }
    if (instance_) { vkDestroyInstance(I(instance_), nullptr); instance_ = nullptr; }
    inited_ = false;
}

} // namespace omni::gpu
