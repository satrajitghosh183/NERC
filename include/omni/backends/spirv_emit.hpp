// omni/backends/spirv_emit.hpp — UIR -> SPIR-V binary emitter (from scratch).
// The reverse of the frontend: lowers UIR back to a valid SPIR-V module. Together with
// the frontend this closes the universal-IR round-trip (PLAN.md §4.3).
#pragma once
#include "omni/uir/ir.hpp"
#include <cstdint>
#include <vector>
#include <string>

namespace omni::backends {

enum class ExecModel : uint32_t { Fragment = 4, GLCompute = 5 };

struct EmitOptions {
    omni::uir::FuncId entry = 0;
    std::string entry_name = "main";
    ExecModel model = ExecModel::GLCompute;
    uint32_t local_size[3] = {1, 1, 1};   // for GLCompute
};

// Emit a complete, validatable SPIR-V module (header + capability + memory model +
// entry point + types/constants + function bodies).
std::vector<uint32_t> emit_spirv(const omni::uir::Module& m, const EmitOptions& opt);

} // namespace omni::backends
