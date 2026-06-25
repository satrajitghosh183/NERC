// omni/frontends/spirv.hpp — hand-written SPIR-V binary -> UIR lifter (from scratch).
// No SPIRV-Tools dependency: we parse the word stream per the SPIR-V spec.
#pragma once
#include "omni/uir/ir.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace omni::frontends {

struct SpirvLiftResult {
    bool ok = false;
    std::string error;
    // Diagnostics for the paper / debugging:
    uint32_t id_bound = 0;
    uint32_t handled_insts = 0;
    uint32_t skipped_insts = 0;   // metadata/decoration/unhandled (kept as Undef if result-producing)
    std::vector<std::string> unhandled_opcodes;
};

// Lift a SPIR-V module (raw words) into `out`. Returns result with ok/error.
SpirvLiftResult lift_spirv(const uint32_t* words, size_t word_count, omni::uir::Module& out);

// Convenience: read a .spv file from disk and lift it.
SpirvLiftResult lift_spirv_file(const std::string& path, omni::uir::Module& out);

} // namespace omni::frontends
