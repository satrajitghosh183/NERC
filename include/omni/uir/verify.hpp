// omni/uir/verify.hpp — structural validity checks for a UIR module.
#pragma once
#include "omni/uir/ir.hpp"
#include <string>
#include <vector>

namespace omni::uir {

bool is_terminator(Op op);

// Returns true if the module is structurally valid; appends human-readable
// problems to `errors` otherwise.
bool verify(const Module& m, std::vector<std::string>& errors);

} // namespace omni::uir
