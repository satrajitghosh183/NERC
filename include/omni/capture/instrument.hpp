// omni/capture/instrument.hpp — the "instrument once" capture pass on UIR.
//
// Inserts a TraceTap after every result-producing instruction whose value is a
// capturable scalar/vector (selectable region-of-interest), so that running the
// instrumented module records each lane's intermediate values. Because the pass
// operates on UIR — not SPIR-V/DXIL/AIR directly — it is written once and lowers to
// every backend (PLAN.md §2.1, §4.4).
#pragma once
#include "omni/uir/ir.hpp"
#include <functional>
#include <vector>

namespace omni::capture {

struct TapSite {
    uint32_t id;            // sequential site id (== TraceTap.imm0)
    uir::ValueId value;     // captured SSA value
    uir::Op def_op;         // defining opcode (for reporting)
    uint32_t line;          // source line
};

// A value is capturable if it is a scalar/vector of int or float (not pointer/void/struct).
bool is_capturable(const uir::Module& m, uir::TypeId t);

// Insert taps into function `f`. `accept` selects which sites to instrument
// (default: all capturable). Returns the tap site table.
std::vector<TapSite> insert_taps(
    uir::Module& m, uir::FuncId f,
    const std::function<bool(const uir::Instruction&)>& accept = {});

} // namespace omni::capture
