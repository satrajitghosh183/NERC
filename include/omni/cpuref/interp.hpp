// omni/cpuref/interp.hpp — bit-exact CPU reference interpreter for UIR.
//
// Executes a UIR function for a single invocation (lane). Driving it across a warp
// of lanes with per-lane inputs gives a hardware-independent golden reference: diff
// it against captured GPU values to localize driver/fast-math/IEEE-754 bugs, and use
// it as the dense reward signal for shader synthesis (PLAN.md §4.5, §11 item 3).
//
// Scope (v1): straight-line + structured single-function fragment/compute logic with a
// pointer/memory model (Variable/Load/Store/AccessChain over scalars, vectors, structs).
#pragma once
#include "omni/uir/ir.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace omni::cpuref {

// A runtime value: up to 4 components, float or int (matches shader scalar/vector types).
struct Val {
    bool is_float = true;
    uint8_t n = 1;
    float f[4] = {0, 0, 0, 0};
    int32_t i[4] = {0, 0, 0, 0};

    static Val scalar_f(float x) { Val v; v.is_float = true; v.n = 1; v.f[0] = x; return v; }
    static Val scalar_i(int32_t x) { Val v; v.is_float = false; v.n = 1; v.i[0] = x; return v; }
    static Val vecf(std::initializer_list<float> xs) {
        Val v; v.is_float = true; v.n = (uint8_t)xs.size(); int k = 0; for (float x : xs) v.f[k++] = x; return v;
    }
};

// Hierarchical storage cell (leaf value, or struct/array of sub-cells) for the memory model.
struct Cell {
    bool leaf = true;
    Val val;
    std::vector<std::unique_ptr<Cell>> members;
};

enum class FastMath { StrictIEEE, FastContract };

class Interp {
public:
    explicit Interp(const uir::Module& m, FastMath fm = FastMath::StrictIEEE) : m_(m), fm_(fm) {}

    // Find a module-scope global variable's storage cell by debug name (e.g., "uv", "pc").
    // Cells for all globals are materialised on first call. Returns nullptr if unknown.
    Cell* global(const std::string& name);

    // Execute function `f` for this lane. Returns false on an unsupported instruction.
    bool run(uir::FuncId f, std::string* err = nullptr);

    // Read a global's leaf value (e.g., the output color) after run().
    Val read_global_leaf(const std::string& name);

    // Optional sink: if set, every executed TraceTap pushes {site_id (imm0), captured value}.
    // This is the CPU-side realisation of instrumentation capture.
    std::vector<std::pair<uint32_t, Val>>* tap_sink = nullptr;

private:
    void ensure_globals();
    std::unique_ptr<Cell> build_cell(uir::TypeId pointee);
    Val eval(uir::ValueId v);
    Val materialize_const(uir::ValueId v);
    Val apply_binop(uir::Op op, const Val& a, const Val& b);

    const uir::Module& m_;
    FastMath fm_;
    bool globals_ready_ = false;
    std::vector<std::unique_ptr<Cell>> arena_;
    std::unordered_map<uir::ValueId, Cell*> ptr_of_;   // pointer value -> cell
    std::unordered_map<std::string, Cell*> global_by_name_;
    std::unordered_map<uir::ValueId, Val> ssa_;        // computed SSA results
};

} // namespace omni::cpuref
