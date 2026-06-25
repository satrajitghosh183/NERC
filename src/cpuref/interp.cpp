#include "omni/cpuref/interp.hpp"
#include <cmath>
#include <cstring>

namespace omni::cpuref {
using namespace omni::uir;

std::unique_ptr<Cell> Interp::build_cell(TypeId pointee) {
    auto c = std::make_unique<Cell>();
    const Type& t = m_.type(pointee);
    if (t.kind == TypeKind::Struct) {
        c->leaf = false;
        for (TypeId mt : t.members) c->members.push_back(build_cell(mt));
    } else if (t.kind == TypeKind::Array || t.kind == TypeKind::RuntimeArray) {
        c->leaf = false;
        uint32_t cnt = t.kind == TypeKind::Array ? t.count : 0;
        for (uint32_t k = 0; k < cnt; ++k) c->members.push_back(build_cell(t.elem));
    } else {
        c->leaf = true;
        // initialise leaf shape from type
        if (t.kind == TypeKind::Vector) { c->val.n = (uint8_t)t.count; c->val.is_float = (m_.type(t.elem).kind == TypeKind::Float); }
        else if (t.kind == TypeKind::Float) { c->val.n = 1; c->val.is_float = true; }
        else { c->val.n = 1; c->val.is_float = false; }
    }
    return c;
}

void Interp::ensure_globals() {
    if (globals_ready_) return;
    globals_ready_ = true;
    for (size_t i = 0; i < m_.num_values(); ++i) {
        const Value& v = m_.value((ValueId)i);
        if (v.kind != ValueKind::GlobalVar) continue;
        const Type& pt = m_.type(v.type);             // pointer type
        TypeId pointee = pt.kind == TypeKind::Pointer ? pt.elem : v.type;
        auto cell = build_cell(pointee);
        Cell* raw = cell.get();
        arena_.push_back(std::move(cell));
        ptr_of_[(ValueId)i] = raw;
        if (!v.name.empty()) global_by_name_[v.name] = raw;
    }
}

Cell* Interp::global(const std::string& name) {
    ensure_globals();
    auto it = global_by_name_.find(name);
    return it == global_by_name_.end() ? nullptr : it->second;
}

Val Interp::read_global_leaf(const std::string& name) {
    Cell* c = global(name);
    return (c && c->leaf) ? c->val : Val{};
}

Val Interp::materialize_const(ValueId v) {
    const Value& val = m_.value(v);
    const Type& t = m_.type(val.type);
    switch (val.kind) {
        case ValueKind::ConstFloat: {
            float f;
            if (t.width == 64) { double d; std::memcpy(&d, &val.const_bits, 8); f = (float)d; }
            else { uint32_t b = (uint32_t)val.const_bits; std::memcpy(&f, &b, 4); }
            return Val::scalar_f(f);
        }
        case ValueKind::ConstInt:  return Val::scalar_i((int32_t)val.const_bits);
        case ValueKind::ConstBool: return Val::scalar_i(val.const_bits ? 1 : 0);
        case ValueKind::ConstComposite: {
            Val r; r.n = (uint8_t)val.elements.size();
            r.is_float = true;
            for (size_t k = 0; k < val.elements.size() && k < 4; ++k) {
                Val e = eval(val.elements[k]);
                r.is_float = e.is_float;
                r.f[k] = e.f[0]; r.i[k] = e.i[0];
            }
            return r;
        }
        case ValueKind::Undef: default: return Val{};
    }
}

Val Interp::eval(ValueId v) {
    if (v == INVALID) return Val{};
    auto it = ssa_.find(v);
    if (it != ssa_.end()) return it->second;
    const Value& val = m_.value(v);
    if (val.kind != ValueKind::InstResult && val.kind != ValueKind::Param)
        return materialize_const(v);
    return Val{}; // param without binding / not yet computed
}

static Val componentwise(const Val& a, const Val& b, float (*op)(float, float)) {
    Val r = a; r.n = a.n; for (int k = 0; k < a.n; ++k) r.f[k] = op(a.f[k], (b.n == 1 ? b.f[0] : b.f[k]));
    return r;
}

Val Interp::apply_binop(Op op, const Val& a, const Val& b) {
    switch (op) {
        case Op::FAdd: return componentwise(a, b, [](float x, float y){ return x + y; });
        case Op::FSub: return componentwise(a, b, [](float x, float y){ return x - y; });
        case Op::FMul: return componentwise(a, b, [](float x, float y){ return x * y; });
        case Op::FDiv: return componentwise(a, b, [](float x, float y){ return x / y; });
        case Op::VectorTimesScalar: { Val r = a; for (int k = 0; k < a.n; ++k) r.f[k] = a.f[k] * b.f[0]; return r; }
        case Op::IAdd: { Val r = a; for (int k = 0; k < a.n; ++k) r.i[k] = a.i[k] + b.i[k]; return r; }
        case Op::ISub: { Val r = a; for (int k = 0; k < a.n; ++k) r.i[k] = a.i[k] - b.i[k]; return r; }
        case Op::IMul: { Val r = a; for (int k = 0; k < a.n; ++k) r.i[k] = a.i[k] * b.i[k]; return r; }
        case Op::Dot: { float s = 0; for (int k = 0; k < a.n; ++k) s += a.f[k] * b.f[k]; return Val::scalar_f(s); }
        default: return Val{};
    }
}

bool Interp::run(FuncId f, std::string* err) {
    ensure_globals();
    const Function& fn = m_.function(f);
    auto fail = [&](const std::string& s){ if (err) *err = s; return false; };

    for (BlockId bid : fn.blocks) {
        const BasicBlock& b = m_.block(bid);
        for (InstId iid : b.insts) {
            const Instruction& in = m_.inst(iid);
            const ValueId* ops = m_.operands(in);
            switch (in.op) {
                case Op::Variable: {
                    const Type& pt = m_.type(in.type);
                    TypeId pointee = pt.kind == TypeKind::Pointer ? pt.elem : in.type;
                    auto cell = build_cell(pointee);
                    ptr_of_[in.result] = cell.get();
                    arena_.push_back(std::move(cell));
                    break;
                }
                case Op::Load: {
                    Cell* c = ptr_of_.count(ops[0]) ? ptr_of_[ops[0]] : nullptr;
                    if (!c || !c->leaf) return fail("Load from non-leaf/unknown pointer");
                    ssa_[in.result] = c->val;
                    break;
                }
                case Op::Store: {
                    Cell* c = ptr_of_.count(ops[0]) ? ptr_of_[ops[0]] : nullptr;
                    if (!c) return fail("Store to unknown pointer");
                    c->val = eval(ops[1]); c->leaf = true;
                    break;
                }
                case Op::AccessChain: {
                    Cell* c = ptr_of_.count(ops[0]) ? ptr_of_[ops[0]] : nullptr;
                    if (!c) return fail("AccessChain on unknown base");
                    for (uint32_t k = 1; k < in.operand_count; ++k) {
                        Val idx = eval(ops[k]);
                        uint32_t ix = (uint32_t)idx.i[0];
                        if (c->leaf || ix >= c->members.size()) return fail("AccessChain index out of range");
                        c = c->members[ix].get();
                    }
                    ptr_of_[in.result] = c;
                    break;
                }
                case Op::CompositeExtract: {
                    Val s = eval(ops[0]);
                    Val r; r.n = 1; r.is_float = s.is_float;
                    r.f[0] = s.f[in.imm0 < 4 ? in.imm0 : 0];
                    r.i[0] = s.i[in.imm0 < 4 ? in.imm0 : 0];
                    ssa_[in.result] = r;
                    break;
                }
                case Op::CompositeConstruct: {
                    Val r; r.n = 0; r.is_float = true; int comp = 0;
                    for (uint32_t k = 0; k < in.operand_count && comp < 4; ++k) {
                        Val e = eval(ops[k]);
                        for (int j = 0; j < e.n && comp < 4; ++j) { r.f[comp] = e.f[j]; r.i[comp] = e.i[j]; ++comp; }
                    }
                    r.n = (uint8_t)comp;
                    ssa_[in.result] = r;
                    break;
                }
                case Op::TraceTap: {
                    if (tap_sink && in.operand_count >= 1)
                        tap_sink->push_back({in.imm0, eval(ops[0])});
                    break;
                }
                case Op::Return: case Op::ReturnValue: return true;
                case Op::Branch: case Op::BranchConditional:
                    // v1 is straight-line; multi-block control flow handled once CFG lands.
                    break;
                default: {
                    if (in.operand_count >= 1) {
                        Val a = eval(ops[0]);
                        Val bb = in.operand_count >= 2 ? eval(ops[1]) : Val{};
                        Val r = apply_binop(in.op, a, bb);
                        if (in.result != INVALID) ssa_[in.result] = r;
                    }
                    break;
                }
            }
        }
    }
    return true;
}

} // namespace omni::cpuref
