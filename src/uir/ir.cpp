#include "omni/uir/ir.hpp"
#include <cstring>

namespace omni::uir {

// ---- op_name table ----------------------------------------------------------
const char* op_name(Op op) {
    switch (op) {
#define X(o) case Op::o: return #o;
        X(Nop) X(Undef) X(Variable) X(Load) X(Store) X(AccessChain)
        X(IAdd) X(ISub) X(IMul) X(SDiv) X(UDiv) X(SMod) X(UMod) X(SNegate)
        X(FAdd) X(FSub) X(FMul) X(FDiv) X(FNegate) X(FRem)
        X(IEqual) X(INotEqual) X(SLessThan) X(SGreaterThan) X(SLessThanEqual)
        X(SGreaterThanEqual) X(ULessThan) X(UGreaterThan) X(ULessThanEqual) X(UGreaterThanEqual)
        X(FOrdEqual) X(FOrdNotEqual) X(FOrdLessThan) X(FOrdGreaterThan)
        X(FOrdLessThanEqual) X(FOrdGreaterThanEqual)
        X(LogicalAnd) X(LogicalOr) X(LogicalNot) X(LogicalEqual)
        X(BitwiseAnd) X(BitwiseOr) X(BitwiseXor) X(Not)
        X(ShiftLeftLogical) X(ShiftRightLogical) X(ShiftRightArithmetic)
        X(ConvertSToF) X(ConvertUToF) X(ConvertFToS) X(ConvertFToU) X(Bitcast)
        X(CompositeConstruct) X(CompositeExtract) X(CompositeInsert)
        X(VectorShuffle) X(VectorTimesScalar) X(Select)
        X(Dot) X(Cross) X(Normalize) X(Length) X(Distance) X(Reflect)
        X(FMin) X(FMax) X(FClamp) X(FMix) X(Pow) X(Exp) X(Exp2) X(Log) X(Log2)
        X(Sin) X(Cos) X(Tan) X(Sqrt) X(InverseSqrt) X(FAbs) X(Floor) X(Ceil)
        X(Fract) X(Step) X(SmoothStep)
        X(DPdx) X(DPdy) X(Fwidth)
        X(ImageSampleImplicitLod) X(ImageSampleExplicitLod)
        X(Phi) X(Branch) X(BranchConditional) X(Switch)
        X(LoopMerge) X(SelectionMerge) X(Return) X(ReturnValue) X(Kill) X(Unreachable)
        X(FunctionCall) X(TraceTap)
#undef X
        default: return "?";
    }
}

// ---- type dedup -------------------------------------------------------------
static std::string type_key(const Type& t) {
    std::string k;
    k.reserve(32);
    k += char('A' + (int)t.kind);
    auto app = [&](uint64_t v) { k += ':'; for (int i = 0; i < 8; ++i) { k += char('0' + (v & 7)); v >>= 3; } };
    app(t.width); app(t.is_signed ? 1 : 0); app(t.elem); app(t.count);
    app(t.storage_class); app(t.ret);
    for (TypeId m : t.members) app(m);
    return k;
}

TypeId Module::add_type(const Type& t) {
    std::string k = type_key(t);
    auto it = type_dedup_.find(k);
    if (it != type_dedup_.end()) return it->second;
    TypeId id = (TypeId)types_.size();
    types_.push_back(t);
    type_dedup_.emplace(std::move(k), id);
    return id;
}

TypeId Module::void_type()  { Type t; t.kind = TypeKind::Void; return add_type(t); }
TypeId Module::bool_type()  { Type t; t.kind = TypeKind::Bool; return add_type(t); }
TypeId Module::int_type(uint16_t w, bool s)   { Type t; t.kind = TypeKind::Int;   t.width = w; t.is_signed = s; return add_type(t); }
TypeId Module::float_type(uint16_t w)         { Type t; t.kind = TypeKind::Float; t.width = w; return add_type(t); }
TypeId Module::vector_type(TypeId e, uint32_t n) { Type t; t.kind = TypeKind::Vector; t.elem = e; t.count = n; return add_type(t); }
TypeId Module::array_type(TypeId e, uint32_t n)  { Type t; t.kind = TypeKind::Array;  t.elem = e; t.count = n; return add_type(t); }
TypeId Module::pointer_type(TypeId p, uint32_t sc) { Type t; t.kind = TypeKind::Pointer; t.elem = p; t.storage_class = sc; return add_type(t); }
TypeId Module::function_type(TypeId ret, std::vector<TypeId> params) {
    Type t; t.kind = TypeKind::Function; t.ret = ret; t.members = std::move(params); return add_type(t);
}
TypeId Module::struct_type(std::vector<TypeId> members) {
    Type t; t.kind = TypeKind::Struct; t.members = std::move(members); return add_type(t);
}

// ---- values / constants -----------------------------------------------------
ValueId Module::new_value(Value v) {
    ValueId id = (ValueId)values_.size();
    values_.push_back(std::move(v));
    return id;
}

ValueId Module::const_int(TypeId t, uint64_t bits) {
    std::string k = "ci:" + std::to_string(t) + ":" + std::to_string(bits);
    auto it = const_dedup_.find(k);
    if (it != const_dedup_.end()) return it->second;
    Value v; v.kind = ValueKind::ConstInt; v.type = t; v.const_bits = bits;
    ValueId id = new_value(std::move(v));
    const_dedup_[k] = id;
    return id;
}

ValueId Module::const_float(TypeId t, double val) {
    uint64_t bits;
    uint16_t w = types_[t].width;
    if (w == 64) { std::memcpy(&bits, &val, 8); }
    else { float f = (float)val; uint32_t b32; std::memcpy(&b32, &f, 4); bits = b32; } // 16 stored as 32 for now
    std::string k = "cf:" + std::to_string(t) + ":" + std::to_string(bits);
    auto it = const_dedup_.find(k);
    if (it != const_dedup_.end()) return it->second;
    Value v; v.kind = ValueKind::ConstFloat; v.type = t; v.const_bits = bits;
    ValueId id = new_value(std::move(v));
    const_dedup_[k] = id;
    return id;
}

ValueId Module::const_bool(bool b) {
    TypeId t = bool_type();
    std::string k = "cb:" + std::to_string(b);
    auto it = const_dedup_.find(k);
    if (it != const_dedup_.end()) return it->second;
    Value v; v.kind = ValueKind::ConstBool; v.type = t; v.const_bits = b ? 1 : 0;
    ValueId id = new_value(std::move(v));
    const_dedup_[k] = id;
    return id;
}

ValueId Module::const_composite(TypeId t, std::vector<ValueId> elems) {
    Value v; v.kind = ValueKind::ConstComposite; v.type = t; v.elements = std::move(elems);
    return new_value(std::move(v));
}

ValueId Module::undef(TypeId t) {
    Value v; v.kind = ValueKind::Undef; v.type = t; return new_value(std::move(v));
}

ValueId Module::global_var(TypeId ptr_type, uint32_t sc, std::string name) {
    Value v; v.kind = ValueKind::GlobalVar; v.type = ptr_type; v.storage_class = sc; v.name = std::move(name);
    return new_value(std::move(v));
}

ValueId Module::param(TypeId t, std::string name) {
    Value v; v.kind = ValueKind::Param; v.type = t; v.name = std::move(name);
    return new_value(std::move(v));
}

// ---- functions / blocks -----------------------------------------------------
FuncId Module::new_function(TypeId fn_type, std::string name) {
    FuncId id = (FuncId)functions_.size();
    Function f; f.id = id; f.type = fn_type; f.name = std::move(name);
    functions_.push_back(std::move(f));
    return id;
}

BlockId Module::new_block(FuncId f) {
    BlockId id = (BlockId)blocks_.size();
    BasicBlock b; b.id = id; b.func = f;
    blocks_.push_back(std::move(b));
    functions_[f].blocks.push_back(id);
    return id;
}

InstId Module::create_inst(BlockId b, Op op, TypeId result_type,
                           const std::vector<ValueId>& operands,
                           uint32_t imm0, uint32_t imm1, uint32_t line) {
    Instruction in;
    in.op = op;
    in.type = result_type;
    in.imm0 = imm0; in.imm1 = imm1; in.line = line;
    in.block = b;
    in.operand_begin = (uint32_t)operands_.size();
    in.operand_count = (uint32_t)operands.size();
    for (ValueId v : operands) operands_.push_back(v);

    InstId id = (InstId)insts_.size();
    if (result_type != INVALID) {
        Value rv; rv.kind = ValueKind::InstResult; rv.type = result_type; rv.inst = id;
        in.result = new_value(std::move(rv));
    }
    insts_.push_back(in);
    return id;
}

InstId Module::emit(BlockId b, Op op, TypeId result_type,
                    const std::vector<ValueId>& operands,
                    uint32_t imm0, uint32_t imm1, uint32_t line) {
    InstId id = create_inst(b, op, result_type, operands, imm0, imm1, line);
    blocks_[b].insts.push_back(id);
    return id;
}

} // namespace omni::uir
