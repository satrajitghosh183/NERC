// omni/uir/ir.hpp — Universal IR (UIR): typed SSA, the API-agnostic core.
// Everything (SPIR-V/DXIL/AIR/WGSL/GLSL frontends, capture pass, CPU reference)
// flows through this. Storage is SoA: parallel pools + a flat operand pool;
// instructions reference operands by [begin,count) range (arena style).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace omni::uir {

// ---- Strong-ish handles (index into Module pools; INVALID == none) ----------
using TypeId  = uint32_t;
using ValueId = uint32_t;
using InstId  = uint32_t;
using BlockId = uint32_t;
using FuncId  = uint32_t;
inline constexpr uint32_t INVALID = 0xFFFFFFFFu;

// ---- Types ------------------------------------------------------------------
enum class TypeKind : uint8_t {
    Void, Bool, Int, Float, Vector, Matrix, Pointer, Struct, Array,
    RuntimeArray, Function, Image, Sampler, SampledImage
};

struct Type {
    TypeKind kind{};
    uint16_t width = 0;        // Int/Float bit width
    bool is_signed = false;    // Int signedness
    TypeId elem = INVALID;     // Vector/Matrix/Array/Pointer pointee/RuntimeArray
    uint32_t count = 0;        // Vector lanes / Matrix cols / Array length
    uint32_t storage_class = 0;// Pointer storage class (SPIR-V numbering)
    TypeId ret = INVALID;      // Function return type
    std::vector<TypeId> members; // Struct members / Function param types
};

// ---- SSA Values -------------------------------------------------------------
enum class ValueKind : uint8_t {
    InstResult, ConstInt, ConstFloat, ConstBool, ConstComposite, ConstNull,
    GlobalVar, Param, Undef
};

struct Value {
    ValueKind kind{};
    TypeId type = INVALID;
    InstId inst = INVALID;          // InstResult: the defining instruction
    uint64_t const_bits = 0;        // ConstInt/ConstFloat/ConstBool bit pattern
    std::vector<ValueId> elements;  // ConstComposite components
    uint32_t storage_class = 0;     // GlobalVar
    std::string name;               // optional debug name
};

// ---- Opcodes (shader-complete subset; extend as frontends need) -------------
enum class Op : uint16_t {
    Nop, Undef,
    // memory
    Variable, Load, Store, AccessChain,
    // int arithmetic
    IAdd, ISub, IMul, SDiv, UDiv, SMod, UMod, SNegate,
    // float arithmetic
    FAdd, FSub, FMul, FDiv, FNegate, FRem,
    // comparison (int)
    IEqual, INotEqual, SLessThan, SGreaterThan, SLessThanEqual, SGreaterThanEqual,
    ULessThan, UGreaterThan, ULessThanEqual, UGreaterThanEqual,
    // comparison (float, ordered)
    FOrdEqual, FOrdNotEqual, FOrdLessThan, FOrdGreaterThan,
    FOrdLessThanEqual, FOrdGreaterThanEqual,
    // logical / bitwise
    LogicalAnd, LogicalOr, LogicalNot, LogicalEqual,
    BitwiseAnd, BitwiseOr, BitwiseXor, Not,
    ShiftLeftLogical, ShiftRightLogical, ShiftRightArithmetic,
    // conversions
    ConvertSToF, ConvertUToF, ConvertFToS, ConvertFToU, Bitcast,
    // composite / vector
    CompositeConstruct, CompositeExtract, CompositeInsert,
    VectorShuffle, VectorTimesScalar, Select,
    // GLSL.std.450-style builtins
    Dot, Cross, Normalize, Length, Distance, Reflect,
    FMin, FMax, FClamp, FMix, Pow, Exp, Exp2, Log, Log2,
    Sin, Cos, Tan, Sqrt, InverseSqrt, FAbs, Floor, Ceil, Fract, Step, SmoothStep,
    // derivatives (quad)
    DPdx, DPdy, Fwidth,
    // texture
    ImageSampleImplicitLod, ImageSampleExplicitLod,
    // control flow
    Phi, Branch, BranchConditional, Switch,
    LoopMerge, SelectionMerge, Return, ReturnValue, Kill, Unreachable,
    // function
    FunctionCall,
    // OmniTrace capture
    TraceTap,
    Count
};

const char* op_name(Op op);

// ---- Instruction ------------------------------------------------------------
struct Instruction {
    Op op{};
    TypeId type = INVALID;       // result type (INVALID == no result / void)
    ValueId result = INVALID;    // SSA result value (INVALID == none)
    uint32_t operand_begin = 0;  // [begin, begin+count) into Module::operands_
    uint32_t operand_count = 0;
    uint32_t imm0 = 0, imm1 = 0; // small inline immediates (member idx, lit, etc.)
    uint32_t line = 0;           // source line (debug)
    BlockId block = INVALID;     // owning block
};

// ---- Basic block / Function -------------------------------------------------
struct BasicBlock {
    BlockId id = INVALID;
    FuncId func = INVALID;
    std::vector<InstId> insts;   // ordered; last is the terminator
    // filled by CFG analysis:
    std::vector<BlockId> preds, succs;
};

struct Function {
    FuncId id = INVALID;
    TypeId type = INVALID;       // function type
    std::vector<ValueId> params;
    std::vector<BlockId> blocks; // blocks[0] is the entry
    std::string name;
};

// ---- Module: the arena that owns everything --------------------------------
class Module {
public:
    // Type construction (dedup'd).
    TypeId void_type();
    TypeId bool_type();
    TypeId int_type(uint16_t width, bool is_signed);
    TypeId float_type(uint16_t width);
    TypeId vector_type(TypeId elem, uint32_t count);
    TypeId pointer_type(TypeId pointee, uint32_t storage_class);
    TypeId array_type(TypeId elem, uint32_t count);
    TypeId function_type(TypeId ret, std::vector<TypeId> params);
    TypeId struct_type(std::vector<TypeId> members);
    TypeId add_type(const Type& t); // generic, dedup'd

    const Type& type(TypeId id) const { return types_[id]; }
    size_t num_types() const { return types_.size(); }

    // Constant / value construction (constants dedup'd).
    ValueId const_int(TypeId t, uint64_t bits);
    ValueId const_float(TypeId t, double v);
    ValueId const_bool(bool v);
    ValueId const_composite(TypeId t, std::vector<ValueId> elems);
    ValueId undef(TypeId t);
    ValueId global_var(TypeId ptr_type, uint32_t storage_class, std::string name = {});
    ValueId param(TypeId t, std::string name = {});

    const Value& value(ValueId id) const { return values_[id]; }
    Value& value(ValueId id) { return values_[id]; }
    size_t num_values() const { return values_.size(); }

    // Functions & blocks.
    FuncId new_function(TypeId fn_type, std::string name);
    BlockId new_block(FuncId f);
    Function& function(FuncId f) { return functions_[f]; }
    const Function& function(FuncId f) const { return functions_[f]; }
    size_t num_functions() const { return functions_.size(); }
    BasicBlock& block(BlockId b) { return blocks_[b]; }
    const BasicBlock& block(BlockId b) const { return blocks_[b]; }
    size_t num_blocks() const { return blocks_.size(); }

    // Create an instruction (allocating its result Value if result_type != INVALID) but do
    // NOT append it to the block's instruction list — caller controls placement. Used by
    // passes that insert instructions mid-block (e.g. the capture pass).
    InstId create_inst(BlockId b, Op op, TypeId result_type,
                       const std::vector<ValueId>& operands,
                       uint32_t imm0 = 0, uint32_t imm1 = 0, uint32_t line = 0);

    // Emit an instruction at the end of block `b` (create_inst + append).
    InstId emit(BlockId b, Op op, TypeId result_type,
                const std::vector<ValueId>& operands,
                uint32_t imm0 = 0, uint32_t imm1 = 0, uint32_t line = 0);

    const Instruction& inst(InstId i) const { return insts_[i]; }
    Instruction& inst(InstId i) { return insts_[i]; }
    size_t num_insts() const { return insts_.size(); }

    // Operand access for a given instruction.
    const ValueId* operands(const Instruction& in) const {
        return in.operand_count ? &operands_[in.operand_begin] : nullptr;
    }
    ValueId result_of(InstId i) const { return insts_[i].result; }

    // Debug source.
    std::string source_text;
    std::string source_file;

private:
    ValueId new_value(Value v);
    std::vector<Type> types_;
    std::vector<Value> values_;
    std::vector<Instruction> insts_;
    std::vector<BasicBlock> blocks_;
    std::vector<Function> functions_;
    std::vector<ValueId> operands_; // flat SoA operand pool
    std::unordered_map<std::string, TypeId> type_dedup_;
    std::unordered_map<std::string, ValueId> const_dedup_;
};

} // namespace omni::uir
