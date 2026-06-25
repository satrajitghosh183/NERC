#include "omni/test.hpp"
#include "omni/uir/ir.hpp"
#include "omni/uir/verify.hpp"
#include <string>

using namespace omni::uir;

TEST(uir, type_dedup) {
    Module m;
    TypeId f32a = m.float_type(32);
    TypeId f32b = m.float_type(32);
    REQUIRE_EQ(f32a, f32b);                 // identical types dedup to one id
    TypeId v3a = m.vector_type(f32a, 3);
    TypeId v3b = m.vector_type(f32b, 3);
    REQUIRE_EQ(v3a, v3b);
    TypeId v4 = m.vector_type(f32a, 4);
    CHECK(v3a != v4);
    CHECK(m.type(v3a).kind == TypeKind::Vector);
    CHECK_EQ(m.type(v3a).count, 3u);
}

TEST(uir, const_dedup) {
    Module m;
    TypeId i32 = m.int_type(32, true);
    ValueId a = m.const_int(i32, 7);
    ValueId b = m.const_int(i32, 7);
    REQUIRE_EQ(a, b);
    ValueId c = m.const_int(i32, 8);
    CHECK(a != c);
    TypeId f32 = m.float_type(32);
    ValueId f1 = m.const_float(f32, 1.5);
    ValueId f2 = m.const_float(f32, 1.5);
    REQUIRE_EQ(f1, f2);
}

// Build a tiny fragment-like function:  out = a * b + c   (floats), then Return.
TEST(uir, build_and_verify_ok) {
    Module m;
    TypeId f32  = m.float_type(32);
    TypeId tvoid = m.void_type();
    TypeId fn = m.function_type(tvoid, {});
    FuncId f = m.new_function(fn, "main");
    BlockId entry = m.new_block(f);

    ValueId a = m.const_float(f32, 2.0);
    ValueId b = m.const_float(f32, 3.0);
    ValueId c = m.const_float(f32, 4.0);

    InstId mul = m.emit(entry, Op::FMul, f32, {a, b});
    ValueId prod = m.result_of(mul);
    CHECK(prod != INVALID);
    m.emit(entry, Op::FAdd, f32, {prod, c});
    m.emit(entry, Op::Return, INVALID, {});

    std::vector<std::string> errs;
    bool ok = verify(m, errs);
    for (auto& e : errs) std::printf("    verify: %s\n", e.c_str());
    REQUIRE(ok);

    // Result value links back to its defining instruction.
    const Value& pv = m.value(prod);
    CHECK(pv.kind == ValueKind::InstResult);
    CHECK_EQ(pv.inst, mul);
    CHECK_EQ(pv.type, f32);
}

TEST(uir, verify_catches_missing_terminator) {
    Module m;
    TypeId f32 = m.float_type(32);
    TypeId fn  = m.function_type(m.void_type(), {});
    FuncId f = m.new_function(fn, "bad");
    BlockId entry = m.new_block(f);
    ValueId a = m.const_float(f32, 1.0);
    m.emit(entry, Op::FAdd, f32, {a, a}); // no terminator
    std::vector<std::string> errs;
    bool ok = verify(m, errs);
    CHECK(!ok);
    CHECK(!errs.empty());
}

TEST(uir, op_names_nonempty) {
    CHECK(std::string(op_name(Op::FMul)) == "FMul");
    CHECK(std::string(op_name(Op::TraceTap)) == "TraceTap");
}
