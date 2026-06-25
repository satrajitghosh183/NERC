// omni/test.hpp — tiny from-scratch unit-test framework (no external deps).
// Usage:
//   #include "omni/test.hpp"
//   TEST(suite_name, case_name) { CHECK(expr); CHECK_EQ(a, b); }
// Link the test file together with omni_test_main (provides main()).
#pragma once
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <sstream>

namespace omni::test {

struct Case {
    const char* suite;
    const char* name;
    std::function<void()> fn;
};

struct Registry {
    std::vector<Case> cases;
    static Registry& get() { static Registry r; return r; }
};

struct Registrar {
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        Registry::get().cases.push_back({suite, name, std::move(fn)});
    }
};

// Thrown on a failed REQUIRE/CHECK to abort the current case.
struct Failure {
    std::string msg;
};

inline thread_local int g_check_failures = 0;

inline void report_fail(const char* file, int line, const std::string& expr, bool fatal) {
    ++g_check_failures;
    std::fprintf(stderr, "    FAIL %s:%d  %s\n", file, line, expr.c_str());
    if (fatal) throw Failure{expr};
}

// Runs every registered case; returns process exit code (0 == all passed).
inline int run_all(const char* filter = nullptr) {
    auto& cases = Registry::get().cases;
    int passed = 0, failed = 0;
    std::string cur_suite;
    for (auto& c : cases) {
        if (filter && std::string(c.suite).find(filter) == std::string::npos &&
            std::string(c.name).find(filter) == std::string::npos)
            continue;
        if (cur_suite != c.suite) { cur_suite = c.suite; std::printf("[%s]\n", cur_suite.c_str()); }
        g_check_failures = 0;
        bool crashed = false;
        try {
            c.fn();
        } catch (const Failure&) {
            crashed = true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "    EXCEPTION: %s\n", e.what());
            crashed = true;
        } catch (...) {
            std::fprintf(stderr, "    EXCEPTION: unknown\n");
            crashed = true;
        }
        bool ok = (g_check_failures == 0) && !crashed;
        std::printf("  %s %s\n", ok ? "ok  " : "FAIL", c.name);
        ok ? ++passed : ++failed;
    }
    std::printf("\n%d passed, %d failed, %d total\n", passed, failed, passed + failed);
    return failed == 0 ? 0 : 1;
}

} // namespace omni::test

#define OMNI_CONCAT_(a, b) a##b
#define OMNI_CONCAT(a, b) OMNI_CONCAT_(a, b)

#define TEST(suite, name)                                                          \
    static void OMNI_CONCAT(omni_test_fn_, __LINE__)();                            \
    static ::omni::test::Registrar OMNI_CONCAT(omni_test_reg_, __LINE__)(          \
        #suite, #name, &OMNI_CONCAT(omni_test_fn_, __LINE__));                     \
    static void OMNI_CONCAT(omni_test_fn_, __LINE__)()

#define CHECK(expr)                                                                \
    do { if (!(expr)) ::omni::test::report_fail(__FILE__, __LINE__, #expr, false); } while (0)

#define REQUIRE(expr)                                                              \
    do { if (!(expr)) ::omni::test::report_fail(__FILE__, __LINE__, #expr, true); } while (0)

#define CHECK_EQ(a, b)                                                             \
    do { auto _a = (a); auto _b = (b); if (!(_a == _b)) {                          \
        std::ostringstream _os; _os << #a " == " #b " (" << _a << " vs " << _b << ")"; \
        ::omni::test::report_fail(__FILE__, __LINE__, _os.str(), false); } } while (0)

#define REQUIRE_EQ(a, b)                                                           \
    do { auto _a = (a); auto _b = (b); if (!(_a == _b)) {                          \
        std::ostringstream _os; _os << #a " == " #b " (" << _a << " vs " << _b << ")"; \
        ::omni::test::report_fail(__FILE__, __LINE__, _os.str(), true); } } while (0)

#define CHECK_NEAR(a, b, tol)                                                      \
    do { double _a = (a), _b = (b), _t = (tol); if (std::fabs(_a - _b) > _t) {     \
        std::ostringstream _os; _os << #a " ~= " #b " (" << _a << " vs " << _b << ", tol " << _t << ")"; \
        ::omni::test::report_fail(__FILE__, __LINE__, _os.str(), false); } } while (0)
