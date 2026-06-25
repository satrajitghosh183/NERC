#include "omni/test.hpp"
// Shared main() for every test executable. Optional argv[1] filters by substring.
int main(int argc, char** argv) {
    return omni::test::run_all(argc > 1 ? argv[1] : nullptr);
}
