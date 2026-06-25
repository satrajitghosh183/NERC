# cmake/PGO.cmake
#
# Profile-guided optimization helpers (item CC.2).
#
# Usage:
#   cmake -DOLMO_PGO=generate -B build_pgo_gen   # build instrumented
#   ./build_pgo_gen/olmo_train conf/...          # run; emits .gcda files
#   cmake -DOLMO_PGO=use -B build_pgo_use        # rebuild with profile feedback
#
# Combined with -DOLMO_ENABLE_LTO=ON for whole-program optimization,
# this typically buys an additional 5-10% on CPU-side dispatch hot paths.

option(OLMO_PGO "Profile-guided optimization stage: generate | use | off" "off")

if(OLMO_PGO STREQUAL "generate")
  message(STATUS "PGO: instrumenting (generate phase)")
  add_compile_options(-fprofile-generate)
  add_link_options(-fprofile-generate)
elseif(OLMO_PGO STREQUAL "use")
  message(STATUS "PGO: using collected profile (use phase)")
  add_compile_options(-fprofile-use -fprofile-correction)
  add_link_options(-fprofile-use)
endif()
