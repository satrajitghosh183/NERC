# Race pipeline — incremental orchestration replacing scripts/race/run_all.sh
#
#   cmake -B build .
#   cmake --build build --target race
#
# Each phase is a custom_command with real output files so `make` only reruns
# work that is stale. Individual phases are also exposed as targets:
#   race_env, race_build, race_correctness, race_data,
#   race_train_cpp, race_train_python, race_infer_cpp, race_infer_python,
#   race_analyze, race

option(OLMO_RACE_BUILD "Register the race pipeline targets" ON)
option(OLMO_RACE_SKIP_PYTHON "Skip Python train/infer phases (05, 07)" OFF)

if(NOT OLMO_RACE_BUILD)
  return()
endif()

set(OLMO_RACE_DIR "${CMAKE_SOURCE_DIR}/scripts/race")
set(OLMO_RACE_RESULTS "${OLMO_RACE_DIR}/results")
set(OLMO_RACE_CONF "${OLMO_RACE_DIR}/configs/race_250m_cpp.conf")
set(OLMO_RACE_JSON "${OLMO_RACE_DIR}/configs/race_250m_cpp.json")
set(OLMO_RACE_DATA "${CMAKE_SOURCE_DIR}/data/race_tokens.npy")
set(OLMO_RACE_CPP_CKPT "${OLMO_RACE_RESULTS}/cpp_ckpt/model.pt")
set(OLMO_RACE_CPP_METRICS "${OLMO_RACE_RESULTS}/cpp_train/metrics.csv")
set(OLMO_RACE_PY_METRICS "${OLMO_RACE_RESULTS}/python_train/metrics.csv")
set(OLMO_RACE_CPP_INFER "${OLMO_RACE_RESULTS}/cpp_infer/results.csv")
set(OLMO_RACE_PY_INFER "${OLMO_RACE_RESULTS}/python_infer/results.csv")
set(OLMO_RACE_REPORT "${OLMO_RACE_RESULTS}/RESULT.md")

file(MAKE_DIRECTORY "${OLMO_RACE_RESULTS}")

set(OLMO_RACE_BUILD_TARGETS
  olmo_train chat prepare_data
  test_cuda_parity test_fused_ce test_fused_qkv_rope
  test_paged_kv test_prefix_cache test_scheduler
)

# Phase 00 — environment preflight
add_custom_command(
  OUTPUT "${OLMO_RACE_RESULTS}/.phase00_env.stamp"
  COMMAND ${CMAKE_COMMAND} -E env
    BUILD_DIR=${CMAKE_BINARY_DIR}
    bash "${OLMO_RACE_DIR}/00_env_check.sh"
  COMMAND ${CMAKE_COMMAND} -E touch "${OLMO_RACE_RESULTS}/.phase00_env.stamp"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Race 00: environment check"
  VERBATIM
)

# Phase 01 — C++ build (normal cmake targets; stamp records completion)
add_custom_command(
  OUTPUT "${OLMO_RACE_RESULTS}/.phase01_build.stamp"
  COMMAND ${CMAKE_COMMAND} -E touch "${OLMO_RACE_RESULTS}/.phase01_build.stamp"
  DEPENDS ${OLMO_RACE_BUILD_TARGETS}
          "${OLMO_RACE_RESULTS}/.phase00_env.stamp"
  COMMENT "Race 01: C++ build"
)

# Phase 02 — correctness gates
add_custom_command(
  OUTPUT "${OLMO_RACE_RESULTS}/.phase02_correctness.stamp"
  COMMAND ${CMAKE_COMMAND} -E env
    BUILD_DIR=${CMAKE_BINARY_DIR}
    bash "${OLMO_RACE_DIR}/02_verify_correctness.sh"
  COMMAND ${CMAKE_COMMAND} -E touch "${OLMO_RACE_RESULTS}/.phase02_correctness.stamp"
  DEPENDS ${OLMO_RACE_BUILD_TARGETS}
          "${OLMO_RACE_RESULTS}/.phase01_build.stamp"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Race 02: correctness verification"
  VERBATIM
)

# Phase 03 — tokenized corpus
add_custom_command(
  OUTPUT "${OLMO_RACE_DATA}"
  COMMAND ${CMAKE_COMMAND} -E env
    BUILD_DIR=${CMAKE_BINARY_DIR}
    bash "${OLMO_RACE_DIR}/03_prepare_data.sh"
  DEPENDS prepare_data
          "${OLMO_RACE_RESULTS}/.phase02_correctness.stamp"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Race 03: prepare tokenized data"
  VERBATIM
)

# Phase 04 — C++ training
add_custom_command(
  OUTPUT "${OLMO_RACE_CPP_METRICS}" "${OLMO_RACE_CPP_CKPT}"
  COMMAND ${CMAKE_COMMAND} -E env
    BUILD_DIR=${CMAKE_BINARY_DIR}
    bash "${OLMO_RACE_DIR}/04_train_cpp.sh"
  DEPENDS olmo_train
          "${OLMO_RACE_DATA}"
          "${OLMO_RACE_CONF}"
          "${OLMO_RACE_RESULTS}/.phase02_correctness.stamp"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Race 04: train C++"
  VERBATIM
)

# Phase 05 — Python training (optional)
if(NOT OLMO_RACE_SKIP_PYTHON)
  add_custom_command(
    OUTPUT "${OLMO_RACE_PY_METRICS}"
    COMMAND ${CMAKE_COMMAND} -E env
      BUILD_DIR=${CMAKE_BINARY_DIR}
      bash "${OLMO_RACE_DIR}/05_train_python.sh"
    DEPENDS "${OLMO_RACE_DATA}"
            "${OLMO_RACE_DIR}/olmo_train_race.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Race 05: train Python"
    VERBATIM
  )
endif()

# Phase 06 — C++ inference
add_custom_command(
  OUTPUT "${OLMO_RACE_CPP_INFER}"
  COMMAND ${CMAKE_COMMAND} -E env
    BUILD_DIR=${CMAKE_BINARY_DIR}
    bash "${OLMO_RACE_DIR}/06_infer_cpp.sh"
  DEPENDS chat
          "${OLMO_RACE_CPP_CKPT}"
          "${OLMO_RACE_JSON}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Race 06: infer C++"
  VERBATIM
)

# Phase 07 — Python inference (optional)
if(NOT OLMO_RACE_SKIP_PYTHON)
  add_custom_command(
    OUTPUT "${OLMO_RACE_PY_INFER}"
    COMMAND ${CMAKE_COMMAND} -E env
      BUILD_DIR=${CMAKE_BINARY_DIR}
      bash "${OLMO_RACE_DIR}/07_infer_python.sh"
    DEPENDS "${OLMO_RACE_PY_METRICS}"
            "${OLMO_RACE_DIR}/python_inference_baseline.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Race 07: infer Python"
    VERBATIM
  )
endif()

# Phase 08 — analyze and emit RESULT.md
set(_OLMO_RACE_ANALYZE_DEPS
  "${OLMO_RACE_CPP_METRICS}"
  "${OLMO_RACE_CPP_INFER}"
)
if(NOT OLMO_RACE_SKIP_PYTHON)
  list(APPEND _OLMO_RACE_ANALYZE_DEPS
    "${OLMO_RACE_PY_METRICS}"
    "${OLMO_RACE_PY_INFER}"
  )
endif()

add_custom_command(
  OUTPUT "${OLMO_RACE_REPORT}"
  COMMAND ${OLMO_PYTHON3} "${OLMO_RACE_DIR}/08_analyze.py"
  DEPENDS ${_OLMO_RACE_ANALYZE_DEPS}
          "${OLMO_RACE_DIR}/08_analyze.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Race 08: analyze results"
  VERBATIM
)

# Per-phase targets (for partial runs / debugging)
add_custom_target(race_env
  DEPENDS "${OLMO_RACE_RESULTS}/.phase00_env.stamp")
add_custom_target(race_build
  DEPENDS "${OLMO_RACE_RESULTS}/.phase01_build.stamp")
add_custom_target(race_correctness
  DEPENDS "${OLMO_RACE_RESULTS}/.phase02_correctness.stamp")
add_custom_target(race_data
  DEPENDS "${OLMO_RACE_DATA}")
add_custom_target(race_train_cpp
  DEPENDS "${OLMO_RACE_CPP_METRICS}")
add_custom_target(race_infer_cpp
  DEPENDS "${OLMO_RACE_CPP_INFER}")
add_custom_target(race_analyze
  DEPENDS "${OLMO_RACE_REPORT}")

if(NOT OLMO_RACE_SKIP_PYTHON)
  add_custom_target(race_train_python
    DEPENDS "${OLMO_RACE_PY_METRICS}")
  add_custom_target(race_infer_python
    DEPENDS "${OLMO_RACE_PY_INFER}")
endif()

# Top-level: full pipeline
set(_OLMO_RACE_ALL_DEPS "${OLMO_RACE_REPORT}")
add_custom_target(race DEPENDS ${_OLMO_RACE_ALL_DEPS})

message(STATUS "Race pipeline: ON (target 'race'; -DOLMO_RACE_SKIP_PYTHON=ON to skip py phases)")
