cmake_minimum_required(VERSION 3.20)
project(omni_shader_debugger LANGUAGES CXX VERSION 0.1.0)

# ---- Standard & flags -------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# Apple Silicon: NEON is baseline; nothing to enable. Allow optional sanitizers.
option(OMNI_ASAN "Build with AddressSanitizer/UBSan" OFF)
if(OMNI_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# Absolute path to the project's data/ dir (sample shaders, corpora) for tests.
add_compile_definitions(OMNI_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ---- Core library -----------------------------------------------------------
file(GLOB_RECURSE OMNI_CORE_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(FILTER OMNI_CORE_SRC EXCLUDE REGEX "/gpu/|/synth/renderer")  # GPU sources added conditionally below
add_library(omni_core STATIC ${OMNI_CORE_SRC})
target_include_directories(omni_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ---- Optional: Vulkan (MoltenVK) for real GPU capture -----------------------
set(OMNI_VULKAN_SDK "/Users/satrajitghosh/Software/VulkanSDK/1.4.328.1/macOS" CACHE PATH "Vulkan SDK root")
if(EXISTS "${OMNI_VULKAN_SDK}/include/vulkan/vulkan.h" AND EXISTS "${OMNI_VULKAN_SDK}/lib/libvulkan.dylib")
  message(STATUS "omni: Vulkan/MoltenVK found at ${OMNI_VULKAN_SDK} -> real GPU capture enabled")
  target_sources(omni_core PRIVATE ${CMAKE_SOURCE_DIR}/src/gpu/vulkan_capture.cpp
                                   ${CMAKE_SOURCE_DIR}/src/synth/renderer.cpp)
  target_include_directories(omni_core PUBLIC ${OMNI_VULKAN_SDK}/include)
  target_link_libraries(omni_core PUBLIC ${OMNI_VULKAN_SDK}/lib/libvulkan.dylib)
  target_compile_definitions(omni_core PUBLIC OMNI_HAVE_VULKAN=1 OMNI_VULKAN_SDK="${OMNI_VULKAN_SDK}")
  set(CMAKE_BUILD_RPATH "${OMNI_VULKAN_SDK}/lib")
else()
  message(STATUS "omni: Vulkan SDK not found -> GPU capture compiled out")
endif()

# ---- Test framework main ----------------------------------------------------
add_library(omni_test_main STATIC ${CMAKE_SOURCE_DIR}/tests/_test_main.cpp)

# ---- Tests ------------------------------------------------------------------
enable_testing()
file(GLOB OMNI_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
foreach(t ${OMNI_TESTS})
  get_filename_component(tname ${t} NAME_WE)
  add_executable(${tname} ${t})
  target_link_libraries(${tname} PRIVATE omni_core omni_test_main)
  add_test(NAME ${tname} COMMAND ${tname})
endforeach()

# ---- Benchmarks -------------------------------------------------------------
file(GLOB OMNI_BENCH CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/bench/bench_*.cpp)
foreach(b ${OMNI_BENCH})
  get_filename_component(bname ${b} NAME_WE)
  add_executable(${bname} ${b})
  target_link_libraries(${bname} PRIVATE omni_core)
endforeach()

# ---- Tools (CLIs) -----------------------------------------------------------
# The debugger-in-the-loop reward CLI: compile -> lift -> CPU-reference exec -> reward JSON.
file(GLOB OMNI_TOOLS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tools/*.cpp)
foreach(t ${OMNI_TOOLS})
  get_filename_component(tname ${t} NAME_WE)
  add_executable(${tname} ${t})
  target_link_libraries(${tname} PRIVATE omni_core)
endforeach()

message(STATUS "omni: ${CMAKE_BUILD_TYPE} build, C++${CMAKE_CXX_STANDARD}, "
        "tests=${OMNI_TESTS}")
