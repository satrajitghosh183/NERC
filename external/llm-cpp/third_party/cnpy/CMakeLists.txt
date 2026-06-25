# Minimal cnpy build - vendored from https://github.com/rogersce/cnpy
# Used as add_subdirectory - no project() to avoid redefining parent project
find_package(ZLIB REQUIRED)

add_library(cnpy STATIC cnpy.cpp)
set_property(TARGET cnpy PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)
