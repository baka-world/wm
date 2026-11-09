cmake_minimum_required(VERSION 3.16)
project(wm_pibt_bridge LANGUAGES CXX)
if(NOT DEFINED PIBT_UPSTREAM)
  message(FATAL_ERROR "Set PIBT_UPSTREAM to the pinned unmodified pibt2 checkout")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# The pinned upstream grid-pathfinding revision uses std::memset without
# including <cstring>. Inject the standard header without changing its source.
add_compile_options(-include cstring)
add_subdirectory("${PIBT_UPSTREAM}/pibt2" upstream-pibt2)
add_library(wm_pibt_bridge SHARED pibt_bridge.cpp)
target_compile_features(wm_pibt_bridge PRIVATE cxx_std_17)
target_include_directories(wm_pibt_bridge PRIVATE "${PIBT_UPSTREAM}/pibt2/include"
  "${PIBT_UPSTREAM}/third_party/grid-pathfinding/graph/include")
target_compile_definitions(wm_pibt_bridge PRIVATE
  WM_SAMPLE_INSTANCE="${PIBT_UPSTREAM}/instances/mapf/sample.txt")
target_link_libraries(wm_pibt_bridge PRIVATE lib-mapf)
