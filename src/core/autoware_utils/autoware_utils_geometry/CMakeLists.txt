cmake_minimum_required(VERSION 3.14)
project(autoware_utils_geometry)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED
  "src/geometry/alt_geometry.cpp"
  "src/geometry/boost_polygon_utils.cpp"
  "src/geometry/ear_clipping.cpp"
  "src/geometry/geometry.cpp"
  "src/geometry/gjk_2d.cpp"
  "src/geometry/pose_deviation.cpp"
  "src/geometry/random_concave_polygon.cpp"
  "src/geometry/random_convex_polygon.cpp"
  "src/geometry/sat_2d.cpp"
  "src/msg/operation.cpp"
)

# Boost.Geometry triggers a spurious -Wmaybe-uninitialized in
# get_rescale_policy.hpp under recent GCC; silence it for this target.
target_compile_options(${PROJECT_NAME} PRIVATE -Wno-maybe-uninitialized)

if(BUILD_TESTING)
  file(GLOB_RECURSE test_files test/*.cpp)

  ament_auto_add_gtest(test_${PROJECT_NAME} ${test_files})
  target_compile_options(test_${PROJECT_NAME} PRIVATE -Wno-maybe-uninitialized)
endif()

ament_auto_package()
