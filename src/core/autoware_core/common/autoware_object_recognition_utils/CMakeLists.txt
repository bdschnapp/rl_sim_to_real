cmake_minimum_required(VERSION 3.14)
project(autoware_object_recognition_utils)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(Boost REQUIRED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/predicted_path_utils.cpp
  src/conversion.cpp
)

if(BUILD_TESTING)
  find_package(ament_cmake_ros REQUIRED)

  file(GLOB_RECURSE test_files test/**/*.cpp)

  ament_add_ros_isolated_gtest(test_object_recognition_utils ${test_files})

  target_link_libraries(test_object_recognition_utils
    ${PROJECT_NAME}
  )

  # Boost.Geometry triggers a spurious -Wmaybe-uninitialized in
  # get_rescale_policy.hpp under recent GCC; silence it for the tests.
  target_compile_options(test_object_recognition_utils PRIVATE -Wno-maybe-uninitialized)
endif()

ament_auto_package()
