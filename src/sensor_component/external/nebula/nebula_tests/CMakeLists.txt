cmake_minimum_required(VERSION 3.14)
project(nebula_tests)

# Default to C++17
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunused-function)
endif ()

set(ROS_DISTRO $ENV{ROS_DISTRO})
if(${ROS_DISTRO} STREQUAL "rolling")
    add_compile_definitions(ROS_DISTRO_ROLLING)
elseif(${ROS_DISTRO} STREQUAL "foxy")
    add_compile_definitions(ROS_DISTRO_FOXY)
elseif(${ROS_DISTRO} STREQUAL "galactic")
    add_compile_definitions(ROS_DISTRO_GALACTIC)
elseif(${ROS_DISTRO} STREQUAL "humble")
    add_compile_definitions(ROS_DISTRO_HUMBLE)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(nebula_common REQUIRED)
find_package(nebula_decoders REQUIRED)
find_package(nebula_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(rosbag2_cpp REQUIRED)
find_package(diagnostic_updater REQUIRED)


if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()

    find_package(ament_cmake_gtest)

    add_definitions(-D_SRC_RESOURCES_DIR_PATH="${PROJECT_SOURCE_DIR}/data/")
    # TODO: FIX this path to point to nebula_decoders/calibration
    add_definitions(-D_SRC_CALIBRATION_DIR_PATH="${PROJECT_SOURCE_DIR}/../nebula_decoders/calibration/")

    set(NEBULA_TEST_INCLUDE_DIRS
        ${nebula_common_INCLUDE_DIRS}
        ${nebula_decoders_INCLUDE_DIRS}
        ${nebula_ros_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${rosbag2_cpp_INCLUDE_DIRS}
        ${diagnostic_updater_INCLUDE_DIRS}
    )

    set(NEBULA_TEST_LIBRARIES
        ${nebula_common_TARGETS}
        ${PCL_LIBRARIES}
        ${rosbag2_cpp_TARGETS}
        ${diagnostic_updater_TARGETS}
    )

    # Continental/Hesai/Velodyne decoders were stripped from nebula_decoders
    # in the aggressive prune (see nebula_decoders/CMakeLists.txt). The tests
    # in continental/, hesai/, and velodyne/ link against targets that no
    # longer exist, so they are disabled here.

endif()

ament_package()
