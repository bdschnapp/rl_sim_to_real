cmake_minimum_required(VERSION 3.14)
project(nebula_hw_interfaces)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunused-function)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(boost_tcp_driver)
find_package(boost_udp_driver)
find_package(nebula_common)
find_package(nebula_msgs)
find_package(pandar_msgs)
find_package(robosense_msgs)
find_package(ros2_socketcan)
find_package(velodyne_msgs)

# Common includes for all targets
include_directories(
    ${nebula_common_INCLUDE_DIRS}
    include
    SYSTEM
)

link_libraries(
    ${nebula_common_TARGETS}
)

# Hesai / Velodyne hw_interfaces removed in the aggressive prune (RoboSense only).

add_library(nebula_hw_interfaces_robosense SHARED
    src/nebula_robosense_hw_interfaces/robosense_hw_interface.cpp
)
target_link_libraries(nebula_hw_interfaces_robosense PUBLIC
    ${boost_tcp_driver_LIBRARIES}
    ${boost_udp_driver_LIBRARIES}
    ${robosense_msgs_TARGETS}
)
target_include_directories(nebula_hw_interfaces_robosense PUBLIC
    ${boost_udp_driver_INCLUDE_DIRS}
    ${boost_tcp_driver_INCLUDE_DIRS}
    ${robosense_msgs_INCLUDE_DIRS}
)

install(TARGETS nebula_hw_interfaces_robosense EXPORT export_nebula_hw_interfaces_robosense)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()

    find_package(ament_cmake_gtest REQUIRED)

    ament_add_gtest(test_udp
        test/common/test_udp.cpp
    )

    target_include_directories(test_udp PUBLIC
        ${nebula_common_INCLUDE_DIRS}
        include
        test)
endif()

ament_export_include_directories("include/${PROJECT_NAME}")
ament_export_targets(export_nebula_hw_interfaces_robosense)

ament_export_dependencies(
    boost_tcp_driver
    boost_udp_driver
    nebula_common
    nebula_msgs
    pandar_msgs
    robosense_msgs
    ros2_socketcan
    velodyne_msgs
)

ament_package()
