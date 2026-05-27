cmake_minimum_required(VERSION 3.14)
project(nebula_ros)

# Default to C++17
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wunused-function)
endif ()

find_package(ament_cmake_auto REQUIRED)
find_package(PCL REQUIRED components common)
find_package(autoware_sensing_msgs REQUIRED)
find_package(continental_msgs REQUIRED)
find_package(continental_srvs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nebula_common REQUIRED)
find_package(nebula_decoders REQUIRED)
find_package(nebula_hw_interfaces REQUIRED)
find_package(nebula_msgs REQUIRED)
find_package(pandar_msgs REQUIRED)
find_package(radar_msgs REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(robosense_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(sync_tooling_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(autoware_utils_debug REQUIRED)
find_package(autoware_internal_debug_msgs REQUIRED)
find_package(agnocastlib)

if("$ENV{ENABLE_AGNOCAST}" AND NOT agnocastlib_FOUND)
    message(FATAL_ERROR "agnocastlib is required when Agnocast is enabled")
endif()

include_directories(
    include
    SYSTEM
    ${nebula_common_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${rclcpp_components_INCLUDE_DIRS}
)

link_libraries(
    ${nebula_common_TARGETS}
    ${YAML_CPP_LIBRARIES}
    ${PCL_LIBRARIES}
)

# Hesai wrapper removed in aggressive prune.

## Robosense
add_library(robosense_ros_wrapper SHARED
    src/robosense/robosense_ros_wrapper.cpp
    src/robosense/decoder_wrapper.cpp
    src/robosense/hw_interface_wrapper.cpp
    src/robosense/hw_monitor_wrapper.cpp
    src/common/parameter_descriptors.cpp
)

target_include_directories(robosense_ros_wrapper PUBLIC
    ${diagnostic_updater_INCLUDE_DIRS}
    ${nebula_decoders_INCLUDE_DIRS}
    ${nebula_hw_interfaces_INCLUDE_DIRS}
    ${robosense_msgs_INCLUDE_DIRS}
)

target_link_libraries(robosense_ros_wrapper PUBLIC
    ${diagnostic_updater_TARGETS}
    ${diagnostic_msgs_TARGETS}
    ${robosense_msgs_TARGETS}
    nebula_decoders::nebula_decoders_robosense
    nebula_decoders::nebula_decoders_robosense_info
    nebula_hw_interfaces::nebula_hw_interfaces_robosense
)

rclcpp_components_register_node(robosense_ros_wrapper
    PLUGIN "RobosenseRosWrapper"
    EXECUTABLE robosense_ros_wrapper_node
)

# Continental wrappers (ARS548, SRR520) removed in aggressive prune.

install(TARGETS robosense_ros_wrapper EXPORT export_robosense_ros_wrapper LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    find_package(ros_testing REQUIRED)
    find_package(ament_cmake_gtest REQUIRED)
    find_package(rcpputils REQUIRED)

    ament_lint_auto_find_test_dependencies()

    # Unit tests
    ament_add_gtest(test_single_consumer_processor
        test/test_single_consumer_processor.cpp
    )
    target_include_directories(test_single_consumer_processor
        PRIVATE
        include
        SYSTEM
        ${rcpputils_INCLUDE_DIRS}
    )

    foreach(MODEL Pandar40P Pandar64 PandarQT64 PandarQT128 Pandar128E4X PandarAT128 PandarXT16 PandarXT32 PandarXT32M ARS548 SRR520 VLP16 VLP32 VLS128 Helios Bpearl Airy)
        string(TOLOWER ${MODEL}_smoke_test test_name)
        add_ros_test(
            test/smoke_test.py
            TARGET ${test_name}
            ARGS "launch_file_path:=${PROJECT_SOURCE_DIR}/launch/nebula_launch.py" "sensor_model:=${MODEL}"
            TIMEOUT "10"
        )
    endforeach()
endif()

ament_export_include_directories("include/${PROJECT_NAME}")
ament_export_targets(export_robosense_ros_wrapper)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
    PCL
    autoware_sensing_msgs
    continental_msgs
    continental_srvs
    diagnostic_msgs
    diagnostic_updater
    geometry_msgs
    nebula_common
    nebula_decoders
    nebula_hw_interfaces
    nebula_msgs
    pandar_msgs
    radar_msgs
    rclcpp_components
    robosense_msgs
    sensor_msgs
    tf2_ros
    velodyne_msgs
    visualization_msgs
    yaml-cpp
    message_filters
    sync_tooling_msgs
    autoware_utils_debug
    autoware_internal_debug_msgs
)

if("$ENV{ENABLE_AGNOCAST}")
    ament_export_dependencies(agnocastlib)
endif()

ament_package()

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
