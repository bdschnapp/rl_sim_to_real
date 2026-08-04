cmake_minimum_required(VERSION 3.5)
project(canbridge)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(can_msgs REQUIRED)

# C++ socketcan bridge node
add_executable(ros2can_bridge src/ros2socketcan.cpp)
target_link_libraries(ros2can_bridge dl boost_system)
ament_target_dependencies(ros2can_bridge rclcpp can_msgs)

install(TARGETS ros2can_bridge
  DESTINATION lib/${PROJECT_NAME})

# Python canbridge node
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  canbridge/canbridge.py
  RENAME canbridge
  DESTINATION lib/${PROJECT_NAME})

# Launch and config files
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()
