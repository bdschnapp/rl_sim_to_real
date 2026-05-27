cmake_minimum_required(VERSION 3.14)
project(autoware_trailer_state_visualizer)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/trailer_state_visualizer.cpp
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "autoware::trailer_state_visualizer::TrailerStateVisualizer"
  EXECUTABLE ${PROJECT_NAME}_node
)

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)
