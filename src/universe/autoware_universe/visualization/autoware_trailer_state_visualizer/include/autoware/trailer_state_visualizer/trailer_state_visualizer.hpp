// Copyright 2024 Electrans
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__TRAILER_STATE_VISUALIZER__TRAILER_STATE_VISUALIZER_HPP_
#define AUTOWARE__TRAILER_STATE_VISUALIZER__TRAILER_STATE_VISUALIZER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <autoware_vehicle_info_utils/vehicle_info.hpp>
#include <autoware_vehicle_msgs/msg/trailer_state.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <memory>
#include <mutex>
#include <string>

namespace autoware::trailer_state_visualizer
{

class TrailerStateVisualizer : public rclcpp::Node
{
public:
  explicit TrailerStateVisualizer(const rclcpp::NodeOptions & options);

private:
  // Subscriptions
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
  rclcpp::Subscription<autoware_vehicle_msgs::msg::TrailerState>::SharedPtr sub_trailer_state_;

  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::TimerBase::SharedPtr timer_;

  // Cached messages
  nav_msgs::msg::Odometry::ConstSharedPtr odometry_ptr_;
  autoware_vehicle_msgs::msg::TrailerState::ConstSharedPtr trailer_state_ptr_;
  std::mutex mutex_;

  // Truck dimensions come from vehicle_info (no manual truck size parameters).
  autoware::vehicle_info_utils::VehicleInfo vehicle_info_;
  double hitch_offset_m_;  // rear axle(base_link) -> hitch offset [m]

  // Trailer parameters
  struct TrailerParams
  {
    double wheelbase;       // hitch to trailer rear axle [m]
    double width;           // [m]
    double front_overhang;  // hitch to front bumper [m]
    double rear_overhang;   // rear axle to rear bumper [m]
    double height;          // [m]
  } trailer_;

  struct MeshParams
  {
    bool enable;
    std::string resource;
    double scale_x;
    double scale_y;
    double scale_z;
    double offset_x;
    double offset_y;
    double offset_z;
    double roll;
    double pitch;
    double yaw;
  };

  MeshParams truck_mesh_;
  MeshParams trailer_mesh_;

  double hitch_marker_z_;  // height of hitch marker above ground [m]
  bool publish_hitch_marker_;
  bool publish_hitch_text_;
  bool publish_trailer_tf_;

  // Callbacks
  void onOdometry(nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void onTrailerState(autoware_vehicle_msgs::msg::TrailerState::ConstSharedPtr msg);
  void onTimer();

  // Marker builders
  visualization_msgs::msg::MarkerArray buildMarkers(
    const nav_msgs::msg::Odometry & odom,
    const autoware_vehicle_msgs::msg::TrailerState & ts) const;

  visualization_msgs::msg::Marker makeBodyBoxMarker(
    int id, double trailer_axle_x, double trailer_axle_y, double trailer_yaw, double trailer_z,
    const std_msgs::msg::Header & header, const std::string & ns) const;

  visualization_msgs::msg::Marker makeMeshMarker(
    int id, double x, double y, double z, double yaw, const std_msgs::msg::Header & header,
    const MeshParams & mesh, const std::string & ns) const;

  static visualization_msgs::msg::Marker makeHitchMarker(
    int id, double x, double y, double z,
    const std_msgs::msg::Header & header);

  static visualization_msgs::msg::Marker makeAngleText(
    int id, double x, double y, double z,
    float hitch_angle_rad, const std_msgs::msg::Header & header);
};

}  // namespace autoware::trailer_state_visualizer

#endif  // AUTOWARE__TRAILER_STATE_VISUALIZER__TRAILER_STATE_VISUALIZER_HPP_
