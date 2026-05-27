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

#include "autoware/trailer_state_visualizer/trailer_state_visualizer.hpp"

#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace autoware::trailer_state_visualizer
{

TrailerStateVisualizer::TrailerStateVisualizer(const rclcpp::NodeOptions & options)
: Node("trailer_state_visualizer", options),
  vehicle_info_(autoware::vehicle_info_utils::VehicleInfoUtils(*this).getVehicleInfo())
{
  // By default, place hitch at rear bumper from vehicle_info (can be overridden).
  const double default_hitch_offset = -vehicle_info_.rear_overhang_m;
  hitch_offset_m_ = declare_parameter<double>("hitch_offset", default_hitch_offset);

  // Trailer box / kinematics parameters
  trailer_.wheelbase = declare_parameter<double>("trailer.wheelbase", 6.0);
  trailer_.width = declare_parameter<double>("trailer.width", vehicle_info_.vehicle_width_m);
  trailer_.front_overhang = declare_parameter<double>("trailer.front_overhang", 0.2);
  trailer_.rear_overhang = declare_parameter<double>("trailer.rear_overhang", 0.5);
  trailer_.height = declare_parameter<double>("trailer.height", vehicle_info_.vehicle_height_m);

  hitch_marker_z_ = declare_parameter<double>("hitch_marker_z", 0.5);
  publish_hitch_marker_ = declare_parameter<bool>("publish_hitch_marker", true);
  publish_hitch_text_ = declare_parameter<bool>("publish_hitch_text", true);
  publish_trailer_tf_ = declare_parameter<bool>("publish_trailer_tf", false);

  const auto parse_vec3 = [this](const std::string & name, const std::vector<double> & default_value) {
    const auto values = declare_parameter<std::vector<double>>(name, default_value);
    if (values.size() != 3U) {
      throw std::invalid_argument(name + " must contain exactly 3 elements");
    }
    return values;
  };

  const auto truck_scale = parse_vec3("truck_mesh.scale", {0.5, 0.5, 0.5});
  const auto truck_offset_xyz = parse_vec3(
    "truck_mesh.offset_xyz", {vehicle_info_.wheel_base_m / 2.0, 0.0, 0.0});
  const auto truck_offset_rpy = parse_vec3("truck_mesh.offset_rpy", {0.0, 0.0, -1.5708});
  truck_mesh_.enable = declare_parameter<bool>("truck_mesh.enable", true);
  truck_mesh_.resource = declare_parameter<std::string>(
    "truck_mesh.resource",
    "package://electrans_robot_vehicle_description/mesh/electrans_tractor.dae");
  truck_mesh_.scale_x = truck_scale.at(0);
  truck_mesh_.scale_y = truck_scale.at(1);
  truck_mesh_.scale_z = truck_scale.at(2);
  truck_mesh_.offset_x = truck_offset_xyz.at(0);
  truck_mesh_.offset_y = truck_offset_xyz.at(1);
  truck_mesh_.offset_z = truck_offset_xyz.at(2);
  truck_mesh_.roll = truck_offset_rpy.at(0);
  truck_mesh_.pitch = truck_offset_rpy.at(1);
  truck_mesh_.yaw = truck_offset_rpy.at(2);

  const auto trailer_scale = parse_vec3("trailer_mesh.scale", {0.5, 0.5, 0.5});
  const auto trailer_offset_xyz = parse_vec3("trailer_mesh.offset_xyz", {0.0, 0.0, 0.0});
  const auto trailer_offset_rpy = parse_vec3("trailer_mesh.offset_rpy", {0.0, 0.0, 0.0});
  trailer_mesh_.enable = declare_parameter<bool>("trailer_mesh.enable", true);
  trailer_mesh_.resource = declare_parameter<std::string>(
    "trailer_mesh.resource",
    "package://electrans_robot_vehicle_description/mesh/electrans_trailer.dae");
  trailer_mesh_.scale_x = trailer_scale.at(0);
  trailer_mesh_.scale_y = trailer_scale.at(1);
  trailer_mesh_.scale_z = trailer_scale.at(2);
  trailer_mesh_.offset_x = trailer_offset_xyz.at(0);
  trailer_mesh_.offset_y = trailer_offset_xyz.at(1);
  trailer_mesh_.offset_z = trailer_offset_xyz.at(2);
  trailer_mesh_.roll = trailer_offset_rpy.at(0);
  trailer_mesh_.pitch = trailer_offset_rpy.at(1);
  trailer_mesh_.yaw = trailer_offset_rpy.at(2);

  const double update_rate = declare_parameter<double>("update_rate", 10.0);

  using std::placeholders::_1;

  sub_odometry_ = create_subscription<nav_msgs::msg::Odometry>(
    "~/input/kinematic_state", rclcpp::QoS{1},
    std::bind(&TrailerStateVisualizer::onOdometry, this, _1));

  sub_trailer_state_ = create_subscription<autoware_vehicle_msgs::msg::TrailerState>(
    "~/input/trailer_state", rclcpp::QoS{1},
    std::bind(&TrailerStateVisualizer::onTrailerState, this, _1));

  pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "~/output/markers", rclcpp::QoS{1});

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  const auto period = std::chrono::duration<double>(1.0 / update_rate);
  timer_ = create_wall_timer(period, std::bind(&TrailerStateVisualizer::onTimer, this));
}

void TrailerStateVisualizer::onOdometry(nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  odometry_ptr_ = msg;
}

void TrailerStateVisualizer::onTrailerState(
  autoware_vehicle_msgs::msg::TrailerState::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  trailer_state_ptr_ = msg;
}

void TrailerStateVisualizer::onTimer()
{
  nav_msgs::msg::Odometry::ConstSharedPtr odom;
  autoware_vehicle_msgs::msg::TrailerState::ConstSharedPtr ts;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    odom = odometry_ptr_;
    ts = trailer_state_ptr_;
  }

  if (!odom || !ts) {
    return;
  }

  pub_markers_->publish(buildMarkers(*odom, *ts));

  if (publish_trailer_tf_) {
    // Trailer reference = trailer rear axle (mirrors base_link convention)
    const double truck_x = odom->pose.pose.position.x;
    const double truck_y = odom->pose.pose.position.y;
    const double truck_yaw = tf2::getYaw(odom->pose.pose.orientation);
    const double hitch_x = truck_x + hitch_offset_m_ * std::cos(truck_yaw);
    const double hitch_y = truck_y + hitch_offset_m_ * std::sin(truck_yaw);
    const double trailer_yaw = truck_yaw - static_cast<double>(ts->hitch_angle);
    const double trailer_axle_x = hitch_x - trailer_.wheelbase * std::cos(trailer_yaw);
    const double trailer_axle_y = hitch_y - trailer_.wheelbase * std::sin(trailer_yaw);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = odom->header.stamp;
    tf_msg.header.frame_id = odom->header.frame_id;
    tf_msg.child_frame_id = "trailer_link";
    tf_msg.transform.translation.x = trailer_axle_x;
    tf_msg.transform.translation.y = trailer_axle_y;
    tf_msg.transform.translation.z = odom->pose.pose.position.z;
    tf_msg.transform.rotation.x = 0.0;
    tf_msg.transform.rotation.y = 0.0;
    tf_msg.transform.rotation.z = std::sin(trailer_yaw / 2.0);
    tf_msg.transform.rotation.w = std::cos(trailer_yaw / 2.0);
    tf_broadcaster_->sendTransform(tf_msg);
  }
}

visualization_msgs::msg::MarkerArray TrailerStateVisualizer::buildMarkers(
  const nav_msgs::msg::Odometry & odom,
  const autoware_vehicle_msgs::msg::TrailerState & ts) const
{
  visualization_msgs::msg::MarkerArray marker_array;

  std_msgs::msg::Header header;
  header.stamp = odom.header.stamp;
  header.frame_id = odom.header.frame_id;

  const double truck_x = odom.pose.pose.position.x;
  const double truck_y = odom.pose.pose.position.y;
  const double truck_yaw = tf2::getYaw(odom.pose.pose.orientation);
  const double trailer_z = odom.pose.pose.position.z;

  // Hitch point in map frame
  const double hitch_x = truck_x + hitch_offset_m_ * std::cos(truck_yaw);
  const double hitch_y = truck_y + hitch_offset_m_ * std::sin(truck_yaw);

  // Trailer heading
  const double trailer_yaw = truck_yaw - static_cast<double>(ts.hitch_angle);
  const double trailer_axle_x = hitch_x - trailer_.wheelbase * std::cos(trailer_yaw);
  const double trailer_axle_y = hitch_y - trailer_.wheelbase * std::sin(trailer_yaw);

  // 0: Truck body marker (reference = rear axle / base_link).
  if (truck_mesh_.enable && !truck_mesh_.resource.empty()) {
    marker_array.markers.push_back(
      makeMeshMarker(0, truck_x, truck_y, trailer_z, truck_yaw, header, truck_mesh_, "truck_body"));
  } else {
    marker_array.markers.push_back(
      makeBodyBoxMarker(0, truck_x, truck_y, truck_yaw, trailer_z, header, "truck_body"));
  }

  // 1: Trailer body marker (reference = trailer rear axle).
  if (trailer_mesh_.enable && !trailer_mesh_.resource.empty()) {
    marker_array.markers.push_back(
      makeMeshMarker(
        1, trailer_axle_x, trailer_axle_y, trailer_z, trailer_yaw, header, trailer_mesh_,
        "trailer_body"));
  } else {
    marker_array.markers.push_back(
      makeBodyBoxMarker(1, trailer_axle_x, trailer_axle_y, trailer_yaw, trailer_z, header, "trailer_body"));
  }

  if (publish_hitch_marker_) {
    // 2: Hitch point sphere
    marker_array.markers.push_back(makeHitchMarker(2, hitch_x, hitch_y, hitch_marker_z_, header));
  }

  if (publish_hitch_text_) {
    // 3: Hitch angle text label
    marker_array.markers.push_back(
      makeAngleText(3, hitch_x, hitch_y, hitch_marker_z_ + 1.5, ts.hitch_angle, header));
  }

  return marker_array;
}

visualization_msgs::msg::Marker TrailerStateVisualizer::makeBodyBoxMarker(
  int id, double trailer_axle_x, double trailer_axle_y, double trailer_yaw, double trailer_z,
  const std_msgs::msg::Header & header, const std::string & ns) const
{
  const double front_length = trailer_.wheelbase + trailer_.front_overhang;
  const double rear_length = trailer_.rear_overhang;
  const double trailer_length = front_length + rear_length;
  const double center_offset_x = 0.5 * (front_length - rear_length);

  visualization_msgs::msg::Marker m;
  m.header = header;
  m.ns = ns;
  m.id = id;
  m.type = visualization_msgs::msg::Marker::CUBE;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.position.x = trailer_axle_x + center_offset_x * std::cos(trailer_yaw);
  m.pose.position.y = trailer_axle_y + center_offset_x * std::sin(trailer_yaw);
  m.pose.position.z = trailer_z + 0.5 * trailer_.height;
  m.pose.orientation.x = 0.0;
  m.pose.orientation.y = 0.0;
  m.pose.orientation.z = std::sin(trailer_yaw / 2.0);
  m.pose.orientation.w = std::cos(trailer_yaw / 2.0);
  m.scale.x = trailer_length;
  m.scale.y = trailer_.width;
  m.scale.z = trailer_.height;
  m.color.r = 1.0F;
  m.color.g = 0.55F;
  m.color.b = 0.1F;
  m.color.a = 0.55F;
  m.lifetime = rclcpp::Duration::from_seconds(0.5);
  return m;
}

visualization_msgs::msg::Marker TrailerStateVisualizer::makeMeshMarker(
  int id, double x, double y, double z, double yaw, const std_msgs::msg::Header & header,
  const MeshParams & mesh, const std::string & ns) const
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);

  const double mx = x + c * mesh.offset_x - s * mesh.offset_y;
  const double my = y + s * mesh.offset_x + c * mesh.offset_y;
  const double mz = z + mesh.offset_z;
  const double myaw = yaw + mesh.yaw;

  visualization_msgs::msg::Marker m;
  m.header = header;
  m.ns = ns;
  m.id = id;
  m.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.mesh_resource = mesh.resource;
  m.pose.position.x = mx;
  m.pose.position.y = my;
  m.pose.position.z = mz;

  tf2::Quaternion q;
  q.setRPY(mesh.roll, mesh.pitch, myaw);
  m.pose.orientation.x = q.x();
  m.pose.orientation.y = q.y();
  m.pose.orientation.z = q.z();
  m.pose.orientation.w = q.w();

  m.scale.x = mesh.scale_x;
  m.scale.y = mesh.scale_y;
  m.scale.z = mesh.scale_z;
  m.color.r = 1.0F;
  m.color.g = 1.0F;
  m.color.b = 1.0F;
  m.color.a = 1.0F;
  m.mesh_use_embedded_materials = true;
  m.lifetime = rclcpp::Duration::from_seconds(0.5);
  return m;
}

visualization_msgs::msg::Marker TrailerStateVisualizer::makeHitchMarker(
  int id, double x, double y, double z, const std_msgs::msg::Header & header)
{
  visualization_msgs::msg::Marker m;
  m.header = header;
  m.ns = "hitch_point";
  m.id = id;
  m.type = visualization_msgs::msg::Marker::SPHERE;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.position.x = x;
  m.pose.position.y = y;
  m.pose.position.z = z;
  m.pose.orientation.w = 1.0;
  m.scale.x = m.scale.y = m.scale.z = 0.3;
  m.color.r = 1.0f;
  m.color.g = 1.0f;
  m.color.b = 0.0f;
  m.color.a = 1.0f;
  m.lifetime = rclcpp::Duration::from_seconds(0.5);
  return m;
}

visualization_msgs::msg::Marker TrailerStateVisualizer::makeAngleText(
  int id, double x, double y, double z,
  float hitch_angle_rad, const std_msgs::msg::Header & header)
{
  visualization_msgs::msg::Marker m;
  m.header = header;
  m.ns = "hitch_angle";
  m.id = id;
  m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.position.x = x;
  m.pose.position.y = y;
  m.pose.position.z = z;
  m.pose.orientation.w = 1.0;
  m.scale.z = 0.5;
  m.color.r = 1.0f;
  m.color.g = 1.0f;
  m.color.b = 1.0f;
  m.color.a = 1.0f;
  m.lifetime = rclcpp::Duration::from_seconds(0.5);

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1)
      << "hitch: " << (static_cast<double>(hitch_angle_rad) * 180.0 / M_PI) << " deg";
  m.text = oss.str();

  return m;
}

}  // namespace autoware::trailer_state_visualizer

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::trailer_state_visualizer::TrailerStateVisualizer)
