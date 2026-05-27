#include <chrono>
#include <memory>

#include "autoware_vehicle_msgs/msg/gear_report.hpp"
#include "rclcpp/rclcpp.hpp"

using autoware_vehicle_msgs::msg::GearReport;
using namespace std::chrono_literals;

class FixedGearReportPublisher : public rclcpp::Node
{
public:
  FixedGearReportPublisher()
  : Node("fixed_gear_report_publisher")
  {
    const auto publish_rate_hz = declare_parameter("publish_rate_hz", 10.0);
    const auto publish_period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_rate_hz));

    publisher_ = create_publisher<GearReport>("/vehicle/status/gear_status", rclcpp::QoS{10});
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(publish_period),
      std::bind(&FixedGearReportPublisher::onTimer, this));
  }

private:
  void onTimer()
  {
    GearReport message;
    message.stamp = now();
    message.report = GearReport::DRIVE;
    publisher_->publish(message);
  }

  rclcpp::Publisher<GearReport>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FixedGearReportPublisher>());
  rclcpp::shutdown();
  return 0;
}
