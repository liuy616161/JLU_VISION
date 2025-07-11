// STD
#include <memory>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

// ROS
#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"

#define pi 3.14159265358979323846

namespace rm_auto_aim
{
class ArmorTargeterNode : public rclcpp::Node
{
  using tf2_filter = tf2_ros::MessageFilter<auto_aim_interfaces::msg::Armors>;

  std::string target_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
  std::shared_ptr<tf2_filter> tf2_filter_;

  message_filters::Subscriber<auto_aim_interfaces::msg::Armors> armors_sub_;

  rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr target_pub_;

public:
  ArmorTargeterNode(const rclcpp::NodeOptions & options) : Node("armor_targeter", options)
  {
    // Subscriber with tf2 message_filter
    // tf2 relevant
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    // subscriber and filter
    armors_sub_.subscribe(this, "detector/armors", rmw_qos_profile_sensor_data);
    target_frame_ = this->declare_parameter<std::string>("frame_id");
    tf2_filter_ = std::make_shared<tf2_filter>(
      armors_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
      this->get_node_clock_interface(), std::chrono::duration<int>(1));
    // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
    tf2_filter_->registerCallback(&ArmorTargeterNode::armorsCallback, this);
    
    // Publisher
    target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
      "targeter/target", rclcpp::SensorDataQoS());
  }

private:
  void armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
  {
    // Tranform armor position from image frame to world coordinate
    for (auto & armor : armors_msg->armors) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = armors_msg->header;
      ps.pose = armor.pose;

      try {
        armor.pose = tf2_buffer_->transform(ps, target_frame_).pose;
      } catch (const tf2::ExtrapolationException & ex) {
        RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
        return;
      }
    }
    
    auto it = std::ranges::min_element(armors_msg->armors,
      [](auto&& x, auto&& y)
      {
        auto distance =
          [](auto&& armor)
          {
            return std::sqrt(
              armor.pose.position.x * armor.pose.position.x +
              armor.pose.position.y * armor.pose.position.y +
              armor.pose.position.z * armor.pose.position.z);
          };
        
        return distance(x) < distance(y);
      });
    
    auto_aim_interfaces::msg::Target target_msg;

    if (it == armors_msg->armors.end())
    {
      target_msg.number.clear();
    }
    else
    {
      target_msg.number = it->number;
      target_msg.angle = std::atan2(it->pose.position.y, it->pose.position.x) * 180 / pi;
    }

    target_pub_->publish(target_msg);
  }
};
}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTargeterNode)
