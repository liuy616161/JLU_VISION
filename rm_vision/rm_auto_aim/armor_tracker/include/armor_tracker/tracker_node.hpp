// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
#define ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_

// STD
#include <memory>
#include <string>
#include <chrono>
#include <vector>

// ROS
#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "std_msgs/msg/float32.hpp"
#include <std_msgs/msg/float64.hpp>
#include "std_msgs/msg/u_int8_multi_array.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include <std_msgs/msg/int8.hpp>
#include "std_msgs/msg/int64.hpp"
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "armor_tracker/tracker.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/tracked_target.hpp"

namespace rm_auto_aim
{
using tf2_filter = tf2_ros::MessageFilter<auto_aim_interfaces::msg::Armors>;

class ArmorTrackerNode : public rclcpp::Node
{
public:
  explicit ArmorTrackerNode(const rclcpp::NodeOptions & options);

private:
  void armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_ptr);

  void yawCallback(const std_msgs::msg::Float64 msg);
  
  void fire_callback(const std_msgs::msg::UInt8MultiArray::SharedPtr msg);

  void transformArmors(auto_aim_interfaces::msg::Armors::SharedPtr armors_msg);

  bool updatePriority(auto_aim_interfaces::msg::Armors::SharedPtr armors_msg);

  double max_armor_distance_;
  std::unique_ptr<Tracker> tracker_;

  double trajectory_bullet_offset_x_;
  double trajectory_bullet_offset_pitch_;
  double trajectory_bullet_offset_yaw_;
  double trajectory_bullet_speed_;
  double trajectory_algorithm_time_;
  double trajectory_control_time_;
  double aim_threshold_armor_;
  double aim_threshold_center_;
  bool is_aiming_center_ = false;
  
  int priority_level = 0;

  // The time when the last message was received
  rclcpp::Time last_time_;

  // Subscriber
  std::string target_frame_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
  message_filters::Subscriber<auto_aim_interfaces::msg::Armors> armors_sub_;
  std::shared_ptr<tf2_filter> tf2_filter_;
  
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr yaw_sub_;
  float robo_yaw = 0;
  
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr fire_data_;
  std::vector<uint8_t> fire_data;  // 用于存储接收到的数据
  
  // Publisher
  rclcpp::Publisher<auto_aim_interfaces::msg::TrackedTarget>::SharedPtr target_pub_;
};
}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
