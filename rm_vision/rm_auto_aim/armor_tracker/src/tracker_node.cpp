// Copyright 2022 Chen Jun
#include "armor_tracker/tracker_node.hpp"

#include "armor_tracker/trajectory.h"

// STD
#include <memory>
#include <vector>
#include <cmath>

#define pi 3.14159265358979323846

namespace rm_auto_aim
{
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions & options)
: Node("armor_tracker", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");
  
  fire_data.resize(5, true);
  fire_data_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>(
    "/contorl_fire", 10, std::bind(&ArmorTrackerNode::fire_callback, this, std::placeholders::_1));
  
  yaw_sub_ = this->create_subscription<std_msgs::msg::Float64>(
    "robo_yaw", 10,std::bind(&ArmorTrackerNode::yawCallback, this, std::placeholders::_1));

  // Maximum allowable armor distance in the XOY plane
  max_armor_distance_ = this->declare_parameter<double>("max_armor_distance");

  // Tracker
  double max_match_distance = this->declare_parameter<double>("tracker.max_match_distance");
  double max_match_yaw_diff = this->declare_parameter<double>("tracker.max_match_yaw_diff");
  double tracking_thres = this->declare_parameter<int>("tracker.tracking_thres");
  double lost_time_thres_ = this->declare_parameter<double>("tracker.lost_time_thres");

  double s2qxyz_max = declare_parameter<double>("ekf.sigma2_q_xyz_max");
  double s2qxyz_min = declare_parameter<double>("ekf.sigma2_q_xyz_min");
  double s2qyaw_max = declare_parameter<double>("ekf.sigma2_q_yaw_max");
  double s2qyaw_min = declare_parameter<double>("ekf.sigma2_q_yaw_min");
  double s2qr = declare_parameter<double>("ekf.sigma2_q_r");
  double r_xyz_factor = declare_parameter<double>("ekf.r_xyz_factor");
  double r_yaw = declare_parameter<double>("ekf.r_yaw");

  tracker_ = std::make_unique<Tracker>(
    max_match_distance, max_match_yaw_diff,
    tracking_thres, lost_time_thres_,
    s2qxyz_max, s2qxyz_min,
    s2qyaw_max, s2qyaw_min,
    s2qr,
    r_xyz_factor, r_yaw);

  trajectory_bullet_offset_x_ = declare_parameter<double>("trajectory.bullet_offset_x");
  trajectory_bullet_offset_pitch_ = declare_parameter<double>("trajectory.bullet_offset_pitch");
  trajectory_bullet_offset_yaw_ = declare_parameter<double>("trajectory.bullet_offset_yaw");
  trajectory_bullet_speed_ = declare_parameter<double>("trajectory.bullet_speed");
  trajectory_algorithm_time_ = declare_parameter<double>("trajectory.algorithm_time");
  trajectory_control_time_ = declare_parameter<double>("trajectory.control_time");
  aim_threshold_armor_ = declare_parameter<double>("aim_threshold.armor");
  aim_threshold_center_ = declare_parameter<double>("aim_threshold.center");

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
  tf2_filter_->registerCallback(&ArmorTrackerNode::armorsCallback, this);

  // Publisher
  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrackedTarget>(
    "tracker/target", rclcpp::SensorDataQoS());
}

void ArmorTrackerNode::fire_callback(const std_msgs::msg::UInt8MultiArray::SharedPtr msg)
{
  fire_data = msg->data;
}

void ArmorTrackerNode::yawCallback(const std_msgs::msg::Float64 msg)
{
  robo_yaw = msg.data;
}

void ArmorTrackerNode::armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
  // Init message
  auto_aim_interfaces::msg::TrackedTarget target_msg;
  rclcpp::Time time = armors_msg->header.stamp;
  target_msg.header.stamp = time;
  target_msg.header.frame_id = target_frame_;

  transformArmors(armors_msg);
  
  // Update tracker
  if (updatePriority(armors_msg)) {
    tracker_->init(armors_msg);
  } else if (tracker_->tracker_state != Tracker::LOST) {
    tracker_->setDt((time - last_time_).seconds());
    tracker_->update(armors_msg);
  }

  last_time_ = time;

  if (
    tracker_->tracker_state == Tracker::LOST ||
    tracker_->tracker_state == Tracker::DETECTING) {
    target_msg.tracking = false;
    target_msg.detecting = tracker_->tracker_state == Tracker::DETECTING;
    target_msg.iffire = false;
  } else {
    const auto & state = tracker_->target_state;
    target_msg.tracking = true;
    target_msg.detecting = false;
    target_msg.id = tracker_->tracked_id;
    target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
    target_msg.position.x = state(0);
    target_msg.position.y = state(2);
    target_msg.position.z = state(4);
    target_msg.v_horizon = -std::sin(robo_yaw) * state(1) + std::cos(robo_yaw) * state(3);
    target_msg.aim_factor = std::abs(state(7));

    if (is_aiming_center_)
    {
      if (target_msg.aim_factor < aim_threshold_armor_)
      {
        is_aiming_center_ = false;
      }
    }
    else
    {
      if (target_msg.aim_factor > aim_threshold_center_)
      {
        is_aiming_center_ = true;
      }
    }
    
    float aim_x, aim_y, aim_z;
    PredictPitchXY &trajectory = PredictPitchXY::getinstance();
    
    if (is_aiming_center_)
    {
      trajectory.aim_center(
        trajectory_algorithm_time_, trajectory_control_time_,
        state(0), state(2), state(4), state(6),
        state(1), state(3), state(5), state(7),
        state(8), tracker_->another_r, tracker_->dz, target_msg.armors_num,
        aim_x, aim_y, aim_z,
        target_msg.r, target_msg.armor_yaw_offset, target_msg.fire);        
    }
    else
    {
      trajectory.aim_armor(
        trajectory_algorithm_time_, trajectory_control_time_,
        state(0), state(2), state(4), state(6),
        state(1), state(3), state(5), state(7),
        state(8), tracker_->another_r, tracker_->dz, target_msg.armors_num,
        aim_x, aim_y, aim_z,
        target_msg.r, target_msg.armor_yaw_offset, target_msg.fire);
    }
    
    target_msg.distance = std::sqrt(aim_x * aim_x + aim_y * aim_y);

    target_msg.pre_pitch = -trajectory_bullet_offset_pitch_ +
      trajectory(trajectory_bullet_offset_x_, trajectory_bullet_speed_, target_msg.distance - target_msg.r, aim_z); //正常的弹道解算
    target_msg.pre_yaw = -trajectory_bullet_offset_yaw_ +
      atan2(aim_y, aim_x) * 180 / pi;
    
    const static std::map<std::string, std::uint8_t> id_to_idx{
      {"1", 0}, {"2", 1}, {"3", 2}, {"4", 3}, {"guard", 4}};
    
    target_msg.iffire = !id_to_idx.contains(target_msg.id) || fire_data[id_to_idx.at(target_msg.id)];
  }

  target_pub_->publish(target_msg);
}

void ArmorTrackerNode::transformArmors(auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
  for (auto & armor : armors_msg->armors) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = armors_msg->header;
    ps.pose = armor.pose;
    try {
      armor.pose = tf2_buffer_->transform(ps, target_frame_).pose;
    } catch (const tf2::ExtrapolationException & ex) {
      RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
      armors_msg->armors.clear();
      return;
    }
  }
  
  // Filter abnormal armors
  std::erase_if(
    armors_msg->armors,
    [this](const auto_aim_interfaces::msg::Armor & armor) {
      return Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() > max_armor_distance_;
    });
}

bool ArmorTrackerNode::updatePriority(auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
  //Filter low level armor
  const static std::map<std::string, int> priority_map{
    {"",4}, {"outpost", 3},{"1", 1},{"2", 0},{"3", 2},{"4", 2},{"5", 3},{"guard", 3},{"base", 1}
  };
  
  if (tracker_->tracker_state == Tracker::LOST) {
    priority_level = 99;
  }

  int highest_level = priority_level;

  //检查装甲板优先级是否改变
  for (const auto & armor : armors_msg->armors) {
    if (priority_map.at(armor.number) < highest_level) {
      highest_level = priority_map.at(armor.number);
    } 
  }

  //抹去低优先级的装甲板
  std::erase_if(
    armors_msg->armors,
    [&](const auto_aim_interfaces::msg::Armor & armor) {
      return priority_map.at(armor.number) > highest_level;
    });
  
  if (highest_level != priority_level)
  {
    priority_level = highest_level;
    return true;
  }
  else
  {
    return false;
  }
}
}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)
