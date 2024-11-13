// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <serial_driver/serial_driver.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "geometry_msgs/msg/twist.hpp"
#include "my_msg_interface/msg/power_heat.hpp"

// C++ system
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include "auto_aim_interfaces/msg/target.hpp"

namespace rm_serial_driver
{
class RMSerialDriver : public rclcpp::Node
{
public:
  explicit RMSerialDriver(const rclcpp::NodeOptions & options);

  ~RMSerialDriver() override;

private:
  void getParams();

  void receiveData();

  void sendData(auto_aim_interfaces::msg::Target::SharedPtr msg);

  void reopenPort();

  void setParam_color(const rclcpp::Parameter & param);

  void resetTracker();

  void NavCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void PowerHeatCallback(const my_msg_interface::msg::PowerHeat::SharedPtr msg);
  void timerCallback() ;

  void Detecting_callback(const std_msgs::msg::Int64 msg);


  //merged pack
  std::vector<uint8_t> vision_data;
  std::vector<uint8_t> nav_data;
  std::vector<uint8_t> power_heat_data;
  std::mutex vision_mutex;
  std::mutex nav_mutex;
  std::mutex power_heat_mutex;
  int rotation;
  // Serial port
  std::unique_ptr<IoContext> owned_ctx_;
  std::string device_name_;
  std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
  std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;

  // Param client to set detect_colr
  using ResultFuturePtr = std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>;
  bool initial_set_param_color = false;
  uint8_t previous_receive_color_ = 0;
  rclcpp::AsyncParametersClient::SharedPtr detector_param_client_;
  ResultFuturePtr set_param_future_color;
  // Service client to reset tracker
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_tracker_client_;

  // Aimimg point receiving from serial port for visualization
  visualization_msgs::msg::Marker aiming_point_;

  // Broadcast tf from odom to gimbal_link
  double timestamp_offset_ = 0;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr Nav_sub_;
  rclcpp::Subscription<my_msg_interface::msg::PowerHeat>::SharedPtr power_heat_sub_;

  // For debug usage
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr latency_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr control_id_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr test_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr send_check_pub_;
  std_msgs::msg::Float32 testmsg;
  std_msgs::msg::Float32 sendcheckmsg;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_pub_;
  int time_count;

  std::thread receive_thread_;

  int detect = 0;

  int last_detect = 0;

  int cnt = 10;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_pub_;

  std_msgs::msg::Float64 yaw_msg;

  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr detecting_sub_;
};
}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
