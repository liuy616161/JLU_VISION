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
#include <std_msgs/msg/int8.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>

// C++ system
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "auto_aim_interfaces/msg/target.hpp"
#include "global_interface/msg/serial.hpp"
#include "global_interface/msg/gimbal.hpp"

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

  void buffMsgCallback(global_interface::msg::Gimbal::SharedPtr gimbal_msg);

  // Serial port
  std::unique_ptr<IoContext> owned_ctx_;
  std::string device_name_;
  std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
  std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;

  // Param client to set detect_colr
  using ResultFuturePtr = std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>;
  bool initial_set_param_color = false;
  uint8_t previous_receive_color_ = 0;
  float imu_pitch = 0;
  float imu_yaw = 0;
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

  // For debug usage
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr latency_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  std::thread receive_thread_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_pub_;
  rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr sentry_decision_pub_;


  std_msgs::msg::Float64 yaw_msg;
  std_msgs::msg::Int8 sentry_decision_msg;
  std_msgs::msg::Int64 task;//0-aim 1-small_buff 2-large_buff
  std_msgs::msg::Float64 pitch_test;
  std_msgs::msg::Float64 pitch_calculate_msg;
  std_msgs::msg::Float64 pitch_imu_msg;
  std_msgs::msg::Int64 exposure_time_msg;// aim and buff are not the same exposure time

  int aim_et;
  int buff_et;
  int detect_color;

    // Task message
  rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr task_pub_;
  rclcpp::Publisher<global_interface::msg::Serial>::SharedPtr serial_msg_pub_;

  rclcpp::Subscription<global_interface::msg::Gimbal>::SharedPtr buff_info_sub_;
  

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr test_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pitch_calculate_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pitch_imu_;
  rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr exposure_time_pub_;

  // std_msgs::msg::Float64 test_msg;

  //   rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pitch_pub_;

  // std_msgs::msg::Float64 pitch_msg;

  // rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr camera_choice_publisher;

  // std_msgs::msg::Int64 camera_msg;

  //   int previous_camera = 1;
};
}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
