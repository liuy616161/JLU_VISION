#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <serial_driver/serial_driver.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <rcpputils/filesystem_helper.hpp>

#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/tracked_target.hpp"
#include "rm_serial_driver/packet.hpp"
#include "tars_msgs/msg/serial_receive.hpp"
#include "tars_msgs/msg/serial_send.hpp"


#include "std_msgs/msg/u_int8_multi_array.hpp"
#include "std_msgs/msg/u_int8.hpp"


namespace rm_serial_driver
{

class RMSerialDriver : public rclcpp::Node
{
using ResultFuturePtr = std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>;

public:
  explicit RMSerialDriver(const rclcpp::NodeOptions & options);
  ~RMSerialDriver() override;

private:
  void getParams();
  void reopenPort();
  void setParam_color(bool& initial_set_param,
    const rclcpp::Parameter & param, rclcpp::AsyncParametersClient::SharedPtr& param_client, ResultFuturePtr& param_future);
  bool isSerialDeviceExists(const std::string& device_name);

  void VisionCallback(auto_aim_interfaces::msg::TrackedTarget::SharedPtr msg);
  void NavCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void TuoluoCallback(const std_msgs::msg::Int8::SharedPtr msg);
  void BackArmorCallback(const auto_aim_interfaces::msg::Target::SharedPtr msg);
  void TimerCallback();
  
  void receiveData();

  // ******************** TF2 ********************
  std::unique_ptr<tf2_ros::TransformBroadcaster> vision_tf_broadcaster_; // Broadcast tf from odom to gimbal_link
  std::unique_ptr<tf2_ros::TransformBroadcaster> nav_tf_broadcaster_;    // Broadcast tf from base_frame to chassis_link

  // ******************** ROS Subscription ********************
  // Vision Subscription
  rclcpp::Subscription<auto_aim_interfaces::msg::TrackedTarget>::SharedPtr target_sub_;

  // Nav Subscription
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr nav_sub_;
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr tuoluo_sub_;

  // Back Vision Subscription
  rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr back_armor_sub_;

  // ******************** ROS Publisher ********************
  // Vision Publisher
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr revivatory_car_pub_;

  // Serial Publisher
  rclcpp::Publisher<tars_msgs::msg::SerialReceive>::SharedPtr serial_rcv_pub_;
  rclcpp::Publisher<tars_msgs::msg::SerialSend>::SharedPtr serial_send_pub_;


  // ******************** ROS Service and Client ********************
  rclcpp::AsyncParametersClient::SharedPtr detector_param_client_;
  rclcpp::AsyncParametersClient::SharedPtr back_detector_param_client_;


  // ******************** ROS Timer ********************
  rclcpp::TimerBase::SharedPtr timer_;


  // ******************** Merged Packages ********************
  VisionPacket vision_packet;
  NavPacket nav_packet;
  BackArmorPacket back_armor_packet;
  TotalSendPacket total_send_packet;
  std::vector<uint8_t> total_send_data;  // packet -> byte stream


  // ******************** Mutex ********************
  std::mutex vision_mutex;
  std::mutex nav_mutex;
  std::mutex back_armor_mutex;


  // ******************** ROS Messages ********************
  std_msgs::msg::Float64 yaw_msg;
  std_msgs::msg::Float64 pitch_msg;
  tars_msgs::msg::SerialReceive rcv_msg;
  tars_msgs::msg::SerialSend send_msg;
  std_msgs::msg::UInt8MultiArray revivatory_car_msg;


  // ******************** ROS Parameters ********************
  std::string device_name_;
  int rotation;
  std::string cmd_vel_topic_;
  double timestamp_offset_ = 0;
  std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;


  // ******************** Related Parameters ********************
  int flag_count = 0; 
  uint8_t another_fire;
  int detect = 0;
  int last_detect = 0;
  int cnt = 10;
  char spinFireflag = 0;
  char gimbalfire = 0;
  char tuoluo_ = 0;
  // Param client to set detect_colr
  bool initial_set_param_color = false;
  bool initial_set_back_param_color = false;
  uint8_t previous_receive_color_ = 0;
  ResultFuturePtr set_param_future_color;
  ResultFuturePtr set_back_param_future_color;
  
  // ******************** Serial port ********************
  std::unique_ptr<IoContext> owned_ctx_;
  std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;
  std::thread receive_thread_;

};
} // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
