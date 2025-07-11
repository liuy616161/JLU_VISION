// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#include <tf2/LinearMath/Quaternion.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <serial_driver/serial_driver.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <arpa/inet.h>

#include <cstdint>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rm_serial_driver/crc.hpp"
#include "rm_serial_driver/rm_serial_driver.hpp"

#define pi 3.14159265358979323846

namespace rm_serial_driver
{

RMSerialDriver::RMSerialDriver(const rclcpp::NodeOptions &options)
    : Node("rm_serial_driver", options),
      owned_ctx_{new IoContext(2)},
      serial_driver_{new drivers::serial_driver::SerialDriver(*owned_ctx_)}
{
  using std::placeholders::_1;

  getParams();
  total_send_data.assign(sizeof(TotalSendPacket), 0x00);
  
  // ****************************** ROS Params ******************************
  timestamp_offset_ = this->declare_parameter("timestamp_offset", 0.0);

  // ****************************** TF2 ******************************
  vision_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  nav_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // ****************************** ROS Subscription ******************************
  // Vision Subscription
  target_sub_ = 
      this->create_subscription<auto_aim_interfaces::msg::TrackedTarget>(
      "front/tracker/target", rclcpp::SensorDataQoS(),
      std::bind(&RMSerialDriver::VisionCallback, this, _1));

  // Nav Subscription
  nav_sub_ = 
      this->create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RMSerialDriver::NavCallback, this, _1));
  
  tuoluo_sub_ = 
      this->create_subscription<std_msgs::msg::Int8>(
      "tuoluo", 5,
      std::bind(&RMSerialDriver::TuoluoCallback, this, _1));

  // Back Vision Subscription
  back_armor_sub_ = 
      this->create_subscription<auto_aim_interfaces::msg::Target>(
      "back/targeter/target", rclcpp::SensorDataQoS(),
      std::bind(&RMSerialDriver::BackArmorCallback, this, _1));

  
  // ****************************** ROS Publisher ******************************
  // Vision Publisher
  yaw_pub_ = this->create_publisher<std_msgs::msg::Float64>("front/robo_yaw", 10);
  revivatory_car_pub_ = this->create_publisher<std_msgs::msg::UInt8MultiArray>("revivatory_car", 10);

  // Serial Publisher
  serial_rcv_pub_ = this->create_publisher<tars_msgs::msg::SerialReceive>("serial_receive", 5);
  serial_send_pub_ = this->create_publisher<tars_msgs::msg::SerialSend>("serial_send", 5);

  // ****************************** ROS Service and Client ******************************
  detector_param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, "front/armor_detector");
  back_detector_param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, "back/armor_detector");

  // ****************************** Serial Driver Init ******************************
  while(true)
  {
    try
    {
      // 首先检查设备是否存在
      if (!isSerialDeviceExists(device_name_))
      {
        RCLCPP_WARN(get_logger(), "Serial device %s not found, retrying in 1 second...", 
                    device_name_.c_str());
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      
      // 尝试初始化和打开串口
      serial_driver_->init_port(device_name_, *device_config_);
      
      if (!serial_driver_->port()->is_open())
      {
        serial_driver_->port()->open();
      }
      
      // 如果成功打开，启动接收线程并退出循环
      if (serial_driver_->port()->is_open())
      {
        RCLCPP_INFO(get_logger(), "Serial port %s opened successfully", device_name_.c_str());
        receive_thread_ = std::thread(&RMSerialDriver::receiveData, this);
        break;
      }
    }
    catch (const std::exception &ex)
    {
      RCLCPP_ERROR(get_logger(), "Error with serial port %s: %s", 
                  device_name_.c_str(), ex.what());
      
      // 发生异常时等待一段时间再重试
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  // ****************************** ROS Timer ******************************
  timer_ = this->create_wall_timer(std::chrono::milliseconds(4), 
            std::bind(&RMSerialDriver::TimerCallback, this));
  
  RCLCPP_INFO(get_logger(), "======== Serial Driver Started! ========");
}

RMSerialDriver::~RMSerialDriver()
{
  if (receive_thread_.joinable())
  {
    receive_thread_.join();
  }

  if (serial_driver_->port()->is_open())
  {
    serial_driver_->port()->close();
  }

  if (owned_ctx_)
  {
    owned_ctx_->waitForExit();
  }
}

bool RMSerialDriver::isSerialDeviceExists(const std::string& device_name)
{
  return rcpputils::fs::exists(rcpputils::fs::path(device_name));
}

void RMSerialDriver::VisionCallback(const auto_aim_interfaces::msg::TrackedTarget::SharedPtr msg)
{
  const static std::map<std::string, uint8_t> id_unit8_map{
      {"", 0}, {"outpost", 0}, {"1", 1}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"guard", 6}, {"base", 7}};

  try
  {
    vision_mutex.lock();
    vision_packet.tracking = msg->tracking;
    vision_packet.id = id_unit8_map.at(msg->id);
    vision_packet.armors_num = msg->armors_num;
    vision_packet.detecting = msg->detecting;
    vision_packet.yaw = msg->pre_yaw * 1000;
    vision_packet.pitch = msg->pre_pitch * 1000;
    vision_packet.fire = msg->fire;
    vision_packet.distance = msg->distance;
    vision_packet.r = msg->r;
    vision_packet.v_horizon = msg->v_horizon;
    vision_packet.armor_yaw_offset = msg->armor_yaw_offset;
    vision_packet.another_fire = msg->iffire;
    vision_mutex.unlock();
  }
  catch (const std::exception &ex)
  {
    RCLCPP_ERROR(get_logger(), "Error from vision callback: %s", ex.what());
    reopenPort();
  }
}

void RMSerialDriver::NavCallback(const geometry_msgs::msg::Twist::SharedPtr msg) 
{
  try
  {
    nav_mutex.lock();
    nav_packet.linear_x = msg->linear.y;
    nav_packet.linear_y = -msg->linear.x;
    nav_packet.angular_z = -msg->angular.z;
    nav_mutex.unlock();
    RCLCPP_INFO(this->get_logger(), "Nav x: %lf, y: %lf, z: %lf", 
                msg->linear.x, msg->linear.y, msg->angular.z);
  }
  catch(const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Error from nav callback: %s", e.what());
  }
}

void RMSerialDriver::BackArmorCallback(const auto_aim_interfaces::msg::Target::SharedPtr msg) 
{
  try
  {
    back_armor_mutex.lock();
    back_armor_packet.number = msg->number[0];
    back_armor_packet.angle = msg->angle;
    back_armor_mutex.unlock();
  }
  catch(const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Error from back armor callback: %s", e.what());
  }
}

void RMSerialDriver::TuoluoCallback(const std_msgs::msg::Int8::SharedPtr msg)
{
  try
  {
    nav_mutex.lock();
    tuoluo_ = msg->data;
    nav_packet.tuoluo = tuoluo_;
    nav_mutex.unlock();
  }
  catch(const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Error from tuoluo callback: %s", e.what());
  }
}


void RMSerialDriver::TimerCallback() 
{
  std::lock(vision_mutex, nav_mutex, back_armor_mutex);

  total_send_packet.vision_pck = vision_packet;
  total_send_packet.nav_pck = nav_packet;
  total_send_packet.back_armor_pck = back_armor_packet;

  total_send_data.resize(sizeof(TotalSendPacket));
  total_send_data = toVector(total_send_packet);

  send_msg.tracking = vision_packet.tracking;
  send_msg.id = vision_packet.id;
  send_msg.armors_num = vision_packet.armors_num;
  send_msg.detecting = vision_packet.detecting;
  send_msg.yaw = vision_packet.yaw;
  send_msg.pitch = vision_packet.pitch;
  send_msg.fire = vision_packet.fire;
  send_msg.distance = vision_packet.distance;
  send_msg.r = vision_packet.r;
  send_msg.v_horizon = vision_packet.v_horizon;
  send_msg.armor_yaw_offset = vision_packet.armor_yaw_offset;
  send_msg.another_fire = vision_packet.another_fire;

  send_msg.linear_x = nav_packet.linear_x;
  send_msg.linear_y = nav_packet.linear_y;
  send_msg.angular_z = nav_packet.angular_z;
  send_msg.tuoluo = nav_packet.tuoluo;

  send_msg.back_armor_number = back_armor_packet.number;
  send_msg.back_armor_angle = back_armor_packet.angle;

  serial_send_pub_->publish(send_msg);

  vision_mutex.unlock();
  nav_mutex.unlock();
  back_armor_mutex.unlock();

  try 
  {
    serial_driver_->port()->send(total_send_data);
    if(flag_count % 1000 == 0) {
      RCLCPP_INFO(this->get_logger(),"+++++ serial send OK ! +++++");
      flag_count = 0;
    }
    flag_count ++;

    /*static auto last_time = this->now();
    auto current_time = this->now();
    RCLCPP_INFO(this->get_logger(), "%dms from last transmition", (int)((current_time - last_time).seconds() * 1000));
    last_time = current_time;*/
  } 
  catch (const std::exception& e) 
  {
    RCLCPP_ERROR(this->get_logger(),"Error from timer callback: %s",e.what());
  }
}


void RMSerialDriver::receiveData()
{
  std::vector<uint8_t> header(1);
  std::vector<uint8_t> rcv_data;
  rcv_data.reserve(sizeof(ReceivePacket));

  while (rclcpp::ok())
  {
    try
    {
      serial_driver_->port()->receive(header);

      if (header[0] == 0x5A)
      {
        rcv_data.resize(sizeof(ReceivePacket) - 1);
        serial_driver_->port()->receive(rcv_data);
        rcv_data.insert(rcv_data.begin(), header[0]);
        ReceivePacket rcv_packet = fromVector(rcv_data);

        bool detect_color_modified = rcv_packet.detect_color != previous_receive_color_;

        if (!initial_set_param_color || detect_color_modified)
        {
          setParam_color(initial_set_param_color,
            rclcpp::Parameter("light.detect_color", rcv_packet.detect_color), detector_param_client_, set_param_future_color);
          previous_receive_color_ = rcv_packet.detect_color;
        }
        
        if (!initial_set_back_param_color || detect_color_modified) 
        {
          setParam_color(initial_set_back_param_color,
            rclcpp::Parameter("light.detect_color", rcv_packet.detect_color), back_detector_param_client_, set_back_param_future_color);
          previous_receive_color_ = rcv_packet.detect_color;
        }
        
        geometry_msgs::msg::TransformStamped t;
        timestamp_offset_ = this->get_parameter("timestamp_offset").as_double();
        t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
        t.header.frame_id = "base_frame";
        t.child_frame_id = "gimbal_link";
        tf2::Quaternion q;
        q.setRPY( rcv_packet.roll / 57.2957 / 1000.0, 
                  -rcv_packet.pitch / 57.2957 / 1000.0, 
                  rcv_packet.yaw / 57.2957 / 1000.0);
        
        yaw_msg.data = rcv_packet.yaw / 57.2957 / 1000.0;
        yaw_pub_->publish(yaw_msg);

        t.transform.rotation = tf2::toMsg(q);
        vision_tf_broadcaster_->sendTransform(t);

        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = this->now();
        transform.header.frame_id = "base_frame";
        transform.child_frame_id = "chassis_link";

        transform.transform.translation.x = 0.0;
        transform.transform.translation.y = 0.0;
        transform.transform.translation.z = 0.0;

        auto quat = tf2::Quaternion();
        quat.setRPY(0.0, 0.0, -(rcv_packet.yaw_delta) * M_PI / 180.0);
        quat.normalize();
        transform.transform.rotation.x = quat.x();
        transform.transform.rotation.y = quat.y();
        transform.transform.rotation.z = quat.z();
        transform.transform.rotation.w = quat.w();

        nav_tf_broadcaster_->sendTransform(transform);

        rcv_msg.detect_color = rcv_packet.detect_color;
        rcv_msg.reset_tracker = rcv_packet.reset_tracker;
        rcv_msg.reserved = rcv_packet.reserved;
        rcv_msg.roll = rcv_packet.roll;
        rcv_msg.pitch = rcv_packet.pitch;
        rcv_msg.yaw = rcv_packet.yaw;
        rcv_msg.control_id = rcv_packet.control_id;
        rcv_msg.revivatory_car = rcv_packet.revivatory_car;

        rcv_msg.yaw_delta = rcv_packet.yaw_delta;

        rcv_msg.game_progress = rcv_packet.game_progress;
        rcv_msg.remain_time = rcv_packet.remain_time;
        rcv_msg.current_hp = rcv_packet.current_hp;
        rcv_msg.projectile = rcv_packet.projectile;
        rcv_msg.sentry_info = rcv_packet.sentry_info;
        rcv_msg.red_outpost_hp = rcv_packet.red_outpost_hp;
        rcv_msg.red_base_hp = rcv_packet.red_base_hp;
        rcv_msg.blue_outpost_hp = rcv_packet.blue_outpost_hp;
        rcv_msg.blue_base_hp = rcv_packet.blue_base_hp;

        uint8_t revivatory_byte = rcv_packet.revivatory_car;
        std::vector<uint8_t> bit_array(5);
        for (int i = 0; i < 5; ++i) {
            bit_array[i] = (revivatory_byte >> (i)) & 0x01;
        }
        revivatory_car_msg.data = bit_array;
        revivatory_car_pub_ ->publish(revivatory_car_msg);

        serial_rcv_pub_->publish(rcv_msg);
      }
      else
      {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 20, "Invalid header: %02X", header[0]);
      }
    }
    catch (const std::exception &ex)
    {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 20, "Error while receiving data: %s", ex.what());
      reopenPort();
    }
  }
}


void RMSerialDriver::getParams()
{
  using FlowControl = drivers::serial_driver::FlowControl;
  using Parity = drivers::serial_driver::Parity;
  using StopBits = drivers::serial_driver::StopBits;

  uint32_t baud_rate{};
  auto fc = FlowControl::NONE;
  auto pt = Parity::NONE;
  auto sb = StopBits::ONE;

  try
  {
    device_name_ = this->declare_parameter<std::string>("device_name", "");
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The device name provided was invalid");
    throw ex;
  }

  try
  {
    rotation = this->declare_parameter<int>("self_rotation", 0);
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The rotation provided was invalid");
    throw ex;
  }

  try
  {
    baud_rate = this->declare_parameter<int>("baud_rate", 0);
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The baud_rate provided was invalid");
    throw ex;
  }

  try
  {
    const auto fc_string = this->declare_parameter<std::string>("flow_control", "");

    if (fc_string == "none")
    {
      fc = FlowControl::NONE;
    }
    else if (fc_string == "hardware")
    {
      fc = FlowControl::HARDWARE;
    }
    else if (fc_string == "software")
    {
      fc = FlowControl::SOFTWARE;
    }
    else
    {
      throw std::invalid_argument{
          "The flow_control parameter must be one of: none, software, or hardware."};
    }
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The flow_control provided was invalid");
    throw ex;
  }

  try
  {
    const auto pt_string = this->declare_parameter<std::string>("parity", "");

    if (pt_string == "none")
    {
      pt = Parity::NONE;
    }
    else if (pt_string == "odd")
    {
      pt = Parity::ODD;
    }
    else if (pt_string == "even")
    {
      pt = Parity::EVEN;
    }
    else
    {
      throw std::invalid_argument{"The parity parameter must be one of: none, odd, or even."};
    }
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The parity provided was invalid");
    throw ex;
  }

  try
  {
    const auto sb_string = this->declare_parameter<std::string>("stop_bits", "");

    if (sb_string == "1" || sb_string == "1.0")
    {
      sb = StopBits::ONE;
    }
    else if (sb_string == "1.5")
    {
      sb = StopBits::ONE_POINT_FIVE;
    }
    else if (sb_string == "2" || sb_string == "2.0")
    {
      sb = StopBits::TWO;
    }
    else
    {
      throw std::invalid_argument{"The stop_bits parameter must be one of: 1, 1.5, or 2."};
    }
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The stop_bits provided was invalid");
    throw ex;
  }

  try
  {
    cmd_vel_topic_ = this->declare_parameter<std::string>("cmd_vel_topic", "");
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The cmd_vel topic name provided was invalid");
    throw ex;
  }

  device_config_ =
      std::make_unique<drivers::serial_driver::SerialPortConfig>(baud_rate, fc, pt, sb);
}

void RMSerialDriver::reopenPort()
{
  RCLCPP_WARN(get_logger(), "Attempting to reopen port");
  try
  {
    if (serial_driver_->port()->is_open())
    {
      serial_driver_->port()->close();
    }
    serial_driver_->port()->open();
    RCLCPP_INFO(get_logger(), "Successfully reopened port");
  }
  catch (const std::exception &ex)
  {
    RCLCPP_ERROR(get_logger(), "Error while reopening port: %s", ex.what());
    if (rclcpp::ok())
    {
      rclcpp::sleep_for(std::chrono::seconds(1));
      reopenPort();
    }
  }
}

void RMSerialDriver::setParam_color(bool& initial_set_param,
  const rclcpp::Parameter &param, rclcpp::AsyncParametersClient::SharedPtr& param_client, ResultFuturePtr& param_future)
{
  if (!param_client->service_is_ready())
  {
    return;
  }

  if (!param_future.valid() ||
  param_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    RCLCPP_INFO(get_logger(), "Setting detect_color to %ld...", param.as_int());
    param_future = param_client->set_parameters(
        {param}, [&, param](const ResultFuturePtr &results)
        {
        for (const auto & result : results.get()) {
          if (!result.successful) {
            RCLCPP_ERROR(get_logger(), "Failed to set parameter: %s", result.reason.c_str());
            return;
          }
        }
        RCLCPP_INFO(get_logger(), "Successfully set detect_color to %ld!", param.as_int());
        initial_set_param = true; });
  }
}

} // namespace rm_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_serial_driver::RMSerialDriver)





