// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#include <tf2/LinearMath/Quaternion.h>

#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/utilities.hpp>
#include <serial_driver/serial_driver.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <std_msgs/msg/float64.hpp>
#include <rclcpp/rclcpp.hpp>


// C++ system
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rm_serial_driver/crc.hpp"
#include "rm_serial_driver/packet.hpp"
#include "rm_serial_driver/rm_serial_driver.hpp"

#include <Eigen/Dense>
#include <arpa/inet.h>
namespace rm_serial_driver
{
  RMSerialDriver::RMSerialDriver(const rclcpp::NodeOptions &options)
      : Node("rm_serial_driver", options),
        owned_ctx_{new IoContext(2)},
        serial_driver_{new drivers::serial_driver::SerialDriver(*owned_ctx_)}
  {
    RCLCPP_INFO(get_logger(), "Start RMSerialDriver!");
    using std::placeholders::_1;

    getParams();
    vision_data.assign(sizeof(new_SendPacket),0x00);
    nav_data.assign(12,0x00);
    power_heat_data.assign(28,0x00);
    time_count = 0;
    // TF broadcaster
    timestamp_offset_ = this->declare_parameter("timestamp_offset", 0.0);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    yaw_pub_ = this->create_publisher<std_msgs::msg::Float64>("/robo_yaw", 100);

    detecting_sub_ = this->create_subscription<std_msgs::msg::Int64>(
    "/detecting", rclcpp::SensorDataQoS(), std::bind(&RMSerialDriver::Detecting_callback, this, _1));

    // Create Publisher
    latency_pub_ = this->create_publisher<std_msgs::msg::Float64>("/latency", 10);
    control_id_pub_ = this->create_publisher<std_msgs::msg::Float64>("/control_id", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/aiming_point", 10);
    test_pub_ = this->create_publisher<std_msgs::msg::Float32>("test_pub",50);
    send_check_pub_ = this->create_publisher<std_msgs::msg::Float32>("send_check_pub",50);
    // Detect parameter client
    detector_param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, "armor_detector");

    // Tracker reset service client
    reset_tracker_client_ = this->create_client<std_srvs::srv::Trigger>("/tracker/reset");

    try
    {
      serial_driver_->init_port(device_name_, *device_config_);
      if (!serial_driver_->port()->is_open())
      {
        serial_driver_->port()->open();
        receive_thread_ = std::thread(&RMSerialDriver::receiveData, this);
      }
    }
    catch (const std::exception &ex)
    {
      RCLCPP_ERROR(
          get_logger(), "Error creating serial port: %s - %s", device_name_.c_str(), ex.what());
      throw ex;
    }

    aiming_point_.header.frame_id = "odom";
    aiming_point_.ns = "aiming_point";
    aiming_point_.type = visualization_msgs::msg::Marker::SPHERE;
    aiming_point_.action = visualization_msgs::msg::Marker::ADD;
    aiming_point_.scale.x = aiming_point_.scale.y = aiming_point_.scale.z = 0.12;
    aiming_point_.color.r = 1.0;
    aiming_point_.color.g = 1.0;
    aiming_point_.color.b = 1.0;
    aiming_point_.color.a = 1.0;
    aiming_point_.lifetime = rclcpp::Duration::from_seconds(0.1);

    // Create Subscription
    target_sub_ = this->create_subscription<auto_aim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS(),
        std::bind(&RMSerialDriver::sendData, this, std::placeholders::_1));
    Nav_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SensorDataQoS(),
        std::bind(&RMSerialDriver::NavCallback, this, std::placeholders::_1));
    power_heat_sub_ = this->create_subscription<my_msg_interface::msg::PowerHeat>(
        "PowerHeat",rclcpp::SystemDefaultsQoS(),
        std::bind(&RMSerialDriver::PowerHeatCallback,this,std::placeholders::_1));
    //creat timer
    timer_pub_ = this->create_wall_timer(
      std::chrono::milliseconds(4), std::bind(&RMSerialDriver::timerCallback, this));

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

  void RMSerialDriver::Detecting_callback(const std_msgs::msg::Int64 msg)
{
  detect = msg.data;
}


  void RMSerialDriver::NavCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    struct temp_nav_struct {
      float linear_x;
      float linear_y;
      float angular_z;
    } temp_nav_data;
    static_assert(sizeof(temp_nav_struct)==12,"temp_nav_data size error");
    temp_nav_data.linear_x = -msg->linear.y;
    temp_nav_data.linear_y = msg->linear.x;
    temp_nav_data.angular_z = msg->angular.z;
    nav_mutex.lock();
      nav_data.resize(sizeof(temp_nav_struct));
      std::copy(reinterpret_cast<uint8_t *>(&temp_nav_data),reinterpret_cast<uint8_t *>(&temp_nav_data)+sizeof(temp_nav_struct),nav_data.begin());
      RCLCPP_INFO(this->get_logger(),"x:%lf,y:%lf,w:%lf",msg->linear.x,msg->linear.y,msg->angular.z);
    nav_mutex.unlock();
  }

  void RMSerialDriver::PowerHeatCallback(const my_msg_interface::msg::PowerHeat::SharedPtr msg) {
    struct temp_power_heat_struct {
      float chassis_power;
      float buffer_energy;
      float rotation;
      float shooter_17mm_1_barrel_heat;
      float shooter_17mm_2_barrel_heat;
      float initial_speed;
      float launching_frequency;
    } temp_power_heat_struct;
    static_assert(sizeof(temp_power_heat_struct)==28,"temp_power_heat_struct size error");
    temp_power_heat_struct.buffer_energy = msg->buffer_energy;
    temp_power_heat_struct.chassis_power = msg->chassis_power;
    temp_power_heat_struct.initial_speed = msg->initial_speed;
    temp_power_heat_struct.launching_frequency = msg->launching_frequency;
    temp_power_heat_struct.shooter_17mm_1_barrel_heat = msg->shooter_17mm_1_barrel_heat;
    temp_power_heat_struct.shooter_17mm_2_barrel_heat = msg->shooter_17mm_2_barrel_heat;
    // temp_power_heat_struct.rotation = rotation;
    temp_power_heat_struct.rotation = this->get_parameter_or("self_rotation",rotation);
    power_heat_mutex.lock();
      power_heat_data.resize(sizeof(temp_power_heat_struct));
      std::copy(reinterpret_cast<uint8_t *>(&temp_power_heat_struct),reinterpret_cast<uint8_t *>(&temp_power_heat_struct)+sizeof(temp_power_heat_struct),power_heat_data.begin());
    power_heat_mutex.unlock();
  
  }
  uint64_t count;
  void RMSerialDriver::timerCallback() {
    //同时锁避免死锁
    std::lock(vision_mutex,nav_mutex,power_heat_mutex);
    //待发送队列为包长加上可去的尾帧
    std::vector<uint8_t> wait_for_send(sizeof(TotalSendPacket)+1);
    wait_for_send[0] =0xA5;
    if (vision_data.data() != nullptr) { 
      //视觉数据在前，视觉的最后两字节放在全包尾帧之前
      std::copy(vision_data.begin(), vision_data.end()-2, wait_for_send.begin());
      //视觉数据的最后两字节放在全包尾帧之前
      std::copy(vision_data.end()-2, vision_data.end(), wait_for_send.end()-3);
    }
    if (nav_data.data() != nullptr ) {
      //导航数据放在视觉包-2字节后
      std::copy(nav_data.begin(), nav_data.end(), (wait_for_send.begin()+18/*vision_data.size()-2*/));
    }
    if(power_heat_data.data() != nullptr ) {
      //裁判系统数据放在 视觉包-2字节，导航包之后
      std::copy(power_heat_data.begin(), power_heat_data.end(), (wait_for_send.begin()+18+12/*vision_data.size()-2+nav_data.size()*/));
    }
    vision_mutex.unlock();
    nav_mutex.unlock();
    power_heat_mutex.unlock();
    wait_for_send.end()[-1] = 0x4A;
    try{
      serial_driver_->port()->send(wait_for_send);
      if(count % 1000 ==0)
        RCLCPP_INFO(this->get_logger(),"serial send OK ! ");
      count++;
    }catch(const std::exception& e){
      RCLCPP_ERROR(this->get_logger(),"%s",e.what());
    }

  }


  void RMSerialDriver::receiveData()
  {
    std::vector<uint8_t> header(1);
    std::vector<uint8_t> data;
    data.reserve(sizeof(ReceivePacket));

    while (rclcpp::ok())
    {
      try
      {
        serial_driver_->port()->receive(header);

        if (header[0] == 0x5A)
        {
          data.resize(sizeof(ReceivePacket) - 1);
          serial_driver_->port()->receive(data);
          data.insert(data.begin(), header[0]);
          ReceivePacket packet = fromVector(data);
          // RCLCPP_INFO(this->get_logger(),"packet.size:%ld packet.header: %#x packet.end_frame %#x",sizeof(packet),packet.header,packet.end_frame);
          // for(const auto byte:data){
          //   RCLCPP_INFO(this->get_logger(),"%#x",byte);
          // }  
          // bool crc_ok =
          //     crc16::Verify_CRC16_Check_Sum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
          // if (crc_ok ) {


//packet.end_frame == 0xA5

          if (1) {

            if (!initial_set_param_color || packet.detect_color != previous_receive_color_) {
              setParam_color(rclcpp::Parameter("detect_color", packet.detect_color));
              previous_receive_color_ = packet.detect_color;
            }

            if (packet.reset_tracker) {
              resetTracker();
            }
            
            geometry_msgs::msg::TransformStamped t;
            timestamp_offset_ = this->get_parameter("timestamp_offset").as_double();
            t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
            t.header.frame_id = "base_link";
            t.child_frame_id = "gimbal_link";
            tf2::Quaternion q;

            std_msgs::msg::Float64 control_id;
            control_id.set__data(packet.control_id);
            control_id_pub_->publish(control_id);
            q.setRPY(packet.roll/57.2957, -packet.pitch/57.2957, packet.yaw/57.2957);
            yaw_msg.data = packet.yaw/57.2957;
            yaw_pub_->publish(yaw_msg);
              // RCLCPP_INFO(this->get_logger(),"\n yaw_pub- %lf -\n",yaw_msg.data);
            test_pub_->publish(testmsg.set__data(packet.useless[1]));
            
            t.transform.rotation = tf2::toMsg(q);
              tf_broadcaster_->sendTransform(t);
          }
          else
          {
            RCLCPP_ERROR(get_logger(), "CRC error!");
          }
      }
      else
      {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 20, "Invalid header: %02X", header[0]);
      }
    }
    catch (const std::exception &ex)
    {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 20, "Error while receiving data: %s", ex.what());
      reopenPort();
    }
  }
}

void RMSerialDriver::sendData(const auto_aim_interfaces::msg::Target::SharedPtr msg)
{
  const static std::map<std::string, uint8_t> id_unit8_map{
      {"", 0}, {"outpost", 0}, {"1", 1}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"guard", 6}, {"base", 7}};

  try
  {
    aiming_point_.pose.position.x = msg->aim_x;
    aiming_point_.pose.position.y = msg->aim_y;
    aiming_point_.pose.position.z = msg->aim_z;
    if (abs(aiming_point_.pose.position.x) > 0.01)
    {
      aiming_point_.header.stamp = this->now();
      marker_pub_->publish(aiming_point_);
    }

    new_SendPacket new_packet;
    new_packet.tracking = msg->tracking;
    new_packet.id = id_unit8_map.at(msg->id);
    new_packet.armors_num = msg->armors_num;
    new_packet.yaw = msg->pre_yaw;
    new_packet.pitch = msg->pre_pitch;
    new_packet.detecting = detect;
    // RCLCPP_WARN(this->get_logger(),"detect: %d", detect);

    // if((new_packet.yaw!=0)&&(new_packet.pitch!=0))
    // {
    //   new_packet.tracking = 1;
    // } else {
    //   new_packet.tracking = 0;
    // }
    new_packet.fire = msg->fire;
    // float show_fire = msg->fire;
    // RCLCPP_INFO(this->get_logger(),"fire: %f", show_fire);
    new_packet.distance = sqrt(msg->aim_x * msg->aim_x + msg->aim_y * msg->aim_y);
//temp change//////////////
    crc16::Append_CRC16_Check_Sum(reinterpret_cast<uint8_t *>(&new_packet), sizeof(new_packet));
    vision_mutex.lock();

      vision_data.resize(sizeof(new_packet));
      vision_data = toVector(new_packet);
      sendcheckmsg.set__data(detect);
    vision_mutex.unlock();
      send_check_pub_->publish(sendcheckmsg);
    // serial_driver_->port()->send(data);

    std_msgs::msg::Float64 latency;
    latency.data = (this->now() - msg->header.stamp).seconds() * 1000.0;
    RCLCPP_DEBUG_STREAM(get_logger(), "Total latency: " + std::to_string(latency.data) + "ms");
    latency_pub_->publish(latency);
  }
  catch (const std::exception &ex)
  {
    RCLCPP_ERROR(get_logger(), "Error while sending data: %s", ex.what());
    reopenPort();
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

void RMSerialDriver::setParam_color(const rclcpp::Parameter &param)
{
  if (!detector_param_client_->service_is_ready())
  {
    RCLCPP_WARN(get_logger(), "Service not ready, skipping parameter set");
    return;
  }

  if (
      !set_param_future_color.valid() ||
      set_param_future_color.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
  {
    RCLCPP_INFO(get_logger(), "Setting detect_color to %ld...", param.as_int());
    set_param_future_color = detector_param_client_->set_parameters(
        {param}, [this, param](const ResultFuturePtr &results)
        {
        for (const auto & result : results.get()) {
          if (!result.successful) {
            RCLCPP_ERROR(get_logger(), "Failed to set parameter: %s", result.reason.c_str());
            return;
          }
        }
        RCLCPP_INFO(get_logger(), "Successfully set detect_color to %ld!", param.as_int());
        initial_set_param_color = true; });
  }
}

void RMSerialDriver::resetTracker()
{
  if (!reset_tracker_client_->service_is_ready())
  {
    RCLCPP_WARN(get_logger(), "Service not ready, skipping tracker reset");
    return;
  }

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  reset_tracker_client_->async_send_request(request);
  RCLCPP_INFO(get_logger(), "Reset tracker!");
}

} // namespace rm_serial_driver

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_serial_driver::RMSerialDriver)
