// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#include <tf2/LinearMath/Quaternion.h>

#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/utilities.hpp>
#include <serial_driver/serial_driver.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


// C++ system
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <math.h>
#include "rm_serial_driver/crc.hpp"
#include "rm_serial_driver/packet.hpp"
#include "rm_serial_driver/rm_serial_driver.hpp"

#include <Eigen/Dense>

namespace rm_serial_driver
{
  RMSerialDriver::RMSerialDriver(const rclcpp::NodeOptions &options)
      : Node("rm_serial_driver", options),
        owned_ctx_{new IoContext(2)},
        serial_driver_{new drivers::serial_driver::SerialDriver(*owned_ctx_)}
  {
    RCLCPP_INFO(get_logger(), "Start RMSerialDriver!");

    getParams();
    yaw_pub_ = this->create_publisher<std_msgs::msg::Float64>("/robo_yaw", 10);
    test_pub_ = this->create_publisher<std_msgs::msg::Float64>("/pitch_test", 10);
    pitch_calculate_ = this->create_publisher<std_msgs::msg::Float64>("/pitch_calculate_msg", 10);
    pitch_imu_ = this->create_publisher<std_msgs::msg::Float64>("/pitch_imu_msg", 10);
    sentry_decision_pub_ = this->create_publisher<std_msgs::msg::Int8>("sentry_decision",10);
    exposure_time_pub_ = this->create_publisher<std_msgs::msg::Int64>("exposure_time",10);
   
    // 从参数服务器获取曝光时间参数
    aim_et_ = this->declare_parameter("exposure_time_aim", 2800);
    buff_et_ = this->declare_parameter("exposure_time_buff", 6800);
    previous_exposure_time_ = this->declare_parameter("previous_exposure_time", 1000);
    // TF broadcaster
    timestamp_offset_ = this->declare_parameter("timestamp_offset", 0.0);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
   // Create Publisher
    task_pub_ = this->create_publisher<global_interface::msg::SerialTask>("/serial_task", 10);
    latency_pub_ = this->create_publisher<std_msgs::msg::Float64>("/latency", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/aiming_point", 10);
    previous_exposure_time_pub_ = this->create_publisher<std_msgs::msg::Int64>("/previous_exposure_time", 10);
    //new//////////////////////
    //qos
    rclcpp::QoS qos(0);
    qos.keep_last(1);
    qos.reliable();
    qos.durability();
    // qos.best_effort()
    // qos.transient_local();
    // qos.durability_volatile();

    rmw_qos_profile_t rmw_qos(rmw_qos_profile_sensor_data);
    rmw_qos.depth = 1;
        
    serial_msg_pub_ = this->create_publisher<global_interface::msg::Serial>("/serial_msg", qos);

    buff_info_sub_ = this->create_subscription<global_interface::msg::Buff>(
        "/buff_msg",
        rclcpp::SensorDataQoS(),
        std::bind(&RMSerialDriver::buffMsgCallback, this, std::placeholders::_1)
    );

    // Detect parameter client
    detector_param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, "armor_detector");

    // Tracker reset service client
    reset_tracker_client_ = this->create_client<std_srvs::srv::Trigger>("/tracker/reset");
    // debug_msg = this->create_publisher<>
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

    exposure_time_sub_ = this->create_subscription<std_msgs::msg::Int64>(
      "exposure_time", 
      qos, 
      std::bind(&RMSerialDriver::exposureTimeCallback, this, std::placeholders::_1)
  );
  }

// 添加回调函数
void RMSerialDriver::exposureTimeCallback(const std_msgs::msg::Int64::SharedPtr msg)
{
    previous_exposure_time_ = msg->data;
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

          bool crc_ok =
              crc16::Verify_CRC16_Check_Sum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
          if (crc_ok)
          
          {
          
        /*
          if (!initial_set_param_color || packet.detect_color != previous_receive_color_)
          {
            setParam_color(rclcpp::Parameter("detect_color", packet.detect_color));
            previous_receive_color_ = packet.detect_color;
          }
        */
          if (packet.reset_tracker)
          {
            resetTracker();
          }
          uint8_t tmp;
          tmp=255-packet.task_mode;
          packet.task_mode=255-tmp;
          geometry_msgs::msg::TransformStamped t;
          timestamp_offset_ = this->get_parameter("timestamp_offset").as_double();
          t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
          t.header.frame_id = "odom";
          t.child_frame_id = "gimbal_link";
          tf2::Quaternion q;

          q.setRPY(packet.roll/57.2957795/1000, -packet.pitch/57.2957795/1000, packet.yaw/57.2957795/1000);

          yaw_msg.data = packet.yaw/57.2957/1000;
          sentry_decision_msg.data = packet.sentry_decision;
          
          task_msg.header.frame_id = "serial";
          task_msg.header.stamp = this->get_clock()->now();
          task_msg.color = packet.detect_color;
          task_msg.mode = packet.task_mode;
          task_msg.direction = packet.rune_direction;
          task_msg.is_stable = packet.rune_stable;

          
          // 根据任务模式设置不同的曝光时间
          int64_t current_exposure_time;
          if(packet.task_mode == 0) {  // 自瞄模式
              current_exposure_time = aim_et_;
          } else  {  // 能量机关模式   
              current_exposure_time = buff_et_;
          }
          
          // 只有当曝光时间改变时才发布消息
          if(current_exposure_time != previous_exposure_time_) {
              exposure_time_msg.data = current_exposure_time;
              exposure_time_pub_->publish(exposure_time_msg);
              
              // 发布前一个曝光时间
              auto previous_msg = std_msgs::msg::Int64();
              previous_msg.data = previous_exposure_time_;
              previous_exposure_time_pub_->publish(previous_msg);
              
              // 更新前一个曝光时间
              previous_exposure_time_ = current_exposure_time;
          }

          task_pub_->publish(task);
          yaw_pub_->publish(yaw_msg);
          sentry_decision_pub_->publish(sentry_decision_msg);
          //exposure_time_pub_->publish(exposure_time_msg);

          //std::cout <<packet.roll<<std::endl;
          //std::cout << -packet.pitch<<std::endl;
          //std::cout << packet.yaw<<std::endl;

          t.transform.rotation = tf2::toMsg(q);
          tf_broadcaster_->sendTransform(t);



          //std::cout<<"serial publish"<<std::endl;
          ////////////////
          global_interface::msg::Serial serial_msg;
                    serial_msg.header.frame_id = "serial";
                    serial_msg.header.stamp = this->get_clock()->now();
                    serial_msg.imu.header.frame_id = "imu_link";
                    serial_msg.imu.header.stamp = this->get_clock()->now();
                    //serial_msg.mode = packet.task_mode;
                    serial_msg.mode = task.data;
                    serial_msg.bullet_speed = 28;
                    serial_msg.shoot_delay = 0;
                    // 下位机乘1000发，会准一点
                    serial_msg.imu.orientation.w = q.w();
                    serial_msg.imu.orientation.x = q.x();
                    serial_msg.imu.orientation.y = q.y();
                    serial_msg.imu.orientation.z = q.z();
                    // std::cout<<"packet.yaw"<<packet.yaw/1000<<std::endl;
                    // std::cout<<"packet.yaw"<<packet.pitch/1000<<std::endl;
                    // std::cout<<"packet.yaw"<<packet.yaw<<std::endl;
                    // std::cout<<"packet.yaw"<<packet.yaw<<std::endl;

                    // serial_msg.imu.angular_velocity.x = gyro[0];
                    // serial_msg.imu.angular_velocity.y = gyro[1];
                    // serial_msg.imu.angular_velocity.z = gyro[2];
                    // serial_msg.imu.linear_acceleration.x = acc[0];
                    // serial_msg.imu.linear_acceleration.y = acc[1];
                    // serial_msg.imu.linear_acceleration.z = acc[2];
            ////////////////
            serial_msg_pub_->publish(std::move(serial_msg));
            imu_pitch = packet.pitch;
            imu_yaw = packet.yaw;
        }
        else
        {
          RCLCPP_ERROR(get_logger(), "CRC error!");
        }
      }
      else
      {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 20, "Invalid header: %02X", header[0]);
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
  if(task.mode == 0)
  {
    //std::cout<<"task"<<task.data<<std::endl;
 const static std::map<std::string, uint8_t> id_unit8_map{
      {"", 0}, {"outpost", 0}, {"1", 1}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"guard", 6}, {"base", 7}};

  try
  {
    // aiming_point_.pose.position.x = msg->aim_x;
    // aiming_point_.pose.position.y = msg->aim_y;
    // aiming_point_.pose.position.z = msg->aim_z;
    if (abs(msg->aim_x) > 0.01)
    {
      aiming_point_.pose.position.x = msg->aim_x;
      aiming_point_.pose.position.y = msg->aim_y;
      aiming_point_.pose.position.z = msg->aim_z;
      aiming_point_.header.stamp = this->now();
      //aiming_point_.dist=sqrt(msg->position.x*msg->position.x+msg->position.y*msg->position.y)
      marker_pub_->publish(aiming_point_);
    }
    SendPacket packet;
    packet.tracking = msg->tracking;
    packet.id = id_unit8_map.at(msg->id);
    packet.armors_num = msg->armors_num;
    packet.yaw = msg->pre_yaw * 1000;

    packet.pitch = msg->pre_pitch * 1000;
    
    packet.fire = msg->fire;
    packet.v_yaw = msg->v_yaw;
    packet.dist = sqrt(msg->position.x*msg->position.x+msg->position.y*msg->position.y);
    packet.flag_spin_mov = msg->flag_spin_mov;
    // packet.aim_x=msg->aim_x;
    // packet.aim_y=msg->aim_y;
    // packet.aim_z=msg->aim_z;
    
    
    crc16::Append_CRC16_Check_Sum(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));

    std::vector<uint8_t> data = toVector(packet);
    serial_driver_->port()->send(data);

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
 
}

void RMSerialDriver::buffMsgCallback(global_interface::msg::Buff::SharedPtr buff_msg)
{

  //std::cout<<"  buff_msg gettttttttt"<<std::endl;
  if(task.mode != 0)
  {
   SendPacket packet;
   /*
   if(buff_msg->predict_pitch == 0 ||  buff_msg->predict_yaw == 0)
   {
     packet.tracking = 0;
   }
   else{
    packet.tracking = 1;
   }
    */
    packet.id = 1;
    packet.armors_num = 1;
    packet.yaw = (buff_msg->predict_yaw) * 1000;
    // std::cout<<"packet.yaw"<<packet.yaw/1000<<std::endl;
    packet.pitch = (buff_msg->predict_pitch) * 1000;
    // std::cout<<"packet.pitch"<<packet.pitch/1000<<std::endl;
    packet.fire = 1;
    packet.v_yaw = 1;


    //pitch_test.data = packet.pitch;
    //test_pub_->publish(pitch_test);
    
    //pitch_calculate_msg.data = gimbal_msg->pitch;
    //pitch_calculate_->publish(pitch_calculate_msg);
    // pitch_imu_msg.data = imu_pitch;
    // pitch_imu_->publish(pitch_imu_msg);

    crc16::Append_CRC16_Check_Sum(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));

    std::vector<uint8_t> data = toVector(packet);
    serial_driver_->port()->send(data);
    // std::cout<<"send finish"<<std::endl;
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
    device_name_ = declare_parameter<std::string>("device_name", "");
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The device name provided was invalid");
    throw ex;
  }

  try
  {
    baud_rate = declare_parameter<int>("baud_rate", 0);
  }
  catch (rclcpp::ParameterTypeException &ex)
  {
    RCLCPP_ERROR(get_logger(), "The baud_rate provided was invalid");
    throw ex;
  }

  try
  {
    const auto fc_string = declare_parameter<std::string>("flow_control", "");

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
    const auto pt_string = declare_parameter<std::string>("parity", "");

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
    const auto sb_string = declare_parameter<std::string>("stop_bits", "");

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
