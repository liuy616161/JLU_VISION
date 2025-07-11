#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <iomanip>
#include <sstream>
#include <string>
#include <mutex>

#include "tars_msgs/msg/serial_receive.hpp"
#include "tars_msgs/msg/serial_send.hpp"

struct Param {
    std::string name = "";
    std::string value = "";
};

class DebugVisualization : public rclcpp::Node
{
public:
  DebugVisualization() : Node("debug_visualization")
  {
    using std::placeholders::_1;

    this->declare_parameter("refresh_rate_hz", 20);
    int refresh_rate = this->get_parameter("refresh_rate_hz").as_int();
    int refresh_ms = 1000 / refresh_rate;

    ser_rcv_sub_ = 
      this->create_subscription<tars_msgs::msg::SerialReceive>(
      "serial_receive", 5,
      std::bind(&DebugVisualization::SerialReceiveCallback, this, _1));
    
    ser_send_sub_ = 
      this->create_subscription<tars_msgs::msg::SerialSend>(
      "serial_send", 5,
      std::bind(&DebugVisualization::SerialSendCallback, this, _1));
    
    timer_ = this->create_wall_timer(std::chrono::milliseconds(refresh_ms), 
            std::bind(&DebugVisualization::TimerCallback, this));
            
    output_buffer_.reserve(4096);
  }
  
  ~DebugVisualization()
  {
    std::cout << std::endl << "Good bye! Have a nice day!" << std::endl;
  }

private:
  void SerialReceiveCallback(const tars_msgs::msg::SerialReceive::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    rcv_msg = *msg;
    new_rcv_data_ = true;
  }

  void SerialSendCallback(const tars_msgs::msg::SerialSend::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    send_msg = *msg;
    new_send_data_ = true;
  }

  void TimerCallback()
  {
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (!new_rcv_data_ && !new_send_data_) {
        return;
      }
      new_rcv_data_ = false;
      new_send_data_ = false;
    }

    output_buffer_.clear();
    
    output_buffer_ += "\033[2J\033[1;1H\n";
    output_buffer_ += "+-------------------------------------------------------------------+\n";
    output_buffer_ += "|                        Debug Visualization                        |\n";
    output_buffer_ += "+-------------------------------------------------------------------+\n";

    append_header_line();
    output_buffer_ += "|                                                                   |\n";

    std::lock_guard<std::mutex> lock(data_mutex_);
    
    append_two_params("detect_color", std::to_string(rcv_msg.detect_color),
                    "tracking", send_msg.tracking ? "true" : "false");
    append_two_params("reset_tracker", std::to_string(rcv_msg.reset_tracker),
                    "id", std::to_string(send_msg.id));
    append_two_params("reserved", std::to_string(rcv_msg.reserved),
                    "armors_num", std::to_string(send_msg.armors_num));
    append_two_params("roll", to_string_with_precision(rcv_msg.roll),
                    "detecting", std::to_string(send_msg.detecting));
    append_two_params("pitch", to_string_with_precision(rcv_msg.pitch),
                    "yaw", to_string_with_precision(send_msg.yaw));
    append_two_params("yaw", to_string_with_precision(rcv_msg.yaw),
                    "pitch", to_string_with_precision(send_msg.pitch));
    append_two_params("control_id", to_string_with_precision(rcv_msg.control_id),
                    "fire", to_string_with_precision(send_msg.fire));
    append_two_params("revivatory_car", uint8_to_binary_string(rcv_msg.revivatory_car),
                    "distance", to_string_with_precision(send_msg.distance));
    append_two_params("", "", 
                    "another_fire", std::to_string(send_msg.another_fire));

    output_buffer_ += "|                                                                   |\n";

    append_two_params("yaw_delta", to_string_with_precision(rcv_msg.yaw_delta),
                    "linear_x", to_string_with_precision(send_msg.linear_x));
    append_two_params("", "", 
                    "linear_y", to_string_with_precision(send_msg.linear_y));
    append_two_params("", "", 
                    "angular_z", to_string_with_precision(send_msg.angular_z));
    append_two_params("", "", 
                    "tuoluo", std::to_string(send_msg.tuoluo));

    output_buffer_ += "|                                                                   |\n";

    append_two_params("game_progress", std::to_string(rcv_msg.game_progress),
                    "back_armor_number", std::to_string(send_msg.back_armor_number));
    append_two_params("remain_time", std::to_string(rcv_msg.remain_time),
                    "back_armor_angle", to_string_with_precision(send_msg.back_armor_angle));
    append_two_params("current_hp", std::to_string(rcv_msg.current_hp),
                    "", "");
    append_two_params("projectile", std::to_string(rcv_msg.projectile),
                    "", "");
    append_two_params("sentry_info", uint8_to_binary_string(rcv_msg.sentry_info),
                    "", "");
    append_two_params("red_outpost_hp", std::to_string(rcv_msg.red_outpost_hp),
                    "", "");
    append_two_params("red_base_hp", std::to_string(rcv_msg.red_base_hp),
                    "", "");
    append_two_params("blue_outpost_hp", std::to_string(rcv_msg.blue_outpost_hp),
                    "", "");
    append_two_params("blue_base_hp", std::to_string(rcv_msg.blue_base_hp),
                    "", "");

    output_buffer_ += "+-------------------------------------------------------------------+\n";

    std::cout << output_buffer_;
  }

  void append_header_line() {
    output_buffer_ += "| ";

    std::string left_part = "Serial Receive Data";
    output_buffer_ += left_part;
    for (size_t i = 0; i < 33 - left_part.length(); ++i) {
        output_buffer_ += " ";
    }
    
    std::string right_part = "Serial Send Data       ";
    for (size_t i = 0; i < 32 - right_part.length(); ++i) {
        output_buffer_ += " ";
    }
    output_buffer_ += right_part;
    output_buffer_ += " |\n";
  }

  template <typename T>
  std::string to_string_with_precision(T value, int precision = 4) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
  }

  std::string uint8_to_binary_string(uint8_t value) {
    static char binary_str[9];
    binary_str[8] = '\0';
    
    for (int i = 7; i >= 0; --i) {
        binary_str[7-i] = ((value >> i) & 1) ? '1' : '0';
    }
    return binary_str;
  }

  void append_two_params(const std::string& left_name, const std::string& left_value,
                        const std::string& right_name, const std::string& right_value) {
    const int name_width = 18;
    const int value_width = 10;

    output_buffer_ += "| ";
    
    if (!left_name.empty()) {
        output_buffer_ += left_name;
        for (size_t i = 0; i < name_width - left_name.length(); ++i) {
            output_buffer_ += " ";
        }
        output_buffer_ += ": ";

        std::string padded_value;
        if (left_value.length() < value_width) {
            padded_value.append(value_width - left_value.length(), ' ');
        }
        padded_value += left_value;
        output_buffer_ += padded_value;
        output_buffer_ += "     ";
    } else {
        output_buffer_.append(name_width + value_width + 7, ' ');
    }

    if (!right_name.empty()) {
        output_buffer_ += right_name;
        for (size_t i = 0; i < name_width - right_name.length(); ++i) {
            output_buffer_ += " ";
        }
        output_buffer_ += ": ";
        
        std::string padded_value;
        if (right_value.length() < value_width) {
            padded_value.append(value_width - right_value.length(), ' ');
        }
        padded_value += right_value;
        output_buffer_ += padded_value;
    } else {
        output_buffer_.append(name_width + value_width + 2, ' ');
    }

    output_buffer_ += " |\n";
  }

  rclcpp::Subscription<tars_msgs::msg::SerialReceive>::SharedPtr ser_rcv_sub_;
  rclcpp::Subscription<tars_msgs::msg::SerialSend>::SharedPtr ser_send_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  tars_msgs::msg::SerialReceive rcv_msg;
  tars_msgs::msg::SerialSend send_msg;

  std::mutex data_mutex_;
  bool new_rcv_data_ = false;
  bool new_send_data_ = false;
  std::string output_buffer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DebugVisualization>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}