#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/u_int8_multi_array.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "std_msgs/msg/float32.hpp"
#include <chrono>
#include <vector>

#include "auto_aim_interfaces/msg/tracked_target.hpp"

// ROS
#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int8.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace rm_auto_aim
{
class ArmorRevivatoryNode: public rclcpp::Node
{
public:  
    ArmorRevivatoryNode(const rclcpp::NodeOptions & options);
    
private:
    void revivatory_car_callback(const std_msgs::msg::UInt8MultiArray::SharedPtr msg);
    
    void timeCallback();
    
    rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr revivatory_car_;
    rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr fire_;
    
    std::vector<uint8_t> now_array_;  // 用于记录此时接收到的数组
    std::vector<uint8_t> last_array_;  // 用于记录上一次接收到的数组
    std::vector<uint8_t> iffire_array_;   //用于记录是否可以开火的数组 1:开火，0:不开火
    std::vector<uint8_t> timekeeping_array_;   //用于记录刚复活无敌时间的数组，0.2×50=10s
    
    rclcpp::TimerBase::SharedPtr timer_;
    std_msgs::msg::UInt8MultiArray fire_msg;
};
}  // namespace rm_auto_aim
