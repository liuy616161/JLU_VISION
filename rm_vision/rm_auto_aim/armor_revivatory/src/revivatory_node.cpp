#include "armor_revivatory/revivatory_node.hpp"

#include "std_msgs/msg/string.hpp"

namespace rm_auto_aim
{
ArmorRevivatoryNode::ArmorRevivatoryNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("armor_revivatory_node", options), now_array_({1, 1, 1, 1, 1}),  last_array_({1, 1, 1, 1, 1}), iffire_array_({1, 1, 1, 1, 1}), 
timekeeping_array_({0, 0, 0, 0, 0})//last_array_要初始化为活着
{
    // 订阅下位机发送车存亡的消息
    revivatory_car_ = this->create_subscription<std_msgs::msg::UInt8MultiArray>(
        "revivatory_car", 10, std::bind(&ArmorRevivatoryNode::revivatory_car_callback, this, std::placeholders::_1));

    fire_ = this->create_publisher<std_msgs::msg::UInt8MultiArray>("contorl_fire", 10);
    
    // 创建一个定时器，每0.2秒触发一次
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(200),  
        std::bind(&ArmorRevivatoryNode::timeCallback, this)  // 定时器触发时调用timerCallback
    );
}

void ArmorRevivatoryNode::revivatory_car_callback(const std_msgs::msg::UInt8MultiArray::SharedPtr msg)
{ 
    now_array_ = msg->data;
}

void ArmorRevivatoryNode::timeCallback()
{
    for (size_t i = 0; i < now_array_.size(); ++i)
    {
        if(now_array_[i] == 0)
        {
            iffire_array_[i] = 1;
            timekeeping_array_[i] = 0;
        }
       
        if((last_array_[i] == 0 && now_array_[i] == 1))
        {
            iffire_array_[i] = 0;
            timekeeping_array_[i] = 0;
        }

        if(iffire_array_[i] == 0)
        {
            timekeeping_array_[i]++;
        }

        if(timekeeping_array_[i] == 48)
        {
            iffire_array_[i] = 1;
            timekeeping_array_[i] = 0;
        }
    }
    
    last_array_ = now_array_;

    fire_msg.data = iffire_array_;
    fire_->publish(fire_msg);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorRevivatoryNode)
