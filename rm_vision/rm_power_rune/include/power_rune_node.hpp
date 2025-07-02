// ROS
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/int64.hpp>
#include "PowerRune.h"
#include "global_interface/msg/serial.hpp"
#include "global_interface/msg/buff.hpp"
#include "global_interface/msg/serial_task.hpp"

namespace power_rune{

class PowerRuneNode : public rclcpp::Node
{
public:
    PowerRuneNode(const rclcpp::NodeOptions & options);


    ~PowerRuneNode() {
        if (video_writer_.isOpened()) {
            video_writer_.release();
            RCLCPP_INFO(this->get_logger(), "视频写入器已释放");
        }
    }
private:
    
      // 添加上次保存时间的成员变量
    rclcpp::Time last_save_time_;
    cv::VideoWriter video_writer_; // 视频写入器
    bool video_writer_initialized_; // 标志，检查是否已初始化
    int save_fps_ = 10; // 保存视频的帧率
    int video_counter_; // 用于生成唯一视频文件名的计数器
    int last_mode_; // 记录上一次的 rune_task_mode

    const double SAVE_INTERVAL = 1.0/save_fps_;  // 单位：秒
    
    // 任务回调函数
    void task_callback(const global_interface::msg::SerialTask task_msg); 
    // 图像回调函数
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);

    std::unique_ptr<PowerRune> power_rune_;
    rclcpp::Subscription<global_interface::msg::SerialTask>::SharedPtr task_sub_;

    rclcpp::Publisher<global_interface::msg::Buff>::SharedPtr buff_pub_;
            global_interface::msg::Buff buff_msg;
    
    
    //debug show
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr  image_show_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr  image_arrow_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr  image_armor_pub_;
    // rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr  image_src_pub_;
    rclcpp::TimerBase::SharedPtr debug_img_timer_;
    
    void publish_debug_img();

    // Camera info part
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    cv::Point2f cam_center_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;

    // Image subscrpition
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    rclcpp::Subscription<global_interface::msg::Serial>::SharedPtr serial_sub_;
    void serial_callback(const global_interface::msg::Serial::ConstSharedPtr serial_msg);

    double pitch_ = 0.0;
    double yaw_ = 0.0;
    double roll_ = 0.0;

    int rune_task_mode=0;
    int fail_count=0;

};

}
#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(power_rune::PowerRuneNode)