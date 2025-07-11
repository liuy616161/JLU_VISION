#include <thread>
#include <cstdint>
#include <stdexcept>

#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/logging.hpp>
#include <rclcpp/parameter_value.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>

namespace rm_auto_aim
{
class V4L2CameraNode : public rclcpp::Node
{
    std::string name_ = this->declare_parameter<std::string>("name");
    int index_ = this->declare_parameter<int>("index");
    int exposure_ = this->declare_parameter<int>("exposure");
    int brightness_ = this->declare_parameter<int>("brightness");
    int width_ = this->declare_parameter<int>("width");
    int height_ = this->declare_parameter<int>("height");
    
    image_transport::CameraPublisher camera_pub_ = image_transport::create_camera_publisher(
        this, "image_raw",
        this->declare_parameter<bool>("use_sensor_data_qos") ?
            rmw_qos_profile_sensor_data : rmw_qos_profile_default);

    std::string frame_id_ =
        this->declare_parameter<std::string>("frame_id");
    std::string camera_info_url_ =
        "package://" + this->declare_parameter<std::string>("camera_info_package") + "/camera/camera_info.yaml";
    
    camera_info_manager::CameraInfoManager camera_info_manager_{this, name_};
    sensor_msgs::msg::CameraInfo camera_info_msg_ = 
        [&]()
        {
            camera_info_manager_.loadCameraInfo(camera_info_url_);
            return camera_info_manager_.getCameraInfo();
        }();
    
    std::thread capture_thread_{
        [&]()
        {
            for ( ; ; )
            {
                try
                {
                    cv::VideoCapture cap(0, cv::CAP_V4L2);
                    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                    cap.set(cv::CAP_PROP_FRAME_WIDTH, width_);
                    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
                    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1.0);
                    cap.set(cv::CAP_PROP_EXPOSURE, exposure_);
                    cap.set(cv::CAP_PROP_BRIGHTNESS, brightness_);
                    cv::Mat img;
                    
                    std_msgs::msg::Header header;
                    header.frame_id = frame_id_;

                    for ( ; ; )
                    {
                        cap >> img;

                        if (img.empty())
                        {
                            throw std::runtime_error("no image");
                        }
                        else
                        {
                            header.stamp = this->now();

                            sensor_msgs::msg::Image image_msg;
                            cv_bridge::CvImage(header, "bgr8", img).toImageMsg(image_msg);

                            camera_info_msg_.header = header;
                            camera_pub_.publish(image_msg, camera_info_msg_);
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    RCLCPP_WARN(this->get_logger(), e.what());
                }
            }
        }};

public:
    explicit V4L2CameraNode(const rclcpp::NodeOptions & options) : Node("v4l2_camera", options)
    {
        RCLCPP_INFO(this->get_logger(), "v4l2 camera initialized");
    }
};
}

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::V4L2CameraNode)