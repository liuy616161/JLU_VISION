// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_node.hpp"

namespace rm_auto_aim
{
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions & options)
: Node("armor_detector", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting DetectorNode!");
  
  detector_ = initDetector();
  
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>("detector/armors", rclcpp::SensorDataQoS());
  
  debug_ = this->declare_parameter<bool>("debug");
  
  if (debug_) {
    createDebugPublishers();
  }

  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "camera_info", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
      cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
      cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
      pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
      cam_info_sub_.reset();
    });

  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "image_raw", rclcpp::SensorDataQoS(rclcpp::KeepLast(1)),
    std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));
}

void ArmorDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  detectArmors(img_msg);

  if (pnp_solver_ != nullptr) {
    armors_msg_.header = img_msg->header;
    armors_msg_.armors.clear();

    auto_aim_interfaces::msg::Armor armor_msg;
    for (const auto & armor : detector_->classified_armors_) {
      cv::Mat rvec, tvec;
      bool success = pnp_solver_->solvePnP(armor, rvec, tvec);

      if (success) {
        armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armor.type)];
        armor_msg.number = armor.number;
        
        armor_msg.pose.position.x = tvec.at<double>(0);
        armor_msg.pose.position.y = tvec.at<double>(1);
        armor_msg.pose.position.z = tvec.at<double>(2);
        
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        
        tf2::Matrix3x3 tf2_rotation_matrix(
          rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
          rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
          rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
          rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
          rotation_matrix.at<double>(2, 2));
        
        tf2::Quaternion tf2_q;
        tf2_rotation_matrix.getRotation(tf2_q);
        armor_msg.pose.orientation = tf2::toMsg(tf2_q);
        armor_msg.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);
        armors_msg_.armors.emplace_back(armor_msg);
      } else {
        RCLCPP_WARN(this->get_logger(), "PnP failed!");
      }
    }

    armors_pub_->publish(armors_msg_);
  }
}

std::unique_ptr<Detector> ArmorDetectorNode::initDetector()
{
  declare_parameter<int>("light.detect_color");
  
  std::vector<int> light_thres {
    (int)declare_parameter<int>("light.thres.red"),
    (int)declare_parameter<int>("light.thres.blue") };

  Detector::LightParams l_params = {
    .min_ratio = declare_parameter<double>("light.min_ratio"),
    .max_ratio = declare_parameter<double>("light.max_ratio"),
    .max_angle = declare_parameter("light.max_angle", 40.0)};

  Detector::ArmorParams a_params = {
    .min_light_ratio = declare_parameter("armor.min_light_ratio", 0.7),
    .min_small_center_distance = declare_parameter("armor.min_small_center_distance", 0.8),
    .max_small_center_distance = declare_parameter("armor.max_small_center_distance", 3.2),
    .min_large_center_distance = declare_parameter("armor.min_large_center_distance", 3.2),
    .max_large_center_distance = declare_parameter("armor.max_large_center_distance", 5.5),
    .max_angle = declare_parameter("armor.max_angle", 35.0)};

  auto detector = std::make_unique<Detector>(light_thres, l_params, a_params);

  // Init classifier
  std::vector<std::vector<double>> rgb_coefficient {
    this->declare_parameter<std::vector<double>>("classifier.rgb_coefficient.red"),
    this->declare_parameter<std::vector<double>>("classifier.rgb_coefficient.blue") };
  int blur_range = this->declare_parameter<int>("classifier.blur_range");
  auto resource_package = this->declare_parameter<std::string>("classifier.model_package");
  auto pkg_path = ament_index_cpp::get_package_share_directory(resource_package);
  auto model_path = pkg_path + "/model/net.onnx";
  auto label_path = pkg_path + "/model/label.txt";
  double threshold = this->declare_parameter<double>("classifier.threshold");
  std::vector<std::string> ignore_classes =
    this->declare_parameter<std::vector<std::string>>("ignore_classes");
  detector->classifier_ = std::make_unique<NumberClassifier>(
    rgb_coefficient, blur_range,
    model_path, label_path, threshold, ignore_classes);

  return detector;
}

void ArmorDetectorNode::detectArmors(
  const sensor_msgs::msg::Image::ConstSharedPtr & img_msg)
{
  // Convert ROS img to cv::Mat
  auto img = cv_bridge::toCvShare(img_msg, "rgb8")->image;

  detector_->detect(img, get_parameter("light.detect_color").as_int());

  auto final_time = this->now();
  auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
  RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

  // Publish debug info
  if (debug_) {
    binary_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "mono8", detector_->binary_img_).toImageMsg());

    auto all_num_img = detector_->getAllNumbersImage();
    number_img_pub_.publish(*cv_bridge::CvImage(img_msg->header, "mono8", all_num_img).toImageMsg());
    
    detector_->drawResults(img);
    // Draw camera center
    cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    // Draw latency
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    result_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "rgb8", img).toImageMsg());
  }
}

void ArmorDetectorNode::createDebugPublishers()
{
  binary_img_pub_ = image_transport::create_publisher(this, "detector/binary_img");
  number_img_pub_ = image_transport::create_publisher(this, "detector/number_img");
  result_img_pub_ = image_transport::create_publisher(this, "detector/result_img");
}
}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)
