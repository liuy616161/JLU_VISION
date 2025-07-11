// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <vector>

#include "armor_detector/detector.hpp"

namespace rm_auto_aim
{
Detector::Detector(
  const std::vector<int>& light_thres, const LightParams & l_, const ArmorParams & a_)
: light_thres_(light_thres), l_(l_), a_(a_)
{
}

void Detector::detect(const cv::Mat & input, int detect_color)
{
  findLights(input, detect_color);
  matchLights(detect_color);
  classifier_->extractNumbers(input, armors_, detect_color);
  classified_armors_ = classifier_->classify(armors_);
}

cv::Mat Detector::preprocessImage(const cv::Mat & rgb_img, int detect_color)
{
  cv::Mat gray_img;
  cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, light_thres_[detect_color], 255, cv::THRESH_BINARY);

  return binary_img;
}

void Detector::findLights(const cv::Mat & rgb_img, int detect_color)
{
  binary_img_ = preprocessImage(rgb_img, detect_color);

  using std::vector;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_img_, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  lights_.clear();

  for (const auto & contour : contours) {
    if (contour.size() < 5) continue;

    auto r_rect = cv::minAreaRect(contour);
    auto light = Light(r_rect);

    if (isLight(light)) {
      auto rect = light.boundingRect();

      if (  // Avoid assertion failed
        0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rgb_img.cols && 0 <= rect.y &&
        0 <= rect.height && rect.y + rect.height <= rgb_img.rows) {
        auto rgb_sum = cv::sum(rgb_img(rect));
        light.color = rgb_sum[0] > rgb_sum[2] ? RED : BLUE;
        lights_.emplace_back(light);
      }
    }
  }
}

bool Detector::isLight(const Light & light)
{
  // The ratio of light (short side / long side)
  float ratio = light.width / light.length;
  bool ratio_ok = l_.min_ratio < ratio && ratio < l_.max_ratio;
  bool angle_ok = light.tilt_angle < l_.max_angle;
  return ratio_ok && angle_ok;
}

void Detector::matchLights(int detect_color)
{
  armors_.clear();

  // Loop all the pairing of lights
  for (auto light_1 = lights_.begin(); light_1 != lights_.end(); light_1++) {
    for (auto light_2 = light_1 + 1; light_2 != lights_.end(); light_2++) {
      if (light_1->color != detect_color || light_2->color != detect_color) {
        continue;
      }
      
      auto type = isArmor(*light_1, *light_2);

      if (type != ArmorType::INVALID) {
        auto armor = Armor(*light_1, *light_2);
        armor.type = type;
        armors_.emplace_back(armor);
      }
    }
  }
}

ArmorType Detector::isArmor(const Light & light_1, const Light & light_2)
{
  // Ratio of the length of 2 lights (short side / long side)
  float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                             : light_2.length / light_1.length;
  bool light_ratio_ok = light_length_ratio > a_.min_light_ratio;

  // Distance between the center of 2 lights (unit : light length)
  float avg_light_length = (light_1.length + light_2.length) / 2;
  float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
  bool center_distance_ok = (a_.min_small_center_distance <= center_distance &&
                             center_distance < a_.max_small_center_distance) ||
                            (a_.min_large_center_distance <= center_distance &&
                             center_distance < a_.max_large_center_distance);

  // Angle of light center connection
  cv::Point2f diff = light_1.center - light_2.center;
  float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
  bool angle_ok = angle < a_.max_angle;
  
  bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

  // Judge armor type
  ArmorType type;
  if (is_armor) {
    type = center_distance > a_.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
  } else {
    type = ArmorType::INVALID;
  }

  return type;
}

cv::Mat Detector::getAllNumbersImage()
{
  if (armors_.empty()) {
    return cv::Mat(cv::Size(20, 14), CV_8U, cv::Scalar{255});
  } else {
    std::vector<cv::Mat> number_imgs;
    number_imgs.reserve(armors_.size());

    for (auto & armor : armors_) {
      number_imgs.emplace_back(armor.number_img);
    }

    cv::Mat all_num_img;
    cv::hconcat(number_imgs, all_num_img);
    return all_num_img;
  }
}

void Detector::drawResults(cv::Mat & img)
{
  // Draw Lights
  for (const auto & light : lights_) {
    cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
    cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
    auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
    cv::line(img, light.top, light.bottom, line_color, 1);
  }

  // Draw armors
  for (const auto & armor : classified_armors_) {
    cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
    cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
  }

  // Show numbers and confidence
  for (const auto & armor : armors_) {
    cv::putText(
      img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);
  }
}

}  // namespace rm_auto_aim
