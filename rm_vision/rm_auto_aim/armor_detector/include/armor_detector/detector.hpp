// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// STD
#include <cmath>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"

namespace rm_auto_aim
{
class Detector
{
public:
  struct LightParams
  {
    // width / height
    double min_ratio;
    double max_ratio;
    // vertical angle
    double max_angle;
  };

  struct ArmorParams
  {
    double min_light_ratio;
    // light pairs distance
    double min_small_center_distance;
    double max_small_center_distance;
    double min_large_center_distance;
    double max_large_center_distance;
    // horizontal angle
    double max_angle;
  };

  Detector(const std::vector<int>& light_thres, const LightParams & l, const ArmorParams & a);

  void detect(const cv::Mat & input, int detect_color);

  // For debug usage
  cv::Mat getAllNumbersImage();
  void drawResults(cv::Mat & img);

  std::vector<int> light_thres_;
  LightParams l_;
  ArmorParams a_;

  std::unique_ptr<NumberClassifier> classifier_;

  // Debug msgs
  cv::Mat binary_img_;
  std::vector<Armor> classified_armors_;

private:
  cv::Mat preprocessImage(const cv::Mat & input, int detect_color);
  void findLights(const cv::Mat& rbg_img, int detect_color);
  void matchLights(int detect_color);
  bool isLight(const Light & possible_light);
  ArmorType isArmor(const Light& light_1, const Light& light_2);

  std::vector<Light> lights_;
  std::vector<Armor> armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_HPP_
