// Copyright 2022 Chen Jun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// STL
#include <algorithm>
#include <iterator>
#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <array>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"

namespace rm_auto_aim
{
NumberClassifier::NumberClassifier(
  const std::vector<std::vector<double>>& rgb_coefficient, int blur_range,
  const std::string& model_path, const std::string& label_path, double thre,
  const std::vector<std::string>& ignore_classes)
: rgb_coefficient_(rgb_coefficient), blur_range_(blur_range),
  threshold(thre), ignore_classes_(ignore_classes)
{
  net_ = cv::dnn::readNetFromONNX(model_path);

  std::ifstream label_file(label_path);
  std::string line;

  while (std::getline(label_file, line)) {
    class_names_.push_back(line);
  }
}

cv::Mat NumberClassifier::convertToGray(const cv::Mat& img, int light_color)
{
  // split rgb
  cv::Mat tmp;
  img.convertTo(tmp, CV_64FC3);
  std::vector<cv::Mat> rgb;
  cv::split(tmp, rgb);

  // mix by coefficient
  auto&& coefficient = rgb_coefficient_[light_color];
  tmp = coefficient[0] * rgb[0] + coefficient[1] * rgb[1] + coefficient[2] * rgb[2];

  // denoise
  cv::GaussianBlur(tmp, tmp, cv::Size(blur_range_, blur_range_), 0);
  tmp.convertTo(tmp, CV_8U);
  return tmp;
}

void NumberClassifier::extractNumbers(const cv::Mat & src, std::vector<Armor> & armors, int light_color)
{
  constexpr int light_length = 12;
  // Image size after warp
  constexpr int warp_height = 28;
  constexpr int small_armor_width = 32;
  constexpr int large_armor_width = 54;
  // Number ROI size
  static const cv::Size roi_size(20, 28);

  int top_light_y = (warp_height - light_length) / 2 - 1;
  int bottom_light_y = top_light_y + light_length;

  for (auto & armor : armors) {
    // Warp perspective transform
    cv::Point2f lights_vertices[4] = {
      armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
      armor.right_light.bottom};

    int warp_width = armor.type == ArmorType::SMALL ? small_armor_width : large_armor_width;

    cv::Point2f target_vertices[4] = {
      cv::Point(0, bottom_light_y),
      cv::Point(0, top_light_y),
      cv::Point(warp_width - 1, top_light_y),
      cv::Point(warp_width - 1, bottom_light_y),
    };

    cv::Mat number_image;
    auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
    cv::warpPerspective(src, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

    // Get ROI
    number_image = number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

    cv::threshold(
      convertToGray(number_image, light_color),
      number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    armor.number_img = number_image;
  }
}

std::vector<Armor> NumberClassifier::classify(std::vector<Armor>& armors)
{
  for (auto & armor : armors) {
    cv::Mat image = armor.number_img.clone();

    // Normalize
    image = image / 255.0;

    // Create blob from image
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob);

    // Set the input blob for the neural network
    net_.setInput(blob);
    // Forward pass the image blob through the model
    cv::Mat outputs = net_.forward();

    // Do softmax
    float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::Mat softmax_prob;
    cv::exp(outputs - max_prob, softmax_prob);
    float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
    softmax_prob /= sum;

    double confidence;
    cv::Point class_id_point;
    minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
    int label_id = class_id_point.x;

    armor.confidence = confidence;
    armor.number = class_names_[label_id];

    std::stringstream result_ss;
    result_ss << armor.number << ": " << std::fixed << std::setprecision(1)
              << armor.confidence * 100.0 << "%";
    armor.classfication_result = result_ss.str();
  }

  std::vector<Armor> result;
  
  std::ranges::copy_if(armors, std::back_inserter(result),
    [this](const Armor & armor) {
      if (armor.confidence < threshold) {
        return false;
      }

      for (const auto & ignore_class : ignore_classes_) {
        if (armor.number == ignore_class) { 
          return false;
        }
      }

      bool mismatch_armor_type = false;
      
      if (armor.type == ArmorType::LARGE) {
        mismatch_armor_type =
          armor.number == "outpost" || armor.number == "2" || armor.number == "guard";
      } else if (armor.type == ArmorType::SMALL) {
        mismatch_armor_type = armor.number == "1" || armor.number == "base";
      }
      return !mismatch_armor_type;
    });
  
  return result;
}

}  // namespace rm_auto_aim