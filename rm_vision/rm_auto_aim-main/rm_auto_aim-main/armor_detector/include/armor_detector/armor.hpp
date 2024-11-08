// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim
{
const int RED = 0;
const int BLUE = 1;

enum class ArmorType { SMALL, LARGE, INVALID };
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

struct Armor
{
  Armor() = default;
  ArmorType type;
  cv::Rect_<float> rect; //
  float landmarks[8]; //4个关键点
  std::string number;
  float prob;
  int color;      //blue:1 , red:0
  double length;
  double width;
  double ratio;   //length s/ width
  cv::Point2f center;
  std::string classfication_result;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_
