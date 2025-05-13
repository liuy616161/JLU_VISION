#pragma once

#include "Calculator.h"
#include "Detector.h"
#include "Param.h"
#include "Utility.h"

#define CONFIG_PATH "/config/config.yaml"

namespace power_rune {



class PowerRune {
   public:
    PowerRune();
    bool runOnce(const cv::Mat& image, double pitch, double yaw, double roll = 0.0);

    cv::Mat get_img_show(){return m_detector.get_imgShow();}
    cv::Mat get_img_arrow(){return m_detector.get_imgArrow();}
    cv::Mat get_img_armor(){return m_detector.get_imgArmor();}
    cv::Mat get_img_src(){return m_detector.get_imgsrc();}


   public:
    Param m_param;
    Detector m_detector;
    Calculator m_calculator;
};

}  // namespace power_rune
