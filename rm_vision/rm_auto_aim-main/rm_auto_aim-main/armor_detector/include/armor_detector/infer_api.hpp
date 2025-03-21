// Copyright 2023 RM Vision Team
// Licensed under the MIT License.

#ifndef OPENVINO_TEST_OPENVINOINFER_H
#define OPENVINO_TEST_OPENVINOINFER_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include "armor_detector/armor.hpp"
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>

namespace rm_auto_aim {

// 性能调优参数
struct PerformanceConfig {
    bool enable_fast_keypoint_regularization = true;  // 启用快速关键点规整化
    bool use_async_inference = true;                  // 使用异步推理
    bool use_int8_inference = false;                  // 使用INT8推理
    int inference_threads = 4;                        // 推理线程数
};

struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

enum ArmorColors {
    BLUE_SMALL,
    BLUE_BIG,
    RED_SMALL,
    RED_BIG,
    GRAY_SMALL,
    GRAY_BIG,
    PURPLE_SMALL,
    PURPLE_BIG
};

class OpenvinoInfer {
public:
    std::vector<Armor> objects;
    const int IMAGE_HEIGHT = 640;
    const int IMAGE_WIDTH = 640;
    double ans;
    std::vector<double> ious;
    std::vector<Armor> tmp_objects;

    // OpenVINO模型相关
    std::shared_ptr<ov::Model> model;
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::Shape input_shape;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor;
    
    ov::InferRequest infer_requests_[2];
int current_request_idx_ = 0;
bool is_first_inference = true;
    
    // 图像和变换相关
    cv::Size raw_size;
    Eigen::Matrix<float, 3, 3> transform_matrix; // 坐标变换矩阵

    // 预分配内存缓冲区
    cv::Mat pre_split[3];
    std::vector<cv::Point2f> points_buffer;
    
    // 检测相关参数
    static constexpr int INPUT_W = 416;
    static constexpr int INPUT_H = 416;
    static constexpr int NUM_CLASSES = 8;
    static constexpr int NUM_COLORS = 8;
    static constexpr float BBOX_CONF_THRESH = 0.75;
    static constexpr float NMS_THRESH = 0.3;
    static constexpr float MERGE_CONF_ERROR = 0.15;
    static constexpr float MERGE_MIN_IOU = 0.9;
    
    // 性能配置
    PerformanceConfig perf_config;

    // 构造函数和析构函数
    OpenvinoInfer() {}
    
    OpenvinoInfer(std::string model_path_xml, std::string model_path_bin) {
        initModel(model_path_xml, model_path_bin);
    }
    
    ~OpenvinoInfer() {}
    
    // 初始化模型
    bool initModel(std::string model_path_xml, std::string model_path_bin);
    
    // 推理函数
    std::vector<Armor> infer(cv::Mat &img, int detect_color);
    
    // 绘制结果
    void drawResults(cv::Mat &img);
    
    // 辅助函数
    double sigmoid(double x) {
        if(x > 0)
            return 1.0 / (1.0 + exp(-x));
        else
            return exp(x) / (1.0 + exp(x));
    }

    double cal_iou(const cv::Rect& r1, const cv::Rect& r2) {
        float x_left = std::fmax(r1.x, r2.x);
        float y_top = std::fmax(r1.y, r2.y); 
        float x_right = std::fmin(r1.x + r1.width, r2.x + r2.width);
        float y_bottom = std::fmin(r1.y + r1.height, r2.y + r2.height);

        if (x_right < x_left || y_bottom < y_top) {
            return 0.0; 
        }

        double in_area = (x_right - x_left) * (y_bottom - y_top);
        double un_area = r1.area() + r2.area() - in_area; 

        return in_area / un_area;
    }

    double meaning(float x, int len) {
        if(len == 0) ans = x;
        else {
            ans = (len * ans + x) / (len+1);
        }
        return ans;
    }

    // 图像预处理，现在返回缩放比例
    cv::Mat scaledResize(cv::Mat& img, float& scale_factor);
    
    // 解码和后处理函数
    void generateProposals(const std::vector<GridAndStride>& grid_strides, 
                          const float* feat_ptr, float prob_threshold, 
                          std::vector<Armor>& proposals);
    
    void nmsMergeBoxes(std::vector<Armor>& proposals, std::vector<Armor>& objects);
    
    // 关键点处理函数
    bool needsRegularization(const Armor& armor);
    void regularizeKeypoints(Armor& armor);
    void fastRegularizeKeypoints(Armor& armor);  // 快速版本
    
    // 生成网格步长
    void generateGridsAndStride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
    
    // 设置性能配置
    void setPerformanceConfig(const PerformanceConfig& config) {
        perf_config = config;
    }
};

}

#endif //OPENVINO_TEST_OPENVINOINFER_H