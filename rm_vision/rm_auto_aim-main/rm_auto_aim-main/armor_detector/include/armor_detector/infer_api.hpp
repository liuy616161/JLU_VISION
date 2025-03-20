// Copyright 2023 RM Vision Team
// Licensed under the MIT License.

#ifndef OPENVINO_TEST_OPENVINOINFER_H
#define OPENVINO_TEST_OPENVINOINFER_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <Eigen/Dense>
#include "armor_detector/armor.hpp"

namespace rm_auto_aim {

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

    std::shared_ptr<ov::Model> model;
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::Shape input_shape;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor;
    cv::Size raw_size;
    Eigen::Matrix<float, 3, 3> transform_matrix; // 坐标变换矩阵

    // 检测相关参数
    static constexpr int INPUT_W = 416;
    static constexpr int INPUT_H = 416;
    static constexpr int NUM_CLASSES = 8;
    static constexpr int NUM_COLORS = 8;
    static constexpr float BBOX_CONF_THRESH = 0.75; // 提高置信度阈值
    static constexpr float NMS_THRESH = 0.3;
    static constexpr float MERGE_CONF_ERROR = 0.15;
    static constexpr float MERGE_MIN_IOU = 0.9;

    void drawResults(cv::Mat &img);

    OpenvinoInfer() {}
    std::vector<Armor> infer(cv::Mat &img, int detect_color);
    
    OpenvinoInfer(std::string model_path_xml, std::string model_path_bin) {
        std::cout << "Start initialize model..." << std::endl;

        // 设置配置
        core.set_property("CPU", ov::enable_profiling(true));
    
        // 步骤1：创建openvino运行时核心
        model = core.read_model(model_path_xml, model_path_bin);

        // 预处理
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::f32);

        // 设置输出精度
        ppp.output().tensor().set_element_type(ov::element::f32);
        
        // 将预处理集成到原始模型
        ppp.build(); 

        // 步骤2：编译模型
        compiled_model = core.compile_model(
            model,
            "CPU",
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
        );

        // 步骤3：创建推理请求
        infer_request = compiled_model.create_infer_request();
    }
    
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

    // 新增：图像预处理函数
    cv::Mat scaledResize(cv::Mat& img);
    
    // 新增：检测关键点
    void generateProposals(std::vector<GridAndStride>& grid_strides, const float* feat_ptr,
                          float prob_threshold, std::vector<Armor>& proposals);
    
    // 新增：非极大值抑制与合并
    void nmsMergeBoxes(std::vector<Armor>& proposals, std::vector<Armor>& objects);
    
    // 新增：验证几何约束
    bool validateGeometry(const Armor& armor);
    
    // 新增：规整化关键点
    void regularizeKeypoints(Armor& armor);

    ~OpenvinoInfer() {}
};

}

#endif //OPENVINO_TEST_OPENVINOINFER_H