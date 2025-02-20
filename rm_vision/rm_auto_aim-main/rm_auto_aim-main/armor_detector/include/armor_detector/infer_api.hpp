//
// Created by nuc on 23-5-7.
//

#ifndef OPENVINO_TEST_OPENVINOINFER_H
#define OPENVINO_TEST_OPENVINOINFER_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include"armor.hpp"
//��ǰ�ȸ���ģ���޸ĺ�Ҫ�޸ĵĵط�

#define mean
namespace rm_auto_aim{

    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };

    enum ArmorTypes
    {
        BLUE_SMALL,
        BLUE_BIG,
        RED_SMALL,
        RED_BIG,
        GRAY_SMALL,
        GRAY_BIG,
        PURPLE_SMALL,
        PURPLE_BIG
    };

// struct ArmorObject
// {
//     int area;
//     cv::Point2f apex[4];
//     cv::Rect_<float> rect;
//     int cls;
//     int color;
//     float prob;
//     std::vector<cv::Point2f> pts;
// };

class OpenvinoInfer {
public:
    std::vector<Armor> objects;
    const int IMAGE_HEIGHT = 640;
    const int IMAGE_WIDTH = 640;
    double ans;
    std::vector<double> ious;
    std::vector<Armor> tmp_objects;    // ��ʱĿ�ꣿ
    std::shared_ptr<ov::Model> model;
    ov::Core core;
    // ov::preprocess::PrePostProcessor *ppp;
    ov::CompiledModel compiled_model;
    ov::Shape input_shape;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor;
    cv::Size raw_size;
    // std::vector<ArmorObject> objects

    void drawResults(cv::Mat &img);

    OpenvinoInfer(){}
    std::vector<Armor> infer(cv::Mat &img,int detect_color);
    OpenvinoInfer(std::string model_path_xml, std::string model_path_bin){
        std::cout << "Start initialize model..." << std::endl;

        // Setting Configuration Values
        core.set_property("CPU", ov::enable_profiling(true));
    
        //Step 1.Create openvino runtime core
        model = core.read_model(model_path_xml, model_path_bin);
        // model = core.import_model();

        // Preprocessing
        ov::preprocess::PrePostProcessor ppp(model);    //PrePostProcessor ������������������Ԥ�����ͺ������裬����׼��ģ�͵��������ݺʹ�����������
        ppp.input().tensor().set_element_type(ov::element::f32);  // ppp.input() ��ȡģ�͵����벿�֡�tensor() ��������������������Ϣ��set_element_type(ov::element::f32) ����������������������Ϊ 32 λ��������f32����
        // ppp.input().tensor().set_element_type(ov::element::u8);    // �޷��Ű�λ����

        // Set output precision           // �����������
        ppp.output().tensor().set_element_type(ov::element::f32);
        // ppp.output().tensor().set_element_type(ov::element::u8);
        
        //��Ԥ��������ԭʼģ��
        ppp.build(); 

        //Step 2. Compile the model    ����ģ��
        compiled_model = core.compile_model(     // �����ѧϰģ�ͱ���ɿ�����ָ��Ӳ����ִ�еĸ�ʽ��
            model,
            "CPU",
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
        );

        infer_request = compiled_model.create_infer_request();
    }
    
    double sigmoid(double x) {
        if(x>0)
            return 1.0 / (1.0 + exp(-x));
        else
            return exp(x) / (1.0 + exp(x));
    }

        double cal_iou(const cv::Rect& r1, const cv::Rect& r2)
    {
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

    double meaning(float x, int len){
        if(len == 0) ans = x;
        else{
            ans = (len * ans + x) / (len+1);
        }
        return ans;
    }

    ~OpenvinoInfer(){
        // delete ppp;
    }
};
}

#endif //OPENVINO_TEST_OPENVINOINFER_H
