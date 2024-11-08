//
// Created by nuc on 23-5-7.
//

#ifndef OPENVINO_TEST_OPENVINOINFER_H
#define OPENVINO_TEST_OPENVINOINFER_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include"armor.hpp"
//用前先根据模型修改好要修改的地方

#define mean
namespace rm_auto_aim{

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
    ov::preprocess::PrePostProcessor *ppp;


    ov::CompiledModel compiled_model;
    ov::Shape input_shape;


    cv::Size raw_size;

    void drawResults(cv::Mat &img);

    OpenvinoInfer(){}
    std::vector<Armor> infer(cv::Mat &img,int detect_color);
    OpenvinoInfer(std::string model_path_xml, std::string model_path_bin){
        // // input_shape = {1, 1, static_cast<unsigned long>(IMAGE_HEIGHT), static_cast<unsigned long>(IMAGE_WIDTH)};
        // input_shape = { 1, static_cast<unsigned long>(IMAGE_HEIGHT), static_cast<unsigned long>(IMAGE_WIDTH),1};
        // model = core.read_model(model_path);
        // // Step . Inizialize Preprocessing for the model
        // ppp = new ov::preprocess::PrePostProcessor(model);
        // // Specify input image format
        // ppp->input().tensor().set_shape(input_shape).set_element_type(ov::element::u8).set_layout("NHWC");
        // ppp->input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);

        // //  Specify model's input layout
        // ppp->input().model().set_layout("NHWC");
        // // Specify output results format
        // ppp->output().tensor().set_element_type(ov::element::f32);
        // // Embed above steps in the graph
        // model = ppp->build();

        input_shape = {1, static_cast<unsigned long>(IMAGE_HEIGHT), static_cast<unsigned long>(IMAGE_WIDTH), 3};
        model = core.read_model(model_path_xml, model_path_bin);
        // Step . Inizialize Preprocessing for the model
        ppp = new ov::preprocess::PrePostProcessor(model);
        // Specify input image format
        ppp->input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR); 
        //NHWC:batchsize,height,width,channels
        // Specify preprocess pipeline to input image without resizing
        ppp->input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
        //  Specify model's input layout
        ppp->input().model().set_layout("NCHW");
        // Specify output results format
        ppp->output().tensor().set_element_type(ov::element::f32);
        // Embed above steps in the graph
        model = ppp->build();




        compiled_model = core.compile_model(model, "CPU",ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
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
        delete ppp;
    }
};
}

#endif //OPENVINO_TEST_OPENVINOINFER_H
