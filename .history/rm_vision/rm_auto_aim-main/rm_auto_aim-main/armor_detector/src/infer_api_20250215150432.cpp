#include "armor_detector/infer_api.hpp"
#include<chrono>

static constexpr int INPUT_W = 416;    // Width of input
static constexpr int INPUT_H = 416;    // Height of input
static constexpr int NUM_CLASSES = 8;  // Number of classes
static constexpr int NUM_COLORS = 8;   // Number of color
static constexpr int TOPK = 128;       // TopK
static constexpr float NMS_THRESH = 0.3;
static constexpr float BBOX_CONF_THRESH = 0.80;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;

namespace rm_auto_aim{

static inline int argmax(const float *ptr, int len) 
{
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }
    return max_arg;
}

std::vector<Armor> OpenvinoInfer::infer(cv::Mat &src_raw, int detect_color){
    

    std::cout<<"start infer"<<std::endl;


    auto start_time=std::chrono::steady_clock::now();





    objects.clear();
    tmp_objects.clear();
    ious.clear();

    //if (src.empty())
    //{
    //    return false;
    //}

    cv::Mat src=src_raw.clone();

    raw_size=cv::Size(src.cols,src.rows);
    auto k=float(416.0 / std::max(src.cols,src.rows));
    int new_width=int(round(src.cols*k));
    int new_height=int(round(src.rows*k));

    cv::resize(src,src,cv::Size(new_width,new_height),0,0,cv::INTER_AREA);
    int d_width=INPUT_W-new_width;
    int d_height=INPUT_H-new_height;        
    cv::copyMakeBorder(src,src,0,d_height,0,d_width,cv::BORDER_ISOLATED);

    cv::Mat pre;
    cv::Mat pre_split[3];
    src.convertTo(pre, CV_32F);
    cv::split(pre, pre_split);
    ///




    auto pre_time=std::chrono::steady_clock::now();





    // Get input tensor by index
    input_tensor = infer_request.get_input_tensor(0);
    
    infer_request.set_input_tensor(input_tensor);

    float* tensor_data = input_tensor.data<float_t>();
    // u_int8_t* tensor_data = input_tensor.data<u_int8_t>();

    auto img_offset = INPUT_H * INPUT_W;
    // Copy img into tensor
    for(int c = 0; c < 3; c++)
    {
        memcpy(tensor_data, pre_split[c].data, INPUT_H * INPUT_W * sizeof(float));
        // memcpy(tensor_data, pre_split[c].data, INPUT_H * INPUT_W * sizeof(u_int8_t));
        tensor_data += img_offset;
    }

    // ov::element::Type input_type = ov::element::f32;
    // ov::Shape input_shape = {1, 3, 416, 416};

    // // std::shared_ptr<unsigned char> input_data_ptr = pre.data;
    // auto input_data_ptr = pre.data;

    // // ת��ͼ������Ϊov::Tensor
    // input_tensor = ov::Tensor(input_type, input_shape, input_data_ptr);

    // auto st = std::chrono::steady_clock::now();
    infer_request.infer();
    // auto end = std::chrono::steady_clock::now();
    // double infer_dt = std::chrono::duration<double,std::milli>(end - st).count();
    // cout << "infer_time:" << infer_dt << endl;
    
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    float* output = output_tensor.data<float_t>();






    auto output_time=std::chrono::steady_clock::now();




    // std::vector<ArmorObject> proposals;
    // float conf_threshold = 0.65 ;
    float nms_threshold = 0.45;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;     

    for (auto stride : strides)
    {
        int num_grid_w = INPUT_W / stride;
        int num_grid_h = INPUT_H / stride;

        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride grid_stride = {g0, g1, stride};
                grid_strides.emplace_back(grid_stride);
            }
        }
    }
    // std::cout<<detect_color<<std::endl;

    const int num_anchors = grid_strides.size();
    // std::cout << num_anchors <<std::endl;
    //Travel all the anchors
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;
        const int basic_pos = anchor_idx * (9 + (NUM_COLORS) + NUM_CLASSES);

        float box_objectness = (output[basic_pos + 8]);
        int box_color = argmax(output + basic_pos + 9, NUM_COLORS);
        int box_class = argmax(output + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);

        float box_prob = box_objectness;
        if (box_prob >= BBOX_CONF_THRESH)
        {
            Armor obj;

            obj.prob = box_objectness;
          switch (box_color) {
            case PURPLE_SMALL:
            case PURPLE_BIG:
                continue; 

            case RED_SMALL:
            case RED_BIG:
                if (detect_color == 1) {
                    continue;
                }
                obj.color = 0;
                break;

            case BLUE_SMALL:
            case BLUE_BIG:
                if (detect_color == 0) {
                    continue; 
                }
                obj.color = 1;
                break;

            default:
                break;
        }

            switch (box_class) {
                case 0:
                    obj.number = "G";
                    break;
                case 1:
                    obj.number = "1";
                    break;
                case 2:
                    obj.number = "2";
                    break;
                case 3:
                    obj.number = "3";
                    break;
                case 4:
                    obj.number = "4";
                    break;
                case 5:
                    obj.number = "5";
                    break;
                case 7:
                    obj.number = "7";
                    break;
                default:
                    break;
            }

            obj.landmarks[0]=(output[basic_pos + 0] + grid0) * stride /new_width *raw_size.width;
            obj.landmarks[1]=(output[basic_pos + 1] + grid1) * stride /new_height *raw_size.height;
            obj.landmarks[2]=(output[basic_pos + 2] + grid0) * stride /new_width *raw_size.width;
            obj.landmarks[3]=(output[basic_pos + 3] + grid1) * stride /new_height *raw_size.height;
            obj.landmarks[4]=(output[basic_pos + 4] + grid0) * stride /new_width *raw_size.width;
            obj.landmarks[5]=(output[basic_pos + 5] + grid1) * stride /new_height*raw_size.height;
            obj.landmarks[6]=(output[basic_pos + 6] + grid0) * stride /new_width *raw_size.width;
            obj.landmarks[7]=(output[basic_pos + 7] + grid1) * stride /new_height *raw_size.height;



            std::cout<<"point 1:"<<obj.

            obj.length = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[6])-cv::Point2f(obj.landmarks[1]-obj.landmarks[7]));
            obj.width = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[2])-cv::Point2f(obj.landmarks[1]-obj.landmarks[3]));
            obj.ratio = obj.length / obj.width;

            obj.classfication_result=obj.number+":"+std::to_string(obj.prob * 100.0);

            // if(obj.ratio<0.4||obj.ratio>0.9) continue;

            if(obj.ratio>0.6) obj.type=ArmorType::SMALL;
            else obj.type=ArmorType::LARGE;
            
            std::vector<cv::Point2f> points;
            points.push_back(cv::Point2f(obj.landmarks[0], obj.landmarks[1]));
            points.push_back(cv::Point2f(obj.landmarks[6], obj.landmarks[7]));
            points.push_back(cv::Point2f(obj.landmarks[4], obj.landmarks[5]));
            points.push_back(cv::Point2f(obj.landmarks[2], obj.landmarks[3]));

            // Create the rectangle
            cv::Rect boundRect=cv::boundingRect(points);
            obj.rect = boundRect;
            //uhui
            //obj.center=cv::Point2f((min_x+max_x)/2,(min_y+max_y)/2);
            objects.push_back(obj);
            boxes.push_back(boundRect);
            confidences.push_back(box_prob);
        }
    } // point anchor loop
    
    
    
    
    
    auto decode_time=std::chrono::steady_clock::now();
    
    
    
    
    
    // }
    // NMS
    //std::cout<<"object_size: "<<objects.size()<<std::endl;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, BBOX_CONF_THRESH, nms_threshold, indices);
    //int index = 0, index_indices = 0;
    for(int valid_index:indices){
        if(valid_index <= (int) objects.size()){
            tmp_objects.push_back(objects[valid_index]);
        }
    }





    auto nms_time=std::chrono::steady_clock::now();




    auto pre_t=std::chrono::duration_cast<std::chrono::milliseconds>(pre_time-start_time);
    auto output_t=std::chrono::duration_cast<std::chrono::milliseconds>(output_time-pre_time);
    auto decode_t=std::chrono::duration_cast<std::chrono::milliseconds>(decode_time-output_time);
    auto nms_t=std::chrono::duration_cast<std::chrono::milliseconds>(nms_time-decode_time);

    std::cout<<"pre_t:"<<std::fixed << std::setprecision(5) << pre_t.count() << "ms"<<std::endl;
    std::cout<<"otuput_t:"<<std::fixed << std::setprecision(5) << output_t.count() << "ms"<<std::endl;
    std::cout<<"decode_t:"<<std::fixed << std::setprecision(5) << decode_t.count() << "ms"<<std::endl;
    std::cout<<"nms_t:"<<std::fixed << std::setprecision(5) << nms_t.count() << "ms"<<std::endl;

    return tmp_objects;
}



void OpenvinoInfer::drawResults(cv::Mat &img){
    for(const auto &tmp_armor : tmp_objects){
        for(int i=0;i<4;i++){
            cv::circle(img,cv::Point2f(tmp_armor.landmarks[i*2],tmp_armor.landmarks[i*2+1]),3,
            cv::Scalar(255, 255, 255),1);

            cv::line(img, cv::Point2f(tmp_armor.landmarks[i%4*2],tmp_armor.landmarks[i%4*2+1]), 
            cv::Point2f(tmp_armor.landmarks[(i+1)%4*2],tmp_armor.landmarks[(i+1)%4*2+1]), cv::Scalar(0, 255, 0), 2);

        }
        cv::putText(
            img,std::to_string(tmp_armor.prob),cv::Point2f(tmp_armor.landmarks[0],tmp_armor.landmarks[1]),
             cv::FONT_HERSHEY_SIMPLEX, 0.8,cv::Scalar(0, 255, 255), 2);
    }
    return ;
}

}