#include "armor_detector/infer_api.hpp"

static constexpr int INPUT_W = 416;    // Width of input
static constexpr int INPUT_H = 416;    // Height of input
static constexpr int NUM_CLASSES = 8;  // Number of classes
static constexpr int NUM_COLORS = 8;   // Number of color
static constexpr int TOPK = 128;       // TopK
static constexpr float NMS_THRESH = 0.3;
static constexpr float BBOX_CONF_THRESH = 0.60;
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

// static void qsort_descent_inplace(std::vector<Armor>& faceobjects, int left, int right)
// {
//     int i = left;
//     int j = right;
//     float p = faceobjects[(left + right) / 2].prob;

//     while (i <= j)
//     {
//         while (faceobjects[i].prob > p)
//             i++;

//         while (faceobjects[j].prob < p)
//             j--;

//         if (i <= j)
//         {
//             // swap
//             std::swap(faceobjects[i], faceobjects[j]);
//             i++;
//             j--;
//         }
//     }
//     // #pragma omp parallel sections
//     // {
//     //     #pragma omp section
//     //     {
//     //         if (left < j) qsort_descent_inplace(faceobjects, left, j);
//     //     }
//     //     #pragma omp section
//     //     {
//     //         if (i < right) qsort_descent_inplace(faceobjects, i, right);
//     //     }
//     // }
//     if (left < j) qsort_descent_inplace(faceobjects, left, j);
//     if (i < right) qsort_descent_inplace(faceobjects, i, right);
// }

// static void qsort_descent_inplace(std::vector<Armor>& objects)
// {
//     if (objects.empty())
//         return;

//     qsort_descent_inplace(objects, 0, objects.size() - 1);
// }


std::vector<Armor> OpenvinoInfer::infer(cv::Mat &src, int detect_color){
    objects.clear();
    tmp_objects.clear();
    ious.clear();

    // if (src.empty())
    // {
    //     return false;
    // }

    // cv::Mat pr_img = scaledResize(src, transfrom_matrix);
    // dw = this->dw;

    //ͼ��任
    //���Ų����߽�
    raw_size=cv::Size(src.cols,src.rows);
    auto k=float(416.0 / std::max(src.cols,src.rows));
    int new_width=int(round(src.cols*k));
    int new_height=int(round(src.rows*k));
    //std::cout<<"1"<<std::endl;
    //std::cout<<new_width<<std::endl;
    //std::cout<<new_height<<std::endl;

    cv::resize(src,src,cv::Size(new_width,new_height),0,0,cv::INTER_AREA);
    int d_width=INPUT_W-new_width;
    int d_height=INPUT_H-new_height;
    cv::copyMakeBorder(src,src,0,d_height,0,d_width,cv::BORDER_ISOLATED);
    //ԭͼ�������Ͻǣ����ĸ������ֱ�Ϊtop,bottom,left,right

    cv::Mat pre;
    cv::Mat pre_split[3];
    src.convertTo(pre, CV_32F);
    cv::split(pre, pre_split);

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

    src=src(cv::Rect(0,0,new_width,new_height));
    cv::resize(src,src,raw_size,0,0,cv::INTER_LINEAR);

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

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        // float x_1 = (output[basic_pos + 0] + grid0) * stride;
        // float y_1 = (output[basic_pos + 1] + grid1) * stride;
        // float x_2 = (output[basic_pos + 2] + grid0) * stride;
        // float y_2 = (output[basic_pos + 3] + grid1) * stride;
        // float x_3 = (output[basic_pos + 4] + grid0) * stride;
        // float y_3 = (output[basic_pos + 5] + grid1) * stride;
        // float x_4 = (output[basic_pos + 6] + grid0) * stride;
        // float y_4 = (output[basic_pos + 7] + grid1) * stride;
        
        float box_objectness = (output[basic_pos + 8]);
        int box_color = argmax(output + basic_pos + 9, NUM_COLORS);
        int box_class = argmax(output + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);

        // cout << "output:" << endl;
        // for (int ii = 0; ii < 25; ii++)
        // {
        //     cout << feat_ptr[basic_pos + ii] << " ";
        // }
        // cout << endl;
        // float color_conf = (feat_ptr[basic_pos + 9 + box_color]);
        // float cls_conf = (feat_ptr[basic_pos + 9 + NUM_COLORS + box_class]);
        // float box_prob = (box_objectness + cls_conf + color_conf) / 3.0;
        float box_prob = box_objectness;
        if (box_prob >= BBOX_CONF_THRESH)
        {
            // std::cout << box_color << std::endl;
            // ArmorObject obj;
            Armor obj;

            if(box_color == PURPLE_SMALL || box_color == PURPLE_BIG)
            {
                continue;
            }
            else if(detect_color == 1 && (box_color == RED_SMALL || box_color == RED_BIG))   // detect blue
            {
                continue;
            }
            else if(detect_color == 0 && (box_color == BLUE_SMALL || box_color == BLUE_BIG))   // detect red
            {
                continue;
            }

            // Eigen::Matrix<float,3,4> apex_norm;
            // Eigen::Matrix<float,3,4> apex_dst;

            // apex_norm << x_1, x_2, x_3, x_4,
            //                 y_1, y_2, y_3, y_4,
            //                 1,   1,   1,   1;
            
            // apex_dst = transform_matrix * apex_norm;

            // for (int i = 0; i < 4; i++)
            // {
            //     obj.apex[i] = cv::Point2f(apex_dst(0,i), apex_dst(1,i));
            //     obj.pts.push_back(obj.apex[i]);
            // }
            obj.prob = box_objectness;

            if(box_color == RED_SMALL || box_color == RED_BIG) obj.color = 0;
            if(box_color == BLUE_SMALL || box_color == BLUE_BIG) obj.color = 1;

            if(box_class==0) obj.number="G";
            if(box_class==1) obj.number="1";
            if(box_class==2) obj.number="2";
            if(box_class==3) obj.number="3";
            if(box_class==4) obj.number="4";
            if(box_class==5) obj.number="5";
            if(box_class==7) obj.number="7";

            obj.landmarks[0]=(output[basic_pos + 0] + grid0) * stride /new_width *raw_size.width;
            obj.landmarks[1]=(output[basic_pos + 1] + grid1) * stride /new_height *raw_size.height;
            obj.landmarks[2]=(output[basic_pos + 2] + grid0) * stride /new_width *raw_size.width;
            obj.landmarks[3]=(output[basic_pos + 3] + grid1) * stride /new_height *raw_size.height;
            obj.landmarks[4]=(output[basic_pos + 4] + grid0) * stride /new_width *raw_size.width;
            obj.landmarks[5]=(output[basic_pos + 5] + grid1) * stride /new_height*raw_size.height;
            obj.landmarks[6]=(output[basic_pos + 6] + grid0) * stride /new_width *raw_size.width;
            obj.landmarks[7]=(output[basic_pos + 7] + grid1) * stride /new_height *raw_size.height;
            obj.length = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[6])-cv::Point2f(obj.landmarks[1]-obj.landmarks[7]));
            obj.width = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[2])-cv::Point2f(obj.landmarks[1]-obj.landmarks[3]));
            obj.ratio = obj.length / obj.width;

            obj.classfication_result=obj.number+":"+std::to_string(obj.prob * 100.0);

            // if(obj.ratio<0.4||obj.ratio>0.9) continue;

            if(obj.ratio>0.6) obj.type=ArmorType::SMALL;
            else obj.type=ArmorType::LARGE;
            
            std::vector<cv::Point2f> points;
            //landmarksΪ������ʱ�룬pointsӦΪ����˳ʱ��
            points.push_back(cv::Point2f(obj.landmarks[0], obj.landmarks[1]));
            points.push_back(cv::Point2f(obj.landmarks[6], obj.landmarks[7]));
            points.push_back(cv::Point2f( obj.landmarks[4], obj.landmarks[5]));
            points.push_back(cv::Point2f( obj.landmarks[2], obj.landmarks[3]));


            // Find the minimum and maximum x and y coordinates
            float min_x = points[0].x;
            float max_x = points[0].x;
            float min_y = points[0].y;
            float max_y = points[0].y;

            for (int i = 1; i < (int)points.size(); i++)
            {
                if (points[i].x < min_x)
                    min_x = points[i].x;
                if (points[i].x > max_x)
                    max_x = points[i].x;
                if (points[i].y < min_y)
                    min_y = points[i].y;
                if (points[i].y > max_y)
                    max_y = points[i].y;
            }

            // Create the rectangle
            cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);
            obj.rect = rect;
            obj.center=cv::Point2f((min_x+max_x)/2,(min_y+max_y)/2);
            objects.push_back(obj);
            boxes.push_back(rect);
            confidences.push_back(box_prob);

            // std::vector<cv::Point2f> tmp(obj.apex, obj.apex + 4);
            // obj.rect = cv::boundingRect(tmp);
            // obj.cls = box_class;
            // obj.color = box_color;
            // obj.prob = box_prob;

            // // cout << "output:";
            // // for (int i = 0; i < 4; i++)
            // // {
            // //     cout << " " << "(" << obj.pts[i].x << "," << obj.pts[i].y << ") "; 
            // // }
            // // cout << "obj_prob:" << obj.prob << " obj_color:" << obj.color << " obj_cls:" << obj.cls << endl;

            // proposals.push_back(obj);
        }
    } // point anchor loop

/*
    qsort_descent_inplace(objects);   // �������Ŷ�����
    if (proposals.size() >= TOPK) 
        proposals.resize(TOPK);
*/

    // cout << "output:" << " ";
    // for (int ii = 0; ii < 25; ii++)
    // {
    //     cout << output[ii] << " ";
    // }
    // cout << endl;
    // u_int8_t* output = output_tensor.data<u_int8_t>();
    // std::cout << &output << std::endl;

    // int img_w = src.cols;
    // int img_h = src.rows;    


    //        std::cout << "The shape of output tensor:"<<output_shape << std::endl;
    // 25200 x 85 Matrix
    // cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());  // �������õ����������ת��Ϊ OpenCV �� cv::Mat ��ʽ
    // float conf_threshold = 0.65 ;    // ���д��붨����һ�����Ŷ���ֵ conf_threshold��ͨ�����ڹ��˵����Ŷȵļ���������磬��Ŀ�����У����ĳ������������Ŷȵ������ֵ������Ժ��Ըý��������/����
    // float nms_threshold = 0.45;      // ���д��붨����һ���Ǽ���ֵ���ƣ�NMS����ֵ nms_threshold�����ڴ����ص��ļ�����Ŀ�����У�NMS ��һ�ֳ��õĺ�������������ɾ���ظ��ļ�������������ŵĿ�
    // std::vector<cv::Rect> boxes;
    // std::vector<int> class_ids;
    // std::vector<float> class_scores;
    // std::vector<float> confidences;
    // // cx,cy,w,h,confidence,c1,c2,...c80
    // for (int i = 0; i < output_buffer.rows; i++) {
    //     //ͨ�����Ŷ���ֵɸѡ
    //     float confidence = output_buffer.at<float>(i, 8);   // ��Ŀ���������У����ͨ������������Ϣ���������ǩ�����ŶȺͱ߽�����ꡣ�� 8 ��ͨ����ָ�ü�������Ŷȷ�����
    //     confidence = sigmoid(confidence);
    //     if (confidence < conf_threshold)
    //     {
    //         continue;
    //     }
    //     //��ɫ������������
    //     cv::Mat color_scores = output_buffer.row(i).colRange(9, 13);  //color   �������ͨ����ʾ����������ɫ��صĵ÷֣��������ڽ�һ���Ĵ��������
    //     cv::Mat classes_scores = output_buffer.row(i).colRange(13, 22); //num   ������ݱ�ʾ�������صĵ÷֣�ͨ�����ڱ�ʾ��ͬ���ĸ��ʻ�÷�
    //     cv::Point class_id,color_id;
    //     int _class_id,_color_id;
    //     double score_color, score_num;
    //     // ��ȡ���������ɫ: ͨ���ҵ����÷ּ�������������ȷ��������ܵļ��������ɫ������ڷ���ͽ�һ���ľ���������Ҫ��
    //     cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    //     cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    //      //std::cout<<score_color<<" "<<color_id.x<<std::endl;
    //     // std::cout<<score_num<<" "<<class_id.x<<std::endl;
    //     // class score: 0~3
    // //    cout<<"class_id.x:"<<class_id.x<<endl;
    // //    cout<<"detect_color:"<<detect_color<<endl;
    //     // None ����Purple ����
    //     if(color_id.x == 2 || color_id.x == 3)
    //     {
    //         continue;
    //     }
    //     else if(detect_color == 0 && color_id.x == 1)   // detect blue
    //     {
    //         continue;
    //     }
    //     else if(detect_color == 1 && color_id.x == 0)   // detect red
    //     {
    //         continue;
    //     }
    //     _class_id = class_id.x;
    //     _color_id = color_id.x;
    //     Armor obj;
    //     obj.prob = confidence;
    //     obj.color = _color_id;
    //     if(_class_id==0) obj.number="G";
    //     if(_class_id==1) obj.number="1";
    //     if(_class_id==2) obj.number="2";
    //     if(_class_id==3) obj.number="3";
    //     if(_class_id==4) obj.number="4";
    //     if(_class_id==5) obj.number="5";
    //     if(_class_id==0) obj.number="0";


    //     obj.landmarks[0]=output_buffer.at<float>(i, 0);
    //     obj.landmarks[1]=output_buffer.at<float>(i, 1);
    //     obj.landmarks[2]=output_buffer.at<float>(i, 2);
    //     obj.landmarks[3]=output_buffer.at<float>(i, 3);
    //     obj.landmarks[4]=output_buffer.at<float>(i, 4);
    //     obj.landmarks[5]=output_buffer.at<float>(i, 5);
    //     obj.landmarks[6]=output_buffer.at<float>(i, 6);
    //     obj.landmarks[7]=output_buffer.at<float>(i, 7);
    //     obj.length = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[6])-cv::Point2f(obj.landmarks[1]-obj.landmarks[7]));
    //     obj.width = cv::norm(cv::Point2f(obj.landmarks[0] - obj.landmarks[2])-cv::Point2f(obj.landmarks[1]-obj.landmarks[3]));
    //     obj.ratio = obj.length / obj.width;


    //     obj.classfication_result=obj.number+":"+std::to_string(obj.prob * 100.0);
    //     if(obj.ratio<0.4||obj.ratio>0.9) continue;

    //     if(obj.ratio>0.6) obj.type=ArmorType::SMALL;
    //     else obj.type=ArmorType::LARGE;

    //     std::vector<cv::Point2f> points;
    //     //landmarksΪ������ʱ�룬pointsӦΪ����˳ʱ��
    //     points.push_back(cv::Point2f(obj.landmarks[0], obj.landmarks[1]));
    //     points.push_back(cv::Point2f(obj.landmarks[6], obj.landmarks[7]));
    //     points.push_back(cv::Point2f( obj.landmarks[4], obj.landmarks[5]));
    //     points.push_back(cv::Point2f( obj.landmarks[2], obj.landmarks[3]));


    //     // Find the minimum and maximum x and y coordinates
    //     float min_x = points[0].x;
    //     float max_x = points[0].x;
    //     float min_y = points[0].y;
    //     float max_y = points[0].y;

    //     for (int i = 1; i < (int)points.size(); i++)
    //     {
    //         if (points[i].x < min_x)
    //             min_x = points[i].x;
    //         if (points[i].x > max_x)
    //             max_x = points[i].x;
    //         if (points[i].y < min_y)
    //             min_y = points[i].y;
    //         if (points[i].y > max_y)
    //             max_y = points[i].y;
    //     }

    //     // Create the rectangle
    //     cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);
    //     obj.rect = rect;
    //     obj.center=cv::Point2f((min_x+max_x)/2,(min_y+max_y)/2);
    //     objects.push_back(obj);
    //     boxes.push_back(rect);
    //     confidences.push_back(score_num);
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
// OpenvinoInfer::OpenvinoInfer(string model_path_xml, string model_path_bin, string device){
//     input_shape = {1, static_cast<unsigned long>(IMAGE_HEIGHT), static_cast<unsigned long>(IMAGE_WIDTH), 3};
//     model = core.read_model(model_path_xml, model_path_bin);
//     // Step . Inizialize Preprocessing for the model
//     ppp = new ov::preprocess::PrePostProcessor(model);
//     // Specify input image format
//     ppp->input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR); 
//     //NHWC:batchsize,height,width,channels
//     // Specify preprocess pipeline to input image without resizing
//     ppp->input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
//     //  Specify model's input layout
//     ppp->input().model().set_layout("NCHW");
//     // Specify output results format
//     ppp->output().tensor().set_element_type(ov::element::f32);
//     // Embed above steps in the graph
//     model = ppp->build();

//     compiled_model = core.compile_model(model, device);
// }
