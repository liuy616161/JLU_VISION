// Copyright 2023 RM Vision Team
// Licensed under the MIT License.

#include "../../include/inference/infer_api.hpp"
#include <algorithm>
#include <cmath>

namespace rm_auto_aim {

static inline int argmax(const float *ptr, int len) 
{
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }
    return max_arg;
}

cv::Mat OpenvinoInfer::scaledResize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols * 1.0f), INPUT_H / (img.rows * 1.0f));
    int unpad_w = std::round(img.cols * r);
    int unpad_h = std::round(img.rows * r);
    
    cv::Mat re;
    cv::resize(img, re, cv::Size(unpad_w, unpad_h));
    
    int dw = INPUT_W - unpad_w;
    int dh = INPUT_H - unpad_h;
    
    dw /= 2;
    dh /= 2;
    
    // 构建变换矩阵
    transform_matrix = Eigen::Matrix<float, 3, 3>::Identity();
    transform_matrix(0, 0) = 1.0f / r;
    transform_matrix(1, 1) = 1.0f / r;
    transform_matrix(0, 2) = -dw / r;
    transform_matrix(1, 2) = -dh / r;
    
    cv::Mat out;
    cv::copyMakeBorder(re, out, dh, dh, dw, dw, cv::BORDER_CONSTANT);
    
    return out;
}

void OpenvinoInfer::generateProposals(std::vector<GridAndStride>& grid_strides, const float* feat_ptr,
                                    float prob_threshold, std::vector<Armor>& proposals) {
    const int num_anchors = grid_strides.size();
    
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;
        const int basic_pos = anchor_idx * (9 + NUM_COLORS + NUM_CLASSES);

        // 提取关键点预测
        float x_1 = (feat_ptr[basic_pos + 0] + grid0) * stride;
        float y_1 = (feat_ptr[basic_pos + 1] + grid1) * stride;
        float x_2 = (feat_ptr[basic_pos + 2] + grid0) * stride;
        float y_2 = (feat_ptr[basic_pos + 3] + grid1) * stride;
        float x_3 = (feat_ptr[basic_pos + 4] + grid0) * stride;
        float y_3 = (feat_ptr[basic_pos + 5] + grid1) * stride;
        float x_4 = (feat_ptr[basic_pos + 6] + grid0) * stride;
        float y_4 = (feat_ptr[basic_pos + 7] + grid1) * stride;
        
        float box_objectness = feat_ptr[basic_pos + 8];
        int box_color = argmax(feat_ptr + basic_pos + 9, NUM_COLORS);
        int box_class = argmax(feat_ptr + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);

        float box_prob = box_objectness;
        
        if (box_prob >= prob_threshold) {
            Armor obj;
            
            // 过滤颜色
            if(box_color == PURPLE_SMALL || box_color == PURPLE_BIG) {
                continue;
            } else if(detect_color == 1 && (box_color == RED_SMALL || box_color == RED_BIG)) {   // detect blue
                continue;
            } else if(detect_color == 0 && (box_color == BLUE_SMALL || box_color == BLUE_BIG)) {   // detect red
                continue;
            }
            
            // 设置属性
            obj.prob = box_prob;
            
            // 设置颜色
            if(box_color == RED_SMALL || box_color == RED_BIG) obj.color = RED;
            if(box_color == BLUE_SMALL || box_color == BLUE_BIG) obj.color = BLUE;
            
            // 设置数字
            if(box_class==0) obj.number="G";
            if(box_class==1) obj.number="1";
            if(box_class==2) obj.number="2";
            if(box_class==3) obj.number="3";
            if(box_class==4) obj.number="4";
            if(box_class==5) obj.number="5";
            if(box_class==7) obj.number="7";
            
            // 使用矩阵变换解码关键点坐标
            Eigen::Matrix<float, 3, 4> apex_norm;
            Eigen::Matrix<float, 3, 4> apex_dst;
            
            // 构建原始预测矩阵
            apex_norm << x_1, x_2, x_3, x_4,
                         y_1, y_2, y_3, y_4,
                         1,   1,   1,   1;
            
            // 应用变换矩阵
            apex_dst = transform_matrix * apex_norm;
            
            // 提取变换后的坐标
            obj.landmarks[0] = apex_dst(0, 0);  // x1
            obj.landmarks[1] = apex_dst(1, 0);  // y1
            obj.landmarks[2] = apex_dst(0, 1);  // x2
            obj.landmarks[3] = apex_dst(1, 1);  // y2
            obj.landmarks[4] = apex_dst(0, 2);  // x3
            obj.landmarks[5] = apex_dst(1, 2);  // y3
            obj.landmarks[6] = apex_dst(0, 3);  // x4
            obj.landmarks[7] = apex_dst(1, 3);  // y4
            
            // 计算装甲板属性
            obj.length = cv::norm(cv::Point2f(obj.landmarks[0], obj.landmarks[1]) - 
                                 cv::Point2f(obj.landmarks[6], obj.landmarks[7]));
            obj.width = cv::norm(cv::Point2f(obj.landmarks[0], obj.landmarks[1]) - 
                                cv::Point2f(obj.landmarks[2], obj.landmarks[3]));
            obj.ratio = obj.length / obj.width;
            
            obj.classfication_result = obj.number + ":" + std::to_string(obj.prob * 100.0);
            
            // 根据比例确定装甲板类型
            if(obj.ratio > 0.6) obj.type = ArmorType::SMALL;
            else obj.type = ArmorType::LARGE;
            
            // 计算边界框和中心点
            std::vector<cv::Point2f> points;
            points.push_back(cv::Point2f(obj.landmarks[0], obj.landmarks[1]));
            points.push_back(cv::Point2f(obj.landmarks[2], obj.landmarks[3]));
            points.push_back(cv::Point2f(obj.landmarks[4], obj.landmarks[5]));
            points.push_back(cv::Point2f(obj.landmarks[6], obj.landmarks[7]));
            
            // 计算边界框
            float min_x = std::min({points[0].x, points[1].x, points[2].x, points[3].x});
            float max_x = std::max({points[0].x, points[1].x, points[2].x, points[3].x});
            float min_y = std::min({points[0].y, points[1].y, points[2].y, points[3].y});
            float max_y = std::max({points[0].y, points[1].y, points[2].y, points[3].y});
            
            obj.rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
            obj.center = cv::Point2f((min_x + max_x) / 2, (min_y + max_y) / 2);
            
            proposals.push_back(obj);
        }
    }
}

void OpenvinoInfer::nmsMergeBoxes(std::vector<Armor>& proposals, std::vector<Armor>& objects) {
    if (proposals.empty()) {
        return;
    }
    
    // 按置信度排序
    std::sort(proposals.begin(), proposals.end(), 
              [](const Armor& a, const Armor& b) { return a.prob > b.prob; });
    
    std::vector<int> picked;
    std::vector<bool> suppressed(proposals.size(), false);
    std::map<int, std::vector<cv::Point2f>> merged_points;
    
    for (size_t i = 0; i < proposals.size(); i++) {
        if (suppressed[i]) continue;
        
        picked.push_back(i);
        std::vector<cv::Point2f> points_group;
        
        // 添加自身的关键点
        for (int j = 0; j < 4; j++) {
            points_group.push_back(cv::Point2f(proposals[i].landmarks[j*2], proposals[i].landmarks[j*2+1]));
        }
        
        merged_points[picked.size()-1] = points_group;
        
        for (size_t j = i + 1; j < proposals.size(); j++) {
            if (suppressed[j]) continue;
            
            // 计算IoU
            double iou = cal_iou(proposals[i].rect, proposals[j].rect);
            
            // 如果IoU大于阈值且类别、颜色相同，认为是同一个目标
            if (iou > NMS_THRESH && 
                proposals[i].number == proposals[j].number && 
                proposals[i].color == proposals[j].color &&
                std::abs(proposals[i].prob - proposals[j].prob) < MERGE_CONF_ERROR) {
                
                // 合并关键点
                for (int k = 0; k < 4; k++) {
                    merged_points[picked.size()-1].push_back(
                        cv::Point2f(proposals[j].landmarks[k*2], proposals[j].landmarks[k*2+1]));
                }
                
                suppressed[j] = true;
            }
        }
    }
    
    // 处理合并后的结果
    objects.resize(picked.size());
    for (size_t i = 0; i < picked.size(); i++) {
        objects[i] = proposals[picked[i]];
        
        // 如果有多组关键点，进行平均
        if (merged_points[i].size() > 4) {
            // 对每个角点位置单独平均
            std::vector<cv::Point2f> avg_points(4, cv::Point2f(0, 0));
            int point_count = merged_points[i].size() / 4;
            
            for (size_t j = 0; j < merged_points[i].size(); j++) {
                avg_points[j % 4] += merged_points[i][j];
            }
            
            for (int j = 0; j < 4; j++) {
                avg_points[j] *= (1.0f / point_count);
                objects[i].landmarks[j*2] = avg_points[j].x;
                objects[i].landmarks[j*2+1] = avg_points[j].y;
            }
            
            // 更新装甲板属性
            regularizeKeypoints(objects[i]);
        }
        
        // 应用几何约束
        if (!validateGeometry(objects[i])) {
            regularizeKeypoints(objects[i]);
        }
    }
}

bool OpenvinoInfer::validateGeometry(const Armor& armor) {
    // 提取四个角点
    cv::Point2f corners[4];
    for (int i = 0; i < 4; i++) {
        corners[i] = cv::Point2f(armor.landmarks[i*2], armor.landmarks[i*2+1]);
    }
    
    // 检查长宽比
    const float min_ratio = 0.3f;
    const float max_ratio = 4.0f;
    if (armor.ratio < min_ratio || armor.ratio > max_ratio) {
        return false;
    }
    
    // 检查是否接近平行四边形
    cv::Point2f d1 = corners[0] - corners[2];
    cv::Point2f d2 = corners[1] - corners[3];
    float diag_ratio = std::min(cv::norm(d1), cv::norm(d2)) / std::max(cv::norm(d1), cv::norm(d2));
    if (diag_ratio < 0.6f) {
        return false;
    }
    
    // 计算装甲板面积
    float area = std::abs((corners[0].x-corners[2].x)*(corners[0].y-corners[2].y) + 
                         (corners[1].x-corners[3].x)*(corners[1].y-corners[3].y)) / 2.0f;
    
    // 最小面积约束
    const float min_area = 20.0f;
    if (area < min_area) {
        return false;
    }
    
    return true;
}

void OpenvinoInfer::regularizeKeypoints(Armor& armor) {
    // 提取四个角点
    cv::Point2f corners[4];
    for (int i = 0; i < 4; i++) {
        corners[i] = cv::Point2f(armor.landmarks[i*2], armor.landmarks[i*2+1]);
    }
    
    // 计算中心点
    cv::Point2f center(0, 0);
    for (int i = 0; i < 4; i++) {
        center += corners[i];
    }
    center *= 0.25f;
    
    // 计算主方向（PCA简化版）
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    for (int i = 0; i < 4; i++) {
        float dx = corners[i].x - center.x;
        float dy = corners[i].y - center.y;
        sum_x += dx;
        sum_y += dy;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    // 计算协方差矩阵的特征向量
    float cov_xx = sum_x2 / 4.0f - (sum_x / 4.0f) * (sum_x / 4.0f);
    float cov_yy = sum_y2 / 4.0f - (sum_y / 4.0f) * (sum_y / 4.0f);
    float cov_xy = sum_xy / 4.0f - (sum_x / 4.0f) * (sum_y / 4.0f);
    
    float trace = cov_xx + cov_yy;
    float det = cov_xx * cov_yy - cov_xy * cov_xy;
    float lambda1 = trace / 2.0f + std::sqrt(trace * trace / 4.0f - det);
    
    // 计算主方向向量
    cv::Point2f dir;
    if (std::abs(cov_xy) > 1e-6f) {
        dir = cv::Point2f(lambda1 - cov_yy, cov_xy);
    } else {
        dir = cv::Point2f(1, 0);  // 退化情况，使用默认方向
    }
    float norm = cv::norm(dir);
    if (norm > 1e-6f) {
        dir /= norm;
    }
    
    // 正交方向
    cv::Point2f ortho(-dir.y, dir.x);
    
    // 计算在两个主方向上的投影
    float proj_main[4] = {0}, proj_ortho[4] = {0};
    for (int i = 0; i < 4; i++) {
        cv::Point2f vec = corners[i] - center;
        proj_main[i] = vec.dot(dir);
        proj_ortho[i] = vec.dot(ortho);
    }
    
    // 找出投影的最大和最小值
    float min_main = *std::min_element(proj_main, proj_main + 4);
    float max_main = *std::max_element(proj_main, proj_main + 4);
    float min_ortho = *std::min_element(proj_ortho, proj_ortho + 4);
    float max_ortho = *std::max_element(proj_ortho, proj_ortho + 4);
    
    // 计算规整化后的宽和高
    float width = std::max(max_main - min_main, 1.0f);
    float height = std::max(max_ortho - min_ortho, 1.0f);
    
    // 使用装甲板的真实宽高比例来调整
    float target_ratio = (armor.type == ArmorType::SMALL) ? 0.7f : 0.4f;
    float current_ratio = height / width;
    
    if (std::abs(current_ratio - target_ratio) > 0.2f) {
        // 调整宽高使比例接近目标
        if (current_ratio > target_ratio) {
            // 过高，增加宽度
            width = height / target_ratio;
        } else {
            // 过宽，增加高度
            height = width * target_ratio;
        }
    }
    
    // 构建规则装甲板的四个角点
    cv::Point2f new_corners[4];
    new_corners[0] = center + dir * (-width/2) + ortho * (-height/2);  // 左上
    new_corners[1] = center + dir * (-width/2) + ortho * (height/2);   // 左下
    new_corners[2] = center + dir * (width/2) + ortho * (height/2);    // 右下
    new_corners[3] = center + dir * (width/2) + ortho * (-height/2);   // 右上
    
    // 更新关键点坐标
    for (int i = 0; i < 4; i++) {
        armor.landmarks[i*2] = new_corners[i].x;
        armor.landmarks[i*2+1] = new_corners[i].y;
    }
    
    // 更新装甲板属性
    armor.length = cv::norm(new_corners[0] - new_corners[3]);
    armor.width = cv::norm(new_corners[0] - new_corners[1]);
    armor.ratio = armor.length / armor.width;
    armor.center = center;
    
    // 更新边界框
    float min_x = std::min({new_corners[0].x, new_corners[1].x, new_corners[2].x, new_corners[3].x});
    float max_x = std::max({new_corners[0].x, new_corners[1].x, new_corners[2].x, new_corners[3].x});
    float min_y = std::min({new_corners[0].y, new_corners[1].y, new_corners[2].y, new_corners[3].y});
    float max_y = std::max({new_corners[0].y, new_corners[1].y, new_corners[2].y, new_corners[3].y});
    
    armor.rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

std::vector<Armor> OpenvinoInfer::infer(cv::Mat &src, int detect_color) {
    objects.clear();
    tmp_objects.clear();
    ious.clear();

    // 图像预处理
    raw_size = cv::Size(src.cols, src.rows);
    cv::Mat pr_img = scaledResize(src);
    
    // 准备输入张量数据
    cv::Mat pre;
    cv::Mat pre_split[3];
    pr_img.convertTo(pre, CV_32F);
    cv::split(pre, pre_split);
    
    // 获取输入张量
    input_tensor = infer_request.get_input_tensor(0);
    infer_request.set_input_tensor(input_tensor);
    
    float* tensor_data = input_tensor.data<float_t>();
    auto img_offset = INPUT_H * INPUT_W;
    
    // 复制图像数据到张量
    for(int c = 0; c < 3; c++) {
        memcpy(tensor_data, pre_split[c].data, INPUT_H * INPUT_W * sizeof(float));
        tensor_data += img_offset;
    }
    
    // 执行推理
    infer_request.infer();
    
    // 处理推理结果
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    float* output = output_tensor.data<float_t>();
    
    // 生成网格和步长
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    
    // 生成网格
    for (auto stride : strides) {
        int num_grid_w = INPUT_W / stride;
        int num_grid_h = INPUT_H / stride;
        
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                grid_strides.emplace_back(GridAndStride{g0, g1, stride});
            }
        }
    }
    
        // 生成候选装甲板
    std::vector<Armor> proposals;
    generateProposals(grid_strides, output, BBOX_CONF_THRESH, proposals);
    
    // 应用NMS合并类似检测
    nmsMergeBoxes(proposals, tmp_objects);
    
    return tmp_objects;
}

void OpenvinoInfer::drawResults(cv::Mat &img) {
    for(const auto &tmp_armor : tmp_objects) {
        // 绘制四个角点
        for(int i = 0; i < 4; i++) {
            cv::circle(img, cv::Point2f(tmp_armor.landmarks[i*2], tmp_armor.landmarks[i*2+1]), 3,
                       cv::Scalar(255, 255, 255), 1);
            
            // 绘制边缘线
            cv::line(img, 
                    cv::Point2f(tmp_armor.landmarks[i*2], tmp_armor.landmarks[i*2+1]), 
                    cv::Point2f(tmp_armor.landmarks[(i+1)%4*2], tmp_armor.landmarks[(i+1)%4*2+1]), 
                    cv::Scalar(0, 255, 0), 2);
        }
        
        // 绘制装甲板信息
        cv::putText(
            img, tmp_armor.classfication_result, 
            cv::Point2f(tmp_armor.landmarks[0], tmp_armor.landmarks[1]),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    }
    return;
}

} // namespace rm_auto_aim