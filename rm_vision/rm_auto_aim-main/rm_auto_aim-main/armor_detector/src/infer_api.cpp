// Copyright 2023 RM Vision Team
// Licensed under the MIT License.

#include "../include/armor_detector/infer_api.hpp"
#include <algorithm>
#include <chrono>
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

bool OpenvinoInfer::initModel(std::string model_path_xml, std::string model_path_bin) {
    std::cout << "Start initialize model..." << std::endl;

    // 性能优化配置
    core.set_property("CPU", ov::enable_profiling(false)); // 禁用性能分析以加快速度
    core.set_property("CPU", ov::inference_num_threads(perf_config.inference_threads));
    // 注意：移除了不兼容的 enable_cpu_memory_pool API

    // 读取模型
    model = core.read_model(model_path_xml, model_path_bin);

    // 预处理配置
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::f32);
    ppp.output().tensor().set_element_type(ov::element::f32);
    
    // 构建模型
    ppp.build(); 

    // 编译模型
    ov::hint::PerformanceMode perf_mode = ov::hint::PerformanceMode::LATENCY;
    compiled_model = core.compile_model(
        model,
        "CPU",
        ov::hint::performance_mode(perf_mode)
    );

   
// 在初始化函数中
infer_requests_[0] = compiled_model.create_infer_request();
infer_requests_[1] = compiled_model.create_infer_request();
    
    
    // 预分配缓冲区
    points_buffer.reserve(50);  // 预分配足够的空间
    
    return true;
}

cv::Mat OpenvinoInfer::scaledResize(cv::Mat& img, float& scale_factor) {
    // 计算缩放比例
    scale_factor = std::min(INPUT_W / (img.cols * 1.0f), INPUT_H / (img.rows * 1.0f));
    int unpad_w = static_cast<int>(img.cols * scale_factor);
    int unpad_h = static_cast<int>(img.rows * scale_factor);
    
    // 快速缩放并保持宽高比
    cv::Mat re;
    cv::resize(img, re, cv::Size(unpad_w, unpad_h), 0, 0, cv::INTER_LINEAR);
    
    // 计算填充尺寸
    int dw = INPUT_W - unpad_w;
    int dh = INPUT_H - unpad_h;
    dw /= 2;
    dh /= 2;
    
    // 构建变换矩阵 (用于后续关键点映射)
    transform_matrix = Eigen::Matrix<float, 3, 3>::Identity();
    transform_matrix(0, 0) = 1.0f / scale_factor;
    transform_matrix(1, 1) = 1.0f / scale_factor;
    transform_matrix(0, 2) = -dw / scale_factor;
    transform_matrix(1, 2) = -dh / scale_factor;
    
    // 边缘填充
    cv::Mat out;
    cv::copyMakeBorder(re, out, dh, dh, dw, dw, cv::BORDER_CONSTANT);
    
    return out;
}

void OpenvinoInfer::generateGridsAndStride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides) {
    grid_strides.clear();
    grid_strides.reserve(INPUT_W * INPUT_H / 64); // 预分配空间
    
    for (auto stride : strides) {
        int num_grid_w = INPUT_W / stride;
        int num_grid_h = INPUT_H / stride;
        
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                grid_strides.emplace_back(GridAndStride{g0, g1, stride});
            }
        }
    }
}

void OpenvinoInfer::generateProposals(const std::vector<GridAndStride>& grid_strides, 
                                     const float* feat_ptr, float prob_threshold, 
                                     std::vector<Armor>& proposals) {
    proposals.clear();
    proposals.reserve(100);  // 预分配容量避免频繁重新分配
    
    const int num_anchors = grid_strides.size();

    // 串行处理所有锚点 (移除了OpenMP并行)
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;
        const int basic_pos = anchor_idx * (9 + NUM_COLORS + NUM_CLASSES);

        // 检查置信度
        float box_objectness = feat_ptr[basic_pos + 8];
        if (box_objectness < prob_threshold) {
            continue; // 快速剪枝
        }
        
        int box_color = argmax(feat_ptr + basic_pos + 9, NUM_COLORS);
        int box_class = argmax(feat_ptr + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);

        // 过滤颜色逻辑保持不变
        if (box_color == PURPLE_SMALL || box_color == PURPLE_BIG) {
            continue;
        }
        
        // 创建对象
        Armor obj;
        
        obj.prob = box_objectness;
        
        // 设置颜色和数字
        if(box_color == RED_SMALL || box_color == RED_BIG) obj.color = RED;
        if(box_color == BLUE_SMALL || box_color == BLUE_BIG) obj.color = BLUE;
        
        if(box_class==0) obj.number="G";
        if(box_class==1) obj.number="1";
        if(box_class==2) obj.number="2";
        if(box_class==3) obj.number="3";
        if(box_class==4) obj.number="4";
        if(box_class==5) obj.number="5";
        if(box_class==7) obj.number="7";
        
        // 提取关键点
        float x_1 = (feat_ptr[basic_pos + 0] + grid0) * stride;
        float y_1 = (feat_ptr[basic_pos + 1] + grid1) * stride;
        float x_2 = (feat_ptr[basic_pos + 2] + grid0) * stride;
        float y_2 = (feat_ptr[basic_pos + 3] + grid1) * stride;
        float x_3 = (feat_ptr[basic_pos + 4] + grid0) * stride;
        float y_3 = (feat_ptr[basic_pos + 5] + grid1) * stride;
        float x_4 = (feat_ptr[basic_pos + 6] + grid0) * stride;
        float y_4 = (feat_ptr[basic_pos + 7] + grid1) * stride;
        
        // 优化的矩阵变换
        Eigen::Matrix<float, 3, 4> apex_norm;
        apex_norm << x_1, x_2, x_3, x_4,
                     y_1, y_2, y_3, y_4,
                     1,   1,   1,   1;
        
        Eigen::Matrix<float, 3, 4> apex_dst = transform_matrix * apex_norm;
        
        // 提取变换后的坐标
        obj.landmarks[0] = apex_dst(0, 0);
        obj.landmarks[1] = apex_dst(1, 0);
        obj.landmarks[2] = apex_dst(0, 1);
        obj.landmarks[3] = apex_dst(1, 1);
        obj.landmarks[4] = apex_dst(0, 2);
        obj.landmarks[5] = apex_dst(1, 2);
        obj.landmarks[6] = apex_dst(0, 3);
        obj.landmarks[7] = apex_dst(1, 3);
        
        // 计算装甲板属性
        obj.length = std::hypot(obj.landmarks[0] - obj.landmarks[6], 
                              obj.landmarks[1] - obj.landmarks[7]);
        obj.width = std::hypot(obj.landmarks[0] - obj.landmarks[2], 
                             obj.landmarks[1] - obj.landmarks[3]);
        
        // 避免除以零
        if (obj.width > 1e-5f) {
            obj.ratio = obj.length / obj.width;
        } else {
            obj.ratio = 0.0f;
            continue; // 跳过无效装甲板
        }
        
        obj.classfication_result = obj.number + ":" + std::to_string(static_cast<int>(obj.prob * 100.0f));
        
        // 根据比例确定装甲板类型
        obj.type = (obj.ratio > 0.6f) ? ArmorType::SMALL : ArmorType::LARGE;
        
        // 计算边界框和中心点
        float min_x = std::min({obj.landmarks[0], obj.landmarks[2], obj.landmarks[4], obj.landmarks[6]});
        float max_x = std::max({obj.landmarks[0], obj.landmarks[2], obj.landmarks[4], obj.landmarks[6]});
        float min_y = std::min({obj.landmarks[1], obj.landmarks[3], obj.landmarks[5], obj.landmarks[7]});
        float max_y = std::max({obj.landmarks[1], obj.landmarks[3], obj.landmarks[5], obj.landmarks[7]});
        
        obj.rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
        obj.center = cv::Point2f((min_x + max_x) * 0.5f, (min_y + max_y) * 0.5f);
        
        proposals.push_back(obj);
    }
}

bool OpenvinoInfer::needsRegularization(const Armor& armor) {
    // 快速检查是否需要规整化
    const float min_ratio = 0.3f;
    const float max_ratio = 4.0f;
    
    // 比例检查
    if (armor.ratio < min_ratio || armor.ratio > max_ratio) {
        return true;
    }
    
    // 面积检查
    if (armor.rect.area() < 20.0f) {
        return true;
    }
    
    // 对角线检查
    cv::Point2f d1(armor.landmarks[0] - armor.landmarks[4], 
                   armor.landmarks[1] - armor.landmarks[5]);
    cv::Point2f d2(armor.landmarks[2] - armor.landmarks[6], 
                   armor.landmarks[3] - armor.landmarks[7]);
                   
    float diag1 = std::hypot(d1.x, d1.y);
    float diag2 = std::hypot(d2.x, d2.y);
    
    // 如果对角线比例不平衡，需要规整化
    float diag_ratio = std::min(diag1, diag2) / std::max(diag1, diag2);
    return diag_ratio < 0.6f;
}

// 优化的关键点规整化函数
void OpenvinoInfer::fastRegularizeKeypoints(Armor& armor) {
    // 提取四个角点
    cv::Point2f corners[4];
    for (int i = 0; i < 4; i++) {
        corners[i] = cv::Point2f(armor.landmarks[i*2], armor.landmarks[i*2+1]);
    }
    
    // 计算中心点 - 使用SIMD友好的计算方式
    cv::Point2f center(0, 0);
    for (int i = 0; i < 4; i++) {
        center.x += corners[i].x;
        center.y += corners[i].y;
    }
    center.x *= 0.25f;
    center.y *= 0.25f;
    
    // 简化的主方向计算
    // 使用对角线作为主轴
    cv::Point2f mainAxis = corners[2] - corners[0];
    float mainAxisLength = std::hypot(mainAxis.x, mainAxis.y);
    
    if (mainAxisLength > 1e-5f) {
        // 归一化主轴
        mainAxis.x /= mainAxisLength;
        mainAxis.y /= mainAxisLength;
    } else {
        // 退化情况
        mainAxis = cv::Point2f(1.0f, 0.0f);
    }
    
    // 正交轴
    cv::Point2f orthoAxis(-mainAxis.y, mainAxis.x);
    
    // 根据装甲板类型确定长宽比
    float targetRatio = (armor.type == ArmorType::SMALL) ? 1.5f : 2.5f;
    
    // 使用类型信息确定尺寸
    float halfWidth, halfHeight;
    
    if (armor.type == ArmorType::SMALL) {
        halfWidth = armor.length * 0.25f;
        halfHeight = halfWidth * targetRatio;
    } else {
        halfWidth = armor.length * 0.4f;
        halfHeight = halfWidth * targetRatio;
    }
    
    // 构建规则装甲板的四个角点
    cv::Point2f new_corners[4];
    new_corners[0] = center + mainAxis * (-halfWidth) + orthoAxis * (-halfHeight);  // 左上
    new_corners[1] = center + mainAxis * (-halfWidth) + orthoAxis * (halfHeight);   // 左下
    new_corners[2] = center + mainAxis * (halfWidth) + orthoAxis * (halfHeight);    // 右下
    new_corners[3] = center + mainAxis * (halfWidth) + orthoAxis * (-halfHeight);   // 右上
    
    // 更新关键点坐标
    for (int i = 0; i < 4; i++) {
        armor.landmarks[i*2] = new_corners[i].x;
        armor.landmarks[i*2+1] = new_corners[i].y;
    }
    
    // 更新装甲板属性
    armor.length = 2.0f * halfWidth;
    armor.width = 2.0f * halfHeight;
    armor.ratio = armor.length / armor.width;
    armor.center = center;
    
    // 更新边界框
    float min_x = std::min({new_corners[0].x, new_corners[1].x, new_corners[2].x, new_corners[3].x});
    float max_x = std::max({new_corners[0].x, new_corners[1].x, new_corners[2].x, new_corners[3].x});
    float min_y = std::min({new_corners[0].y, new_corners[1].y, new_corners[2].y, new_corners[3].y});
    float max_y = std::max({new_corners[0].y, new_corners[1].y, new_corners[2].y, new_corners[3].y});
    
    armor.rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

// 原有的精确规整化函数，仅作参考，使用时优先使用fastRegularizeKeypoints
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
            width = height / target_ratio;
        } else {
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
    armor.length = width;
    armor.width = height;
    armor.ratio = armor.length / armor.width;
    armor.center = center;
    
    // 更新边界框
    float min_x = std::min({new_corners[0].x, new_corners[1].x, new_corners[2].x, new_corners[3].x});
    float max_x = std::max({new_corners[0].x, new_corners[1].x, new_corners[2].x, new_corners[3].x});
    float min_y = std::min({new_corners[0].y, new_corners[1].y, new_corners[2].y, new_corners[3].y});
    float max_y = std::max({new_corners[0].y, new_corners[1].y, new_corners[2].y, new_corners[3].y});
    
    armor.rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

void OpenvinoInfer::nmsMergeBoxes(std::vector<Armor>& proposals, std::vector<Armor>& output_objects) {
    if (proposals.empty()) {
        output_objects.clear();
        return;
    }
    
    // 按置信度排序
    std::sort(proposals.begin(), proposals.end(), 
              [](const Armor& a, const Armor& b) { return a.prob > b.prob; });
    
    std::vector<bool> suppressed(proposals.size(), false);
    output_objects.clear();
    output_objects.reserve(proposals.size() / 2);  // 预分配空间
    
    // 创建存储合并点的数据结构
    std::vector<std::vector<cv::Point2f>> merged_groups;
    merged_groups.reserve(proposals.size() / 2);
    
    // NMS处理
    for (size_t i = 0; i < proposals.size(); i++) {
        if (suppressed[i]) continue;
        
        // 创建新的合并组
        std::vector<cv::Point2f> points_group;
        points_group.reserve(16);  // 预计最多4个物体合并，每个4个点
        
        // 添加当前物体的关键点
        for (int j = 0; j < 4; j++) {
            points_group.push_back(cv::Point2f(proposals[i].landmarks[j*2], 
                                              proposals[i].landmarks[j*2+1]));
        }
        
        // 添加到合并组 - 去掉未使用的变量 group_idx
        merged_groups.push_back(points_group);
        
        // 添加当前物体到输出
        output_objects.push_back(proposals[i]);
        
        // 检查其他物体是否应该合并到当前组
        for (size_t j = i + 1; j < proposals.size(); j++) {
            if (suppressed[j]) continue;
            
            // 计算IoU
            double iou = cal_iou(proposals[i].rect, proposals[j].rect);
            
            // 如果IoU大于阈值且类别、颜色相同，认为是同一个目标
            if (iou > NMS_THRESH && 
                proposals[i].number == proposals[j].number && 
                proposals[i].color == proposals[j].color &&
                std::abs(proposals[i].prob - proposals[j].prob) < MERGE_CONF_ERROR) {
                
                // 合并关键点到当前组的最后一个组
                for (int k = 0; k < 4; k++) {
                    merged_groups.back().push_back(cv::Point2f(proposals[j].landmarks[k*2],
                                                     proposals[j].landmarks[k*2+1]));
                }
                
                suppressed[j] = true;
            }
        }
    }
    
    // 处理合并后的关键点 - 串行处理，移除OpenMP
    const size_t num_objects = output_objects.size();
    
    for (size_t i = 0; i < num_objects; i++) {
        // 只有当有多个检测时才进行平均
        if (merged_groups[i].size() > 4) {
            const size_t num_points = merged_groups[i].size();
            const size_t num_detections = num_points / 4;
            std::array<cv::Point2f, 4> avg_points = {cv::Point2f(0,0), cv::Point2f(0,0), 
                                                    cv::Point2f(0,0), cv::Point2f(0,0)};
            
            // 按照角点位置分组平均
            for (size_t j = 0; j < num_points; j++) {
                avg_points[j % 4] += merged_groups[i][j];
            }
            
            // 计算平均位置
            for (size_t j = 0; j < 4; j++) {
                avg_points[j] *= (1.0f / num_detections);
                output_objects[i].landmarks[j*2] = avg_points[j].x;
                output_objects[i].landmarks[j*2+1] = avg_points[j].y;
            }
            
            // 更新装甲板属性
            output_objects[i].length = std::hypot(avg_points[0].x - avg_points[3].x, 
                                                avg_points[0].y - avg_points[3].y);
            output_objects[i].width = std::hypot(avg_points[0].x - avg_points[1].x, 
                                               avg_points[0].y - avg_points[1].y);
            
            if (output_objects[i].width > 1e-5f) {
                output_objects[i].ratio = output_objects[i].length / output_objects[i].width;
            }
            
            // 更新中心点和边界框
            output_objects[i].center = (avg_points[0] + avg_points[1] + 
                                      avg_points[2] + avg_points[3]) * 0.25f;
            
            float min_x = std::min({avg_points[0].x, avg_points[1].x, 
                                   avg_points[2].x, avg_points[3].x});
            float max_x = std::max({avg_points[0].x, avg_points[1].x, 
                                   avg_points[2].x, avg_points[3].x});
            float min_y = std::min({avg_points[0].y, avg_points[1].y, 
                                   avg_points[2].y, avg_points[3].y});
            float max_y = std::max({avg_points[0].y, avg_points[1].y, 
                                   avg_points[2].y, avg_points[3].y});
            
            output_objects[i].rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
        }
        
        // 判断是否需要规整化，并应用合适的规整化
        if (needsRegularization(output_objects[i])) {
            if (perf_config.enable_fast_keypoint_regularization) {
                fastRegularizeKeypoints(output_objects[i]);
            } else {
                regularizeKeypoints(output_objects[i]);
            }
        }
    }
}

std::vector<Armor> OpenvinoInfer::infer(cv::Mat &src, int detect_color) {
    objects.clear();
    tmp_objects.clear();
    ious.clear();
    
    // 图像预处理
    raw_size = cv::Size(src.cols, src.rows);
    float scale_factor;
    cv::Mat pr_img = scaledResize(src, scale_factor);
    
    // 准备输入张量数据
    cv::Mat pre;
    pr_img.convertTo(pre, CV_32F);
    cv::split(pre, pre_split);
    
    if (perf_config.use_async_inference) {
        // 获取当前请求
        auto& curr_request = infer_requests_[current_request_idx_];
        
        // 准备当前请求输入数据
        input_tensor = curr_request.get_input_tensor(0);
        curr_request.set_input_tensor(input_tensor);
        
        float* tensor_data = input_tensor.data<float_t>();
        auto img_offset = INPUT_H * INPUT_W;
        
        // 复制图像数据到张量
        for(int c = 0; c < 3; c++) {
            memcpy(tensor_data, pre_split[c].data, INPUT_H * INPUT_W * sizeof(float));
            tensor_data += img_offset;
        }
        
        // 启动当前推理
        curr_request.start_async();
        
        // 处理上一次的结果
        if (!is_first_inference) {
            int prev_idx = (current_request_idx_ + 1) % 2;
            auto& prev_request = infer_requests_[prev_idx];
            
            try {
                // 等待上一个推理完成
                prev_request.wait();
                
                // 获取结果
                ov::Tensor output_tensor = prev_request.get_output_tensor();
                float* output = output_tensor.data<float_t>();
                
                // 处理结果
                std::vector<Armor> proposals;
                proposals.reserve(100);
                
                std::vector<int> strides = {8, 16, 32};
                std::vector<GridAndStride> grid_strides;
                generateGridsAndStride(strides, grid_strides);
                
                generateProposals(grid_strides, output, BBOX_CONF_THRESH, proposals);
                nmsMergeBoxes(proposals, tmp_objects);
            } catch (const ov::Exception& e) {
                RCLCPP_WARN(rclcpp::get_logger("OpenvinoInfer"), 
                           "异步推理错误: %s", e.what());
            }
        } else {
            is_first_inference = false;
        }
        
        // 切换到下一个请求
        current_request_idx_ = (current_request_idx_ + 1) % 2;
        
        return tmp_objects;
    } else {
        // 同步推理方式
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
        
        // 生成候选装甲板
        std::vector<Armor> proposals;
        proposals.reserve(100);
        
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;
        generateGridsAndStride(strides, grid_strides);
        
        generateProposals(grid_strides, output, BBOX_CONF_THRESH, proposals);
        
        // 应用NMS合并类似检测
        nmsMergeBoxes(proposals, tmp_objects);
        
        return tmp_objects;
    }
}

void OpenvinoInfer::drawResults(cv::Mat &img) {
    for(const auto &tmp_armor : tmp_objects) {
        // 绘制四个角点
        for(int i = 0; i < 4; i++) {
            cv::circle(img, cv::Point2f(tmp_armor.landmarks[i*2], tmp_armor.landmarks[i*2+1]), 3,
                       cv::Scalar(255, 255, 255), 1);
            
            // 绘制边缘线
            int next_idx = (i + 1) % 4;
            cv::line(img, 
                    cv::Point2f(tmp_armor.landmarks[i*2], tmp_armor.landmarks[i*2+1]), 
                    cv::Point2f(tmp_armor.landmarks[next_idx*2], tmp_armor.landmarks[next_idx*2+1]), 
                    cv::Scalar(0, 255, 0), 2);
        }
        
        // 绘制装甲板信息
        cv::putText(
            img, tmp_armor.classfication_result, 
            cv::Point2f(tmp_armor.landmarks[0], tmp_armor.landmarks[1] - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    }
}

} // namespace rm_auto_aim