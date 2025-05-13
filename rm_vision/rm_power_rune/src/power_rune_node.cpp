#include <thread>

#include "PowerRune.h"
#include "power_rune_node.hpp"

namespace power_rune{


    PowerRuneNode::PowerRuneNode(const rclcpp::NodeOptions & options)
    : Node("power_rune_node", options),
      last_save_time_(this->now())
    {
        

        RCLCPP_INFO(this->get_logger(),"Starting PowerRuneNode !");

        // Power_Rune
        power_rune_ = std::make_unique<PowerRune>();

        cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
        //cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
        //cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        //pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
        
        Param::IMAGE_WIDTH = camera_info -> width;

        RCLCPP_INFO(this->get_logger(), "Param:   width:%d   height:%d",camera_info -> width,camera_info -> height );

        Param::IMAGE_HEIGHT = camera_info -> height;

        buff_pub_ = this->create_publisher<global_interface::msg::Buff>("buff_msg", rclcpp::SensorDataQoS());

        //在power_rune/config.yaml中更正tvec_c2g
        /*
        for(int i=0;i<9;i++) Param::INTRINSIC_MATRIX.at<double>(i/3,i%3)=camera_info->k[i];
        for(int i=0;i<5;i++) Param::DIST_COEFFS.at<double>(i)=camera_info->d[i];
        //Param::INTRINSIC_MATRIX = camera_info -> k;
        //哈工大为opencv格式，君为ros格式，此处需要改正
        //Param::DIST_COEFFS = camera_info -> d;
        */
        cam_info_sub_.reset();
        });

        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(rclcpp::KeepLast(1)),
            std::bind(&PowerRuneNode::imageCallback, this, std::placeholders::_1));

           task_sub_ = this->create_subscription<std_msgs::msg::Int64>(
        "/task_mode", rclcpp::SensorDataQoS(),
            std::bind(&PowerRuneNode::task_callback, this, std::placeholders::_1));

        serial_sub_ = this->create_subscription<global_interface::msg::Serial>(
            "/serial_msg", rclcpp::SensorDataQoS(),
            std::bind(&PowerRuneNode::serial_callback, this, std::placeholders::_1));

        #if SHOW_IMAGE>=1
        image_show_pub_=this->create_publisher<sensor_msgs::msg::Image>("img_show",10);
        image_armor_pub_=this->create_publisher<sensor_msgs::msg::Image>("img_armor",10);
        image_arrow_pub_=this->create_publisher<sensor_msgs::msg::Image>("img_arrow",10);
        // image_src_pub_=this->create_publisher<sensor_msgs::msg::Image>("img_src",10);
        
        debug_img_timer_=this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&PowerRuneNode::publish_debug_img,this)
        );
        #endif

        last_save_time_ = this->now();


    }

    #if SHOW_IMAGE>=1
    void PowerRuneNode::publish_debug_img(){
        // auto st = std::chrono::steady_clock::now();
        auto img_show_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", power_rune_->get_img_show()).toImageMsg();
        image_show_pub_->publish(*img_show_msg);
        auto img_arrow_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", power_rune_->get_img_arrow()).toImageMsg();
        image_arrow_pub_->publish(*img_arrow_msg);
        auto img_armor_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", power_rune_->get_img_armor()).toImageMsg();
        image_armor_pub_->publish(*img_armor_msg);
        // auto img_src_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", power_rune_->get_img_src()).toImageMsg();
        // image_src_pub_->publish(*img_src_msg);
        //         auto end = std::chrono::steady_clock::now();
        // double infer_dt = std::chrono::duration<double,std::milli>(end - st).count();
        // std::cout << "infer_time:" << infer_dt << std::endl;
    }
    #endif
    
    void PowerRuneNode::task_callback(const std_msgs::msg::Int64::ConstSharedPtr task_msg)
    {
        rune_task_mode=task_msg->data;
    }   

   
    void PowerRuneNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
    {   
        if(rune_task_mode==0){
            return;
        }   
        
        if(rune_task_mode==1||rune_task_mode==2) Param::COLOR=Color::RED;
        else Param::COLOR=Color::BLUE;

        if(rune_task_mode==1||rune_task_mode==3) Param::MODE=Mode::SMALL;
        else Param::MODE=Mode::BIG;
                

        auto img = cv_bridge::toCvShare(img_msg, "rgb8")->image;
        cv::Mat imgbgr=img.clone();
        auto start{std::chrono::steady_clock::now()};
        cv::cvtColor(imgbgr,imgbgr,cv::COLOR_RGB2BGR);
        
        //std::cout<<"image width:"<<imgbgr.cols<<"   image height:"<<imgbgr.rows<<std::endl;

        if(power_rune_->runOnce(imgbgr, pitch_, yaw_, 0.0)==false){

            if(power_rune_->m_detector.m_status!=power_rune::Status::SUCCESS){
                
                std::cout<<fail_count<<"       "<<power_rune_->m_detector.centerR2low<<std::endl;
                fail_count++;
                if(fail_count>=10&&power_rune_->m_detector.centerR2low>=10){
                    
                    fail_count=0;
                    power_rune_->m_detector.centerR2low=0;
                    buff_msg.distance = 0;
                    buff_msg.angletofirst = 0;
                    buff_msg.angletolast = 0;
                    buff_msg.angle = 0;
                    buff_msg.predict_pitch = pitch_ - 5;
                    buff_msg.predict_yaw = yaw_;
                    
                    buff_pub_->publish(buff_msg);
                }
            }
            return ;
        }

        fail_count=0;
        std::pair<double, double> pitch_yaw=power_rune_->m_calculator.getPredictPitchYaw();
        if(Param::MODE==Mode::BIG) buff_msg.mode=1;
        else buff_msg.mode=0;
        buff_msg.distance = power_rune_->m_calculator.getDistance();
        buff_msg.angletofirst = power_rune_->m_calculator.getAngleRel();
        buff_msg.angletolast = power_rune_->m_calculator.getAngleLast();
        buff_msg.angle=buff_msg.angletofirst-buff_msg.angletolast;
        buff_msg.predict_pitch = pitch_yaw.first;
        buff_msg.predict_yaw = pitch_yaw.second;
        //buff_msg.mode = Param::MODE;
        //buff_msg.buff_angle = power_rune_->m_calculator.getBuffAngle();
        //buff_msg.predict_point3d_cam=power_rune_->m_calculator.getPredictRobot();

        buff_pub_->publish(buff_msg);

        auto future_time = start + std::chrono::milliseconds(1000 / power_rune::Param::FPS);
        if (std::chrono::steady_clock::now() < future_time) {
            std::this_thread::sleep_until(future_time);
        }    
        //
        
        // 检查是否达到保存间隔
        rclcpp::Time current_time = this->now();


        /*
        if ((current_time - last_save_time_).seconds() >= SAVE_INTERVAL) {
            static int frame_count = 0;
            cv::Mat install = power_rune_->get_img_src().clone();
            if (!install.empty()) {
                //cv::cvtColor(install, install, cv::COLOR_BGR2RGB);
                std::string filename = "/home/tars-go/Documents/src1/frame_" + std::to_string(frame_count++) + ".png";
                if (!cv::imwrite(filename, install)) {
                    RCLCPP_WARN(this->get_logger(), "Failed to save image: %s", filename.c_str());
                }
                last_save_time_ = current_time;  // 更新上次保存时间
            }
        }
        */


        
    }

    void PowerRuneNode::serial_callback(const global_interface::msg::Serial::ConstSharedPtr serial_msg)
    {
        // 获取四元数
        double x = serial_msg->imu.orientation.x;
        double y = serial_msg->imu.orientation.y;
        double z = serial_msg->imu.orientation.z;
        double w = serial_msg->imu.orientation.w;

        // 四元数转欧拉角
        // roll (x-axis rotation)
        double sinr_cosp = 2.0 * (w * x + y * z);
        double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        roll_ = std::atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = 2.0 * (w * y - z * x);
        if (std::abs(sinp) >= 1)
            pitch_ = std::copysign(M_PI / 2, sinp); // 使用90度，如果超出范围
        else
            pitch_ = std::asin(sinp);

        pitch_=-pitch_;

        // yaw (z-axis rotation)
        double siny_cosp = 2.0 * (w * z + x * y);
        double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        yaw_ = std::atan2(siny_cosp, cosy_cosp);

        // 将弧度转换为角度
        roll_ = roll_ * 180.0 / M_PI;
        pitch_ = pitch_ * 180.0 / M_PI;
        yaw_ = yaw_ * 180.0 / M_PI;
    }
}

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(power_rune::PowerRuneNode)