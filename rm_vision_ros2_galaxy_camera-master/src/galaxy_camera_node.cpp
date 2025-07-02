#include "GxIAPI.h"
#include "DxImageProc.h"
#include <condition_variable>
#include <mutex>
// ROS
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/parameter_value.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rclcpp/subscription.hpp>
#include <std_msgs/msg/int64.hpp>


#define GX_SUCCESS(X) (X == GX_STATUS_SUCCESS)

namespace galaxy_camera
{
class GalaxyCameraNode : public rclcpp::Node
{
public:
  explicit GalaxyCameraNode(const rclcpp::NodeOptions & options) : Node("galaxy_camera", options)
  {
    GX_STATUS status;
    RCLCPP_INFO(this->get_logger(), "Starting GalaxyCameraNode!");

    // Init lib
    do {
      status = GXInitLib();
      if (!GX_SUCCESS(status)) {
        RCLCPP_FATAL(this->get_logger(), "Init GxIAPI failed, code = %x!", status);
        exit(status);
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    } while (!GX_SUCCESS(status));

    // Constantly try opening one camera
    while (true) {
      uint32_t device_count = 0;
      status = GXUpdateDeviceList(&device_count, 100);    // 枚举所有设备
      if (device_count < 1) {
        RCLCPP_WARN(this->get_logger(), "No camera found. device_count = %d", device_count);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      status = GXOpenDeviceByIndex(1, &camera_handle_);     // 通过序号打开设备
      if (!GX_SUCCESS(status)) {
        RCLCPP_ERROR(this->get_logger(), "Can not open camera, status = %d", status);
      } else {
        break;
      }
    };

    // Get camera infomation
    GXGetInt(camera_handle_, GX_INT_WIDTH, &img_info_.nWidthValue);        // 图像宽度
    GXGetInt(camera_handle_, GX_INT_WIDTH_MAX, &img_info_.nWidthMax);      // 最大宽度
    GXGetInt(camera_handle_, GX_INT_HEIGHT, &img_info_.nHeightValue);
    GXGetInt(camera_handle_, GX_INT_HEIGHT_MAX, &img_info_.nHeightMax);
    // GXSendCommand(camera_handle_, GX_BOOL_REVERSE_Y);
    image_msg_.data.reserve(img_info_.nHeightMax * img_info_.nWidthMax * 3);    // 分配内存

    bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", true);    // 使用传感器数据的质量服务
    auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    //auto qos = rmw_qos_profile_default;
    camera_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);

    declareParameters();

    // GXSetBool(camera_handle_, GX_BOOL_REVERSE_Y, true);
    GXSendCommand(camera_handle_, GX_COMMAND_ACQUISITION_START);    // 发送控制命令


    // Load camera info
    camera_name_ = this->declare_parameter("camera_name", "narrow_stereo");
    camera_info_manager_ =
      std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    auto camera_info_url =
      this->declare_parameter("camera_info_url", "package://galaxy_camera/config/camera_info.yaml");
    if (camera_info_manager_->validateURL(camera_info_url)) {
      camera_info_manager_->loadCameraInfo(camera_info_url);
      camera_info_msg_ = camera_info_manager_->getCameraInfo();
    } else {
      RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s", camera_info_url.c_str());
    }

    params_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&GalaxyCameraNode::parametersCallback, this, std::placeholders::_1));

    
    exposure_sub_ = this->create_subscription<std_msgs::msg::Int64>(
      "/exposure_time",10,std::bind(&GalaxyCameraNode::exposureCallback,this,std::placeholders::_1));

    gain_sub_ = this->create_subscription<std_msgs::msg::Int64>(
      "/gain",10,std::bind(&GalaxyCameraNode::gainCallback,this,std::placeholders::_1));
 


      


    capture_thread_ = std::thread{[this]() -> void {
      GX_FRAME_DATA bayer_frame {};
      GX_STATUS status;
      std::vector<char> bayer_buffer_holder;

      // Initialize frame   初始化帧
      int64_t payloadSize;
      GXGetInt(camera_handle_, GX_INT_PAYLOAD_SIZE, &payloadSize);
      bayer_buffer_holder.reserve(payloadSize);
      bayer_frame.pImgBuf = bayer_buffer_holder.data();

      RCLCPP_INFO(this->get_logger(), "Publishing image!");

      image_msg_.header.frame_id = "camera_optical_frame";
      image_msg_.encoding = "rgb8";

      while (rclcpp::ok()) {
        if (fail_conut_ > 5) {
          RCLCPP_FATAL(this->get_logger(), "Retry camera!");
          // rclcpp::shutdown();
          while (true) {
            uint32_t device_count = 0;
            status = GXUpdateDeviceList(&device_count, 100);    // 枚举所有设备
            if (device_count < 1) {
              RCLCPP_WARN(this->get_logger(), "No camera found. device_count = %d", device_count);
              std::this_thread::sleep_for(std::chrono::seconds(1));
              continue;
            }
            status = GXOpenDeviceByIndex(1, &camera_handle_);     // 通过序号打开设备
            if (!GX_SUCCESS(status)) {
              // std::cout<<device_count<<std::endl;
              RCLCPP_ERROR(this->get_logger(), "Can not open camera, status = %d", status);
            } else {
              GXSetBool(camera_handle_, GX_BOOL_REVERSE_X, 1);
              GXSetBool(camera_handle_, GX_BOOL_REVERSE_Y, 1);
              fail_conut_=0;
              break;
            }
          };
        }
        // double pnValue;
        // GXGetFloat(camera_handle_, GX_FLOAT_CURRENT_ACQUISITION_FRAME_RATE, &pnValue);
        // std::cout << "当前采集帧率: " << pnValue << std::endl;

        // int64_t link_cur;
        // GXGetInt(camera_handle_, GX_INT_DEVICE_LINK_CURRENT_THROUGHPUT, &link_cur);
        // std::cout<< "当前设备带宽: " << link_cur << std::endl;

        // Fetch image
        status = GXGetImage(camera_handle_, &bayer_frame, 700);  // 直接获取一帧图像

        if (GX_SUCCESS(status)) {
          DX_PIXEL_COLOR_FILTER bayer_type;   
          switch (bayer_frame.nPixelFormat) {  // 每个像素在图像中存储的颜色信息的格式
                case GX_PIXEL_FORMAT_BAYER_GR8: bayer_type = BAYERGR; break;
            case GX_PIXEL_FORMAT_BAYER_RG8: bayer_type = BAYERRG; break;
            case GX_PIXEL_FORMAT_BAYER_GB8: bayer_type = BAYERGB; break;
            case GX_PIXEL_FORMAT_BAYER_BG8: bayer_type = BAYERBG; break;
            default: RCLCPP_FATAL(this->get_logger(), "Unsupported Bayer layout: %d!", bayer_frame.nPixelFormat); return;
          }

          image_msg_.header.stamp = this->now();
          image_msg_.height = bayer_frame.nHeight;
          image_msg_.width = bayer_frame.nWidth;
          image_msg_.step = bayer_frame.nWidth * 3;
          image_msg_.data.resize(image_msg_.width * image_msg_.height * 3);

          status = DxRaw8toRGB24(bayer_frame.pImgBuf, image_msg_.data.data(),
                                 bayer_frame.nWidth, bayer_frame.nHeight,
                                 RAW2RGB_NEIGHBOUR, bayer_type, false);

          if (!GX_SUCCESS(status)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to convert Bayer to RGB, status = %d", status);
            continue;
          }

          RCLCPP_DEBUG(this->get_logger(), "Publish image: %dx%d", bayer_frame.nWidth, bayer_frame.nHeight);

          camera_info_msg_.header = image_msg_.header;
          camera_pub_.publish(image_msg_, camera_info_msg_);

          fail_conut_ = 0;
        } else {
          RCLCPP_WARN(this->get_logger(), "Get buffer failed, status = %d", status);
          GXSendCommand(camera_handle_, GX_COMMAND_ACQUISITION_STOP);
          status=GXSendCommand(camera_handle_, GX_COMMAND_ACQUISITION_START);
          RCLCPP_INFO(this->get_logger(),"status = %d",status);
          fail_conut_++;
        }
        if(fail_conut_>5)
        {
          GXCloseDevice(camera_handle_);
        }
      }
    }};

    RCLCPP_INFO(this->get_logger(), "GalaxyCamera initialized");
  }

  ~GalaxyCameraNode() override
  {
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    if (camera_handle_) {
      GXSendCommand(camera_handle_, GX_COMMAND_ACQUISITION_STOP);
      GXCloseDevice(camera_handle_);
    }
    GXCloseLib();
    RCLCPP_INFO(this->get_logger(), "GalaxyCameraNode destroyed!");
  }

private:
  void declareParameters()
  {
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    // name: 参数的名称，通常与实际参数名一致。
    // type: 参数的数据类型，定义为 rcl_interfaces::msg::ParameterType，可以是整数、浮点数、布尔值、字符串等。
    // description: 对参数的描述，提供参数用途或如何使用的详细信息。
    // additional_constraints: 额外约束，可能是一个字符串，用于描述参数的取值范围或其他限制条件。
    // read_only: 布尔值，指示该参数是否为只读，是否可以在运行时被修改。
    // floating_point_range: 如果参数是浮点数类型，可以指定其取值范围。
    // integer_range: 如果参数是整数类型，可以指定其取值范围。
    
    double f_value;
    GX_FLOAT_RANGE f_range;
    GX_STATUS status;
    param_desc.integer_range.resize(1);
    param_desc.integer_range[0].step = 1;
<<<<<<< HEAD
=======

    // 带宽
    status = GXSetEnum(camera_handle_, GX_ENUM_DEVICE_LINK_THROUGHPUT_LIMIT_MODE, GX_DEVICE_LINK_THROUGHPUT_LIMIT_MODE_OFF);
    /*
    GX_DEVICE_LINK_THROUGHPUT_LIMIT_MODE_OFF 关闭带宽限制模式
    GX_DEVICE_LINK_THROUGHPUT_LIMIT_MODE_ON
    */
    // status = GXSetInt(camera_handle_, GX_INT_DEVICE_LINK_THROUGHPUT_LIMIT, 400000000);

    // bool reverse;
    // GXGetBool(camera_handle_, GX_BOOL_REVERSE_X, &reverse);

    //GXSetBool(camera_handle_, GX_BOOL_REVERSE_X, 0);

    // std::cout << reverse <<std::endl;
    // std::cout << "1" << std::endl;

    // GXSendCommand(camera_handle_, )



    bool reverst;
    GXGetBool(camera_handle_, GX_BOOL_REVERSE_X, &reverst);

    GXSetBool(camera_handle_, GX_BOOL_REVERSE_X, 1);
    GXSetBool(camera_handle_, GX_BOOL_REVERSE_Y, 1);
    //std::cout<<reverst<<std::endl;



>>>>>>> 75f713b (clean)
    // Exposure time
    param_desc.description = "Exposure time in microseconds";    // 微妙
    GXGetFloat(camera_handle_, GX_FLOAT_EXPOSURE_TIME, &f_value);        // 获取浮点类型值的当前值
    GXGetFloatRange(camera_handle_, GX_FLOAT_EXPOSURE_TIME, &f_range);   // 获取Float类型值的最小值、最大值、步长等信息
    param_desc.integer_range[0].from_value = f_range.dMin;
    param_desc.integer_range[0].to_value = f_range.dMax;
    RCLCPP_INFO(this->get_logger(), "Exposure time: %f, range %f ~ %f", f_value, f_range.dMin, f_range.dMax);
    status = GXSetEnum(camera_handle_, GX_ENUM_EXPOSURE_AUTO, GX_EXPOSURE_AUTO_OFF); // Disable auto exposure
    if (!GX_SUCCESS(status)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to disable auto exposure, status = %d", status);
    }
    status = GXSetEnum(camera_handle_, GX_ENUM_EXPOSURE_MODE, GX_EXPOSURE_MODE_TIMED); // Set exposure to timed mode 曝光时间寄存器
    if (!GX_SUCCESS(status)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set exposure to timed mode, status = %d", status);
    }
    double exposure_time = this->declare_parameter("exposure_time", 5000, param_desc);
    status = GXSetFloat(camera_handle_, GX_FLOAT_EXPOSURE_TIME, exposure_time);
    if (!GX_SUCCESS(status)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set exposure time, status = %d", status);
    }

    // Gain
    param_desc.description = "Gain";
    GXGetFloat(camera_handle_, GX_FLOAT_GAIN, &f_value);
    GXGetFloatRange(camera_handle_, GX_FLOAT_GAIN, &f_range);
    param_desc.integer_range[0].from_value = f_range.dMin;
    param_desc.integer_range[0].to_value = f_range.dMax;
    RCLCPP_INFO(this->get_logger(), "Gain: %f, range %f ~ %f", f_value, f_range.dMin, f_range.dMax);
    double gain = this->declare_parameter("gain", f_value, param_desc);
    status = GXSetEnum(camera_handle_, GX_ENUM_GAIN_AUTO, GX_GAIN_AUTO_OFF); // Disable auto gain
    if (!GX_SUCCESS(status)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to disable auto gain, status = %d", status);
    }
    status = GXSetEnum(camera_handle_, GX_ENUM_GAIN_SELECTOR, GX_GAIN_SELECTOR_ALL); // Gain for all channels
    if (!GX_SUCCESS(status)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to select gain for all channels, status = %d", status);
    }
    status = GXSetFloat(camera_handle_, GX_FLOAT_GAIN, gain);
    if (!GX_SUCCESS(status)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to set gain, status = %d", status);
    }
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    GX_STATUS status;
    RCLCPP_ERROR(this->get_logger(), "parametersCallback!");
    for (const auto & param : parameters) {
      try {
        if (param.get_name() == "exposure_time") {
          status = GXSetFloat(camera_handle_, GX_FLOAT_EXPOSURE_TIME, param.as_int());
          if (!GX_SUCCESS(status)) {
            result.successful = false;
            result.reason = "Failed to set exposure time, status = " + std::to_string(status);
          }
        } else if (param.get_name() == "gain") {
          status = GXSetFloat(camera_handle_, GX_FLOAT_GAIN, param.as_double());
          if (!GX_SUCCESS(status)) {
            result.successful = false;
            result.reason = "Failed to set gain, status = " + std::to_string(status);
          }
        } else {
          result.successful = false;
          result.reason = "Unknown parameter: " + param.get_name();
        }
        RCLCPP_INFO(this->get_logger(), "Parameter set %s: %s",(result.successful ? "successfully" : "failed"), param.get_name().c_str());
      } catch (rclcpp::ParameterTypeException &e) {
        RCLCPP_ERROR(this->get_logger(), "Error setting parameter \"%s\": %s", param.get_name().c_str(), e.what());
      }
    }
    return result;
  }



  
  //sub exposure_time
  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr exposure_sub_;
  
  void exposureCallback(const std_msgs::msg::Int64::SharedPtr msg ){
      last_et = et;
      et = msg->data;
      if(last_et != et)
      {
        GX_STATUS status = GXSetFloat(camera_handle_,GX_FLOAT_EXPOSURE_TIME ,et);
        if (!GX_SUCCESS(status)) {
            RCLCPP_ERROR(this->get_logger(),"Failed to change exposure time!");
          }else {
            RCLCPP_INFO(this->get_logger(),"Succeeded to change exposure time!");
          }
      }
      
  }
   
  //sub gain
  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr gain_sub_;

  void gainCallback(const std_msgs::msg::Int64::SharedPtr msg ){
      last_gn = gn;
      gn = msg->data;
      if(last_gn != gn)
      {
        GX_STATUS status = GXSetFloat(camera_handle_,GX_FLOAT_GAIN ,gn);
        if (!GX_SUCCESS(status)) {
            RCLCPP_ERROR(this->get_logger(),"Failed to change gain!");
          }else {
            RCLCPP_INFO(this->get_logger(),"Succeeded to change gain!");
          }
      }
  }



  sensor_msgs::msg::Image image_msg_;

  image_transport::CameraPublisher camera_pub_;

  GX_DEV_HANDLE camera_handle_;
  struct {
    int64_t nWidthValue, nWidthMax;
    int64_t nHeightValue, nHeightMax;
  } img_info_;

  std::string camera_name_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;

  int et = 2800;
  int last_et =2800;
<<<<<<< HEAD
=======


  int gn = 6.0;
  int last_gn = 6.0;



>>>>>>> 75f713b (clean)
  int fail_conut_ = 0;
  std::thread capture_thread_;

  OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
};
}  // namespace galaxy_camera

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(galaxy_camera::GalaxyCameraNode)