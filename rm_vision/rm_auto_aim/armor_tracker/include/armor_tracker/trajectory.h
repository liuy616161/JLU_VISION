#ifndef _TRAJECTORY_H
#define _TRAJECTORY_H

#include <cmath>
#include <iostream>
#include <vector>
#include <utility>
#include "math.h"

#define pi 3.14159265358979323846

class PredictPitchXY
{
private:
  PredictPitchXY() = default;
  ~PredictPitchXY() = default;
  
  // 单例模式
  static PredictPitchXY instance;
  PredictPitchXY(const PredictPitchXY &) = delete;
  PredictPitchXY &operator=(const PredictPitchXY &) = delete;

  double time_acc = 0;
  
  static constexpr int maxstep = 100;        // 迭代最大次数
  static constexpr double minerr_y = 0.01;    // y轴容许误差，y轴迭代后允许的最大误差
  static constexpr double time_step = 0.0004;  // 时间步长
  static constexpr double k = 0.01003; // 阻力系数(大弹丸：0.00556、小弹丸：0.01903、发光大弹丸：0.00530)
  static constexpr double dt = 0.0005; // 每次仿真计算的步长
  static constexpr double g = 9.8;

  /**
    * @brief X、Y resistance(RK4)
    */
  double dropshotRK45(double d0, double v0, double X_r, double Y_r);

  struct tar
  {
    double x, y, z, r, yaw_offset;
  };

  static std::pair<int, tar> predict_nearest(
    double delay,
    double xw, double yw, double zw, double armor_yaw,
    double vxw, double vyw, double vzw, double v_yaw,
    double r1, double r2, double dz, int armors_num);

public:
  double operator()(double bulletOffsetX, double bulletSpeedNow, double distance, double hrRec);

  static PredictPitchXY &getinstance();
  
  void aim_armor(
    double algorithm_time, double control_time,
    double xw, double yw, double zw, double armor_yaw,
    double vxw, double vyw, double vzw, double v_yaw,
    double r1, double r2, double dz, int armors_num,
    float& aim_x, float& aim_y, float& aim_z,
    float& aim_armor_r, float& aim_armor_yaw_offset, float& fire) const;
  
  void aim_center(
    double algorithm_time, double control_time,
    double xw, double yw, double zw, double armor_yaw,
    double vxw, double vyw, double vzw, double v_yaw,
    double r1, double r2, double dz, int armors_num,
    float& aim_x, float& aim_y, float& aim_z,
    float& aim_armor_r, float& aim_armor_yaw_offset, float& fire) const;
};

#endif