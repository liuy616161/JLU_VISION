#include "armor_tracker/trajectory.h"

PredictPitchXY PredictPitchXY::instance;  // 一定要初始化静态变量

PredictPitchXY& PredictPitchXY::getinstance() { return instance; } // 返回单例(感觉用不着)

double PredictPitchXY::operator()(double bulletOffsetX, double bulletSpeedNow, double distance, double hrRec)
{
    return dropshotRK45(bulletOffsetX, bulletSpeedNow, distance, hrRec) * 180.0 / pi;
}

double PredictPitchXY::dropshotRK45(double d0, double v0, double X_r, double Y_r)
{
  double theta = 5 * pi / 180.0;  // 初始化角度

  for (int i = 0; i < maxstep; i++)
  {
    time_acc = 0;
    double theta_d = theta;  // 将theta赋给过程量
    double X_d = d0 * cos(theta_d);   // 2024infantry枪管相对于云台中心的水平长度
    double Y_d = d0 * sin(theta_d) ;  // 2024infantry枪管相对于云台中心的垂直高度
    double v0_d = v0;

    while (X_d < X_r)  // 迭代
    {
      // 计算各阶

      auto k1_v = (-k * pow(v0_d, 2) - g * sin(theta_d)) * time_step;
      auto k1_theta = (-g * cos(theta_d) / v0_d) * time_step;
      // 由于大弹丸很难产生马格努斯效应（横向摩擦轮怎么可能产生后向旋转），故忽略升力
      // 其实这里和公式是不完全相同的，未考虑t随步长的改变，因为t在推导过程中简化约去了

      auto k1_v_2 = v0_d + k1_v / 4.0;
      auto k1_theta_2 = theta_d + k1_theta / 4.0;

      auto k2_v = (-k * pow(k1_v_2, 2) - g * sin(k1_theta_2)) * time_step;
      auto k2_theta = (-g * cos(k1_theta_2) / k1_v_2) * time_step;
      auto k12_v_3 = v0_d + 3.0 / 32.0 * k1_v + 9.0 / 32.0 * k2_v;
      auto k12_theta_3 = theta_d + 3.0 / 32.0 * k1_theta + 9.0 / 32.0 * k2_theta;

      auto k3_v = (-k * pow(k12_v_3, 2) - g * sin(k12_theta_3)) * time_step;
      auto k3_theta = (-g * cos(k12_theta_3) / k12_v_3) * time_step;
      auto k123_v_4 = v0_d + 1932.0 / 2179.0 * k1_v - 7200.0 / 2179.0 * k2_v + 7296.0 / 2179.0 * k3_v;
      auto k123_theta_4 =
          theta_d + 1932.0 / 2179.0 * k1_theta - 7200.0 / 2179.0 * k2_theta + 7296.0 / 2179.0 * k3_theta;

      auto k4_v = (-k * pow(k123_v_4, 2) - g * sin(k123_theta_4)) * time_step;
      auto k4_theta = (-g * cos(k123_theta_4) / k123_v_4) * time_step;
      auto k1234_v_5 = v0_d + 439.0 / 216.0 * k1_v - 8.0 * k2_v + 3680.0 / 513.0 * k3_v - 845.0 / 4140.0 * k4_v;
      auto k1234_theta_5 =
          theta_d + 439.0 / 216.0 * k1_theta - 8.0 * k2_theta + 3680.0 / 513.0 * k3_theta - 845.0 / 4140.0 * k4_theta;

      auto k5_v = (-k * pow(k1234_v_5, 2) - g * sin(k1234_theta_5)) * time_step;
      auto k5_theta = (-g * cos(k1234_theta_5) / k1234_v_5) * time_step;
      auto k12345_v_6 =
          v0_d - 8.0 / 27.0 * k1_v + 2.0 * k2_v - 3544.0 / 2565.0 * k3_v + 1859.0 / 4104.0 * k4_v - 11.0 / 40.0 * k5_v;
      auto k12345_theta_6 = theta_d - 8.0 / 27.0 * k1_theta + 2.0 * k2_theta - 3544.0 / 2565.0 * k3_theta +
                            1859.0 / 4104.0 * k4_theta - 11.0 / 40.0 * k5_theta;

      auto k6_v = (-k * pow(k12345_v_6, 2) - g * sin(k12345_theta_6)) * time_step;
      auto k6_theta = (-g * cos(k12345_theta_6) / k12345_v_6) * time_step;

      auto vclass_5 = v0_d + 16.0 / 135.0 * k1_v + 6656.0 / 12825.0 * k3_v + 28561.0 / 56430.0 * k4_v -
                      9.0 / 50.0 * k5_v + 2.0 / 55.0 * k6_v;
      auto thetaclass_5 = theta_d + 16.0 / 135.0 * k1_theta + 6656.0 / 12825.0 * k3_theta +
                          28561.0 / 56430.0 * k4_theta - 9.0 / 50.0 * k5_theta + 2.0 / 55.0 * k6_theta;
                          
      v0_d = vclass_5;
      theta_d = thetaclass_5;
      X_d += time_step * v0_d * cos(theta_d);
      Y_d += time_step * v0_d * sin(theta_d);
      time_acc += time_step;
    }

    // 评估迭代结果，修正theta
    double error = Y_r - Y_d;  // error可以pid调节

    if (abs(error) < minerr_y)
    {
      return theta;
    }  // 合适则输出本次迭代使用的theta
    else
    {
      theta += atan((error) / X_r);
    }
  }

  // 迭代失败
  return atan(Y_r / X_r);
}

std::pair<int, PredictPitchXY::tar> PredictPitchXY::predict_nearest(
  double delay,
  double xw, double yw, double zw, double armor_yaw,
  double vxw, double vyw, double vzw, double v_yaw,
  double r1, double r2, double dz, int armors_num)
{
  bool use_1 = true;
  std::vector<tar> targets(armors_num);

  for (int i = 0; i < armors_num; i++)
  {
    double x = xw + vxw * delay, y = yw + vyw * delay;
    double r = use_1 ? r1 : r2;
    double yaw = armor_yaw + i * pi * 2 / armors_num + v_yaw * delay;

    targets[i].x = x - r * cos(yaw);
    targets[i].y = y - r * sin(yaw);
    targets[i].z = use_1 ? zw : dz + zw;  //+ vzw * timeDelay;
    targets[i].r = r;
    targets[i].yaw_offset = yaw - atan2(targets[i].y, targets[i].x);

    use_1 ^= armors_num == 4;
  }

  double dis_diff_min = targets[0].x * targets[0].x + targets[0].y * targets[0].y;
  int idx = 0;

  for (int i = 1; i < armors_num; i++)
  {
    double temp_dis_diff = targets[i].x * targets[i].x + targets[i].y * targets[i].y;

    if (temp_dis_diff < dis_diff_min)
    {
      dis_diff_min = temp_dis_diff;
      idx = i;
    }
  }

  return {idx, targets[idx]};
}

void PredictPitchXY::aim_armor(
  double algorithm_time, double control_time,
  double xw, double yw, double zw, double armor_yaw,
  double vxw, double vyw, double vzw, double v_yaw,
  double r1, double r2, double dz, int armors_num,
  float& aim_x, float& aim_y, float& aim_z,
  float& aim_armor_r, float& aim_armor_yaw_offset, float& fire) const
{
  auto [idx, aim] = predict_nearest(algorithm_time + time_acc, xw, yw, zw, armor_yaw, vxw, vyw, vzw, v_yaw, r1, r2, dz, armors_num);
  aim_x = aim.x;
  aim_y = aim.y;
  aim_z = aim.z;
  aim_armor_r = 0;
  aim_armor_yaw_offset = aim.yaw_offset;

  fire =
    idx ==
    predict_nearest(algorithm_time + control_time + time_acc, xw, yw, zw, armor_yaw, vxw, vyw, vzw, v_yaw, r1, r2, dz, armors_num).first
    ? 1.0 : -1.0;
}

void PredictPitchXY::aim_center(
  double algorithm_time, double control_time,
  double xw, double yw, double zw, double armor_yaw,
  double vxw, double vyw, double vzw, double v_yaw,
  double r1, double r2, double dz, int armors_num,
  float& aim_x, float& aim_y, float& aim_z,
  float& aim_armor_r, float& aim_armor_yaw_offset, float& fire) const
{
  auto [_, aim_center] = predict_nearest(algorithm_time + time_acc, xw, yw, zw, 0, vxw, vyw, vzw, 0, 0, 0, 0, 1);
  aim_x = aim_center.x;
  aim_y = aim_center.y;
  auto [__, aim_target] = predict_nearest(algorithm_time + control_time + time_acc, xw, yw, zw, armor_yaw, vxw, vyw, vzw, v_yaw, r1, r2, dz, armors_num);
  aim_z = aim_target.z;
  aim_armor_r = aim_target.r;
  aim_armor_yaw_offset = aim_target.yaw_offset;
  fire = 1.0;
}