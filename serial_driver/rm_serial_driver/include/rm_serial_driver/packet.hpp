#ifndef RM_SERIAL_DRIVER__PACKET_HPP_
#define RM_SERIAL_DRIVER__PACKET_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rm_serial_driver
{

struct ReceivePacket
{
  uint8_t header = 0;  // 0x5A

  // vision 
  uint8_t detect_color : 1;  // 0-red 1-blue
  uint8_t reset_tracker : 1;
  uint8_t reserved : 6;
  float roll;
  float pitch;
  float yaw;
  float control_id;
  uint8_t revivatory_car; // 定义1为活着，0为死了，0变为1判定为复活10s内不击打 

  // nav
  float yaw_delta; // 云台和底盘相对角度

  // referee system
  // game state
  uint8_t game_progress;  // 当前比赛状态 0:未开始比赛 1:准备阶段 2:自检阶段 
                          //            3:五秒倒计时 4:比赛中   5:比赛结算中
  uint16_t remain_time;   // 比赛剩余时间 单位:s

  // robot state
  uint16_t current_hp;  // 当前血量
  uint16_t projectile;  // 当前剩余允许发弹量
  uint8_t sentry_info;  // bit 0: 装甲板是否被攻击 0:否 1:是
                        // bit 1: 是否脱战 0:否 1:是
                        // bit 2: RFID 是否检测到堡垒 0:否 1:是
                        // bit 3: RFID 是否检测到补给区(与兑换站不重叠) 0:否 1:是
                        // bit 4: RFID 是否检测到补给区(与兑换站重叠) 0:否 1:是
                        // bit 5: 当前剩余能量值是否小于30% 0:否 1:是
                        // bit 6-7: 0

  // building state
  uint16_t red_outpost_hp;  // 红方前哨站血量
  uint16_t red_base_hp;     // 红方基地血量
  uint16_t blue_outpost_hp; // 蓝方前哨站血量
  uint16_t blue_base_hp;    // 蓝方基地血量

  uint8_t end_frame;                                 
} __attribute__((packed));
static_assert(sizeof(ReceivePacket) == 40, "ReceivePacket size error");


struct VisionPacket
{
  bool tracking : 1 = false;
  uint8_t id : 3 = 0;          // 0-outpost 6-guard 7-base
  uint8_t armors_num : 3 = 0;  // 2-balance 3-outpost 4-normal
  uint8_t detecting : 1 = false;
  float yaw = 0;
  float pitch = 0;
  float fire = 0;
  float distance = 0;
  float r = 0;
  float v_horizon = 0;
  float armor_yaw_offset = 0;
  uint8_t another_fire = false;
} __attribute__((packed));
static_assert(sizeof(VisionPacket) == 30, "Vision Packet size error");

struct NavPacket
{
  float linear_x = 0.0;
  float linear_y = 0.0;
  float angular_z = 0.0;
  uint8_t tuoluo = 0;
} __attribute__((packed));
static_assert(sizeof(NavPacket) == 13, "Nav Packet size error");


struct BackArmorPacket
{
  uint8_t number = 0;
  float angle = 0;
} __attribute__((packed));
static_assert(sizeof(BackArmorPacket) == 5, "Back Armor Packet size error");


struct TotalSendPacket
{
  uint8_t header = 0xA5;
  VisionPacket vision_pck;
  NavPacket nav_pck;
  BackArmorPacket back_armor_pck;
  uint8_t end_frame = 0x4A;
} __attribute__((packed));
static_assert(sizeof(TotalSendPacket) == (sizeof(VisionPacket) + sizeof(NavPacket) + sizeof(BackArmorPacket) + 2), "Total Send Packet size error");


inline ReceivePacket fromVector(const std::vector<uint8_t> & data)
{
  ReceivePacket packet;
  std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&packet));
  return packet;
}

inline std::vector<uint8_t> toVector(const VisionPacket & data)
{
  std::vector<uint8_t> packet(sizeof(VisionPacket));
  std::copy(
    reinterpret_cast<const uint8_t *>(&data),
    reinterpret_cast<const uint8_t *>(&data) + sizeof(VisionPacket), packet.begin());
  return packet;
}

inline std::vector<uint8_t> toVector(const NavPacket & data)
{
  std::vector<uint8_t> packet(sizeof(NavPacket));
  std::copy(
    reinterpret_cast<const uint8_t *>(&data),
    reinterpret_cast<const uint8_t *>(&data) + sizeof(NavPacket), packet.begin());
  return packet;
}

inline std::vector<uint8_t> toVector(const TotalSendPacket & data)
{
  std::vector<uint8_t> packet(sizeof(TotalSendPacket));
  std::copy(
    reinterpret_cast<const uint8_t *>(&data),
    reinterpret_cast<const uint8_t *>(&data) + sizeof(TotalSendPacket), packet.begin());
  return packet;
}

}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__PACKET_HPP_
