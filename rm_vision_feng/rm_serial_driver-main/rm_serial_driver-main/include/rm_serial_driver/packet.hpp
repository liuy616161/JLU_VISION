// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#ifndef RM_SERIAL_DRIVER__PACKET_HPP_
#define RM_SERIAL_DRIVER__PACKET_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rm_serial_driver
{
struct ReceivePacket
{
  uint8_t header = 0x5A;
  uint8_t detect_color : 1;  // 0-red 1-blue
  uint8_t reset_tracker : 1;
  uint8_t reserved : 6;
  float roll;
  float pitch;
  float yaw;
  float control_id;
  float useless[11];
  uint8_t rubbish;
  uint8_t end_frame;
  // uint16_t checksum = 0;                                   
} __attribute__((packed));
static_assert(sizeof(ReceivePacket) == 64, "ReceivePacket size error");

struct new_SendPacket //
{
  uint8_t header = 0xA5;
  bool tracking : 1;
  uint8_t id : 3;          // 0-outpost 6-guard 7-base
  uint8_t armors_num : 3;  // 2-balance 3-outpost 4-normal
  // bool fire : 1;
  uint8_t detecting : 1;
  float yaw;
  float pitch;
  float fire;
  float distance;
  uint16_t checksum = 0;
} __attribute__((packed));
static_assert(sizeof(new_SendPacket) == 20, "new_SendPacket size error");

struct TotalSendPacket //
{
  uint8_t header = 0xA5;
  bool tracking : 1;
  uint8_t id : 3;          // 0-outpost 6-guard 7-base
  uint8_t armors_num : 3;  // 2-balance 3-outpost 4-normal
  // bool fire : 1;
  uint8_t reserved : 1;
  float yaw;
  float pitch;
  float fire;
  float distance;
  float linear_x;
  float linear_y;
  float angular_z;
  float chassis_power;
  float buffer_energy;
  float rotation;
  float shooter_17mm_1_barrel_heat;
  float shooter_17mm_2_barrel_heat;
  float initial_speed;
  float launching_frequency;
  uint16_t checksum = 0;
} __attribute__((packed));
static_assert(sizeof(TotalSendPacket) == 60, "new_SendPacket size error");

struct PowerHeatDataStruct { 
    uint16_t chassis_voltage; 
    uint16_t chassis_current; 
    float chassis_power; 
    uint16_t buffer_energy; 
    uint16_t shooter_17mm_1_barrel_heat; 
    uint16_t shooter_17mm_2_barrel_heat; 
    uint16_t shooter_42mm_barrel_heat; }; 
static_assert(sizeof(PowerHeatDataStruct) == 16, "PowerHeatDataStruct must be 16 bytes long with packing");

inline ReceivePacket fromVector(const std::vector<uint8_t> & data)
{
  ReceivePacket packet;
  std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&packet));
  return packet;
}

inline std::vector<uint8_t> toVector(const new_SendPacket & data)
{
  std::vector<uint8_t> packet(sizeof(new_SendPacket));
  std::copy(
    reinterpret_cast<const uint8_t *>(&data),
    reinterpret_cast<const uint8_t *>(&data) + sizeof(new_SendPacket), packet.begin());
  return packet;
}

}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__PACKET_HPP_
