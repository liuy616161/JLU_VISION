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
  bool reset_tracker : 1;
  uint8_t task_mode : 2;   // 0-aim 1-small buff   2 big buff
  bool rune_direction : 1;  // 0-anti-clockwise 1-clockwise
  bool rune_stable : 1;    // 0-unstable 1-stable
  uint8_t change_exposure : 2;   //0-stay  1-add 2-sub
  float roll;
  float pitch;
  float yaw;
  int8_t sentry_decision;
  uint16_t checksum = 0;                                   
} __attribute__((packed));


struct SendPacket
{
  uint8_t header = 0xA5;
  int8_t tracking;
  // int8_t fire;
  uint8_t id : 3;          // 0-outpost 6-guard 7-base
  uint8_t armors_num : 3;  // 2-balance 3-outpost 4-normal
  uint8_t flag_spin_mov : 1;
  uint8_t reserved : 1;
  uint8_t exposure_time : 6;
  float yaw;
  float pitch;
  float fire;
  float v_yaw;
  float dist;
  // float aim_x;
  // float aim_y;
  // float aim_z;
  uint16_t checksum = 0;
} __attribute__((packed));



inline ReceivePacket fromVector(const std::vector<uint8_t> & data)
{
  ReceivePacket packet;
  std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&packet));
  return packet;
}

inline std::vector<uint8_t> toVector(const SendPacket & data)
{
  std::vector<uint8_t> packet(sizeof(SendPacket));
  std::copy(
    reinterpret_cast<const uint8_t *>(&data),
    reinterpret_cast<const uint8_t *>(&data) + sizeof(SendPacket), packet.begin());
  return packet;
}


}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__PACKET_HPP_
