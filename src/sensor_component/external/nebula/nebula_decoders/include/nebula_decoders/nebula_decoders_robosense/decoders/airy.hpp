// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "nebula_decoders/nebula_decoders_robosense/decoders/robosense_packet.hpp"
#include "nebula_decoders/nebula_decoders_robosense/decoders/robosense_sensor.hpp"

#include "boost/endian/buffers.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

using namespace boost::endian;  // NOLINT(build/namespaces)

namespace nebula::drivers
{
namespace robosense_packet::airy
{
#pragma pack(push, 1)

struct Header
{
  big_uint32_buf_t pkt_head;
  uint8_t reserved0[8];
  big_uint32_buf_t packet_count;
  uint8_t reserved1[4];
  Timestamp timestamp;
  big_uint8_buf_t reserved2;
  big_uint8_buf_t lidar_type;
  big_uint8_buf_t lidar_model;
  uint8_t reserved3[9];
};

struct Packet : public PacketBase<8, 48, 1, 100>
{
  using body_t = Body<Block<Unit, Packet::n_channels>, Packet::n_blocks>;
  Header header;
  body_t body;
  uint8_t reserved_tail[4];
  big_uint16_buf_t tail;
};

struct SensorCalibration
{
  ChannelAngleCorrection vertical[96];
  ChannelAngleCorrection horizontal[96];

  [[nodiscard]] RobosenseCalibrationConfiguration get_calibration() const
  {
    RobosenseCalibrationConfiguration calibration;
    calibration.set_channel_size(96);
    for (size_t i = 0; i < 96; ++i) {
      ChannelCorrection channel_correction;
      channel_correction.azimuth = horizontal[i].get_angle();
      channel_correction.elevation = vertical[i].get_angle();
      calibration.calibration[i] = channel_correction;
    }
    return calibration;
  }
};

struct ImuCalibration
{
  uint8_t raw[28];
};

struct InfoPacket
{
  big_uint64_buf_t header;
  big_uint16_buf_t motor_speed_setting;
  IpAddress ethernet_source_ip;
  IpAddress ethernet_destination_ip;
  MacAddress lidar_mac;
  big_uint16_buf_t msop_port;
  uint8_t reserved0[2];
  big_uint16_buf_t difop_port;
  uint8_t reserved1[10];
  FirmwareVersion mainboard_firmware_version;
  FirmwareVersion baseboard_firmware_version;
  FirmwareVersion app_firmware_version;
  FirmwareVersion motor_firmware_version;
  uint8_t reserved2[232];
  SerialNumber serial_number;
  uint8_t reserved3[2];
  big_uint8_buf_t return_mode;
  big_uint8_buf_t time_sync_mode;
  big_uint8_buf_t time_sync_status;
  Timestamp time;
  uint8_t reserved4[60];
  big_uint16_buf_t realtime_motor_speed;
  uint8_t reserved5[93];
  SensorCalibration sensor_calibration;
  big_uint16_buf_t mainboard_total_input_voltage;
  uint8_t reserved6[20];
  big_uint16_buf_t device_input_voltage;
  big_uint16_buf_t baseboard_12v_voltage;
  uint8_t reserved7[10];
  big_uint16_buf_t mainboard_transmit_temperature;
  uint8_t reserved8[10];
  ImuCalibration imu_calibration;
  uint8_t reserved9[126];
  big_uint16_buf_t tail;
};

#pragma pack(pop)
}  // namespace robosense_packet::airy

class Airy
: public RobosenseSensor<robosense_packet::airy::Packet, robosense_packet::airy::InfoPacket>
{
public:
  using angle_corrector_t =
    AngleCorrectorCalibrationBased<96, robosense_packet::airy::Packet::degree_subdivisions>;
  static constexpr float min_range = 0.1F;
  static constexpr float max_range = 60.0F;
  static constexpr size_t max_scan_buffer_points = 120000;

  [[nodiscard]] uint32_t get_global_channel_id(
    uint32_t block_id, uint32_t channel_id) const override
  {
    const bool upper_bank = (block_id % 2) != 0;
    return static_cast<uint32_t>(channel_id + (upper_bank ? 48 : 0));
  }

  int get_packet_relative_point_time_offset(
    const uint32_t block_id, const uint32_t channel_id,
    const std::shared_ptr<const RobosenseSensorConfiguration> & /*sensor_configuration*/) override
  {
    // RoboSense documentation revision 1.2 lists per-channel emission offsets. Until we obtain a
    // definitive timing table, approximate the firing sequence with monotonically increasing
    // offsets that maintain ordering within a packet. These offsets are derived from a combination
    // of the block index (column) and the global channel id.
    constexpr int channel_spacing_ns = 2'000;         // 2 µs between consecutive channels
    constexpr int column_spacing_ns = 96 * channel_spacing_ns;  // Four columns per packet

    const auto global_channel = static_cast<int>(channel_id);
    const int column_index = static_cast<int>(block_id / 2);

    return column_index * column_spacing_ns + global_channel * channel_spacing_ns;
  }

  ReturnMode get_return_mode(const robosense_packet::airy::InfoPacket & info_packet) override
  {
    switch (info_packet.return_mode.value()) {
      case 0x00:
        return ReturnMode::SINGLE_STRONGEST;
      case 0x01:
        return ReturnMode::SINGLE_FIRST;
      case 0x02:
        return ReturnMode::SINGLE_LAST;
      default:
        return ReturnMode::UNKNOWN;
    }
  }

  RobosenseCalibrationConfiguration get_sensor_calibration(
    const robosense_packet::airy::InfoPacket & info_packet) override
  {
    return info_packet.sensor_calibration.get_calibration();
  }

  bool get_sync_status(const robosense_packet::airy::InfoPacket & info_packet) override
  {
    return info_packet.time_sync_status.value() == 0x01;
  }

  std::map<std::string, std::string> get_sensor_info(
    const robosense_packet::airy::InfoPacket & info_packet) override
  {
    std::map<std::string, std::string> sensor_info;
    sensor_info["motor_speed_setting"] = std::to_string(info_packet.motor_speed_setting.value());
    sensor_info["source_ip"] = info_packet.ethernet_source_ip.to_string();
    sensor_info["destination_ip"] = info_packet.ethernet_destination_ip.to_string();
    sensor_info["mac_addr"] = info_packet.lidar_mac.to_string();
    sensor_info["msop_port"] = std::to_string(info_packet.msop_port.value());
    sensor_info["difop_port"] = std::to_string(info_packet.difop_port.value());
    sensor_info["serial_number"] = info_packet.serial_number.to_string();
    sensor_info["return_mode_raw"] = std::to_string(info_packet.return_mode.value());
    sensor_info["time_sync_mode_raw"] = std::to_string(info_packet.time_sync_mode.value());
    sensor_info["time_sync_status_raw"] = std::to_string(info_packet.time_sync_status.value());
    sensor_info["realtime_motor_speed"] = std::to_string(info_packet.realtime_motor_speed.value());
    sensor_info["mainboard_total_input_voltage"] =
      std::to_string(info_packet.mainboard_total_input_voltage.value());
    sensor_info["device_input_voltage"] = std::to_string(info_packet.device_input_voltage.value());
    sensor_info["baseboard_12v_voltage"] =
      std::to_string(info_packet.baseboard_12v_voltage.value());
    sensor_info["mainboard_transmit_temperature"] =
      std::to_string(info_packet.mainboard_transmit_temperature.value());
    return sensor_info;
  }
};

}  // namespace nebula::drivers
