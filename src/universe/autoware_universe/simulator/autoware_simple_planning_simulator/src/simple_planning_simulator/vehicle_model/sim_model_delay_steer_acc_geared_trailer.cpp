// Copyright 2025 The Autoware Foundation.
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

#include "autoware/simple_planning_simulator/vehicle_model/sim_model_delay_steer_acc_geared_trailer.hpp"

#include "autoware_vehicle_msgs/msg/gear_command.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace autoware::simulator::simple_planning_simulator
{

SimModelDelaySteerAccGearedTrailer::SimModelDelaySteerAccGearedTrailer(
  double vx_lim, double steer_lim, double vx_rate_lim, double steer_rate_lim,
  double wheelbase, double hitch_offset, double trailer_wheelbase, double max_hitch_angle,
  double dt,
  double acc_delay, double acc_time_constant,
  double steer_delay, double steer_time_constant,
  double steer_dead_band, double steer_bias,
  double debug_acc_scaling_factor, double debug_steer_scaling_factor)
: SimModelInterface(7 /* dim x */, 2 /* dim u */),
  MIN_TIME_CONSTANT(0.03),
  vx_lim_(vx_lim),
  vx_rate_lim_(vx_rate_lim),
  steer_lim_(steer_lim),
  steer_rate_lim_(steer_rate_lim),
  wheelbase_(wheelbase),
  hitch_offset_(hitch_offset),
  trailer_wheelbase_(trailer_wheelbase),
  max_hitch_angle_(max_hitch_angle),
  acc_delay_(acc_delay),
  acc_time_constant_(std::max(acc_time_constant, MIN_TIME_CONSTANT)),
  steer_delay_(steer_delay),
  steer_time_constant_(std::max(steer_time_constant, MIN_TIME_CONSTANT)),
  steer_dead_band_(steer_dead_band),
  steer_bias_(steer_bias),
  debug_acc_scaling_factor_(std::max(debug_acc_scaling_factor, 0.0)),
  debug_steer_scaling_factor_(std::max(debug_steer_scaling_factor, 0.0))
{
  initializeInputQueue(dt);
}

double SimModelDelaySteerAccGearedTrailer::getX()
{
  return state_(IDX::X);
}
double SimModelDelaySteerAccGearedTrailer::getY()
{
  return state_(IDX::Y);
}
double SimModelDelaySteerAccGearedTrailer::getYaw()
{
  return state_(IDX::YAW);
}
double SimModelDelaySteerAccGearedTrailer::getVx()
{
  return state_(IDX::VX);
}
double SimModelDelaySteerAccGearedTrailer::getVy()
{
  return 0.0;
}
double SimModelDelaySteerAccGearedTrailer::getAx()
{
  return state_(IDX::ACCX);
}
double SimModelDelaySteerAccGearedTrailer::getWz()
{
  return state_(IDX::VX) * std::tan(state_(IDX::STEER)) / wheelbase_;
}
double SimModelDelaySteerAccGearedTrailer::getSteer()
{
  // measured value includes bias
  return state_(IDX::STEER) + steer_bias_;
}
double SimModelDelaySteerAccGearedTrailer::getHitchAngle()
{
  return state_(IDX::HITCH_ANGLE);
}

void SimModelDelaySteerAccGearedTrailer::initializeInputQueue(const double & dt)
{
  const size_t acc_q_size = static_cast<size_t>(std::round(acc_delay_ / dt));
  acc_input_queue_.assign(acc_q_size, 0.0);

  const size_t steer_q_size = static_cast<size_t>(std::round(steer_delay_ / dt));
  steer_input_queue_.assign(steer_q_size, 0.0);
}

void SimModelDelaySteerAccGearedTrailer::update(const double & dt)
{
  Eigen::VectorXd delayed_input = Eigen::VectorXd::Zero(dim_u_);

  acc_input_queue_.push_back(input_(IDX_U::ACCX_DES));
  delayed_input(IDX_U::ACCX_DES) = acc_input_queue_.front();
  acc_input_queue_.pop_front();

  steer_input_queue_.push_back(input_(IDX_U::STEER_DES));
  delayed_input(IDX_U::STEER_DES) = steer_input_queue_.front();
  steer_input_queue_.pop_front();

  const auto prev_state = state_;
  updateRungeKutta(dt, delayed_input);

  // velocity limit
  state_(IDX::VX) = std::max(-vx_lim_, std::min(state_(IDX::VX), vx_lim_));

  // jackknife limit — clamp hitch angle to prevent physical impossibility
  state_(IDX::HITCH_ANGLE) =
    std::max(-max_hitch_angle_, std::min(state_(IDX::HITCH_ANGLE), max_hitch_angle_));

  // gear-based direction enforcement
  updateStateWithGear(state_, prev_state, gear_, dt);
}

Eigen::VectorXd SimModelDelaySteerAccGearedTrailer::calcModel(
  const Eigen::VectorXd & state, const Eigen::VectorXd & input)
{
  auto sat = [](double val, double u, double l) { return std::max(std::min(val, u), l); };

  const double vel = sat(state(IDX::VX), vx_lim_, -vx_lim_);
  const double acc = sat(state(IDX::ACCX), vx_rate_lim_, -vx_rate_lim_);
  const double yaw = state(IDX::YAW);
  const double steer = state(IDX::STEER);
  const double beta = state(IDX::HITCH_ANGLE);

  const double acc_des =
    sat(input(IDX_U::ACCX_DES), vx_rate_lim_, -vx_rate_lim_) * debug_acc_scaling_factor_;
  const double steer_des =
    sat(input(IDX_U::STEER_DES), steer_lim_, -steer_lim_) * debug_steer_scaling_factor_;

  // Steering dynamics (first-order with dead band)
  // NOTE: getSteer() reads state_(IDX::STEER) + bias; at integration sub-steps we use the
  //       current sub-state's steer instead of the member state_, so compute measured here.
  const double steer_measured = steer + steer_bias_;
  const double steer_diff = steer_measured - steer_des;
  const double steer_diff_db = std::invoke([&]() -> double {
    if (steer_diff > steer_dead_band_) {
      return steer_diff - steer_dead_band_;
    } else if (steer_diff < -steer_dead_band_) {
      return steer_diff + steer_dead_band_;
    }
    return 0.0;
  });
  const double steer_rate =
    sat(-steer_diff_db / steer_time_constant_, steer_rate_lim_, -steer_rate_lim_);

  Eigen::VectorXd d_state = Eigen::VectorXd::Zero(dim_x_);

  // Truck bicycle kinematics
  d_state(IDX::X) = vel * std::cos(yaw);
  d_state(IDX::Y) = vel * std::sin(yaw);
  d_state(IDX::YAW) = vel * std::tan(steer) / wheelbase_;
  d_state(IDX::VX) = acc;
  d_state(IDX::STEER) = steer_rate;
  d_state(IDX::ACCX) = -(acc - acc_des) / acc_time_constant_;

  // Trailer hitch-angle kinematics (non-holonomic constraint at hitch)
  //
  //   β̇ = v · [ tan(δ)/L₁ · (1 + M·cos(β)/L₂)  −  sin(β)/L₂ ]
  //
  // When M = 0 (hitch exactly at rear axle) this reduces to the classic:
  //   β̇ = v · [ tan(δ)/L₁  −  sin(β)/L₂ ]
  d_state(IDX::HITCH_ANGLE) = vel * (
    std::tan(steer) / wheelbase_ * (1.0 + hitch_offset_ * std::cos(beta) / trailer_wheelbase_) -
    std::sin(beta) / trailer_wheelbase_);

  return d_state;
}

void SimModelDelaySteerAccGearedTrailer::updateStateWithGear(
  Eigen::VectorXd & state, const Eigen::VectorXd & prev_state,
  const uint8_t gear, const double dt)
{
  const auto setStopState = [&]() {
    state(IDX::VX) = 0.0;
    state(IDX::X) = prev_state(IDX::X);
    state(IDX::Y) = prev_state(IDX::Y);
    state(IDX::YAW) = prev_state(IDX::YAW);
    // Trailer position relative to truck is unchanged when truck is stopped
    state(IDX::HITCH_ANGLE) = prev_state(IDX::HITCH_ANGLE);
    state(IDX::ACCX) = (state(IDX::VX) - prev_state(IDX::VX)) / std::max(dt, 1.0e-5);
  };

  using autoware_vehicle_msgs::msg::GearCommand;
  if (
    gear == GearCommand::DRIVE || gear == GearCommand::DRIVE_2 || gear == GearCommand::DRIVE_3 ||
    gear == GearCommand::DRIVE_4 || gear == GearCommand::DRIVE_5 || gear == GearCommand::DRIVE_6 ||
    gear == GearCommand::DRIVE_7 || gear == GearCommand::DRIVE_8 || gear == GearCommand::DRIVE_9 ||
    gear == GearCommand::DRIVE_10 || gear == GearCommand::DRIVE_11 ||
    gear == GearCommand::DRIVE_12 || gear == GearCommand::DRIVE_13 ||
    gear == GearCommand::DRIVE_14 || gear == GearCommand::DRIVE_15 ||
    gear == GearCommand::DRIVE_16 || gear == GearCommand::DRIVE_17 ||
    gear == GearCommand::DRIVE_18 || gear == GearCommand::LOW || gear == GearCommand::LOW_2) {
    if (state(IDX::VX) < 0.0) {
      setStopState();
    }
  } else if (gear == GearCommand::REVERSE || gear == GearCommand::REVERSE_2) {
    if (state(IDX::VX) > 0.0) {
      setStopState();
    }
  } else {  // PARK or unknown
    setStopState();
  }
}

}  // namespace autoware::simulator::simple_planning_simulator
