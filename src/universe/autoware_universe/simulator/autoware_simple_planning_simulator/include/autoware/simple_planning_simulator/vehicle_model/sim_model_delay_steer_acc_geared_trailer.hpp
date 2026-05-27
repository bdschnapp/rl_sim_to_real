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

#ifndef AUTOWARE__SIMPLE_PLANNING_SIMULATOR__VEHICLE_MODEL__SIM_MODEL_DELAY_STEER_ACC_GEARED_TRAILER_HPP_  // NOLINT
#define AUTOWARE__SIMPLE_PLANNING_SIMULATOR__VEHICLE_MODEL__SIM_MODEL_DELAY_STEER_ACC_GEARED_TRAILER_HPP_  // NOLINT

#include "autoware/simple_planning_simulator/vehicle_model/sim_model_interface.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

#include <deque>
#include <iostream>

namespace autoware::simulator::simple_planning_simulator
{

/**
 * @class SimModelDelaySteerAccGearedTrailer
 * @brief Extends DELAY_STEER_ACC_GEARED with an articulated trailer coupled via
 *        a revolute (hitch) joint at a fixed offset from the truck rear axle.
 *
 * A single additional state β (hitch_angle = θ_truck - θ_trailer) is integrated
 * alongside the standard truck dynamics.
 *
 * Trailer kinematics (from the non-holonomic constraint at the hitch):
 *
 *   β̇ = v · [ tan(δ)/L₁ · (1 + M·cos(β)/L₂)  −  sin(β)/L₂ ]
 *
 * Symbols:
 *   L₁  – truck wheelbase           [m]
 *   L₂  – trailer wheelbase (hitch → trailer rear axle) [m]
 *   M   – hitch offset from truck rear axle (+forward / –rearward) [m]
 *   δ   – front steering angle       [rad]
 *   β   – hitch angle                [rad], positive = trailer swings left
 *
 * State vector (dim 7):  [ X, Y, YAW, VX, STEER, ACCX, HITCH_ANGLE ]
 * Input vector (dim 2):  [ ACCX_DES, STEER_DES ]
 * Gear is handled via setGear() / gear_ as in the parent model.
 */
class SimModelDelaySteerAccGearedTrailer : public SimModelInterface
{
public:
  /**
   * @brief constructor
   * @param [in] vx_lim             velocity limit [m/s]
   * @param [in] steer_lim          steering limit [rad]
   * @param [in] vx_rate_lim        acceleration limit [m/s²]
   * @param [in] steer_rate_lim     steering-rate limit [rad/s]
   * @param [in] wheelbase          truck wheelbase L₁ [m]
   * @param [in] hitch_offset       truck rear-axle → hitch M [m]  (+ = forward)
   * @param [in] trailer_wheelbase  hitch → trailer rear-axle L₂ [m]
   * @param [in] max_hitch_angle    jackknife clamping limit [rad]
   * @param [in] dt                 simulation time step [s]
   * @param [in] acc_delay          acceleration command delay [s]
   * @param [in] acc_time_constant  acceleration first-order time constant [s]
   * @param [in] steer_delay        steering command delay [s]
   * @param [in] steer_time_constant steering first-order time constant [s]
   * @param [in] steer_dead_band    steering dead band [rad]
   * @param [in] steer_bias         steering bias added to measured angle [rad]
   * @param [in] debug_acc_scaling_factor   scaling for acc command (1.0 = normal)
   * @param [in] debug_steer_scaling_factor scaling for steer command (1.0 = normal)
   */
  SimModelDelaySteerAccGearedTrailer(
    double vx_lim, double steer_lim, double vx_rate_lim, double steer_rate_lim,
    double wheelbase, double hitch_offset, double trailer_wheelbase, double max_hitch_angle,
    double dt,
    double acc_delay, double acc_time_constant,
    double steer_delay, double steer_time_constant,
    double steer_dead_band, double steer_bias,
    double debug_acc_scaling_factor, double debug_steer_scaling_factor);

  ~SimModelDelaySteerAccGearedTrailer() = default;

  /**
   * @brief get hitch (articulation) angle β = θ_truck − θ_trailer [rad]
   */
  double getHitchAngle() override;

private:
  const double MIN_TIME_CONSTANT;

  enum IDX {
    X = 0,
    Y,
    YAW,
    VX,
    STEER,
    ACCX,
    HITCH_ANGLE,  //!< β
  };

  enum IDX_U {
    ACCX_DES = 0,
    STEER_DES,
    DRIVE_SHIFT,  // not in the input vector; gear handled via setGear()
  };

  const double vx_lim_;
  const double vx_rate_lim_;
  const double steer_lim_;
  const double steer_rate_lim_;
  const double wheelbase_;          //!< truck wheelbase L₁ [m]
  const double hitch_offset_;       //!< truck rear-axle → hitch M [m]
  const double trailer_wheelbase_;  //!< hitch → trailer rear-axle L₂ [m]
  const double max_hitch_angle_;    //!< jackknife clamping limit [rad]

  std::deque<double> acc_input_queue_;
  std::deque<double> steer_input_queue_;

  const double acc_delay_;
  const double acc_time_constant_;
  const double steer_delay_;
  const double steer_time_constant_;
  const double steer_dead_band_;
  const double steer_bias_;
  const double debug_acc_scaling_factor_;
  const double debug_steer_scaling_factor_;

  void initializeInputQueue(const double & dt);

  double getX() override;
  double getY() override;
  double getYaw() override;
  double getVx() override;
  double getVy() override;
  double getAx() override;
  double getWz() override;
  double getSteer() override;

  void update(const double & dt) override;

  Eigen::VectorXd calcModel(
    const Eigen::VectorXd & state, const Eigen::VectorXd & input) override;

  void updateStateWithGear(
    Eigen::VectorXd & state, const Eigen::VectorXd & prev_state,
    const uint8_t gear, const double dt);
};

}  // namespace autoware::simulator::simple_planning_simulator

// NOLINTNEXTLINE
#endif  // AUTOWARE__SIMPLE_PLANNING_SIMULATOR__VEHICLE_MODEL__SIM_MODEL_DELAY_STEER_ACC_GEARED_TRAILER_HPP_
