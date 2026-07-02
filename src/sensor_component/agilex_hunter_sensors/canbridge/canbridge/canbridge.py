#!/usr/bin/env python3

import math

import rclpy
import time
from rclpy.node import Node
from can_msgs.msg import Frame
from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import ControlModeReport, SteeringReport, VelocityReport


class CANParser(Node):
    HUNTER_SYSTEM_STATE_ID = 0x211
    HUNTER_MOTION_STATE_ID = 0x221
    # Ackermann wheelbase [m]; must match vehicle_info wheel_base (0.65).
    WHEEL_BASE_M = 0.65
    # The chassis CAN speed field is NOT mm/s: calibrated 2026-07-01 with
    # tape-measured 16 ft straight drives. We calibrate to the MAP frame
    # (NDT distance / raw wheel distance): that is what the EKF fuses and
    # what the RL bridge observes, so map-consistent odometry is what kills
    # the dead-reckoning lag. 1.804 comes from the second, cleaner drive
    # (NDT 4.98 m / raw 2.76 m); the first drive gave 1.88. Applied
    # symmetrically to feedback (raw -> m/s) and commands (m/s -> CAN units)
    # so commanded speeds are executed 1:1.
    SPEED_SCALE = 1.804
    # The CAN steering field is NOT milli-rad of tire angle either: calibrated
    # 2026-07-01 with a constant-steering full circle (gyro/NDT agreed on 362
    # deg within 0.5%). Reported 0.637 "rad" drove a 1.46 m radius turn ->
    # true tire angle atan(0.65/1.46) = 0.419 rad. True = raw/1000 x 0.657.
    # Physical lock is ~±637 raw units = ±0.42 rad true, matching the
    # vehicle_info max_steer_angle (0.436) closely. Applied symmetrically:
    # feedback reports true tire rad (so RL obs[0] and heading_rate are
    # correct) and commands are converted from true tire rad to CAN units.
    STEER_SCALE = 0.657
    HUNTER_MODE_STANDBY = 0x00
    HUNTER_MODE_CAN_CONTROL = 0x01
    HUNTER_MODE_REMOTE = 0x02

    def __init__(self):
        super().__init__('controller_canbridge')

        self.declare_parameter('can_interface', 'can0')
        can_interface = self.get_parameter('can_interface').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            Control,
            '/control/command/control_cmd',
            self.controllerCallback,
            10)
        self.subscription = self.create_subscription(
            Frame,
            f'/CAN/{can_interface}/receive',
            self.vehicleFeedback,
            10)
        time.sleep(5)
        timer_period = 0.02  # 20ms
        self.timer = self.create_timer(timer_period, self.sendCanMessages)
        self.can_publisher = self.create_publisher(Frame, f'/CAN/{can_interface}/transmit', 10)

        self.VelocityPublisher = self.create_publisher(
            VelocityReport, '/vehicle/status/velocity_status', 10)
        self.SteeringPublisher = self.create_publisher(
            SteeringReport, '/vehicle/status/steering_status', 10)
        self.ControlModePublisher = self.create_publisher(
            ControlModeReport, '/vehicle/status/control_mode', 10)
        self.initFlag = False

        self.toSendSpeed = 0
        self.toSendTurn = 0

        self.speedFeedback = 0
        self.steeringFeedback = 0

        # clearing faults
        self.statusSetting = Frame()
        self.statusSetting.dlc = 1
        self.statusSetting.id = 0x441
        self.statusSetting.data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

        # put into CAN Control Mode
        self.CanMode = Frame()
        self.CanMode.dlc = 1
        self.CanMode.id = 0x421
        self.CanMode.data = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

        # turn off brakes
        self.brakesOff = Frame()
        self.brakesOff.dlc = 1
        self.brakesOff.id = 0x131
        self.brakesOff.data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

        # send Speed and Turning
        self.sendMovment = Frame()
        self.sendMovment.dlc = 8
        self.sendMovment.id = 0x111

    def vehicleFeedback(self, msg):
        toSendVelocity = VelocityReport()
        toSendSteering = SteeringReport()
        if msg.id == self.HUNTER_SYSTEM_STATE_ID:
            payload = msg.data

            control_mode = ControlModeReport()
            control_mode.stamp = self.get_clock().now().to_msg()

            chassis_mode = payload[1]
            if chassis_mode == self.HUNTER_MODE_CAN_CONTROL:
                control_mode.mode = ControlModeReport.AUTONOMOUS
            elif chassis_mode == self.HUNTER_MODE_REMOTE:
                control_mode.mode = ControlModeReport.MANUAL
            else:
                control_mode.mode = ControlModeReport.DISENGAGED

            self.ControlModePublisher.publish(control_mode)

        if msg.id == self.HUNTER_MOTION_STATE_ID:
            payload = msg.data

            self.speedFeedback = (payload[0] << 8) | payload[1]
            if self.speedFeedback >= 0x8000:
                self.speedFeedback -= 0x10000
            self.speedFeedback = self.speedFeedback / 1000 * self.SPEED_SCALE  # in m/s (map frame)

            self.steeringFeedback = (payload[6] << 8) | payload[7]
            if self.steeringFeedback >= 0x8000:
                self.steeringFeedback -= 0x10000
            self.steeringFeedback = self.steeringFeedback / 1000 * self.STEER_SCALE  # true tire rad
            toSendSteering.stamp = self.get_clock().now().to_msg()

            # self.get_logger().info(
            #     "speed FBK: %s m/s |  steer FBK: %s Rad"
            #     % (str(self.speedFeedback), str(self.steeringFeedback)))
            toSendVelocity.header.stamp = self.get_clock().now().to_msg()
            toSendVelocity.header.frame_id = 'base_link'
            toSendVelocity.longitudinal_velocity = self.speedFeedback
            # Kinematic yaw rate from steering feedback. Downstream twist
            # consumers (lidar deskew, concat motion compensation, gyro-odom
            # fallback) read this via vehicle_velocity_converter's angular.z;
            # leaving it 0 made them treat every turn as straight-line motion.
            toSendVelocity.heading_rate = (
                self.speedFeedback * math.tan(self.steeringFeedback) / self.WHEEL_BASE_M)
            toSendSteering.steering_tire_angle = self.steeringFeedback
            self.SteeringPublisher.publish(toSendSteering)
            self.VelocityPublisher.publish(toSendVelocity)

    def sendCanMessages(self):
        if not self.initFlag:
            self.can_publisher.publish(self.statusSetting)
            self.can_publisher.publish(self.CanMode)
            self.can_publisher.publish(self.brakesOff)
            self.initFlag = True
        else:
            self.can_publisher.publish(self.sendMovment)

    # Hard safety cap in map-frame m/s: robot must never drive faster than this
    # in either direction. 0.6 allows the RL models' 0.5 m/s operating speed
    # with margin. (Before the SPEED_SCALE calibration this was 0.2 in CHASSIS
    # units, which actually executed ~0.38 map-m/s — every autonomous run was
    # silently 25% slower than the policy expected.)
    MAX_SPEED_MPS = 0.6

    def controllerCallback(self, msg):
        velocity = msg.longitudinal.velocity          # m/s
        steering_angle = msg.lateral.steering_tire_angle  # radians

        if velocity > self.MAX_SPEED_MPS:
            velocity = self.MAX_SPEED_MPS
        elif velocity < -self.MAX_SPEED_MPS:
            velocity = -self.MAX_SPEED_MPS

        max_speed = 333  # CAN units = MAX_SPEED_MPS/SPEED_SCALE*1000 (hardware max ±1500)
        self.toSendSpeed = min(abs(int(velocity * 1000 / self.SPEED_SCALE)), max_speed)
        if velocity < 0:
            self.toSendSpeed = -self.toSendSpeed

        # Convert commanded true tire angle [rad] to CAN units via the
        # calibrated STEER_SCALE (raw units = rad/0.657*1000 ~= 1522x). The old
        # empirical 1320x under-delivered: a full 0.436 rad command produced
        # only ~0.38 rad true tire angle. Cap at the observed physical lock
        # (±637 raw units, RC full-lock feedback); firmware clamps beyond.
        max_turn = 640  # CAN units ~= physical lock (0.42 rad true)
        self.toSendTurn = int(steering_angle * 1000 / self.STEER_SCALE)
        if self.toSendTurn > max_turn:
            self.toSendTurn = max_turn
        elif self.toSendTurn < -max_turn:
            self.toSendTurn = -max_turn

        lower_speed_byte = self.toSendSpeed & 0xFF
        higher_speed_byte = (self.toSendSpeed >> 8) & 0xFF
        lower_turn_byte = self.toSendTurn & 0xFF
        higher_turn_byte = (self.toSendTurn >> 8) & 0xFF
        self.sendMovment.data = [
            higher_speed_byte, lower_speed_byte,
            0x00, 0x00, 0x00, 0x00,
            higher_turn_byte, lower_turn_byte,
        ]


def main(args=None):
    rclpy.init(args=args)
    can_parser = CANParser()
    rclpy.spin(can_parser)
    can_parser.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
