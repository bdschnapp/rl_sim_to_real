#!/usr/bin/env python3

import rclpy
import time
from rclpy.node import Node
from can_msgs.msg import Frame
from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import ControlModeReport, SteeringReport, VelocityReport


class CANParser(Node):
    HUNTER_SYSTEM_STATE_ID = 0x211
    HUNTER_MOTION_STATE_ID = 0x221
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
            self.speedFeedback /= 1000  # in m/s

            self.steeringFeedback = (payload[6] << 8) | payload[7]
            if self.steeringFeedback >= 0x8000:
                self.steeringFeedback -= 0x10000
            self.steeringFeedback /= 1000  # in Rad
            toSendSteering.stamp = self.get_clock().now().to_msg()

            # self.get_logger().info(
            #     "speed FBK: %s m/s |  steer FBK: %s Rad"
            #     % (str(self.speedFeedback), str(self.steeringFeedback)))
            toSendVelocity.header.stamp = self.get_clock().now().to_msg()
            toSendVelocity.header.frame_id = 'base_link'
            toSendVelocity.longitudinal_velocity = self.speedFeedback
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

    # Hard safety cap: robot must never drive faster than this in either direction.
    MAX_SPEED_MPS = 0.2

    def controllerCallback(self, msg):
        velocity = msg.longitudinal.velocity          # m/s
        steering_angle = msg.lateral.steering_tire_angle  # radians

        if velocity > self.MAX_SPEED_MPS:
            velocity = self.MAX_SPEED_MPS
        elif velocity < -self.MAX_SPEED_MPS:
            velocity = -self.MAX_SPEED_MPS

        max_speed = 200  # software speed limit (CAN units; hardware max is ±1500)
        self.toSendSpeed = min(abs(int(velocity * 1000)), max_speed)
        if velocity < 0:
            self.toSendSpeed = -self.toSendSpeed

        # Map full Autoware steering range (±0.436 rad ≈ ±25°) to full hardware
        # range (±576 CAN units). Previously used 576x which only reached ~44%
        # of hardware steering at max command. Clamp for safety.
        max_turn = 576  # CAN units, hardware limit
        self.toSendTurn = int(steering_angle * 1320)
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
