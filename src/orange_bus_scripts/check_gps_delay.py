#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from statistics import mean, pstdev

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from rosidl_runtime_py.utilities import get_message

# ----- Constants: EXACT match to your C++ -----
K_TO_NANO = int(1e9)
K_MILLI_TO_NANO = int(1e6)
K_GPS_SECONDS_IN_WEEK = 60 * 60 * 24 * 7  # 604800
K_UNIX_TIME_OFFSET = 315_964_782_000_000_000  # ns

TOPIC = "/applanix/lvx_client/gsof/ins_solution_49"
MSG_TYPE = "applanix_msgs/msg/NavigationSolutionGsof49"  # ensure applanix_msgs is installed


def gps_to_ros_time_ns(week: int, time_ms: int) -> int:
    """Exact C++ mirror: week*604800*1e9 + time_ms*1e6 + 315964782000000000"""
    return (K_GPS_SECONDS_IN_WEEK * int(week) * K_TO_NANO) + \
           (int(time_ms) * K_MILLI_TO_NANO) + \
           K_UNIX_TIME_OFFSET


class GpsDriftMonitor(Node):
    def __init__(self):
        super().__init__("gps_drift_monitor")

        # Resolve message type
        Msg = get_message(MSG_TYPE)

        # QoS: reliable, keep last 10
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(Msg, TOPIC, self.cb, qos)

        # Collect deltas (ns) for the 10-second window
        self._deltas_ns = []
        self._skipped = 0
        self._received = 0

        # Print every 10 seconds
        self.create_timer(10.0, self.report)

        self.get_logger().info(
            f"Listening on {TOPIC}; will report every 10s. Comparing receive_time_ns − gps_time_ns."
        )

    def cb(self, msg):
        now_ns = self.get_clock().now().nanoseconds  # receive time (node clock, typically system/steady)
        try:
            # Extract (week, time_ms) from msg.gps_time
            gt = msg.gps_time
            week = int(gt.week)
            if hasattr(gt, "time_msec"):
                time_ms = int(gt.time_msec)
            elif hasattr(gt, "time"):
                time_ms = int(gt.time)  # many Applanix variants store ms here
            else:
                raise AttributeError("gps_time has neither time_msec nor time")
            gps_ns = gps_to_ros_time_ns(week, time_ms)
            delta = now_ns - gps_ns
            self._deltas_ns.append(delta)
        except Exception as e:
            self._skipped += 1
        finally:
            self._received += 1

    def report(self):
        n = len(self._deltas_ns)
        if n == 0:
            self.get_logger().warn(
                f"No samples in last 10s. received={self._received}, skipped={self._skipped}"
            )
            self._skipped = 0
            self._received = 0
            return

        dmin = min(self._deltas_ns)
        dmax = max(self._deltas_ns)
        dmean = mean(self._deltas_ns)
        dstdev = pstdev(self._deltas_ns) if n > 1 else 0.0
        drange = dmax - dmin

        def to_ms(x): return x / 1e6

        self.get_logger().info(
            "Δ = receive_time_ns − gps_time_ns over last ~10s\n"
            f"  samples: {n}, skipped: {self._skipped}\n"
            f"  min   : {dmin} ns  ({to_ms(dmin):.3f} ms)\n"
            f"  max   : {dmax} ns  ({to_ms(dmax):.3f} ms)\n"
            f"  mean  : {dmean:.1f} ns  ({to_ms(dmean):.3f} ms)\n"
            f"  std   : {dstdev:.1f} ns  ({to_ms(dstdev):.3f} ms)\n"
            f"  range : {drange} ns  ({to_ms(drange):.3f} ms)"
        )

        # Reset window
        self._deltas_ns.clear()
        self._skipped = 0
        self._received = 0


def main():
    rclpy.init()
    node = GpsDriftMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
