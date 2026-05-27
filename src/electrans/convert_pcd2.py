#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField

# Map ROS PointField datatypes to numpy (little-endian)
PF2NP = {
    PointField.INT8:    '<i1',
    PointField.UINT8:   '<u1',
    PointField.INT16:   '<i2',
    PointField.UINT16:  '<u2',
    PointField.INT32:   '<i4',
    PointField.UINT32:  '<u4',
    PointField.FLOAT32: '<f4',
    PointField.FLOAT64: '<f8',
}

class PCDSchemaRelay(Node):
    def __init__(self):
        super().__init__('pcd_schema_relay')

        # Parameters
        self.declare_parameter('in_topic', '/rslidar_points_front_raw')
        self.declare_parameter('out_topic', '/sensing/lidar/front/rslidar_points_front')
        self.declare_parameter('intensity_scale', 1.0)    # float->uint8 scaling
        self.declare_parameter('return_type', 0)          # constant uint8
        self.declare_parameter('ts_mode', 'absolute')     # 'absolute' or 'relative'
        self.declare_parameter('ts_scale', 1.0)           # multiply input timestamp by this to get seconds

        in_topic = self.get_parameter('in_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('out_topic').get_parameter_value().string_value

        qos_sub = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,   # was BEST_EFFORT
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        qos_pub = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,   # was BEST_EFFORT
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.sub = self.create_subscription(PointCloud2, in_topic, self.cb, qos_sub)
        self.pub = self.create_publisher(PointCloud2, out_topic, qos_pub)

        # Prebuild output fields
        self.out_fields = [
            PointField(name='x',           offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',           offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',           offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity',   offset=12, datatype=PointField.UINT8,   count=1),
            PointField(name='return_type', offset=13, datatype=PointField.UINT8,   count=1),
            PointField(name='channel',     offset=14, datatype=PointField.UINT16,  count=1),
            PointField(name='azimuth',     offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='elevation',   offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='distance',    offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name='time',        offset=28, datatype=PointField.UINT32,  count=1),
        ]
        self.point_step = 32

        self.get_logger().info(f"Relaying {in_topic} → {out_topic}")

    def _field_map(self, msg: PointCloud2):
        """Return dict: name -> (offset, numpy dtype)"""
        fmap = {}
        for f in msg.fields:
            if f.datatype not in PF2NP:
                continue
            fmap[f.name] = (f.offset, PF2NP[f.datatype])
        return fmap

    def _view_field(self, raw: memoryview, n: int, step: int, offset: int, np_dtype: str):
        """Strided view into raw bytes for one field."""
        # numpy expects a Python buffer; memoryview is fine
        return np.ndarray(shape=(n,), dtype=np_dtype, buffer=raw, offset=offset, strides=(step,))

    def cb(self, msg: PointCloud2):
        fmap = self._field_map(msg)
        required = ['x','y','z','intensity','ring','timestamp']
        for k in required:
            if k not in fmap:
                self.get_logger().warn(f"Missing field '{k}' in incoming PointCloud2; skipping frame.")
                return

        # Counts
        total_points = (msg.row_step * msg.height) // msg.point_step
        if total_points == 0:
            return

        raw = memoryview(bytes(msg.data))  # copy once to ensure contiguous buffer for numpy
        step = msg.point_step

        # Strided views
        off_x,  dt_x  = fmap['x']
        off_y,  dt_y  = fmap['y']
        off_z,  dt_z  = fmap['z']
        off_i,  dt_i  = fmap['intensity']
        off_rn, dt_rn = fmap['ring']
        off_ts, dt_ts = fmap['timestamp']

        x  = self._view_field(raw, total_points, step, off_x,  dt_x).astype('<f4', copy=False)
        y  = self._view_field(raw, total_points, step, off_y,  dt_y).astype('<f4', copy=False)
        z  = self._view_field(raw, total_points, step, off_z,  dt_z).astype('<f4', copy=False)
        iF = self._view_field(raw, total_points, step, off_i,  dt_i).astype('<f4', copy=False)
        ring = self._view_field(raw, total_points, step, off_rn, dt_rn).astype('<u2', copy=False)
        tsF = self._view_field(raw, total_points, step, off_ts, dt_ts).astype('<f8', copy=False)

        # Derived fields
        # Intensity -> uint8
        scale = float(self.get_parameter('intensity_scale').get_parameter_value().double_value)
        iU8 = np.clip(iF * scale, 0.0, 255.0).astype('<u1', copy=False)

        # Return type constant
        ret_c = int(self.get_parameter('return_type').get_parameter_value().integer_value)
        retU8 = np.full_like(iU8, ret_c, dtype='<u1')

        # Channel from ring
        chanU16 = ring  # already <u2

        # D = hypot(X, Y, Z)
        D = np.sqrt(x*x + y*y + z*z, dtype=np.float32)
        # A = atan2(Y, X)
        A = np.arctan2(y, x, dtype=np.float32)
        # E = atan2(Z, D)   (your spec)
        E = np.arctan2(z.astype(np.float32, copy=False), D, dtype=np.float32)

        # T = nanoseconds since header stamp
        ts_mode = self.get_parameter('ts_mode').get_parameter_value().string_value
        ts_scale = float(self.get_parameter('ts_scale').get_parameter_value().double_value)

        # header time in seconds
        hdr_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if ts_mode == 'relative':
            # timestamp already relative seconds from header
            rel_sec = tsF * ts_scale
        else:
            # absolute seconds; subtract header
            rel_sec = (tsF * ts_scale) - hdr_sec

        rel_ns = np.clip(np.round(rel_sec * 1e9), 0, (1<<32)-1).astype('<u4', copy=False)

        # Pack output (N x 32 bytes)
        N = total_points
        buf = np.zeros((N, self.point_step), dtype=np.uint8)

        f32v = buf.view('<f4')
        u8v  = buf.view('<u1')
        u16v = buf.view('<u2')
        u32v = buf.view('<u4')

        # floats
        f32v[:,0] = x
        f32v[:,1] = y
        f32v[:,2] = z
        f32v[:,4] = A
        f32v[:,5] = E
        f32v[:,6] = D
        # u8/16/32 at byte offsets 12/13/14/28
        u8v[:,12]  = iU8
        u8v[:,13]  = retU8
        u16v[:,7]  = chanU16     # 7 * 2 = 14 bytes
        u32v[:,7]  = rel_ns      # 7 * 4 = 28 bytes

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width  = N
        out.fields = self.out_fields
        out.is_bigendian = False
        out.point_step = self.point_step
        out.row_step = self.point_step * N
        out.is_dense = True
        out.data = buf.tobytes()

        self.pub.publish(out)

def main():
    rclpy.init()
    node = PCDSchemaRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
