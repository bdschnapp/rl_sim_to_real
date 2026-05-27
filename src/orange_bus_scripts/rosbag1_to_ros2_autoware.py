from builtin_interfaces.msg import Time
import struct
import sys
import os
import yaml
import numpy as np
from rosbags.rosbag1 import Reader as Ros1Reader
from rosbags.rosbag2 import Writer as Ros2Writer
from rosbags.serde import deserialize_cdr, serialize_cdr, ros1_to_cdr, cdr_to_ros1
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from importlib import import_module
from std_msgs.msg import Header
import tf_transformations
from nav_msgs.msg import Odometry
from geometry_msgs.msg import AccelWithCovarianceStamped
from sensor_msgs.msg import PointCloud2, PointField
from geodesy import utm
from scipy.spatial.transform import Rotation as R

from rosbags.highlevel.anyreader import AnyReader

# autoware msgs
from autoware_vehicle_msgs.msg import ControlModeReport, GearReport, SteeringReport, VelocityReport


# Define ROS1/ROS2 CameraInfo
CAMERAINFO_DEF = '''
std_msgs/msg/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/msg/RegionOfInterest roi
'''

# PointXYZIRCAEDT fields definitions
FIELDS = [
    PointField(name='x',          offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y',          offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z',          offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='intensity',  offset=12, datatype=PointField.UINT8,   count=1),
    PointField(name='return_type',offset=13, datatype=PointField.UINT8,   count=1),
    PointField(name='channel',    offset=14, datatype=PointField.UINT16,  count=1),
    PointField(name='azimuth',    offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name='elevation',  offset=20, datatype=PointField.FLOAT32, count=1),
    PointField(name='distance',   offset=24, datatype=PointField.FLOAT32, count=1),
    PointField(name='time_stamp', offset=28, datatype=PointField.UINT32,  count=1),
]
POINT_STEP = 32  # total bytes


ts = get_typestore(Stores.EMPTY)
ts.register(get_types_from_msg(CAMERAINFO_DEF, 'sensor_msgs/msg/CameraInfo'))
CameraInfoUpper = ts.types['sensor_msgs/msg/CameraInfo']


typestore_ros1 = get_typestore(Stores.ROS1_NOETIC)   # or the actual ROS1 version of your bag
typestore_ros2 = get_typestore(Stores.ROS2_HUMBLE)

# Register the autoware vehicle msgs
# Path to Autoware `.msg` files

LATEST_HEADING_RATE = 0.0  # Default value for heading rate if not provided
LATEST_VEHICLE_SPEED = 0.0  # Default value for vehicle speed if not provided
LATEST_VEHICLE_SPEED_LATERAL = 0.0  # Default value for lateral speed if not provided

autoware_msg_dir = '/home/minghao/autoware/src/core/autoware_msgs/autoware_vehicle_msgs/msg'
# List of required messages
msgs_to_register = [
    'ControlModeReport.msg',
    'GearReport.msg',
    'SteeringReport.msg',
    'VelocityReport.msg',
]
# Register each message type
for msg_file in msgs_to_register:
    msg_path = os.path.join(autoware_msg_dir, msg_file)
    if os.path.exists(msg_path):
        with open(msg_path, 'r') as f:
            msg_def = f.read()
        # Register the message definition
        typestore_ros2.register(get_types_from_msg(
            msg_def, f'autoware_vehicle_msgs/msg/{msg_file[:-4]}'))
    else:
        print(f"Warning: {msg_path} does not exist. Skipping registration.")



# --- PointCloud2 helper ---
def ros1_pointcloud_to_ros2(msg_ros1):
    # --- 1) build a numpy dtype for the incoming ROS1 cloud ---
    # we know your ROS1 PointXYZIRT is laid out as:
    #   x:float32 @0, y:float32 @4, z:float32 @8,
    #   intensity:uint8 @16, ring:uint16 @18, timestamp:float64 @24
    dtype_in = np.dtype({
      'names':   ['x','y','z','intensity','ring','timestamp'],
      'formats': ['<f4','<f4','<f4','<u1','<u2','<f8'],
      'offsets': [0,    4,    8,    16,         18,        24      ],
      'itemsize': msg_ros1.point_step
    })

    # view the raw data as that structured array
    arr_in = np.frombuffer(msg_ros1.data, dtype=dtype_in, count=msg_ros1.width * msg_ros1.height)

    # drop any totally invalid xyz
    valid = np.isfinite(arr_in['x']) & np.isfinite(arr_in['y']) & np.isfinite(arr_in['z'])
    arr_in = arr_in[valid]

    # find the earliest per-point timestamp for the header
    t0 = float(arr_in['timestamp'].min())

    # --- 2) build the NumPy array for the Autoware layout ---
    dtype_out = np.dtype({
      'names':   ['x','y','z','intensity','return_type','channel','azimuth','elevation','distance','time_stamp'],
      'formats': ['f4','f4','f4','u1','u1','u2','f4','f4','f4','u4'],
      'offsets': [0,   4,   8,   12,          13,         14,        16,         20,         24,           28      ],
      'itemsize': POINT_STEP
    })
    arr_out = np.zeros(arr_in.shape, dtype=dtype_out)

    # copy the basics
    arr_out['x']         = arr_in['x']
    arr_out['y']         = arr_in['y']
    arr_out['z']         = arr_in['z']
    arr_out['intensity'] = arr_in['intensity']
    arr_out['return_type']= 0                         # you said always zero
    arr_out['channel']   = arr_in['ring']

    # derived: distance, azimuth, elevation
    xy = np.vstack((arr_in['x'], arr_in['y'])).T
    dists = np.linalg.norm(np.vstack((arr_in['x'],arr_in['y'],arr_in['z'])).T, axis=1)
    arr_out['distance']  = dists.astype('f4')
    arr_out['azimuth']   = np.arctan2(arr_in['y'], arr_in['x']).astype('f4')
    arr_out['elevation'] = np.arctan2(arr_in['z'], dists).astype('f4')

    # timestamp offset in **nanoseconds** since t0
    offs = ((arr_in['timestamp'] - t0) * 1e9).astype('u4')
    arr_out['time_stamp'] = offs

    # --- 3) pack into a ROS2 PointCloud2 ---
    pc2 = PointCloud2(
        header=Header(
            stamp=make_ros2_time(t0),
            frame_id=msg_ros1.header.frame_id
        ),
        height=1,
        width=arr_out.shape[0],
        fields=FIELDS,
        is_bigendian=False,
        point_step=POINT_STEP,
      row_step=POINT_STEP * arr_out.shape[0],
      is_dense=True,
      data = []
    )
    pc2._data = np.frombuffer(arr_out.tobytes(), dtype=np.uint8)

    return pc2


# --- YAML transform helper ---

def tf_to_matrix(translation, rpy):
    rot_mat = R.from_euler('xyz', rpy).as_matrix()
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rot_mat
    tf_mat[:3, 3] = translation
    return tf_mat

def load_transform_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    tf = data['sensor_kit_base_link']['applanix']
    trans = np.array([tf['x'], tf['y'], tf['z']])
    # Roll, pitch, yaw are in radians (check your YAML, but appears so)
    roll, pitch, yaw = tf['roll'], tf['pitch'], tf['yaw']
    transform = tf_to_matrix(
        translation=trans,
        rpy=[roll, pitch, yaw]
    )
    return np.linalg.inv(transform)


# --- Message conversion helpers ---

def make_ros2_time(ts):
    """
    Converts a float (seconds), int (seconds), or int (nanoseconds, if very large) into a ROS2 Time msg.
    """
    # Heuristics: if ts > 1e12, it's nanoseconds (ROS1/rosbags); else seconds
    if isinstance(ts, float):
        secs = int(ts)
        nsecs = int((ts - secs) * 1e9)
    elif isinstance(ts, int):
        if ts > 1e12:  # nanoseconds
            secs = ts // 10**9
            nsecs = ts % 10**9
        else:          # seconds
            secs = ts
            nsecs = 0
    else:
        raise ValueError("Unsupported timestamp type for make_ros2_time")

    return Time(sec=secs, nanosec=nsecs)

def parse_navsol_full_raw(msg_bytes):
    fmt = (
        '<'   # little-endian
        # TimeDistance (5)
        'd'   # time1
        'd'   # time2
        'd'   # distance
        'B'   # time_types
        'B'   # distance_type
        # LLA (3)
        'd'   # latitude
        'd'   # longitude
        'd'   # altitude
        # NED (3)
        'f'   # north
        'f'   # east
        'f'   # down
        # Remaining fields
        'd'   # roll
        'd'   # pitch
        'd'   # heading
        'd'   # wander_angle
        'f'   # track_angle
        'f'   # total_speed
        'f'   # ang_rate_long
        'f'   # ang_rate_trans
        'f'   # ang_rate_down
        'f'   # acc_long
        'f'   # acc_trans
        'f'   # acc_down
        'B'   # alignment_status
        'B'   # gnss_status
        'f'   # north_pos_rms
        'f'   # east_pos_rms
        'f'   # down_pos_rms
        'f'   # roll_rms
        'f'   # pitch_rms
        'f'   # heading_rms
        'f'   # north_vel_rms
        'f'   # east_vel_rms
        'f'   # down_vel_rms
    )
    # Calculate expected size and check if msg_bytes has enough length
    expected_len = struct.calcsize(fmt)
    if len(msg_bytes) < expected_len:
        raise ValueError(f"Message too short: {len(msg_bytes)} < {expected_len}")

    values = struct.unpack(fmt, msg_bytes[:expected_len])
    keys = [
        # TimeDistance
        'td_time1', 'td_time2', 'td_distance', 'td_time_types', 'td_distance_type',
        # LLA
        'latitude', 'longitude', 'altitude',
        # NED velocity
        'vel_north', 'vel_east', 'vel_down',
        # Remaining fields
        'roll', 'pitch', 'heading', 'wander_angle',
        'track_angle', 'total_speed',
        'ang_rate_long', 'ang_rate_trans', 'ang_rate_down',
        'acc_long', 'acc_trans', 'acc_down',
        'alignment_status', 'gnss_status',
        'north_pos_rms', 'east_pos_rms', 'down_pos_rms',
        'roll_rms', 'pitch_rms', 'heading_rms',
        'north_vel_rms', 'east_vel_rms', 'down_vel_rms'
    ]
    return dict(zip(keys, values))


def parse_watono_chasis_sensor(rawdata: bytes) -> dict:
    offset = 0

    # --- Parse Header (stamp.sec, stamp.nanosec, frame_id) ---
    seq, stamp_sec, stamp_nsec = struct.unpack_from('<III', rawdata, offset)
    offset += 12

    # Parse frame_id string (ROS1 strings are [length(uint32)] + [char * length])
    (frame_id_len,) = struct.unpack_from('<I', rawdata, offset)
    offset += 4
    frame_id = rawdata[offset:offset + frame_id_len].decode('utf-8')
    offset += frame_id_len

    header = {
        'stamp': {
            'sec': stamp_sec,
            'nanosec': stamp_nsec
        },
        'frame_id': frame_id
    }

    # --- Parse remaining fields (custom payload) ---
    fmt = '<I6b3f'
    expected_size = struct.calcsize(fmt)
    if len(rawdata) - offset < expected_size:
        raise ValueError(f"Not enough bytes: got {len(rawdata) - offset}, expected {expected_size}")

    unpacked = struct.unpack_from(fmt, rawdata, offset)
    keys = [
        'mcu_rolling_count',
        'mcu_ready',
        'mcu_auto_engaged',
        'mcu_auto_button_pressed',
        'door_open',
        'door_close',
        'parking_brake',
        'veh_steering_rear',
        'veh_steering_front',
        'veh_speed'
    ]

    msg = dict(zip(keys, unpacked))
    msg['header'] = header
    return msg


def decode_watono_chasis_sensor(rawdata):
    msg = parse_watono_chasis_sensor(rawdata)

    stamp = Time(
        sec=msg['header']['stamp']['sec'],
        nanosec=msg['header']['stamp']['nanosec']
    )

    # ControlModeReport
    control_mode = ControlModeReport()
    control_mode.stamp = stamp
    control_mode.mode = (
        ControlModeReport.AUTONOMOUS if msg['mcu_auto_engaged']
        else ControlModeReport.MANUAL
    )

    # GearReport (basic logic based on brake; customize as needed)
    gear = GearReport()
    gear.stamp = stamp
    # gear.report = GearReport.PARK if msg['parking_brake'] else GearReport.DRIVE
    gear.report = GearReport.DRIVE #TODO: Implement actual gear logic

    # SteeringReport
    steering = SteeringReport()
    steering.stamp = stamp
    steering.steering_tire_angle = msg['veh_steering_front']

    # VelocityReport
    velocity = VelocityReport()
    velocity.header.stamp = stamp
    velocity.header.frame_id = 'base_link'
    # velocity.longitudinal_velocity = msg['veh_speed'] / 3.6  # kph → m/s
    velocity.longitudinal_velocity = LATEST_VEHICLE_SPEED  # Use the global variable
    velocity.lateral_velocity = LATEST_VEHICLE_SPEED_LATERAL  # Use the global variable
    velocity.heading_rate = LATEST_HEADING_RATE  # Use the global variable

    return control_mode, gear, steering, velocity



def fill_odometry_and_acceleration(msg, header_stamp, base_link_to_applanix_transform):
    global LATEST_HEADING_RATE, LATEST_VEHICLE_SPEED, LATEST_VEHICLE_SPEED_LATERAL
    # Our MGRS projection is 17TNJ
    REF_EASTING = 500000.0
    REF_NORTHING = 4800000.0
    REF_ALTITUDE = 0.0  # MGRS altitude is not used, so we set it to zero

    ## return None for invalid messages
    if msg['alignment_status'] != 3:
        print(f"Invalid alignment status: {msg['alignment_status']}")
        return None, None

    odom = Odometry()
    odom.header.stamp = header_stamp
    odom.header.frame_id = "map"
    odom.child_frame_id = "base_link"

    # convert MGRS to UTM coordinates
    utm_coords = utm.fromLatLong(msg['latitude'], msg['longitude'])
    applanix_translation_x = utm_coords.easting - REF_EASTING
    applanix_translation_y = utm_coords.northing - REF_NORTHING
    applanix_translation_z = msg['altitude'] - REF_ALTITUDE
    applanix_orientation_roll = (msg['roll']+180.0) * np.pi / 180.0  # Convert to radians
    applanix_orientation_pitch = -msg['pitch'] * np.pi / 180.0
    applanix_orientation_yaw = (90-msg['heading']) * np.pi / 180.0  # Convert to radians
    # get the transform matrix for the applanix to map
    applanix_transform = tf_to_matrix(
        translation=[applanix_translation_x, applanix_translation_y, applanix_translation_z],
        rpy=[applanix_orientation_roll, applanix_orientation_pitch, applanix_orientation_yaw]
    )
    # get the transform matrix for base_link to map
    base_link_transform = applanix_transform @ base_link_to_applanix_transform
    # fill translation
    odom.pose.pose.position.x = base_link_transform[0, 3]
    odom.pose.pose.position.y = base_link_transform[1, 3]
    odom.pose.pose.position.z = base_link_transform[2, 3]
    # convert angle
    quat = R.from_matrix(base_link_transform[:3, :3]).as_quat()
    odom.pose.pose.orientation.x = quat[0]
    odom.pose.pose.orientation.y = quat[1]
    odom.pose.pose.orientation.z = quat[2]
    odom.pose.pose.orientation.w = quat[3]
    # convert velocity
    vel_north = msg['vel_north']
    vel_east = msg['vel_east']
    vel_up = -msg['vel_down']  # Down is negative in ROS
    # Convert to base_link frame using the base_link_transform rotation
    vel_base_link = np.linalg.inv(base_link_transform[:3, :3]) @ np.array([vel_east, vel_north, vel_up])
    odom.twist.twist.linear.x = vel_base_link[0]
    odom.twist.twist.linear.y = vel_base_link[1]
    odom.twist.twist.linear.z = vel_base_link[2]
    # Fill angular velocity (rad/s)
    yaw_rate = -msg['ang_rate_down'] * np.pi / 180.0
    roll_rate = msg['ang_rate_long'] * np.pi / 180.0
    pitch_rate = -msg['ang_rate_trans'] * np.pi / 180.0
    odom.twist.twist.angular.x = roll_rate
    odom.twist.twist.angular.y = pitch_rate
    odom.twist.twist.angular.z = yaw_rate
    # Fill covariance
    north_pos_rms = msg['north_pos_rms']
    east_pos_rms = msg['east_pos_rms']
    down_pos_rms = msg['down_pos_rms']
    north_vel_rms = msg['north_vel_rms']
    east_vel_rms = msg['east_vel_rms']
    down_vel_rms = msg['down_vel_rms']
    roll_rms = msg['roll_rms']
    pitch_rms = msg['pitch_rms']
    heading_rms = msg['heading_rms']
    odom.pose.covariance = np.float64([
        north_pos_rms**2, 0, 0, 0, 0, 0,
        0, east_pos_rms**2, 0, 0, 0, 0,
        0, 0, down_pos_rms**2, 0, 0, 0,
        0, 0, 0, roll_rms**2, 0, 0,
        0, 0, 0, 0, pitch_rms**2, 0,
        0, 0, 0, 0, 0, heading_rms**2
    ])
    odom.twist.covariance = np.float64([
        north_vel_rms**2, 0, 0, 0, 0, 0,
        0, east_vel_rms**2, 0, 0, 0, 0,
        0, 0, down_vel_rms**2, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    ])

    # Generate the Acceleration message
    acc = AccelWithCovarianceStamped()
    acc.header.stamp = header_stamp
    acc.header.frame_id = "base_link"
    # Fill linear acceleration (m/s^2)
    acc.accel.accel.linear.x = msg['acc_long']
    acc.accel.accel.linear.y = -msg['acc_trans']
    acc.accel.accel.linear.z = -msg['acc_down']

    # Save the LATEST_HEADING_RATE
    LATEST_HEADING_RATE = yaw_rate
    # Save the LATEST_VEHICLE_SPEED
    LATEST_VEHICLE_SPEED = vel_base_link[0]
    # Save the LATEST_VEHICLE_SPEED_LATERAL
    LATEST_VEHICLE_SPEED_LATERAL = vel_base_link[1]
    return odom, acc



# --- List of direct-copy topics ---
copy_topics_types = {
    "/pylon_camera_node_back/camera_info": 'sensor_msgs/msg/CameraInfo',
    "/pylon_camera_node_back/image_rect/compressed": 'sensor_msgs/msg/CompressedImage',
    "/pylon_camera_node_backright/camera_info": 'sensor_msgs/msg/CameraInfo',
    "/pylon_camera_node_backright/image_rect/compressed": 'sensor_msgs/msg/CompressedImage',
    "/pylon_camera_node_center/camera_info": 'sensor_msgs/msg/CameraInfo',
    "/pylon_camera_node_center/image_rect/compressed": 'sensor_msgs/msg/CompressedImage',
    "/pylon_camera_node_left/camera_info": 'sensor_msgs/msg/CameraInfo',
    "/pylon_camera_node_left/image_rect/compressed": 'sensor_msgs/msg/CompressedImage',
    "/pylon_camera_node_merging/camera_info": 'sensor_msgs/msg/CameraInfo',
    "/pylon_camera_node_merging/image_rect/compressed": 'sensor_msgs/msg/CompressedImage',
    "/pylon_camera_node_right/camera_info": 'sensor_msgs/msg/CameraInfo',
    "/pylon_camera_node_right/image_rect/compressed": 'sensor_msgs/msg/CompressedImage',
    "/rslidar_points_BP_F": 'sensor_msgs/msg/PointCloud2',
    "/rslidar_points_BP_R": 'sensor_msgs/msg/PointCloud2',
    "/rslidar_points_front": 'sensor_msgs/msg/PointCloud2',
}


def main(yaml_path, bag_paths, output_bag_dir):
    base_link_to_applanix_transform = load_transform_from_yaml(yaml_path)
    print(f"Loaded transform base_link<-applanix: {base_link_to_applanix_transform}")

    # Prepare ROS2 bag writer
    # remove output directory if exists
    if os.path.exists(output_bag_dir):
        import shutil
        shutil.rmtree(output_bag_dir)
    with Ros2Writer(output_bag_dir) as writer:
        # Register output topics and types for ROS2
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import AccelWithCovarianceStamped
        kinematic_conn = writer.add_connection(
            '/localization/kinematic_state', 'nav_msgs/msg/Odometry')
        accel_conn = writer.add_connection('/localization/acceleration',
                                        'geometry_msgs/msg/AccelWithCovarianceStamped')
        # vehicle status messages
        control_mode_conn = writer.add_connection(
            '/vehicle/status/control_mode', 'autoware_vehicle_msgs/msg/ControlModeReport', typestore=typestore_ros2)
        gear_status_conn = writer.add_connection(
            '/vehicle/status/gear_status', 'autoware_vehicle_msgs/msg/GearReport', typestore=typestore_ros2)
        steering_conn = writer.add_connection(
            '/vehicle/status/steering_status', 'autoware_vehicle_msgs/msg/SteeringReport', typestore=typestore_ros2)
        velocity_conn = writer.add_connection(
            '/vehicle/status/velocity_status', 'autoware_vehicle_msgs/msg/VelocityReport', typestore=typestore_ros2)

        topic_to_conn = {}
        for topic, msgtype in copy_topics_types.items():
            topic_to_conn[topic] = writer.add_connection(topic, msgtype)

        for bagfile in bag_paths:
            print(f"Processing bag: {bagfile}")
            with Ros1Reader(bagfile) as reader:
                for connection, timestamp, rawdata in reader.messages():
                    # print(f"Processing topic: {connection.topic} at {timestamp}")
                    if connection.topic == '/lvx_client_node/gsof/full':
                        # Convert custom message
                        msg = parse_navsol_full_raw(rawdata)
                        # Convert ROS timestamp (genpy.Time) to nanosec for ROS2
                        header_stamp = msg.header.stamp if hasattr(msg, 'header') else None
                        # Fallback to bag timestamp if missing
                        if header_stamp is None:
                            # nanosec (rosbags expects int)
                            header_stamp = make_ros2_time(timestamp)
                        odom, accel = fill_odometry_and_acceleration(msg, header_stamp, base_link_to_applanix_transform)
                        # print(writer.connections)
                        if odom is not None:
                            writer.write(kinematic_conn, timestamp,
                                     typestore_ros2.serialize_cdr(odom, 'nav_msgs/msg/Odometry'))
                        if accel is not None:
                            writer.write(accel_conn, timestamp,
                                     typestore_ros2.serialize_cdr(accel, 'geometry_msgs/msg/AccelWithCovarianceStamped'))
                    elif connection.topic in copy_topics_types:
                        try:
                            msgtype = copy_topics_types[connection.topic]
                            if msgtype == 'sensor_msgs/msg/CameraInfo':
                                # Deserialize ROS1, map fields, and serialize for ROS2
                                msg_ros1 = typestore_ros1.deserialize_ros1(rawdata, connection.msgtype)
                                msg_ros2 = CameraInfoUpper(
                                    header=msg_ros1.header,
                                    height=msg_ros1.height,
                                    width=msg_ros1.width,
                                    distortion_model=msg_ros1.distortion_model,
                                    d=msg_ros1.D,
                                    k=msg_ros1.K,
                                    r=msg_ros1.R,
                                    p=msg_ros1.P,
                                    binning_x=msg_ros1.binning_x,
                                    binning_y=msg_ros1.binning_y,
                                    roi=msg_ros1.roi
                                )
                                conn_obj = topic_to_conn[connection.topic]
                                # cdr_bytes = ts.serialize_ros2(msg_ros2, conn_obj.msgtype)
                                cdr_bytes = typestore_ros2.serialize_cdr(msg_ros2, msgtype)
                                writer.write(conn_obj, timestamp, cdr_bytes)
                            elif msgtype == 'sensor_msgs/msg/PointCloud2':
                                # Deserialize ROS1 PointCloud2, convert to custom format, and serialize for ROS2
                                msg_ros1 = typestore_ros1.deserialize_ros1(rawdata, connection.msgtype)
                                msg_ros2 = ros1_pointcloud_to_ros2(msg_ros1)
                                conn_obj = topic_to_conn[connection.topic]
                                cdr_bytes = typestore_ros2.serialize_cdr(msg_ros2, msgtype)
                                writer.write(conn_obj, timestamp, cdr_bytes)
                                # try:
                                #     cdr = typestore_ros2.serialize_cdr(msg_ros2, msgtype)
                                #     writer.write(conn_obj, timestamp, cdr)
                                # except Exception:
                                #     import traceback, sys
                                #     print(f"\n--- FAILED serializing PointCloud2 on topic {connection.topic} ---")
                                #     traceback.print_exc()
                                #     print("Dump of all msg_ros2 attributes:")
                                #     for name in dir(msg_ros2):
                                #         if name.startswith('_'): continue
                                #         val = getattr(msg_ros2, name)
                                #         print(f" {name:12s} ({type(val)})")
                                #     sys.exit(1)
                            else:
                                # Default path for other msgs
                                msg_ros1 = typestore_ros1.deserialize_ros1(rawdata, connection.msgtype)
                                cdr_bytes = typestore_ros2.serialize_cdr(msg_ros1, msgtype)
                                conn_obj = topic_to_conn[connection.topic]
                                writer.write(conn_obj, timestamp, cdr_bytes)
                        except Exception as e:
                            print(f"Failed to write {connection.topic}: {e}")
                    elif connection.topic == '/watono_chasis_sensor':
                        control_mode, gear_status, steering_status, velocity_status = decode_watono_chasis_sensor(rawdata)
                        if control_mode is not None:
                            cdr_bytes = typestore_ros2.serialize_cdr(control_mode, 'autoware_vehicle_msgs/msg/ControlModeReport')
                            writer.write(control_mode_conn, timestamp, cdr_bytes)
                        if gear_status is not None:
                            cdr_bytes = typestore_ros2.serialize_cdr(gear_status, 'autoware_vehicle_msgs/msg/GearReport')
                            writer.write(gear_status_conn, timestamp, cdr_bytes)
                        if steering_status is not None:
                            cdr_bytes = typestore_ros2.serialize_cdr(steering_status, 'autoware_vehicle_msgs/msg/SteeringReport')
                            writer.write(steering_conn, timestamp, cdr_bytes)
                        if velocity_status is not None:
                            cdr_bytes = typestore_ros2.serialize_cdr(velocity_status, 'autoware_vehicle_msgs/msg/VelocityReport')
                            writer.write(velocity_conn, timestamp, cdr_bytes)

    print("All done!")


if __name__ == "__main__":
    YAML_PATH = "/home/minghao/autoware/src/orange_bus_scripts/autoware_sensor_kit_base_link.yaml"
    ROS1_BAG_FOLDER = "/media/minghao/Data2TB/WatonoBus/2025-03-06_15-03-04-RingroadAndMerging"
    OUTPUT_BAG_DIR = "/media/minghao/Data2TB/WatonoBus/2025-03-06_15-03-04-RingroadAndMerging_ROS2"
    # ROS1_BAG_FOLDER = "/media/minghao/Data2TB/WatonoBus/ExampleROS1"
    # OUTPUT_BAG_DIR = "/media/minghao/Data2TB/WatonoBus/ExampleROS2"
    # List all .bag files in the folder and natsort
    import glob
    import natsort
    bag_files = natsort.natsorted(glob.glob(os.path.join(ROS1_BAG_FOLDER, '*.bag')))
    if not bag_files:
        print(f"No .bag files found in {ROS1_BAG_FOLDER}")
        sys.exit(1)
    main(YAML_PATH, bag_files, OUTPUT_BAG_DIR)
