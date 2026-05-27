import os
import yaml
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

# ---------------- Configuration ----------------
LAUNCH_FILE = "/home/minghao/autoware/src/orange_bus_scripts/transform_pub.launch"  # Update this
OUTPUT_YAML = "/home/minghao/autoware/src/orange_bus_scripts/autoware_sensor_kit_base_link.yaml"
DESIRED_ROOT_FRAME = "base_link"

# Manual transform: "x y z roll pitch yaw parent child rate"
# MANUAL_BASELINK_TF_STR = "-3.39 0 -0.9 0 0 0 rslidar_front base_link 50"
# ------------------------------------------------


def remove_comments(text):
    import re
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)


def parse_launch_file(path):
    with open(path, 'r') as f:
        content = f.read()
    clean_content = remove_comments(content)
    root = ET.fromstring(clean_content)
    transforms = []
    for node in root.findall('node'):
        args = node.attrib.get('args')
        if not args:
            continue
        parts = args.split()
        if len(parts) < 8:
            continue
        x, y, z = map(float, parts[0:3])
        yaw, pitch, roll = map(float, parts[3:6])
        parent, child = parts[6:8]
        transforms.append({
            'parent': parent,
            'child': child,
            'transform': {'translation': [x, y, z], 'rotation': [roll, pitch, yaw]}
        })
    return transforms


def parse_manual_tf(args_str):
    parts = args_str.split()
    x, y, z = map(float, parts[0:3])
    yaw, pitch, roll = map(float, parts[3:6])
    parent, child = parts[6:8]
    return {
        'parent': parent,
        'child': child,
        'transform': {'translation': [x, y, z], 'rotation': [roll, pitch, yaw]}
    }


def tf_to_matrix(translation, rpy):
    rot_mat = R.from_euler('xyz', rpy).as_matrix()
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rot_mat
    tf_mat[:3, 3] = translation
    return tf_mat


def matrix_to_tf(matrix):
    translation = matrix[:3, 3].tolist()
    rpy = R.from_matrix(matrix[:3, :3]).as_euler('xyz').tolist()
    return translation, rpy


def build_tf_graph(transforms):
    graph = {}
    for tf in transforms:
        parent = tf['parent']
        child = tf['child']
        T = tf_to_matrix(tf['transform']['translation'], tf['transform']['rotation'])
        graph[(parent, child)] = T
        graph[(child, parent)] = np.linalg.inv(T)
    return graph


def find_chain(graph, source, target, visited=None):
    visited = visited or set()
    visited.add(source)
    for (a, b), T in graph.items():
        if a == source and b not in visited:
            if b == target:
                return [T]
            sub_chain = find_chain(graph, b, target, visited)
            if sub_chain:
                return [T] + sub_chain
    return None


def compute_transform_to_base(graph, child, base):
    if child == base:
        return np.eye(4)
    chain = find_chain(graph, base, child)
    if chain is None:
        return None
    result = np.eye(4)
    for T in chain:
        result = result @ T
    return result


def main():
    # Step 1: Load and clean transforms
    transforms = parse_launch_file(LAUNCH_FILE)

    # Step 2: Add manual base_link transform
    # manual_tf = parse_manual_tf(MANUAL_BASELINK_TF_STR)
    # transforms.append(manual_tf)

    # Step 3: Build graph
    tf_graph = build_tf_graph(transforms)

    # Step 4: Convert all frames to base_link
    sensors_yaml = {}
    # all_children = set(tf['child'] for tf in transforms)
    all_children = set()
    for (parent, child), _ in tf_graph.items():
        all_children.add(child)
        all_children.add(parent)

    print("All children found in transforms:")
    for child in sorted(all_children):
        print(f" - {child}")

    for child in sorted(all_children):
        if child == DESIRED_ROOT_FRAME:
            continue
        full_tf = compute_transform_to_base(tf_graph, child, DESIRED_ROOT_FRAME)
        if full_tf is None:
            print(f"⚠️ Could not resolve transform from {child} to {DESIRED_ROOT_FRAME}, skipping.")
            continue
        translation, rpy = matrix_to_tf(full_tf)
        sensors_yaml[child] = {
            'x': round(translation[0], 6),
            'y': round(translation[1], 6),
            'z': round(translation[2], 6),
            'roll': round(rpy[0], 6),
            'pitch': round(rpy[1], 6),
            'yaw': round(rpy[2], 6),
        }

    # Step 5: Save YAML
    final_yaml = {"sensor_kit_base_link": sensors_yaml}
    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)
    print(f"✅ Saved calibration to {OUTPUT_YAML}")


if __name__ == "__main__":
    main()
