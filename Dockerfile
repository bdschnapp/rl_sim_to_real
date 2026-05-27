# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Overlay image (post-prune)
#
# Builds the surviving Electrans packages on top of the pre-built autoware:latest
# base layer (which still contains all pre-prune autoware binaries — they're
# just no longer referenced by our pruned launch files).
#
#   docker build -f Dockerfile -t electrans:latest .
#
# This dev desktop has parallelization problems, so the build is forced
# sequential (one package, one compile job at a time).
# ─────────────────────────────────────────────────────────────────────────────
FROM autoware:latest
LABEL maintainer="Electrans"

ARG DEBIAN_FRONTEND=noninteractive

# ── Overlay our pruned launch / config files on top of the base install ───────
# autoware_launch — top-level launch + components (stubs for planning/control/api).
COPY src/launcher/autoware_launch/autoware_launch/launch \
     /autoware_ws/install/autoware_launch/share/autoware_launch/launch/

COPY src/launcher/autoware_launch/autoware_launch/config \
     /autoware_ws/install/autoware_launch/share/autoware_launch/config/

# tier4_simulator_launch — pruned simulator.launch.xml (no MOT/prediction/elevation_map/traffic_light/scenario_adapter)
COPY src/universe/autoware_universe/launch/tier4_simulator_launch/launch \
     /autoware_ws/install/tier4_simulator_launch/share/tier4_simulator_launch/launch/

# tier4_perception_launch — pruned perception.launch.xml (obstacle_seg + OGM only)
# Note: object_recognition/ and traffic_light_recognition/ subdirs are gone from source
# but still sit in the install image; harmless because nothing includes them.
COPY src/universe/autoware_universe/launch/tier4_perception_launch/launch \
     /autoware_ws/install/tier4_perception_launch/share/tier4_perception_launch/launch/

# tier4_localization_launch — pruned pose_twist_estimator.launch.xml (NDT + gyro + pose_initializer only)
COPY src/universe/autoware_universe/launch/tier4_localization_launch/launch \
     /autoware_ws/install/tier4_localization_launch/share/tier4_localization_launch/launch/

# tier4_system_launch — collapsed system.launch.xml (service_log_checker only)
COPY src/universe/autoware_universe/launch/tier4_system_launch/launch \
     /autoware_ws/install/tier4_system_launch/share/tier4_system_launch/launch/

# autoware_dummy_perception_publisher — dummy_perception_publisher.launch.xml with shape_estimation/feature_remover removed
COPY src/universe/autoware_universe/simulator/autoware_dummy_perception_publisher/launch \
     /autoware_ws/install/autoware_dummy_perception_publisher/share/autoware_dummy_perception_publisher/launch/

# autoware_core — autoware_core.launch.xml with planning/control/api includes removed
COPY src/core/autoware_core/autoware_core/launch \
     /autoware_ws/install/autoware_core/share/autoware_core/launch/

# Drop stale launch files that were deleted in the prune so they can't be picked up by mistake.
RUN rm -f /autoware_ws/install/autoware_launch/share/autoware_launch/launch/e2e_simulator.launch.xml \
       && rm -f /autoware_ws/install/autoware_launch/share/autoware_launch/launch/components/tier4_autoware_api_component.launch.xml

# ── New Electrans overlay packages ───────────────────────────────────────────
# Both are pure CMake/launch/URDF packages (no heavy C++ build).
COPY src/launcher/autoware_launch/vehicle/electrans_robot_vehicle_launch \
     /electrans_ws/src/electrans_robot_vehicle_launch/

COPY src/launcher/autoware_launch/sensor_kit/electrans_robot_sensor_kit_launch \
     /electrans_ws/src/electrans_robot_sensor_kit_launch/

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# ── Sequential overlay build ─────────────────────────────────────────────────
# Forced sequential: one package and one compile job at a time (dev desktop
# parallelization constraint).
WORKDIR /electrans_ws
RUN /bin/bash -o pipefail -c "\
    source /opt/ros/humble/setup.bash && \
    source /autoware_ws/install/setup.bash && \
    MAKEFLAGS='-j1' CMAKE_BUILD_PARALLEL_LEVEL=1 \
    colcon build \
      --executor sequential \
      --parallel-workers 1 \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      2>&1 | tee /electrans_ws/build.log" && \
    echo "==> Phase 2 build complete"

# ── Strip build artefacts ─────────────────────────────────────────────────────
RUN rm -rf /electrans_ws/build /electrans_ws/log

# ── Default command ───────────────────────────────────────────────────────────
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
