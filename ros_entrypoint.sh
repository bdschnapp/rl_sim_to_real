#!/bin/bash
set -e

# Source the base ROS 2 installation
source "/opt/ros/$ROS_DISTRO/setup.bash"

# Source the standard Autoware packages (Phase 1 base workspace)
if [ -f "/autoware_ws/install/setup.bash" ]; then
    source /autoware_ws/install/setup.bash
fi

# Source the custom Electrans packages (Phase 2 overlay workspace)
if [ -f "/electrans_ws/install/setup.bash" ]; then
    source /electrans_ws/install/setup.bash
fi

exec "$@"
