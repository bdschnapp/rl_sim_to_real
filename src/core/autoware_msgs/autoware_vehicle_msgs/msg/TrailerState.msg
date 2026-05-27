# Articulated truck-trailer state message.
# The hitch joint connects the truck to the trailer via a revolute joint.
#
# Sign convention:
#   hitch_angle = theta_truck - theta_trailer
#   Positive hitch_angle => trailer swings to the left of the truck heading.
#
builtin_interfaces/Time stamp
float32 hitch_angle    # articulation angle beta [rad]
float32 hitch_rate     # time derivative of hitch_angle [rad/s]
