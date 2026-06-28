# tractor_trailer_rl_lab_map  (2026-06-27)

Fresh lab map for the tractor-trailer RL work. Built from an offline KISS-ICP
aggregation of a raw-sensor mapping bag (lidar + wheel odom, no IMU), cleaned in
CloudCompare, with the lanelet authored algorithmically (no GUI) from hand-picked
centerline waypoints.

## Files (Autoware map_path contents)
- `pointcloud_map.pcd`     — NDT/localization cloud, floor+ceiling+operator removed (~363k pts)
- `lanelet2_map.osm`       — 1 BIDIRECTIONAL lane (one_way=no), full width 2.8 m (half-width 1.40 m)
- `map_projector_info.yaml`— projector_type: Local (origin = robot start pose, heading +X)

## Provenance / how to regenerate the lanelet
- `make_lanelet.py`   — WAYPOINTS + WIDTH=2.8 + SMOOTH_RANGE=(1,3); straight segments except a
                        spline arc through the corner trio. Run -> lanelet2_map.osm.
- `clicked_points.csv`— the centerline waypoints used.
- `final_map_verify.png` — lane polygon overlaid on the final cloud.
Source bag + full pipeline: Electrans_project/lab_map_capture/ (build_pcd.py, KISS-ICP poses,
remove_inside_lanelets.py).

## Use
Load as Autoware map_path, e.g.:
  map_path:=$HOME/Ben/Electrans/autoware_map/tractor_trailer_rl_lab_map
