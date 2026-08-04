# tractor_trailer_rl_3d_map  (2026-07-30)

First FULL-3D map (ground + ceiling kept, 2.5 m real elevation change) for the
tractor-trailer RL work. Replaces the walls-only slab approach of
tractor_trailer_rl_lab_map — the localization crop box / init covariance must
be reverted for this map (see "Deploy" below).

## Files
- `pointcloud_map.pcd`      — NDT cloud, 4,619,119 pts @ 5 cm voxel, FULL 3D.
- `lanelet2_map.osm`        — 1 bidirectional lane, VARIABLE width: 2.8 m in the
                              narrow east-west alley tapering to 5.1 m (two-lane)
                              from the T-junction through the south corridor.
                              Node `ele` follows the real ground (−0.14..−2.31 m).
- `map_projector_info.yaml` — projector_type: Local (origin = robot start pose,
                              base_link at ground, heading +X).

## Provenance
Bag: robot ~/bags/mapping_20260730-102142 (desktop copy ~/Ben/Thesis/bags/).
Odometry: GenZ-ICP 0.3.2, custom config genz_custom/c1_range40.yaml (pretuned
configs diverge on this bag!) + Open3D pose-graph loop closure. Revisit wall
misalignment p95 = 0.040 m. Known caveat: absolute elevation of the south dip
is uncertain (KISS says −0.7 m, GenZ −2.4 m, scan-ground checks suggest between);
map is internally consistent to ≤6 cm, which is what NDT consumes.
Cleaning: Ben's CloudCompare crop (warp-transferred) + scan-persistence ghost
filter (operator on foot removed). Transient objects (A..D in lanelet_verify_final.png,
biggest 1.5 m tall at x≈7.6 in-lane) were present all drive but are NOT permanent;
they remain in the pcd — remove physically before driving or ignore (NDT tolerant).
Full pipeline + intermediates: Electrans_project/map_capture_20260730/.

## Deploy (3D map — REQUIRED config reverts vs lab map)
1. crop_box_filter_measurement_range.param.yaml: min_z 0.0 -> -30.0, max_z 2.0 -> 50.0
2. initialpose_shim.py RVIZ_PARTICLE_COVARIANCE: widen z (0.01 -> ~0.25-1.0) and
   roll/pitch (0.01 -> ~0.02-0.05); flat-floor assumption does not hold.
3. First live test: re-check NVTL against the 2.3 convergence gate (calibrated on
   the walls-only regime); map_height_fitter biases downhill on slopes.
Launch: map_path:=$HOME/Ben/Electrans/autoware_map/tractor_trailer_rl_3d_map
