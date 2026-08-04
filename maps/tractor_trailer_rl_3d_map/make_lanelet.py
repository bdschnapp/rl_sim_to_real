#!/usr/bin/env python
"""Generate a Lanelet2 .osm from centerline waypoints — elevation-aware version.

Same recipe as lab_map_capture/make_lanelet.py (straight segments with spline
arcs through selected waypoint ranges, +/- WIDTH/2 bounds, one bidirectional
lane) with ONE change for the 3D map: every node's `ele` is sampled from the
map's local ground grid (ground_grid.npz, built from pointcloud_map_3d_final)
instead of the June map's hard-coded ELE = -0.5. The lane surface follows the
ramp.

Usage:
    /home/ben/Ben/Thesis/kiss_icp_venv/bin/python make_lanelet.py [clicked_points.csv]
-> writes lanelet2_map.osm
"""
import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# ---- EDIT THESE -------------------------------------------------------------
WAYPOINTS = None       # None = read clicked_points.csv; or list of (x, y)
SMOOTH_RANGES = [(2, 5)]     # list of inclusive 0-based (a, b) waypoint index ranges
                       # to spline-smooth (corners); everything else STRAIGHT.
                       # e.g. [(1, 3)] to arc through waypoints 2-4 (1-based 2..4)
WIDTHS   = [2.8, 2.8, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1]
                       # full lane width (m) PER WAYPOINT; linearly tapered in
                       # between (narrow alley -> two-lane road)
SPACING  = 0.5         # resampled centerline spacing (m)
ONE_WAY  = False       # bidirectional single lane (drive up & back)
SPEED_KMH = 5.0
OUT      = HERE / "lanelet2_map.osm"
# -----------------------------------------------------------------------------

_g = np.load(HERE / "ground_grid.npz")
_gz, _xy0, _cell = _g["gz"], _g["xy0"], float(_g["cell"])

def ground_z(x, y):
    """Bilinear sample of the local ground elevation grid."""
    fx = (x - _xy0[0]) / _cell - 0.5
    fy = (y - _xy0[1]) / _cell - 0.5
    i0, j0 = int(np.floor(fx)), int(np.floor(fy))
    ti, tj = fx - i0, fy - j0
    i0 = np.clip(i0, 0, _gz.shape[0] - 2)
    j0 = np.clip(j0, 0, _gz.shape[1] - 2)
    return float((_gz[i0, j0] * (1 - ti) + _gz[i0 + 1, j0] * ti) * (1 - tj) +
                 (_gz[i0, j0 + 1] * (1 - ti) + _gz[i0 + 1, j0 + 1] * ti) * tj)

def _straight(p, q, spacing):
    n = max(int(np.linalg.norm(np.subtract(q, p)) / spacing) + 1, 2)
    return np.column_stack([np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)])

def _corner_bezier(wp, a, b, spacing):
    """Cubic Bezier from wp[a] to wp[b], tangent-matched to the adjacent
    straights (G1 joins, smooth curvature). Interior waypoints only hint."""
    wp = np.asarray(wp, float)
    p0, p3 = wp[a], wp[b]
    t0 = wp[a] - wp[a - 1] if a > 0 else wp[a + 1] - wp[a]
    t1 = wp[b + 1] - wp[b] if b + 1 < len(wp) else wp[b] - wp[b - 1]
    t0 /= np.linalg.norm(t0); t1 /= np.linalg.norm(t1)
    chord = np.linalg.norm(p3 - p0)
    p1, p2 = p0 + t0 * 0.42 * chord, p3 - t1 * 0.42 * chord
    n = max(int(1.6 * chord / spacing) + 1, 8)
    u = np.linspace(0, 1, n)[:, None]
    return ((1-u)**3*p0 + 3*(1-u)**2*u*p1 + 3*(1-u)*u**2*p2 + u**3*p3)

def _arc(pts, spacing):
    pts = np.asarray(pts, float)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    u = np.linspace(0, seg[-1], max(int(seg[-1] / spacing) + 1, 2))
    from scipy.interpolate import splprep, splev
    tck, _ = splprep([pts[:, 0], pts[:, 1]], u=seg, k=min(3, len(pts) - 1), s=0)
    x, y = splev(u, tck)
    return np.column_stack([x, y])

def build_centerline(wp, spacing, smooth_ranges):
    wp = np.asarray(wp, float)
    ranges = sorted(smooth_ranges)
    parts, i = [], 0
    for a, b in ranges:
        for k in range(i, a):
            parts.append(_straight(wp[k], wp[k + 1], spacing))
        parts.append(_corner_bezier(wp, a, b, spacing))
        i = b
    for k in range(i, len(wp) - 1):
        parts.append(_straight(wp[k], wp[k + 1], spacing))
    cl = np.vstack(parts)
    keep = np.r_[True, np.any(np.abs(np.diff(cl, axis=0)) > 1e-9, axis=1)]
    return cl[keep]

def normals(cl):
    d = np.gradient(cl, axis=0)
    t = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return np.column_stack([-t[:, 1], t[:, 0]])

def main():
    wp = WAYPOINTS
    csv = sys.argv[1] if len(sys.argv) > 1 else HERE / "clicked_points.csv"
    if wp is None:
        wp = [tuple(map(float, l.split(","))) for l in open(csv) if l.strip()]
        print(f"waypoints from {csv}: {len(wp)}")
    cl = build_centerline(wp, SPACING, SMOOTH_RANGES)
    if len(cl) >= 9:                       # round the straight/spline joints
        from scipy.signal import savgol_filter
        cl = np.column_stack([savgol_filter(cl[:, 0], 9, 2),
                              savgol_filter(cl[:, 1], 9, 2)])
    nl = normals(cl)
    # per-point width: waypoint widths interpolated along centerline arc-length
    s_cl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))]
    wp_a = np.asarray(wp, float)
    s_wp = [s_cl[np.argmin(np.linalg.norm(cl - w, axis=1))] for w in wp_a]
    width = np.interp(s_cl, s_wp, WIDTHS[:len(wp_a)])
    left  = cl + (width[:, None] / 2.0) * nl
    right = cl - (width[:, None] / 2.0) * nl

    nid = [0]; lines = []
    def node(x, y):
        nid[0] += 1
        lines.append(f'  <node id="{nid[0]}" lat="0.0" lon="0.0">')
        lines.append(f'    <tag k="local_x" v="{x:.4f}"/>')
        lines.append(f'    <tag k="local_y" v="{y:.4f}"/>')
        lines.append(f'    <tag k="ele" v="{ground_z(x, y):.4f}"/>')
        lines.append(f'  </node>')
        return nid[0]
    header = ['<?xml version="1.0" encoding="UTF-8"?>',
              '<osm version="0.6" generator="make_lanelet.py (map_capture_20260730)">',
              '  <MetaInfo format_version="1" map_version="1"/>']
    lines = header
    lnodes = [node(x, y) for x, y in left]
    rnodes = [node(x, y) for x, y in right]

    wid = nid[0] + 1
    def way(wid, nodes, ltype, lsub):
        out = [f'  <way id="{wid}">']
        out += [f'    <nd ref="{r}"/>' for r in nodes]
        out += [f'    <tag k="type" v="{ltype}"/>',
                f'    <tag k="subtype" v="{lsub}"/>',
                f'  </way>']
        return out
    lines += way(wid,     lnodes, "line_thin", "solid")
    lines += way(wid + 1, rnodes, "line_thin", "solid")

    rid = wid + 2
    lines += [f'  <relation id="{rid}">',
              f'    <member type="way" ref="{wid}" role="left"/>',
              f'    <member type="way" ref="{wid + 1}" role="right"/>',
              f'    <tag k="type" v="lanelet"/>',
              f'    <tag k="subtype" v="road"/>',
              f'    <tag k="location" v="private"/>',
              f'    <tag k="one_way" v="{"yes" if ONE_WAY else "no"}"/>',
              f'    <tag k="speed_limit" v="{SPEED_KMH:.1f}"/>',
              f'    <tag k="participant:vehicle" v="yes"/>',
              f'  </relation>',
              '</osm>']
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    ele = [ground_z(x, y) for x, y in cl]
    print(f"centerline {len(cl)} pts, width {width.min():.1f}..{width.max():.1f} m, one_way={ONE_WAY} -> {OUT}")
    print(f"elevation along lane: {min(ele):.2f} .. {max(ele):.2f} m")

if __name__ == "__main__":
    main()
