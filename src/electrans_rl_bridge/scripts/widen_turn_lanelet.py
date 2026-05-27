#!/usr/bin/env python3
"""Widen the outside edge of the MVSL turn lanelet (LL402).

After the bidirectional collapse, LL402 still has a tight inside-corner
(way 401) and a relatively snug outside (way 400) that puts the lane
centerline close enough to the inside that an RL policy taking the
optimal interior path nearly clips the corner. To open up the turn, we
shift way 400's four MIDDLE nodes ~0.3 m radially outward from the
inside apex (estimated at world (-6.2, +1.2)). The endpoints (nodes 9
and 378, shared with LL7 / LL381) stay put so the lane geometry remains
connected at the seams.

Idempotent: writes ABSOLUTE target positions rather than relative deltas
so re-running converges to the same map.

Run:
  python3 src/electrans_rl_bridge/scripts/widen_turn_lanelet.py \
      [/path/to/lanelet2_map.osm]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import math

# Original (as-authored) positions of LL402's outside-bound middle nodes.
# Endpoints (nodes 9 and 378) intentionally absent — they're shared with
# adjacent lanelets and must stay in place.
TURN_OUTSIDE_NODES_ORIGINAL = {
    "392": (-6.9444, -1.2548),
    "393": (-7.5505, -0.9304),
    "394": (-8.1764, -0.3109),
    "395": (-8.5071, +0.2918),
}

# Approximate inside-apex of the turn (way 401's curve summit). Outward
# shift direction is the unit vector pointing from this apex through
# each outside node.
TURN_INSIDE_APEX = (-6.2, +1.2)

# Magnitude of the outward shift in meters. Each middle node gets pushed
# this far along the unit-radial from TURN_INSIDE_APEX, regardless of how
# many times the script has run (the targets are absolute, so re-runs
# converge — bump this constant when you want to widen further).
WIDEN_M = 0.4


def _compute_targets():
    apex_x, apex_y = TURN_INSIDE_APEX
    targets = {}
    for nid, (x, y) in TURN_OUTSIDE_NODES_ORIGINAL.items():
        dx, dy = x - apex_x, y - apex_y
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        targets[nid] = (x + WIDEN_M * ux, y + WIDEN_M * uy)
    return targets


TURN_OUTSIDE_NODES = _compute_targets()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "osm_path",
        nargs="?",
        default=os.path.expanduser("~/Ben/Electrans/autoware_map/mvsl/lanelet2_map.osm"),
        help="lanelet2_map.osm to edit (default: MVSL lab map)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.osm_path):
        print(f"ERROR: {args.osm_path} not found.", file=sys.stderr)
        return 1

    # Distinct backup from the bidirectional-conversion backup, so we can
    # revert just the turn-widening without losing the bidirectional fix.
    backup = args.osm_path + ".bak.pre_widen"
    if not os.path.exists(backup):
        shutil.copy2(args.osm_path, backup)
        print(f"Backed up pre-widen state -> {backup}")
    else:
        print(f"Pre-widen backup already exists, leaving {backup} untouched")

    tree = ET.parse(args.osm_path)
    root = tree.getroot()

    n_updated = 0
    for node in root.findall("node"):
        nid = node.get("id")
        if nid not in TURN_OUTSIDE_NODES:
            continue
        tx, ty = TURN_OUTSIDE_NODES[nid]
        for tag in node.findall("tag"):
            k = tag.get("k")
            if k == "local_x":
                tag.set("v", f"{tx:.4f}")
            elif k == "local_y":
                tag.set("v", f"{ty:.4f}")
        n_updated += 1
        print(f"  node {nid} -> ({tx:+.4f}, {ty:+.4f})")

    if n_updated != len(TURN_OUTSIDE_NODES):
        print(f"WARNING: expected {len(TURN_OUTSIDE_NODES)} matches, "
              f"updated {n_updated}", file=sys.stderr)

    tree.write(args.osm_path, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote {args.osm_path}  (updated {n_updated} nodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
