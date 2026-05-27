#!/usr/bin/env python3
"""Convert a unidirectional lanelet2 OSM map into a bidirectional one.

Specific to the MVSL lab map's structure:
  - Forward chain LL7 -> LL402 -> LL381
  - Reverse chain LL388 -> LL413 -> LL85   (covers the same physical track)
We keep the forward chain, drop the reverse chain, and tag the survivors with
one_way=no so a single lanelet represents each physical road section.

Run:
  python3 src/electrans_rl_bridge/scripts/make_bidirectional_map.py \
      [/path/to/lanelet2_map.osm]
The script writes the modified file in place and saves a backup next to it
(<file>.bak.unidirectional) the first time it runs. Subsequent runs are
idempotent (already-bidirectional maps are left alone).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

# Lanelets representing the "reverse" direction of each physical road section.
# Keep the "forward" lanelet of each pair; this set gets dropped.
REVERSE_DIRECTION_LANELETS = {"85", "388", "413"}

# Lanelets to keep + flip to bidirectional.
FORWARD_DIRECTION_LANELETS = {"7", "381", "402"}

# Bound-expansion pass: the unidirectional pairs cover slightly offset
# strips of the same physical road (the two original lanelets sat ~0.3 m
# apart). After dropping the reverse half, the surviving lanelet's bounds
# cover only one side of that asymmetry, so its centerline (= bound midpoint)
# ends up offset from the actual road center. For each surviving way listed
# below, we replace the local_y tag of every node on that way with the
# specified value, expanding the lanelet to span the full physical road.
# Values come from the corresponding bound's coordinate in the dropped
# lanelet (e.g. way 6's south edge moves from y=-1.5 to the y=-1.8 of LL85's
# old right bound, way 81).
WAY_Y_OVERRIDES = {
    "6":   -1.81,  # LL7 left bound; was -1.5, expand to LL85's old right bound (way 81)
}
WAY_X_OVERRIDES = {
    "380": -8.93,  # LL381 left bound; was -8.73, expand to LL388's old right bound (way 384)
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "osm_path",
        nargs="?",
        default=os.path.expanduser("~/Ben/Electrans/autoware_map/mvsl/lanelet2_map.osm"),
        help="lanelet2_map.osm to convert (default: MVSL lab map)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.osm_path):
        print(f"ERROR: {args.osm_path} not found.", file=sys.stderr)
        return 1

    # Backup the original (only if a backup doesn't already exist — preserves
    # the truly-original file on repeated runs).
    backup = args.osm_path + ".bak.unidirectional"
    if not os.path.exists(backup):
        shutil.copy2(args.osm_path, backup)
        print(f"Backed up original -> {backup}")
    else:
        print(f"Backup already exists, leaving {backup} untouched")

    tree = ET.parse(args.osm_path)
    root = tree.getroot()

    # ---- Pass 1: remove reverse-direction lanelet relations.
    removed_relations = []
    for rel in list(root.findall("relation")):
        rid = rel.get("id")
        if rid in REVERSE_DIRECTION_LANELETS:
            # Capture its member way refs before deleting, so we know what
            # ways might now be orphaned.
            way_refs = {
                m.get("ref") for m in rel.findall("member") if m.get("type") == "way"
            }
            root.remove(rel)
            removed_relations.append((rid, way_refs))
    print(f"Removed {len(removed_relations)} reverse-direction lanelets: "
          f"{[r[0] for r in removed_relations]}")

    # ---- Pass 2: find ways that are no longer referenced by any relation.
    referenced_ways = {
        m.get("ref")
        for rel in root.findall("relation")
        for m in rel.findall("member")
        if m.get("type") == "way"
    }
    candidate_orphan_ways = {ref for _, refs in removed_relations for ref in refs}
    orphan_ways = candidate_orphan_ways - referenced_ways

    removed_ways = []
    for way in list(root.findall("way")):
        wid = way.get("id")
        if wid in orphan_ways:
            nd_refs = {nd.get("ref") for nd in way.findall("nd")}
            root.remove(way)
            removed_ways.append((wid, nd_refs))
    print(f"Removed {len(removed_ways)} orphan ways: {[w[0] for w in removed_ways]}")

    # ---- Pass 3: find nodes no longer referenced by any way.
    referenced_nodes = {
        nd.get("ref")
        for way in root.findall("way")
        for nd in way.findall("nd")
    }
    candidate_orphan_nodes = {ref for _, refs in removed_ways for ref in refs}
    orphan_nodes = candidate_orphan_nodes - referenced_nodes

    removed_nodes = 0
    for node in list(root.findall("node")):
        if node.get("id") in orphan_nodes:
            root.remove(node)
            removed_nodes += 1
    print(f"Removed {removed_nodes} orphan nodes")

    # ---- Pass 3.5: expand surviving lanelet bounds to span the full
    # physical road. The kept lanelet's bounds originally covered ~one lane
    # of the pair; we shift the appropriate boundary outward so the centerline
    # (auto-computed as the bound midpoint) lines up with the road center.
    if WAY_Y_OVERRIDES or WAY_X_OVERRIDES:
        # Build map: node_id -> set of way_ids that reference it.
        node_to_ways = {}
        for way in root.findall("way"):
            for nd in way.findall("nd"):
                node_to_ways.setdefault(nd.get("ref"), set()).add(way.get("id"))

        n_y_updates = 0
        n_x_updates = 0
        for node in root.findall("node"):
            nid = node.get("id")
            refs = node_to_ways.get(nid, set())
            for tag in node.findall("tag"):
                k = tag.get("k")
                if k == "local_y" and any(w in WAY_Y_OVERRIDES for w in refs):
                    # If a node is on multiple ways, all override targets
                    # should agree; we take the first matching way's value.
                    target = next(
                        WAY_Y_OVERRIDES[w] for w in refs if w in WAY_Y_OVERRIDES
                    )
                    tag.set("v", f"{target:.4f}")
                    n_y_updates += 1
                elif k == "local_x" and any(w in WAY_X_OVERRIDES for w in refs):
                    target = next(
                        WAY_X_OVERRIDES[w] for w in refs if w in WAY_X_OVERRIDES
                    )
                    tag.set("v", f"{target:.4f}")
                    n_x_updates += 1
        print(f"Bound-expansion: updated local_y on {n_y_updates} nodes, "
              f"local_x on {n_x_updates} nodes")

    # ---- Pass 4: set one_way=no on the surviving forward lanelets.
    flipped = []
    for rel in root.findall("relation"):
        if rel.get("id") not in FORWARD_DIRECTION_LANELETS:
            continue
        one_way_tag = next(
            (t for t in rel.findall("tag") if t.get("k") == "one_way"), None
        )
        if one_way_tag is None:
            ET.SubElement(rel, "tag", {"k": "one_way", "v": "no"})
        elif one_way_tag.get("v") != "no":
            one_way_tag.set("v", "no")
        else:
            continue  # already bidirectional
        flipped.append(rel.get("id"))
    print(f"Tagged one_way=no on {len(flipped)} lanelets: {flipped}")

    # Preserve XML declaration on write.
    tree.write(args.osm_path, encoding="UTF-8", xml_declaration=True)
    print(f"Wrote {args.osm_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
