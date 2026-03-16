#!/usr/bin/env python3
"""
Clean up KITTI label files for IAC racecar PointPillars training.

Operates directly on label files in OpenPCDet/data/kitti/training/label_2/:
  1. Fix bounding box dimensions (h, w, l) to match actual Dallara racecar
  2. Fix z-coordinate to correct for the new height
  3. Remove labels with fewer than MIN_POINTS LiDAR points inside the box

Creates a backup of original labels before modifying.

Must be run from OpenPCDet/tools/ so configs resolve correctly.

Usage:
    cd /p/cavalier/jay/OpenPCDet/tools
    python /p/cavalier/jay/scripts/cleanup_labels.py [--min_points 5] [--dry_run]
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# ── Corrected IAC Dallara Racecar Dimensions (meters) ──────────────────────
# From spec sheet: 4876mm L × 1930mm W × 1156.5mm H
NEW_H = 1.16   # height (rounded from 1.1565)
NEW_W = 1.93   # width
NEW_L = 4.88   # length (rounded from 4.876)

# Old values used in preprocessing
OLD_H = 1.50
OLD_OPPONENT_H = 1.50  # was OPPONENT_H in preprocess_kitti.py


def points_in_box(points, cx, cy, cz, dx, dy, dz, heading):
    """Check how many points fall inside a 3D oriented bounding box."""
    pts = points[:, :3] - np.array([cx, cy, cz])
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    rot = np.array([
        [cos_h, -sin_h, 0],
        [sin_h,  cos_h, 0],
        [0,      0,     1],
    ])
    pts_aligned = pts @ rot.T
    mask = (
        (np.abs(pts_aligned[:, 0]) <= dx / 2) &
        (np.abs(pts_aligned[:, 1]) <= dy / 2) &
        (np.abs(pts_aligned[:, 2]) <= dz / 2)
    )
    return int(mask.sum())


def fix_label_line(parts):
    """
    Fix dimensions and z-coordinate in a parsed KITTI label line.

    KITTI label format: class trunc occ alpha bb1 bb2 bb3 bb4 h w l x y z ry
                        0     1     2   3     4   5   6   7   8 9 10 11 12 13 14

    The preprocessing did:  z_kitti = z_original - OLD_OPPONENT_H / 2
    This is the bottom-center z in KITTI convention.

    OpenPCDet's boxes3d_kitti_camera_to_lidar does: z_lidar = z_kitti + h/2
    to get center-z.

    So the original center-z = z_kitti + OLD_H/2.
    New z_kitti (bottom-center) = original_center_z - NEW_H/2
                                = (z_kitti + OLD_H/2) - NEW_H/2
                                = z_kitti + (OLD_H - NEW_H) / 2
    """
    z_kitti = float(parts[13])
    z_kitti_new = z_kitti + (OLD_H - NEW_H) / 2.0

    parts[8] = f"{NEW_H:.2f}"
    parts[9] = f"{NEW_W:.2f}"
    parts[10] = f"{NEW_L:.2f}"
    parts[13] = f"{z_kitti_new:.2f}"

    return parts


def get_box_lidar(parts):
    """
    Convert KITTI label parts to LiDAR-frame box [cx, cy, cz, dx, dy, dz, heading].

    With identity calibration (our setup):
      KITTI camera [x, y, z, l, h, w, r] → LiDAR [x, y, z+h/2, l, w, h, -(r+pi/2)]
    """
    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
    ry = float(parts[14])

    # With identity Tr_velo_to_cam, camera coords = LiDAR coords
    # OpenPCDet adds h/2 to z to get center
    cz = z + h / 2.0
    heading = -(ry + math.pi / 2.0)

    return x, y, cz, l, w, h, heading


def process_frame(label_path, bin_path, min_points, dry_run=False):
    """
    Process one frame: fix dimensions, filter by point count.

    Returns: (original_count, kept_count, removed_count)
    """
    with open(label_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return 0, 0, 0

    # Load point cloud
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    new_lines = []
    removed = 0

    for line in lines:
        parts = line.split()
        if len(parts) < 15 or parts[0] != "Car":
            continue

        # Fix dimensions first
        parts = fix_label_line(parts)

        # Get the corrected box in LiDAR frame for point counting
        cx, cy, cz, dx, dy, dz, heading = get_box_lidar(parts)
        count = points_in_box(points, cx, cy, cz, dx, dy, dz, heading)

        if count >= min_points:
            new_lines.append(" ".join(parts))
        else:
            removed += 1

    if not dry_run:
        with open(label_path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")

    return len(lines), len(new_lines), removed


def main():
    ap = argparse.ArgumentParser(description="Clean up KITTI labels for IAC racecar")
    ap.add_argument("--data_dir", default="/p/cavalier/jay/OpenPCDet/data/kitti/training",
                    help="Path to KITTI training directory")
    ap.add_argument("--min_points", type=int, default=5,
                    help="Minimum LiDAR points inside box to keep label")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print changes without modifying files")
    ap.add_argument("--no_backup", action="store_true",
                    help="Skip creating backup of label directory")
    args = ap.parse_args()

    label_dir = Path(args.data_dir) / "label_2"
    vel_dir = Path(args.data_dir) / "velodyne"

    if not label_dir.is_dir():
        sys.exit(f"ERROR: label dir not found: {label_dir}")
    if not vel_dir.is_dir():
        sys.exit(f"ERROR: velodyne dir not found: {vel_dir}")

    # Backup
    backup_dir = label_dir.parent / "label_2_backup"
    if not args.dry_run and not args.no_backup and not backup_dir.exists():
        print(f"Creating backup: {backup_dir}")
        shutil.copytree(label_dir, backup_dir)
        print(f"  Backup complete.")

    # Process all label files
    label_files = sorted(label_dir.glob("*.txt"))
    print(f"Processing {len(label_files)} label files "
          f"(min_points={args.min_points}, dry_run={args.dry_run})")
    print(f"  Correcting dimensions: h={OLD_H}->{NEW_H}, w=2.00->{NEW_W}, l=5.00->{NEW_L}")

    total_original = 0
    total_kept = 0
    total_removed = 0
    frames_modified = 0

    for i, label_path in enumerate(label_files):
        if i % 5000 == 0:
            print(f"  [{i}/{len(label_files)}]")

        bin_path = vel_dir / f"{label_path.stem}.bin"
        if not bin_path.exists():
            continue

        orig, kept, removed = process_frame(
            label_path, bin_path, args.min_points, dry_run=args.dry_run
        )
        total_original += orig
        total_kept += kept
        total_removed += removed
        if removed > 0:
            frames_modified += 1

    print(f"\n{'='*60}")
    print(f"CLEANUP {'(DRY RUN) ' if args.dry_run else ''}COMPLETE")
    print(f"{'='*60}")
    print(f"  Original labels:   {total_original}")
    print(f"  Kept labels:       {total_kept}")
    print(f"  Removed labels:    {total_removed} ({100*total_removed/max(total_original,1):.1f}%)")
    print(f"  Frames modified:   {frames_modified}")
    print(f"  Dimensions fixed:  h={NEW_H}, w={NEW_W}, l={NEW_L}")
    if not args.dry_run:
        print(f"  Backup at:         {backup_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
