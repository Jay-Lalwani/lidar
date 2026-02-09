#!/usr/bin/env python3
"""
Preprocess IAC KITTI LiDAR data for OpenPCDet PointPillars training.

Copies and transforms raw exported data into the KITTI directory layout that
OpenPCDet expects, applying:
  - Sequential 6-digit file renaming (sorted by timestamp)
  - Label normalisation  (class -> "Car", z adjustment, ry conversion)
  - Dummy image generation (OpenPCDet requires image_2/ to exist)
  - Automatic train/val split (80/20 by default)
  - Empty-frame filtering in ImageSets (frames with no labels are excluded)
  - testing/ symlink (required by create_kitti_infos)

Usage:
    python preprocess_kitti.py \\
        --input  /p/cavalier/data/processed/2025_01_06_PolimoveUnimore_KITTI \\
        --output /p/cavalier/jay/OpenPCDet/data/kitti
"""

import argparse
import glob
import math
import os
import random
import shutil
import struct
import sys
import zlib

# ---------------------------------------------------------------------------
# Constants  (must match the export_kitti.py that generated the raw data)
# ---------------------------------------------------------------------------
OPPONENT_H = 1.5                                   # opponent car height
KNOWN_CLASSES = {"Polimove", "Unimore", "KAIST", "Car"}
FAKE_BBOX = "0.00 0.00 50.00 50.00"                # gives difficulty=Easy


# ── helpers ─────────────────────────────────────────────────────────────────

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    angle = angle % (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle


def process_label_line(line: str) -> str | None:
    """
    Transform one KITTI label line for OpenPCDet:
      - Rename class -> Car
      - z:  center-height -> bottom-center  (z_new = z - H/2)
      - ry: ego yaw -> KITTI rotation_y     (ry = -yaw - pi/2)
      - Alpha: recomputed for consistency
      - 2D bbox replaced with dummy values
      - Extra fields (pitch, roll, score) stripped -> exactly 15 fields
    Returns None for empty / malformed / unknown-class lines.
    """
    parts = line.strip().split()
    if len(parts) < 15 or parts[0] not in KNOWN_CLASSES:
        return None

    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    x, y, z  = float(parts[11]), float(parts[12]), float(parts[13])
    yaw      = float(parts[14])

    # Adjustments
    z_new  = z - OPPONENT_H / 2.0
    ry_new = normalize_angle(-yaw - math.pi / 2.0)
    angle_to_obj = math.atan2(y, x) if (x != 0 or y != 0) else 0.0
    alpha_new = normalize_angle(ry_new - angle_to_obj)

    return (
        f"Car {parts[1]} {parts[2]} {alpha_new:.2f} "
        f"{FAKE_BBOX} "
        f"{h:.2f} {w:.2f} {l:.2f} "
        f"{x:.2f} {y:.2f} {z_new:.2f} "
        f"{ry_new:.2f}"
    )


def process_label_file(src_path: str) -> list[str]:
    """Read label file, transform each line, return non-None results."""
    with open(src_path) as f:
        return [r for line in f if (r := process_label_line(line)) is not None]


def create_dummy_png(path: str, width: int = 100, height: int = 100):
    """Write a minimal black RGB PNG (no external deps)."""
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw  = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Preprocess IAC KITTI data for OpenPCDet")
    ap.add_argument("--input",  required=True, help="Raw KITTI export directory")
    ap.add_argument("--output", required=True, help="OpenPCDet data/kitti directory")
    ap.add_argument("--split_mode", choices=["random", "sequential"], default="random")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Validate input dirs
    src_vel   = os.path.join(args.input, "velodyne")
    src_label = os.path.join(args.input, "label_2")
    src_calib = os.path.join(args.input, "calib")
    for d in (src_vel, src_label, src_calib):
        if not os.path.isdir(d):
            sys.exit(f"ERROR: directory not found: {d}")

    # Discover and sort by timestamp
    bin_files = sorted(glob.glob(os.path.join(src_vel, "*.bin")))
    timestamps = sorted(
        [os.path.splitext(os.path.basename(f))[0] for f in bin_files],
        key=int,
    )
    n_total = len(timestamps)
    print(f"Found {n_total} point cloud files")
    if n_total == 0:
        sys.exit("ERROR: no .bin files found")

    # Output paths
    out_train    = os.path.join(args.output, "training")
    out_vel      = os.path.join(out_train, "velodyne")
    out_label    = os.path.join(out_train, "label_2")
    out_calib    = os.path.join(out_train, "calib")
    out_image    = os.path.join(out_train, "image_2")
    out_imgsets  = os.path.join(args.output, "ImageSets")

    if not args.dry_run:
        for d in (out_vel, out_label, out_calib, out_image, out_imgsets):
            os.makedirs(d, exist_ok=True)

    # Dummy image template
    template = os.path.join(out_image, "_dummy_template.png")
    if not args.dry_run:
        create_dummy_png(template)

    # ── Process frames ──────────────────────────────────────────────────────
    non_empty_ids = []  # IDs that have at least one labelled object
    total_objects = 0

    for idx, ts in enumerate(timestamps):
        fid = f"{idx:06d}"
        if idx % 5000 == 0:
            print(f"  [{idx}/{n_total}]  ts={ts} -> {fid}")
        if args.dry_run:
            continue

        # Velodyne (straight copy)
        shutil.copy2(
            os.path.join(src_vel, f"{ts}.bin"),
            os.path.join(out_vel, f"{fid}.bin"),
        )

        # Labels (transform)
        lbl_src = os.path.join(src_label, f"{ts}.txt")
        lbl_dst = os.path.join(out_label, f"{fid}.txt")
        lines = process_label_file(lbl_src) if os.path.exists(lbl_src) else []
        with open(lbl_dst, "w") as f:
            for l in lines:
                f.write(l + "\n")
        if lines:
            non_empty_ids.append(fid)
            total_objects += len(lines)

        # Calib (copy)
        cal_src = os.path.join(src_calib, f"{ts}.txt")
        if os.path.exists(cal_src):
            shutil.copy2(cal_src, os.path.join(out_calib, f"{fid}.txt"))

        # Image (hard-link from template)
        img_dst = os.path.join(out_image, f"{fid}.png")
        if not os.path.exists(img_dst):
            try:
                os.link(template, img_dst)
            except OSError:
                shutil.copy2(template, img_dst)

    if args.dry_run:
        print("Dry run — no files written.")
        return

    # ── Train / val split (non-empty frames only) ──────────────────────────
    n_val   = int(len(non_empty_ids) * args.val_ratio)
    n_train = len(non_empty_ids) - n_val

    if args.split_mode == "random":
        rng = random.Random(args.seed)
        shuffled = list(non_empty_ids)
        rng.shuffle(shuffled)
        train_ids = sorted(shuffled[:n_train])
        val_ids   = sorted(shuffled[n_train:])
    else:
        train_ids = non_empty_ids[:n_train]
        val_ids   = non_empty_ids[n_train:]

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", val_ids)]:
        path = os.path.join(out_imgsets, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(ids) + "\n")
        print(f"  ImageSets/{name}.txt -> {len(ids)} samples")

    # ── testing/ symlink (needed by create_kitti_infos) ─────────────────────
    testing_link = os.path.join(args.output, "testing")
    if os.path.islink(testing_link):
        os.remove(testing_link)
    if not os.path.exists(testing_link):
        os.symlink("training", testing_link)
        print("  Created testing -> training symlink")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"  Total frames:       {n_total}")
    print(f"  Frames with labels: {len(non_empty_ids)}")
    print(f"  Total objects:      {total_objects}")
    print(f"  Empty frames:       {n_total - len(non_empty_ids)}")
    print(f"  Train / Val:        {len(train_ids)} / {len(val_ids)}")
    print(f"  Output:             {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
