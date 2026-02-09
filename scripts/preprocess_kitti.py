#!/usr/bin/env python3
"""
Preprocess the IAC KITTI dataset for OpenPCDet PointPillars training.

Copies and transforms data from the raw export directory into the OpenPCDet
expected KITTI directory structure, applying:
  - Sequential 6-digit file renaming (sorted chronologically)
  - Label normalization (class -> "Car", z adjustment, ry conversion, fake 2D bbox)
  - Dummy image generation (required by OpenPCDet info generation)
  - Train/val ImageSets creation (80/20 split)

Usage:
    python preprocess_kitti.py \
        --input /p/cavalier/data/processed/2025_01_06_PolimoveUnimore_KITTI \
        --output /p/cavalier/jay/OpenPCDet/data/kitti \
        --split_mode random  # or "sequential"
"""

import argparse
import glob
import math
import os
import random
import shutil
import sys

# ---------------------------------------------------------------------------
# Constants (must match export_kitti.py)
# ---------------------------------------------------------------------------
OPPONENT_H = 1.5  # Height used in labels
KNOWN_CLASSES = {"Polimove", "Unimore", "KAIST", "Car"}
FAKE_BBOX = "0.00 0.00 50.00 50.00"  # Gives 2D height=51 -> difficulty "Easy"


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    angle = angle % (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle


def process_label_line(line):
    """
    Process a single KITTI label line:
      - Rename class to 'Car'
      - Adjust z by -H/2  (center -> bottom-center for KITTI convention)
      - Convert yaw -> ry = -yaw - pi/2
      - Replace 2D bbox with fake values
      - Strip extra fields (pitch, roll) to get exactly 16 fields
    Returns the transformed line, or None if the line is empty / should be skipped.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 15:
        return None  # Malformed

    cls_name = parts[0]
    if cls_name not in KNOWN_CLASSES:
        return None  # Skip unknown classes

    # --- Fields (0-indexed) ---
    # 0:  class
    # 1:  truncated
    # 2:  occluded
    # 3:  alpha
    # 4-7: bbox 2D (left, top, right, bottom)
    # 8:  h
    # 9:  w
    # 10: l
    # 11: x
    # 12: y
    # 13: z
    # 14: ry (yaw in our export)
    # 15: score (optional)

    truncated = parts[1]
    occluded = parts[2]
    alpha = parts[3]
    # Skip original bbox (parts[4:8])
    h = float(parts[8])
    w = float(parts[9])
    l = float(parts[10])
    x = float(parts[11])
    y = float(parts[12])
    z = float(parts[13])
    yaw = float(parts[14])  # This is the ego-frame heading from export
    score = float(parts[15]) if len(parts) >= 16 else 1.0

    # --- Adjustments ---
    # 1. Z: center -> bottom-center (OpenPCDet adds h/2 back internally)
    z_new = z - OPPONENT_H / 2.0

    # 2. ry: convert ego-frame yaw to KITTI ry
    #    OpenPCDet heading = -(pi/2 + ry), and we want heading = yaw
    #    => ry = -yaw - pi/2
    ry_new = normalize_angle(-yaw - math.pi / 2.0)

    # 3. Alpha: recompute for consistency
    #    alpha = ry - atan2(z_cam, x_cam)  [standard KITTI]
    #    With identity Tr, "camera" frame = ego frame, so:
    #    observation angle from ego to opponent
    angle_to_obj = math.atan2(y, x) if (x != 0 or y != 0) else 0.0
    alpha_new = normalize_angle(ry_new - angle_to_obj)

    # Build output: exactly 16 fields (15 standard + score)
    out = (
        f"Car {truncated} {occluded} {alpha_new:.2f} "
        f"{FAKE_BBOX} "
        f"{h:.2f} {w:.2f} {l:.2f} "
        f"{x:.2f} {y:.2f} {z_new:.2f} "
        f"{ry_new:.2f} {score:.2f}"
    )
    return out


def process_label_file(src_path):
    """Read a label file, process each line, return list of processed lines."""
    with open(src_path, "r") as f:
        raw_lines = f.readlines()

    processed = []
    for line in raw_lines:
        result = process_label_line(line)
        if result is not None:
            processed.append(result)
    return processed


def create_dummy_image(path, width=100, height=100):
    """Create a minimal black PNG image (no external deps)."""
    import struct
    import zlib

    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    # IDAT - raw image data (all zeros = black RGB)
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00" + b"\x00\x00\x00" * width  # filter byte + RGB
    compressed = zlib.compress(raw_data)
    idat = make_chunk(b"IDAT", compressed)

    # IEND
    iend = make_chunk(b"IEND", b"")

    # Write PNG
    signature = b"\x89PNG\r\n\x1a\n"
    with open(path, "wb") as f:
        f.write(signature + ihdr + idat + iend)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess IAC KITTI dataset for OpenPCDet"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw KITTI export (e.g. .../2025_01_06_PolimoveUnimore_KITTI)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to OpenPCDet kitti data dir (e.g. .../OpenPCDet/data/kitti)",
    )
    parser.add_argument(
        "--split_mode",
        choices=["random", "sequential"],
        default="random",
        help="How to split train/val: 'random' (shuffled 80/20) or 'sequential' (first 80%% train)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (only used with --split_mode random)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print stats but don't copy files",
    )
    args = parser.parse_args()

    # --- Validate input ---
    src_velodyne = os.path.join(args.input, "velodyne")
    src_label = os.path.join(args.input, "label_2")
    src_calib = os.path.join(args.input, "calib")

    for d in [src_velodyne, src_label, src_calib]:
        if not os.path.isdir(d):
            print(f"ERROR: Expected directory not found: {d}")
            sys.exit(1)

    # --- Discover files ---
    bin_files = sorted(glob.glob(os.path.join(src_velodyne, "*.bin")))
    print(f"Found {len(bin_files)} point cloud files in {src_velodyne}")

    if not bin_files:
        print("ERROR: No .bin files found!")
        sys.exit(1)

    # Extract timestamps and sort chronologically
    timestamps = []
    for bf in bin_files:
        ts = os.path.splitext(os.path.basename(bf))[0]
        timestamps.append(ts)

    # Timestamps are nanosecond integers - sorting lexicographically works
    # since they're all the same length, but let's be explicit
    timestamps.sort(key=lambda t: int(t))

    n_total = len(timestamps)
    print(f"Total frames: {n_total}")

    # Verify matching label and calib files
    missing_labels = 0
    missing_calibs = 0
    for ts in timestamps:
        if not os.path.exists(os.path.join(src_label, f"{ts}.txt")):
            missing_labels += 1
        if not os.path.exists(os.path.join(src_calib, f"{ts}.txt")):
            missing_calibs += 1

    if missing_labels > 0:
        print(f"WARNING: {missing_labels} timestamps missing label files")
    if missing_calibs > 0:
        print(f"WARNING: {missing_calibs} timestamps missing calib files")

    # --- Create output directories ---
    out_training = os.path.join(args.output, "training")
    out_velodyne = os.path.join(out_training, "velodyne")
    out_label = os.path.join(out_training, "label_2")
    out_calib = os.path.join(out_training, "calib")
    out_image = os.path.join(out_training, "image_2")
    out_imagesets = os.path.join(args.output, "ImageSets")

    if not args.dry_run:
        for d in [out_velodyne, out_label, out_calib, out_image, out_imagesets]:
            os.makedirs(d, exist_ok=True)

    # --- Create dummy image template ---
    dummy_img_path = os.path.join(out_image, "_dummy_template.png")
    if not args.dry_run:
        create_dummy_image(dummy_img_path, width=100, height=100)

    # --- Process and copy files ---
    ts_to_id = {}  # timestamp -> 6-digit ID string
    label_stats = {"total_objects": 0, "empty_frames": 0, "classes_seen": set()}

    print("Processing frames...")
    for idx, ts in enumerate(timestamps):
        new_id = f"{idx:06d}"
        ts_to_id[ts] = new_id

        if idx % 5000 == 0:
            print(f"  Frame {idx}/{n_total} (ts={ts} -> id={new_id})")

        if args.dry_run:
            # Still process labels for stats
            label_src = os.path.join(src_label, f"{ts}.txt")
            if os.path.exists(label_src):
                with open(label_src, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            label_stats["classes_seen"].add(parts[0])
            continue

        # 1. Copy velodyne .bin (straight copy, no transformation needed)
        vel_src = os.path.join(src_velodyne, f"{ts}.bin")
        vel_dst = os.path.join(out_velodyne, f"{new_id}.bin")
        shutil.copy2(vel_src, vel_dst)

        # 2. Process and write labels
        label_src = os.path.join(src_label, f"{ts}.txt")
        label_dst = os.path.join(out_label, f"{new_id}.txt")
        if os.path.exists(label_src):
            processed_lines = process_label_file(label_src)
            label_stats["total_objects"] += len(processed_lines)
            if not processed_lines:
                label_stats["empty_frames"] += 1
            with open(label_dst, "w") as f:
                for pl in processed_lines:
                    f.write(pl + "\n")
        else:
            # Write empty label file
            label_stats["empty_frames"] += 1
            with open(label_dst, "w") as f:
                pass

        # 3. Copy calib file as-is (identity transforms are fine)
        calib_src = os.path.join(src_calib, f"{ts}.txt")
        calib_dst = os.path.join(out_calib, f"{new_id}.txt")
        if os.path.exists(calib_src):
            shutil.copy2(calib_src, calib_dst)

        # 4. Create dummy image (hard link from template for efficiency)
        img_dst = os.path.join(out_image, f"{new_id}.png")
        if not os.path.exists(img_dst):
            try:
                os.link(dummy_img_path, img_dst)
            except OSError:
                # Fall back to copy if hard link fails (e.g., cross-device)
                shutil.copy2(dummy_img_path, img_dst)

    # --- Create train/val splits ---
    all_ids = [f"{i:06d}" for i in range(n_total)]

    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    if args.split_mode == "random":
        rng = random.Random(args.seed)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        train_ids = sorted(shuffled[:n_train])
        val_ids = sorted(shuffled[n_train:])
    else:  # sequential
        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train:]

    print(f"\nSplit ({args.split_mode}): {len(train_ids)} train, {len(val_ids)} val")

    if not args.dry_run:
        for name, ids in [("train", train_ids), ("val", val_ids), ("test", val_ids)]:
            path = os.path.join(out_imagesets, f"{name}.txt")
            with open(path, "w") as f:
                for sample_id in ids:
                    f.write(sample_id + "\n")
            print(f"  Wrote {path} ({len(ids)} samples)")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete!")
    print(f"  Total frames:   {n_total}")
    print(f"  Train / Val:    {len(train_ids)} / {len(val_ids)}")
    if not args.dry_run:
        print(f"  Total objects:  {label_stats['total_objects']}")
        print(f"  Empty frames:   {label_stats['empty_frames']}")
    else:
        print(f"  Classes seen:   {label_stats['classes_seen']}")
        print(f"  (dry run - no files written)")
    print(f"\nOutput directory: {args.output}")
    print(f"  training/velodyne/  -> {n_total} .bin files")
    print(f"  training/label_2/   -> {n_total} .txt files")
    print(f"  training/calib/     -> {n_total} .txt files")
    print(f"  training/image_2/   -> {n_total} .png files (dummy)")
    print(f"  ImageSets/          -> train.txt, val.txt, test.txt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
