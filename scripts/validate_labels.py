#!/usr/bin/env python3
"""
Validate LiDAR labels by visualizing OpenPCDet's exact data loading pipeline.

Uses the SAME KittiDataset class and config that training uses, so what you
see in the visualization is EXACTLY what the model receives (minus augmentation).
Generates interactive Plotly HTML files you can open in a browser.

Must be run from OpenPCDet/tools/ so configs resolve correctly.

Usage:
    cd /p/cavalier/jay/OpenPCDet/tools
    python /p/cavalier/jay/scripts/validate_labels.py [--samples 5] [--out /p/cavalier/jay/logs]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ── OpenPCDet imports (same as train.py) ────────────────────────────────────
# Must run from tools/ directory for config resolution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "OpenPCDet"))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.utils import common_utils

# ── Plotly (generate standalone HTML, no server needed) ─────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("ERROR: pip install plotly  (no other deps needed)")


# ── Box geometry ────────────────────────────────────────────────────────────

def boxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    """
    Convert (N, 7) boxes [x, y, z, dx, dy, dz, heading] to (N, 8, 3) corners.

    OpenPCDet convention (LiDAR frame):
        x = forward, y = left, z = up
        dx = length (along heading), dy = width, dz = height
        heading = rotation about z-axis (0 = facing +x)
        (x, y, z) = center of the box
    """
    n = boxes.shape[0]
    corners = np.zeros((n, 8, 3))

    for i in range(n):
        cx, cy, cz, dx, dy, dz, heading = boxes[i]

        # Half-extents
        hdx, hdy, hdz = dx / 2, dy / 2, dz / 2

        # 8 corners of axis-aligned box (before rotation)
        #   Order: bottom-face (CCW from front-right), then top-face same order
        #   "front" = +dx direction, "right" = -dy direction
        template = np.array([
            [ hdx, -hdy, -hdz],  # 0: front-right-bottom
            [ hdx,  hdy, -hdz],  # 1: front-left-bottom
            [-hdx,  hdy, -hdz],  # 2: rear-left-bottom
            [-hdx, -hdy, -hdz],  # 3: rear-right-bottom
            [ hdx, -hdy,  hdz],  # 4: front-right-top
            [ hdx,  hdy,  hdz],  # 5: front-left-top
            [-hdx,  hdy,  hdz],  # 6: rear-left-top
            [-hdx, -hdy,  hdz],  # 7: rear-right-top
        ])

        # Rotate about z-axis
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rot = np.array([
            [cos_h, -sin_h, 0],
            [sin_h,  cos_h, 0],
            [0,      0,     1],
        ])
        rotated = template @ rot.T  # (8, 3)

        # Translate to center
        corners[i] = rotated + np.array([cx, cy, cz])

    return corners


def box_wireframe_traces(corners: np.ndarray, box_info: list[str]) -> list:
    """
    Create Plotly traces for wireframe boxes.
    corners: (N, 8, 3)
    box_info: list of N label strings for hover text
    """
    traces = []
    # Edges: connect corners to form wireframe
    # Bottom face: 0-1-2-3-0, Top face: 4-5-6-7-4, Pillars: 0-4, 1-5, 2-6, 3-7
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # pillars
    ]
    # Also draw a direction indicator: midpoint of front-top edge
    for i in range(corners.shape[0]):
        c = corners[i]  # (8, 3)
        xs, ys, zs = [], [], []
        for a, b in edges:
            xs.extend([c[a, 0], c[b, 0], None])
            ys.extend([c[a, 1], c[b, 1], None])
            zs.extend([c[a, 2], c[b, 2], None])

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color="red", width=4),
            name=f"Box {i}",
            hovertext=box_info[i],
            hoverinfo="text",
            showlegend=(i == 0),
            legendgroup="boxes",
        ))

        # Heading arrow: center → front-face center
        center = c.mean(axis=0)
        front_center = (c[0] + c[1] + c[4] + c[5]) / 4  # front face center
        traces.append(go.Scatter3d(
            x=[center[0], front_center[0]],
            y=[center[1], front_center[1]],
            z=[center[2], front_center[2]],
            mode="lines",
            line=dict(color="yellow", width=6),
            name=f"Heading {i}",
            showlegend=False,
        ))

    return traces


# ── Main ────────────────────────────────────────────────────────────────────

def load_dataset_no_aug(cfg_file: str):
    """Load KittiDataset with all augmentation disabled (see true labels)."""
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger("validate.log", rank=0)

    # Disable ALL augmentation so we see the raw transformed labels
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        "gt_sampling",
        "random_world_flip",
        "random_world_rotation",
        "random_world_scaling",
    ]

    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH).resolve()
            if not Path(cfg.DATA_CONFIG.DATA_PATH).is_absolute()
            else Path(cfg.DATA_CONFIG.DATA_PATH),
        training=True,
        logger=logger,
    )
    return dataset


def read_raw_label(dataset, idx: int) -> str:
    """Read the raw label file for a given dataset index."""
    info = dataset.kitti_infos[idx]
    sample_idx = info["point_cloud"]["lidar_idx"]
    label_file = dataset.root_path / "training" / "label_2" / f"{sample_idx}.txt"
    if label_file.exists():
        return label_file.read_text()
    return "(no label file)"


def visualize_sample(dataset, idx: int, out_dir: Path):
    """Load one sample through OpenPCDet and create a Plotly visualization."""
    info = dataset.kitti_infos[idx]
    sample_idx = info["point_cloud"]["lidar_idx"]

    # ── Load through OpenPCDet's exact pipeline (minus augmentation) ────────
    data_dict = dataset[idx]

    points = data_dict["points"]  # (N, 5) or (N, 4): x, y, z, intensity, [idx]
    gt_boxes = data_dict.get("gt_boxes", np.zeros((0, 7)))  # (M, 7+): x,y,z,dx,dy,dz,heading,[class]

    if gt_boxes.ndim == 2 and gt_boxes.shape[1] > 7:
        gt_boxes = gt_boxes[:, :7]  # strip class label column

    n_pts = points.shape[0]
    n_boxes = gt_boxes.shape[0]
    raw_label = read_raw_label(dataset, idx)

    print(f"  Sample {sample_idx}: {n_pts} points, {n_boxes} boxes")

    # ── Build Plotly figure ─────────────────────────────────────────────────
    # Subsample points for performance (Plotly struggles with >100K points)
    max_pts = 80000
    if n_pts > max_pts:
        choice = np.random.choice(n_pts, max_pts, replace=False)
        vis_pts = points[choice]
    else:
        vis_pts = points

    fig = go.Figure()

    # Point cloud colored by Z (height)
    fig.add_trace(go.Scatter3d(
        x=vis_pts[:, 0], y=vis_pts[:, 1], z=vis_pts[:, 2],
        mode="markers",
        marker=dict(
            size=1,
            color=vis_pts[:, 2],
            colorscale="Viridis",
            colorbar=dict(title="Z (m)", x=1.02),
            opacity=0.6,
        ),
        name=f"Points ({n_pts})",
        hovertemplate="x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<extra></extra>",
    ))

    # Ego vehicle marker at origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=6, color="white", symbol="diamond"),
        text=["EGO"],
        textposition="top center",
        textfont=dict(color="white", size=12),
        name="Ego vehicle",
    ))

    # Bounding boxes
    if n_boxes > 0:
        corners = boxes_to_corners(gt_boxes)  # (M, 8, 3)
        box_labels = []
        for i in range(n_boxes):
            bx = gt_boxes[i]
            box_labels.append(
                f"Box {i}<br>"
                f"center=({bx[0]:.1f}, {bx[1]:.1f}, {bx[2]:.1f})<br>"
                f"size=({bx[3]:.1f}L x {bx[4]:.1f}W x {bx[5]:.1f}H)<br>"
                f"heading={bx[6]:.2f} rad ({math.degrees(bx[6]):.1f}°)"
            )
        fig.add_traces(box_wireframe_traces(corners, box_labels))

    # Coordinate axes at origin (length 5m)
    for axis, color, label in [
        ([5, 0, 0], "red",   "+X (forward)"),
        ([0, 5, 0], "green", "+Y (left)"),
        ([0, 0, 5], "blue",  "+Z (up)"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis[0]], y=[0, axis[1]], z=[0, axis[2]],
            mode="lines+text",
            line=dict(color=color, width=5),
            text=["", label],
            textposition="top center",
            textfont=dict(color=color, size=10),
            name=label,
            showlegend=False,
        ))

    # Layout
    raw_lines = raw_label.strip().split("\n")
    raw_preview = "<br>".join(raw_lines[:5])
    if len(raw_lines) > 5:
        raw_preview += f"<br>... ({len(raw_lines)} lines total)"

    fig.update_layout(
        title=dict(text=(
            f"Sample {sample_idx} — {n_pts} points, {n_boxes} boxes<br>"
            f"<sub>Loaded via OpenPCDet KittiDataset (no augmentation) | "
            f"Red wireframes = gt_boxes | Yellow = heading direction</sub>"
        )),
        scene=dict(
            xaxis_title="X (forward, m)",
            yaxis_title="Y (left, m)",
            zaxis_title="Z (up, m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=-0.5, y=-1.0, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_dark",
        width=1400,
        height=900,
        annotations=[
            dict(
                text=(
                    f"<b>Raw label file ({sample_idx}.txt):</b><br>{raw_preview}<br><br>"
                    f"<b>OpenPCDet gt_boxes (LiDAR frame):</b><br>"
                    + "<br>".join(
                        f"  [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, "
                        f"{b[3]:.1f}L, {b[4]:.1f}W, {b[5]:.1f}H, "
                        f"h={b[6]:.2f}]"
                        for b in gt_boxes
                    ) if n_boxes > 0 else "  (no boxes)"
                ),
                xref="paper", yref="paper",
                x=0.01, y=-0.02,
                showarrow=False,
                font=dict(family="monospace", size=11, color="lightgray"),
                align="left",
                bgcolor="rgba(0,0,0,0.7)",
                borderpad=8,
            )
        ],
    )

    out_file = out_dir / f"validate_{sample_idx}.html"
    fig.write_html(str(out_file), include_plotlyjs="cdn")
    print(f"    -> {out_file}")
    return out_file


def main():
    ap = argparse.ArgumentParser(description="Validate LiDAR labels via OpenPCDet + Plotly")
    ap.add_argument("--cfg", default="cfgs/kitti_models/pointpillar.yaml",
                     help="Model config (relative to tools/)")
    ap.add_argument("--samples", type=int, default=5,
                     help="Number of random samples to visualize")
    ap.add_argument("--indices", type=str, default=None,
                     help="Comma-separated dataset indices to visualize (overrides --samples)")
    ap.add_argument("--out", type=str, default="/p/cavalier/jay/logs",
                     help="Output directory for HTML files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset (augmentation disabled)...")
    dataset = load_dataset_no_aug(args.cfg)
    n = len(dataset)
    print(f"Dataset has {n} samples")

    # Select indices
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    else:
        rng = np.random.RandomState(args.seed)
        indices = sorted(rng.choice(n, min(args.samples, n), replace=False))

    print(f"Visualizing {len(indices)} samples: {indices}")

    html_files = []
    for idx in indices:
        html_files.append(visualize_sample(dataset, idx, out_dir))

    # Create an index page linking to all samples
    index_html = out_dir / "validate_index.html"
    links = "\n".join(
        f'<li><a href="{f.name}" target="_blank">{f.stem}</a></li>'
        for f in html_files
    )
    index_html.write_text(f"""<!DOCTYPE html>
<html><head><title>Label Validation Index</title></head>
<body style="font-family: sans-serif; padding: 20px; background: #1a1a2e; color: #eee;">
<h1>LiDAR Label Validation</h1>
<p>Generated via OpenPCDet KittiDataset (no augmentation).<br>
Red wireframes = ground truth boxes. Yellow arrows = heading direction.<br>
Coordinate axes at origin: <span style="color:red">+X forward</span>,
<span style="color:green">+Y left</span>,
<span style="color:blue">+Z up</span>.</p>
<ul>{links}</ul>
</body></html>""")
    print(f"\nIndex page: {index_html}")
    print("Open in browser to inspect.")


if __name__ == "__main__":
    main()
