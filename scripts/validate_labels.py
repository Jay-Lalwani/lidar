#!/usr/bin/env python3
"""
Validate LiDAR labels by visualizing OpenPCDet's exact data loading pipeline.

Uses the SAME KittiDataset class and config that training uses, so what you
see in the visualization is EXACTLY what the model receives (minus augmentation).
Generates interactive Plotly HTML files you can open in a browser.

Optionally overlay model predictions by passing --ckpt <path_to_checkpoint>.

Must be run from OpenPCDet/tools/ so configs resolve correctly.

Usage:
    cd /p/cavalier/jay/OpenPCDet/tools
    python /p/cavalier/jay/scripts/validate_labels.py [--samples 5] [--out /p/cavalier/jay/logs]
    python /p/cavalier/jay/scripts/validate_labels.py --indices 7919 --ckpt ../output/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth
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
from pcdet.models import build_network
from pcdet.utils import common_utils

# ── Plotly (generate standalone HTML, no server needed) ─────────────────────
try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("ERROR: pip install plotly  (no other deps needed)")


# ── Point-in-box utility ───────────────────────────────────────────────────

def points_in_box(points, box):
    """
    Find which points fall inside a 3D oriented bounding box.

    Args:
        points: (N, 3+) array
        box: (7,) array [cx, cy, cz, dx, dy, dz, heading]

    Returns:
        mask: (N,) boolean mask of points inside the box
    """
    cx, cy, cz, dx, dy, dz, heading = box
    pts = points[:, :3] - np.array([cx, cy, cz])
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    rot = np.array([[cos_h, -sin_h, 0],
                    [sin_h,  cos_h, 0],
                    [0,      0,     1]])
    pts_aligned = pts @ rot.T
    return ((np.abs(pts_aligned[:, 0]) <= dx / 2) &
            (np.abs(pts_aligned[:, 1]) <= dy / 2) &
            (np.abs(pts_aligned[:, 2]) <= dz / 2))


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

        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rot = np.array([[cos_h, -sin_h, 0],
                        [sin_h,  cos_h, 0],
                        [0,      0,     1]])
        corners[i] = (template @ rot.T) + np.array([cx, cy, cz])

    return corners


def box_wireframe_traces(corners: np.ndarray, box_info: list[str],
                         color="red", name_prefix="Box",
                         heading_color="yellow") -> list:
    """
    Create Plotly traces for wireframe boxes.
    corners: (N, 8, 3)
    box_info: list of N label strings for hover text
    """
    traces = []
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # pillars
    ]
    for i in range(corners.shape[0]):
        c = corners[i]
        xs, ys, zs = [], [], []
        for a, b in edges:
            xs.extend([c[a, 0], c[b, 0], None])
            ys.extend([c[a, 1], c[b, 1], None])
            zs.extend([c[a, 2], c[b, 2], None])

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=4),
            name=f"{name_prefix} {i}",
            hovertext=box_info[i],
            hoverinfo="text",
            showlegend=(i == 0),
            legendgroup=name_prefix,
        ))

        # Heading arrow: center → front-face center
        center = c.mean(axis=0)
        front_center = (c[0] + c[1] + c[4] + c[5]) / 4
        traces.append(go.Scatter3d(
            x=[center[0], front_center[0]],
            y=[center[1], front_center[1]],
            z=[center[2], front_center[2]],
            mode="lines",
            line=dict(color=heading_color, width=6),
            name=f"{name_prefix} Heading {i}",
            showlegend=False,
        ))

    return traces


# ── Dataset loading ─────────────────────────────────────────────────────────

def load_dataset_no_aug(cfg_file: str, split: str = 'train'):
    """Load KittiDataset with all augmentation disabled (see true labels)."""
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger("validate.log", rank=0)

    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = [
        "gt_sampling", "random_world_flip",
        "random_world_rotation", "random_world_scaling",
    ]

    # If split is 'val', set training=False to load validation set (defined in config as 'test': 'val')
    training = (split == 'train')

    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH).resolve()
            if not Path(cfg.DATA_CONFIG.DATA_PATH).is_absolute()
            else Path(cfg.DATA_CONFIG.DATA_PATH),
        training=training,
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


def run_inference(model, dataset, idx):
    """Run model inference on a single sample. Returns (pred_boxes, pred_scores)."""
    import torch
    data_dict = dataset[idx]
    batch = dataset.collate_batch([data_dict])
    for key, val in batch.items():
        if not isinstance(val, np.ndarray):
            continue
        if val.dtype.kind in {'U', 'S'}:
            continue
        batch[key] = torch.from_numpy(val).float().cuda()

    with torch.no_grad():
        pred_dicts, _ = model(batch)
    pred = pred_dicts[0]
    return pred['pred_boxes'].cpu().numpy(), pred['pred_scores'].cpu().numpy()


# ── Custom HTML/JS controls ────────────────────────────────────────────────

CUSTOM_CONTROLS_CSS = """
<style>
  #controls-panel {
    position: fixed; top: 10px; right: 10px; z-index: 9999;
    background: rgba(20, 20, 35, 0.95); border: 1px solid #444;
    border-radius: 10px; padding: 16px 20px; min-width: 280px;
    font-family: 'Segoe UI', system-ui, sans-serif; color: #ddd;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5); font-size: 13px;
    max-height: 90vh; overflow-y: auto;
    backdrop-filter: blur(8px);
  }
  #controls-panel h3 {
    margin: 0 0 12px 0; font-size: 15px; color: #fff;
    border-bottom: 1px solid #555; padding-bottom: 8px;
    letter-spacing: 0.5px;
  }
  #controls-panel .ctrl-group {
    margin-bottom: 10px;
  }
  #controls-panel label {
    display: block; margin-bottom: 4px; font-size: 12px;
    color: #aaa; text-transform: uppercase; letter-spacing: 0.5px;
  }
  #controls-panel input[type=range] {
    width: 100%; accent-color: #6c8cff; cursor: pointer;
  }
  #controls-panel .val-display {
    float: right; color: #8af; font-weight: 600;
  }
  #controls-panel select {
    width: 100%; padding: 4px 6px; background: #2a2a3e;
    border: 1px solid #555; border-radius: 4px; color: #ddd;
    font-size: 13px; cursor: pointer;
  }
  #controls-panel .toggle-btn {
    display: inline-block; padding: 4px 10px; margin: 2px 3px 2px 0;
    background: #2a2a3e; border: 1px solid #555; border-radius: 4px;
    color: #ddd; cursor: pointer; font-size: 12px;
    transition: all 0.15s;
  }
  #controls-panel .toggle-btn.active {
    background: #3a5a9f; border-color: #6c8cff; color: #fff;
  }
  #controls-panel .toggle-btn:hover {
    background: #3a3a5e;
  }
  #controls-panel .stat-row {
    display: flex; justify-content: space-between;
    padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  #controls-panel .stat-label { color: #999; }
  #controls-panel .stat-value { color: #cde; font-weight: 500; }
  #controls-panel .section-header {
    font-size: 12px; color: #8af; margin: 12px 0 6px 0;
    text-transform: uppercase; letter-spacing: 1px;
    font-weight: 600;
  }
  #collapse-btn {
    position: absolute; top: 8px; right: 12px;
    background: none; border: none; color: #888; cursor: pointer;
    font-size: 18px; line-height: 1;
  }
  #collapse-btn:hover { color: #fff; }
  #controls-body.collapsed { display: none; }
  .box-detail-card {
    background: rgba(255,255,255,0.04); border-radius: 6px;
    padding: 8px 10px; margin-bottom: 6px;
    border-left: 3px solid;
  }
  .box-detail-card.gt { border-left-color: #ff4444; }
  .box-detail-card.pred { border-left-color: #00ff88; }
  .box-detail-card .box-title {
    font-weight: 600; font-size: 13px; margin-bottom: 4px;
  }
  .box-detail-card .box-title.gt { color: #ff6666; }
  .box-detail-card .box-title.pred { color: #00ff88; }
  .snap-btn {
    background: none; border: none; color: #8af;
    cursor: pointer; padding: 0 4px; font-size: 14px;
    float: right;
  }
  .snap-btn:hover { color: #fff; text-shadow: 0 0 5px #8af; }
</style>
"""


def build_controls_html(sample_idx, n_pts, gt_boxes, gt_point_counts,
                        gt_distances, pred_boxes, pred_scores,
                        pred_point_counts, pred_distances,
                        point_cloud_range, raw_label):
    """Build the HTML for the interactive control panel."""

    # ── Stats section ──
    stats_html = f"""
    <div class="section-header">📊 Scene Statistics</div>
    <div class="stat-row"><span class="stat-label">Sample ID</span><span class="stat-value">{sample_idx}</span></div>
    <div class="stat-row"><span class="stat-label">Total Points</span><span class="stat-value">{n_pts:,}</span></div>
    <div class="stat-row"><span class="stat-label">GT Boxes</span><span class="stat-value">{len(gt_boxes)}</span></div>
    <div class="stat-row"><span class="stat-label">Predictions</span><span class="stat-value">{len(pred_boxes)}</span></div>
    <div class="stat-row"><span class="stat-label">PC Range X</span><span class="stat-value">[{point_cloud_range[0]:.0f}, {point_cloud_range[3]:.0f}]</span></div>
    <div class="stat-row"><span class="stat-label">PC Range Y</span><span class="stat-value">[{point_cloud_range[1]:.0f}, {point_cloud_range[4]:.0f}]</span></div>
    <div class="stat-row"><span class="stat-label">PC Range Z</span><span class="stat-value">[{point_cloud_range[2]:.0f}, {point_cloud_range[5]:.0f}]</span></div>
    """

    # ── GT box details ──
    gt_cards = ""
    if len(gt_boxes) > 0:
        gt_cards += '<div class="section-header">🟥 Ground Truth Boxes</div>'
        for i, bx in enumerate(gt_boxes):
            dist = gt_distances[i]
            pts = gt_point_counts[i]
            heading_deg = math.degrees(bx[6])
            gt_cards += f"""
            <div class="box-detail-card gt">
              <div class="box-title gt">
                GT {i}
                <button class="snap-btn" title="Snap to Box"
                        onclick="snapToBox({bx[0]:.2f}, {bx[1]:.2f}, {bx[2]:.2f})">⌖</button>
              </div>
              <div class="stat-row"><span class="stat-label">Distance</span><span class="stat-value">{dist:.1f} m</span></div>
              <div class="stat-row"><span class="stat-label">Points Inside</span><span class="stat-value">{pts}</span></div>
              <div class="stat-row"><span class="stat-label">Center</span><span class="stat-value">({bx[0]:.1f}, {bx[1]:.1f}, {bx[2]:.1f})</span></div>
              <div class="stat-row"><span class="stat-label">Size (L×W×H)</span><span class="stat-value">{bx[3]:.2f} × {bx[4]:.2f} × {bx[5]:.2f}</span></div>
              <div class="stat-row"><span class="stat-label">Heading</span><span class="stat-value">{bx[6]:.2f} rad ({heading_deg:.1f}°)</span></div>
              <div class="stat-row"><span class="stat-label">Z bottom</span><span class="stat-value">{bx[2] - bx[5]/2:.2f} m</span></div>
            </div>"""

    # ── Pred box details ──
    pred_cards = ""
    if len(pred_boxes) > 0:
        pred_cards += '<div class="section-header">🟩 Predicted Boxes</div>'
        for i in range(len(pred_boxes)):
            bx = pred_boxes[i]
            dist = pred_distances[i]
            pts = pred_point_counts[i]
            score = pred_scores[i]
            heading_deg = math.degrees(bx[6])
            pred_cards += f"""
            <div class="box-detail-card pred">
              <div class="box-title pred">
                Pred {i} (score: {score:.3f})
                <button class="snap-btn" title="Snap to Box"
                        onclick="snapToBox({bx[0]:.2f}, {bx[1]:.2f}, {bx[2]:.2f})">⌖</button>
              </div>
              <div class="stat-row"><span class="stat-label">Distance</span><span class="stat-value">{dist:.1f} m</span></div>
              <div class="stat-row"><span class="stat-label">Points Inside</span><span class="stat-value">{pts}</span></div>
              <div class="stat-row"><span class="stat-label">Center</span><span class="stat-value">({bx[0]:.1f}, {bx[1]:.1f}, {bx[2]:.1f})</span></div>
              <div class="stat-row"><span class="stat-label">Size (L×W×H)</span><span class="stat-value">{bx[3]:.2f} × {bx[4]:.2f} × {bx[5]:.2f}</span></div>
              <div class="stat-row"><span class="stat-label">Heading</span><span class="stat-value">{bx[6]:.2f} rad ({heading_deg:.1f}°)</span></div>
              <div class="stat-row"><span class="stat-label">Z bottom</span><span class="stat-value">{bx[2] - bx[5]/2:.2f} m</span></div>
            </div>"""

    # ── Raw label preview ──
    raw_lines = raw_label.strip().split("\n")
    raw_preview = "<br>".join(raw_lines[:6])
    if len(raw_lines) > 6:
        raw_preview += f"<br>... ({len(raw_lines)} lines total)"

    return f"""
    {CUSTOM_CONTROLS_CSS}
    <div id="controls-panel">
      <button id="collapse-btn" onclick="
        document.getElementById('controls-body').classList.toggle('collapsed');
        this.textContent = this.textContent === '−' ? '+' : '−';
      ">−</button>
      <h3>🎛️ Visualization Controls</h3>
      <div id="controls-body">

        <div class="ctrl-group">
          <label>Point Size <span class="val-display" id="ps-val">1.0</span></label>
          <input type="range" id="point-size-slider" min="0.2" max="5" step="0.1" value="1.0"
                 oninput="updatePointSize(this.value)">
        </div>

        <div class="ctrl-group">
          <label>Point Opacity <span class="val-display" id="po-val">0.6</span></label>
          <input type="range" id="point-opacity-slider" min="0.05" max="1" step="0.05" value="0.6"
                 oninput="updatePointOpacity(this.value)">
        </div>

        <div class="ctrl-group">
          <label>Box Line Width <span class="val-display" id="bw-val">4</span></label>
          <input type="range" id="box-width-slider" min="1" max="10" step="0.5" value="4"
                 oninput="updateBoxWidth(this.value)">
        </div>

        <div class="ctrl-group">
          <label>Color By</label>
          <select id="color-mode" onchange="updateColorMode(this.value)">
            <option value="z">Height (Z)</option>
            <option value="distance">Distance from Ego</option>
            <option value="intensity">Intensity</option>
            <option value="x">X (Forward)</option>
            <option value="y">Y (Lateral)</option>
          </select>
        </div>

        <div class="ctrl-group">
          <label>Toggle Layers</label>
          <span class="toggle-btn active" onclick="toggleLayer(this, 'points')">Points</span>
          <span class="toggle-btn active" onclick="toggleLayer(this, 'gt')">GT Boxes</span>
          <span class="toggle-btn active" onclick="toggleLayer(this, 'pred')">Predictions</span>
          <span class="toggle-btn active" onclick="toggleLayer(this, 'axes')">Axes</span>
          <span class="toggle-btn active" onclick="toggleLayer(this, 'ego')">Ego</span>
          <span class="toggle-btn active" onclick="toggleLayer(this, 'rings')">Range Rings</span>
        </div>

        {stats_html}
        {gt_cards}
        {pred_cards}

        <div class="section-header">📄 Raw Label File</div>
        <div style="font-family: monospace; font-size: 11px; color: #999; word-break: break-all;">
          {raw_preview}
        </div>

      </div>
    </div>
    """


def build_controls_js(point_data, pc_range):
    """Build JavaScript for interactive controls."""
    return f"""
    <script>
    // Store point data for re-coloring
    var pointData = {json.dumps(point_data)};
    var pcRange = {json.dumps(pc_range)};

    function getPlotDiv() {{
      return document.querySelector('.plotly-graph-div') || document.querySelector('[class*="js-plotly"]');
    }}

    function updatePointSize(val) {{
      document.getElementById('ps-val').textContent = parseFloat(val).toFixed(1);
      var gd = getPlotDiv();
      if (gd) Plotly.restyle(gd, {{'marker.size': parseFloat(val)}}, [0]);
    }}

    function updatePointOpacity(val) {{
      document.getElementById('po-val').textContent = parseFloat(val).toFixed(2);
      var gd = getPlotDiv();
      if (gd) Plotly.restyle(gd, {{'marker.opacity': parseFloat(val)}}, [0]);
    }}

    function updateBoxWidth(val) {{
      document.getElementById('bw-val').textContent = parseFloat(val).toFixed(1);
      var gd = getPlotDiv();
      if (!gd) return;
      // Update all box traces (indices > 1, i.e. not points or ego)
      var indices = [];
      for (var i = 2; i < gd.data.length; i++) {{
        if (gd.data[i].mode === 'lines' && gd.data[i].line) {{
          indices.push(i);
        }}
      }}
      if (indices.length > 0) Plotly.restyle(gd, {{'line.width': parseFloat(val)}}, indices);
    }}

    function updateColorMode(mode) {{
      var gd = getPlotDiv();
      if (!gd) return;
      var colors, title;
      switch(mode) {{
        case 'z':
          colors = pointData.z; title = 'Z (m)'; break;
        case 'distance':
          colors = pointData.distance; title = 'Distance (m)'; break;
        case 'intensity':
          colors = pointData.intensity; title = 'Intensity'; break;
        case 'x':
          colors = pointData.x; title = 'X (m)'; break;
        case 'y':
          colors = pointData.y; title = 'Y (m)'; break;
      }}
      Plotly.restyle(gd, {{
        'marker.color': [colors],
        'marker.colorbar.title.text': title
      }}, [0]);
    }}

    function toggleLayer(btn, group) {{
      var gd = getPlotDiv();
      if (!gd) return;
      btn.classList.toggle('active');
      var vis = btn.classList.contains('active');

      for (var i = 0; i < gd.data.length; i++) {{
        var trace = gd.data[i];
        var match = false;
        switch(group) {{
          case 'points':  match = (i === 0); break;
          case 'ego':     match = (trace.name && trace.name.includes('Ego')); break;
          case 'gt':      match = (trace.name && (trace.name.startsWith('GT') || trace.name.includes('GT Heading'))); break;
          case 'pred':    match = (trace.name && (trace.name.startsWith('Pred') || trace.name.includes('Pred Heading'))); break;
          case 'axes':    match = (trace.name && trace.name.startsWith('+')); break;
          case 'rings':   match = (trace.name && trace.name.includes('Range')); break;
        }}
        if (match) {{
          Plotly.restyle(gd, {{'visible': vis}}, [i]);
        }}
      }}
    }}

    function snapToBox(cx, cy, cz) {{
      // Snap camera to look at box center (cx, cy, cz)
      // We need to normalize coordinates to the scene's bounding box
      var xMin = pcRange[0], yMin = pcRange[1], zMin = pcRange[2];
      var xMax = pcRange[3], yMax = pcRange[4], zMax = pcRange[5];

      var spanX = xMax - xMin;
      var spanY = yMax - yMin;
      var spanZ = zMax - zMin;
      var maxSpan = Math.max(spanX, spanY, spanZ);

      var sceneCenterX = (xMin + xMax) / 2;
      var sceneCenterY = (yMin + yMax) / 2;
      var sceneCenterZ = (zMin + zMax) / 2;

      // Normalized target position (center of camera view)
      var tx = (cx - sceneCenterX) / maxSpan;
      var ty = (cy - sceneCenterY) / maxSpan;
      var tz = (cz - sceneCenterZ) / maxSpan;

      // Camera position (eye)
      // Offset to be "zoomed in" closer to the target
      // Standard view diagonal is large. We want to be close.
      var zoomDist = 0.25; // Distance from target in normalized units
      
      // Default angle: Look from top-back-left
      // Direction vector pointing from camera to target
      var dirX = 1, dirY = 1, dirZ = -1; 
      // Eye = Target - (Direction * zoomDist)
      // So if looking down (-z), eye z is higher.
      
      var ex = tx - (dirX * zoomDist * 0.5); // weak offset in X
      var ey = ty - (dirY * zoomDist * 1.5); // strong offset in Y (from side?)
      var ez = tz + (0.5 * zoomDist);        // above
      
      // Let's use a simpler fixed offset relative to current view or fixed "shoulder view"
      // View from (-0.5, -0.5, 0.5) direction relative to target
      var ex = tx - 0.2;
      var ey = ty - 0.2;
      var ez = tz + 0.2;

      var gd = getPlotDiv();
      if (gd) {{
        Plotly.relayout(gd, {{
          'scene.camera.center': {{x: tx, y: ty, z: tz}},
          'scene.camera.eye': {{x: ex, y: ey, z: ez}}
        }});
      }}
    }}
    </script>
    """


# ── Main visualization ─────────────────────────────────────────────────────

def visualize_sample(dataset, idx: int, out_dir: Path, model=None,
                     score_thresh: float = 0.25):
    """Load one sample through OpenPCDet and create a Plotly visualization."""
    info = dataset.kitti_infos[idx]
    sample_idx = info["point_cloud"]["lidar_idx"]

    # ── Load through OpenPCDet's exact pipeline (minus augmentation) ────────
    data_dict = dataset[idx]

    points = data_dict["points"]  # (N, 5) or (N, 4)
    gt_boxes = data_dict.get("gt_boxes", np.zeros((0, 7)))

    if gt_boxes.ndim == 2 and gt_boxes.shape[1] > 7:
        gt_boxes = gt_boxes[:, :7]

    n_pts = points.shape[0]
    n_boxes = gt_boxes.shape[0]
    raw_label = read_raw_label(dataset, idx)

    # ── Compute per-point distance from ego ─────────────────────────────────
    pt_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    pt_intensity = points[:, 3] if points.shape[1] > 3 else np.zeros(n_pts)

    # ── Compute point counts inside GT boxes ────────────────────────────────
    gt_point_counts = []
    gt_distances = []
    for i in range(n_boxes):
        mask = points_in_box(points, gt_boxes[i])
        gt_point_counts.append(int(mask.sum()))
        gt_distances.append(float(np.sqrt(
            gt_boxes[i, 0]**2 + gt_boxes[i, 1]**2)))

    # ── Optional model predictions ──────────────────────────────────────────
    pred_boxes = np.zeros((0, 7))
    pred_scores = np.zeros(0)
    if model is not None:
        pb, ps = run_inference(model, dataset, idx)
        mask = ps > score_thresh
        pred_boxes = pb[mask]
        pred_scores = ps[mask]
    n_preds = pred_boxes.shape[0]

    # Compute point counts inside predicted boxes
    pred_point_counts = []
    pred_distances = []
    for i in range(n_preds):
        mask = points_in_box(points, pred_boxes[i])
        pred_point_counts.append(int(mask.sum()))
        pred_distances.append(float(np.sqrt(
            pred_boxes[i, 0]**2 + pred_boxes[i, 1]**2)))

    print(f"  Sample {sample_idx}: {n_pts} points, {n_boxes} GT boxes"
          + (f", {n_preds} preds (>{score_thresh})" if model else ""))

    # ── Build Plotly figure ─────────────────────────────────────────────────
    fig = go.Figure()

    # Point cloud — hover shows x, y, z, distance, intensity
    hover_texts = [
        f"x={points[i,0]:.2f}  y={points[i,1]:.2f}  z={points[i,2]:.2f}<br>"
        f"dist={pt_dist[i]:.1f}m  int={pt_intensity[i]:.2f}"
        for i in range(n_pts)
    ]

    fig.add_trace(go.Scatter3d(
        x=points[:, 0].tolist(),
        y=points[:, 1].tolist(),
        z=points[:, 2].tolist(),
        mode="markers",
        marker=dict(
            size=1,
            color=points[:, 2].tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Z (m)", x=1.02, len=0.5, y=0.75),
            opacity=0.6,
        ),
        name=f"Points ({n_pts:,})",
        hovertext=hover_texts,
        hoverinfo="text",
    ))

    # Ego vehicle marker at origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=8, color="white", symbol="diamond"),
        text=["EGO"],
        textposition="top center",
        textfont=dict(color="white", size=12),
        name="Ego vehicle",
    ))

    # ── Range rings (BEV circles at 25m, 50m, 75m, 100m, 150m, 200m) ──────
    for radius in [25, 50, 75, 100, 150, 200]:
        theta = np.linspace(0, 2 * np.pi, 100)
        ring_x = radius * np.cos(theta)
        ring_y = radius * np.sin(theta)
        ring_z = np.full_like(ring_x, -1.5)  # Draw at ground level
        fig.add_trace(go.Scatter3d(
            x=ring_x.tolist(), y=ring_y.tolist(), z=ring_z.tolist(),
            mode="lines",
            line=dict(color="rgba(100,150,255,0.25)", width=1, dash="dot"),
            name=f"Range {radius}m",
            showlegend=False,
            hovertext=f"{radius}m",
            hoverinfo="text",
        ))

    # ── GT bounding boxes with rich hover info ──────────────────────────────
    if n_boxes > 0:
        corners = boxes_to_corners(gt_boxes)
        box_labels = []
        for i in range(n_boxes):
            bx = gt_boxes[i]
            box_labels.append(
                f"<b>GT {i}</b><br>"
                f"distance: {gt_distances[i]:.1f} m<br>"
                f"points inside: {gt_point_counts[i]}<br>"
                f"center: ({bx[0]:.1f}, {bx[1]:.1f}, {bx[2]:.1f})<br>"
                f"size: {bx[3]:.2f}L × {bx[4]:.2f}W × {bx[5]:.2f}H<br>"
                f"heading: {bx[6]:.2f} rad ({math.degrees(bx[6]):.1f}°)<br>"
                f"z-bottom: {bx[2] - bx[5]/2:.2f} m"
            )
        fig.add_traces(box_wireframe_traces(corners, box_labels,
                                              color="red", name_prefix="GT"))

    # ── Predicted boxes with rich hover info ────────────────────────────────
    if n_preds > 0:
        pred_corners = boxes_to_corners(pred_boxes)
        pred_labels = [
            f"<b>Pred {i}</b> (score: {pred_scores[i]:.3f})<br>"
            f"distance: {pred_distances[i]:.1f} m<br>"
            f"points inside: {pred_point_counts[i]}<br>"
            f"center: ({pred_boxes[i,0]:.1f}, {pred_boxes[i,1]:.1f}, {pred_boxes[i,2]:.1f})<br>"
            f"size: {pred_boxes[i,3]:.2f}L × {pred_boxes[i,4]:.2f}W × {pred_boxes[i,5]:.2f}H<br>"
            f"heading: {pred_boxes[i,6]:.2f} rad ({math.degrees(pred_boxes[i,6]):.1f}°)<br>"
            f"z-bottom: {pred_boxes[i,2] - pred_boxes[i,5]/2:.2f} m"
            for i in range(n_preds)
        ]
        fig.add_traces(box_wireframe_traces(pred_corners, pred_labels,
                                              color="#00FF00", name_prefix="Pred",
                                              heading_color="lightgreen"))

    # ── Coordinate axes (length 10m) ────────────────────────────────────────
    for axis, color, label in [
        ([10, 0, 0], "red",   "+X (forward)"),
        ([0, 10, 0], "green", "+Y (left)"),
        ([0, 0, 5],  "blue",  "+Z (up)"),
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

    # ── Layout ──────────────────────────────────────────────────────────────
    pred_subtitle = ""
    if model is not None:
        pred_subtitle = f" | 🟩 Preds ({n_preds} > {score_thresh})"
    fig.update_layout(
        title=dict(text=(
            f"<b>Sample {sample_idx}</b> — {n_pts:,} points, "
            f"{n_boxes} GT boxes{pred_subtitle}<br>"
            f"<sub>OpenPCDet KittiDataset (no aug) | "
            f"🟥 GT | 🟡 Heading | Use the panel on the right for controls</sub>"
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
        margin=dict(r=300),  # room for the control panel
    )

    # ── Point cloud range from config ───────────────────────────────────────
    pc_range = list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)

    # ── Export HTML with custom controls injected ───────────────────────────
    # First get the base plotly HTML
    base_html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    # Subsample point data for JS (max 200k to keep HTML manageable)
    step = max(1, n_pts // 200000)
    point_data_for_js = {
        "z": points[::step, 2].tolist(),
        "x": points[::step, 0].tolist(),
        "y": points[::step, 1].tolist(),
        "distance": pt_dist[::step].tolist(),
        "intensity": pt_intensity[::step].tolist(),
    }

    # Build the control panel and JS
    controls_html = build_controls_html(
        sample_idx, n_pts,
        gt_boxes, gt_point_counts, gt_distances,
        pred_boxes, pred_scores, pred_point_counts, pred_distances,
        pc_range, raw_label,
    )
    controls_js = build_controls_js(point_data_for_js, pc_range)

    # Inject controls right before </body>
    final_html = base_html.replace(
        "</body>",
        f"{controls_html}\n{controls_js}\n</body>"
    )

    out_file = out_dir / f"validate_{sample_idx}.html"
    out_file.write_text(final_html)
    print(f"    -> {out_file}")
    return out_file


def main():
    ap = argparse.ArgumentParser(description="Validate LiDAR labels via OpenPCDet + Plotly")
    ap.add_argument("--cfg", default="cfgs/kitti_models/pointpillar.yaml",
                     help="Model config (relative to tools/)")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"],
                     help="Dataset split to use (train or val)")
    ap.add_argument("--samples", type=int, default=5,
                     help="Number of random samples to visualize")
    ap.add_argument("--indices", type=str, default=None,
                     help="Comma-separated dataset indices to visualize (overrides --samples)")
    ap.add_argument("--out", type=str, default="/p/cavalier/jay/logs",
                     help="Output directory for HTML files")
    ap.add_argument("--ckpt", type=str, default=None,
                     help="Optional: path to model checkpoint for prediction overlay")
    ap.add_argument("--score_thresh", type=float, default=0.25,
                     help="Score threshold for predicted boxes (only with --ckpt)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset (split={args.split}, augmentation disabled)...")
    dataset = load_dataset_no_aug(args.cfg, split=args.split)
    n = len(dataset)
    print(f"Dataset has {n} samples")

    # Optionally load model for prediction overlay
    model = None
    if args.ckpt:
        import torch
        print(f"Loading model from {args.ckpt}...")
        model = build_network(model_cfg=cfg.MODEL,
                              num_class=len(cfg.CLASS_NAMES),
                              dataset=dataset)
        model.load_params_from_file(filename=args.ckpt,
                                    logger=common_utils.create_logger("viz.log"),
                                    to_cpu=False)
        model.cuda()
        model.eval()
        print("Model loaded.")

    # Select indices
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    else:
        rng = np.random.RandomState(args.seed)
        indices = sorted(rng.choice(n, min(args.samples, n), replace=False))

    print(f"Visualizing {len(indices)} samples: {indices}")

    html_files = []
    for idx in indices:
        html_files.append(visualize_sample(dataset, idx, out_dir,
                                           model=model,
                                           score_thresh=args.score_thresh))

    # Create an index page linking to all samples
    index_html = out_dir / "validate_index.html"
    links = "\n".join(
        f'<li><a href="{f.name}" target="_blank">{f.stem}</a></li>'
        for f in html_files
    )
    index_html.write_text(f"""<!DOCTYPE html>
<html><head><title>Label Validation Index</title></head>
<body style="font-family: 'Segoe UI', system-ui, sans-serif; padding: 20px; background: #1a1a2e; color: #eee;">
<h1>🔍 LiDAR Label Validation</h1>
<p>Generated via OpenPCDet KittiDataset (no augmentation).<br>
🟥 Red wireframes = ground truth boxes. 🟡 Yellow arrows = heading direction.<br>
🟩 Green wireframes = model predictions (if checkpoint provided).<br>
Coordinate axes at origin: <span style="color:red">+X forward</span>,
<span style="color:green">+Y left</span>,
<span style="color:blue">+Z up</span>.</p>
<ul style="line-height: 2;">{links}</ul>
<p style="color: #666; font-size: 12px; margin-top: 30px;">
  Each visualization includes an interactive control panel for:
  point size / opacity / color-by mode, box line width, layer toggles,
  per-box statistics (distance, point count, dimensions), and range rings.
</p>
</body></html>""")
    print(f"\nIndex page: {index_html}")
    print("Open in browser to inspect.")


if __name__ == "__main__":
    main()
