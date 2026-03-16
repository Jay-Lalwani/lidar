#!/usr/bin/env python3
"""
3D-Only Evaluation for LiDAR-based Object Detection (bypasses KITTI 2D/camera pipeline).

Computes AP directly in LiDAR coordinates for custom data that lacks real
camera calibration and images. Uses the same AP algorithm as KITTI (greedy
matching, score-sorted detections) with R40 interpolation.

Metrics reported:
  - BEV AP  @ IoU 0.3, 0.5, 0.7
  - 3D  AP  @ IoU 0.3, 0.5, 0.7
  - Recall   @ IoU 0.3, 0.5, 0.7
  - Per-distance breakdown (0-30m, 30-60m, 60-100m, 100m+)
  - Score distribution, detection counts

Must be run from OpenPCDet/tools/ so configs resolve correctly.

Usage:
    cd /p/cavalier/jay/OpenPCDet/tools
    python /p/cavalier/jay/scripts/evaluate_3d.py \
        --cfg_file cfgs/kitti_models/pointpillar.yaml \
        --ckpt /p/cavalier/jay/OpenPCDet/output/kitti_models/pointpillar/5458562/ckpt/checkpoint_epoch_80.pth \
        --score_thresh 0.1
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numba
import numpy as np
import torch
import tqdm

# ── OpenPCDet imports ────────────────────────────────────────────────────────
# Script must be run from OpenPCDet/tools/ directory (cwd)
sys.path.insert(0, str(Path.cwd()))
import _init_path  # noqa: F401, E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

# ── Reuse OpenPCDet's GPU-accelerated rotated IoU ────────────────────────────
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import (
    rotate_iou_gpu_eval,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  3D IoU computation in LiDAR frame
# ═══════════════════════════════════════════════════════════════════════════════

def boxes_bev_iou_lidar(boxes_a, boxes_b):
    """
    Compute BEV IoU between two sets of LiDAR boxes.

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        iou: (N, M) BEV IoU matrix
    """
    # rotate_iou_gpu_eval expects [cx, cy, dx, dy, angle] in 5 columns
    bev_a = boxes_a[:, [0, 1, 3, 4, 6]].astype(np.float32)
    bev_b = boxes_b[:, [0, 1, 3, 4, 6]].astype(np.float32)
    return rotate_iou_gpu_eval(bev_a, bev_b, criterion=-1)


@numba.jit(nopython=True)
def _height_overlap(boxes_a, boxes_b, bev_iou):
    """
    Compute 3D IoU given BEV IoU, adding height overlap.

    boxes format: [x, y, z, dx, dy, dz, heading]
    z is center height, dz is full box height.
    """
    N, M = bev_iou.shape
    result = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        z_a_min = boxes_a[i, 2] - boxes_a[i, 5] / 2.0
        z_a_max = boxes_a[i, 2] + boxes_a[i, 5] / 2.0
        vol_a = boxes_a[i, 3] * boxes_a[i, 4] * boxes_a[i, 5]
        for j in range(M):
            if bev_iou[i, j] <= 0:
                continue
            z_b_min = boxes_b[j, 2] - boxes_b[j, 5] / 2.0
            z_b_max = boxes_b[j, 2] + boxes_b[j, 5] / 2.0
            z_overlap = max(0.0, min(z_a_max, z_b_max) - max(z_a_min, z_b_min))
            if z_overlap <= 0:
                result[i, j] = 0.0
                continue
            # BEV IoU gives intersection_bev / union_bev
            # We need: intersection_3d / union_3d
            # intersection_3d = intersection_bev * z_overlap
            # union_3d = vol_a + vol_b - intersection_3d
            vol_b = boxes_b[j, 3] * boxes_b[j, 4] * boxes_b[j, 5]
            # Recover BEV intersection area from BEV IoU:
            # bev_iou = inter_bev / (area_a + area_b - inter_bev)
            area_a = boxes_a[i, 3] * boxes_a[i, 4]
            area_b = boxes_b[j, 3] * boxes_b[j, 4]
            inter_bev = bev_iou[i, j] * (area_a + area_b) / (1.0 + bev_iou[i, j])
            inter_3d = inter_bev * z_overlap
            union_3d = vol_a + vol_b - inter_3d
            if union_3d > 0:
                result[i, j] = inter_3d / union_3d
    return result


def boxes_3d_iou_lidar(boxes_a, boxes_b):
    """
    Compute 3D IoU between two sets of LiDAR boxes.

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        iou: (N, M) 3D IoU matrix
    """
    bev_iou = boxes_bev_iou_lidar(boxes_a, boxes_b)
    return _height_overlap(
        boxes_a.astype(np.float64),
        boxes_b.astype(np.float64),
        bev_iou.astype(np.float64),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  AP computation (R40-style, matching KITTI protocol but without 2D filtering)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ap_r40(precision, recall):
    """
    Compute AP using R40 (40-point interpolation), same as KITTI AP_R40.
    """
    recall_thresholds = np.linspace(0.0, 1.0, 41)  # 41 points including 0 and 1
    precisions_at_recall = np.zeros(41)
    for i, r_thresh in enumerate(recall_thresholds):
        # Find max precision at recall >= r_thresh
        mask = recall >= r_thresh
        if mask.any():
            precisions_at_recall[i] = precision[mask].max()
        else:
            precisions_at_recall[i] = 0.0
    return precisions_at_recall.mean()


def compute_ap_11(precision, recall):
    """Compute AP using 11-point interpolation (legacy KITTI)."""
    recall_thresholds = np.linspace(0.0, 1.0, 11)
    precisions_at_recall = np.zeros(11)
    for i, r_thresh in enumerate(recall_thresholds):
        mask = recall >= r_thresh
        if mask.any():
            precisions_at_recall[i] = precision[mask].max()
    return precisions_at_recall.mean()


def evaluate_detections(all_gt_boxes, all_dt_boxes, all_dt_scores,
                        iou_thresholds=(0.3, 0.5, 0.7), metric='3d'):
    """
    Compute AP for a single class across all samples.

    Args:
        all_gt_boxes: list of (N_i, 7) arrays   — GT boxes per sample
        all_dt_boxes: list of (M_i, 7) arrays    — detection boxes per sample
        all_dt_scores: list of (M_i,) arrays     — detection scores per sample
        iou_thresholds: tuple of IoU thresholds  — compute AP at each
        metric: '3d' or 'bev'

    Returns:
        results: dict with AP, precision, recall for each IoU threshold
    """
    iou_fn = boxes_3d_iou_lidar if metric == '3d' else boxes_bev_iou_lidar

    # Flatten all detections with sample index
    dt_entries = []  # (score, sample_idx, det_idx_in_sample)
    for s_idx in range(len(all_dt_boxes)):
        for d_idx in range(len(all_dt_scores[s_idx])):
            dt_entries.append((all_dt_scores[s_idx][d_idx], s_idx, d_idx))
    # Sort by descending score
    dt_entries.sort(key=lambda x: -x[0])

    total_gt = sum(len(g) for g in all_gt_boxes)

    # Precompute IoU matrices
    iou_matrices = []
    for s_idx in range(len(all_gt_boxes)):
        gt = all_gt_boxes[s_idx]
        dt = all_dt_boxes[s_idx]
        if len(gt) == 0 or len(dt) == 0:
            iou_matrices.append(np.zeros((len(gt), len(dt)), dtype=np.float64))
        else:
            iou_matrices.append(iou_fn(gt, dt))

    results = {}
    for iou_thresh in iou_thresholds:
        # Track which GT boxes have been matched
        gt_matched = [np.zeros(len(g), dtype=bool) for g in all_gt_boxes]
        tp = np.zeros(len(dt_entries))
        fp = np.zeros(len(dt_entries))

        for i, (score, s_idx, d_idx) in enumerate(dt_entries):
            iou_mat = iou_matrices[s_idx]
            if iou_mat.shape[0] == 0:
                fp[i] = 1
                continue
            # Find best matching GT
            ious = iou_mat[:, d_idx]
            best_gt = ious.argmax()
            best_iou = ious[best_gt]

            if best_iou >= iou_thresh and not gt_matched[s_idx][best_gt]:
                tp[i] = 1
                gt_matched[s_idx][best_gt] = True
            else:
                fp[i] = 1

        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / max(total_gt, 1)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap_r40 = compute_ap_r40(precision, recall)
        ap_11 = compute_ap_11(precision, recall)
        final_recall = recall[-1] if len(recall) > 0 else 0.0

        results[iou_thresh] = {
            'ap_r40': ap_r40,
            'ap_11': ap_11,
            'recall': final_recall,
            'precision_curve': precision,
            'recall_curve': recall,
            'total_tp': int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0,
            'total_fp': int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0,
            'total_gt': total_gt,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-distance breakdown
# ═══════════════════════════════════════════════════════════════════════════════

def distance_breakdown(all_gt_boxes, all_dt_boxes, all_dt_scores,
                       ranges=((0, 30), (30, 60), (60, 100), (100, 999)),
                       iou_threshold=0.7, metric='3d'):
    """Compute AP for each distance range separately."""
    results = {}
    for d_min, d_max in ranges:
        gt_filtered = []
        dt_filtered = []
        sc_filtered = []
        for gt, dt, sc in zip(all_gt_boxes, all_dt_boxes, all_dt_scores):
            # Filter GT by distance
            if len(gt) > 0:
                gt_dist = np.sqrt(gt[:, 0]**2 + gt[:, 1]**2)
                gt_mask = (gt_dist >= d_min) & (gt_dist < d_max)
                gt_filtered.append(gt[gt_mask])
            else:
                gt_filtered.append(gt)
            # Filter DT by distance
            if len(dt) > 0:
                dt_dist = np.sqrt(dt[:, 0]**2 + dt[:, 1]**2)
                dt_mask = (dt_dist >= d_min) & (dt_dist < d_max)
                dt_filtered.append(dt[dt_mask])
                sc_filtered.append(sc[dt_mask])
            else:
                dt_filtered.append(dt)
                sc_filtered.append(sc)

        total_gt = sum(len(g) for g in gt_filtered)
        if total_gt == 0:
            results[f'{d_min}-{d_max}m'] = {'ap_r40': 0.0, 'recall': 0.0, 'n_gt': 0, 'n_dt': sum(len(d) for d in dt_filtered)}
            continue

        res = evaluate_detections(gt_filtered, dt_filtered, sc_filtered,
                                  iou_thresholds=(iou_threshold,), metric=metric)
        r = res[iou_threshold]
        results[f'{d_min}-{d_max}m'] = {
            'ap_r40': r['ap_r40'],
            'recall': r['recall'],
            'n_gt': total_gt,
            'n_dt': sum(len(d) for d in dt_filtered),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='3D-only model evaluation')
    parser.add_argument('--cfg_file', type=str,
                        default='cfgs/kitti_models/pointpillar.yaml',
                        help='Config file for model')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--score_thresh', type=float, default=0.1,
                        help='Score threshold for detections')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save detailed results (.npz)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger("evaluate_3d.log", rank=0)

    # ── Print config summary ────────────────────────────────────────────────
    print("=" * 70)
    print("  3D-Only LiDAR Evaluation")
    print("=" * 70)
    print(f"  Config:      {args.cfg_file}")
    print(f"  Checkpoint:  {args.ckpt}")
    print(f"  Score thresh: {args.score_thresh}")
    print(f"  Point Cloud Range: {cfg.DATA_CONFIG.POINT_CLOUD_RANGE}")
    print(f"  Class Names: {cfg.CLASS_NAMES}")
    print("=" * 70)

    # ── Build dataloader (val set) ──────────────────────────────────────────
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False
    )
    print(f"\nVal set: {len(test_set)} samples")

    # ── Build model & load checkpoint ───────────────────────────────────────
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                          dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # ── Run inference ───────────────────────────────────────────────────────
    all_gt_boxes = []
    all_dt_boxes = []
    all_dt_scores = []
    all_dt_labels = []
    score_dist = []
    n_empty_gt = 0
    n_empty_dt = 0

    print("\n>>> Running inference on val set...")
    start_time = time.time()

    with torch.no_grad():
        for batch_dict in tqdm.tqdm(test_loader, desc="eval", dynamic_ncols=True):
            load_data_to_gpu(batch_dict)
            pred_dicts, ret_dict = model(batch_dict)

            batch_size = batch_dict['batch_size']
            for b_idx in range(batch_size):
                # GT boxes from batch_dict (in LiDAR frame)
                gt_mask = batch_dict['gt_boxes'][b_idx].cpu().numpy()
                # Remove padding rows (all zeros)
                valid_mask = gt_mask[:, 3] > 0  # dx > 0 means valid box
                gt = gt_mask[valid_mask, :7]  # [x, y, z, dx, dy, dz, heading]
                all_gt_boxes.append(gt)
                if len(gt) == 0:
                    n_empty_gt += 1

                # Predictions
                pred = pred_dicts[b_idx]
                pred_boxes = pred['pred_boxes'].cpu().numpy()
                pred_scores = pred['pred_scores'].cpu().numpy()
                pred_labels = pred['pred_labels'].cpu().numpy()

                # Filter by score threshold
                mask = pred_scores >= args.score_thresh
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]

                all_dt_boxes.append(pred_boxes)
                all_dt_scores.append(pred_scores)
                all_dt_labels.append(pred_labels)
                score_dist.extend(pred_scores.tolist())
                if len(pred_boxes) == 0:
                    n_empty_dt += 1

    elapsed = time.time() - start_time
    n_samples = len(all_gt_boxes)
    total_gt = sum(len(g) for g in all_gt_boxes)
    total_dt = sum(len(d) for d in all_dt_boxes)

    print(f"\n>>> Inference complete: {elapsed:.1f}s "
          f"({elapsed/n_samples:.3f}s/sample)")
    print(f"    Samples: {n_samples}")
    print(f"    Total GT boxes: {total_gt}")
    print(f"    Total detections (score >= {args.score_thresh}): {total_dt}")
    print(f"    Avg GT/sample: {total_gt/max(n_samples,1):.1f}")
    print(f"    Avg DT/sample: {total_dt/max(n_samples,1):.1f}")
    print(f"    Samples with no GT: {n_empty_gt}")
    print(f"    Samples with no DT: {n_empty_dt}")

    # ── Score distribution ──────────────────────────────────────────────────
    if score_dist:
        scores = np.array(score_dist)
        print(f"\n{'=' * 70}")
        print("  Score Distribution")
        print(f"{'=' * 70}")
        print(f"    Min:    {scores.min():.4f}")
        print(f"    Max:    {scores.max():.4f}")
        print(f"    Mean:   {scores.mean():.4f}")
        print(f"    Median: {np.median(scores):.4f}")
        for thresh in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            n = (scores >= thresh).sum()
            print(f"    >= {thresh:.1f}:  {n:6d} ({100*n/len(scores):.1f}%)")

    # ── Compute 3D AP ───────────────────────────────────────────────────────
    iou_thresholds = (0.3, 0.5, 0.7)

    print(f"\n{'=' * 70}")
    print("  3D AP Results (LiDAR frame)")
    print(f"{'=' * 70}")
    results_3d = evaluate_detections(all_gt_boxes, all_dt_boxes, all_dt_scores,
                                     iou_thresholds=iou_thresholds, metric='3d')
    for iou_t in iou_thresholds:
        r = results_3d[iou_t]
        print(f"\n  IoU threshold: {iou_t:.1f}")
        print(f"    3D AP (R40):   {r['ap_r40']*100:6.2f}%")
        print(f"    3D AP (11pt):  {r['ap_11']*100:6.2f}%")
        print(f"    Recall:        {r['recall']*100:6.2f}%  "
              f"({r['total_tp']}/{r['total_gt']} matched)")
        if r['total_tp'] + r['total_fp'] > 0:
            final_prec = r['total_tp'] / (r['total_tp'] + r['total_fp'])
            print(f"    Precision:     {final_prec*100:6.2f}%  "
                  f"({r['total_tp']}/{r['total_tp']+r['total_fp']})")

    # ── Compute BEV AP ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  BEV AP Results (LiDAR frame)")
    print(f"{'=' * 70}")
    results_bev = evaluate_detections(all_gt_boxes, all_dt_boxes, all_dt_scores,
                                      iou_thresholds=iou_thresholds, metric='bev')
    for iou_t in iou_thresholds:
        r = results_bev[iou_t]
        print(f"\n  IoU threshold: {iou_t:.1f}")
        print(f"    BEV AP (R40):  {r['ap_r40']*100:6.2f}%")
        print(f"    BEV AP (11pt): {r['ap_11']*100:6.2f}%")
        print(f"    Recall:        {r['recall']*100:6.2f}%  "
              f"({r['total_tp']}/{r['total_gt']} matched)")

    # ── Per-distance breakdown ──────────────────────────────────────────────
    for iou_t in [0.5, 0.7]:
        print(f"\n{'=' * 70}")
        print(f"  Per-Distance 3D AP @ IoU {iou_t}")
        print(f"{'=' * 70}")
        dist_res = distance_breakdown(all_gt_boxes, all_dt_boxes, all_dt_scores,
                                      iou_threshold=iou_t, metric='3d')
        print(f"  {'Range':<12} {'AP (R40)':>10} {'Recall':>10} {'# GT':>8} {'# DT':>8}")
        print(f"  {'─'*50}")
        for rng, vals in dist_res.items():
            print(f"  {rng:<12} {vals['ap_r40']*100:9.2f}% "
                  f"{vals['recall']*100:9.2f}%  {vals['n_gt']:7d}  {vals['n_dt']:7d}")

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<20} {'IoU 0.3':>10} {'IoU 0.5':>10} {'IoU 0.7':>10}")
    print(f"  {'─'*52}")
    print(f"  {'BEV AP (R40)':<20}", end="")
    for t in iou_thresholds:
        print(f" {results_bev[t]['ap_r40']*100:9.2f}%", end="")
    print()
    print(f"  {'3D AP (R40)':<20}", end="")
    for t in iou_thresholds:
        print(f" {results_3d[t]['ap_r40']*100:9.2f}%", end="")
    print()
    print(f"  {'Recall':<20}", end="")
    for t in iou_thresholds:
        print(f" {results_3d[t]['recall']*100:9.2f}%", end="")
    print()
    print(f"{'=' * 70}")

    # ── Save results ────────────────────────────────────────────────────────
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'n_samples': n_samples,
            'total_gt': total_gt,
            'total_dt': total_dt,
            'score_thresh': args.score_thresh,
        }
        for iou_t in iou_thresholds:
            r3 = results_3d[iou_t]
            rb = results_bev[iou_t]
            t_str = str(iou_t).replace('.', '')
            save_data[f'3d_ap_r40_{t_str}'] = r3['ap_r40']
            save_data[f'3d_recall_{t_str}'] = r3['recall']
            save_data[f'bev_ap_r40_{t_str}'] = rb['ap_r40']
            save_data[f'bev_recall_{t_str}'] = rb['recall']
            save_data[f'3d_precision_curve_{t_str}'] = r3['precision_curve']
            save_data[f'3d_recall_curve_{t_str}'] = r3['recall_curve']
        np.savez(str(save_path), **save_data)
        print(f"\nDetailed results saved to: {save_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
