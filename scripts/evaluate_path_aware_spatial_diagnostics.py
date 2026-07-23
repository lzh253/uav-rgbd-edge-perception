from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = FINALLY_ROOT / "configs" / "path_aware_spatial_diagnostics_v1.json"
PROTOCOL_CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"
BASELINE_CONFIG_PATH = FINALLY_ROOT / "configs" / "baseline_training_long.json"
SEMANTIC_CONFIG_PATH = FINALLY_ROOT / "configs" / "semantic_comm_training.json"

sys.path.insert(0, str(SCRIPT_DIR))
from train_baselines import ensure_dir, load_json, read_ids
from evaluate_path_planning import (
    ModelPredictor,
    astar,
    binary_iou,
    evaluate_prediction,
    load_rgb_for_preview,
    make_problem,
    path_length,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out
    except (TypeError, ValueError):
        return default


def finite(value: Any, default: float = 0.0) -> float:
    out = to_float(value)
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def mean(values: list[Any]) -> float:
    clean = [to_float(x) for x in values]
    clean = [x for x in clean if not math.isnan(x)]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def std(values: list[Any]) -> float:
    clean = [to_float(x) for x in values]
    clean = [x for x in clean if not math.isnan(x)]
    if len(clean) <= 1:
        return 0.0
    m = sum(clean) / len(clean)
    return float(math.sqrt(sum((x - m) ** 2 for x in clean) / (len(clean) - 1)))


def pearson(xs: list[Any], ys: list[Any]) -> float:
    pairs = [(to_float(x), to_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return math.nan
    x_arr = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y_arr = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(x_arr.std()) == 0.0 or float(y_arr.std()) == 0.0:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def fmt(value: Any, digits: int = 4) -> str:
    out = to_float(value)
    if math.isnan(out):
        return "nan"
    return f"{out:.{digits}f}"


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def model_meta(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": model_cfg.get("family", model_cfg.get("name", "")),
        "display_name": model_cfg.get("display_name", model_cfg.get("name", "")),
        "checkpoint_selection": model_cfg.get("checkpoint_selection", ""),
        "pair_id": model_cfg.get("pair_id", ""),
        "seed": model_cfg.get("seed", ""),
        "role": model_cfg.get("role", ""),
        "model_type": model_cfg.get("type", ""),
    }


def connected_component_metrics(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, dict[str, float]]:
    if int(mask.sum()) == 0:
        return np.zeros(mask.shape, dtype=np.int32), {
            "component_count": 0.0,
            "largest_component_pixels": 0.0,
            "mean_component_pixels": 0.0,
        }
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    if num_labels <= 1:
        return labels, {
            "component_count": 0.0,
            "largest_component_pixels": 0.0,
            "mean_component_pixels": 0.0,
        }
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    return labels, {
        "component_count": float(num_labels - 1),
        "largest_component_pixels": float(areas.max()),
        "mean_component_pixels": float(areas.mean()),
    }


def make_path_mask(path: list[tuple[int, int]] | None, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if path is None:
        return mask
    for y, x in path:
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            mask[y, x] = True
    return mask


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def compute_spatial_metrics(
    problem: Any,
    prob: np.ndarray,
    threshold: float,
    result: dict[str, Any],
    connectivity: int,
    corridor_radius: int,
) -> dict[str, Any]:
    pred = result["pred_mask"].astype(bool)
    gt = problem.gt_mask.astype(bool)
    total_pixels = int(gt.size)
    path = result["path"]
    fp_mask = np.logical_and(pred, ~gt)
    fn_mask = np.logical_and(~pred, gt)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(fp_mask.sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fn = int(fn_mask.sum())
    pred_pos = int(pred.sum())
    gt_pos = int(gt.sum())

    fp_labels, fp_components = connected_component_metrics(fp_mask, connectivity)
    pred_labels, pred_components = connected_component_metrics(pred, connectivity)
    start_label = int(pred_labels[problem.start]) if pred[problem.start] else 0
    goal_label = int(pred_labels[problem.goal]) if pred[problem.goal] else 0
    start_component_pixels = 0
    start_component_fp_pixels = 0
    if start_label > 0:
        start_component = pred_labels == start_label
        start_component_pixels = int(start_component.sum())
        start_component_fp_pixels = int(np.logical_and(start_component, fp_mask).sum())

    path_mask = make_path_mask(path, gt.shape)
    path_cells = int(path_mask.sum())
    path_fp_mask = np.logical_and(path_mask, fp_mask)
    path_gt_mask = np.logical_and(path_mask, gt)
    collision_cells = int(path_fp_mask.sum())
    corridor = dilate(path_mask, corridor_radius) if path is not None else np.zeros(gt.shape, dtype=bool)
    corridor_cells = int(corridor.sum())
    corridor_fp_cells = int(np.logical_and(corridor, fp_mask).sum())
    corridor_gt_negative_cells = int(np.logical_and(corridor, ~gt).sum())
    corridor_fn_cells = int(np.logical_and(corridor, fn_mask).sum())

    bridge_labels = np.unique(fp_labels[path_fp_mask]) if collision_cells > 0 else np.asarray([], dtype=np.int32)
    bridge_labels = bridge_labels[bridge_labels > 0]
    bridge_component_pixels = 0
    largest_bridge_component_pixels = 0
    if len(bridge_labels) > 0:
        bridge_areas = [int((fp_labels == int(label)).sum()) for label in bridge_labels]
        bridge_component_pixels = int(sum(bridge_areas))
        largest_bridge_component_pixels = int(max(bridge_areas))

    path_probs = prob[path_mask] if path_cells > 0 else np.asarray([], dtype=np.float32)
    collision_probs = prob[path_fp_mask] if collision_cells > 0 else np.asarray([], dtype=np.float32)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    pred_start_component_fp_fraction = start_component_fp_pixels / max(1, start_component_pixels)

    return {
        "threshold": threshold,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "tn_pixels": tn,
        "fn_pixels": fn,
        "pred_positive_pixels": pred_pos,
        "gt_positive_pixels": gt_pos,
        "pred_positive_rate": pred_pos / max(1, total_pixels),
        "gt_positive_rate": gt_pos / max(1, total_pixels),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "map_iou": binary_iou(pred, gt),
        "fp_component_count": fp_components["component_count"],
        "largest_fp_component_pixels": fp_components["largest_component_pixels"],
        "largest_fp_component_ratio_image": fp_components["largest_component_pixels"] / max(1, total_pixels),
        "largest_fp_component_ratio_fp": fp_components["largest_component_pixels"] / max(1, fp),
        "mean_fp_component_pixels": fp_components["mean_component_pixels"],
        "pred_component_count": pred_components["component_count"],
        "largest_pred_component_pixels": pred_components["largest_component_pixels"],
        "start_goal_same_pred_component": bool(start_label > 0 and start_label == goal_label),
        "pred_start_component_pixels": start_component_pixels,
        "pred_start_component_fp_pixels": start_component_fp_pixels,
        "pred_start_component_fp_fraction": pred_start_component_fp_fraction,
        "path_cells": path_cells,
        "path_gt_cells": int(path_gt_mask.sum()),
        "path_false_positive_cells": collision_cells,
        "path_false_positive_fraction": collision_cells / max(1, path_cells) if path is not None else math.nan,
        "path_mean_probability": float(path_probs.mean()) if path_probs.size else math.nan,
        "path_collision_mean_probability": float(collision_probs.mean()) if collision_probs.size else math.nan,
        "path_corridor_cells": corridor_cells,
        "path_corridor_fp_cells": corridor_fp_cells,
        "path_corridor_fp_rate": corridor_fp_cells / max(1, corridor_cells) if path is not None else math.nan,
        "path_corridor_fp_negative_rate": corridor_fp_cells / max(1, corridor_gt_negative_cells) if path is not None else math.nan,
        "path_corridor_fn_cells": corridor_fn_cells,
        "collision_bridge_component_count": float(len(bridge_labels)),
        "collision_bridge_component_pixels": bridge_component_pixels,
        "largest_collision_bridge_component_pixels": largest_bridge_component_pixels,
        "collision_bridge_component_area_ratio_image": bridge_component_pixels / max(1, total_pixels),
    }


def path_aware_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    safety = finite(row.get("planning_safety_utility"))
    collision = finite(row.get("collision_rate_among_found"))
    corridor = finite(row.get("mean_path_corridor_fp_rate_found"))
    bridge = finite(row.get("mean_collision_bridge_component_area_ratio_image_found"))
    blocked = finite(row.get("start_or_goal_blocked_rate"))
    return (
        safety
        - float(weights["collision_rate_penalty_weight"]) * collision
        - float(weights["corridor_fp_rate_penalty_weight"]) * corridor
        - float(weights["collision_bridge_area_penalty_weight"]) * bridge
        - float(weights["blocked_rate_penalty_weight"]) * blocked
    )


def summarize_threshold_rows(
    split: str,
    model_cfg: dict[str, Any],
    threshold: float,
    rows: list[dict[str, Any]],
    selected_samples: int,
    total_samples: int,
    weights: dict[str, float],
) -> dict[str, Any]:
    found_rows = [row for row in rows if bool_value(row["path_found"])]
    success_rows = [row for row in rows if bool_value(row["success"])]
    blocked_rows = [row for row in rows if not bool_value(row["start_on_pred"]) or not bool_value(row["goal_on_pred"])]
    collision_found = [row for row in found_rows if int(row["collision_cells"]) > 0]
    denom = max(1, len(rows))
    path_found_rate = len(found_rows) / denom
    success_rate = len(success_rows) / denom
    unsafe_found_rate = path_found_rate - success_rate
    meta = model_meta(model_cfg)
    summary: dict[str, Any] = {
        "split": split,
        "model": model_cfg["name"],
        **meta,
        "threshold": threshold,
        "total_split_samples": total_samples,
        "selected_evaluable_samples": selected_samples,
        "evaluated_samples": len(rows),
        "path_found_rate": path_found_rate,
        "success_rate_collision_free": success_rate,
        "unsafe_found_rate": unsafe_found_rate,
        "planning_safety_utility": success_rate - unsafe_found_rate,
        "start_or_goal_blocked_rate": len(blocked_rows) / denom,
        "collision_rate_among_found": len(collision_found) / max(1, len(found_rows)),
        "mean_collision_cells_found": mean([row["collision_cells"] for row in found_rows]),
        "mean_collision_fraction_found": mean([row["collision_fraction"] for row in found_rows]),
        "mean_path_length_ratio_success": mean([row["path_length_ratio"] for row in success_rows]),
        "mean_map_iou": mean([row["map_iou"] for row in rows]),
        "mean_false_positive_rate": mean([row["false_positive_rate"] for row in rows]),
        "mean_false_negative_rate": mean([row["false_negative_rate"] for row in rows]),
        "mean_pred_positive_rate": mean([row["pred_positive_rate"] for row in rows]),
        "mean_largest_fp_component_ratio_image": mean([row["largest_fp_component_ratio_image"] for row in rows]),
        "mean_largest_fp_component_ratio_fp": mean([row["largest_fp_component_ratio_fp"] for row in rows]),
        "mean_pred_start_component_fp_fraction": mean([row["pred_start_component_fp_fraction"] for row in rows]),
        "start_goal_same_pred_component_rate": mean([1.0 if bool_value(row["start_goal_same_pred_component"]) else 0.0 for row in rows]),
        "mean_path_false_positive_fraction_found": mean([row["path_false_positive_fraction"] for row in found_rows]),
        "mean_path_corridor_fp_rate_found": mean([row["path_corridor_fp_rate"] for row in found_rows]),
        "mean_path_corridor_fp_negative_rate_found": mean([row["path_corridor_fp_negative_rate"] for row in found_rows]),
        "mean_collision_bridge_component_count_found": mean([row["collision_bridge_component_count"] for row in found_rows]),
        "mean_collision_bridge_component_area_ratio_image_found": mean([row["collision_bridge_component_area_ratio_image"] for row in found_rows]),
        "mean_path_collision_probability_found": mean([row["path_collision_mean_probability"] for row in found_rows]),
        "mean_inference_time_ms": mean([row["inference_time_ms"] for row in rows]),
        "mean_planning_time_ms": mean([row["planning_time_ms"] for row in rows]),
    }
    summary["path_aware_score"] = path_aware_score(summary, weights)
    return summary


def select_thresholds(
    summary_rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    selection_split: str,
    final_split: str,
    metric: str,
    strategy: str,
) -> list[dict[str, Any]]:
    by_key = {(row["split"], row["model"], float(row["threshold"])): row for row in summary_rows}
    out: list[dict[str, Any]] = []
    for model in models:
        model_name = str(model["name"])
        val_rows = [row for row in summary_rows if row["split"] == selection_split and row["model"] == model_name]
        if not val_rows:
            continue

        def rank(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
            return (
                finite(row.get(metric), -1e9),
                finite(row.get("planning_safety_utility"), -1e9),
                finite(row.get("success_rate_collision_free"), -1e9),
                -finite(row.get("collision_rate_among_found"), 1.0),
                finite(row.get("mean_map_iou"), -1e9),
                -abs(float(row["threshold"]) - 0.5),
            )

        best = max(val_rows, key=rank)
        threshold = float(best["threshold"])
        test = by_key[(final_split, model_name, threshold)]
        out.append(flatten_selection_row(strategy, metric, threshold, best, test))
    return out


def flatten_selection_row(
    strategy: str,
    selection_metric: str,
    threshold: float,
    val: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "path_found_rate",
        "success_rate_collision_free",
        "unsafe_found_rate",
        "planning_safety_utility",
        "path_aware_score",
        "collision_rate_among_found",
        "start_or_goal_blocked_rate",
        "mean_map_iou",
        "mean_false_positive_rate",
        "mean_largest_fp_component_ratio_image",
        "mean_pred_start_component_fp_fraction",
        "mean_path_false_positive_fraction_found",
        "mean_path_corridor_fp_rate_found",
        "mean_collision_bridge_component_area_ratio_image_found",
    ]
    row = {
        "strategy": strategy,
        "selection_metric": selection_metric,
        "model": test["model"],
        "family": test.get("family", ""),
        "display_name": test.get("display_name", ""),
        "checkpoint_selection": test.get("checkpoint_selection", ""),
        "pair_id": test.get("pair_id", ""),
        "seed": test.get("seed", ""),
        "selected_threshold": threshold,
    }
    for key in keys:
        row[f"val_{key}"] = val.get(key, math.nan)
        row[f"test_{key}"] = test.get(key, math.nan)
    return row


def default_threshold_rows(
    summary_rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    selection_split: str,
    final_split: str,
    default_threshold: float,
) -> list[dict[str, Any]]:
    by_key = {(row["split"], row["model"], float(row["threshold"])): row for row in summary_rows}
    rows: list[dict[str, Any]] = []
    for model in models:
        model_name = str(model["name"])
        key = (selection_split, model_name, default_threshold)
        test_key = (final_split, model_name, default_threshold)
        if key in by_key and test_key in by_key:
            rows.append(flatten_selection_row("fixed_0_50", "fixed_threshold", default_threshold, by_key[key], by_key[test_key]))
    return rows


def pixel_iou_strategy_rows(
    summary_rows: list[dict[str, Any]],
    calibration_transfer_rows: list[dict[str, str]],
    selection_split: str,
    final_split: str,
) -> list[dict[str, Any]]:
    by_key = {(row["split"], row["model"], float(row["threshold"])): row for row in summary_rows}
    rows: list[dict[str, Any]] = []
    for transfer in calibration_transfer_rows:
        if transfer.get("selection_metric") != "iou":
            continue
        model = transfer["model"]
        threshold = to_float(transfer["selected_threshold"])
        key = (selection_split, model, threshold)
        test_key = (final_split, model, threshold)
        if math.isnan(threshold) or key not in by_key or test_key not in by_key:
            continue
        row = flatten_selection_row("pixel_iou_transfer", "pixel_iou", threshold, by_key[key], by_key[test_key])
        row["pixel_transfer_test_iou"] = transfer.get("test_iou", "")
        row["pixel_transfer_test_f1"] = transfer.get("test_f1", "")
        rows.append(row)
    return rows


def aggregate_strategy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["strategy"]), str(row["family"]), str(row.get("checkpoint_selection", "")))].append(row)
    metrics = [
        "test_planning_safety_utility",
        "test_success_rate_collision_free",
        "test_path_found_rate",
        "test_collision_rate_among_found",
        "test_mean_map_iou",
        "test_mean_path_corridor_fp_rate_found",
        "test_mean_collision_bridge_component_area_ratio_image_found",
        "test_path_aware_score",
    ]
    out: list[dict[str, Any]] = []
    for (strategy, family, checkpoint), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "strategy": strategy,
            "family": family,
            "checkpoint_selection": checkpoint,
            "members": len(group),
        }
        for metric in metrics:
            values = [row.get(metric, math.nan) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = std(values)
        out.append(item)
    return out


def spatial_correlations(summary_rows: list[dict[str, Any]], final_split: str) -> list[dict[str, Any]]:
    rows = [row for row in summary_rows if row["split"] == final_split]
    metrics = [
        "mean_map_iou",
        "mean_false_positive_rate",
        "mean_false_negative_rate",
        "mean_largest_fp_component_ratio_image",
        "mean_pred_start_component_fp_fraction",
        "mean_path_false_positive_fraction_found",
        "mean_path_corridor_fp_rate_found",
        "mean_collision_bridge_component_area_ratio_image_found",
        "collision_rate_among_found",
        "start_or_goal_blocked_rate",
        "path_aware_score",
    ]
    return [
        {
            "metric": metric,
            "pearson_with_path_safety": pearson([row[metric] for row in rows], [row["planning_safety_utility"] for row in rows]),
            "n_model_threshold_points": len(rows),
        }
        for metric in metrics
    ]


def plot_strategy_comparison(path: Path, family_rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in family_rows
        if row["strategy"] in {"fixed_0_50", "pixel_iou_transfer", "safety_utility", "path_aware"}
        and row["checkpoint_selection"] == "best"
    ]
    families = sorted({row["family"] for row in selected})
    strategies = ["fixed_0_50", "pixel_iou_transfer", "safety_utility", "path_aware"]
    labels = {
        "fixed_0_50": "0.50",
        "pixel_iou_transfer": "pixel IoU",
        "safety_utility": "safety",
        "path_aware": "path-aware",
    }
    x = np.arange(len(families))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    colors = ["#64748b", "#2563eb", "#0f766e", "#dc2626"]
    for idx, strategy in enumerate(strategies):
        rows = {(row["family"], row["strategy"]): row for row in selected}
        safety = [finite(rows.get((family, strategy), {}).get("test_planning_safety_utility_mean"), math.nan) for family in families]
        risk = [finite(rows.get((family, strategy), {}).get("test_mean_path_corridor_fp_rate_found_mean"), math.nan) for family in families]
        axes[0].bar(x + (idx - 1.5) * width, safety, width=width, label=labels[strategy], color=colors[idx])
        axes[1].bar(x + (idx - 1.5) * width, risk, width=width, label=labels[strategy], color=colors[idx])
    axes[0].set_title("Test path safety utility")
    axes[1].set_title("Path-corridor false positive rate")
    for ax in axes:
        ax.set_xticks(x, families, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_spatial_risk_vs_safety(path: Path, summary_rows: list[dict[str, Any]], final_split: str) -> None:
    rows = [row for row in summary_rows if row["split"] == final_split]
    families = sorted({str(row["family"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(families))))
    color_map = {family: colors[idx] for idx, family in enumerate(families)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    pairs = [
        ("mean_path_corridor_fp_rate_found", "corridor FP rate"),
        ("mean_largest_fp_component_ratio_image", "largest FP component/image"),
        ("collision_rate_among_found", "collision/found"),
    ]
    for ax, (metric, label) in zip(axes, pairs):
        for family in families:
            sub = [row for row in rows if row["family"] == family]
            ax.scatter(
                [finite(row.get(metric), math.nan) for row in sub],
                [finite(row.get("planning_safety_utility"), math.nan) for row in sub],
                s=22,
                alpha=0.72,
                label=family,
                color=color_map[family],
            )
        ax.set_xlabel(label)
        ax.set_ylabel("path safety utility")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_family_spatial_risk(path: Path, family_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in family_rows if row["strategy"] == "path_aware" and row["checkpoint_selection"] == "best"]
    labels = [row["family"] for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(x, [finite(row["test_planning_safety_utility_mean"]) for row in rows], color="#0f766e")
    axes[0].set_title("Path-aware selected safety")
    axes[1].bar(x, [finite(row["test_collision_rate_among_found_mean"]) for row in rows], color="#dc2626")
    axes[1].set_title("Collision/found")
    axes[2].bar(x, [finite(row["test_mean_path_corridor_fp_rate_found_mean"]) for row in rows], color="#f97316")
    axes[2].set_title("Corridor FP rate")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_collision_preview(
    out_path: Path,
    rgb: np.ndarray,
    problem: Any,
    prob: np.ndarray,
    threshold: float,
    connectivity: int,
    title: str,
) -> None:
    result = evaluate_prediction(problem, prob, threshold, connectivity)
    pred = result["pred_mask"].astype(bool)
    gt = problem.gt_mask.astype(bool)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    path = result["path"]
    path_mask = make_path_mask(path, gt.shape)
    collision_path = np.logical_and(path_mask, fp)
    error_rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    error_rgb[gt] = [0.75, 0.75, 0.75]
    error_rgb[fp] = [1.0, 0.45, 0.05]
    error_rgb[fn] = [0.05, 0.35, 1.0]
    error_rgb[collision_path] = [1.0, 0.0, 0.0]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB + path", fontsize=9)
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT road", fontsize=9)
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Pred @ {threshold:.2f}", fontsize=9)
    axes[3].imshow(error_rgb)
    axes[3].set_title("FP/FN/collision", fontsize=9)
    axes[4].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[4].set_title("Probability", fontsize=9)
    if path is not None:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        for ax in (axes[0], axes[2], axes[3], axes[4]):
            ax.plot(xs, ys, color="#ef4444", linewidth=1.8)
    for ax in axes:
        ax.scatter([problem.start[1]], [problem.start[0]], s=30, color="#22c55e")
        ax.scatter([problem.goal[1]], [problem.goal[0]], s=30, color="#f43f5e")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def generate_collision_previews(
    config: dict[str, Any],
    run_dir: Path,
    models: list[dict[str, Any]],
    protocol_config: dict[str, Any],
    baseline_config: dict[str, Any],
    semantic_config: dict[str, Any],
    strategy_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    problems_by_split: dict[str, dict[str, Any]],
    device: torch.device,
    log: Any,
) -> list[dict[str, Any]]:
    final_split = str(config["final_split"])
    preview_models = set(str(x) for x in config.get("preview_models", []))
    max_total = int(config.get("max_collision_previews_total", 12))
    max_per_model = int(config.get("max_collision_previews_per_model", 3))
    connectivity = int(config["connectivity"])
    preview_dir = run_dir / "collision_previews"
    ensure_dir(preview_dir)
    model_cfgs = {str(model["name"]): model for model in models}
    path_aware_thresholds = {
        row["model"]: float(row["selected_threshold"])
        for row in strategy_rows
        if row["strategy"] == "path_aware"
    }
    candidates = [
        row
        for row in sample_rows
        if row["split"] == final_split
        and row["model"] in preview_models
        and bool_value(row["path_found"])
        and int(row["collision_cells"]) > 0
        and abs(float(row["threshold"]) - float(path_aware_thresholds.get(row["model"], -999))) < 1e-9
    ]
    candidates.sort(
        key=lambda row: (
            int(row["collision_cells"]),
            finite(row["path_corridor_fp_rate"]),
            finite(row["collision_bridge_component_area_ratio_image"]),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    per_model_count: dict[str, int] = defaultdict(int)
    for row in candidates:
        model = str(row["model"])
        if per_model_count[model] >= max_per_model:
            continue
        selected.append(row)
        per_model_count[model] += 1
        if len(selected) >= max_total:
            break

    out_rows: list[dict[str, Any]] = []
    sample_ids = sorted({str(row["sample_id"]) for row in selected})
    for model_name in sorted({str(row["model"]) for row in selected}):
        model_selected = [row for row in selected if row["model"] == model_name]
        if not model_selected:
            continue
        predictor = ModelPredictor(
            model_cfgs[model_name],
            sample_ids,
            protocol_config,
            baseline_config,
            semantic_config,
            device,
        )
        for idx, row in enumerate(model_selected, start=1):
            sample_id = str(row["sample_id"])
            problem = problems_by_split[final_split][sample_id]
            threshold = float(row["threshold"])
            seed = int(config["seed"]) + idx
            prob, _elapsed = predictor.predict(problem, seed)
            rgb = load_rgb_for_preview(sample_id, protocol_config, int(config["image_size"]))
            threshold_tag = f"{threshold:.2f}".replace(".", "p")
            filename = f"{model_name}_{sample_id}_thr{threshold_tag}_collision.png"
            out_path = preview_dir / filename
            title = f"{model_name} sample={sample_id} collision={row['collision_cells']}"
            save_collision_preview(out_path, rgb, problem, prob, threshold, connectivity, title)
            out = dict(row)
            out["preview_path"] = str(out_path.relative_to(FINALLY_ROOT).as_posix())
            out_rows.append(out)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log(f"Saved {len(out_rows)} collision previews")
    return out_rows


def markdown_strategy_table(rows: list[dict[str, Any]]) -> str:
    preferred = [
        ("semantic_clean_step4_single", "best"),
        ("semantic_original_recipe", "best"),
        ("semantic_stable_v2_lr2e4_clip10", "best"),
        ("rgb_baseline", "best"),
    ]
    lines = [
        "| strategy | family | ckpt | members | safety | collision/found | corridor FP | path-aware score |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    by_key = {(row["family"], row["checkpoint_selection"], row["strategy"]): row for row in rows}
    for family, ckpt in preferred:
        for strategy in ["fixed_0_50", "pixel_iou_transfer", "safety_utility", "path_aware"]:
            row = by_key.get((family, ckpt, strategy))
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{strategy}`",
                        f"`{family}`",
                        f"`{ckpt}`",
                        str(row["members"]),
                        fmt(row["test_planning_safety_utility_mean"]),
                        fmt(row["test_collision_rate_among_found_mean"]),
                        fmt(row["test_mean_path_corridor_fp_rate_found_mean"]),
                        fmt(row["test_path_aware_score_mean"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def write_report(
    report_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    strategy_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    preview_rows: list[dict[str, Any]],
) -> None:
    ensure_dir(report_path.parent)
    path_aware_rows = [row for row in strategy_rows if row["strategy"] == "path_aware"]
    best_path_aware = max(path_aware_rows, key=lambda row: finite(row["test_planning_safety_utility"], -1e9))
    safety_rows = [row for row in strategy_rows if row["strategy"] == "safety_utility"]
    best_safety = max(safety_rows, key=lambda row: finite(row["test_planning_safety_utility"], -1e9))
    independent_correlation_rows = [row for row in correlation_rows if row["metric"] != "path_aware_score"]
    strongest_corr = sorted(
        independent_correlation_rows,
        key=lambda row: abs(finite(row["pearson_with_path_safety"], 0.0)),
        reverse=True,
    )[:5]
    rel = run_dir.relative_to(FINALLY_ROOT).as_posix()
    lines = [
        "# 第十四步：Path-aware selection 与空间假阳性连通诊断报告",
        "",
        "生成日期：2026-05-18",
        "",
        "## 实验目的",
        "",
        "第十三步已经证明：概率校准好、像素 IoU 高，不一定能带来安全路径。本步进一步检查 probability map 的空间结构，尤其是 A* 路径走廊附近的假阳性、碰撞路径穿过的假阳性连通块，以及预测连通域的纯度。",
        "",
        "本步不训练新模型，复用第十二步的 24 个非 oracle best/last checkpoint，在 validation 上选择阈值，再报告 test 结果。",
        "",
        "## 新增指标",
        "",
        "- `path_corridor_fp_rate`：A* 路径膨胀走廊内有多少比例是假阳性。",
        "- `collision_bridge_component_area_ratio_image`：碰撞路径穿过的假阳性连通块面积占整图比例。",
        "- `pred_start_component_fp_fraction`：预测中与起点连通的区域里，有多少是假阳性。",
        "- `path_aware_score`：在 safety utility 基础上惩罚 collision/found、路径走廊假阳性、碰撞桥连通块和起终点阻塞。",
        "",
        "## 策略对比",
        "",
        markdown_strategy_table(family_rows),
        "",
        "## 核心发现",
        "",
        f"- path-aware 选择后 test safety utility 最高的模型是 `{best_path_aware['model']}`，阈值 {fmt(best_path_aware['selected_threshold'])}，safety utility {fmt(best_path_aware['test_planning_safety_utility'])}。",
        f"- 传统 safety utility 选择后 test safety utility 最高的模型是 `{best_safety['model']}`，阈值 {fmt(best_safety['selected_threshold'])}，safety utility {fmt(best_safety['test_planning_safety_utility'])}。",
        "- 当前 path-aware score 是诊断性原型，没有超过纯 safety utility 选择；它的主要价值是把路径碰撞拆成可解释的空间风险项。",
        "- 与第十三步相比，本步把问题从“概率是否校准”推进到“假阳性是否形成会诱导 A* 的空间通路”。",
        "",
        "`path_aware_score` 本身包含 safety utility，因此不把它当成独立解释变量。与 path safety utility 相关性较强的非派生空间指标：",
        "",
    ]
    for row in strongest_corr:
        lines.append(f"- `{row['metric']}`：Pearson {fmt(row['pearson_with_path_safety'])}。")
    if preview_rows:
        lines.extend(
            [
                "",
                "本步还保存了碰撞案例预览图，用于论文 qualitative analysis。预览图中红色路径表示 A* 预测路径，橙色区域是假阳性，蓝色区域是假阴性，红色亮点/区域表示路径实际穿过的假阳性碰撞位置。",
            ]
        )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `configs/path_aware_spatial_diagnostics_v1.json`：第十四步 path-aware 空间诊断配置。",
            "- `scripts/evaluate_path_aware_spatial_diagnostics.py`：第十四步空间诊断、path-aware 选择、绘图和报告脚本。",
            f"- `{rel}/spatial_sample_rows.csv`：逐 split、模型、样本、阈值的空间错误指标。",
            f"- `{rel}/spatial_threshold_summary.csv`：逐 split、模型、阈值的聚合空间诊断指标。",
            f"- `{rel}/path_aware_thresholds_by_val.csv`：validation path-aware score 选出的阈值。",
            f"- `{rel}/selected_path_aware_test_summary.csv`：path-aware 阈值迁移到 test 后的路径与空间指标。",
            f"- `{rel}/selection_strategy_comparison.csv`：固定 0.5、像素 IoU、safety utility、path-aware 四种选择策略对比。",
            f"- `{rel}/family_strategy_summary.csv`：family/checkpoint 级策略对比 mean/std。",
            f"- `{rel}/spatial_risk_correlations.csv`：空间风险指标与 path safety utility 的相关性。",
            f"- `{rel}/collision_case_index.csv`：碰撞案例预览图索引。",
            f"- `{rel}/path_aware_strategy_comparison.png`：策略对比图。",
            f"- `{rel}/spatial_risk_vs_safety.png`：空间风险指标与 path safety utility 散点图。",
            f"- `{rel}/family_spatial_risk_summary.png`：path-aware 选择后的 family 级空间风险图。",
            f"- `{rel}/collision_previews/*.png`：smoke/full 运行保存的典型碰撞案例预览图；正式索引以 `collision_case_index.csv` 为准。",
            f"- `{rel}/summary.json`：机器可读完整汇总。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run path-aware spatial false-positive diagnostics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    source_model_config = load_json(Path(config["source_model_config"]))
    protocol_config = load_json(PROTOCOL_CONFIG_PATH)
    baseline_config = load_json(BASELINE_CONFIG_PATH)
    semantic_config = load_json(SEMANTIC_CONFIG_PATH)
    run_name = str(config["run_name"])
    run_dir = Path(config["outputs"]["root"]) / run_name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "logs")
    log_path = run_dir / "logs" / f"path_aware_spatial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    started = time.perf_counter()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    thresholds = [float(x) for x in config["thresholds"]]
    splits = [str(x) for x in config["splits"]]
    selection_split = str(config["selection_split"])
    final_split = str(config["final_split"])
    connectivity = int(config["connectivity"])
    corridor_radius = int(config["corridor_radius"])
    seed = int(config["seed"])
    max_samples = int(args.max_samples if args.max_samples >= 0 else config.get("max_samples_per_split", 0))
    weights = {key: float(value) for key, value in config["path_aware_score"].items() if key.endswith("_weight")}

    models = [
        model
        for model in source_model_config["models"]
        if bool(config.get("include_oracle", False)) or str(model.get("type")) != "oracle"
    ]
    log(f"Path-aware spatial diagnostics run={run_name} models={len(models)} splits={splits} thresholds={len(thresholds)} device={device}")

    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "source_model_config_snapshot.json").write_text(json.dumps(source_model_config, indent=2), encoding="utf-8")
    (run_dir / "protocol_config_snapshot.json").write_text(json.dumps(protocol_config, indent=2), encoding="utf-8")

    all_sample_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    sample_selection_rows: list[dict[str, Any]] = []
    total_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    problems_by_split: dict[str, dict[str, Any]] = {}

    for split_index, split in enumerate(splits):
        outputs_root = Path(protocol_config["outputs_root"])
        sample_ids = read_ids(outputs_root / "protocol" / f"{split}.txt")
        if max_samples > 0:
            sample_ids = sample_ids[:max_samples]
        total_counts[split] = len(sample_ids)
        problems = []
        split_selection_rows = []
        for index, sample_id in enumerate(sample_ids, start=1):
            problem, selection = make_problem(sample_id, protocol_config, config)
            selection["split"] = split
            split_selection_rows.append(selection)
            if problem is not None:
                problems.append(problem)
            if index % 100 == 0 or index == len(sample_ids):
                log(f"{split}: selected {len(problems)}/{index}")
        selected_counts[split] = len(problems)
        problems_by_split[split] = {problem.sample_id: problem for problem in problems}
        sample_selection_rows.extend(split_selection_rows)
        write_csv(run_dir / f"sample_selection_{split}.csv", split_selection_rows)

        predictor_sample_ids = [problem.sample_id for problem in problems]
        predictors = [
            ModelPredictor(model, predictor_sample_ids, protocol_config, baseline_config, semantic_config, device)
            for model in models
        ]
        for model_index, predictor in enumerate(predictors):
            model_cfg = models[model_index]
            log(f"{split}: evaluating {predictor.name} ({model_index + 1}/{len(predictors)})")
            model_threshold_rows: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds}
            for problem_index, problem in enumerate(problems, start=1):
                forward_seed = seed + split_index * 1000000 + model_index * 100000 + problem_index
                prob, inference_time_ms = predictor.predict(problem, forward_seed)
                inference_rows.append(
                    {
                        "split": split,
                        "model": predictor.name,
                        "sample_id": problem.sample_id,
                        "inference_time_ms": inference_time_ms,
                    }
                )
                for threshold in thresholds:
                    result = evaluate_prediction(problem, prob, threshold, connectivity)
                    spatial = compute_spatial_metrics(problem, prob, threshold, result, connectivity, corridor_radius)
                    row = {
                        "split": split,
                        "model": predictor.name,
                        **model_meta(model_cfg),
                        "sample_id": problem.sample_id,
                        "component_pixels": problem.component_pixels,
                        "start_y": problem.start[0],
                        "start_x": problem.start[1],
                        "goal_y": problem.goal[0],
                        "goal_x": problem.goal[1],
                        "oracle_length": problem.oracle_length,
                        "threshold": threshold,
                        "start_on_pred": result["start_on_pred"],
                        "goal_on_pred": result["goal_on_pred"],
                        "path_found": result["path_found"],
                        "success": result["success"],
                        "collision_cells": result["collision_cells"],
                        "collision_fraction": result["collision_fraction"],
                        "path_length": result["path_length"],
                        "path_length_ratio": result["path_length_ratio"],
                        "inference_time_ms": inference_time_ms,
                        "planning_time_ms": result["planning_time_ms"],
                        **spatial,
                    }
                    model_threshold_rows[threshold].append(row)
                    all_sample_rows.append(row)
                if problem_index % 50 == 0 or problem_index == len(problems):
                    log(f"{split}: {predictor.name} processed {problem_index}/{len(problems)}")
            for threshold in thresholds:
                all_summary_rows.append(
                    summarize_threshold_rows(
                        split,
                        model_cfg,
                        threshold,
                        model_threshold_rows[threshold],
                        selected_counts[split],
                        total_counts[split],
                        weights,
                    )
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    safety_selected_rows = select_thresholds(
        all_summary_rows,
        models,
        selection_split,
        final_split,
        "planning_safety_utility",
        "safety_utility",
    )
    path_aware_rows = select_thresholds(
        all_summary_rows,
        models,
        selection_split,
        final_split,
        "path_aware_score",
        "path_aware",
    )
    fixed_rows = default_threshold_rows(all_summary_rows, models, selection_split, final_split, 0.5)
    pixel_rows = pixel_iou_strategy_rows(
        all_summary_rows,
        read_csv(Path(config["calibration_threshold_transfer_csv"])),
        selection_split,
        final_split,
    )
    strategy_rows = fixed_rows + pixel_rows + safety_selected_rows + path_aware_rows
    family_rows = aggregate_strategy(strategy_rows)
    correlation_rows = spatial_correlations(all_summary_rows, final_split)

    write_csv(run_dir / "sample_selection_all.csv", sample_selection_rows)
    write_csv(run_dir / "inference_manifest.csv", inference_rows)
    write_csv(run_dir / "spatial_sample_rows.csv", all_sample_rows)
    write_csv(run_dir / "spatial_threshold_summary.csv", all_summary_rows)
    write_csv(run_dir / "safety_thresholds_by_val.csv", safety_selected_rows)
    write_csv(run_dir / "path_aware_thresholds_by_val.csv", path_aware_rows)
    write_csv(run_dir / "selected_path_aware_test_summary.csv", path_aware_rows)
    write_csv(run_dir / "selection_strategy_comparison.csv", strategy_rows)
    write_csv(run_dir / "family_strategy_summary.csv", family_rows)
    write_csv(run_dir / "spatial_risk_correlations.csv", correlation_rows)

    plot_strategy_comparison(run_dir / "path_aware_strategy_comparison.png", family_rows)
    plot_spatial_risk_vs_safety(run_dir / "spatial_risk_vs_safety.png", all_summary_rows, final_split)
    plot_family_spatial_risk(run_dir / "family_spatial_risk_summary.png", family_rows)
    preview_rows = generate_collision_previews(
        config,
        run_dir,
        models,
        protocol_config,
        baseline_config,
        semantic_config,
        strategy_rows,
        all_sample_rows,
        problems_by_split,
        device,
        log,
    )
    write_csv(run_dir / "collision_case_index.csv", preview_rows)

    elapsed_sec = time.perf_counter() - started
    summary = {
        "run_name": run_name,
        "models": len(models),
        "splits": splits,
        "thresholds": thresholds,
        "selection_split": selection_split,
        "final_split": final_split,
        "selected_counts": selected_counts,
        "total_counts": total_counts,
        "elapsed_sec": elapsed_sec,
        "outputs": {
            "spatial_sample_rows": str(run_dir / "spatial_sample_rows.csv"),
            "spatial_threshold_summary": str(run_dir / "spatial_threshold_summary.csv"),
            "selection_strategy_comparison": str(run_dir / "selection_strategy_comparison.csv"),
            "family_strategy_summary": str(run_dir / "family_strategy_summary.csv"),
            "spatial_risk_correlations": str(run_dir / "spatial_risk_correlations.csv"),
            "collision_case_index": str(run_dir / "collision_case_index.csv"),
            "report": str(config["outputs"]["report"]),
            "log": str(log_path),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(Path(config["outputs"]["report"]), config, run_dir, strategy_rows, family_rows, correlation_rows, preview_rows)
    log(f"Finished path-aware spatial diagnostics in {elapsed_sec:.1f}s")


if __name__ == "__main__":
    main()
