from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
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
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_path_planning import (  # noqa: E402
    ModelPredictor,
    astar,
    evaluate_prediction,
    load_label_mask,
    load_rgb_for_preview,
    make_problem,
)
from train_baselines import ensure_dir, load_json, read_ids  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return math.nan
    return float(np.mean(finite))


def finite_median(values: list[float]) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return math.nan
    return float(np.median(finite))


def scaled_planning_config(config: dict[str, Any], resolution: int) -> dict[str, Any]:
    scale = float(resolution) / float(config["reference_resolution"])
    return {
        "image_size": int(resolution),
        "connectivity": int(config["connectivity"]),
        "min_gt_component_pixels": int(round(float(config["base_min_gt_component_pixels"]) * scale * scale)),
        "min_start_goal_distance": float(config["base_min_start_goal_distance"]) * scale,
    }


def prepare_model_configs(source_planning_config: dict[str, Any], keep_names: set[str]) -> list[dict[str, Any]]:
    models = []
    for model in source_planning_config["models"]:
        if model["name"] in keep_names:
            models.append(deepcopy(model))
    missing = sorted(keep_names - {m["name"] for m in models})
    if missing:
        raise KeyError(f"Missing model configs in source planning config: {missing}")
    return models


def component_stats(mask: np.ndarray, connectivity: int = 8) -> dict[str, Any]:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    count = max(0, int(num_labels) - 1)
    if count == 0:
        return {
            "component_count": 0,
            "largest_component_pixels": 0,
            "small_component_count": 0,
        }
    sizes = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    return {
        "component_count": count,
        "largest_component_pixels": int(sizes.max()),
        "small_component_count": int((sizes < 25).sum()),
    }


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def build_topology_rows(
    sample_ids: list[str],
    protocol_config: dict[str, Any],
    config: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ref_res = int(config["reference_resolution"])
    target_res = max(int(r) for r in config["resolutions"])
    ref_planning = scaled_planning_config(config, ref_res)
    target_planning = scaled_planning_config(config, target_res)
    for sample_id in sample_ids:
        gt_ref = load_label_mask(sample_id, protocol_config, ref_res)
        gt_target = load_label_mask(sample_id, protocol_config, target_res)
        gt_ref_up = cv2.resize(gt_ref.astype(np.uint8), (target_res, target_res), interpolation=cv2.INTER_NEAREST).astype(bool)
        stable = np.logical_and(gt_target, gt_ref_up)
        lost = np.logical_and(gt_target, ~gt_ref_up)
        extra = np.logical_and(~gt_target, gt_ref_up)
        target_road_pixels = int(gt_target.sum())
        ref_up_road_pixels = int(gt_ref_up.sum())
        target_stats = component_stats(gt_target, int(config["connectivity"]))
        ref_up_stats = component_stats(gt_ref_up, int(config["connectivity"]))
        problem_ref, selection_ref = make_problem(sample_id, protocol_config, ref_planning)
        problem_target, selection_target = make_problem(sample_id, protocol_config, target_planning)
        rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "scene_prefix": sample_id.split("_")[0],
                "reference_resolution": ref_res,
                "target_resolution": target_res,
                "topology_iou_ref_up_vs_target": binary_iou(gt_ref_up, gt_target),
                "target_road_pixels": target_road_pixels,
                "ref_up_road_pixels": ref_up_road_pixels,
                "lost_target_road_pixels": int(lost.sum()),
                "extra_ref_road_pixels": int(extra.sum()),
                "stable_road_pixels": int(stable.sum()),
                "lost_target_road_fraction": float(lost.sum() / max(1, target_road_pixels)),
                "extra_ref_road_fraction_vs_target": float(extra.sum() / max(1, target_road_pixels)),
                "target_component_count": target_stats["component_count"],
                "ref_up_component_count": ref_up_stats["component_count"],
                "component_count_delta_ref_up_minus_target": int(ref_up_stats["component_count"] - target_stats["component_count"]),
                "target_largest_component_pixels": target_stats["largest_component_pixels"],
                "ref_up_largest_component_pixels": ref_up_stats["largest_component_pixels"],
                "eligible_ref_resolution": bool(selection_ref.get("selected", False)),
                "eligible_target_resolution": bool(selection_target.get("selected", False)),
                "ref_skip_reason": selection_ref.get("skip_reason", ""),
                "target_skip_reason": selection_target.get("skip_reason", ""),
            }
        )
    return rows


def summarize_topology(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)
    summaries = []
    for split, split_rows in sorted(grouped.items()):
        summaries.append(
            {
                "split": split,
                "samples": len(split_rows),
                "mean_topology_iou_ref_up_vs_target": finite_mean([r["topology_iou_ref_up_vs_target"] for r in split_rows]),
                "median_topology_iou_ref_up_vs_target": finite_median([r["topology_iou_ref_up_vs_target"] for r in split_rows]),
                "mean_lost_target_road_fraction": finite_mean([r["lost_target_road_fraction"] for r in split_rows]),
                "median_lost_target_road_fraction": finite_median([r["lost_target_road_fraction"] for r in split_rows]),
                "mean_extra_ref_road_fraction_vs_target": finite_mean([r["extra_ref_road_fraction_vs_target"] for r in split_rows]),
                "eligible_at_reference": int(sum(bool(r["eligible_ref_resolution"]) for r in split_rows)),
                "eligible_at_target": int(sum(bool(r["eligible_target_resolution"]) for r in split_rows)),
                "eligible_at_both": int(sum(bool(r["eligible_ref_resolution"]) and bool(r["eligible_target_resolution"]) for r in split_rows)),
                "target_more_components_than_ref_up": int(
                    sum(int(r["target_component_count"]) > int(r["ref_up_component_count"]) for r in split_rows)
                ),
                "ref_up_more_components_than_target": int(
                    sum(int(r["ref_up_component_count"]) > int(r["target_component_count"]) for r in split_rows)
                ),
            }
        )
    return summaries


def aggregate_planning_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["resolution"], row["split"], row["model"], row["threshold"])
        grouped[key].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        resolution, split, model, threshold = key
        found = [bool(r["path_found"]) for r in group]
        success = [bool(r["success"]) for r in group]
        found_count = sum(found)
        success_count = sum(success)
        unsafe_found = sum(bool(r["path_found"]) and not bool(r["success"]) for r in group)
        path_found_rate = found_count / max(1, len(group))
        success_rate = success_count / max(1, len(group))
        out.append(
            {
                "resolution": resolution,
                "split": split,
                "model": model,
                "threshold": threshold,
                "evaluated_samples": len(group),
                "path_found_rate": path_found_rate,
                "success_rate_collision_free": success_rate,
                "unsafe_found_rate": unsafe_found / max(1, len(group)),
                "collision_rate_among_found": unsafe_found / max(1, found_count),
                "planning_safety_utility": success_rate - (path_found_rate - success_rate),
                "mean_map_iou": finite_mean([float(r["map_iou"]) for r in group]),
                "mean_collision_fraction_found": finite_mean(
                    [float(r["collision_fraction"]) for r in group if bool(r["path_found"])]
                ),
                "mean_path_false_positive_fraction_found": finite_mean(
                    [float(r["path_false_positive_fraction"]) for r in group if bool(r["path_found"])]
                ),
                "mean_inference_ms": finite_mean([float(r["inference_time_ms"]) for r in group]),
                "mean_planning_ms": finite_mean([float(r["planning_time_ms"]) for r in group]),
            }
        )
    return out


def choose_thresholds(summary_rows: list[dict[str, Any]], selection_split: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row["split"] == selection_split:
            grouped[(int(row["resolution"]), str(row["model"]))].append(row)
    selected = []
    for (resolution, model), candidates in sorted(grouped.items()):
        best = max(
            candidates,
            key=lambda r: (
                float(r["planning_safety_utility"]),
                float(r["success_rate_collision_free"]),
                float(r["path_found_rate"]),
                float(r["mean_map_iou"]),
                -float(r["collision_rate_among_found"]),
            ),
        )
        selected.append(
            {
                "resolution": resolution,
                "model": model,
                "selected_threshold": best["threshold"],
                "selection_split": selection_split,
                "selection_success_rate_collision_free": best["success_rate_collision_free"],
                "selection_planning_safety_utility": best["planning_safety_utility"],
                "selection_path_found_rate": best["path_found_rate"],
                "selection_mean_map_iou": best["mean_map_iou"],
                "selection_collision_rate_among_found": best["collision_rate_among_found"],
            }
        )
    return selected


def selected_test_rows(summary_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]], final_split: str) -> list[dict[str, Any]]:
    index = {
        (int(r["resolution"]), str(r["model"]), float(r["threshold"])): r
        for r in summary_rows
        if r["split"] == final_split
    }
    out = []
    for selected in threshold_rows:
        key = (int(selected["resolution"]), str(selected["model"]), float(selected["selected_threshold"]))
        test = index.get(key)
        if test is None:
            continue
        merged = dict(selected)
        for k, v in test.items():
            if k not in {"resolution", "model", "threshold"}:
                merged[f"test_{k}"] = v
        out.append(merged)
    return out


def run_resolution_planning(
    config: dict[str, Any],
    protocol_config: dict[str, Any],
    baseline_config: dict[str, Any],
    semantic_config: dict[str, Any],
    model_configs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device_name = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    thresholds = [float(t) for t in config["thresholds"]]
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    outputs_root = Path(protocol_config["outputs_root"])
    random.seed(int(config["seed"]))
    for resolution in [int(r) for r in config["resolutions"]]:
        planning_config = scaled_planning_config(config, resolution)
        baseline_cfg = deepcopy(baseline_config)
        semantic_cfg = deepcopy(semantic_config)
        baseline_cfg["image_size"] = resolution
        semantic_cfg["image_size"] = resolution
        for split in [str(config["selection_split"]), str(config["final_split"])]:
            sample_ids = read_ids(outputs_root / "protocol" / f"{split}.txt")
            max_samples = int(config.get("max_samples_per_split", 0))
            if max_samples > 0:
                sample_ids = sample_ids[:max_samples]
            predictors = [
                ModelPredictor(model_cfg, sample_ids, protocol_config, baseline_cfg, semantic_cfg, device)
                for model_cfg in model_configs
            ]
            for sample_i, sample_id in enumerate(sample_ids):
                problem, selection = make_problem(sample_id, protocol_config, planning_config)
                selection_rows.append(
                    {
                        "resolution": resolution,
                        "split": split,
                        **selection,
                    }
                )
                if problem is None:
                    continue
                for predictor in predictors:
                    prob, inference_ms = predictor.predict(problem, seed=int(config["seed"]) + sample_i)
                    for threshold in thresholds:
                        result = evaluate_prediction(
                            problem,
                            prob,
                            threshold=threshold,
                            connectivity=int(planning_config["connectivity"]),
                        )
                        pred_mask = result["pred_mask"]
                        path = result["path"]
                        path_fp_fraction = math.nan
                        if path is not None:
                            path_fp_fraction = float(sum(not bool(problem.gt_mask[y, x]) for y, x in path) / max(1, len(path)))
                        rows.append(
                            {
                                "resolution": resolution,
                                "split": split,
                                "sample_id": sample_id,
                                "scene_prefix": sample_id.split("_")[0],
                                "model": predictor.name,
                                "threshold": threshold,
                                "path_found": bool(result["path_found"]),
                                "success": bool(result["success"]),
                                "collision_cells": int(result["collision_cells"]),
                                "collision_fraction": float(result["collision_fraction"])
                                if np.isfinite(result["collision_fraction"])
                                else math.nan,
                                "path_false_positive_fraction": path_fp_fraction,
                                "map_iou": float(result["map_iou"]),
                                "path_length_ratio": float(result["path_length_ratio"])
                                if np.isfinite(result["path_length_ratio"])
                                else math.nan,
                                "start_on_pred": bool(result["start_on_pred"]),
                                "goal_on_pred": bool(result["goal_on_pred"]),
                                "inference_time_ms": float(inference_ms),
                                "planning_time_ms": float(result["planning_time_ms"]),
                                "pred_road_fraction": float(pred_mask.mean()),
                                "gt_component_pixels": int(problem.component_pixels),
                            }
                        )
            del predictors
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows, selection_rows


def make_coarsening_error_rgb(gt_target: np.ndarray, gt_ref_up: np.ndarray) -> np.ndarray:
    img = np.zeros((gt_target.shape[0], gt_target.shape[1], 3), dtype=np.float32)
    stable = np.logical_and(gt_target, gt_ref_up)
    lost = np.logical_and(gt_target, ~gt_ref_up)
    extra = np.logical_and(~gt_target, gt_ref_up)
    img[stable] = (0.82, 0.82, 0.82)
    img[lost] = (0.16, 0.45, 0.95)
    img[extra] = (0.95, 0.45, 0.06)
    return img


def draw_path(ax: plt.Axes, path: list[tuple[int, int]] | None, color: str, linewidth: float = 1.4, linestyle: str = "-") -> None:
    if not path:
        return
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle)


def draw_start_goal(ax: plt.Axes, start: tuple[int, int], goal: tuple[int, int]) -> None:
    ax.scatter([start[1]], [start[0]], s=22, c="#22c55e", edgecolors="white", linewidths=0.7, zorder=4)
    ax.scatter([goal[1]], [goal[0]], s=22, c="#ef4444", edgecolors="white", linewidths=0.7, zorder=4)


def build_resolution_figure(
    config: dict[str, Any],
    protocol_config: dict[str, Any],
    baseline_config: dict[str, Any],
    semantic_config: dict[str, Any],
    model_configs: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    target_res = max(int(r) for r in config["resolutions"])
    ref_res = int(config["reference_resolution"])
    planning_config = scaled_planning_config(config, target_res)
    semantic_cfg = deepcopy(semantic_config)
    baseline_cfg = deepcopy(baseline_config)
    semantic_cfg["image_size"] = target_res
    baseline_cfg["image_size"] = target_res
    semantic_model_cfg = next(m for m in model_configs if m["name"] == "semantic_clean")
    final_split = str(config["final_split"])
    semantic_threshold = next(
        float(r["selected_threshold"])
        for r in selected_rows
        if int(r["resolution"]) == target_res and r["model"] == "semantic_clean"
    )
    candidates = [
        r
        for r in topology_rows
        if r["split"] == final_split and bool(r["eligible_target_resolution"])
    ]
    candidates.sort(key=lambda r: (float(r["topology_iou_ref_up_vs_target"]), -float(r["lost_target_road_fraction"])))
    chosen = []
    used_prefixes: set[str] = set()
    for row in candidates:
        prefix = str(row["scene_prefix"])
        if prefix in used_prefixes:
            continue
        chosen.append(row)
        used_prefixes.add(prefix)
        if len(chosen) >= int(config["figure_case_count"]):
            break
    if len(chosen) < int(config["figure_case_count"]):
        for row in candidates:
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= int(config["figure_case_count"]):
                break
    sample_ids = [str(r["sample_id"]) for r in chosen]
    predictor = ModelPredictor(semantic_model_cfg, sample_ids, protocol_config, baseline_cfg, semantic_cfg, torch.device(config.get("device", "cpu") if torch.cuda.is_available() else "cpu"))
    fig, axes = plt.subplots(
        len(sample_ids),
        6,
        figsize=(15.8, 2.55 * max(1, len(sample_ids))),
        gridspec_kw={"wspace": 0.035, "hspace": 0.13},
    )
    if len(sample_ids) == 1:
        axes = np.expand_dims(axes, axis=0)
    col_titles = ["RGB + oracle", "GT 256", "GT 128->256", "Coarsening error", "Semantic 256 pred.", "Semantic prob."]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10.5)
    for r, sample_id in enumerate(sample_ids):
        problem, _selection = make_problem(sample_id, protocol_config, planning_config)
        if problem is None:
            continue
        rgb = load_rgb_for_preview(sample_id, protocol_config, target_res)
        gt_target = load_label_mask(sample_id, protocol_config, target_res)
        gt_ref = load_label_mask(sample_id, protocol_config, ref_res)
        gt_ref_up = cv2.resize(gt_ref.astype(np.uint8), (target_res, target_res), interpolation=cv2.INTER_NEAREST).astype(bool)
        prob, _elapsed_ms = predictor.predict(problem, seed=int(config["seed"]) + r)
        result = evaluate_prediction(problem, prob, semantic_threshold, int(config["connectivity"]))
        pred_mask = result["pred_mask"]
        panels = [
            rgb,
            gt_target,
            gt_ref_up,
            make_coarsening_error_rgb(gt_target, gt_ref_up),
            pred_mask,
            prob,
        ]
        cmaps = [None, "gray", "gray", None, "gray", "viridis"]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap=cmaps[c], vmin=0 if c in {1, 2, 4, 5} else None, vmax=1 if c in {1, 2, 4, 5} else None)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#d1d5db")
        draw_path(axes[r, 0], problem.oracle_path, "#2563eb", 1.2)
        draw_start_goal(axes[r, 0], problem.start, problem.goal)
        draw_path(axes[r, 4], result["path"], "#f97316" if result["success"] else "#ef4444", 1.25)
        draw_start_goal(axes[r, 4], problem.start, problem.goal)
        draw_path(axes[r, 5], result["path"], "#f97316" if result["success"] else "#ef4444", 1.15)
        draw_start_goal(axes[r, 5], problem.start, problem.goal)
        label = (
            f"{sample_id}\n"
            f"IoU128up={float(chosen[r]['topology_iou_ref_up_vs_target']):.3f}\n"
            f"lost={float(chosen[r]['lost_target_road_fraction']):.2f}"
        )
        axes[r, 0].set_ylabel(label, fontsize=8.7, rotation=0, ha="right", va="center", labelpad=52)
    fig.text(
        0.5,
        0.012,
        "Blue/orange/red paths denote oracle, safe semantic, and unsafe semantic routes. Coarsening error: gray=preserved road, blue=target road lost by 128x128 coarsening, orange=extra road from coarse grid.",
        ha="center",
        fontsize=8.3,
    )
    png_path = output_dir / "resolution_case_sheet.png"
    pdf_path = output_dir / "resolution_case_sheet.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"figure_png": str(png_path), "figure_pdf": str(pdf_path)}


def write_report(
    config: dict[str, Any],
    report_path: Path,
    output_dir: Path,
    topology_summary: list[dict[str, Any]],
    selected_test: list[dict[str, Any]],
    figure_paths: dict[str, str],
) -> None:
    test_topology = next((r for r in topology_summary if r["split"] == config["final_split"]), {})
    lines = [
        "# Resolution Diagnostic",
        "",
        "This diagnostic studies topology loss at the 128 x 128 grid scale. It does not replace the main protocol and does not claim a fully retrained high-resolution benchmark.",
        "",
        "## Scope Boundary",
        "",
        "- The 128 x 128 setting is treated as a compact route-level semantic message for low-bandwidth UAV-to-UGV communication.",
        "- The 256 x 256 model evaluation reuses fully convolutional checkpoints trained under the original 128 x 128 protocol, so it is a resolution-transfer diagnostic.",
        "- The topology analysis directly compares a 256 x 256 ground-truth road mask against the same mask coarsened through 128 x 128 and then upsampled.",
        "",
        "## Topology Summary",
        "",
    ]
    if test_topology:
        lines.extend(
            [
                f"- Test samples: {test_topology['samples']}",
                f"- Mean IoU between 128->256 coarsened GT and direct 256 GT: {float(test_topology['mean_topology_iou_ref_up_vs_target']):.4f}",
                f"- Median IoU between 128->256 coarsened GT and direct 256 GT: {float(test_topology['median_topology_iou_ref_up_vs_target']):.4f}",
                f"- Mean fraction of 256-road pixels lost by 128 coarsening: {float(test_topology['mean_lost_target_road_fraction']):.4f}",
                f"- Samples eligible for planning at both resolutions: {test_topology['eligible_at_both']}",
                "",
            ]
        )
    lines.extend(["## Selected Test Results", ""])
    for row in selected_test:
        if int(row["resolution"]) not in {128, 256}:
            continue
        lines.append(
            "- "
            f"{row['model']} at {row['resolution']} px: "
            f"threshold={float(row['selected_threshold']):.2f}, "
            f"IoU={float(row['test_mean_map_iou']):.4f}, "
            f"safe={float(row['test_success_rate_collision_free']):.4f}, "
            f"found={float(row['test_path_found_rate']):.4f}, "
            f"safety utility={float(row['test_planning_safety_utility']):.4f}."
        )
    lines.extend(
        [
            "",
            "## Suggested Manuscript Use",
            "",
            "The safest interpretation is that 128 x 128 is a deliberate coarse semantic payload, not a final obstacle-avoidance map. The topology table quantifies how much road geometry can be lost by coarsening, and the 256 x 256 transfer check shows whether the same learned checkpoints remain plausible when evaluated on a finer grid. The response letter should present this as a limitation-aware diagnostic rather than a new main claim.",
            "",
            "## Output Files",
            "",
            f"- `{output_dir / 'topology_resolution_rows.csv'}`",
            f"- `{output_dir / 'topology_resolution_summary.csv'}`",
            f"- `{output_dir / 'planner_resolution_rows.csv'}`",
            f"- `{output_dir / 'planner_resolution_summary.csv'}`",
            f"- `{output_dir / 'planner_resolution_selected_test.csv'}`",
            f"- `{figure_paths['figure_png']}`",
            f"- `{figure_paths['figure_pdf']}`",
            "",
        ]
    )
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(FINALLY_ROOT / "configs" / "resolution_diagnostic.json"))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    output_dir = Path(config["output_root"])
    ensure_dir(output_dir)
    protocol_config = load_json(Path(config["protocol_config"]))
    baseline_config = load_json(Path(config["baseline_config"]))
    semantic_config = load_json(Path(config["semantic_config"]))
    source_planning_config = load_json(Path(config["source_planning_config"]))
    model_configs = prepare_model_configs(source_planning_config, set(config["models"]))

    outputs_root = Path(protocol_config["outputs_root"])
    split_ids: dict[str, list[str]] = {}
    for split in sorted({str(config["selection_split"]), str(config["final_split"])}):
        ids = read_ids(outputs_root / "protocol" / f"{split}.txt")
        max_samples = int(config.get("max_samples_per_split", 0))
        if max_samples > 0:
            ids = ids[:max_samples]
        split_ids[split] = ids

    started = time.time()
    topology_rows: list[dict[str, Any]] = []
    for split, ids in split_ids.items():
        topology_rows.extend(build_topology_rows(ids, protocol_config, config, split))
    topology_summary = summarize_topology(topology_rows)

    planner_rows, sample_selection_rows = run_resolution_planning(
        config,
        protocol_config,
        baseline_config,
        semantic_config,
        model_configs,
    )
    planner_summary = aggregate_planning_rows(planner_rows)
    threshold_selection = choose_thresholds(planner_summary, str(config["selection_split"]))
    selected_test = selected_test_rows(planner_summary, threshold_selection, str(config["final_split"]))

    figure_paths = build_resolution_figure(
        config,
        protocol_config,
        baseline_config,
        semantic_config,
        model_configs,
        topology_rows,
        selected_test,
        output_dir,
    )

    write_csv(output_dir / "topology_resolution_rows.csv", topology_rows)
    write_csv(output_dir / "topology_resolution_summary.csv", topology_summary)
    write_csv(output_dir / "planner_sample_selection.csv", sample_selection_rows)
    write_csv(output_dir / "planner_resolution_rows.csv", planner_rows)
    write_csv(output_dir / "planner_resolution_summary.csv", planner_summary)
    write_csv(output_dir / "planner_resolution_threshold_selection.csv", threshold_selection)
    write_csv(output_dir / "planner_resolution_selected_test.csv", selected_test)
    (output_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        "run_name": config["run_name"],
        "elapsed_seconds": time.time() - started,
        "device_requested": config.get("device"),
        "device_used": "cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu",
        "topology_summary": topology_summary,
        "selected_test": selected_test,
        "figure_paths": figure_paths,
        "scope_boundary": "Resolution-transfer diagnostic using existing 128x128-trained checkpoints; not a fully retrained 256x256 benchmark.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(config, Path(config["report_path"]), output_dir, topology_summary, selected_test, figure_paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
