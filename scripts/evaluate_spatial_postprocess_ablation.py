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
DEFAULT_CONFIG = FINALLY_ROOT / "configs" / "spatial_postprocess_ablation_v1.json"
PROTOCOL_CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"
BASELINE_CONFIG_PATH = FINALLY_ROOT / "configs" / "baseline_training_long.json"
SEMANTIC_CONFIG_PATH = FINALLY_ROOT / "configs" / "semantic_comm_training.json"

sys.path.insert(0, str(SCRIPT_DIR))
from train_baselines import ensure_dir, load_json, read_ids
from evaluate_path_planning import ModelPredictor, astar, binary_iou, make_problem, path_length
from evaluate_path_aware_spatial_diagnostics import compute_spatial_metrics, model_meta, summarize_threshold_rows


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
        return float(value)
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


def load_thresholds(config: dict[str, Any]) -> dict[str, float]:
    strategy = str(config["threshold_strategy"])
    rows = [row for row in read_csv(Path(config["threshold_selection_csv"])) if row.get("strategy") == strategy]
    thresholds = {row["model"]: to_float(row["selected_threshold"]) for row in rows}
    if not thresholds:
        raise RuntimeError(f"No thresholds found for strategy={strategy}")
    return thresholds


def remove_small_components(mask: np.ndarray, min_pixels: int, connectivity: int) -> np.ndarray:
    if min_pixels <= 1 or int(mask.sum()) == 0:
        return mask.astype(bool)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    out = np.zeros(mask.shape, dtype=bool)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels:
            out |= labels == label
    return out


def keep_largest_component(mask: np.ndarray, connectivity: int) -> np.ndarray:
    if int(mask.sum()) == 0:
        return mask.astype(bool)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    if num_labels <= 1:
        return np.zeros(mask.shape, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    return labels == best_label


def morph_open(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iterations).astype(bool)


def component_confidence_filter(
    mask: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    connectivity: int,
    min_component_pixels: int,
    min_mean_probability_margin: float,
) -> np.ndarray:
    if int(mask.sum()) == 0:
        return mask.astype(bool)
    min_mean = threshold + min_mean_probability_margin
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    out = np.zeros(mask.shape, dtype=bool)
    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue
        if float(prob[component].mean()) < min_mean:
            continue
        out |= component
    return out


def apply_postprocess(mask: np.ndarray, prob: np.ndarray, threshold: float, variant: dict[str, Any], connectivity: int) -> np.ndarray:
    kind = str(variant["type"])
    if kind == "none":
        return mask.astype(bool)
    if kind == "remove_small_components":
        return remove_small_components(mask, int(variant["min_component_pixels"]), connectivity)
    if kind == "keep_largest_component":
        return keep_largest_component(mask, connectivity)
    if kind == "morph_open":
        return morph_open(mask, int(variant["kernel_size"]), int(variant.get("iterations", 1)))
    if kind == "component_confidence":
        return component_confidence_filter(
            mask,
            prob,
            threshold,
            connectivity,
            int(variant["min_component_pixels"]),
            float(variant["min_mean_probability_margin"]),
        )
    if kind == "pipeline":
        out = mask.astype(bool)
        for step in variant["steps"]:
            out = apply_postprocess(out, prob, threshold, step, connectivity)
        return out
    raise ValueError(f"Unknown postprocess type: {kind}")


def evaluate_mask(problem: Any, pred_mask: np.ndarray, connectivity: int) -> dict[str, Any]:
    pred_mask = pred_mask.astype(bool)
    start_on_pred = bool(pred_mask[problem.start])
    goal_on_pred = bool(pred_mask[problem.goal])
    started = time.perf_counter()
    path = astar(pred_mask, problem.start, problem.goal, connectivity)
    planning_time_ms = (time.perf_counter() - started) * 1000.0
    found = path is not None
    collision_cells = 0
    length = math.nan
    length_ratio = math.nan
    collision_fraction = math.nan
    if path is not None:
        collision_cells = int(sum(not bool(problem.gt_mask[y, x]) for y, x in path))
        length = path_length(path)
        length_ratio = length / problem.oracle_length if problem.oracle_length > 0 else math.nan
        collision_fraction = collision_cells / max(1, len(path))
    return {
        "pred_mask": pred_mask,
        "path": path,
        "start_on_pred": start_on_pred,
        "goal_on_pred": goal_on_pred,
        "path_found": found,
        "success": bool(found and collision_cells == 0),
        "collision_cells": collision_cells,
        "collision_fraction": collision_fraction,
        "path_length": length,
        "path_length_ratio": length_ratio,
        "map_iou": binary_iou(pred_mask, problem.gt_mask),
        "planning_time_ms": planning_time_ms,
    }


def select_postprocess(summary_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    selection_split = str(config["selection_split"])
    final_split = str(config["final_split"])
    by_key = {(row["split"], row["model"], row["postprocess"]): row for row in summary_rows}
    out: list[dict[str, Any]] = []
    models = sorted({row["model"] for row in summary_rows})
    for model in models:
        val_rows = [row for row in summary_rows if row["split"] == selection_split and row["model"] == model]
        if not val_rows:
            continue

        def rank(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
            return (
                finite(row["planning_safety_utility"], -1e9),
                -finite(row["collision_rate_among_found"], 1.0),
                -finite(row["mean_path_corridor_fp_rate_found"], 1.0),
                finite(row["mean_map_iou"], -1e9),
                1.0 if row["postprocess"] == "raw" else 0.0,
            )

        chosen_val = max(val_rows, key=rank)
        chosen_test = by_key[(final_split, model, chosen_val["postprocess"])]
        row = {
            "model": model,
            "family": chosen_test.get("family", ""),
            "display_name": chosen_test.get("display_name", ""),
            "checkpoint_selection": chosen_test.get("checkpoint_selection", ""),
            "pair_id": chosen_test.get("pair_id", ""),
            "seed": chosen_test.get("seed", ""),
            "threshold_strategy": chosen_test["threshold_strategy"],
            "threshold": chosen_test["threshold"],
            "selected_postprocess": chosen_test["postprocess"],
        }
        for key, value in chosen_val.items():
            if key in {"split", "model", "family", "display_name", "checkpoint_selection", "pair_id", "seed", "threshold_strategy", "threshold", "postprocess"}:
                continue
            row[f"val_{key}"] = value
        for key, value in chosen_test.items():
            if key in {"split", "model", "family", "display_name", "checkpoint_selection", "pair_id", "seed", "threshold_strategy", "threshold", "postprocess"}:
                continue
            row[f"test_{key}"] = value
        out.append(row)
    return out


def delta_vs_raw(selected_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], final_split: str) -> list[dict[str, Any]]:
    raw_by_model = {row["model"]: row for row in summary_rows if row["split"] == final_split and row["postprocess"] == "raw"}
    out: list[dict[str, Any]] = []
    for row in selected_rows:
        raw = raw_by_model[row["model"]]
        out.append(
            {
                "model": row["model"],
                "family": row.get("family", ""),
                "checkpoint_selection": row.get("checkpoint_selection", ""),
                "selected_postprocess": row["selected_postprocess"],
                "threshold": row["threshold"],
                "test_safety_delta_vs_raw": finite(row["test_planning_safety_utility"]) - finite(raw["planning_safety_utility"]),
                "test_success_delta_vs_raw": finite(row["test_success_rate_collision_free"]) - finite(raw["success_rate_collision_free"]),
                "test_path_found_delta_vs_raw": finite(row["test_path_found_rate"]) - finite(raw["path_found_rate"]),
                "test_collision_found_delta_vs_raw": finite(row["test_collision_rate_among_found"]) - finite(raw["collision_rate_among_found"]),
                "test_corridor_fp_delta_vs_raw": finite(row["test_mean_path_corridor_fp_rate_found"]) - finite(raw["mean_path_corridor_fp_rate_found"]),
                "test_map_iou_delta_vs_raw": finite(row["test_mean_map_iou"]) - finite(raw["mean_map_iou"]),
            }
        )
    return out


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, key_values)}
        item["members"] = len(group)
        for metric in metrics:
            values = [row.get(metric, math.nan) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = std(values)
        out.append(item)
    return out


def plot_variant_summary(path: Path, variant_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in variant_rows if row["split"] == "test"]
    labels = [row["postprocess"] for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes[0].bar(x, [finite(row["planning_safety_utility_mean"]) for row in rows], color="#0f766e")
    axes[0].set_title("Mean test safety")
    axes[1].bar(x, [finite(row["collision_rate_among_found_mean"]) for row in rows], color="#dc2626")
    axes[1].set_title("Mean collision/found")
    axes[2].bar(x, [finite(row["mean_path_corridor_fp_rate_found_mean"]) for row in rows], color="#f97316")
    axes[2].set_title("Mean corridor FP")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_delta_summary(path: Path, delta_rows: list[dict[str, Any]]) -> None:
    grouped = aggregate_rows(
        delta_rows,
        ["selected_postprocess"],
        ["test_safety_delta_vs_raw", "test_collision_found_delta_vs_raw", "test_corridor_fp_delta_vs_raw"],
    )
    labels = [row["selected_postprocess"] for row in grouped]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes[0].bar(x, [finite(row["test_safety_delta_vs_raw_mean"]) for row in grouped], color="#0f766e")
    axes[0].set_title("Safety delta vs raw")
    axes[1].bar(x, [finite(row["test_collision_found_delta_vs_raw_mean"]) for row in grouped], color="#dc2626")
    axes[1].set_title("Collision/found delta")
    axes[2].bar(x, [finite(row["test_corridor_fp_delta_vs_raw_mean"]) for row in grouped], color="#f97316")
    axes[2].set_title("Corridor FP delta")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_family_selected(path: Path, family_rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['family']} {row['checkpoint_selection']}" for row in family_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    axes[0].bar(x, [finite(row["test_planning_safety_utility_mean"]) for row in family_rows], color="#0f766e")
    axes[0].set_title("Selected postprocess safety")
    axes[1].bar(x, [finite(row["test_collision_rate_among_found_mean"]) for row in family_rows], color="#dc2626")
    axes[1].set_title("Selected postprocess collision/found")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def markdown_variant_table(rows: list[dict[str, Any]]) -> str:
    test_rows = [row for row in rows if row["split"] == "test"]
    lines = [
        "| postprocess | members | safety | collision/found | corridor FP | mask IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['postprocess']}`",
                    str(row["members"]),
                    fmt(row["planning_safety_utility_mean"]),
                    fmt(row["collision_rate_among_found_mean"]),
                    fmt(row["mean_path_corridor_fp_rate_found_mean"]),
                    fmt(row["mean_map_iou_mean"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(
    report_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    selected_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> None:
    ensure_dir(report_path.parent)
    best = max(selected_rows, key=lambda row: finite(row["test_planning_safety_utility"], -1e9))
    improved_safety = [row for row in delta_rows if finite(row["test_safety_delta_vs_raw"]) > 1e-12]
    reduced_collision = [row for row in delta_rows if finite(row["test_collision_found_delta_vs_raw"]) < -1e-12]
    reduced_corridor = [row for row in delta_rows if finite(row["test_corridor_fp_delta_vs_raw"]) < -1e-12]
    rel = run_dir.relative_to(FINALLY_ROOT).as_posix()
    lines = [
        "# 第十六步：轻量空间后处理消融报告",
        "",
        "生成日期：2026-05-19",
        "",
        "## 实验目的",
        "",
        "第十五步说明空间风险适合作为 safety utility 的约束或 tie-breaker。第十六步进一步验证：不训练新模型，只对预测二值 mask 做轻量后处理，是否可以降低 collision/found 或路径走廊假阳性。",
        "",
        "本步使用第十五步 `constrained_safety_5pp` 选出的阈值作为每个模型的固定基础阈值；后处理变体只用 validation 选择，再报告 test 结果。",
        "",
        "## 后处理变体",
        "",
        "- `raw`：不做后处理。",
        "- `remove_small_64/128/256`：删除小于指定像素数的预测可通行连通块。",
        "- `keep_largest`：只保留最大的预测可通行连通块。",
        "- `open3`：3x3 形态学开运算，用来切断细小假阳性桥。",
        "- `open3_remove64`：先开运算，再删除小连通块。",
        "- `component_conf_64_tplus05`：删除小连通块，同时要求连通块平均概率至少高于阈值 0.05。",
        "",
        "## test 上所有模型的后处理均值",
        "",
        markdown_variant_table(variant_rows),
        "",
        "## 核心发现",
        "",
        f"- validation 选择后，test safety utility 最高的单模型结果是 `{best['model']}`，后处理 `{best['selected_postprocess']}`，test safety utility {fmt(best['test_planning_safety_utility'])}。",
        f"- 与 raw 相比，{len(improved_safety)}/24 个模型提升 test safety utility，{len(reduced_collision)}/24 个模型降低 collision/found，{len(reduced_corridor)}/24 个模型降低 path corridor FP。",
        f"- selected postprocess 的平均 safety delta 为 {fmt(mean([row['test_safety_delta_vs_raw'] for row in delta_rows]))}，平均 collision/found delta 为 {fmt(mean([row['test_collision_found_delta_vs_raw'] for row in delta_rows]))}。",
        "",
        "严谨结论：轻量后处理是有价值的消融，但不是万能修复。它可以在部分模型上降低碰撞或路径走廊假阳性；如果平均收益有限，就应把它写成空间错误结构的后处理分析，而不是主方法性能来源。",
        "",
        "## 产物",
        "",
        "- `configs/spatial_postprocess_ablation_v1.json`：第十六步空间后处理消融配置。",
        "- `scripts/evaluate_spatial_postprocess_ablation.py`：第十六步后处理评估、选择、绘图和报告脚本。",
        f"- `{rel}/config_snapshot.json`：配置快照。",
        f"- `{rel}/source_model_config_snapshot.json`：模型来源配置快照。",
        f"- `{rel}/threshold_selection_snapshot.csv`：第十五步阈值选择快照。",
        f"- `{rel}/postprocess_sample_rows.csv`：逐 split、模型、样本、后处理变体的路径和空间指标。",
        f"- `{rel}/postprocess_summary.csv`：逐 split、模型、后处理变体的聚合指标。",
        f"- `{rel}/selected_postprocess_by_val.csv`：validation 选择的后处理变体。",
        f"- `{rel}/selected_postprocess_test_summary.csv`：选择后处理迁移到 test 的指标。",
        f"- `{rel}/postprocess_delta_vs_raw.csv`：selected postprocess 相对 raw 的 test delta。",
        f"- `{rel}/family_postprocess_summary.csv`：family/checkpoint 级 selected postprocess 汇总。",
        f"- `{rel}/postprocess_variant_summary.csv`：后处理变体整体均值汇总。",
        f"- `{rel}/postprocess_variant_summary.png`：各后处理变体整体对比图。",
        f"- `{rel}/postprocess_delta_vs_raw.png`：selected postprocess 相对 raw 的 delta 图。",
        f"- `{rel}/family_selected_postprocess.png`：family 级 selected postprocess 结果图。",
        f"- `{rel}/summary.json`：机器可读完整汇总。",
        f"- `{rel}/logs/*.log`：运行日志。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lightweight spatial post-processing ablations.")
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
    log_path = run_dir / "logs" / f"spatial_postprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    started = time.perf_counter()
    thresholds_by_model = load_thresholds(config)
    splits = [str(x) for x in config["splits"]]
    connectivity = int(config["connectivity"])
    corridor_radius = int(config["corridor_radius"])
    seed = int(config["seed"])
    variants = [dict(v) for v in config["postprocess_variants"]]
    weights = {key: float(value) for key, value in config["path_aware_score"].items() if key.endswith("_weight")}
    max_samples = int(args.max_samples if args.max_samples >= 0 else config.get("max_samples_per_split", 0))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    models = [
        model
        for model in source_model_config["models"]
        if (bool(config.get("include_oracle", False)) or str(model.get("type")) != "oracle") and str(model["name"]) in thresholds_by_model
    ]

    log(f"Spatial postprocess ablation run={run_name} models={len(models)} variants={len(variants)} splits={splits} device={device}")
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "source_model_config_snapshot.json").write_text(json.dumps(source_model_config, indent=2), encoding="utf-8")
    write_csv(run_dir / "threshold_selection_snapshot.csv", [row for row in read_csv(Path(config["threshold_selection_csv"])) if row.get("strategy") == config["threshold_strategy"]])

    all_sample_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    sample_selection_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    total_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}

    for split_index, split in enumerate(splits):
        sample_ids = read_ids(Path(protocol_config["outputs_root"]) / "protocol" / f"{split}.txt")
        if max_samples > 0:
            sample_ids = sample_ids[:max_samples]
        total_counts[split] = len(sample_ids)
        problems = []
        split_selection = []
        for index, sample_id in enumerate(sample_ids, start=1):
            problem, selection = make_problem(sample_id, protocol_config, config)
            selection["split"] = split
            split_selection.append(selection)
            if problem is not None:
                problems.append(problem)
            if index % 100 == 0 or index == len(sample_ids):
                log(f"{split}: selected {len(problems)}/{index}")
        selected_counts[split] = len(problems)
        sample_selection_rows.extend(split_selection)
        write_csv(run_dir / f"sample_selection_{split}.csv", split_selection)

        predictor_sample_ids = [problem.sample_id for problem in problems]
        predictors = [
            ModelPredictor(model, predictor_sample_ids, protocol_config, baseline_config, semantic_config, device)
            for model in models
        ]
        for model_index, predictor in enumerate(predictors):
            model_cfg = models[model_index]
            model_name = predictor.name
            threshold = float(thresholds_by_model[model_name])
            log(f"{split}: evaluating {model_name} threshold={threshold:.2f} ({model_index + 1}/{len(predictors)})")
            variant_rows: dict[str, list[dict[str, Any]]] = {str(variant["name"]): [] for variant in variants}
            for problem_index, problem in enumerate(problems, start=1):
                forward_seed = seed + split_index * 1000000 + model_index * 100000 + problem_index
                prob, inference_time_ms = predictor.predict(problem, forward_seed)
                base_mask = prob >= threshold
                inference_rows.append(
                    {
                        "split": split,
                        "model": model_name,
                        "sample_id": problem.sample_id,
                        "threshold": threshold,
                        "inference_time_ms": inference_time_ms,
                    }
                )
                for variant in variants:
                    post_name = str(variant["name"])
                    pred_mask = apply_postprocess(base_mask, prob, threshold, variant, connectivity)
                    result = evaluate_mask(problem, pred_mask, connectivity)
                    spatial = compute_spatial_metrics(problem, prob, threshold, result, connectivity, corridor_radius)
                    row = {
                        "split": split,
                        "model": model_name,
                        **model_meta(model_cfg),
                        "sample_id": problem.sample_id,
                        "threshold_strategy": config["threshold_strategy"],
                        "threshold": threshold,
                        "postprocess": post_name,
                        "postprocess_type": variant["type"],
                        "component_pixels": problem.component_pixels,
                        "start_y": problem.start[0],
                        "start_x": problem.start[1],
                        "goal_y": problem.goal[0],
                        "goal_x": problem.goal[1],
                        "oracle_length": problem.oracle_length,
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
                    variant_rows[post_name].append(row)
                    all_sample_rows.append(row)
                if problem_index % 50 == 0 or problem_index == len(problems):
                    log(f"{split}: {model_name} processed {problem_index}/{len(problems)}")
            for variant in variants:
                post_name = str(variant["name"])
                summary = summarize_threshold_rows(
                    split,
                    model_cfg,
                    threshold,
                    variant_rows[post_name],
                    selected_counts[split],
                    total_counts[split],
                    weights,
                )
                summary["threshold_strategy"] = config["threshold_strategy"]
                summary["postprocess"] = post_name
                summary["postprocess_type"] = variant["type"]
                all_summary_rows.append(summary)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    selected_rows = select_postprocess(all_summary_rows, config)
    delta_rows = delta_vs_raw(selected_rows, all_summary_rows, str(config["final_split"]))
    family_rows = aggregate_rows(
        selected_rows,
        ["family", "checkpoint_selection"],
        [
            "test_planning_safety_utility",
            "test_success_rate_collision_free",
            "test_path_found_rate",
            "test_collision_rate_among_found",
            "test_mean_map_iou",
            "test_mean_path_corridor_fp_rate_found",
        ],
    )
    variant_rows = aggregate_rows(
        all_summary_rows,
        ["split", "postprocess"],
        [
            "planning_safety_utility",
            "success_rate_collision_free",
            "path_found_rate",
            "collision_rate_among_found",
            "mean_map_iou",
            "mean_path_corridor_fp_rate_found",
        ],
    )

    write_csv(run_dir / "sample_selection_all.csv", sample_selection_rows)
    write_csv(run_dir / "inference_manifest.csv", inference_rows)
    write_csv(run_dir / "postprocess_sample_rows.csv", all_sample_rows)
    write_csv(run_dir / "postprocess_summary.csv", all_summary_rows)
    write_csv(run_dir / "selected_postprocess_by_val.csv", selected_rows)
    write_csv(run_dir / "selected_postprocess_test_summary.csv", selected_rows)
    write_csv(run_dir / "postprocess_delta_vs_raw.csv", delta_rows)
    write_csv(run_dir / "family_postprocess_summary.csv", family_rows)
    write_csv(run_dir / "postprocess_variant_summary.csv", variant_rows)
    plot_variant_summary(run_dir / "postprocess_variant_summary.png", variant_rows)
    plot_delta_summary(run_dir / "postprocess_delta_vs_raw.png", delta_rows)
    plot_family_selected(run_dir / "family_selected_postprocess.png", family_rows)

    elapsed = time.perf_counter() - started
    summary = {
        "run_name": run_name,
        "models": len(models),
        "variants": [variant["name"] for variant in variants],
        "splits": splits,
        "threshold_strategy": config["threshold_strategy"],
        "selected_counts": selected_counts,
        "total_counts": total_counts,
        "elapsed_sec": elapsed,
        "outputs": {
            "postprocess_sample_rows": str(run_dir / "postprocess_sample_rows.csv"),
            "postprocess_summary": str(run_dir / "postprocess_summary.csv"),
            "selected_postprocess_by_val": str(run_dir / "selected_postprocess_by_val.csv"),
            "postprocess_delta_vs_raw": str(run_dir / "postprocess_delta_vs_raw.csv"),
            "report": str(config["outputs"]["report"]),
            "log": str(log_path),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(Path(config["outputs"]["report"]), config, run_dir, selected_rows, delta_rows, variant_rows, family_rows)
    log(f"Finished spatial postprocess ablation in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
