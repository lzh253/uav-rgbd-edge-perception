from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
PROTOCOL_CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"
BASELINE_CONFIG_PATH = FINALLY_ROOT / "configs" / "baseline_training_long.json"
SEMANTIC_CONFIG_PATH = FINALLY_ROOT / "configs" / "semantic_comm_training.json"
SWEEP_CONFIG_PATH = FINALLY_ROOT / "configs" / "path_planning_threshold_sweep.json"

sys.path.insert(0, str(SCRIPT_DIR))
from train_baselines import ensure_dir, load_json, read_ids
from evaluate_path_planning import (
    ModelPredictor,
    evaluate_prediction,
    make_problem,
    mean,
    plot_summary,
    summarize_model,
    write_csv,
)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def select_rows(rows: list[dict[str, Any]], split: str, model: str, threshold: float) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["split"] == split and row["model"] == model and abs(float(row["threshold"]) - threshold) < 1e-9
    ]


def summarize_threshold(
    split: str,
    model_name: str,
    threshold: float,
    rows: list[dict[str, Any]],
    selected_samples: int,
    total_samples: int,
) -> dict[str, Any]:
    scoped_rows = select_rows(rows, split, model_name, threshold)
    summary = summarize_model(model_name, scoped_rows, selected_samples, total_samples)
    summary["split"] = split
    summary["threshold"] = threshold
    path_found_rate = float(summary["path_found_rate"])
    success_rate = float(summary["success_rate_collision_free"])
    summary["unsafe_found_rate"] = path_found_rate - success_rate
    summary["planning_safety_utility"] = success_rate - summary["unsafe_found_rate"]
    return summary


def select_best_thresholds(
    summary_rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    selection_split: str,
    final_split: str,
    thresholds: list[float],
    selection_metric: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    best_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    summary_index = {
        (str(row["split"]), str(row["model"]), float(row["threshold"])): row
        for row in summary_rows
    }
    default_threshold = 0.5 if 0.5 in thresholds else thresholds[len(thresholds) // 2]

    for model in models:
        model_name = str(model["name"])
        if str(model["type"]) == "oracle":
            selected_threshold = default_threshold
            selection_reason = "oracle_reference_fixed_threshold"
        else:
            candidates = [
                row
                for row in summary_rows
                if row["split"] == selection_split and row["model"] == model_name
            ]
            if not candidates:
                raise RuntimeError(f"No validation summary rows for model={model_name}")

            def score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
                return (
                    safe_float(row.get(selection_metric), -1.0),
                    safe_float(row.get("path_found_rate"), -1.0),
                    safe_float(row.get("mean_map_iou"), -1.0),
                    -safe_float(row.get("collision_rate_among_found"), 1.0),
                    -abs(float(row["threshold"]) - default_threshold),
                )

            best = max(candidates, key=score)
            selected_threshold = float(best["threshold"])
            selection_reason = f"max_{selection_metric}_on_{selection_split}"

        val_row = summary_index[(selection_split, model_name, selected_threshold)]
        test_row = summary_index[(final_split, model_name, selected_threshold)]
        test_default = summary_index.get((final_split, model_name, default_threshold), test_row)
        best_rows.append(
            {
                "model": model_name,
                "model_type": model["type"],
                "selected_threshold": selected_threshold,
                "selection_reason": selection_reason,
                "val_success_rate_collision_free": val_row["success_rate_collision_free"],
                "val_path_found_rate": val_row["path_found_rate"],
                "val_start_or_goal_blocked_rate": val_row["start_or_goal_blocked_rate"],
                "val_collision_rate_among_found": val_row["collision_rate_among_found"],
                "val_mean_map_iou": val_row["mean_map_iou"],
                "test_success_rate_collision_free": test_row["success_rate_collision_free"],
                "test_path_found_rate": test_row["path_found_rate"],
                "test_start_or_goal_blocked_rate": test_row["start_or_goal_blocked_rate"],
                "test_collision_rate_among_found": test_row["collision_rate_among_found"],
                "test_mean_map_iou": test_row["mean_map_iou"],
                "test_success_gain_vs_threshold_0_50": safe_float(test_row["success_rate_collision_free"]) - safe_float(test_default["success_rate_collision_free"]),
                "test_path_found_gain_vs_threshold_0_50": safe_float(test_row["path_found_rate"]) - safe_float(test_default["path_found_rate"]),
            }
        )
        final_copy = dict(test_row)
        final_copy["selected_by"] = selection_split
        final_rows.append(final_copy)
    return best_rows, final_rows


def plot_threshold_success_curves(path: Path, summary_rows: list[dict[str, Any]], thresholds: list[float], models: list[str], splits: list[str]) -> None:
    fig, axes = plt.subplots(1, len(splits), figsize=(6.4 * len(splits), 4.2), sharey=True)
    if len(splits) == 1:
        axes = [axes]
    colors = plt.get_cmap("tab10").colors
    for ax, split in zip(axes, splits):
        for idx, model in enumerate(models):
            if model == "oracle_gt":
                continue
            y = []
            for threshold in thresholds:
                row = next(
                    item
                    for item in summary_rows
                    if item["split"] == split and item["model"] == model and abs(float(item["threshold"]) - threshold) < 1e-9
                )
                y.append(float(row["success_rate_collision_free"]))
            ax.plot(thresholds, y, marker="o", linewidth=1.8, label=model, color=colors[idx % len(colors)])
        ax.set_title(f"{split}: collision-free success")
        ax.set_xlabel("threshold")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("rate")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_threshold_tradeoff(path: Path, summary_rows: list[dict[str, Any]], thresholds: list[float], models: list[str], split: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    colors = plt.get_cmap("tab10").colors
    metrics = [
        ("path_found_rate", "path found"),
        ("success_rate_collision_free", "collision-free success"),
        ("start_or_goal_blocked_rate", "start/goal blocked"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for idx, model in enumerate(models):
            if model == "oracle_gt":
                continue
            y = []
            for threshold in thresholds:
                row = next(
                    item
                    for item in summary_rows
                    if item["split"] == split and item["model"] == model and abs(float(item["threshold"]) - threshold) < 1e-9
                )
                y.append(float(row[metric]))
            ax.plot(thresholds, y, marker="o", linewidth=1.6, label=model, color=colors[idx % len(colors)])
        ax.set_title(f"{split}: {title}")
        ax.set_xlabel("threshold")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("rate")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_selected_thresholds(path: Path, best_rows: list[dict[str, Any]]) -> None:
    labels = [row["model"] for row in best_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metrics = [
        ("test_success_rate_collision_free", "test collision-free success"),
        ("test_path_found_rate", "test path found"),
        ("test_mean_map_iou", "test mask IoU"),
    ]
    colors = ["#64748b", "#2563eb", "#0f766e", "#16a34a", "#eab308", "#f97316", "#dc2626"]
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(x, [float(row[metric]) for row in best_rows], color=colors[: len(labels)])
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation-driven threshold sweep for A* path planning.")
    parser.add_argument("--config", type=Path, default=SWEEP_CONFIG_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_config = load_json(args.config)
    protocol_config = load_json(PROTOCOL_CONFIG_PATH)
    baseline_config = load_json(BASELINE_CONFIG_PATH)
    semantic_config = load_json(SEMANTIC_CONFIG_PATH)
    run_name = str(sweep_config["run_name"])
    output_root = Path(sweep_config["outputs"]["root"])
    run_dir = output_root / run_name
    ensure_dir(run_dir)
    (run_dir / "config_snapshot.json").write_text(json.dumps(sweep_config, indent=2), encoding="utf-8")
    (run_dir / "protocol_config_snapshot.json").write_text(json.dumps(protocol_config, indent=2), encoding="utf-8")

    thresholds = [float(x) for x in sweep_config["thresholds"]]
    splits = [str(x) for x in sweep_config["splits"]]
    selection_split = str(sweep_config["selection_split"])
    final_split = str(sweep_config["final_split"])
    selection_metric = str(sweep_config["selection_metric"])
    safety_selection_metric = str(sweep_config.get("safety_selection_metric", "planning_safety_utility"))
    seed = int(sweep_config["seed"])
    connectivity = int(sweep_config["connectivity"])
    max_samples = int(args.max_samples if args.max_samples >= 0 else sweep_config.get("max_samples_per_split", 0))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print(f"Threshold sweep run={run_name} splits={splits} thresholds={thresholds} device={device}")
    all_rows: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    inference_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for split_index, split in enumerate(splits):
        outputs_root = Path(protocol_config["outputs_root"])
        sample_ids = read_ids(outputs_root / "protocol" / f"{split}.txt")
        if max_samples > 0:
            sample_ids = sample_ids[:max_samples]
        total_counts[split] = len(sample_ids)
        problems = []
        selection_rows = []
        for index, sample_id in enumerate(sample_ids, start=1):
            problem, selection = make_problem(sample_id, protocol_config, sweep_config)
            selection["split"] = split
            selection_rows.append(selection)
            if problem is not None:
                problems.append(problem)
            if index % 100 == 0 or index == len(sample_ids):
                print(f"{split}: selected {len(problems)}/{index}")
                write_progress(run_dir / "progress.json", {"stage": "problem_selection", "split": split, "processed": index, "total": len(sample_ids), "selected": len(problems)})
        selected_counts[split] = len(problems)
        all_selection_rows.extend(selection_rows)
        write_csv(run_dir / f"sample_selection_{split}.csv", selection_rows)

        predictor_sample_ids = [problem.sample_id for problem in problems]
        predictors = [
            ModelPredictor(model_cfg, predictor_sample_ids, protocol_config, baseline_config, semantic_config, device)
            for model_cfg in sweep_config["models"]
        ]
        for model_index, predictor in enumerate(predictors):
            print(f"{split}: evaluating {predictor.name} ({model_index + 1}/{len(predictors)})")
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
                    all_rows.append(
                        {
                            "split": split,
                            "model": predictor.name,
                            "model_type": predictor.type,
                            "sample_id": problem.sample_id,
                            "threshold": threshold,
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
                            "map_iou": result["map_iou"],
                            "inference_time_ms": inference_time_ms,
                            "planning_time_ms": result["planning_time_ms"],
                        }
                    )
                if problem_index % 50 == 0 or problem_index == len(problems):
                    write_progress(
                        run_dir / "progress.json",
                        {
                            "stage": "threshold_evaluation",
                            "split": split,
                            "model": predictor.name,
                            "model_index": model_index + 1,
                            "models_total": len(predictors),
                            "processed": problem_index,
                            "total": len(problems),
                        },
                    )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    model_names = [str(model["name"]) for model in sweep_config["models"]]
    summary_rows = [
        summarize_threshold(split, model_name, threshold, all_rows, selected_counts[split], total_counts[split])
        for split in splits
        for model_name in model_names
        for threshold in thresholds
    ]
    best_success_rows, selected_success_test_rows = select_best_thresholds(
        summary_rows,
        sweep_config["models"],
        selection_split,
        final_split,
        thresholds,
        selection_metric,
    )
    best_safety_rows, selected_safety_test_rows = select_best_thresholds(
        summary_rows,
        sweep_config["models"],
        selection_split,
        final_split,
        thresholds,
        safety_selection_metric,
    )
    write_csv(run_dir / "sample_selection_all.csv", all_selection_rows)
    write_csv(run_dir / "inference_manifest.csv", inference_rows)
    write_csv(run_dir / "threshold_rows.csv", all_rows)
    write_csv(run_dir / "threshold_summary.csv", summary_rows)
    write_csv(run_dir / "best_thresholds_by_val.csv", best_success_rows)
    write_csv(run_dir / "selected_threshold_test_summary.csv", selected_success_test_rows)
    write_csv(run_dir / "best_safety_thresholds_by_val.csv", best_safety_rows)
    write_csv(run_dir / "selected_safety_threshold_test_summary.csv", selected_safety_test_rows)
    plot_threshold_success_curves(run_dir / "threshold_success_curves.png", summary_rows, thresholds, model_names, splits)
    plot_threshold_tradeoff(run_dir / "threshold_tradeoff_test.png", summary_rows, thresholds, model_names, final_split)
    plot_selected_thresholds(run_dir / "selected_threshold_test_comparison.png", best_success_rows)
    plot_summary(run_dir / "selected_threshold_test_planning_comparison.png", selected_success_test_rows)
    plot_selected_thresholds(run_dir / "selected_safety_threshold_test_comparison.png", best_safety_rows)
    plot_summary(run_dir / "selected_safety_threshold_test_planning_comparison.png", selected_safety_test_rows)

    elapsed_sec = time.perf_counter() - started
    summary = {
        "run_name": run_name,
        "splits": splits,
        "thresholds": thresholds,
        "selection_split": selection_split,
        "final_split": final_split,
        "selection_metric": selection_metric,
        "safety_selection_metric": safety_selection_metric,
        "selected_counts": selected_counts,
        "total_counts": total_counts,
        "device": str(device),
        "elapsed_sec": elapsed_sec,
        "best_thresholds_by_val": best_success_rows,
        "best_safety_thresholds_by_val": best_safety_rows,
        "outputs": {
            "threshold_rows_csv": str(run_dir / "threshold_rows.csv"),
            "threshold_summary_csv": str(run_dir / "threshold_summary.csv"),
            "best_thresholds_csv": str(run_dir / "best_thresholds_by_val.csv"),
            "selected_threshold_test_summary_csv": str(run_dir / "selected_threshold_test_summary.csv"),
            "best_safety_thresholds_csv": str(run_dir / "best_safety_thresholds_by_val.csv"),
            "selected_safety_threshold_test_summary_csv": str(run_dir / "selected_safety_threshold_test_summary.csv"),
            "threshold_success_curves": str(run_dir / "threshold_success_curves.png"),
            "threshold_tradeoff_test": str(run_dir / "threshold_tradeoff_test.png"),
            "selected_threshold_test_comparison": str(run_dir / "selected_threshold_test_comparison.png"),
            "selected_safety_threshold_test_comparison": str(run_dir / "selected_safety_threshold_test_comparison.png"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_progress(run_dir / "progress.json", {"stage": "done", "elapsed_sec": elapsed_sec})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
