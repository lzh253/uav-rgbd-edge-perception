from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
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
PROTOCOL_CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"
BASELINE_CONFIG_PATH = FINALLY_ROOT / "configs" / "baseline_training_long.json"
SEMANTIC_CONFIG_PATH = FINALLY_ROOT / "configs" / "semantic_comm_training.json"
PLANNING_CONFIG_PATH = FINALLY_ROOT / "configs" / "path_planning_eval.json"

sys.path.insert(0, str(SCRIPT_DIR))
from train_baselines import TraversabilityDataset, ensure_dir, load_json, make_model as make_baseline_model, read_ids
from train_semantic_comm import make_model as make_semantic_model


@dataclass
class PlanningProblem:
    sample_id: str
    gt_mask: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    oracle_path: list[tuple[int, int]]
    oracle_length: float
    component_pixels: int


class ModelPredictor:
    def __init__(
        self,
        model_cfg: dict[str, Any],
        sample_ids: list[str],
        protocol_config: dict[str, Any],
        baseline_config: dict[str, Any],
        semantic_config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.cfg = model_cfg
        self.name = str(model_cfg["name"])
        self.type = str(model_cfg["type"])
        self.device = device
        self.id_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        self.model: torch.nn.Module | None = None
        self.dataset: TraversabilityDataset | None = None

        if self.type == "oracle":
            return
        if self.type == "baseline":
            mode = str(model_cfg["mode"])
            self.dataset = TraversabilityDataset(sample_ids, mode, protocol_config, baseline_config, str(protocol_config.get("split", "")))
            self.model = make_baseline_model(mode, int(baseline_config["base_channels"]))
        elif self.type == "semantic":
            variant_name = str(model_cfg["variant"])
            variants = {str(v["name"]): v for v in semantic_config["variants"]}
            if variant_name not in variants:
                raise KeyError(f"Unknown semantic variant: {variant_name}")
            self.dataset = TraversabilityDataset(sample_ids, "rgbd", protocol_config, semantic_config, str(protocol_config.get("split", "")))
            self.model = make_semantic_model(semantic_config, variants[variant_name])
        else:
            raise ValueError(f"Unsupported model type: {self.type}")

        checkpoint_path = Path(str(model_cfg["checkpoint"]))
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    def predict(
        self,
        problem: PlanningProblem,
        seed: int,
    ) -> tuple[np.ndarray, float]:
        if self.type == "oracle":
            return problem.gt_mask.astype(np.float32), 0.0
        if self.dataset is None or self.model is None:
            raise RuntimeError(f"Model predictor {self.name} is not initialized")
        idx = self.id_to_index[problem.sample_id]
        x, _y, _sample_id = self.dataset[idx]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        x = x.unsqueeze(0).to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0, 0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return probs.astype(np.float32), elapsed_ms


def load_label_mask(sample_id: str, protocol_config: dict[str, Any], image_size: int) -> np.ndarray:
    dataset_root = Path(protocol_config["dataset_root"])
    label_path = dataset_root / protocol_config["label_dir"] / f"{sample_id}.png"
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(label_path)
    if label.ndim == 3:
        label = label[:, :, 0]
    label = cv2.resize(label, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    passable_ids = [int(x) for x in protocol_config["passable_class_ids"]]
    mask = np.zeros(label.shape, dtype=bool)
    for class_id in passable_ids:
        mask |= label == class_id
    return mask


def load_rgb_for_preview(sample_id: str, protocol_config: dict[str, Any], image_size: int) -> np.ndarray:
    dataset_root = Path(protocol_config["dataset_root"])
    rgb_path = dataset_root / protocol_config["rgb_dir"] / f"{sample_id}.jpg"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return rgb


def largest_component(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray | None, int]:
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    if num_labels <= 1:
        return None, 0
    component_sizes = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(component_sizes)) + 1
    best_size = int(stats[best_label, cv2.CC_STAT_AREA])
    return labels == best_label, best_size


def choose_start_goal(
    component: np.ndarray,
    min_distance: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    coords = np.argwhere(component)
    if coords.size == 0:
        return None
    height, width = component.shape
    center_x = (width - 1) / 2.0
    ys = coords[:, 0]
    bottom_cutoff = np.percentile(ys, 90)
    top_cutoff = np.percentile(ys, 10)
    bottom_pool = coords[ys >= bottom_cutoff]
    top_pool = coords[ys <= top_cutoff]
    if len(bottom_pool) == 0 or len(top_pool) == 0:
        return None
    start_arr = bottom_pool[np.argmin(np.abs(bottom_pool[:, 1] - center_x))]
    goal_arr = top_pool[np.argmin(np.abs(top_pool[:, 1] - center_x))]
    start = (int(start_arr[0]), int(start_arr[1]))
    goal = (int(goal_arr[0]), int(goal_arr[1]))
    if euclidean(start, goal) >= min_distance:
        return start, goal
    distances = np.sqrt((coords[:, 0] - start[0]) ** 2 + (coords[:, 1] - start[1]) ** 2)
    farthest = coords[int(np.argmax(distances))]
    goal = (int(farthest[0]), int(farthest[1]))
    if euclidean(start, goal) < min_distance:
        return None
    return start, goal


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def path_length(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(euclidean(path[i - 1], path[i]) for i in range(1, len(path))))


def astar(
    mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    connectivity: int,
) -> list[tuple[int, int]] | None:
    height, width = mask.shape
    if not (0 <= start[0] < height and 0 <= start[1] < width):
        return None
    if not (0 <= goal[0] < height and 0 <= goal[1] < width):
        return None
    if not mask[start] or not mask[goal]:
        return None
    if connectivity == 8:
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
    else:
        neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

    open_heap: list[tuple[float, float, tuple[int, int]]] = [(euclidean(start, goal), 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_g = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    while open_heap:
        _f, current_g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)
        cy, cx = current
        for dy, dx, step_cost in neighbors:
            ny, nx = cy + dy, cx + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width or not mask[ny, nx]:
                continue
            candidate = (ny, nx)
            tentative_g = current_g + step_cost
            if tentative_g < best_g.get(candidate, math.inf):
                came_from[candidate] = current
                best_g[candidate] = tentative_g
                heapq.heappush(open_heap, (tentative_g + euclidean(candidate, goal), tentative_g, candidate))
    return None


def binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def make_problem(
    sample_id: str,
    protocol_config: dict[str, Any],
    planning_config: dict[str, Any],
) -> tuple[PlanningProblem | None, dict[str, Any]]:
    image_size = int(planning_config["image_size"])
    connectivity = int(planning_config["connectivity"])
    gt_mask = load_label_mask(sample_id, protocol_config, image_size)
    component, component_pixels = largest_component(gt_mask, connectivity)
    selection = {
        "sample_id": sample_id,
        "selected": False,
        "component_pixels": component_pixels,
        "skip_reason": "",
    }
    if component is None or component_pixels < int(planning_config["min_gt_component_pixels"]):
        selection["skip_reason"] = "no_large_gt_component"
        return None, selection
    pair = choose_start_goal(component, float(planning_config["min_start_goal_distance"]))
    if pair is None:
        selection["skip_reason"] = "no_valid_start_goal_pair"
        return None, selection
    start, goal = pair
    oracle_path = astar(gt_mask, start, goal, connectivity)
    if oracle_path is None:
        selection["skip_reason"] = "oracle_path_not_found"
        return None, selection
    selection.update(
        {
            "selected": True,
            "start_y": start[0],
            "start_x": start[1],
            "goal_y": goal[0],
            "goal_x": goal[1],
            "oracle_length": path_length(oracle_path),
        }
    )
    return (
        PlanningProblem(
            sample_id=sample_id,
            gt_mask=gt_mask,
            start=start,
            goal=goal,
            oracle_path=oracle_path,
            oracle_length=path_length(oracle_path),
            component_pixels=component_pixels,
        ),
        selection,
    )


def evaluate_prediction(
    problem: PlanningProblem,
    prob: np.ndarray,
    threshold: float,
    connectivity: int,
) -> dict[str, Any]:
    pred_mask = prob >= threshold
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


def save_preview(
    out_path: Path,
    rgb: np.ndarray,
    problem: PlanningProblem,
    pred_mask: np.ndarray,
    prob: np.ndarray,
    path: list[tuple[int, int]] | None,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB + path", fontsize=9)
    axes[1].imshow(problem.gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("ground truth", fontsize=9)
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("predicted road", fontsize=9)
    axes[3].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title(title, fontsize=9)
    if path is not None:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        axes[0].plot(xs, ys, color="#f97316", linewidth=2.2)
        axes[2].plot(xs, ys, color="#f97316", linewidth=1.8)
    for ax in (axes[0], axes[2]):
        ax.scatter([problem.start[1]], [problem.start[0]], s=30, color="#22c55e")
        ax.scatter([problem.goal[1]], [problem.goal[0]], s=30, color="#ef4444")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def mean(values: list[float]) -> float:
    clean = [float(x) for x in values if not math.isnan(float(x))]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def summarize_model(
    model_name: str,
    rows: list[dict[str, Any]],
    selected_samples: int,
    total_samples: int,
) -> dict[str, Any]:
    model_rows = [row for row in rows if row["model"] == model_name]
    found_rows = [row for row in model_rows if row["path_found"]]
    success_rows = [row for row in model_rows if row["success"]]
    blocked_rows = [row for row in model_rows if not row["start_on_pred"] or not row["goal_on_pred"]]
    collision_found = [row for row in found_rows if int(row["collision_cells"]) > 0]
    denom = max(1, len(model_rows))
    return {
        "model": model_name,
        "total_split_samples": total_samples,
        "selected_evaluable_samples": selected_samples,
        "evaluated_samples": len(model_rows),
        "path_found_rate": len(found_rows) / denom,
        "success_rate_collision_free": len(success_rows) / denom,
        "start_or_goal_blocked_rate": len(blocked_rows) / denom,
        "collision_rate_among_found": len(collision_found) / max(1, len(found_rows)),
        "mean_collision_fraction_found": mean([float(row["collision_fraction"]) for row in found_rows]),
        "mean_path_length_success": mean([float(row["path_length"]) for row in success_rows]),
        "mean_path_length_ratio_success": mean([float(row["path_length_ratio"]) for row in success_rows]),
        "mean_map_iou": mean([float(row["map_iou"]) for row in model_rows]),
        "mean_inference_time_ms": mean([float(row["inference_time_ms"]) for row in model_rows]),
        "mean_planning_time_ms": mean([float(row["planning_time_ms"]) for row in model_rows]),
    }


def plot_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    labels = [row["model"] for row in summary_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#64748b", "#2563eb", "#0f766e", "#16a34a", "#eab308", "#f97316", "#dc2626"]
    axes[0].bar(x, [row["success_rate_collision_free"] for row in summary_rows], color=colors[: len(labels)])
    axes[0].set_title("Collision-free success rate")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, [row["path_found_rate"] for row in summary_rows], color=colors[: len(labels)])
    axes[1].set_title("Path found rate")
    axes[1].set_ylim(0, 1)
    axes[2].bar(x, [row["mean_map_iou"] for row in summary_rows], color=colors[: len(labels)])
    axes[2].set_title("Mean mask IoU on selected samples")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained traversability models with A* path planning.")
    parser.add_argument("--config", type=Path, default=PLANNING_CONFIG_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planning_config = load_json(args.config)
    protocol_config = load_json(PROTOCOL_CONFIG_PATH)
    baseline_config = load_json(BASELINE_CONFIG_PATH)
    semantic_config = load_json(SEMANTIC_CONFIG_PATH)
    run_name = str(planning_config["run_name"])
    output_root = Path(planning_config["outputs"]["root"])
    run_dir = output_root / run_name
    previews_dir = run_dir / "previews"
    ensure_dir(run_dir)
    ensure_dir(previews_dir)
    (run_dir / "config_snapshot.json").write_text(json.dumps(planning_config, indent=2), encoding="utf-8")
    (run_dir / "protocol_config_snapshot.json").write_text(json.dumps(protocol_config, indent=2), encoding="utf-8")

    split = str(planning_config["split"])
    outputs_root = Path(protocol_config["outputs_root"])
    sample_ids = read_ids(outputs_root / "protocol" / f"{split}.txt")
    max_samples = int(args.max_samples if args.max_samples >= 0 else planning_config.get("max_samples", 0))
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]
    image_size = int(planning_config["image_size"])
    threshold = float(planning_config["prediction_threshold"])
    connectivity = int(planning_config["connectivity"])
    seed = int(planning_config["seed"])
    save_previews_per_model = int(planning_config["save_previews_per_model"])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Planning evaluation run={run_name} split={split} samples={len(sample_ids)} device={device}")

    selection_rows: list[dict[str, Any]] = []
    problems: list[PlanningProblem] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        problem, selection = make_problem(sample_id, protocol_config, planning_config)
        selection_rows.append(selection)
        if problem is not None:
            problems.append(problem)
        if index % 100 == 0 or index == len(sample_ids):
            print(f"Prepared planning problems {index}/{len(sample_ids)} selected={len(problems)}")
            write_progress(run_dir / "progress.json", {"stage": "problem_selection", "processed": index, "total": len(sample_ids), "selected": len(problems)})
    write_csv(run_dir / "sample_selection.csv", selection_rows)

    predictors = [
        ModelPredictor(model_cfg, [p.sample_id for p in problems], protocol_config, baseline_config, semantic_config, device)
        for model_cfg in planning_config["models"]
    ]

    rows: list[dict[str, Any]] = []
    preview_counts = {predictor.name: 0 for predictor in predictors}
    for model_index, predictor in enumerate(predictors):
        print(f"Evaluating {predictor.name} ({model_index + 1}/{len(predictors)})")
        for problem_index, problem in enumerate(problems, start=1):
            forward_seed = seed + model_index * 100000 + problem_index
            prob, inference_time_ms = predictor.predict(problem, forward_seed)
            result = evaluate_prediction(problem, prob, threshold, connectivity)
            rows.append(
                {
                    "model": predictor.name,
                    "model_type": predictor.type,
                    "sample_id": problem.sample_id,
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
                    "threshold": threshold,
                }
            )
            if preview_counts[predictor.name] < save_previews_per_model:
                rgb = load_rgb_for_preview(problem.sample_id, protocol_config, image_size)
                preview_path = previews_dir / f"{predictor.name}_{problem.sample_id}_path.png"
                save_preview(preview_path, rgb, problem, result["pred_mask"], prob, result["path"], predictor.name)
                preview_counts[predictor.name] += 1
            if problem_index % 50 == 0 or problem_index == len(problems):
                write_progress(
                    run_dir / "progress.json",
                    {
                        "stage": "model_evaluation",
                        "model": predictor.name,
                        "model_index": model_index + 1,
                        "models_total": len(predictors),
                        "processed": problem_index,
                        "total": len(problems),
                    },
                )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = [
        summarize_model(str(model_cfg["name"]), rows, len(problems), len(sample_ids))
        for model_cfg in planning_config["models"]
    ]
    write_csv(run_dir / "planning_rows.csv", rows)
    write_csv(run_dir / "planning_summary.csv", summary_rows)
    plot_summary(run_dir / "path_planning_comparison.png", summary_rows)
    summary = {
        "run_name": run_name,
        "split": split,
        "total_split_samples": len(sample_ids),
        "selected_evaluable_samples": len(problems),
        "skipped_samples": len(sample_ids) - len(problems),
        "device": str(device),
        "threshold": threshold,
        "summary_rows": summary_rows,
        "outputs": {
            "sample_selection_csv": str(run_dir / "sample_selection.csv"),
            "planning_rows_csv": str(run_dir / "planning_rows.csv"),
            "planning_summary_csv": str(run_dir / "planning_summary.csv"),
            "comparison_plot": str(run_dir / "path_planning_comparison.png"),
            "previews_dir": str(previews_dir),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_progress(run_dir / "progress.json", {"stage": "done", "selected": len(problems), "total": len(sample_ids)})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
