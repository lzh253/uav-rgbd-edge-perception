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
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = FINALLY_ROOT / "configs" / "planner_robustness.json"

sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_path_planning import (  # noqa: E402
    BASELINE_CONFIG_PATH,
    PROTOCOL_CONFIG_PATH,
    SEMANTIC_CONFIG_PATH,
    STRONG_BASELINE_CONFIG_PATH,
    ModelPredictor,
    PlanningProblem,
    binary_iou,
    choose_start_goal,
    largest_component,
    load_json,
    load_label_mask,
    path_length,
    read_ids,
    write_csv,
)


@dataclass
class RobustProblem:
    sample_id: str
    gt_mask: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    oracle_path: list[tuple[int, int]]
    oracle_length: float
    component_pixels: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def neighbors_for(connectivity: int) -> list[tuple[int, int, float]]:
    if connectivity == 8:
        return [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
    return [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def graph_search(
    mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    connectivity: int,
    planner: str,
) -> list[tuple[int, int]] | None:
    height, width = mask.shape
    if not (0 <= start[0] < height and 0 <= start[1] < width):
        return None
    if not (0 <= goal[0] < height and 0 <= goal[1] < width):
        return None
    if not mask[start] or not mask[goal]:
        return None

    use_heuristic = planner == "astar"
    open_heap: list[tuple[float, float, tuple[int, int]]] = [(euclidean(start, goal) if use_heuristic else 0.0, 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_g = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    neighbors = neighbors_for(connectivity)

    while open_heap:
        _priority, current_g, current = heapq.heappop(open_heap)
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
                heuristic = euclidean(candidate, goal) if use_heuristic else 0.0
                heapq.heappush(open_heap, (tentative_g + heuristic, tentative_g, candidate))
    return None


def make_problem_for_setting(
    sample_id: str,
    protocol_config: dict[str, Any],
    setting: dict[str, Any],
    image_size: int,
) -> tuple[RobustProblem | None, dict[str, Any]]:
    connectivity = int(setting["connectivity"])
    radius = int(setting.get("footprint_radius", 0))
    planner = str(setting["planner"])
    gt_mask = erode_mask(load_label_mask(sample_id, protocol_config, image_size), radius)
    component, component_pixels = largest_component(gt_mask, connectivity)
    selection = {
        "setting": setting["name"],
        "sample_id": sample_id,
        "selected": False,
        "component_pixels": component_pixels,
        "skip_reason": "",
    }
    if component is None or component_pixels < int(setting["min_gt_component_pixels"]):
        selection["skip_reason"] = "no_large_gt_component"
        return None, selection
    pair = choose_start_goal(component, float(setting["min_start_goal_distance"]))
    if pair is None:
        selection["skip_reason"] = "no_valid_start_goal_pair"
        return None, selection
    start, goal = pair
    oracle_path = graph_search(gt_mask, start, goal, connectivity, planner)
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
        RobustProblem(
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


def evaluate_prediction_robust(
    problem: RobustProblem,
    prob: np.ndarray,
    threshold: float,
    setting: dict[str, Any],
) -> dict[str, Any]:
    connectivity = int(setting["connectivity"])
    radius = int(setting.get("footprint_radius", 0))
    planner = str(setting["planner"])
    pred_mask = erode_mask(prob >= threshold, radius)
    started = time.perf_counter()
    path = graph_search(pred_mask, problem.start, problem.goal, connectivity, planner)
    planning_time_ms = (time.perf_counter() - started) * 1000.0
    found = path is not None
    collision_cells = 0
    collision_fraction = math.nan
    length = math.nan
    length_ratio = math.nan
    if path is not None:
        collision_cells = int(sum(not bool(problem.gt_mask[y, x]) for y, x in path))
        collision_fraction = collision_cells / max(1, len(path))
        length = path_length(path)
        length_ratio = length / problem.oracle_length if problem.oracle_length > 0 else math.nan
    return {
        "start_on_pred": bool(pred_mask[problem.start]),
        "goal_on_pred": bool(pred_mask[problem.goal]),
        "path_found": found,
        "success": bool(found and collision_cells == 0),
        "collision_cells": collision_cells,
        "collision_fraction": collision_fraction,
        "path_length": length,
        "path_length_ratio": length_ratio,
        "map_iou": binary_iou(pred_mask, problem.gt_mask),
        "planning_time_ms": planning_time_ms,
    }


def mean(values: list[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def summarize(setting_name: str, model_name: str, rows: list[dict[str, Any]], selected_samples: int, total_samples: int) -> dict[str, Any]:
    scoped = [row for row in rows if row["setting"] == setting_name and row["model"] == model_name]
    found = [row for row in scoped if row["path_found"]]
    success = [row for row in scoped if row["success"]]
    blocked = [row for row in scoped if not row["start_on_pred"] or not row["goal_on_pred"]]
    collision_found = [row for row in found if int(row["collision_cells"]) > 0]
    denom = max(1, len(scoped))
    found_rate = len(found) / denom
    success_rate = len(success) / denom
    unsafe_found_rate = found_rate - success_rate
    return {
        "setting": setting_name,
        "model": model_name,
        "total_split_samples": total_samples,
        "selected_evaluable_samples": selected_samples,
        "evaluated_samples": len(scoped),
        "path_found_rate": found_rate,
        "safe_success_rate": success_rate,
        "unsafe_found_rate": unsafe_found_rate,
        "collision_rate_among_found": len(collision_found) / max(1, len(found)),
        "path_safety_utility": success_rate - unsafe_found_rate,
        "start_or_goal_blocked_rate": len(blocked) / denom,
        "mean_collision_fraction_found": mean([row["collision_fraction"] for row in found]),
        "mean_path_length_ratio_success": mean([row["path_length_ratio"] for row in success]),
        "mean_map_iou": mean([row["map_iou"] for row in scoped]),
        "mean_inference_time_ms": mean([row["inference_time_ms"] for row in scoped]),
        "mean_planning_time_ms": mean([row["planning_time_ms"] for row in scoped]),
    }


def build_delta_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_setting: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        by_setting.setdefault(row["setting"], {})[row["model"]] = row
    out: list[dict[str, Any]] = []
    for setting, rows in by_setting.items():
        semantic = rows.get("semantic_clean")
        rgb = rows.get("rgb_baseline")
        rgbd = rows.get("rgbd_concat_baseline")
        if semantic is None:
            continue
        out.append(
            {
                "setting": setting,
                "semantic_safety": semantic["path_safety_utility"],
                "rgb_safety": rgb["path_safety_utility"] if rgb else math.nan,
                "rgbd_safety": rgbd["path_safety_utility"] if rgbd else math.nan,
                "semantic_minus_rgb_safety": semantic["path_safety_utility"] - (rgb["path_safety_utility"] if rgb else math.nan),
                "semantic_minus_rgbd_safety": semantic["path_safety_utility"] - (rgbd["path_safety_utility"] if rgbd else math.nan),
                "semantic_safe_success": semantic["safe_success_rate"],
                "semantic_collision_found": semantic["collision_rate_among_found"],
                "semantic_map_iou": semantic["mean_map_iou"],
            }
        )
    return out


def write_report(report_path: Path, config: dict[str, Any], summary_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> None:
    by_setting = {row["setting"]: row for row in delta_rows}
    lines = [
        "# Planner Robustness Experiment",
        "",
        "This report summarizes the planner-side robustness checks.",
        "The experiment reuses the validation-selected checkpoints and thresholds; it does not introduce a new training protocol.",
        "",
        "## Tested Perturbations",
        "",
    ]
    for setting in config["settings"]:
        lines.append(
            f"- `{setting['name']}`: planner={setting['planner']}, connectivity={setting['connectivity']}, "
            f"footprint_radius={setting['footprint_radius']}, min_start_goal_distance={setting['min_start_goal_distance']}."
        )
    lines.extend(["", "## Semantic Safety Deltas", ""])
    for setting in config["settings"]:
        row = by_setting.get(setting["name"])
        if not row:
            continue
        lines.append(
            f"- `{setting['name']}`: semantic safety={row['semantic_safety']:.4f}, "
            f"semantic-RGB={row['semantic_minus_rgb_safety']:.4f}, "
            f"semantic-RGB-D concat={row['semantic_minus_rgbd_safety']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation for the Manuscript",
            "",
            "These results should be used as a robustness check, not as a deployment claim. The settings probe whether the model ranking is stable when the graph search and grid clearance assumptions are changed. They still do not model UGV dynamics, control, slip, localization, or real wireless delay.",
            "",
            "## Output Files",
            "",
            f"- `{Path(config['outputs']['root']) / config['run_name'] / 'planner_robustness_summary.csv'}`",
            f"- `{Path(config['outputs']['root']) / config['run_name'] / 'planner_robustness_delta_vs_nominal.csv'}`",
            f"- `{Path(config['outputs']['root']) / config['run_name'] / 'planner_robustness_rows.csv'}`",
            f"- `{Path(config['outputs']['root']) / config['run_name'] / 'sample_selection_by_setting.csv'}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planner robustness checks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    protocol_config = load_json(PROTOCOL_CONFIG_PATH)
    baseline_config = load_json(BASELINE_CONFIG_PATH)
    semantic_config = load_json(SEMANTIC_CONFIG_PATH)
    strong_config = load_json(STRONG_BASELINE_CONFIG_PATH)
    source_model_config = load_json(Path(config["source_model_config"]))
    threshold_csv = str(config.get("threshold_csv", "")).strip()
    thresholds = {}
    if threshold_csv:
        thresholds = {row["model"]: float(row["threshold"]) for row in read_csv(Path(threshold_csv))}
    model_names = set(str(name) for name in config["model_names"])
    model_cfgs = [model for model in source_model_config["models"] if str(model["name"]) in model_names]

    run_dir = Path(config["outputs"]["root"]) / str(config["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "source_model_config_snapshot.json").write_text(json.dumps(source_model_config, indent=2), encoding="utf-8")

    split = str(config["split"])
    image_size = int(config["image_size"])
    outputs_root = Path(protocol_config["outputs_root"])
    sample_ids = read_ids(outputs_root / "protocol" / f"{split}.txt")
    max_samples = int(args.max_samples if args.max_samples >= 0 else config.get("max_samples", 0))
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Planner robustness run={config['run_name']} split={split} samples={len(sample_ids)} device={device}")

    all_rows: list[dict[str, Any]] = []
    all_selection: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for setting_index, setting in enumerate(config["settings"], start=1):
        print(f"Preparing setting {setting['name']} ({setting_index}/{len(config['settings'])})")
        problems: list[RobustProblem] = []
        selection_rows: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            problem, selection = make_problem_for_setting(sample_id, protocol_config, setting, image_size)
            selection_rows.append(selection)
            if problem is not None:
                problems.append(problem)
        all_selection.extend(selection_rows)

        planning_problems = [
            PlanningProblem(
                sample_id=problem.sample_id,
                gt_mask=problem.gt_mask,
                start=problem.start,
                goal=problem.goal,
                oracle_path=problem.oracle_path,
                oracle_length=problem.oracle_length,
                component_pixels=problem.component_pixels,
            )
            for problem in problems
        ]
        predictors = [
            ModelPredictor(model_cfg, [p.sample_id for p in planning_problems], protocol_config, baseline_config, semantic_config, strong_config, device)
            for model_cfg in model_cfgs
        ]
        problem_by_id = {problem.sample_id: problem for problem in problems}
        for model_index, predictor in enumerate(predictors):
            print(f"  Evaluating {setting['name']} / {predictor.name}")
            threshold = thresholds.get(predictor.name, 0.5)
            for problem_index, planning_problem in enumerate(planning_problems, start=1):
                robust_problem = problem_by_id[planning_problem.sample_id]
                prob, inference_time_ms = predictor.predict(planning_problem, int(config["seed"]) + setting_index * 1000000 + model_index * 100000 + problem_index)
                result = evaluate_prediction_robust(robust_problem, prob, threshold, setting)
                all_rows.append(
                    {
                        "setting": setting["name"],
                        "planner": setting["planner"],
                        "connectivity": setting["connectivity"],
                        "footprint_radius": setting["footprint_radius"],
                        "min_start_goal_distance": setting["min_start_goal_distance"],
                        "model": predictor.name,
                        "sample_id": robust_problem.sample_id,
                        "threshold": threshold,
                        "component_pixels": robust_problem.component_pixels,
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
        for model_cfg in model_cfgs:
            summary_rows.append(summarize(str(setting["name"]), str(model_cfg["name"]), all_rows, len(problems), len(sample_ids)))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    delta_rows = build_delta_rows(summary_rows)
    write_csv(run_dir / "sample_selection_by_setting.csv", all_selection)
    write_csv(run_dir / "planner_robustness_rows.csv", all_rows)
    write_csv(run_dir / "planner_robustness_summary.csv", summary_rows)
    write_csv(run_dir / "planner_robustness_delta_vs_nominal.csv", delta_rows)
    write_report(Path(config["outputs"]["report"]), config, summary_rows, delta_rows)
    payload = {
        "run_name": config["run_name"],
        "split": split,
        "settings": [setting["name"] for setting in config["settings"]],
        "models": [model["name"] for model in model_cfgs],
        "outputs": {
            "run_dir": str(run_dir),
            "summary_csv": str(run_dir / "planner_robustness_summary.csv"),
            "delta_csv": str(run_dir / "planner_robustness_delta_vs_nominal.csv"),
            "report": str(config["outputs"]["report"]),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
