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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
PROTOCOL_CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"
BASELINE_CONFIG_PATH = FINALLY_ROOT / "configs" / "baseline_training_long.json"
SEMANTIC_CONFIG_PATH = FINALLY_ROOT / "configs" / "semantic_comm_training.json"
DEFAULT_CONFIG = FINALLY_ROOT / "configs" / "probability_calibration_v1.json"

sys.path.insert(0, str(SCRIPT_DIR))
from train_baselines import TraversabilityDataset, ensure_dir, load_json, make_model as make_baseline_model, read_ids
from train_semantic_comm import make_model as make_semantic_model


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def mean(values: list[float]) -> float:
    clean = [x for x in values if not math.isnan(x)]
    if not clean:
        return math.nan
    return float(sum(clean) / len(clean))


def std(values: list[float]) -> float:
    clean = [x for x in values if not math.isnan(x)]
    if len(clean) <= 1:
        return 0.0
    mu = mean(clean)
    return float(math.sqrt(sum((x - mu) ** 2 for x in clean) / (len(clean) - 1)))


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) <= 1:
        return math.nan
    x_arr = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y_arr = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(x_arr.std()) == 0.0 or float(y_arr.std()) == 0.0:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def make_model(model_cfg: dict[str, Any], baseline_config: dict[str, Any], semantic_config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if model_cfg["type"] == "baseline":
        model = make_baseline_model(str(model_cfg["mode"]), int(baseline_config["base_channels"]))
    elif model_cfg["type"] == "semantic":
        variant_name = str(model_cfg.get("variant", "clean"))
        variants = {str(v["name"]): v for v in semantic_config["variants"]}
        model = make_semantic_model(semantic_config, variants[variant_name])
    else:
        raise ValueError(f"Unsupported model type for calibration: {model_cfg['type']}")
    checkpoint = torch.load(Path(str(model_cfg["checkpoint"])), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_dataset(model_cfg: dict[str, Any], sample_ids: list[str], protocol_config: dict[str, Any], baseline_config: dict[str, Any], semantic_config: dict[str, Any], split: str) -> TraversabilityDataset:
    if model_cfg["type"] == "baseline":
        return TraversabilityDataset(sample_ids, str(model_cfg["mode"]), protocol_config, baseline_config, split)
    if model_cfg["type"] == "semantic":
        return TraversabilityDataset(sample_ids, "rgbd", protocol_config, semantic_config, split)
    raise ValueError(f"Unsupported model type for calibration: {model_cfg['type']}")


def threshold_metrics_from_counts(tp: float, fp: float, tn: float, fn: float) -> dict[str, float]:
    eps = 1e-12
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    specificity = tn / max(tn + fp, eps)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, eps)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    iou = tp / max(tp + fp + fn, eps)
    pred_positive_rate = (tp + fp) / max(tp + fp + tn + fn, eps)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "iou": float(iou),
        "pred_positive_rate": float(pred_positive_rate),
    }


def auc_from_score_hist(pos_hist: np.ndarray, total_hist: np.ndarray) -> tuple[float, float]:
    neg_hist = total_hist - pos_hist
    total_pos = float(pos_hist.sum())
    total_neg = float(neg_hist.sum())
    if total_pos <= 0 or total_neg <= 0:
        return math.nan, math.nan

    pos_desc = pos_hist[::-1].astype(np.float64)
    neg_desc = neg_hist[::-1].astype(np.float64)
    tp = np.cumsum(pos_desc)
    fp = np.cumsum(neg_desc)
    recall = tp / total_pos
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    auprc = float(np.sum((recall - recall_prev) * precision))

    tpr = np.concatenate([[0.0], recall, [1.0]])
    fpr = np.concatenate([[0.0], fp / total_neg, [1.0]])
    auroc = float(np.trapz(tpr, fpr))
    return auprc, auroc


def evaluate_model_split(
    model_cfg: dict[str, Any],
    split: str,
    sample_ids: list[str],
    protocol_config: dict[str, Any],
    baseline_config: dict[str, Any],
    semantic_config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    thresholds: list[float],
    calibration_bins: int,
    score_bins: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = make_dataset(model_cfg, sample_ids, protocol_config, baseline_config, semantic_config, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    model = make_model(model_cfg, baseline_config, semantic_config, device)

    thresholds_arr = np.asarray(thresholds, dtype=np.float32)
    tp = np.zeros(len(thresholds_arr), dtype=np.float64)
    fp = np.zeros(len(thresholds_arr), dtype=np.float64)
    tn = np.zeros(len(thresholds_arr), dtype=np.float64)
    fn = np.zeros(len(thresholds_arr), dtype=np.float64)

    cal_count = np.zeros(calibration_bins, dtype=np.float64)
    cal_prob_sum = np.zeros(calibration_bins, dtype=np.float64)
    cal_pos_sum = np.zeros(calibration_bins, dtype=np.float64)
    score_count = np.zeros(score_bins, dtype=np.float64)
    score_pos = np.zeros(score_bins, dtype=np.float64)

    n_pixels = 0.0
    positives = 0.0
    prob_sum = 0.0
    prob_pos_sum = 0.0
    prob_neg_sum = 0.0
    brier_sum = 0.0
    nll_sum = 0.0
    inference_ms = 0.0

    started = time.perf_counter()
    for batch_idx, (x, y, _sample_ids) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        target = y.numpy().astype(np.float32).reshape(-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_started = time.perf_counter()
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).detach().cpu().numpy().astype(np.float32).reshape(-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms += (time.perf_counter() - forward_started) * 1000.0

        target_bool = target >= 0.5
        n = float(target.size)
        pos = float(target_bool.sum())
        n_pixels += n
        positives += pos
        prob_sum += float(probs.sum())
        prob_pos_sum += float(probs[target_bool].sum())
        prob_neg_sum += float(probs[~target_bool].sum())
        clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
        brier_sum += float(np.square(probs - target).sum())
        nll_sum += float((-(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))).sum())

        cal_idx = np.minimum((probs * calibration_bins).astype(np.int64), calibration_bins - 1)
        cal_count += np.bincount(cal_idx, minlength=calibration_bins)
        cal_prob_sum += np.bincount(cal_idx, weights=probs, minlength=calibration_bins)
        cal_pos_sum += np.bincount(cal_idx, weights=target, minlength=calibration_bins)

        score_idx = np.minimum((probs * score_bins).astype(np.int64), score_bins - 1)
        score_count += np.bincount(score_idx, minlength=score_bins)
        score_pos += np.bincount(score_idx, weights=target, minlength=score_bins)

        for idx, threshold in enumerate(thresholds_arr):
            pred = probs >= threshold
            tp[idx] += float(np.logical_and(pred, target_bool).sum())
            fp[idx] += float(np.logical_and(pred, ~target_bool).sum())
            tn[idx] += float(np.logical_and(~pred, ~target_bool).sum())
            fn[idx] += float(np.logical_and(~pred, target_bool).sum())

        if batch_idx % 20 == 0:
            print(f"  {model_cfg['name']} {split}: batch {batch_idx}/{len(loader)}")

    ece = 0.0
    signed_ece = 0.0
    mce = 0.0
    bin_rows: list[dict[str, Any]] = []
    for idx in range(calibration_bins):
        count = float(cal_count[idx])
        conf = float(cal_prob_sum[idx] / count) if count > 0 else math.nan
        acc = float(cal_pos_sum[idx] / count) if count > 0 else math.nan
        gap = conf - acc if count > 0 else math.nan
        if count > 0:
            weight = count / max(n_pixels, 1.0)
            ece += weight * abs(gap)
            signed_ece += weight * gap
            mce = max(mce, abs(gap))
        bin_rows.append(
            {
                "model": model_cfg["name"],
                "split": split,
                "bin": idx,
                "bin_lower": idx / calibration_bins,
                "bin_upper": (idx + 1) / calibration_bins,
                "count": int(count),
                "mean_confidence": conf,
                "empirical_positive_rate": acc,
                "calibration_gap_conf_minus_acc": gap,
            }
        )

    auprc, auroc = auc_from_score_hist(score_pos, score_count)
    threshold_rows: list[dict[str, Any]] = []
    for idx, threshold in enumerate(thresholds):
        metrics = threshold_metrics_from_counts(tp[idx], fp[idx], tn[idx], fn[idx])
        threshold_rows.append(
            {
                "model": model_cfg["name"],
                "split": split,
                "threshold": threshold,
                "tp": int(tp[idx]),
                "fp": int(fp[idx]),
                "tn": int(tn[idx]),
                "fn": int(fn[idx]),
                **metrics,
            }
        )

    score_rows = []
    for idx in range(score_bins):
        count = float(score_count[idx])
        score_rows.append(
            {
                "model": model_cfg["name"],
                "split": split,
                "score_bin": idx,
                "bin_lower": idx / score_bins,
                "bin_upper": (idx + 1) / score_bins,
                "count": int(count),
                "positive_count": float(score_pos[idx]),
            }
        )

    elapsed = time.perf_counter() - started
    pos_rate = positives / max(n_pixels, 1.0)
    neg_count = n_pixels - positives
    summary = {
        "model": model_cfg["name"],
        "split": split,
        "family": model_cfg.get("family", model_cfg["name"]),
        "display_name": model_cfg.get("display_name", model_cfg.get("family", model_cfg["name"])),
        "checkpoint_selection": model_cfg.get("checkpoint_selection", ""),
        "seed": model_cfg.get("seed", ""),
        "pair_id": model_cfg.get("pair_id", model_cfg["name"]),
        "model_type": model_cfg.get("type", ""),
        "pixels": int(n_pixels),
        "positives": int(positives),
        "positive_rate": pos_rate,
        "mean_probability": prob_sum / max(n_pixels, 1.0),
        "mean_probability_on_positive": prob_pos_sum / max(positives, 1.0),
        "mean_probability_on_negative": prob_neg_sum / max(neg_count, 1.0),
        "brier_score": brier_sum / max(n_pixels, 1.0),
        "nll": nll_sum / max(n_pixels, 1.0),
        "ece": ece,
        "signed_ece_conf_minus_acc": signed_ece,
        "mce": mce,
        "auprc": auprc,
        "auroc": auroc,
        "inference_time_ms": inference_ms,
        "elapsed_seconds": elapsed,
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, threshold_rows, bin_rows, score_rows


def select_threshold_transfer(threshold_rows: list[dict[str, Any]], selection_split: str, final_split: str) -> list[dict[str, Any]]:
    by_model_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in threshold_rows:
        by_model_split[(str(row["model"]), str(row["split"]))].append(row)
    models = sorted({str(row["model"]) for row in threshold_rows})
    transfer_rows: list[dict[str, Any]] = []
    criteria = ["iou", "f1", "balanced_accuracy"]
    for model in models:
        val_rows = by_model_split.get((model, selection_split), [])
        test_rows = by_model_split.get((model, final_split), [])
        if not val_rows or not test_rows:
            continue
        test_by_threshold = {float(row["threshold"]): row for row in test_rows}
        for criterion in criteria:
            selected = max(
                val_rows,
                key=lambda row: (
                    to_float(row[criterion]),
                    to_float(row["iou"]),
                    to_float(row["f1"]),
                    -abs(to_float(row["threshold"]) - 0.5),
                ),
            )
            threshold = float(selected["threshold"])
            test_row = test_by_threshold[threshold]
            default_row = test_by_threshold.get(0.5, test_row)
            transfer_rows.append(
                {
                    "model": model,
                    "selection_metric": criterion,
                    "selected_threshold": threshold,
                    f"val_{criterion}": selected[criterion],
                    "val_iou": selected["iou"],
                    "val_f1": selected["f1"],
                    "val_precision": selected["precision"],
                    "val_recall": selected["recall"],
                    "test_iou": test_row["iou"],
                    "test_f1": test_row["f1"],
                    "test_precision": test_row["precision"],
                    "test_recall": test_row["recall"],
                    "test_pred_positive_rate": test_row["pred_positive_rate"],
                    "test_iou_gain_vs_threshold_0_5": to_float(test_row["iou"]) - to_float(default_row["iou"]),
                    "test_f1_gain_vs_threshold_0_5": to_float(test_row["f1"]) - to_float(default_row["f1"]),
                }
            )
    return transfer_rows


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        out = {key: value for key, value in zip(keys, group_key)}
        out["members"] = len(group_rows)
        out["models"] = ";".join(str(row["model"]) for row in group_rows)
        for metric in metrics:
            values = [to_float(row.get(metric)) for row in group_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = std(values)
            out[f"{metric}_min"] = min([x for x in values if not math.isnan(x)], default=math.nan)
            out[f"{metric}_max"] = max([x for x in values if not math.isnan(x)], default=math.nan)
        output.append(out)
    return sorted(output, key=lambda row: tuple(str(row.get(key, "")) for key in keys))


def join_with_path_safety(calibration_rows: list[dict[str, Any]], path_safety_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    safety_by_model = {str(row["model"]): row for row in path_safety_rows}
    joined: list[dict[str, Any]] = []
    for row in calibration_rows:
        if row["split"] != "test":
            continue
        safety = safety_by_model.get(str(row["model"]), {})
        out = dict(row)
        for key in [
            "threshold",
            "planning_safety_utility",
            "success_rate_collision_free",
            "path_found_rate",
            "collision_rate_among_found",
            "mean_map_iou",
        ]:
            out[f"path_{key}"] = to_float(safety.get(key))
        joined.append(out)
    return joined


def plot_calibration_vs_safety(path: Path, joined_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in joined_rows if not math.isnan(to_float(row.get("path_planning_safety_utility")))]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    specs = [
        ("ece", "ECE", False),
        ("brier_score", "Brier score", False),
        ("mean_probability", "Mean probability", False),
    ]
    colors = {"best": "#2563eb", "last": "#f97316", "": "#64748b"}
    for ax, (metric, label, _invert) in zip(axes, specs):
        for row in rows:
            tag = str(row.get("checkpoint_selection", ""))
            ax.scatter(to_float(row.get(metric)), to_float(row.get("path_planning_safety_utility")), color=colors.get(tag, "#64748b"), s=44, alpha=0.82)
        ax.set_xlabel(label)
        ax.set_ylabel("path safety utility")
        ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=label, markersize=8)
        for label, color in [("best", "#2563eb"), ("last", "#f97316")]
    ]
    axes[-1].legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_reliability(path: Path, bin_rows: list[dict[str, Any]], plot_models: list[str], split: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.plot([0, 1], [0, 1], color="#111827", linewidth=1.2, linestyle="--", label="perfect")
    colors = plt.get_cmap("tab10").colors
    for idx, model in enumerate(plot_models):
        rows = [row for row in bin_rows if row["model"] == model and row["split"] == split and int(row["count"]) > 0]
        if not rows:
            continue
        xs = [to_float(row["mean_confidence"]) for row in rows]
        ys = [to_float(row["empirical_positive_rate"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=model, color=colors[idx % len(colors)])
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical positive rate")
    ax.set_title(f"{split} reliability curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_probability_histograms(path: Path, score_rows: list[dict[str, Any]], plot_models: list[str], split: str) -> None:
    fig, axes = plt.subplots(len(plot_models), 1, figsize=(8.5, max(2.0 * len(plot_models), 4.0)), sharex=True)
    if len(plot_models) == 1:
        axes = [axes]
    for ax, model in zip(axes, plot_models):
        rows = [row for row in score_rows if row["model"] == model and row["split"] == split]
        if not rows:
            continue
        xs = [(to_float(row["bin_lower"]) + to_float(row["bin_upper"])) / 2.0 for row in rows]
        total = np.asarray([to_float(row["count"]) for row in rows], dtype=np.float64)
        pos = np.asarray([to_float(row["positive_count"]) for row in rows], dtype=np.float64)
        neg = np.maximum(total - pos, 0)
        norm = max(total.sum(), 1.0)
        ax.fill_between(xs, neg / norm, color="#94a3b8", alpha=0.75, label="negative")
        ax.fill_between(xs, (neg + pos) / norm, neg / norm, color="#16a34a", alpha=0.75, label="positive")
        ax.set_ylabel(model, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("predicted probability")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_threshold_transfer(path: Path, transfer_rows: list[dict[str, Any]], model_meta: dict[str, dict[str, Any]]) -> None:
    rows = [row for row in transfer_rows if row["selection_metric"] == "iou"]
    families = []
    for row in rows:
        family = model_meta.get(str(row["model"]), {}).get("family", row["model"])
        tag = model_meta.get(str(row["model"]), {}).get("checkpoint_selection", "")
        label = f"{family}:{tag}"
        if label not in families:
            families.append(label)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = model_meta.get(str(row["model"]), {})
        label = f"{meta.get('family', row['model'])}:{meta.get('checkpoint_selection', '')}"
        grouped[label].append(row)
    labels = families
    x = np.arange(len(labels))
    means = [mean([to_float(row["test_iou"]) for row in grouped[label]]) for label in labels]
    errors = [std([to_float(row["test_iou"]) for row in grouped[label]]) for label in labels]
    thresholds = [mean([to_float(row["selected_threshold"]) for row in grouped[label]]) for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    axes[0].bar(x, means, yerr=errors, capsize=4, color="#2563eb", alpha=0.86)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("test IoU at val-selected IoU threshold")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, thresholds, color="#0f766e", alpha=0.86)
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("val-selected threshold")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def markdown_family_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| family | ckpt | members | Brier | ECE | signed ECE | AUPRC | mean prob | path safety | path collision/found | path mask IoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['family']}` | `{row['checkpoint_selection']}` | {row['members']} | "
            f"{fmt(row['brier_score_mean'])} ± {fmt(row['brier_score_std'])} | "
            f"{fmt(row['ece_mean'])} ± {fmt(row['ece_std'])} | "
            f"{fmt(row['signed_ece_conf_minus_acc_mean'])} ± {fmt(row['signed_ece_conf_minus_acc_std'])} | "
            f"{fmt(row['auprc_mean'])} ± {fmt(row['auprc_std'])} | "
            f"{fmt(row['mean_probability_mean'])} ± {fmt(row['mean_probability_std'])} | "
            f"{fmt(row['path_planning_safety_utility_mean'])} ± {fmt(row['path_planning_safety_utility_std'])} | "
            f"{fmt(row['path_collision_rate_among_found_mean'])} ± {fmt(row['path_collision_rate_among_found_std'])} | "
            f"{fmt(row['path_mean_map_iou_mean'])} ± {fmt(row['path_mean_map_iou_std'])} |"
        )
    return "\n".join(lines)


def write_report(
    report_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    family_rows: list[dict[str, Any]],
    joined_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
) -> None:
    best_safety = max(joined_rows, key=lambda row: to_float(row.get("path_planning_safety_utility")))
    best_ece = min(joined_rows, key=lambda row: to_float(row.get("ece")))
    best_brier = min(joined_rows, key=lambda row: to_float(row.get("brier_score")))
    corr_ece = next((row for row in correlation_rows if row["metric"] == "ece"), None)
    corr_brier = next((row for row in correlation_rows if row["metric"] == "brier_score"), None)
    lines = [
        "# 第十三步：概率校准与阈值迁移诊断报告",
        "",
        f"生成日期：{datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## 实验目的",
        "",
        "第十一步和第十二步说明：像素级 IoU、checkpoint selection 与路径规划 safety utility 之间并不是简单同向关系。第十三步专门检查概率图本身：模型是否校准良好，概率阈值从 validation 迁移到 test 是否稳定，以及校准指标能否解释路径安全效用差距。",
        "",
        "本步不训练新模型，复用第十二步的 best/last checkpoint 展开配置；对 val/test 全量样本计算像素级概率指标，并与第十二步 safety-selected 路径规划结果连接分析。",
        "",
        "## 指标说明",
        "",
        "- Brier score：概率平方误差，越低越好。",
        "- NLL：像素级负对数似然，越低越好。",
        "- ECE：expected calibration error，越低越好。",
        "- signed ECE：预测置信度减真实正例率；正值表示整体过度自信，负值表示整体偏保守。",
        "- AUPRC/AUROC：阈值无关排序能力。",
        "- path safety：第十二步 validation safety utility 选阈值后的 test safety utility。",
        "",
        "## family 级 test 校准与路径安全结果",
        "",
        markdown_family_table(family_rows),
        "",
        "## 核心发现",
        "",
        f"- 路径 safety utility 最高的模型是 `{best_safety['model']}`，safety utility 为 {fmt(best_safety['path_planning_safety_utility'])}。",
        f"- ECE 最低的模型是 `{best_ece['model']}`，ECE 为 {fmt(best_ece['ece'])}。",
        f"- Brier score 最低的模型是 `{best_brier['model']}`，Brier 为 {fmt(best_brier['brier_score'])}。",
    ]
    if corr_ece:
        lines.append(f"- ECE 与路径 safety utility 的 Pearson 相关为 {fmt(corr_ece['pearson_with_path_safety'])}。")
    if corr_brier:
        lines.append(f"- Brier score 与路径 safety utility 的 Pearson 相关为 {fmt(corr_brier['pearson_with_path_safety'])}。")
    counterexample = next((row for row in joined_rows if row["model"] == "semantic_original_recipe_seed20260518_best"), None)
    lines.extend(
        [
            "",
            (
                "一个关键反证是：`semantic_original_recipe_seed20260518_best` 同时取得最低 ECE "
                f"{fmt(counterexample['ece']) if counterexample else ''}、最低 Brier {fmt(counterexample['brier_score']) if counterexample else ''}，"
                f"但它的路径 safety utility 只有 {fmt(counterexample['path_planning_safety_utility']) if counterexample else ''}，"
                f"collision/found 为 {fmt(counterexample['path_collision_rate_among_found']) if counterexample else ''}。也就是说，像素概率校准和像素 IoU 做得好，仍然可能在 A* 路径上产生危险连通。"
            ),
            "",
            "如果校准指标与路径 safety utility 相关性较弱，就说明问题不只是概率是否校准，而是概率图的空间连通结构、局部假阳性分布和路径约束之间存在更复杂的冲突。这样的结论会把下一步自然推向 path-aware selection。",
            "",
            "## 产物",
            "",
            "- `configs/probability_calibration_v1.json`：第十三步概率校准诊断配置。",
            "- `scripts/evaluate_probability_calibration.py`：第十三步校准评估、聚合、绘图和报告脚本。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/model_split_calibration.csv`：逐模型逐 split 校准指标。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/threshold_metrics.csv`：逐模型逐 split 逐阈值像素指标。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/val_selected_threshold_transfer.csv`：validation 像素指标选阈值后的 test 迁移结果。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/calibration_bins.csv`：reliability diagram 所需分箱数据。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/calibration_path_safety_join.csv`：test 校准指标与路径 safety utility 的连接表。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/family_calibration_path_summary.csv`：family/checkpoint 级 mean/std 汇总。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/calibration_vs_safety.png`：校准指标与路径 safety utility 散点图。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/reliability_curves_test.png`：代表模型 reliability curve。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/probability_histograms_test.png`：代表模型概率分布图。",
            f"- `{run_dir.relative_to(FINALLY_ROOT).as_posix()}/threshold_transfer_iou.png`：validation 像素 IoU 选阈值后的 test 迁移图。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pixel probability calibration for traversability models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-models", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    protocol_config = load_json(PROTOCOL_CONFIG_PATH)
    baseline_config = load_json(BASELINE_CONFIG_PATH)
    semantic_config = load_json(SEMANTIC_CONFIG_PATH)
    source_model_config = load_json(Path(config["source_model_config"]))
    models = [model for model in source_model_config["models"] if model.get("type") != "oracle"]
    if args.max_models > 0:
        models = models[: args.max_models]
    model_meta = {str(model["name"]): model for model in models}

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_root = Path(config["outputs"]["root"])
    run_dir = output_root / str(config["run_name"])
    ensure_dir(run_dir)
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "source_model_config_snapshot.json").write_text(json.dumps(source_model_config, indent=2), encoding="utf-8")

    thresholds = [float(x) for x in config["thresholds"]]
    calibration_bins = int(config["calibration_bins"])
    score_bins = int(config["score_bins"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])

    all_calibration_rows: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print(f"Calibration run={config['run_name']} models={len(models)} device={device}")
    for split in config["splits"]:
        sample_ids = read_ids(Path(protocol_config["outputs_root"]) / "protocol" / f"{split}.txt")
        print(f"Split {split}: samples={len(sample_ids)}")
        for index, model_cfg in enumerate(models, start=1):
            print(f"[{index}/{len(models)}] evaluating {model_cfg['name']} split={split}")
            summary, threshold_rows, bin_rows, score_rows = evaluate_model_split(
                model_cfg=model_cfg,
                split=str(split),
                sample_ids=sample_ids,
                protocol_config=protocol_config,
                baseline_config=baseline_config,
                semantic_config=semantic_config,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                thresholds=thresholds,
                calibration_bins=calibration_bins,
                score_bins=score_bins,
            )
            all_calibration_rows.append(summary)
            all_threshold_rows.extend(threshold_rows)
            all_bin_rows.extend(bin_rows)
            all_score_rows.extend(score_rows)

    transfer_rows = select_threshold_transfer(all_threshold_rows, str(config["selection_split"]), str(config["final_split"]))
    transfer_rows = [{**row, **{k: v for k, v in model_meta.get(row["model"], {}).items() if k in {"family", "display_name", "checkpoint_selection", "seed", "pair_id"}}} for row in transfer_rows]
    path_safety_rows = read_csv(Path(config["path_planning_safety_csv"]))
    joined_rows = join_with_path_safety(all_calibration_rows, path_safety_rows)

    aggregate_metrics = [
        "brier_score",
        "nll",
        "ece",
        "signed_ece_conf_minus_acc",
        "mce",
        "auprc",
        "auroc",
        "mean_probability",
        "mean_probability_on_positive",
        "mean_probability_on_negative",
        "path_planning_safety_utility",
        "path_success_rate_collision_free",
        "path_path_found_rate",
        "path_collision_rate_among_found",
        "path_mean_map_iou",
    ]
    family_rows = aggregate_rows(joined_rows, ["family", "checkpoint_selection"], aggregate_metrics)

    correlation_rows = []
    for metric in ["brier_score", "nll", "ece", "signed_ece_conf_minus_acc", "mce", "auprc", "auroc", "mean_probability", "path_mean_map_iou"]:
        correlation_rows.append(
            {
                "metric": metric,
                "pearson_with_path_safety": pearson(
                    [to_float(row.get(metric)) for row in joined_rows],
                    [to_float(row.get("path_planning_safety_utility")) for row in joined_rows],
                ),
            }
        )

    write_csv(run_dir / "model_split_calibration.csv", all_calibration_rows)
    write_csv(run_dir / "threshold_metrics.csv", all_threshold_rows)
    write_csv(run_dir / "val_selected_threshold_transfer.csv", transfer_rows)
    write_csv(run_dir / "calibration_bins.csv", all_bin_rows)
    write_csv(run_dir / "score_histograms.csv", all_score_rows)
    write_csv(run_dir / "calibration_path_safety_join.csv", joined_rows)
    write_csv(run_dir / "family_calibration_path_summary.csv", family_rows)
    write_csv(run_dir / "calibration_path_correlations.csv", correlation_rows)

    plot_models = [name for name in config.get("plot_models", []) if name in model_meta]
    plot_calibration_vs_safety(run_dir / "calibration_vs_safety.png", joined_rows)
    plot_reliability(run_dir / "reliability_curves_test.png", all_bin_rows, plot_models, "test")
    plot_probability_histograms(run_dir / "probability_histograms_test.png", all_score_rows, plot_models, "test")
    plot_threshold_transfer(run_dir / "threshold_transfer_iou.png", transfer_rows, model_meta)

    elapsed = time.perf_counter() - started
    summary = {
        "run_name": config["run_name"],
        "models": len(models),
        "splits": config["splits"],
        "device": str(device),
        "elapsed_seconds": elapsed,
        "outputs": {
            "model_split_calibration": str(run_dir / "model_split_calibration.csv"),
            "threshold_metrics": str(run_dir / "threshold_metrics.csv"),
            "val_selected_threshold_transfer": str(run_dir / "val_selected_threshold_transfer.csv"),
            "calibration_path_safety_join": str(run_dir / "calibration_path_safety_join.csv"),
            "family_calibration_path_summary": str(run_dir / "family_calibration_path_summary.csv"),
            "report": str(config["outputs"]["report"]),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(Path(config["outputs"]["report"]), config, run_dir, family_rows, joined_rows, correlation_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
