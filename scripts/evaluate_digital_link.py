from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
CORE_SCRIPTS = FINALLY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CORE_SCRIPTS))

from evaluate_path_planning import evaluate_prediction, make_problem  # noqa: E402
from train_baselines import TraversabilityDataset, ensure_dir, load_json, read_ids  # noqa: E402
from train_strict_bottleneck import StrictRGBDBottleneckNet  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else math.nan


def finite_std(values: list[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.std(clean, ddof=0)) if clean else math.nan


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou": iou,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def update_counts(prob: np.ndarray, target: np.ndarray, threshold: float, counts: dict[str, int]) -> None:
    pred = prob >= threshold
    label = target.astype(bool)
    counts["tp"] += int(np.logical_and(pred, label).sum())
    counts["fp"] += int(np.logical_and(pred, ~label).sum())
    counts["tn"] += int(np.logical_and(~pred, ~label).sum())
    counts["fn"] += int(np.logical_and(~pred, label).sum())


def theoretical_bpsk_ber(ebn0_db: float) -> float:
    linear = 10.0 ** (float(ebn0_db) / 10.0)
    return 0.5 * math.erfc(math.sqrt(linear))


def transmit_payload(
    z: torch.Tensor,
    profile: dict[str, Any],
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    mode = str(profile["mode"])
    if mode == "clean":
        return z, {
            "payload_bits": int(z[0].numel() * 32),
            "bit_errors": 0,
            "transmitted_bits": 0,
            "raw_bit_errors": 0,
            "raw_transmitted_bits": 0,
            "empirical_ber": 0.0,
            "raw_empirical_ber": 0.0,
            "theoretical_ber": 0.0,
        }

    flat = z.flatten(1)
    z_min = flat.min(dim=1).values.view(-1, 1, 1, 1)
    z_max = flat.max(dim=1).values.view(-1, 1, 1, 1)
    scale = (z_max - z_min).clamp_min(1e-6)
    q = torch.round((z - z_min) / scale * 255.0).to(torch.int64)
    q_hat = q
    bit_errors = 0
    transmitted_bits = int(q.numel() * 8)
    raw_bit_errors = 0
    raw_transmitted_bits = 0
    theoretical = 0.0

    if mode in {"bpsk_awgn", "bpsk_awgn_rep3"}:
        set_seed(seed)
        shifts = torch.arange(7, -1, -1, device=z.device, dtype=torch.int64)
        source_bits = ((q.unsqueeze(-1) >> shifts) & 1).to(torch.int64)
        channel_bits = (
            source_bits.unsqueeze(-1).expand(*source_bits.shape, 3).reshape(*source_bits.shape[:-1], 24)
            if mode == "bpsk_awgn_rep3"
            else source_bits
        )
        symbols = 1.0 - 2.0 * channel_bits.to(z.dtype)
        ebn0_db = float(profile["ebn0_db"])
        ebn0_linear = 10.0 ** (ebn0_db / 10.0)
        sigma = math.sqrt(1.0 / (2.0 * ebn0_linear))
        received = symbols + torch.randn_like(symbols) * sigma
        channel_bits_hat = (received < 0.0).to(torch.int64)
        raw_bit_errors = int((channel_bits_hat != channel_bits).sum().item())
        raw_transmitted_bits = int(channel_bits.numel())
        if mode == "bpsk_awgn_rep3":
            decoded_bits = (channel_bits_hat.reshape(*source_bits.shape, 3).sum(dim=-1) >= 2).to(torch.int64)
        else:
            decoded_bits = channel_bits_hat
        bit_errors = int((decoded_bits != source_bits).sum().item())
        transmitted_bits = int(source_bits.numel())
        q_hat = torch.sum(decoded_bits << shifts, dim=-1)
        theoretical = theoretical_bpsk_ber(ebn0_db)
    elif mode != "q8":
        raise ValueError(f"Unsupported profile mode: {mode}")

    z_hat = q_hat.to(z.dtype) / 255.0 * scale + z_min
    payload_bits = int(q[0].numel() * 8 * (3 if mode == "bpsk_awgn_rep3" else 1) + 64)
    return z_hat, {
        "payload_bits": payload_bits,
        "bit_errors": bit_errors,
        "transmitted_bits": transmitted_bits if mode in {"bpsk_awgn", "bpsk_awgn_rep3"} else 0,
        "raw_bit_errors": raw_bit_errors,
        "raw_transmitted_bits": raw_transmitted_bits,
        "empirical_ber": bit_errors / max(1, transmitted_bits)
        if mode in {"bpsk_awgn", "bpsk_awgn_rep3"}
        else 0.0,
        "raw_empirical_ber": raw_bit_errors / max(1, raw_transmitted_bits)
        if mode in {"bpsk_awgn", "bpsk_awgn_rep3"}
        else 0.0,
        "theoretical_ber": theoretical,
    }


def load_model(config: dict[str, Any], device: torch.device) -> tuple[StrictRGBDBottleneckNet, dict[str, Any]]:
    checkpoint = torch.load(Path(config["checkpoint"]), map_location=device, weights_only=False)
    variant = checkpoint["variant"]
    train_config = checkpoint["config"]
    model = StrictRGBDBottleneckNet(
        base_channels=int(train_config["base_channels"]),
        latent_channels=int(variant["latent_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, train_config


def prepare_problems(
    sample_ids: list[str],
    protocol_config: dict[str, Any],
    planning_config: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    problems: list[Any] = []
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        problem, selection = make_problem(sample_id, protocol_config, planning_config)
        rows.append(selection)
        if problem is not None:
            problems.append(problem)
    return problems, rows


def aggregate_planning(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["profile"]), str(row["split"]), float(row["threshold"]))].append(row)
    output = []
    for (profile, split, threshold), group in sorted(groups.items()):
        found_count = sum(bool(row["path_found"]) for row in group)
        success_count = sum(bool(row["success"]) for row in group)
        unsafe_found = sum(bool(row["path_found"]) and not bool(row["success"]) for row in group)
        path_found_rate = found_count / max(1, len(group))
        success_rate = success_count / max(1, len(group))
        output.append(
            {
                "profile": profile,
                "split": split,
                "threshold": threshold,
                "rows": len(group),
                "unique_samples": len({row["sample_id"] for row in group}),
                "repeats": len({row["repeat"] for row in group}),
                "path_found_rate": path_found_rate,
                "success_rate_collision_free": success_rate,
                "unsafe_found_rate": unsafe_found / max(1, len(group)),
                "collision_rate_among_found": unsafe_found / max(1, found_count),
                "planning_safety_utility": success_rate - (path_found_rate - success_rate),
                "mean_map_iou": finite_mean([row["map_iou"] for row in group]),
                "std_map_iou": finite_std([row["map_iou"] for row in group]),
                "mean_collision_fraction_found": finite_mean(
                    [row["collision_fraction"] for row in group if bool(row["path_found"])]
                ),
                "mean_inference_ms": finite_mean([row["inference_ms"] for row in group]),
                "mean_encoder_ms": finite_mean([row["encoder_ms"] for row in group]),
                "mean_channel_ms": finite_mean([row["channel_ms"] for row in group]),
                "mean_decoder_ms": finite_mean([row["decoder_ms"] for row in group]),
            }
        )
    return output


def choose_thresholds(summary_rows: list[dict[str, Any]], selection_split: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row["split"] == selection_split:
            groups[str(row["profile"])].append(row)
    selected = []
    for profile, candidates in sorted(groups.items()):
        best = max(
            candidates,
            key=lambda row: (
                float(row["planning_safety_utility"]),
                float(row["success_rate_collision_free"]),
                float(row["mean_map_iou"]),
                float(row["path_found_rate"]),
            ),
        )
        selected.append(
            {
                "profile": profile,
                "selected_threshold": best["threshold"],
                "selection_planning_safety_utility": best["planning_safety_utility"],
                "selection_success_rate_collision_free": best["success_rate_collision_free"],
                "selection_mean_map_iou": best["mean_map_iou"],
            }
        )
    return selected


def merge_selected_test(
    summaries: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    final_split: str,
) -> list[dict[str, Any]]:
    index = {
        (str(row["profile"]), float(row["threshold"])): row
        for row in summaries
        if row["split"] == final_split
    }
    output = []
    for selection in selected:
        key = (str(selection["profile"]), float(selection["selected_threshold"]))
        test = index[key]
        merged = dict(selection)
        for name, value in test.items():
            if name not in {"profile", "threshold"}:
                merged[f"test_{name}"] = value
        output.append(merged)
    return output


def aggregate_segmentation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["profile"]), str(row["split"]))].append(row)
    output = []
    for (profile, split), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "profile": profile,
            "split": split,
            "repeats": len(group),
            "samples_per_repeat": group[0]["samples"],
        }
        for metric in ("accuracy", "precision", "recall", "specificity", "f1", "iou", "balanced_accuracy"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = finite_mean(values)
            item[f"{metric}_std"] = finite_std(values)
        output.append(item)
    return output


def plot_results(
    path: Path,
    profile_rows: list[dict[str, Any]],
    segmentation_summary: list[dict[str, Any]],
    final_split: str,
    link_rates: list[float],
) -> None:
    bpsk = [row for row in profile_rows if row["mode"] == "bpsk_awgn"]
    bpsk.sort(key=lambda row: float(row["ebn0_db"]))
    result_index = {
        str(row["profile"]): row
        for row in segmentation_summary
        if str(row["split"]) == final_split
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    x = [float(row["ebn0_db"]) for row in bpsk]
    axes[0].semilogy(x, [float(row["empirical_ber"]) for row in bpsk], "o-", label="Empirical")
    axes[0].semilogy(x, [float(row["theoretical_ber"]) for row in bpsk], "--", label="Theory")
    axes[0].set_xlabel("$E_b/N_0$ (dB)")
    axes[0].set_ylabel("Bit-error rate")
    axes[0].set_title("Uncoded BPSK/AWGN")
    axes[0].legend()
    axes[1].errorbar(
        x,
        [float(result_index[str(row["name"])]["iou_mean"]) for row in bpsk],
        yerr=[float(result_index[str(row["name"])]["iou_std"]) for row in bpsk],
        fmt="o-",
        color="#2563eb",
        capsize=3,
    )
    axes[1].set_xlabel("$E_b/N_0$ (dB)")
    axes[1].set_ylabel("Mask IoU at threshold 0.5")
    axes[1].set_ylim(0.4, 0.72)
    axes[1].set_title("Task-level degradation")
    q8_payload_bits = float(next(row["payload_bits"] for row in profile_rows if row["name"] == "q8_ideal"))
    rep3_payload_bits = float(next(row["payload_bits"] for row in profile_rows if row["mode"] == "bpsk_awgn_rep3"))
    positions = np.arange(len(link_rates))
    width = 0.36
    axes[2].bar(
        positions - width / 2,
        [q8_payload_bits / (rate * 1e6) * 1000.0 for rate in link_rates],
        width,
        label="Uncoded",
        color="#ea580c",
    )
    axes[2].bar(
        positions + width / 2,
        [rep3_payload_bits / (rate * 1e6) * 1000.0 for rate in link_rates],
        width,
        label="Repetition-3",
        color="#0f766e",
    )
    axes[2].set_xticks(positions, [str(rate) for rate in link_rates])
    axes[2].set_xlabel("Payload rate (Mbit/s)")
    axes[2].set_ylabel("Transmission time (ms)")
    axes[2].set_title("Reliability--latency trade-off")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strict semantic payload over uncoded BPSK/AWGN.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    protocol_config = load_json(Path(config["protocol_config"]))
    output_root = Path(config["output_root"])
    ensure_dir(output_root)
    device_name = str(config.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model, train_config = load_model(config, device)
    planning_config = {
        "image_size": int(config["image_size"]),
        "connectivity": int(config["connectivity"]),
        "min_gt_component_pixels": int(config["min_gt_component_pixels"]),
        "min_start_goal_distance": float(config["min_start_goal_distance"]),
    }

    outputs_root = Path(protocol_config["outputs_root"])
    splits = [str(config["selection_split"]), str(config["final_split"])]
    problems_by_split: dict[str, list[Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    for split in splits:
        ids = read_ids(outputs_root / "protocol" / f"{split}.txt")
        problems, selections = prepare_problems(ids, protocol_config, planning_config)
        problems_by_split[split] = problems
        selection_rows.extend({"split": split, **row} for row in selections)

    thresholds = [float(value) for value in config["thresholds"]]
    planning_rows: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    started = time.time()

    for profile_index, profile in enumerate(config["profiles"]):
        profile_name = str(profile["name"])
        repeats = int(profile.get("repeats", 1))
        aggregate_errors = 0
        aggregate_bits = 0
        aggregate_raw_errors = 0
        aggregate_raw_bits = 0
        timing_rows: list[dict[str, float]] = []
        for repeat in range(repeats):
            repeat_seed = int(config["seed"]) + profile_index * 100000 + repeat * 1000
            for split in splits:
                problems = problems_by_split[split]
                sample_ids = [problem.sample_id for problem in problems]
                dataset = TraversabilityDataset(sample_ids, "rgbd", protocol_config, train_config, split)
                id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
                counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                for problem_index, problem in enumerate(problems):
                    x, _target, _sample_id = dataset[id_to_index[problem.sample_id]]
                    x = x.unsqueeze(0).to(device, non_blocking=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        z = model.encode(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    with torch.no_grad():
                        z_hat, channel_stats = transmit_payload(
                            z,
                            profile,
                            seed=repeat_seed + problem_index,
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    with torch.no_grad():
                        logits = model.decode(z_hat)
                        prob = torch.sigmoid(logits).detach().cpu().numpy()[0, 0].astype(np.float32)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    encoder_ms = (t1 - t0) * 1000.0
                    channel_ms = (t2 - t1) * 1000.0
                    decoder_ms = (t3 - t2) * 1000.0
                    inference_ms = (t3 - t0) * 1000.0
                    aggregate_errors += int(channel_stats["bit_errors"])
                    aggregate_bits += int(channel_stats["transmitted_bits"])
                    aggregate_raw_errors += int(channel_stats["raw_bit_errors"])
                    aggregate_raw_bits += int(channel_stats["raw_transmitted_bits"])
                    timing_rows.append(
                        {
                            "encoder_ms": encoder_ms,
                            "channel_ms": channel_ms,
                            "decoder_ms": decoder_ms,
                            "inference_ms": inference_ms,
                        }
                    )
                    update_counts(prob, problem.gt_mask, 0.5, counts)
                    for threshold in thresholds:
                        result = evaluate_prediction(problem, prob, threshold, int(config["connectivity"]))
                        planning_rows.append(
                            {
                                "profile": profile_name,
                                "repeat": repeat,
                                "split": split,
                                "sample_id": problem.sample_id,
                                "threshold": threshold,
                                "path_found": bool(result["path_found"]),
                                "success": bool(result["success"]),
                                "collision_cells": int(result["collision_cells"]),
                                "collision_fraction": float(result["collision_fraction"])
                                if np.isfinite(result["collision_fraction"])
                                else math.nan,
                                "map_iou": float(result["map_iou"]),
                                "encoder_ms": encoder_ms,
                                "channel_ms": channel_ms,
                                "decoder_ms": decoder_ms,
                                "inference_ms": inference_ms,
                            }
                        )
                segmentation_rows.append(
                    {
                        "profile": profile_name,
                        "repeat": repeat,
                        "split": split,
                        "samples": len(problems),
                        **counts,
                        **metrics_from_counts(counts),
                    }
                )
                print(
                    f"profile={profile_name} repeat={repeat + 1}/{repeats} split={split} "
                    f"samples={len(problems)} iou@0.5={segmentation_rows[-1]['iou']:.4f}"
                )

        if profile["mode"] == "clean":
            payload_bits = int(model.latent_channels * 16 * 16 * 32)
        elif profile["mode"] == "bpsk_awgn_rep3":
            payload_bits = int(model.latent_channels * 16 * 16 * 8 * 3 + 64)
        else:
            payload_bits = int(model.latent_channels * 16 * 16 * 8 + 64)
        profile_rows.append(
            {
                "name": profile_name,
                "label": profile["label"],
                "mode": profile["mode"],
                "ebn0_db": profile.get("ebn0_db", ""),
                "repeats": repeats,
                "payload_bits": payload_bits,
                "empirical_ber": aggregate_raw_errors / max(1, aggregate_raw_bits)
                if profile["mode"] in {"bpsk_awgn", "bpsk_awgn_rep3"}
                else 0.0,
                "residual_ber": aggregate_errors / max(1, aggregate_bits)
                if profile["mode"] in {"bpsk_awgn", "bpsk_awgn_rep3"}
                else 0.0,
                "theoretical_ber": theoretical_bpsk_ber(float(profile["ebn0_db"]))
                if profile["mode"] in {"bpsk_awgn", "bpsk_awgn_rep3"}
                else 0.0,
                "theoretical_residual_ber": (
                    3.0 * theoretical_bpsk_ber(float(profile["ebn0_db"])) ** 2
                    - 2.0 * theoretical_bpsk_ber(float(profile["ebn0_db"])) ** 3
                )
                if profile["mode"] == "bpsk_awgn_rep3"
                else (
                    theoretical_bpsk_ber(float(profile["ebn0_db"]))
                    if profile["mode"] == "bpsk_awgn"
                    else 0.0
                ),
                "mean_encoder_ms": finite_mean([row["encoder_ms"] for row in timing_rows]),
                "mean_channel_ms": finite_mean([row["channel_ms"] for row in timing_rows]),
                "mean_decoder_ms": finite_mean([row["decoder_ms"] for row in timing_rows]),
                "mean_inference_ms": finite_mean([row["inference_ms"] for row in timing_rows]),
            }
        )

    planning_summary = aggregate_planning(planning_rows)
    threshold_selection = choose_thresholds(planning_summary, str(config["selection_split"]))
    selected_test = merge_selected_test(planning_summary, threshold_selection, str(config["final_split"]))
    segmentation_summary = aggregate_segmentation(segmentation_rows)
    link_rates = [float(value) for value in config["link_rates_mbps"]]
    latency_rows = []
    q8_bits = int(model.latent_channels * 16 * 16 * 8 + 64)
    for scheme, bits in (("uncoded_q8", q8_bits), ("repetition3_q8", int((q8_bits - 64) * 3 + 64))):
        for rate in link_rates:
            latency_rows.append(
                {
                    "payload_scheme": scheme,
                    "link_rate_mbps": rate,
                    "payload_bits": bits,
                    "transmission_time_ms": bits / (rate * 1e6) * 1000.0,
                }
            )

    write_csv(output_root / "sample_selection.csv", selection_rows)
    write_csv(output_root / "profile_summary.csv", profile_rows)
    write_csv(output_root / "segmentation_repeat.csv", segmentation_rows)
    write_csv(output_root / "segmentation_summary.csv", segmentation_summary)
    write_csv(output_root / "planning_rows.csv", planning_rows)
    write_csv(output_root / "planning_summary.csv", planning_summary)
    write_csv(output_root / "threshold_selection.csv", threshold_selection)
    write_csv(output_root / "selected_test_results.csv", selected_test)
    write_csv(output_root / "link_latency.csv", latency_rows)
    plot_results(
        output_root / "digital_link_summary.png",
        profile_rows,
        segmentation_summary,
        str(config["final_split"]),
        link_rates,
    )
    summary = {
        "run_name": config["run_name"],
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "strict_checkpoint": config["checkpoint"],
        "payload_shape": [model.latent_channels, 16, 16],
        "profile_rows": profile_rows,
        "segmentation_summary": segmentation_summary,
        "selected_test": selected_test,
        "link_latency": latency_rows,
        "scope": "Uncoded BPSK/AWGN sensitivity diagnostic with protected 64-bit quantizer metadata; not a complete UAV radio protocol.",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
