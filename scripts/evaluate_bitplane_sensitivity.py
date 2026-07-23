from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ELECTRONICS_ROOT = SCRIPT_DIR.parent
FINALLY_ROOT = ELECTRONICS_ROOT
CORE_SCRIPTS = FINALLY_ROOT / "scripts"
sys.path.insert(0, str(CORE_SCRIPTS))
sys.path.insert(0, str(SCRIPT_DIR))

from train_baselines import TraversabilityDataset, load_json, read_ids, set_seed  # noqa: E402
from train_strict_bottleneck import StrictRGBDBottleneckNet  # noqa: E402


DEFAULT_CONFIG_PATH = ELECTRONICS_ROOT / "configs" / "digital_link_evaluation.json"
DEFAULT_OUTPUT_ROOT = ELECTRONICS_ROOT / "experiments" / "bitplane_sensitivity_v1"


def update_counts(prob: np.ndarray, target: np.ndarray, counts: dict[str, int]) -> None:
    pred = prob >= 0.5
    label = target.astype(bool)
    counts["tp"] += int(np.logical_and(pred, label).sum())
    counts["fp"] += int(np.logical_and(pred, ~label).sum())
    counts["tn"] += int(np.logical_and(~pred, ~label).sum())
    counts["fn"] += int(np.logical_and(~pred, label).sum())


def finalize(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / max(1, tp + fp + fn),
    }


def quantize(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = z.flatten(1)
    z_min = flat.min(dim=1).values[:, None, None, None]
    z_max = flat.max(dim=1).values[:, None, None, None]
    scale = (z_max - z_min).clamp_min(1e-6)
    q = torch.round((z - z_min) / scale * 255.0).to(torch.int64)
    return q, z_min, scale


def dequantize(q: torch.Tensor, z_min: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) / 255.0 * scale + z_min


def load_model(config: dict, device: torch.device) -> StrictRGBDBottleneckNet:
    checkpoint = torch.load(Path(config["checkpoint"]), map_location=device, weights_only=False)
    model = StrictRGBDBottleneckNet(
        base_channels=int(checkpoint["config"]["base_channels"]),
        latent_channels=int(checkpoint["variant"]["latent_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate which quantized latent bit planes are most task-sensitive."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--flip-probability", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.config.resolve()
    config = load_json(config_path)
    protocol = load_json(Path(config["protocol_config"]))
    training = load_json(Path(config["training_config"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)

    test_ids = read_ids(Path(protocol["outputs_root"]) / "protocol" / "test.txt")
    dataset = TraversabilityDataset(test_ids, "rgbd", protocol, training, "test")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    flip_probability = float(args.flip_probability)
    repeats = int(args.repeats)
    if not 0.0 < flip_probability < 1.0:
        raise ValueError("--flip-probability must be in (0, 1)")
    if repeats < 1:
        raise ValueError("--repeats must be positive")
    profiles: list[tuple[str, int | None]] = [("ideal_q8", None)]
    profiles.extend((f"bit_{bit}", bit) for bit in range(7, -1, -1))

    repeat_rows: list[dict] = []
    with torch.inference_mode():
        for profile_index, (profile, bit) in enumerate(profiles):
            profile_repeats = 1 if bit is None else repeats
            for repeat in range(profile_repeats):
                set_seed(20260723 + profile_index * 101 + repeat, deterministic=True, benchmark=False)
                counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                flipped = 0
                source_bits = 0
                for x, target, _ in loader:
                    x = x.to(device, non_blocking=True)
                    z = model.encode(x)
                    q, z_min, scale = quantize(z)
                    q_hat = q
                    if bit is not None:
                        flip_mask = torch.rand(q.shape, device=device) < flip_probability
                        q_hat = torch.bitwise_xor(q, flip_mask.to(torch.int64) << bit)
                        flipped += int(flip_mask.sum().item())
                        source_bits += int(q.numel() * 8)
                    prob = torch.sigmoid(model.decode(dequantize(q_hat, z_min, scale)))
                    update_counts(prob.cpu().numpy(), target.numpy(), counts)
                metrics = finalize(counts)
                repeat_rows.append(
                    {
                        "profile": profile,
                        "bit_index": "" if bit is None else bit,
                        "bit_weight": "" if bit is None else 2**bit,
                        "conditional_flip_probability": 0.0 if bit is None else flip_probability,
                        "payload_ber": 0.0 if bit is None else flipped / max(1, source_bits),
                        "repeat": repeat,
                        **metrics,
                    }
                )

    summary_rows: list[dict] = []
    for profile, bit in profiles:
        group = [row for row in repeat_rows if row["profile"] == profile]
        summary_rows.append(
            {
                "profile": profile,
                "bit_index": "" if bit is None else bit,
                "bit_weight": "" if bit is None else 2**bit,
                "conditional_flip_probability": 0.0 if bit is None else flip_probability,
                "payload_ber_mean": float(np.mean([float(row["payload_ber"]) for row in group])),
                "iou_mean": float(np.mean([float(row["iou"]) for row in group])),
                "iou_std": float(np.std([float(row["iou"]) for row in group], ddof=0)),
                "f1_mean": float(np.mean([float(row["f1"]) for row in group])),
                "f1_std": float(np.std([float(row["f1"]) for row in group], ddof=0)),
            }
        )

    write_csv(output_root / "bitplane_repeat.csv", repeat_rows)
    write_csv(output_root / "bitplane_summary.csv", summary_rows)
    (output_root / "config_snapshot.json").write_text(
        json.dumps(
            {
                "source_config": str(config_path),
                "checkpoint": config["checkpoint"],
                "split": "full_test_648",
                "conditional_flip_probability": flip_probability,
                "repeats": repeats,
                "threshold": 0.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ideal_iou = float(summary_rows[0]["iou_mean"])
    bit_rows = summary_rows[1:]
    labels = [f"b{row['bit_index']}\n({row['bit_weight']})" for row in bit_rows]
    iou_values = [float(row["iou_mean"]) for row in bit_rows]
    iou_std = [float(row["iou_std"]) for row in bit_rows]
    degradation = [ideal_iou - value for value in iou_values]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))
    axes[0].bar(labels, iou_values, yerr=iou_std, capsize=3, color="#4C78A8")
    axes[0].axhline(ideal_iou, color="#D14A61", ls="--", lw=1.5)
    axes[0].set_ylabel("Held-out test IoU")
    axes[0].set_xlabel("Corrupted bit plane (decimal weight)")
    axes[0].set_ylim(min(iou_values) - 0.025, ideal_iou + 0.015)
    axes[0].set_title(
        f"(a) Equal-probability bit-plane corruption\n"
        f"(dashed line: ideal q8 = {ideal_iou:.3f})"
    )
    axes[1].bar(labels, degradation, color="#F28E2B")
    axes[1].set_ylabel("IoU loss from ideal q8")
    axes[1].set_xlabel("Corrupted bit plane (decimal weight)")
    axes[1].set_title("(b) Bit significance for the task")
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    payload_ber_percent = flip_probability / 8.0 * 100.0
    fig.suptitle(
        f"One selected bit plane flipped with {flip_probability * 100:.1f}% "
        f"conditional probability ({payload_ber_percent:.3f}% payload BER), "
        f"{repeats} repeats",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_root / "bitplane_sensitivity.png", dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
