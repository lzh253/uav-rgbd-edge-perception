from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.segmentation import deeplabv3_resnet50
except Exception:  # pragma: no cover - handled at runtime for local env differences.
    deeplabv3_resnet50 = None

from train_baselines import (
    ConvBlock,
    TeeStream,
    compute_pos_weight,
    dataloader_for,
    ensure_dir,
    load_epoch_csv,
    load_json,
    plot_curves,
    read_ids,
    run_epoch,
    save_epoch_csv,
    save_prediction_previews,
    set_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
PROTOCOL_CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"
STRONG_BASELINE_CONFIG_PATH = FINALLY_ROOT / "configs" / "strong_baselines.json"


class DualBranchFusionNoBottleneck(nn.Module):
    """RGB-D dual-branch fusion baseline without the compact semantic bottleneck."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        b = int(base_channels)
        self.rgb_enc1 = ConvBlock(3, b)
        self.rgb_enc2 = ConvBlock(b, b * 2)
        self.rgb_enc3 = ConvBlock(b * 2, b * 4)
        self.depth_enc1 = ConvBlock(1, b)
        self.depth_enc2 = ConvBlock(b, b * 2)
        self.depth_enc3 = ConvBlock(b * 2, b * 4)
        self.fuse1 = ConvBlock(b * 2, b)
        self.fuse2 = ConvBlock(b * 4, b * 2)
        self.fuse3 = ConvBlock(b * 8, b * 4)
        self.bridge = ConvBlock(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, 1, kernel_size=1)

    def encode_branch(self, rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r1 = self.rgb_enc1(rgb)
        r2 = self.rgb_enc2(F.max_pool2d(r1, 2))
        r3 = self.rgb_enc3(F.max_pool2d(r2, 2))
        d1 = self.depth_enc1(depth)
        d2 = self.depth_enc2(F.max_pool2d(d1, 2))
        d3 = self.depth_enc3(F.max_pool2d(d2, 2))
        f1 = self.fuse1(torch.cat([r1, d1], dim=1))
        f2 = self.fuse2(torch.cat([r2, d2], dim=1))
        f3 = self.fuse3(torch.cat([r3, d3], dim=1))
        return f1, f2, f3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3]
        depth = x[:, 3:4]
        f1, f2, f3 = self.encode_branch(rgb, depth)
        bridge = self.bridge(F.max_pool2d(f3, 2))
        d3 = self.up3(bridge)
        d3 = self.dec3(torch.cat([d3, f3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, f2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, f1], dim=1))
        return self.out(d1)


class DeepLabV3BinaryWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        if deeplabv3_resnet50 is None:
            raise RuntimeError("torchvision DeepLabV3 is unavailable in this Python environment.")
        self.net = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=1, aux_loss=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]


def make_model(architecture: str, config: dict[str, Any], model_config: dict[str, Any] | None = None) -> nn.Module:
    if architecture == "dual_fusion_no_bottleneck":
        return DualBranchFusionNoBottleneck(base_channels=int(config["base_channels"]))
    if architecture == "deeplabv3_resnet50_rgb":
        return DeepLabV3BinaryWrapper()
    raise ValueError(f"Unsupported strong baseline architecture: {architecture}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def representation_stats(architecture: str, config: dict[str, Any]) -> dict[str, Any]:
    image_size = int(config["image_size"])
    if architecture == "dual_fusion_no_bottleneck":
        b = int(config["base_channels"])
        bridge_h = image_size // 8
        bridge_w = image_size // 8
        bridge_channels = b * 8
        bridge_values = bridge_channels * bridge_h * bridge_w
        semantic_values = 16 * bridge_h * bridge_w
        input_values = 4 * image_size * image_size
        return {
            "input_values_rgbd": input_values,
            "uncompressed_bridge_shape": [bridge_channels, bridge_h, bridge_w],
            "uncompressed_bridge_values": bridge_values,
            "reference_semantic_bottleneck_values_16x16x16": semantic_values,
            "bridge_to_semantic_value_ratio": bridge_values / semantic_values,
            "input_to_bridge_value_ratio": input_values / bridge_values,
        }
    if architecture == "deeplabv3_resnet50_rgb":
        return {
            "input_values_rgb": 3 * image_size * image_size,
            "note": "DeepLabV3 is an RGB architecture baseline and does not define a transmitted compact semantic tensor.",
        }
    return {}


def write_combined_summary(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        metrics = summary["test_metrics_from_best_checkpoint"]
        params = summary["parameter_count"]
        rows.append(
            {
                "name": summary["name"],
                "architecture": summary["architecture"],
                "mode": summary["mode"],
                "best_epoch": summary["best_epoch"],
                "best_val_iou": summary["best_val_iou"],
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_iou": metrics["iou"],
                "test_specificity": metrics["specificity"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "trainable_parameters": params["trainable_parameters"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    if rows:
        with (run_dir / "combined_test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (run_dir / "combined_summary.json").write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")


def train_one_model(
    model_cfg: dict[str, Any],
    run_dir: Path,
    protocol_config: dict[str, Any],
    strong_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed = int(args.seed if args.seed >= 0 else strong_config["seed"])
    set_seed(
        seed,
        deterministic=bool(strong_config.get("deterministic", False)),
        benchmark=bool(strong_config.get("cudnn_benchmark", True)),
    )
    name = str(model_cfg["name"])
    architecture = str(model_cfg["architecture"])
    mode = str(model_cfg["mode"])
    model_dir = run_dir / name
    checkpoints_dir = model_dir / "checkpoints"
    metrics_dir = model_dir / "metrics"
    previews_dir = model_dir / "previews"
    for directory in (checkpoints_dir, metrics_dir, previews_dir):
        ensure_dir(directory)

    outputs_root = Path(protocol_config["outputs_root"])
    train_ids = read_ids(outputs_root / "protocol" / "train.txt")
    val_ids = read_ids(outputs_root / "protocol" / "val.txt")
    test_ids = read_ids(outputs_root / "protocol" / "test.txt")
    batch_size = int(args.batch_size or model_cfg.get("batch_size", strong_config["batch_size"]))
    epochs = int(args.epochs or model_cfg.get("num_epochs", strong_config["num_epochs"]))
    learning_rate = float(args.lr or model_cfg.get("learning_rate", strong_config["learning_rate"]))
    train_limit = int(args.train_limit)
    val_limit = int(args.val_limit)
    test_limit = int(args.test_limit)

    train_dataset, train_loader = dataloader_for(train_ids, mode, "train", protocol_config, strong_config, batch_size, True, train_limit)
    val_dataset, val_loader = dataloader_for(val_ids, mode, "val", protocol_config, strong_config, batch_size, False, val_limit)
    test_dataset, test_loader = dataloader_for(test_ids, mode, "test", protocol_config, strong_config, batch_size, False, test_limit)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_model(architecture, strong_config, model_cfg).to(device)
    param_count = count_parameters(model)
    pos_weight_value = compute_pos_weight(train_dataset, max_samples=int(args.pos_weight_samples))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(strong_config["weight_decay"]),
    )
    scheduler_factor = float(strong_config.get("scheduler_factor", 0.5))
    scheduler_patience = int(strong_config.get("scheduler_patience", 4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience)
    threshold = float(strong_config["threshold"])
    gradient_clip_norm = float(args.gradient_clip_norm if args.gradient_clip_norm >= 0 else strong_config.get("gradient_clip_norm", 0.0))

    metadata = {
        "name": name,
        "architecture": architecture,
        "mode": mode,
        "run_dir": str(model_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "device": str(device),
        "pos_weight": pos_weight_value,
        "parameter_count": param_count,
        "representation": representation_stats(architecture, strong_config),
        "deterministic": bool(strong_config.get("deterministic", False)),
        "cudnn_benchmark": bool(strong_config.get("cudnn_benchmark", True)),
        "gradient_clip_norm": gradient_clip_norm,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "early_stopping_patience": int(args.early_stopping_patience if args.early_stopping_patience >= 0 else strong_config.get("early_stopping_patience", 0)),
        "early_stopping_min_delta": float(args.early_stopping_min_delta if args.early_stopping_min_delta >= 0 else strong_config.get("early_stopping_min_delta", 0.0)),
        "evaluation_scope": "Architecture-capacity and deployment-complexity controls.",
    }
    (model_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    epoch_metrics_path = metrics_dir / "epoch_metrics.csv"
    epoch_rows: list[dict[str, Any]] = []
    best_val_iou = -math.inf
    best_epoch = 0
    start_epoch = 1
    latest_checkpoint = checkpoints_dir / "latest_model.pth"
    if args.resume and latest_checkpoint.exists():
        ckpt = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        epoch_rows = load_epoch_csv(epoch_metrics_path)
        if epoch_rows:
            best_row = max(epoch_rows, key=lambda row: float(row["val_iou"]))
            best_val_iou = float(best_row["val_iou"])
            best_epoch = int(best_row["epoch"])
        print(f"Resumed name={name} from epoch {start_epoch - 1}; best_val_iou={best_val_iou:.4f}")

    patience = int(metadata["early_stopping_patience"])
    min_delta = float(metadata["early_stopping_min_delta"])
    epochs_without_improvement = 0
    started = time.time()
    print(
        f"\n=== Training strong baseline name={name} arch={architecture} "
        f"mode={mode} epochs={epochs} train={len(train_dataset)} val={len(val_dataset)} "
        f"params={param_count['trainable_parameters']} pos_weight={pos_weight_value:.4f} ==="
    )
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, threshold, optimizer, gradient_clip_norm)
        val_metrics = run_epoch(model, val_loader, device, criterion, threshold)
        scheduler.step(val_metrics.iou)
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            **{f"train_{k}": v for k, v in asdict(train_metrics).items()},
            **{f"val_{k}": v for k, v in asdict(val_metrics).items()},
        }
        epoch_rows.append(row)
        save_epoch_csv(epoch_metrics_path, epoch_rows)
        plot_curves(metrics_dir / "training_curves.png", epoch_rows)
        improved = val_metrics.iou > best_val_iou + min_delta
        if improved:
            best_val_iou = val_metrics.iou
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "name": name,
                    "architecture": architecture,
                    "mode": mode,
                    "epoch": epoch,
                    "val_metrics": asdict(val_metrics),
                    "metadata": metadata,
                    "model_config": model_cfg,
                    "strong_baseline_config": strong_config,
                    "protocol_config": protocol_config,
                },
                checkpoints_dir / "best_model.pth",
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "name": name,
                "architecture": architecture,
                "mode": mode,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_iou": best_val_iou,
                "metadata": metadata,
                "model_config": model_cfg,
                "strong_baseline_config": strong_config,
                "protocol_config": protocol_config,
            },
            latest_checkpoint,
        )
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_iou={val_metrics.iou:.4f} val_f1={val_metrics.f1:.4f}"
        )
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping name={name} at epoch {epoch}; "
                f"best_epoch={best_epoch} best_val_iou={best_val_iou:.4f}"
            )
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "name": name,
            "architecture": architecture,
            "mode": mode,
            "epoch": epoch_rows[-1]["epoch"] if epoch_rows else 0,
            "metadata": metadata,
            "model_config": model_cfg,
            "strong_baseline_config": strong_config,
            "protocol_config": protocol_config,
        },
        checkpoints_dir / "last_model.pth",
    )

    best = torch.load(checkpoints_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, criterion, threshold)
    val_metrics_final = run_epoch(model, val_loader, device, criterion, threshold)
    save_prediction_previews(model, test_dataset, device, mode, previews_dir, int(strong_config["outputs"]["save_prediction_previews"]), threshold)

    summary = {
        "name": name,
        "architecture": architecture,
        "mode": mode,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "final_val_metrics_from_best_checkpoint": asdict(val_metrics_final),
        "test_metrics_from_best_checkpoint": asdict(test_metrics),
        "elapsed_seconds": time.time() - started,
        "metadata": metadata,
        "parameter_count": param_count,
        "representation": metadata["representation"],
    }
    best_path = checkpoints_dir / "best_model.pth"
    if best_path.exists():
        summary["best_checkpoint_size_mb"] = best_path.stat().st_size / (1024 * 1024)
    (metrics_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controlled strong baselines.")
    parser.add_argument("--protocol-config", type=Path, default=PROTOCOL_CONFIG_PATH)
    parser.add_argument("--config", type=Path, default=STRONG_BASELINE_CONFIG_PATH)
    parser.add_argument("--run-name", default="strong_baselines_v1")
    parser.add_argument("--models", nargs="+", default=None, help="Model names to run from the config.")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--pos-weight-samples", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--gradient-clip-norm", type=float, default=-1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=-1, help="Use -1 to read from config; 0 disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=-1.0, help="Use -1 to read from config.")
    parser.add_argument("--resume", action="store_true", help="Resume each model from checkpoints/latest_model.pth when available.")
    parser.add_argument("--no-log-to-file", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_config = load_json(args.protocol_config)
    strong_config = load_json(args.config)
    set_seed(
        int(args.seed if args.seed >= 0 else strong_config["seed"]),
        deterministic=bool(strong_config.get("deterministic", False)),
        benchmark=bool(strong_config.get("cudnn_benchmark", True)),
    )
    selected = set(args.models or [])
    model_cfgs = [m for m in strong_config["models"] if not selected or str(m["name"]) in selected]
    if selected:
        found = {str(m["name"]) for m in model_cfgs}
        missing = selected - found
        if missing:
            raise KeyError(f"Requested model(s) not found in config: {sorted(missing)}")
    run_dir = Path(strong_config["outputs"]["root"]) / args.run_name
    ensure_dir(run_dir)
    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if not args.no_log_to_file:
        logs_dir = run_dir / "logs"
        ensure_dir(logs_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"train_strong_baselines_{timestamp}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        sys.stdout = TeeStream(sys.__stdout__, log_handle)
        sys.stderr = TeeStream(sys.__stderr__, log_handle)
        print(f"Logging to: {log_path}")
    (run_dir / "strong_baseline_config_snapshot.json").write_text(json.dumps(strong_config, indent=2), encoding="utf-8")
    (run_dir / "protocol_config_snapshot.json").write_text(json.dumps(protocol_config, indent=2), encoding="utf-8")

    summaries = []
    for model_cfg in model_cfgs:
        summaries.append(train_one_model(model_cfg, run_dir, protocol_config, strong_config, args))
    write_combined_summary(run_dir, summaries)
    print(f"\nStrong baseline run finished: {run_dir}")
    if log_handle is not None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
