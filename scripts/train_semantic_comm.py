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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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
SEMANTIC_CONFIG_PATH = FINALLY_ROOT / "configs" / "semantic_comm_training.json"


class SemanticChannel(nn.Module):
    def __init__(self, quant_bits: int = 0, noise_std: float = 0.0, dropout_prob: float = 0.0):
        super().__init__()
        self.quant_bits = int(quant_bits)
        self.noise_std = float(noise_std)
        self.dropout_prob = float(dropout_prob)

    def fake_quantize(self, z: torch.Tensor) -> torch.Tensor:
        if self.quant_bits <= 0:
            return z
        levels = float((2 ** self.quant_bits) - 1)
        flat = z.flatten(1)
        z_min = flat.min(dim=1).values.view(-1, 1, 1, 1)
        z_max = flat.max(dim=1).values.view(-1, 1, 1, 1)
        scale = (z_max - z_min).clamp_min(1e-6)
        q = torch.round((z - z_min) / scale * levels) / levels * scale + z_min
        return z + (q - z).detach()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fake_quantize(z)
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        if self.dropout_prob > 0:
            out = F.dropout(out, p=self.dropout_prob, training=True)
        return out


class DualBranchSemanticCommNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        latent_channels: int = 16,
        quant_bits: int = 0,
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        b = int(base_channels)
        lc = int(latent_channels)
        self.latent_channels = lc
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
        self.to_semantic = nn.Conv2d(b * 8, lc, kernel_size=1)
        self.channel = SemanticChannel(quant_bits=quant_bits, noise_std=noise_std, dropout_prob=dropout_prob)
        self.from_semantic = nn.Conv2d(lc, b * 8, kernel_size=1)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, 1, kernel_size=1)
        self.last_semantic_shape: tuple[int, ...] | None = None

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
        semantic = self.to_semantic(bridge)
        self.last_semantic_shape = tuple(semantic.shape[1:])
        transmitted = self.channel(semantic)
        decoded = self.from_semantic(transmitted)
        d3 = self.up3(decoded)
        d3 = self.dec3(torch.cat([d3, f3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, f2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, f1], dim=1))
        return self.out(d1)


def semantic_stats(config: dict, variant: dict) -> dict:
    image_size = int(config["image_size"])
    latent_channels = int(config["latent_channels"])
    semantic_h = image_size // 8
    semantic_w = image_size // 8
    semantic_values = latent_channels * semantic_h * semantic_w
    input_values = 4 * image_size * image_size
    quant_bits = int(variant.get("quant_bits", 0))
    semantic_bits = semantic_values * (quant_bits if quant_bits > 0 else 32)
    input_bits = input_values * 32
    return {
        "input_values": input_values,
        "semantic_values": semantic_values,
        "value_compression_ratio": input_values / semantic_values,
        "input_bits_float32": input_bits,
        "semantic_bits": semantic_bits,
        "bit_compression_ratio": input_bits / semantic_bits,
        "semantic_shape_per_sample": [latent_channels, semantic_h, semantic_w],
    }


def make_model(config: dict, variant: dict) -> nn.Module:
    return DualBranchSemanticCommNet(
        base_channels=int(config["base_channels"]),
        latent_channels=int(config["latent_channels"]),
        quant_bits=int(variant.get("quant_bits", 0)),
        noise_std=float(variant.get("noise_std", 0.0)),
        dropout_prob=float(variant.get("dropout_prob", 0.0)),
    )


def write_combined_summary(run_dir: Path, summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        metrics = summary["test_metrics_from_best_checkpoint"]
        comm = summary["communication"]
        rows.append(
            {
                "variant": summary["variant"],
                "best_epoch": summary["best_epoch"],
                "best_val_iou": summary["best_val_iou"],
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_iou": metrics["iou"],
                "semantic_values": comm["semantic_values"],
                "value_compression_ratio": comm["value_compression_ratio"],
                "semantic_bits": comm["semantic_bits"],
                "bit_compression_ratio": comm["bit_compression_ratio"],
            }
        )
    if rows:
        with (run_dir / "combined_test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        plot_comparison(run_dir / "semantic_comm_comparison.png", rows)
    (run_dir / "combined_summary.json").write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")


def plot_comparison(path: Path, rows: list[dict]) -> None:
    labels = [row["variant"] for row in rows]
    x = range(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, [row["test_iou"] for row in rows], color="#3b82f6")
    axes[0].set_title("Final Test IoU")
    axes[0].set_xticks(list(x), labels, rotation=20)
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, [row["bit_compression_ratio"] for row in rows], color="#10b981")
    axes[1].set_title("Bit Compression Ratio")
    axes[1].set_xticks(list(x), labels, rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one_variant(
    variant: dict,
    run_dir: Path,
    protocol_config: dict,
    semantic_config: dict,
    args: argparse.Namespace,
) -> dict:
    deterministic = bool(semantic_config.get("deterministic", False))
    cudnn_benchmark = bool(semantic_config.get("cudnn_benchmark", True))
    set_seed(int(semantic_config["seed"]), deterministic=deterministic, benchmark=cudnn_benchmark)
    name = variant["name"]
    variant_dir = run_dir / name
    checkpoints_dir = variant_dir / "checkpoints"
    metrics_dir = variant_dir / "metrics"
    previews_dir = variant_dir / "previews"
    for directory in (checkpoints_dir, metrics_dir, previews_dir):
        ensure_dir(directory)

    outputs_root = Path(protocol_config["outputs_root"])
    train_ids = read_ids(outputs_root / "protocol" / "train.txt")
    val_ids = read_ids(outputs_root / "protocol" / "val.txt")
    test_ids = read_ids(outputs_root / "protocol" / "test.txt")
    batch_size = int(args.batch_size or semantic_config["batch_size"])
    epochs = int(args.epochs or semantic_config["num_epochs"])
    train_limit = int(args.train_limit)
    val_limit = int(args.val_limit)
    test_limit = int(args.test_limit)

    train_dataset, train_loader = dataloader_for(train_ids, "rgbd", "train", protocol_config, semantic_config, batch_size, True, train_limit)
    val_dataset, val_loader = dataloader_for(val_ids, "rgbd", "val", protocol_config, semantic_config, batch_size, False, val_limit)
    test_dataset, test_loader = dataloader_for(test_ids, "rgbd", "test", protocol_config, semantic_config, batch_size, False, test_limit)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_model(semantic_config, variant).to(device)
    initialized_from = ""
    if args.init_checkpoint and not args.resume:
        init_path = Path(args.init_checkpoint)
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state_dict"])
        initialized_from = str(init_path)
    pos_weight_value = compute_pos_weight(train_dataset, max_samples=int(args.pos_weight_samples))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr or semantic_config["learning_rate"]),
        weight_decay=float(semantic_config["weight_decay"]),
    )
    learning_rate = float(optimizer.param_groups[0]["lr"])
    scheduler_factor = float(semantic_config.get("scheduler_factor", 0.5))
    scheduler_patience = int(semantic_config.get("scheduler_patience", 4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience)
    threshold = float(semantic_config["threshold"])
    gradient_clip_norm = float(semantic_config.get("gradient_clip_norm", 0.0))
    communication = semantic_stats(semantic_config, variant)

    metadata = {
        "variant": name,
        "run_dir": str(variant_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "device": str(device),
        "learning_rate": learning_rate,
        "weight_decay": float(semantic_config["weight_decay"]),
        "pos_weight": pos_weight_value,
        "model": "DualBranchSemanticCommNet",
        "initialized_from": initialized_from,
        "deterministic": deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "gradient_clip_norm": gradient_clip_norm,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "variant_config": variant,
        "communication": communication,
        "early_stopping_patience": int(args.early_stopping_patience if args.early_stopping_patience >= 0 else semantic_config.get("early_stopping_patience", 0)),
        "early_stopping_min_delta": float(args.early_stopping_min_delta if args.early_stopping_min_delta >= 0 else semantic_config.get("early_stopping_min_delta", 0.0)),
    }
    (variant_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    epoch_metrics_path = metrics_dir / "epoch_metrics.csv"
    epoch_rows: list[dict] = []
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
        print(f"Resumed variant={name} from epoch {start_epoch - 1}; best_val_iou={best_val_iou:.4f}")

    patience = int(metadata["early_stopping_patience"])
    min_delta = float(metadata["early_stopping_min_delta"])
    epochs_without_improvement = 0
    started = time.time()
    print(
        f"\n=== Training semantic variant={name} epochs={epochs} train={len(train_dataset)} "
        f"val={len(val_dataset)} pos_weight={pos_weight_value:.4f} "
        f"semantic_shape={communication['semantic_shape_per_sample']} "
        f"bit_compression={communication['bit_compression_ratio']:.2f}x ==="
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
                    "variant": variant,
                    "epoch": epoch,
                    "val_metrics": asdict(val_metrics),
                    "metadata": metadata,
                    "semantic_config": semantic_config,
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
                "variant": variant,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_iou": best_val_iou,
                "metadata": metadata,
                "semantic_config": semantic_config,
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
                f"Early stopping variant={name} at epoch {epoch}; "
                f"best_epoch={best_epoch} best_val_iou={best_val_iou:.4f}"
            )
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": variant,
            "epoch": epoch_rows[-1]["epoch"] if epoch_rows else 0,
            "metadata": metadata,
            "semantic_config": semantic_config,
            "protocol_config": protocol_config,
        },
        checkpoints_dir / "last_model.pth",
    )

    best = torch.load(checkpoints_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, criterion, threshold)
    val_metrics_final = run_epoch(model, val_loader, device, criterion, threshold)
    save_prediction_previews(
        model,
        test_dataset,
        device,
        "rgbd",
        previews_dir,
        int(semantic_config["outputs"]["save_prediction_previews"]),
        threshold,
    )

    summary = {
        "variant": name,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "final_val_metrics_from_best_checkpoint": asdict(val_metrics_final),
        "test_metrics_from_best_checkpoint": asdict(test_metrics),
        "communication": communication,
        "elapsed_seconds": time.time() - started,
        "metadata": metadata,
    }
    (metrics_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RGB-D semantic communication traversability models.")
    parser.add_argument("--protocol-config", type=Path, default=PROTOCOL_CONFIG_PATH)
    parser.add_argument("--semantic-config", type=Path, default=SEMANTIC_CONFIG_PATH)
    parser.add_argument("--run-name", default="semantic_comm_run")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--pos-weight-samples", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--early-stopping-patience", type=int, default=-1)
    parser.add_argument("--early-stopping-min-delta", type=float, default=-1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--no-log-to-file", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_config = load_json(args.protocol_config)
    semantic_config = load_json(args.semantic_config)
    set_seed(
        int(semantic_config["seed"]),
        deterministic=bool(semantic_config.get("deterministic", False)),
        benchmark=bool(semantic_config.get("cudnn_benchmark", True)),
    )
    all_variants = {variant["name"]: variant for variant in semantic_config["variants"]}
    variant_names = args.variants or list(all_variants.keys())
    variants = []
    for name in variant_names:
        if name not in all_variants:
            raise ValueError(f"Unknown variant {name}. Available: {sorted(all_variants)}")
        variants.append(all_variants[name])

    run_dir = Path(semantic_config["outputs"]["root"]) / args.run_name
    ensure_dir(run_dir)
    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if not args.no_log_to_file:
        logs_dir = run_dir / "logs"
        ensure_dir(logs_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"train_{timestamp}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        sys.stdout = TeeStream(sys.__stdout__, log_handle)
        sys.stderr = TeeStream(sys.__stderr__, log_handle)
        print(f"Logging to: {log_path}")

    (run_dir / "semantic_config_snapshot.json").write_text(json.dumps(semantic_config, indent=2), encoding="utf-8")
    (run_dir / "protocol_config_snapshot.json").write_text(json.dumps(protocol_config, indent=2), encoding="utf-8")
    summaries = []
    try:
        for variant in variants:
            summaries.append(train_one_variant(variant, run_dir, protocol_config, semantic_config, args))
        write_combined_summary(run_dir, summaries)
        print(f"\nSemantic communication run finished: {run_dir}")
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()


if __name__ == "__main__":
    main()
