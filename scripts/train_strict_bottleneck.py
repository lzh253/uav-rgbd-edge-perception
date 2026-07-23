from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
CORE_SCRIPTS = FINALLY_ROOT / "scripts"
sys.path.insert(0, str(CORE_SCRIPTS))

from train_baselines import (  # noqa: E402
    ConvBlock,
    compute_pos_weight,
    dataloader_for,
    ensure_dir,
    load_json,
    plot_curves,
    read_ids,
    run_epoch,
    save_epoch_csv,
    save_prediction_previews,
    set_seed,
)


class IdentityChannel(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class StrictRGBDBottleneckNet(nn.Module):
    """RGB-D encoder/decoder whose only cross-boundary tensor is the bottleneck."""

    def __init__(self, base_channels: int = 32, latent_channels: int = 16):
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
        self.fuse3 = ConvBlock(b * 8, b * 4)
        self.bridge = ConvBlock(b * 4, b * 8)
        self.to_semantic = nn.Conv2d(b * 8, lc, kernel_size=1)

        self.channel: nn.Module = IdentityChannel()
        self.from_semantic = ConvBlock(lc, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 4, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 2, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b, b)
        self.out = nn.Conv2d(b, 1, kernel_size=1)
        self.last_semantic_shape: tuple[int, ...] | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3]
        depth = x[:, 3:4]
        r1 = self.rgb_enc1(rgb)
        r2 = self.rgb_enc2(F.max_pool2d(r1, 2))
        r3 = self.rgb_enc3(F.max_pool2d(r2, 2))
        d1 = self.depth_enc1(depth)
        d2 = self.depth_enc2(F.max_pool2d(d1, 2))
        d3 = self.depth_enc3(F.max_pool2d(d2, 2))
        f3 = self.fuse3(torch.cat([r3, d3], dim=1))
        bridge = self.bridge(F.max_pool2d(f3, 2))
        return self.to_semantic(bridge)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        d3 = self.up3(self.from_semantic(z))
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self.dec1(d1)
        return self.out(d1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        semantic = self.encode(x)
        self.last_semantic_shape = tuple(semantic.shape[1:])
        return self.decode(self.channel(semantic))


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=dims)
        denominator = prob.sum(dim=dims) + target.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return bce + self.dice_weight * dice_loss


class JointGeometricAugmentation(Dataset):
    """Apply label-preserving aerial-view transforms to RGB, depth, and mask jointly."""

    def __init__(
        self,
        dataset: Dataset,
        horizontal_flip_probability: float,
        vertical_flip_probability: float,
        random_rot90: bool,
    ):
        self.dataset = dataset
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.vertical_flip_probability = float(vertical_flip_probability)
        self.random_rot90 = bool(random_rot90)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y, sample_id = self.dataset[index]
        if self.random_rot90:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=(-2, -1))
                y = torch.rot90(y, k, dims=(-2, -1))
        if torch.rand(1).item() < self.horizontal_flip_probability:
            x = torch.flip(x, dims=(-1,))
            y = torch.flip(y, dims=(-1,))
        if torch.rand(1).item() < self.vertical_flip_probability:
            x = torch.flip(x, dims=(-2,))
            y = torch.flip(y, dims=(-2,))
        return x.contiguous(), y.contiguous(), sample_id


def payload_stats(image_size: int, latent_channels: int) -> dict[str, float | int | list[int]]:
    spatial = image_size // 8
    values = latent_channels * spatial * spatial
    rgbd_float32_bits = 4 * image_size * image_size * 32
    rgb8_depth16_bits = image_size * image_size * (3 * 8 + 16)
    q8_bits = values * 8 + 64
    return {
        "semantic_shape": [latent_channels, spatial, spatial],
        "semantic_values": values,
        "float32_payload_bits": values * 32,
        "q8_payload_bits_including_minmax": q8_bits,
        "compression_vs_rgbd_float32": rgbd_float32_bits / q8_bits,
        "compression_vs_rgb8_depth16": rgb8_depth16_bits / q8_bits,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_criterion(loss_name: str, pos_weight: float, dice_weight: float, device: torch.device) -> nn.Module:
    weight = torch.tensor([pos_weight], device=device)
    if loss_name == "bce_dice":
        return BCEDiceLoss(weight, dice_weight=dice_weight)
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    raise ValueError(f"Unsupported loss: {loss_name}")


def plot_summary(path: Path, rows: list[dict]) -> None:
    labels = [str(row["variant"]) for row in rows]
    x = range(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(x, [float(row["test_iou"]) for row in rows], color="#2563eb")
    axes[0].set_title("Held-out test IoU")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, [float(row["test_f1"]) for row in rows], color="#059669")
    axes[1].set_title("Held-out test F1")
    axes[1].set_ylim(0, 1)
    axes[2].bar(x, [float(row["q8_payload_kbit"]) for row in rows], color="#ea580c")
    axes[2].set_title("8-bit payload (kbit)")
    for axis in axes:
        axis.set_xticks(list(x), labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def train_variant(
    variant: dict,
    config: dict,
    protocol_config: dict,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    name = str(variant["name"])
    seed = int(variant.get("seed", config["seed"]))
    deterministic = bool(config.get("deterministic", True))
    set_seed(seed, deterministic=deterministic, benchmark=not deterministic)

    variant_dir = output_root / name
    checkpoints_dir = variant_dir / "checkpoints"
    metrics_dir = variant_dir / "metrics"
    previews_dir = variant_dir / "previews"
    for path in (checkpoints_dir, metrics_dir, previews_dir):
        ensure_dir(path)

    outputs_root = Path(protocol_config["outputs_root"])
    train_ids = read_ids(outputs_root / "protocol" / "train.txt")
    val_ids = read_ids(outputs_root / "protocol" / "val.txt")
    test_ids = read_ids(outputs_root / "protocol" / "test.txt")
    batch_size = int(args.batch_size or config["batch_size"])
    epochs = int(args.epochs or config["num_epochs"])
    train_dataset, train_loader = dataloader_for(
        train_ids, "rgbd", "train", protocol_config, config, batch_size, True, int(args.train_limit)
    )
    val_dataset, val_loader = dataloader_for(
        val_ids, "rgbd", "val", protocol_config, config, batch_size, False, int(args.val_limit)
    )
    test_dataset, test_loader = dataloader_for(
        test_ids, "rgbd", "test", protocol_config, config, batch_size, False, int(args.test_limit)
    )

    pos_weight = compute_pos_weight(train_dataset, max_samples=int(config.get("pos_weight_samples", 0)))
    augmentation = config.get("augmentation", {})
    if bool(augmentation.get("enabled", False)):
        train_dataset = JointGeometricAugmentation(
            train_dataset,
            horizontal_flip_probability=float(augmentation.get("horizontal_flip_probability", 0.5)),
            vertical_flip_probability=float(augmentation.get("vertical_flip_probability", 0.5)),
            random_rot90=bool(augmentation.get("random_rot90", True)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(config["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    latent_channels = int(variant["latent_channels"])
    model = StrictRGBDBottleneckNet(
        base_channels=int(config["base_channels"]),
        latent_channels=latent_channels,
    ).to(device)
    loss_name = str(variant.get("loss", config["loss"]))
    criterion = build_criterion(loss_name, pos_weight, float(config.get("dice_weight", 0.5)), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant.get("learning_rate", config["learning_rate"])),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config.get("scheduler_factor", 0.5)),
        patience=int(config.get("scheduler_patience", 4)),
    )

    threshold = float(config["threshold"])
    gradient_clip = float(config.get("gradient_clip_norm", 1.0))
    patience = int(config.get("early_stopping_patience", 14))
    min_delta = float(config.get("early_stopping_min_delta", 0.001))
    best_val_iou = -math.inf
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict] = []
    start = time.time()

    metadata = {
        "variant": name,
        "seed": seed,
        "architecture": "StrictRGBDBottleneckNet_no_skip_features",
        "base_channels": int(config["base_channels"]),
        "latent_channels": latent_channels,
        "loss": loss_name,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "device": str(device),
        "payload": payload_stats(int(config["image_size"]), latent_channels),
        "cross_boundary_tensors": ["semantic_bottleneck_only"],
    }
    (variant_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Training {name}: latent={latent_channels}, seed={seed}, device={device}, loss={loss_name}")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, threshold, optimizer, gradient_clip)
        val_metrics = run_epoch(model, val_loader, device, criterion, threshold)
        scheduler.step(val_metrics.iou)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
            **{f"val_{key}": value for key, value in asdict(val_metrics).items()},
        }
        rows.append(row)
        save_epoch_csv(metrics_dir / "epoch_metrics.csv", rows)
        plot_curves(metrics_dir / "training_curves.png", rows)

        improved = val_metrics.iou > best_val_iou + min_delta
        if improved:
            best_val_iou = val_metrics.iou
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": asdict(val_metrics),
                    "variant": variant,
                    "config": config,
                    "protocol_config": protocol_config,
                    "metadata": metadata,
                },
                checkpoints_dir / "best_model.pth",
            )
        else:
            stale_epochs += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_iou": best_val_iou,
                "variant": variant,
                "config": config,
                "protocol_config": protocol_config,
                "metadata": metadata,
            },
            checkpoints_dir / "latest_model.pth",
        )
        print(
            f"{name} epoch {epoch:03d}/{epochs}: "
            f"train_loss={train_metrics.loss:.4f}, val_iou={val_metrics.iou:.4f}, "
            f"val_f1={val_metrics.f1:.4f}"
        )
        if patience > 0 and stale_epochs >= patience:
            print(f"{name} early stop: best epoch {best_epoch}, val IoU {best_val_iou:.4f}")
            break

    best = torch.load(checkpoints_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, criterion, threshold)
    final_val_metrics = run_epoch(model, val_loader, device, criterion, threshold)
    save_prediction_previews(
        model,
        test_dataset,
        device,
        "rgbd",
        previews_dir,
        int(config.get("preview_count", 8)),
        threshold,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": best_epoch,
            "variant": variant,
            "config": config,
            "protocol_config": protocol_config,
            "metadata": metadata,
        },
        checkpoints_dir / "validation_selected_model.pth",
    )

    payload = payload_stats(int(config["image_size"]), latent_channels)
    summary = {
        "variant": name,
        "seed": seed,
        "latent_channels": latent_channels,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "val_iou": final_val_metrics.iou,
        "val_f1": final_val_metrics.f1,
        "test_iou": test_metrics.iou,
        "test_f1": test_metrics.f1,
        "test_precision": test_metrics.precision,
        "test_recall": test_metrics.recall,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": (checkpoints_dir / "validation_selected_model.pth").stat().st_size,
        "q8_payload_kbit": float(payload["q8_payload_bits_including_minmax"]) / 1000.0,
        "elapsed_seconds": time.time() - start,
        **{f"payload_{key}": value for key, value in payload.items()},
    }
    (metrics_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strict no-skip RGB-D bottleneck models.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    protocol_config = load_json(Path(config["protocol_config"]))
    output_root = Path(config["output_root"])
    ensure_dir(output_root)
    (output_root / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    variants = list(config["variants"])
    if args.variants:
        wanted = set(args.variants)
        variants = [variant for variant in variants if variant["name"] in wanted]
        missing = wanted - {variant["name"] for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    summaries = []
    for variant in variants:
        summaries.append(train_variant(variant, config, protocol_config, output_root, args))
    write_csv(output_root / "strict_bottleneck_summary.csv", summaries)
    (output_root / "strict_bottleneck_summary.json").write_text(
        json.dumps({"summaries": summaries}, indent=2),
        encoding="utf-8",
    )
    plot_summary(output_root / "strict_bottleneck_summary.png", summaries)
    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
