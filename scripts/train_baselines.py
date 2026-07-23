from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
PROTOCOL_CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"
BASELINE_CONFIG_PATH = FINALLY_ROOT / "configs" / "baseline_training.json"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    iou: float
    specificity: float
    balanced_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int, deterministic: bool = False, benchmark: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = bool(benchmark)
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_depth(depth: np.ndarray, lower: float, upper: float) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() <= 10:
        return np.zeros(depth.shape, dtype=np.float32)
    lo = float(np.percentile(depth[valid], lower))
    hi = float(np.percentile(depth[valid], upper))
    if abs(hi - lo) < 1e-12:
        hi = lo + 1e-6
    out = ((depth - lo) / (hi - lo)).clip(0.0, 1.0)
    out[~valid] = 0.0
    return out.astype(np.float32)


class TraversabilityDataset(Dataset):
    def __init__(
        self,
        sample_ids: list[str],
        mode: str,
        protocol_config: dict,
        baseline_config: dict,
        split: str,
    ):
        if mode not in {"rgb", "depth", "rgbd"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.sample_ids = sample_ids
        self.mode = mode
        self.split = split
        self.protocol_config = protocol_config
        self.baseline_config = baseline_config
        self.dataset_root = Path(protocol_config["dataset_root"])
        self.rgb_dir = self.dataset_root / protocol_config["rgb_dir"]
        self.label_dir = self.dataset_root / protocol_config["label_dir"]
        outputs_root = Path(protocol_config["outputs_root"])
        depth_cfg = protocol_config["depth_generation"]
        self.depth_dir = outputs_root / depth_cfg["output_dir"] / depth_cfg["npy_dir"]
        self.passable_ids = [int(x) for x in protocol_config["passable_class_ids"]]
        self.image_size = int(baseline_config["image_size"])
        depth_norm = baseline_config["depth_normalization"]
        self.depth_lower = float(depth_norm["lower_percentile"])
        self.depth_upper = float(depth_norm["upper_percentile"])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_rgb(self, sample_id: str) -> np.ndarray:
        path = self.rgb_dir / f"{sample_id}.jpg"
        rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        return np.transpose(rgb, (2, 0, 1))

    def _load_depth(self, sample_id: str) -> np.ndarray:
        path = self.depth_dir / f"{sample_id}.npy"
        depth = np.load(path)
        depth = normalize_depth(depth, self.depth_lower, self.depth_upper)
        depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return depth[None, :, :].astype(np.float32)

    def _load_mask(self, sample_id: str) -> np.ndarray:
        path = self.label_dir / f"{sample_id}.png"
        label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(path)
        if label.ndim == 3:
            label = label[:, :, 0]
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros(label.shape, dtype=np.float32)
        for class_id in self.passable_ids:
            mask[label == class_id] = 1.0
        return mask[None, :, :]

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        parts = []
        if self.mode in {"rgb", "rgbd"}:
            parts.append(self._load_rgb(sample_id))
        if self.mode in {"depth", "rgbd"}:
            parts.append(self._load_depth(sample_id))
        x = np.concatenate(parts, axis=0)
        y = self._load_mask(sample_id)
        return torch.from_numpy(x), torch.from_numpy(y), sample_id


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bridge = ConvBlock(b * 4, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        bridge = self.bridge(F.max_pool2d(e3, 2))
        d3 = self.up3(bridge)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def make_model(mode: str, base_channels: int) -> nn.Module:
    in_channels = {"rgb": 3, "depth": 1, "rgbd": 4}[mode]
    return SmallUNet(in_channels=in_channels, base_channels=base_channels)


def compute_pos_weight(dataset: Dataset, max_samples: int = 0) -> float:
    total_pos = 0.0
    total_pixels = 0.0
    indices = range(len(dataset))
    if max_samples and max_samples > 0:
        indices = range(min(max_samples, len(dataset)))
    for idx in indices:
        _x, y, _sample_id = dataset[idx]
        pos = float(y.sum().item())
        total_pos += pos
        total_pixels += float(y.numel())
    total_neg = total_pixels - total_pos
    if total_pos <= 0:
        return 1.0
    return float(total_neg / total_pos)


def update_confusion_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float, counts: dict[str, int]) -> None:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    labels = targets >= 0.5
    counts["tp"] += int((preds & labels).sum().item())
    counts["fp"] += int((preds & ~labels).sum().item())
    counts["tn"] += int((~preds & ~labels).sum().item())
    counts["fn"] += int((~preds & labels).sum().item())


def metrics_from_counts(loss: float, counts: dict[str, int]) -> Metrics:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    eps = 1e-9
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    balanced_accuracy = 0.5 * (recall + specificity)
    return Metrics(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 0.0,
) -> Metrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    batches = 0
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for x, y, _sample_ids in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm and gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        update_confusion_from_logits(logits.detach(), y, threshold, counts)

    return metrics_from_counts(total_loss / max(1, batches), counts)


def save_epoch_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_epoch_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(float(value))
                else:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
            rows.append(parsed)
    return rows


def plot_curves(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0, 0].plot(epochs, [row["val_loss"] for row in rows], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [row["val_iou"] for row in rows], label="val IoU")
    axes[0, 1].plot(epochs, [row["val_f1"] for row in rows], label="val F1")
    axes[0, 1].set_title("Val IoU/F1")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [row["val_precision"] for row in rows], label="precision")
    axes[1, 0].plot(epochs, [row["val_recall"] for row in rows], label="recall")
    axes[1, 0].set_title("Val Precision/Recall")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, [row["learning_rate"] for row in rows])
    axes[1, 1].set_title("Learning Rate")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tensor_rgb_for_preview(x: torch.Tensor, mode: str) -> np.ndarray:
    if mode in {"rgb", "rgbd"}:
        rgb = x[:3].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb * std + mean).clip(0, 1)
        return (rgb * 255).astype(np.uint8)
    depth = x[0].cpu().numpy()
    return (plt.get_cmap("Spectral")(1 - depth)[..., :3] * 255).astype(np.uint8)


def save_prediction_previews(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    mode: str,
    out_dir: Path,
    max_items: int,
    threshold: float,
) -> None:
    ensure_dir(out_dir)
    model.eval()
    n = min(max_items, len(dataset))
    for idx in range(n):
        x, y, sample_id = dataset[idx]
        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(device).float())
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        rgb = tensor_rgb_for_preview(x, mode)
        target = y[0].cpu().numpy()
        pred = (prob >= threshold).astype(np.float32)
        err = np.abs(pred - target)

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(rgb)
        axes[0].set_title(f"{sample_id} input")
        axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("target road mask")
        axes[2].imshow(prob, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("pred probability")
        axes[3].imshow(err, cmap="hot", vmin=0, vmax=1)
        axes[3].set_title("binary error")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / f"{sample_id}_{mode}_prediction.png", dpi=150)
        plt.close(fig)


def dataloader_for(
    sample_ids: list[str],
    mode: str,
    split: str,
    protocol_config: dict,
    baseline_config: dict,
    batch_size: int,
    shuffle: bool,
    limit: int = 0,
) -> tuple[Dataset, DataLoader]:
    dataset = TraversabilityDataset(sample_ids, mode, protocol_config, baseline_config, split)
    if limit and limit > 0:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(baseline_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def train_one_mode(
    mode: str,
    run_dir: Path,
    protocol_config: dict,
    baseline_config: dict,
    args: argparse.Namespace,
) -> dict:
    deterministic = bool(baseline_config.get("deterministic", False))
    cudnn_benchmark = bool(baseline_config.get("cudnn_benchmark", True))
    set_seed(int(baseline_config["seed"]), deterministic=deterministic, benchmark=cudnn_benchmark)
    mode_dir = run_dir / mode
    checkpoints_dir = mode_dir / "checkpoints"
    metrics_dir = mode_dir / "metrics"
    previews_dir = mode_dir / "previews"
    for directory in (checkpoints_dir, metrics_dir, previews_dir):
        ensure_dir(directory)

    outputs_root = Path(protocol_config["outputs_root"])
    train_ids = read_ids(outputs_root / "protocol" / "train.txt")
    val_ids = read_ids(outputs_root / "protocol" / "val.txt")
    test_ids = read_ids(outputs_root / "protocol" / "test.txt")
    batch_size = int(args.batch_size or baseline_config["batch_size"])
    epochs = int(args.epochs or baseline_config["num_epochs"])
    train_limit = int(args.train_limit)
    val_limit = int(args.val_limit)
    test_limit = int(args.test_limit)

    train_dataset, train_loader = dataloader_for(train_ids, mode, "train", protocol_config, baseline_config, batch_size, True, train_limit)
    val_dataset, val_loader = dataloader_for(val_ids, mode, "val", protocol_config, baseline_config, batch_size, False, val_limit)
    test_dataset, test_loader = dataloader_for(test_ids, mode, "test", protocol_config, baseline_config, batch_size, False, test_limit)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = make_model(mode, int(baseline_config["base_channels"])).to(device)
    pos_weight_value = compute_pos_weight(train_dataset, max_samples=int(args.pos_weight_samples))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr or baseline_config["learning_rate"]),
        weight_decay=float(baseline_config["weight_decay"]),
    )
    scheduler_factor = float(baseline_config.get("scheduler_factor", 0.5))
    scheduler_patience = int(baseline_config.get("scheduler_patience", 4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience)
    threshold = float(baseline_config["threshold"])
    gradient_clip_norm = float(baseline_config.get("gradient_clip_norm", 0.0))

    metadata = {
        "mode": mode,
        "run_dir": str(mode_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "device": str(device),
        "pos_weight": pos_weight_value,
        "model": "SmallUNet",
        "deterministic": deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "gradient_clip_norm": gradient_clip_norm,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "early_stopping_patience": int(args.early_stopping_patience if args.early_stopping_patience >= 0 else baseline_config.get("early_stopping_patience", 0)),
        "early_stopping_min_delta": float(args.early_stopping_min_delta if args.early_stopping_min_delta >= 0 else baseline_config.get("early_stopping_min_delta", 0.0)),
    }
    (mode_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
        print(f"Resumed mode={mode} from epoch {start_epoch - 1}; best_val_iou={best_val_iou:.4f}")

    patience = int(metadata["early_stopping_patience"])
    min_delta = float(metadata["early_stopping_min_delta"])
    epochs_without_improvement = 0
    started = time.time()
    print(f"\n=== Training mode={mode} epochs={epochs} train={len(train_dataset)} val={len(val_dataset)} pos_weight={pos_weight_value:.4f} ===")
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
                    "mode": mode,
                    "epoch": epoch,
                    "val_metrics": asdict(val_metrics),
                    "metadata": metadata,
                    "baseline_config": baseline_config,
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
                "mode": mode,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_iou": best_val_iou,
                "metadata": metadata,
                "baseline_config": baseline_config,
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
                f"Early stopping mode={mode} at epoch {epoch}; "
                f"best_epoch={best_epoch} best_val_iou={best_val_iou:.4f}"
            )
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mode": mode,
            "epoch": epoch_rows[-1]["epoch"] if epoch_rows else 0,
            "metadata": metadata,
            "baseline_config": baseline_config,
            "protocol_config": protocol_config,
        },
        checkpoints_dir / "last_model.pth",
    )

    best = torch.load(checkpoints_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device, criterion, threshold)
    val_metrics_final = run_epoch(model, val_loader, device, criterion, threshold)
    save_prediction_previews(model, test_dataset, device, mode, previews_dir, int(baseline_config["outputs"]["save_prediction_previews"]), threshold)

    summary = {
        "mode": mode,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "final_val_metrics_from_best_checkpoint": asdict(val_metrics_final),
        "test_metrics_from_best_checkpoint": asdict(test_metrics),
        "elapsed_seconds": time.time() - started,
        "metadata": metadata,
    }
    (metrics_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_combined_summary(run_dir: Path, summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        metrics = summary["test_metrics_from_best_checkpoint"]
        rows.append(
            {
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
            }
        )
    with (run_dir / "combined_test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "combined_summary.json").write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strict RGB/Depth/RGB-D traversability baselines.")
    parser.add_argument("--protocol-config", type=Path, default=PROTOCOL_CONFIG_PATH)
    parser.add_argument("--baseline-config", type=Path, default=BASELINE_CONFIG_PATH)
    parser.add_argument("--run-name", default="baseline_run")
    parser.add_argument("--modes", nargs="+", choices=["rgb", "depth", "rgbd"], default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--pos-weight-samples", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--early-stopping-patience", type=int, default=-1, help="Use -1 to read from config; 0 disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=-1.0, help="Use -1 to read from config.")
    parser.add_argument("--resume", action="store_true", help="Resume each mode from checkpoints/latest_model.pth when available.")
    parser.add_argument("--no-log-to-file", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_config = load_json(args.protocol_config)
    baseline_config = load_json(args.baseline_config)
    set_seed(
        int(baseline_config["seed"]),
        deterministic=bool(baseline_config.get("deterministic", False)),
        benchmark=bool(baseline_config.get("cudnn_benchmark", True)),
    )
    modes = args.modes or baseline_config["modes"]
    run_dir = Path(baseline_config["outputs"]["root"]) / args.run_name
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
    (run_dir / "baseline_config_snapshot.json").write_text(json.dumps(baseline_config, indent=2), encoding="utf-8")
    (run_dir / "protocol_config_snapshot.json").write_text(json.dumps(protocol_config, indent=2), encoding="utf-8")

    summaries = []
    for mode in modes:
        summaries.append(train_one_mode(mode, run_dir, protocol_config, baseline_config, args))
    write_combined_summary(run_dir, summaries)
    print(f"\nBaseline run finished: {run_dir}")
    if log_handle is not None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
