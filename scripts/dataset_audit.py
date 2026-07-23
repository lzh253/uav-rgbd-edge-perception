from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    group_id: str
    official_split: str
    protocol_split: str
    rgb_path: Path
    label_path: Path
    label_vis_path: Path
    depth_preview_path: Path
    rgb_exists: bool
    label_exists: bool
    label_vis_exists: bool
    depth_preview_exists: bool
    width: int | None
    height: int | None
    label_width: int | None
    label_height: int | None
    depth_width: int | None
    depth_height: int | None
    depth_mode: str
    label_ids: tuple[int, ...]
    passable_pixels: int
    total_pixels: int
    passable_ratio: float


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_id(sample_id: str) -> str:
    return sample_id.split("_", 1)[0]


def choose_validation_groups(train_ids: list[str], fraction: float, seed: int) -> set[str]:
    groups = sorted({group_id(sid) for sid in train_ids})
    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)
    target_count = max(1, round(len(groups) * fraction))
    return set(shuffled[:target_count])


def safe_image_info(path: Path) -> tuple[int | None, int | None, str]:
    if not path.exists():
        return None, None, ""
    with Image.open(path) as img:
        return img.width, img.height, img.mode


def load_label(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def make_passable_mask(label: np.ndarray, passable_ids: Iterable[int]) -> np.ndarray:
    mask = np.zeros(label.shape, dtype=np.uint8)
    for class_id in passable_ids:
        mask[label == int(class_id)] = 1
    return mask


def build_records(config: dict) -> list[SampleRecord]:
    dataset_root = Path(config["dataset_root"])
    rgb_dir = dataset_root / config["rgb_dir"]
    label_dir = dataset_root / config["label_dir"]
    label_vis_dir = dataset_root / config["label_visualization_dir"]
    depth_dir = dataset_root / config["depth_preview_dir"]
    image_sets_dir = dataset_root / config["image_sets_dir"]

    official_train_ids = read_ids(image_sets_dir / config["official_train_file"])
    official_test_ids = read_ids(image_sets_dir / config["official_val_file"])

    split_cfg = config["protocol_split"]
    val_groups = choose_validation_groups(
        official_train_ids,
        float(split_cfg["validation_group_fraction"]),
        int(split_cfg["random_seed"]),
    )

    records: list[SampleRecord] = []
    for official_split, ids in (("official_train", official_train_ids), ("official_val", official_test_ids)):
        for sid in ids:
            gid = group_id(sid)
            if official_split == "official_val":
                protocol_split = "test"
            elif gid in val_groups:
                protocol_split = "val"
            else:
                protocol_split = "train"

            rgb_path = rgb_dir / f"{sid}.jpg"
            label_path = label_dir / f"{sid}.png"
            label_vis_path = label_vis_dir / f"{sid}.png"
            depth_preview_path = depth_dir / f"depth_{sid}.jpg"

            rgb_width, rgb_height, _rgb_mode = safe_image_info(rgb_path)
            label_width, label_height, _label_mode = safe_image_info(label_path)
            depth_width, depth_height, depth_mode = safe_image_info(depth_preview_path)

            label_ids: tuple[int, ...] = tuple()
            passable_pixels = 0
            total_pixels = 0
            passable_ratio = 0.0
            if label_path.exists():
                label = load_label(label_path)
                mask = make_passable_mask(label, config["passable_class_ids"])
                label_ids = tuple(int(v) for v in sorted(np.unique(label)))
                passable_pixels = int(mask.sum())
                total_pixels = int(mask.size)
                passable_ratio = float(passable_pixels / total_pixels) if total_pixels else 0.0

            records.append(
                SampleRecord(
                    sample_id=sid,
                    group_id=gid,
                    official_split=official_split,
                    protocol_split=protocol_split,
                    rgb_path=rgb_path,
                    label_path=label_path,
                    label_vis_path=label_vis_path,
                    depth_preview_path=depth_preview_path,
                    rgb_exists=rgb_path.exists(),
                    label_exists=label_path.exists(),
                    label_vis_exists=label_vis_path.exists(),
                    depth_preview_exists=depth_preview_path.exists(),
                    width=rgb_width,
                    height=rgb_height,
                    label_width=label_width,
                    label_height=label_height,
                    depth_width=depth_width,
                    depth_height=depth_height,
                    depth_mode=depth_mode,
                    label_ids=label_ids,
                    passable_pixels=passable_pixels,
                    total_pixels=total_pixels,
                    passable_ratio=passable_ratio,
                )
            )
    return records


def write_split_files(records: list[SampleRecord], protocol_dir: Path) -> dict:
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for rec in records:
        splits[rec.protocol_split].append(rec.sample_id)

    for split, ids in splits.items():
        (protocol_dir / f"{split}.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")

    groups_by_split = {
        split: sorted({group_id(sid) for sid in ids})
        for split, ids in splits.items()
    }
    metadata = {
        "splits": {split: {"num_images": len(ids), "num_groups": len(groups_by_split[split])} for split, ids in splits.items()},
        "groups": groups_by_split,
    }
    (protocol_dir / "splits.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_index(records: list[SampleRecord], protocol_dir: Path) -> None:
    fieldnames = [
        "sample_id",
        "group_id",
        "official_split",
        "protocol_split",
        "rgb_path",
        "label_path",
        "label_visualization_path",
        "depth_preview_path",
        "rgb_exists",
        "label_exists",
        "label_visualization_exists",
        "depth_preview_exists",
        "rgb_width",
        "rgb_height",
        "label_width",
        "label_height",
        "depth_width",
        "depth_height",
        "depth_mode",
        "label_ids",
        "passable_pixels",
        "total_pixels",
        "passable_ratio",
    ]
    with (protocol_dir / "dataset_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "sample_id": rec.sample_id,
                    "group_id": rec.group_id,
                    "official_split": rec.official_split,
                    "protocol_split": rec.protocol_split,
                    "rgb_path": str(rec.rgb_path),
                    "label_path": str(rec.label_path),
                    "label_visualization_path": str(rec.label_vis_path),
                    "depth_preview_path": str(rec.depth_preview_path),
                    "rgb_exists": rec.rgb_exists,
                    "label_exists": rec.label_exists,
                    "label_visualization_exists": rec.label_vis_exists,
                    "depth_preview_exists": rec.depth_preview_exists,
                    "rgb_width": rec.width,
                    "rgb_height": rec.height,
                    "label_width": rec.label_width,
                    "label_height": rec.label_height,
                    "depth_width": rec.depth_width,
                    "depth_height": rec.depth_height,
                    "depth_mode": rec.depth_mode,
                    "label_ids": " ".join(str(x) for x in rec.label_ids),
                    "passable_pixels": rec.passable_pixels,
                    "total_pixels": rec.total_pixels,
                    "passable_ratio": f"{rec.passable_ratio:.8f}",
                }
            )


def aggregate_label_distribution(records: list[SampleRecord], reports_dir: Path) -> dict:
    counts: dict[str, dict[int, int]] = {split: {} for split in ("train", "val", "test", "all")}
    image_counts: dict[str, dict[int, int]] = {split: {} for split in ("train", "val", "test", "all")}

    for rec in records:
        if not rec.label_exists:
            continue
        label = load_label(rec.label_path)
        unique, pixel_counts = np.unique(label, return_counts=True)
        for split in (rec.protocol_split, "all"):
            for class_id, count in zip(unique, pixel_counts):
                cid = int(class_id)
                counts[split][cid] = counts[split].get(cid, 0) + int(count)
                image_counts[split][cid] = image_counts[split].get(cid, 0) + 1

    with (reports_dir / "label_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label_id", "pixels", "images_with_label", "pixel_percent"])
        writer.writeheader()
        for split in ("train", "val", "test", "all"):
            total = sum(counts[split].values())
            for class_id in sorted(counts[split]):
                writer.writerow(
                    {
                        "split": split,
                        "label_id": class_id,
                        "pixels": counts[split][class_id],
                        "images_with_label": image_counts[split][class_id],
                        "pixel_percent": f"{counts[split][class_id] / total:.8f}" if total else "0.00000000",
                    }
                )
    return {
        split: {
            str(class_id): {
                "pixels": counts[split][class_id],
                "images_with_label": image_counts[split][class_id],
            }
            for class_id in sorted(counts[split])
        }
        for split in counts
    }


def resize_keep_aspect(img: Image.Image, width: int) -> Image.Image:
    ratio = width / img.width
    return img.resize((width, max(1, round(img.height * ratio))), Image.Resampling.BILINEAR)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")


def overlay_mask(rgb: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    rgb = rgb.convert("RGB")
    color = Image.new("RGB", rgb.size, (255, 220, 0))
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(rgb.size, Image.Resampling.NEAREST)
    return Image.composite(Image.blend(rgb, color, alpha), rgb, mask_img)


def panel_with_title(img: Image.Image, title: str, width: int) -> Image.Image:
    title_height = 28
    img = resize_keep_aspect(img, width)
    canvas = Image.new("RGB", (width, img.height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    canvas.paste(img, (0, title_height))
    return canvas


def make_preview(records: list[SampleRecord], config: dict, previews_dir: Path) -> list[str]:
    preview_cfg = config["preview"]
    width = int(preview_cfg["image_width"])
    samples_per_split = int(preview_cfg["samples_per_split"])
    alpha = float(preview_cfg["mask_alpha"])
    created: list[str] = []

    for split in ("train", "val", "test"):
        split_records = [rec for rec in records if rec.protocol_split == split and rec.rgb_exists and rec.label_exists]
        chosen = split_records[:samples_per_split]
        rows: list[Image.Image] = []
        for rec in chosen:
            rgb = Image.open(rec.rgb_path).convert("RGB")
            label = load_label(rec.label_path)
            mask = make_passable_mask(label, config["passable_class_ids"])
            label_vis = Image.open(rec.label_vis_path).convert("RGB") if rec.label_vis_exists else mask_to_image(label > 0)
            depth = Image.open(rec.depth_preview_path).convert("RGB") if rec.depth_preview_exists else Image.new("RGB", rgb.size, "gray")
            overlay = overlay_mask(rgb, mask, alpha)

            panels = [
                panel_with_title(rgb, f"{rec.sample_id} RGB", width),
                panel_with_title(label_vis, "semantic label vis", width),
                panel_with_title(mask_to_image(mask), "passable mask id=10", width),
                panel_with_title(overlay, "RGB + passable overlay", width),
                panel_with_title(depth, "existing depth preview", width),
            ]
            row_height = max(panel.height for panel in panels)
            row = Image.new("RGB", (sum(panel.width for panel in panels), row_height), "white")
            x = 0
            for panel in panels:
                row.paste(panel, (x, 0))
                x += panel.width
            rows.append(row)

        if rows:
            sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
            y = 0
            for row in rows:
                sheet.paste(row, (0, y))
                y += row.height
            out_path = previews_dir / f"{split}_preview_sheet.png"
            sheet.save(out_path)
            created.append(str(out_path))
    return created


def write_summary(config: dict, records: list[SampleRecord], split_metadata: dict, label_distribution: dict, reports_dir: Path, previews: list[str]) -> None:
    missing = {
        "rgb": [rec.sample_id for rec in records if not rec.rgb_exists],
        "label": [rec.sample_id for rec in records if not rec.label_exists],
        "label_visualization": [rec.sample_id for rec in records if not rec.label_vis_exists],
        "depth_preview": [rec.sample_id for rec in records if not rec.depth_preview_exists],
    }
    passable_by_split = {}
    for split in ("train", "val", "test"):
        split_records = [rec for rec in records if rec.protocol_split == split]
        total_pixels = sum(rec.total_pixels for rec in split_records)
        passable_pixels = sum(rec.passable_pixels for rec in split_records)
        passable_by_split[split] = {
            "num_images": len(split_records),
            "passable_pixels": passable_pixels,
            "total_pixels": total_pixels,
            "passable_ratio": passable_pixels / total_pixels if total_pixels else 0.0,
        }

    depth_modes = sorted({rec.depth_mode for rec in records if rec.depth_mode})
    depth_sizes = sorted({f"{rec.depth_width}x{rec.depth_height}" for rec in records if rec.depth_width and rec.depth_height})
    summary = {
        "protocol_name": config["protocol_name"],
        "dataset_root": config["dataset_root"],
        "passable_class_ids": config["passable_class_ids"],
        "non_passable_policy": config["non_passable_policy"],
        "class_0_note": config["class_0_note"],
        "num_records": len(records),
        "split_metadata": split_metadata,
        "missing_files": {key: {"count": len(value), "first_20": value[:20]} for key, value in missing.items()},
        "passable_by_split": passable_by_split,
        "label_distribution": label_distribution,
        "depth_preview_audit": {
            "modes": depth_modes,
            "sizes": depth_sizes[:20],
            "warning": "Existing DepthImages are previews saved as JPG. They are useful for visual checking, but raw Depth Anything outputs should be regenerated before final RGB-D training.",
        },
        "preview_files": previews,
    }
    (reports_dir / "dataset_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    config = load_config()
    outputs_root = Path(config["outputs_root"])
    protocol_dir = outputs_root / "protocol"
    reports_dir = outputs_root / "reports"
    previews_dir = outputs_root / "previews"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(config)
    split_metadata = write_split_files(records, protocol_dir)
    write_index(records, protocol_dir)
    label_distribution = aggregate_label_distribution(records, reports_dir)
    previews = make_preview(records, config, previews_dir)
    write_summary(config, records, split_metadata, label_distribution, reports_dir, previews)

    print("Dataset audit completed.")
    print(f"Records: {len(records)}")
    print(json.dumps(split_metadata["splits"], indent=2))
    print(f"Index: {protocol_dir / 'dataset_index.csv'}")
    print(f"Summary: {reports_dir / 'dataset_audit_summary.json'}")


if __name__ == "__main__":
    main()
