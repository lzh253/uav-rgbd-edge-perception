from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def depth_stats(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    depth = np.load(path)
    valid = np.isfinite(depth) & (depth > 0)
    row = {
        "exists": True,
        "shape": "x".join(str(x) for x in depth.shape),
        "dtype": str(depth.dtype),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(depth.size),
        "valid_ratio": float(valid.sum() / depth.size) if depth.size else 0.0,
    }
    if valid.sum() > 0:
        values = depth[valid]
        row.update(
            {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "p01": float(np.percentile(values, 1)),
                "p50": float(np.percentile(values, 50)),
                "p99": float(np.percentile(values, 99)),
            }
        )
    else:
        row.update({"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0})
    return row


def colorize_depth(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() <= 10:
        return Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8))
    lo = np.percentile(depth[valid], 1)
    hi = np.percentile(depth[valid], 99)
    if abs(hi - lo) < 1e-12:
        hi = lo + 1e-6
    x = ((depth - lo) / (hi - lo)).clip(0, 1)
    x[~valid] = 0
    cmap = matplotlib.colormaps["Spectral"]
    rgb = (cmap(1 - x)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def resize_keep_aspect(img: Image.Image, width: int) -> Image.Image:
    ratio = width / img.width
    return img.resize((width, max(1, round(img.height * ratio))), Image.Resampling.BILINEAR)


def panel(img: Image.Image, title: str, width: int) -> Image.Image:
    title_height = 28
    img = resize_keep_aspect(img.convert("RGB"), width)
    canvas = Image.new("RGB", (width, img.height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    canvas.paste(img, (0, title_height))
    return canvas


def make_preview(rows: list[dict], config: dict, previews_dir: Path, limit_per_split: int) -> list[str]:
    dataset_root = Path(config["dataset_root"])
    rgb_dir = dataset_root / config["rgb_dir"]
    outputs_root = Path(config["outputs_root"])
    npy_dir = outputs_root / config["depth_generation"]["output_dir"] / config["depth_generation"]["npy_dir"]
    created: list[str] = []

    for split in ("train", "val", "test"):
        chosen = [row for row in rows if row["split"] == split and row["exists"]][:limit_per_split]
        if not chosen:
            continue
        rendered_rows: list[Image.Image] = []
        for row in chosen:
            sample_id = row["sample_id"]
            rgb = Image.open(rgb_dir / f"{sample_id}.jpg").convert("RGB")
            depth = np.load(npy_dir / f"{sample_id}.npy")
            depth_img = colorize_depth(depth)
            panels = [
                panel(rgb, f"{sample_id} RGB", 360),
                panel(depth_img, "raw depth preview", 360),
            ]
            row_height = max(item.height for item in panels)
            canvas = Image.new("RGB", (sum(item.width for item in panels), row_height), "white")
            x = 0
            for item in panels:
                canvas.paste(item, (x, 0))
                x += item.width
            rendered_rows.append(canvas)

        sheet = Image.new("RGB", (rendered_rows[0].width, sum(item.height for item in rendered_rows)), "white")
        y = 0
        for item in rendered_rows:
            sheet.paste(item, (0, y))
            y += item.height
        out_path = previews_dir / f"raw_depth_{split}_preview_sheet.png"
        sheet.save(out_path)
        created.append(str(out_path))
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit locally generated raw depth files.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--preview-limit", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs_root = Path(config["outputs_root"])
    protocol_dir = outputs_root / "protocol"
    reports_dir = outputs_root / "reports"
    previews_dir = outputs_root / "previews"
    npy_dir = outputs_root / config["depth_generation"]["output_dir"] / config["depth_generation"]["npy_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for split in ("train", "val", "test"):
        for sample_id in read_ids(protocol_dir / f"{split}.txt"):
            npy_path = npy_dir / f"{sample_id}.npy"
            stats = depth_stats(npy_path)
            row = {
                "sample_id": sample_id,
                "split": split,
                "npy_path": str(npy_path),
                "exists": bool(stats["exists"]),
                "shape": stats.get("shape", ""),
                "dtype": stats.get("dtype", ""),
                "valid_pixels": stats.get("valid_pixels", 0),
                "total_pixels": stats.get("total_pixels", 0),
                "valid_ratio": stats.get("valid_ratio", 0.0),
                "min": stats.get("min", 0.0),
                "max": stats.get("max", 0.0),
                "mean": stats.get("mean", 0.0),
                "std": stats.get("std", 0.0),
                "p01": stats.get("p01", 0.0),
                "p50": stats.get("p50", 0.0),
                "p99": stats.get("p99", 0.0),
            }
            rows.append(row)

    audit_csv = reports_dir / "raw_depth_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"splits": {}, "overall": {}}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        existing = [row for row in split_rows if row["exists"]]
        summary["splits"][split] = {
            "expected": len(split_rows),
            "existing": len(existing),
            "missing": len(split_rows) - len(existing),
            "coverage": len(existing) / len(split_rows) if split_rows else 0.0,
            "shapes": sorted({row["shape"] for row in existing}),
            "dtypes": sorted({row["dtype"] for row in existing}),
        }
    existing_all = [row for row in rows if row["exists"]]
    summary["overall"] = {
        "expected": len(rows),
        "existing": len(existing_all),
        "missing": len(rows) - len(existing_all),
        "coverage": len(existing_all) / len(rows) if rows else 0.0,
        "audit_csv": str(audit_csv),
    }
    summary["preview_files"] = make_preview(rows, config, previews_dir, args.preview_limit)

    audit_json = reports_dir / "raw_depth_audit_summary.json"
    audit_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
