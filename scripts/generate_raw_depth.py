from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger as da3_logger
from depth_anything_3.utils.visualize import visualize_depth


SCRIPT_DIR = Path(__file__).resolve().parent
FINALLY_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = FINALLY_ROOT / "configs" / "experiment_protocol.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_ids(protocol_dir: Path, split: str) -> list[tuple[str, str]]:
    splits = ["train", "val", "test"] if split == "all" else [split]
    rows: list[tuple[str, str]] = []
    for item in splits:
        for sample_id in read_ids(protocol_dir / f"{item}.txt"):
            rows.append((sample_id, item))
    return rows


def normalize_to_u16(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() <= 10:
        return np.zeros(depth.shape, dtype=np.uint16), 0.0, 0.0
    depth_min = float(np.percentile(depth[valid], 1))
    depth_max = float(np.percentile(depth[valid], 99))
    if abs(depth_max - depth_min) < 1e-12:
        depth_max = depth_min + 1e-6
    norm = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    norm[~valid] = 0
    return (norm * 65535.0).round().astype(np.uint16), depth_min, depth_max


def depth_stats(depth: np.ndarray) -> dict[str, float | int | str]:
    valid = np.isfinite(depth) & (depth > 0)
    stats: dict[str, float | int | str] = {
        "shape": "x".join(str(x) for x in depth.shape),
        "dtype": str(depth.dtype),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(depth.size),
        "valid_ratio": float(valid.sum() / depth.size) if depth.size else 0.0,
    }
    if valid.sum() > 0:
        values = depth[valid]
        stats.update(
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
        stats.update({"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0})
    return stats


def write_manifest_header(path: Path) -> None:
    fields = [
        "timestamp",
        "sample_id",
        "split",
        "status",
        "error",
        "rgb_path",
        "npy_path",
        "png16_path",
        "preview_path",
        "model_name",
        "process_res",
        "process_res_method",
        "seconds",
        "shape",
        "dtype",
        "valid_pixels",
        "total_pixels",
        "valid_ratio",
        "min",
        "max",
        "mean",
        "std",
        "p01",
        "p50",
        "p99",
        "png16_norm_min",
        "png16_norm_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def append_manifest(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def load_model(model_name: str, device: str) -> DepthAnything3:
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model


def run_inference_sequential(
    model: DepthAnything3,
    rgb_path: Path,
    process_res: int,
    process_res_method: str,
):
    """Run DA3 inference while avoiding Windows ThreadPool permission failures."""
    imgs_cpu, extrinsics, intrinsics = model.input_processor(
        [str(rgb_path)],
        None,
        None,
        process_res,
        process_res_method,
        num_workers=1,
        sequential=True,
        print_progress=False,
        desc=None,
    )
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
    ex_t_norm = model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
    raw_output = model._run_model_forward(
        imgs,
        ex_t_norm,
        in_t,
        export_feat_layers=[],
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    )
    prediction = model._convert_to_prediction(raw_output)
    prediction = model._align_to_input_extrinsics_intrinsics(extrinsics, intrinsics, prediction)
    prediction = model._add_processed_images(prediction, imgs_cpu)
    return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw Depth Anything depth arrays into a local output directory.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--offset", type=int, default=0, help="Number of selected samples to skip before applying --limit.")
    parser.add_argument("--run-name", default="raw_depth_run")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--process-res", type=int, default=None)
    parser.add_argument("--process-res-method", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose-da3", action="store_true", help="Show Depth Anything internal INFO logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.verbose_da3:
        da3_logger.level = 1
    config = load_config(args.config)
    depth_cfg = config["depth_generation"]

    model_name = args.model_name or depth_cfg["model_name"]
    process_res = int(args.process_res or depth_cfg["process_res"])
    process_res_method = args.process_res_method or depth_cfg["process_res_method"]

    outputs_root = Path(config["outputs_root"])
    dataset_root = Path(config["dataset_root"])
    rgb_dir = dataset_root / config["rgb_dir"]
    protocol_dir = outputs_root / "protocol"
    reports_dir = outputs_root / "reports"
    depth_root = outputs_root / depth_cfg["output_dir"]
    npy_dir = depth_root / depth_cfg["npy_dir"]
    png16_dir = depth_root / depth_cfg["png16_dir"]
    preview_dir = depth_root / depth_cfg["preview_dir"]

    for directory in (reports_dir, depth_root, npy_dir, png16_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = reports_dir / f"depth_generation_{args.run_name}.csv"
    summary_path = reports_dir / f"depth_generation_{args.run_name}_summary.json"
    write_manifest_header(manifest_path)

    items = selected_ids(protocol_dir, args.split)
    if args.offset and args.offset > 0:
        items = items[args.offset :]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    print(f"Loading model: {model_name}")
    print(f"Device: {args.device}")
    model = load_model(model_name, args.device)
    print(f"Generating raw depth for {len(items)} samples")

    counts = {"generated": 0, "skipped_existing": 0, "failed": 0}
    started = time.time()

    for index, (sample_id, split) in enumerate(items, start=1):
        timestamp = datetime.now().isoformat(timespec="seconds")
        rgb_path = rgb_dir / f"{sample_id}.jpg"
        npy_path = npy_dir / f"{sample_id}.npy"
        png16_path = png16_dir / f"{sample_id}.png"
        preview_path = preview_dir / f"{sample_id}.png"

        base_row = {
            "timestamp": timestamp,
            "sample_id": sample_id,
            "split": split,
            "status": "",
            "error": "",
            "rgb_path": str(rgb_path),
            "npy_path": str(npy_path),
            "png16_path": str(png16_path),
            "preview_path": str(preview_path),
            "model_name": model_name,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "seconds": "0.0000",
            "shape": "",
            "dtype": "",
            "valid_pixels": 0,
            "total_pixels": 0,
            "valid_ratio": "0.00000000",
            "min": "0.00000000",
            "max": "0.00000000",
            "mean": "0.00000000",
            "std": "0.00000000",
            "p01": "0.00000000",
            "p50": "0.00000000",
            "p99": "0.00000000",
            "png16_norm_min": "0.00000000",
            "png16_norm_max": "0.00000000",
        }

        if npy_path.exists() and png16_path.exists() and preview_path.exists() and not args.overwrite:
            base_row["status"] = "skipped_existing"
            counts["skipped_existing"] += 1
            append_manifest(manifest_path, base_row)
            continue

        one_started = time.time()
        try:
            prediction = run_inference_sequential(
                model=model,
                rgb_path=rgb_path,
                process_res=process_res,
                process_res_method=process_res_method,
            )
            depth = prediction.depth[0].astype(np.float32)
            np.save(npy_path, depth)

            png16, norm_min, norm_max = normalize_to_u16(depth)
            Image.fromarray(png16, mode="I;16").save(png16_path)

            depth_vis = visualize_depth(depth)
            Image.fromarray(depth_vis).save(preview_path)

            stats = depth_stats(depth)
            elapsed = time.time() - one_started
            base_row.update(
                {
                    "status": "generated",
                    "seconds": f"{elapsed:.4f}",
                    "shape": stats["shape"],
                    "dtype": stats["dtype"],
                    "valid_pixels": stats["valid_pixels"],
                    "total_pixels": stats["total_pixels"],
                    "valid_ratio": f"{stats['valid_ratio']:.8f}",
                    "min": f"{stats['min']:.8f}",
                    "max": f"{stats['max']:.8f}",
                    "mean": f"{stats['mean']:.8f}",
                    "std": f"{stats['std']:.8f}",
                    "p01": f"{stats['p01']:.8f}",
                    "p50": f"{stats['p50']:.8f}",
                    "p99": f"{stats['p99']:.8f}",
                    "png16_norm_min": f"{norm_min:.8f}",
                    "png16_norm_max": f"{norm_max:.8f}",
                }
            )
            counts["generated"] += 1
            if index == 1 or index == len(items) or index % max(1, args.progress_every) == 0:
                print(f"[{index}/{len(items)}] generated {sample_id} shape={depth.shape} seconds={elapsed:.2f}")
        except Exception as exc:  # noqa: BLE001 - keep manifest robust for long runs
            base_row["status"] = "failed"
            base_row["error"] = traceback.format_exc().replace("\n", "\\n")
            counts["failed"] += 1
            print(f"[{index}/{len(items)}] failed {sample_id}: {exc}")
        append_manifest(manifest_path, base_row)

    summary = {
        "run_name": args.run_name,
        "model_name": model_name,
        "device": args.device,
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "process_res": process_res,
        "process_res_method": process_res_method,
        "counts": counts,
        "total_requested": len(items),
        "elapsed_seconds": time.time() - started,
        "manifest_path": str(manifest_path),
        "npy_dir": str(npy_dir),
        "png16_dir": str(png16_dir),
        "preview_dir": str(preview_dir),
        "authoritative_depth_input": "npy float32 files",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
