# Bandwidth-Constrained RGB-D Edge Perception

This repository contains the training and evaluation code for a UAV-to-ground traversability study. The pipeline combines aerial RGB images with monocular depth priors, encodes them into a compact semantic tensor, and evaluates the decoded road grid using segmentation, digital-link, calibration, and route-topology metrics.

## Repository Layout

- `configs/`: fixed experiment and evaluation settings.
- `protocol/`: train, validation, and test splits used in the experiments.
- `scripts/`: data preparation, training, link simulation, planning, and diagnostic code.
- `results/`: compact aggregate tables reported in the study.
- `checkpoints/`: instructions for the validation-selected model released separately.

## Setup

Create the environment with Conda:

```bash
conda env create -f environment.yml
conda activate uav-rgbd-edge
```

or install the Python dependencies directly:

```bash
pip install -r requirements.txt
```

## Data

Download AeroScapes from its official source and place it under:

```text
data/AeroScapes/
├── JPEGImages/
├── SegmentationClass/
├── Visualizations/
└── ImageSets/
```

The fixed split files are provided in `protocol/`. Raw images, labels, and generated depth arrays are not redistributed here.

Generate the monocular depth priors:

```bash
python scripts/generate_raw_depth.py --config configs/experiment_protocol.json --split all
```

## Training

Matched RGB, depth-only, and naive RGB-D baselines:

```bash
python scripts/train_baselines.py \
  --protocol-config configs/experiment_protocol.json \
  --baseline-config configs/baseline_training_long_cuda_safe.json \
  --run-name matched_baselines \
  --modes rgb depth rgbd
```

Strict no-skip semantic bottleneck and channel-count ablation:

```bash
python scripts/train_strict_bottleneck.py \
  --config configs/strict_bottleneck_training.json
```

The stable latent-8 recipe can be reproduced with:

```bash
python scripts/train_strict_bottleneck.py \
  --config configs/strict_l8_stable_training.json
```

## Evaluation

Bit-level 8-bit payload serialization with BPSK/AWGN:

```bash
python scripts/evaluate_digital_link.py \
  --config configs/digital_link_evaluation.json
```

Planner evaluation and validation-selected threshold sweep:

```bash
python scripts/evaluate_path_planning.py
python scripts/evaluate_path_planning_threshold_sweep.py
```

Additional scripts cover probability calibration, spatial-risk diagnostics, post-processing ablations, planner robustness, and resolution transfer. Compact reference outputs are stored in `results/`.

## Checkpoint

The validation-selected latent-8 checkpoint is distributed with the `v1.0.0` GitHub release. Extract it to:

```text
checkpoints/strict_l8_validation_selected_model.pth
```

## Reproducibility Notes

- Data splits and random seeds are fixed in the supplied protocol and configuration files.
- Thresholds are selected on the validation split and transferred unchanged to the test split.
- The 128 x 128 grid is a route-level semantic message, not a final local obstacle-avoidance map.
- The digital-link experiments model payload quantization and controlled BPSK/AWGN stress; they are not a complete UAV radio-stack implementation.

## License

See `LICENSE_NOTICE.md`.
