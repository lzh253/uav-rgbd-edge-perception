# Model Card

## Intended Use

The model decodes a compact RGB-D latent tensor into a coarse binary road-traversability grid for route-level planning experiments.

## Inputs and Outputs

- Input: 128 x 128 UAV-view RGB image and a monocular depth prior.
- Transmitted representation: latent-8 tensor with shape 8 x 16 x 16.
- Output: 128 x 128 road-traversability probability grid.

## Evaluation

The released checkpoint was selected on the validation split. Reported evaluation includes pixel IoU/F1, A* and Dijkstra route-safety metrics, payload quantization, BPSK/AWGN stress, calibration, and spatial-risk diagnostics.

## Limitations

The depth prior is monocular and non-metric. The output grid is a coarse route-level message and is not a substitute for onboard obstacle avoidance. The digital-link study is a controlled physical-layer abstraction rather than a complete air-to-ground radio stack.
