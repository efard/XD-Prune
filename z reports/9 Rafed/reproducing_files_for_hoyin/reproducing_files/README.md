# Reproduction package

This package contains the fixed inputs and settings needed to reproduce the
current GEN/SNOW evaluation and to run standard structured-pruning competitors
under the same experimental budget.

It does not include dataset images or labels. Obtain MIO-TCD and ACDC
separately, preserve the supplied split membership, and update only the dataset
root paths in `configs/dataset_GEN.yaml` and `configs/dataset_SNOW.yaml`.

## Primary controls

| Item | Fixed value |
|---|---|
| Architecture | YOLO26n |
| GEN baseline | `models/GEN_baseline_best.pt` |
| SNOW baseline | `models/SNOW_baseline_best.pt` |
| Primary random seed | 42 |
| Local channel-pruning ratio | 25% of the selected live root channels |
| Target whole-model parameter reduction | 10.25% |
| Allowed target handling | Use the closest feasible dependency-safe model and report the exact achieved reduction |
| Recovery budget | 20 epochs per domain |
| Input size | 640 x 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Final learning-rate fraction | 0.01 |
| Momentum / beta1 | 0.9 |
| Weight decay | 0.0005 |
| Warm-up | 1 epoch |
| Mixed precision training | Enabled |
| Deterministic mode | Enabled |
| Training workers | 4 |

The recorded T6 run retained the Ultralytics default mosaic probability of
1.0 and used `close_mosaic: 0`. There was no separate BatchNorm-only
recalibration pass; BatchNorm statistics were updated during ordinary
fine-tuning.

## Dataset splits

| Domain | Dataset | Train | Validation | Test |
|---|---|---:|---:|---:|
| GEN | MIO-TCD | 88,000 | 11,000 | 11,000 |
| SNOW | ACDC snow | 400 | 100 | 500 |

The CSVs in `manifests/` give the exact image membership and source-relative
path for every split. The test sets were not used for pruning decisions.

## Evaluation settings

Publication-facing evaluation uses:

- validation split;
- FP32 inference (`half: false`);
- 640 x 640 input;
- batch size 16;
- rectangular batches;
- confidence threshold 0.001;
- non-maximum-suppression IoU threshold 0.70;
- maximum 300 detections;
- no test-time augmentation;
- class-aware non-maximum suppression;
- seed 42.

Use `configs/evaluation_fp32.yaml`. The research variant changes only
`save_json` to preserve post-NMS detections.

## Competitor protocol

Run these three competitors:

1. global L1 channel-magnitude structured pruning;
2. standard Torch-Pruning without the proposed dual-domain ranking;
3. random structured channel pruning with seed 42.

For a fair comparison:

- start from the supplied baseline checkpoint;
- use dependency-consistent channel pruning, not zeroing or whole-layer
  replacement;
- use the validated root pool in `group_definitions/`;
- protect every root listed in `protected_root_manifest.csv`, including the
  Detect head and packed-attention projections;
- target 10.25% whole-model parameter reduction and record the exact achieved
  percentage;
- use the same 20-epoch recovery settings;
- evaluate with the supplied FP32 configuration and exact validation split;
- report parameters, GFLOPs, mAP50-95, mAP50, mAP75, precision, recall, model
  size, and—when available—PYNQ-Z2 latency/FPS/memory.

The primary competitor comparison should be run on GEN as requested. To
support the dual-domain claim, the final dependency-safe architecture should
also be reproduced from the SNOW baseline and evaluated on the fixed SNOW
split using the same reduction and recovery budget.

## Reference results

The `tables/` directory contains the current 25%-local group ranking and the
T5-T7 results. The strongest completed recovery result is T6, which performs
one 20-epoch fine-tuning session after pruning.

## File integrity

`CHECKSUMS.sha256` identifies every file in this package. Paths are relative to
the package root. Recompute SHA-256 hashes after transfer and compare them
before running experiments.

