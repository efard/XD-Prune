YOLO26n simplified Stage 5 handoff package
Generated: 2026-07-30T10:20:13-06:00

Purpose
-------
This compact package contains only:
1. Information useful for writing the experiment report.
2. Recovered best.pt candidate models for later PYNQ deployment.

Report materials
----------------
- Stage 1 baseline validation and environment records, when found.
- Stage 2 complete per-layer result tables and summaries.
- Stage 3 complete near-target candidate result tables and summaries.
- Stage 4 C007/L9 recovery tables, summaries, settings, architecture audits,
  final validation outputs, and training curves/results.

PYNQ candidate models
---------------------
- YOLO26n_recovered_C007_GEN_best.pt
- YOLO26n_recovered_C007_SNOW_best.pt
- YOLO26n_recovered_L9_GEN_best.pt
- YOLO26n_recovered_L9_SNOW_best.pt

These are the recovered Stage 4 best checkpoints. Stage 5 should select the
final candidate before PYNQ conversion and benchmarking. Until that selection
is complete, both C007 and L9 are retained.

Excluded
--------
- Raw datasets
- Cache files
- last.pt files
- Raw Stage 2/3 replacement checkpoints
- Unused intermediate checkpoints
- Optimizer state
- Full experiment script copies
- Large temporary artifacts

Source runs
-----------
Stage 1: /home/afm176/yolo_project/5_reproduction/stage1_results
Stage 2: /home/afm176/yolo_project/5_reproduction/stage2_results/layer_replacement_sweep_20260729_151557
Stage 3: /home/afm176/yolo_project/5_reproduction/stage3_results/near_target_20260729_172754
Stage 4: /home/afm176/yolo_project/5_reproduction/stage4_results/recovery_C007_L9_20260729_192450
