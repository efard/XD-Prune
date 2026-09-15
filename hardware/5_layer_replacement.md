# 5 — Layer Replacement pruning

- baseline validation
- dual-domain single-layer sweep
- P10 near-target combination sweep
- selected P10 recovery/evaluation
- GEN P56 efficiency-greedy Layer Replacement

## Required files

```text
2_scripts/5.1_validate_supplied_baselines.py
2_scripts/5.2_layer_replacement_sweep_dual_domain.py
2_scripts/5.3_near_target_combination_sweep.py
2_scripts/5.4_recover_and_evaluate_selected.py
2_scripts/5.5_run_gen50_efficiency_greedy.py
2_scripts/layer_replacement_adapter.py
3_models/2_baseline_full/full_gen_640.pt
3_models/3_baseline_snow/SNOW_baseline_best.pt
1_data/dataset_GEN_local.yaml
1_data/dataset_SNOW_local.yaml
1_data_full_val_images.txt *
1_data_full_val_images.txt *
```

Edit dataset_GEN_local.yaml and dataset_SNOW_local.yaml's "..." to the downloaded dataset path before processing.

Generate your own GEN_full_val_images.txt and SNOW_full_val_images.txt depends on dataset path.

Keep `layer_replacement_adapter.py` in the same `2_scripts/` directory as these scripts.

## Step 1 — Validate supplied baselines

```bash
python 2_scripts/5.1_validate_supplied_baselines.py \
  --gen-model 3_models/2_baseline_full/full_gen_640.pt \
  --snow-model 3_models/3_baseline_snow/SNOW_baseline_best.pt \
  --gen-data 1_data/dataset_GEN_local.yaml \
  --snow-data 1_data/dataset_SNOW_local.yaml \
  --gen-manifest 1_data/GEN_full_val_images.txt \
  --snow-manifest 1_data/SNOW_full_val_images.txt \
  --project 7_results_prun_LR/reference/ultralytics_runs \
  --out-dir 7_results_prun_LR/reference \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0
```

Main input for the next stage:

```text
7_results_prun_LR/reference/baseline_validation_fp32.csv
```

## Step 2 — Single-layer dual-domain sweep

```bash
python 2_scripts/5.2_layer_replacement_sweep_dual_domain.py \
  --gen-model 3_models/2_baseline_full/full_gen_640.pt \
  --snow-model 3_models/3_baseline_snow/SNOW_baseline_best.pt \
  --gen-data 1_data/dataset_GEN_local.yaml \
  --snow-data 1_data/dataset_SNOW_local.yaml \
  --stage1-csv 7_results_prun_LR/reference/baseline_validation_fp32.csv \
  --split val \
  --target-layers all \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --multi-input-policy concat \
  --project 7_results_prun_LR/p10/stage2_ultralytics \
  --out-dir 7_results_prun_LR/p10/stage2
```

Important Stage-2 outputs include:

```text
layer_metadata_dual_domain.csv
raw_dual_domain_layer_replacement.csv
baseline_GEN_SNOW_stage2.csv
baseline_repeatability_gate.csv
saved_replaced_models/
```

## Step 3 — P10 near-target combination sweep

The restored Stage-3 script requires the entire Stage-2 output directory because it reads the four filenames listed above.

```bash
python 2_scripts/5.3_near_target_combination_sweep.py \
  --gen-model 3_models/2_baseline_full/full_gen_640.pt \
  --snow-model 3_models/3_baseline_snow/SNOW_baseline_best.pt \
  --gen-data 1_data/dataset_GEN_local.yaml \
  --snow-data 1_data/dataset_SNOW_local.yaml \
  --stage2-dir 7_results_prun_LR/p10/stage2 \
  --split val \
  --target-percent 10.25 \
  --relative-tolerance 0.05 \
  --max-combination-size 3 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --multi-input-policy concat \
  --project 7_results_prun_LR/p10/stage3_ultralytics \
  --out-dir 7_results_prun_LR/p10/stage3
```

Main outputs include:

```text
near_target_candidate_search_space.csv
raw_near_target_combination_results.csv
near_target_results_sorted_by_target_distance.csv
near_target_results_sorted_by_GEN_signed_AD.csv
near_target_results_sorted_by_SNOW_signed_AD.csv
saved_near_target_models/
stage3_summary.json
```

## Step 4 — Recover/evaluate selected P10 candidates

The supplied recovery script is intentionally specific to:
- `C007`: layers 9 and 19
- `L9`: layer 9

It reads the raw model paths from the fresh Stage-2/Stage-3 CSV files.

```bash
python 2_scripts/5.4_recover_and_evaluate_selected.py \
  --stage1-csv 7_results_prun_LR/reference/baseline_validation_fp32.csv \
  --stage2-raw-csv 7_results_prun_LR/p10/stage2/raw_dual_domain_layer_replacement.csv \
  --stage3-raw-csv 7_results_prun_LR/p10/stage3/raw_near_target_combination_results.csv \
  --gen-baseline-model 3_models/2_baseline_full/full_gen_640.pt \
  --snow-baseline-model 3_models/3_baseline_snow/SNOW_baseline_best.pt \
  --gen-data 1_data/dataset_GEN_local.yaml \
  --snow-data 1_data/dataset_SNOW_local.yaml \
  --out-dir 7_results_prun_LR/p10/recovery \
  --device 0 \
  --train-workers 4 \
  --val-workers 8 \
  --mode full
```

Do **not** use the archived old Stage-2/Stage-3 CSVs as active inputs after relocating the repo. Their embedded model paths point to `/home/afm176/...`. Rerun Steps 2 and 3 so the paths are regenerated correctly.

## P56 — GEN efficiency-greedy Layer Replacement

Use the fresh single-layer Stage-2 CSV explicitly so the script does not need its legacy auto-discovery paths:

```bash
python 2_scripts/5.5run_gen50_efficiency_greedy.py \
  --project-root . \
  --baseline-model 3_models/2_baseline_full/full_gen_640.pt \
  --data 1_data/dataset_GEN_local.yaml \
  --out-dir 7_results_prun_LR/p56/run \
  --single-layer-csv 7_results_prun_LR/p10/stage2/raw_dual_domain_layer_replacement.csv \
  --target-percent 50 \
  --epochs 20 \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --seed 42 \
  --device 0
```

## Final model storage

After verification, place final selected checkpoints under:

```text
3_models_pt/layer_replacement/p10/
3_models_pt/layer_replacement/p56/
```

The experiment in `7_results_prun_LR/`.
