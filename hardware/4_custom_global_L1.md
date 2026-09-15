# 4 — (limited) Global L1 pruning

- P10 target search/replay/recovery for GEN and SNOW
- P56 pipeline for GEN

## Required files

```text
2_scripts/4.1_trace_mode_t4_audit.py
2_scripts/4.2_incremental_global_l1_target_search.py
2_scripts/4.4_full_20epoch_recovery.py
2_scripts/4.5_run_exact_gen56_pipeline.py
2_scripts/4.ex_stage5b_exact_core.py
2_scripts/4.3_stage5c_exact_core.py
2_scripts/4.ex_stage5e_exact_core.py
6_results_prun_L1/T4_25pct_group_ranking.csv
6_results_prun_L1/protected_root_manifest.csv
6_results_prun_L1/validated_group_manifest.csv
3_models/2_baseline_full/full_gen_640.pt
3_models/3_baseline_snow/SNOW_baseline_best.pt
```

The Python environment also needs Torch-Pruning (`torch_pruning`).

## Step 0 — Optional scope/trace audit

```bash
python 2_scripts/4.1_trace_mode_t4_audit.py \
  --gen-model 3_models/2_baseline_full/full_gen_640.pt \
  --snow-model 3_models/3_baseline_snow/SNOW_baseline_best.pt \
  --t4 6_results_prun_L1/T4_25pct_group_ranking.csv \
  --validated-manifest 6_results_prun_L1/validated_group_manifest.csv \
  --protected-manifest 6_results_prun_L1/protected_root_manifest.csv \
  --output-dir 6_results_prun_L1/audit \
  --imgsz 640 \
  --device cuda:0
```

## P10 — Step 1: incremental Global L1 target search

```bash
python 2_scripts/4.2_incremental_global_l1_target_search.py \
  --gen-model 3_models/2_baseline_full/full_gen_640.pt \
  --snow-model 3_models/3_baseline_snow/SNOW_baseline_best.pt \
  --t4 6_results_prun_L1/T4_25pct_group_ranking.csv \
  --protected-manifest 6_results_prun_L1/protected_root_manifest.csv \
  --output-dir 6_results_prun_L1/p10/search \
  --imgsz 640 \
  --seed 42 \
  --maximum-root-pruning-fraction 0.50 \
  --absolute-minimum-channels 4 \
  --device cuda:0
```

This writes the chosen replay plans for GEN and SNOW.

## P10 — Step 2: deterministic replay and raw validation

```bash
python 2_scripts/4.3_stage5c_exact_core.py \
  --gen-model 3_models/2_baseline_full/full_gen_640.pt \
  --snow-model 3_models/3_baseline_snow/SNOW_baseline_best.pt \
  --gen-data 1_data/GEN/dataset_GEN_local.yaml \
  --snow-data 1_data/SNOW/dataset_SNOW_local.yaml \
  --search-dir 6_results_prun_L1/p10/search \
  --t4 6_results_prun_L1/T4_25pct_group_ranking.csv \
  --protected-manifest 6_results_prun_L1/protected_root_manifest.csv \
  --output-dir 6_results_prun_L1/p10/replay \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --seed 42 \
  --device cuda:0 \
  --val-device 0
```

Expected raw model paths include:

```text
6_results_prun_L1/p10/replay/GEN/GEN_global_L1_42root_raw.pt
6_results_prun_L1/p10/replay/SNOW/SNOW_global_L1_42root_raw.pt
```

## P10 — Step 3: 20-epoch recovery

GEN:

```bash
python 2_scripts/4.4_full_20epoch_recovery.py \
  --domain GEN \
  --raw-model 6_results_prun_L1/p10/replay/GEN/GEN_global_L1_42root_raw.pt \
  --data 1_data/GEN/dataset_GEN_local.yaml \
  --output-dir 6_results_prun_L1/p10/recovery_GEN \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --epochs 20 \
  --seed 42 \
  --device 0
```

Repeat with `--domain SNOW`, the SNOW raw model and SNOW dataset.

## P56 GEN pipeline

The runner imports the files as sibling modules: `run_exact_gen56_pipeline.py`, `stage5b_exact_core.py`, `stage5c_exact_core.py` and `stage5e_exact_core.py` in `2_scripts/`.

```bash
python 2_scripts/4.5_run_exact_gen56_pipeline.py \
  --project-root . \
  --gen-model 3_models/2_baseline_full/full_gen_640.pt \
  --gen-data 1_data/GEN/dataset_GEN_local.yaml \
  --t4 6_results_prun_L1/T4_25pct_group_ranking.csv \
  --protected-manifest 6_results_prun_L1/protected_root_manifest.csv \
  --output-dir 6_results_prun_L1/p56/run \
  --target-percent 56.4956 \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --seed 42 \
  --device 0
```

The pipeline first attempts the original 50% per-root cap and can restart from the untouched baseline with a 90% cap when needed by its defined protocol.

## Final model storage

After a run is verified, copy the selected final raw/recovered checkpoints into:

```text
3_models/L1_pruned/
```

CSV/JSON/YAML in `6_results_prun_L1/`.
