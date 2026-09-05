# 2 — Profiling

layer-wise and end-to-end profiling

## Required files

```text
2_scripts/2.1_profile_yolo26n_layers.py
2_scripts/2.2_profile_yolo26n_end_to_end.py
3_models/2_baseline_full/full_gen_640.pt
3_models/3_baseline_snow/SNOW_baseline_best.pt
1_data/dataset_GEN_local.yaml
1_data/dataset_SNOW_local.yaml
```

Edit dataset_GEN_local.yaml and dataset_SNOW_local.yaml's "..." to the downloaded dataset path before processing.

## Step 1 — Layer-wise latency

GEN example:

```bash
python 2_scripts/2.1_profile_yolo26n_layers.py \
  --model 3_models/2_baseline_full/full_gen_640.pt \
  --imgsz 640 \
  --device 0 \
  --warmup 20 \
  --iters 100 \
  --seed 42 \
  --out 4_results_profi/GEN_layer_latency.csv
```

Repeat with the SNOW baseline and a SNOW output filename.

The script uses a fixed random input and times top-level YOLO layers after warm-up.

## Step 2 — End-to-end validation profiling

GEN example:

```bash
python 2_scripts/2.2_profile_yolo26n_end_to_end.py \
  --model 3_models/2_baseline_full/full_gen_640.pt \
  --data 1_data/dataset_GEN_local.yaml \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --project 4_results_profi/ultralytics_runs \
  --name GEN_end_to_end \
  --out 4_results_profi/GEN_end_to_end.csv
```

Repeat for SNOW.

## Output

```text
4_results_profi/
├── GEN_layer_latency.csv
├── SNOW_layer_latency.csv
├── GEN_end_to_end.csv
├── SNOW_end_to_end.csv
└── ultralytics_runs/
```
