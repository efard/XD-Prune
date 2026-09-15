# 3 — Error injection

layer-based Gaussian activation error injection: accuracy

## Required files

```text
2_scripts/3_error_injection_yolo26n.py
3_models/2_baseline_full/full_gen_640.pt
3_models/3_baseline_snow/SNOW_baseline_best.pt
1_data/GEN/dataset_GEN_local.yaml
1_data/SNOW/dataset_SNOW_local.yaml
```

Edit dataset_GEN_local.yaml and dataset_SNOW_local.yaml's "..." to the downloaded dataset path before processing.

## Run GEN

```bash
python 2_scripts/3_error_injection_yolo26n.py \
  --model 3_models/2_baseline_full/full_gen_640.pt \
  --data 1_data/dataset_GEN_local.yaml \
  --split val \
  --noise-std-ratio 0.01 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --out 5_results_err_inj/GEN_error_injection.csv
```

Repeat for SNOW.

By default the script tests all non-Detect top-level layers. To test only selected layers:

```text
--target-layers 0,1,2,9,19
```

Use `--include-detect` only when Detect layers are intentionally part of the experiment.

## Noise definition

With the default argument:

```text
noise_std = activation_std × 0.01
```

## Output

```text
5_results_err_inj/
├── GEN_error_injection.csv
└── SNOW_error_injection.csv
```
