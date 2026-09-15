# 1 — Model training

## Required files

```text
2_scripts/1.1_convert_mio_to_yolo.py
2_scripts/1.2_train_yolo26n.py
3_models_pt/baseline/GEN_baseline_best.pt
3_models_pt/baseline/SNOW_baseline_best.pt
```

External inputs:
- raw images and annotation CSV
- the starting YOLO26n `.pt` checkpoint to train from
- dataset YAML for training dataset

## Step 1 — Convert MIO-TCD to YOLO format

From the repository root:

```bash
python 2_scripts/1.1_convert_mio_to_yolo.py \
  --image-dir /path/to/MIO-TCD-Localization/train \
  --ann-csv /path/to/gt_train.csv \
  --out 1_data/mio_yolo_11000_seed42 \
  --total-images 11000 \
  --train-images 7700 \
  --val-images 1650 \
  --test-images 1650 \
  --seed 42
```

The converter deterministically samples/splits the selected images using seed 42.

## Step 2 — Train YOLO26n

```bash
python 2_scripts/1.2_train_yolo26n.py \
  --model /path/to/starting_yolo26n.pt \
  --data 1_data/mio_yolo_11000_seed42/mio_tcd.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --project 3_models_pt/training_runs \
  --name baseline_light
```

Add `--run-test` only when a final test-split evaluation is required.

## Output

- `best.pt` and `last.pt`
- training `args.yaml`
- `results.csv`
- dataset YAML
