# Repeatable PYNQ-Z2 NCNN Model Test

This package assumes that the following one-time setup is already complete:

```text
/home/xilinx/yolo26_ps/
├── 1_dataset/
├── 1_models/
├── 1_scripts/
│   ├── build/yolo26_ncnn_runner
│   └── run_model.sh
├── 2_results/
└── third_party/ncnn/
```

The new `measure_model.sh` is a wrapper around the existing runner. It keeps
every model on the same settings and packages each result separately.

## One-time installation

Copy `measure_model.sh` to:

```text
/home/xilinx/yolo26_ps/1_scripts/measure_model.sh
```

Then run:

```bash
chmod +x /home/xilinx/yolo26_ps/1_scripts/measure_model.sh
```

The provided notebook can be uploaded anywhere under the PYNQ Jupyter home
directory.

## For every new model

### 1. Prepare the model folder on the local PC

Use a short model name without spaces, for example:

```text
pruned_t6/
├── model.ncnn.param
├── model.ncnn.bin
└── metadata.yaml
```

Keep the original `.pt` file on the PC/server. It is still required for
parameter count and FLOPs.

### 2. Upload using MobaXterm

Drag the entire folder into:

```text
/home/xilinx/yolo26_ps/1_models/
```

### 3. Quick smoke test

```bash
cd /home/xilinx/yolo26_ps

bash 1_scripts/measure_model.sh \
  pruned_t6 \
  benchmark \
  1
```

### 4. Full performance benchmark

```bash
bash 1_scripts/measure_model.sh \
  pruned_t6 \
  benchmark
```

Automatic results:

```text
2_results/pruned_t6/
├── performance.csv
├── model_size.csv
├── power.csv
├── run_config.txt
├── model_hashes.txt
└── metadata.yaml
```

Downloadable package:

```text
2_results/pruned_t6_results.tar.gz
```

Main performance fields:

- `median_compute_ms`: primary latency per image
- `p95_compute_ms`: slower-case latency
- `compute_fps`: FPS excluding image-file reading
- `peak_rss_mib`: peak process memory
- `normalized_board_cpu_percent`: CPU usage normalized over both cores

### 5. Accuracy inference

Only use a validation dataset whose class IDs, names, and order match the model
metadata.

```bash
bash 1_scripts/measure_model.sh \
  pruned_t6 \
  validate
```

Additional outputs:

```text
2_results/pruned_t6/predictions.csv
2_results/pruned_t6/images.csv
```

Transfer these files to the PC/server and run the existing
`evaluate_predictions.py` to obtain mAP50-95, mAP50, Precision, Recall, and
per-class AP.

### 6. Performance and accuracy together

```bash
bash 1_scripts/measure_model.sh \
  pruned_t6 \
  all
```

## Run from Jupyter Notebook

Both MobaXterm and Jupyter call the same shell script, so the experimental
settings remain identical.

Open `measure_new_model.ipynb`, then change only:

```python
MODEL_NAME = "pruned_t6"
MODE = "benchmark"
IMAGE_LIMIT = 1
```

Use `IMAGE_LIMIT = 1` for the first test, then change it to `0` for the full
benchmark.

## Where each metric is measured

| Metric | Location |
|---|---|
| Latency, FPS, peak memory, CPU usage | PYNQ benchmark |
| NCNN deployment size | PYNQ benchmark |
| PYNQ detections | PYNQ validation |
| mAP50-95, mAP50, Precision, Recall, per-class AP | PC/server evaluator |
| Parameters, FLOPs, source `.pt` size | PC/server using original `.pt` |
| Power | External USB/DC meter during PYNQ benchmark |

`power.csv` is a blank template because the PYNQ-Z2 software environment does
not provide a reliable whole-board power sensor. Record idle, average inference,
and peak power from the external meter, then calculate:

```text
Dynamic power = average inference power - idle power
Energy per image = average inference power / compute FPS
```

## Non-technical explanation

The script performs five tasks:

1. Checks that the three required NCNN files exist.
2. Records file hashes so the result can be linked to the exact model.
3. Runs every model with the same benchmark settings.
4. Saves all CSV files in a model-specific result folder.
5. Creates one compressed archive for easy download through MobaXterm.

It does not change, prune, quantize, or retrain the model.
