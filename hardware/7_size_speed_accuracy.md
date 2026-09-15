# 7 — Model size, speed and accuracy measurement

Available components:
- server/PC NCNN accuracy evaluator
- PYNQ NCNN C++ runner
- PYNQ model-size/benchmark shell wrappers
- restored `CMakeLists.txt` for building the runner/server

Remaining setup requirements:
- NCNN Python runtime for server-side NCNN accuracy
- NCNN/OpenCV C++ installation on PYNQ
- PYNQ-local validation and benchmark image lists
- path updates in `measure_model.sh` and `run_model.sh` for the new repository structure

## Part A — NCNN accuracy on server/PC

Required:

```text
2_scripts/7.1_evaluate_all_ncnn.py
3_models/<model directories>/
1_data/dataset_GEN_local.yaml
1_data/dataset_SNOW_local.yaml
```

Edit dataset_GEN_local.yaml and dataset_SNOW_local.yaml's "..." to the downloaded dataset path before processing.

Run:

```bash
python 2_scripts/7.1_evaluate_all_ncnn.py \
  --model-root 3_models/<model directories> \
  --gen-data 1_data/dataset_GEN_local.yaml \
  --snow-data 1_data/dataset_SNOW_local.yaml \
  --output-dir 8_results_ncnn/accuracy \
  --workers 4
```

Edit NCNN model directories.

The evaluator recursively discovers directories containing `model.ncnn.param`, `model.ncnn.bin`, and `metadata.yaml`. It determines GEN/SNOW from the class names in metadata and uses each model's exported input size.

Expected outputs include:

```text
ncnn_accuracy_results.csv
ncnn_accuracy_per_class.csv
ncnn_accuracy_summary.json
```

The historical full run evaluated the fixed full validation split (GEN 11,000 images; SNOW 100 images).

### Environment note

The historical log used custom externally mounted Python package paths for NCNN. The supplied `requirements.txt` alone does not capture that runtime, so test the NCNN Python package installation explicitly on the reproduction machine.

## Part B — Build the PYNQ C++ runner

On PYNQ:

```bash
cd /path/to/repo/2_scripts_pynq
mkdir -p build
cd build
cmake .. -DNCNN_ROOT=/path/to/ncnn/install
make -j2
```

The supplied CMake project builds:

```text
yolo26_ncnn_runner
yolo26_ncnn_server
```

It targets C++14 and links NCNN, OpenCV, OpenMP, Threads, `dl`, `m`, and `atomic`.

## Part C — Prepare PYNQ image lists

Create:

```text
1_data/pynq/val_images.txt
1_data/pynq/benchmark_images.txt
```

Each line must point to an image that is actually accessible on the PYNQ board.

## Part D — Update shell-script repository paths

The archived shell scripts use the previous paths:

```text
1_models/
1_scripts/
1_dataset/
2_results/
```

For the new repository, update path definitions only so they resolve to:

```text
3_models_ncnn/
2_scripts_pynq/
1_data/pynq/
8_results_ncnn/pynq/
```

Do not change the benchmark settings unless starting a new experiment protocol.

## Part E — Measure size and speed

After the path update:

```bash
bash 2_scripts_pynq/measure_model.sh <model_name> benchmark
```

Validation only:

```bash
bash 2_scripts_pynq/measure_model.sh <model_name> validate
```

Both:

```bash
bash 2_scripts_pynq/measure_model.sh <model_name> all
```

A one-image limit may be used only as a smoke test:

```bash
bash 2_scripts_pynq/measure_model.sh <model_name> benchmark 1
```

The fixed benchmark settings in the supplied wrapper are:

```text
threads = 2
warmup = 5
repeat = 3
benchmark confidence = 0.25
validation confidence = 0.001
IoU = 0.70
max_det = 300
```

NCNN deployment size is recorded as:

```text
size(model.ncnn.param) + size(model.ncnn.bin)
```

Important output files include:

```text
model_size.csv
performance.csv
run_config.txt
model_hashes.txt
metadata.yaml
```

The PYNQ validation runner generates predictions/image summaries; use the server-side evaluator for the reported NCNN mAP unless a separate board-side metric calculator is intentionally added.
