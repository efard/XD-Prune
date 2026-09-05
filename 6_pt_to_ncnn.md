# 6 — PT to NCNN export

- FP32/FP16/INT8 × 320/480/640 model export
- extra INT8 NCNN export (input resolution through `--imgsz`)
- export of Layer Replacement checkpoints when `layer_replacement_adapter.py` is importable

## Required files

```text
2_scripts/6.1_export_baseline_ncnn.py
2_scripts/6.2_export_pt_to_int8_ncnn.py
2_scripts/layer_replacement_adapter.py
3_models/<source checkpoint>.pt
```

For INT8, also provide an NCNN tools directory containing:

```text
ncnn2table
ncnn2int8
```

The INT8 exporter also imports OpenCV (`cv2`).

## FP32 or Fp16 export by `export_baseline_ncnn.py`

Before running, set the `model_path` in the script to the desired `.pt` checkpoint. The archived script currently exports at 640×640 with FP32 settings, change to FP32/FP16 × 320/480/640 if needed.

Then run:

```bash
python 2_scripts/6.1_export_baseline_ncnn.py
```

## INT8 export

Example at 320×320:

```bash
python 2_scripts/6.2_export_pt_to_int8_ncnn.py \
  --model 3_models_pt/layer_replacement/p10/GEN_C007_recovered_best.pt \
  --calib /path/to/calibration/images \
  --ncnn-tools /path/to/ncnn/tools \
  --output 3_models_ncnn/LR_GEN_P10_320_INT8 \
  --imgsz 320 \
  --threads 8 \
  --max-calib 10000
```

For another resolution, change only `--imgsz` and output directory, for example:

```text
--imgsz 480
--imgsz 640
```

Each final NCNN directory should contain:

```text
model.ncnn.param
model.ncnn.bin
metadata.yaml
```
