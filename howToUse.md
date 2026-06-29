# environment
```
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
## nvidia gpu
```
nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```
### check
```
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## ultralytics
```
pip install -U ultralytics
```
### module load opencv/4.13.0
```
pip install ultralytics --no-deps
pip install numpy pandas matplotlib pyyaml pillow psutil tqdm requests scipy
pip install "polars>=0.20.0" "nvidia-ml-py>=12.0.0" "ultralytics-thop>=2.0.18"
yolo checks
```
### if re-install later
```
pip install -r /home/afm176/yolo_project/requirements.txt
```
### check
```
python -c "import cv2; print('cv2:', cv2.__version__)"
python -c "from ultralytics import YOLO; print('YOLO import ok')"
```

## final checking
```
python -V
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import ultralytics, pandas, yaml, PIL, psutil; print('packages ok')"
```

---

# extract
## MIO-TCD-Localization.tar
```
tar -xvf 1_data/raw/MIO-TCD-Localization.tar -C 1_data/extracted/MIO-TCD-Localization
```
## snow dataset (pending)

---

# variables
## initial
```
export IMG_DIR="1_data/extracted/MIO-TCD-Localization/train"
export ANN_CSV="1_data/extracted/MIO-TCD-Localization/gt_train.csv"
export YOLO_DATA="1_data/mio_yolo_11000_seed42"
```
## after train
```
export BEST="runs/baseline/baseline_yolo26n_mio_11000_seed42/weights/best.pt"
export LAST="runs/baseline/baseline_yolo26n_mio_11000_seed42/weights/last.pt"
```

---

# convert data to yolo format
```
python 1_scripts/1.1_convert_mio_to_yolo.py \
  --image-dir "$IMG_DIR" \
  --ann-csv "$ANN_CSV" \
  --out "$YOLO_DATA" \
  --total-images 11000 \
  --train-images 7700 \
  --val-images 1650 \
  --test-images 1650 \
  --seed 42 \
  --copy-mode symlink \
  2>&1 | tee logs/01_convert_mio_to_yolo.log
```
## verify
```
cat "$YOLO_DATA/split_summary.csv"
cat "$YOLO_DATA/class_balance.csv"
cat "$YOLO_DATA/primary_class_balance.csv"
cat "$YOLO_DATA/mio_tcd.yaml"

find "$YOLO_DATA/images/train" -type l -o -type f | wc -l
find "$YOLO_DATA/images/val" -type l -o -type f | wc -l
find "$YOLO_DATA/images/test" -type l -o -type f | wc -l
```
should be something like:
train: 7700,
val: 1650,
test: 1650

---

# train baseline YOLO26n
## test
```
python 1_scripts/1.3_train_yolo26n.py \
  --model 1_models/yolo26n_official.pt \
  --data "$YOLO_DATA/mio_tcd.yaml" \
  --epochs 1 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --project runs/debug \
  --name debug_gpu_batch16_test \
  2>&1 | tee logs/01_debug_gpu_batch16_test.log
```
## real run
```
python 1_scripts/1.3_train_yolo26n.py \
  --model 1_models/yolo26n_official.pt \
  --data "$YOLO_DATA/mio_tcd.yaml" \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --project runs/baseline \
  --name baseline_yolo26n_mio_11000_seed42 \
  2>&1 | tee logs/02_train_baseline_yolo26n.log
```

---

# profiling / layer latency
## layer latency
```
python 1_scripts/2.1_profile_yolo26n_layers.py \
  --model "$BEST" \
  --imgsz 640 \
  --device 0 \
  --warmup 20 \
  --iters 100 \
  --seed 42 \
  --out 2_profiling/yolo26n_layer_latency_seed42.csv \
  2>&1 | tee logs/03_profile_layer_latency.log
```

## end to end
```
python 1_scripts/2.2_profile_yolo26n_end_to_end.py \
  --model "$BEST" \
  --data "$YOLO_DATA/mio_tcd.yaml" \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --project 2_profiling \
  --name end_to_end_val_seed42 \
  --out 2_profiling/yolo26n_end_to_end_val_seed42.csv \
  2>&1 | tee logs/04_profile_end_to_end_val.log
```

---

# error injection / mAP50-95 sensitivity check
## test
```
python 1_scripts/2.3_error_injection_yolo26n.py \
  --model "$BEST" \
  --data "$YOLO_DATA/mio_tcd.yaml" \
  --split val \
  --target-layers 0,1,2 \
  --noise-std-ratio 0.01 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --out 2_error_inj/yolo26n_error_injection_smoke_val_seed42.csv \
  2>&1 | tee logs/05_error_injection_smoke.log
```
## full run
```
python 1_scripts/2.3_error_injection_yolo26n.py \
  --model "$BEST" \
  --data "$YOLO_DATA/mio_tcd.yaml" \
  --split val \
  --noise-std-ratio 0.01 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --seed 42 \
  --out 2_error_inj/yolo26n_error_injection_all_val_seed42_noise001.csv \
  2>&1 | tee logs/06_error_injection_all_val.log
```
