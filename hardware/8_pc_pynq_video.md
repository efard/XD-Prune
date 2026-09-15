# Task 8 — PC-to-PYNQ live video NCNN test

## Status

**Require: PYNQ C++ dependencies are installed and the NCNN model is copied to the board.**

The restored CMake project can build the TCP NCNN server. The PC client and PYNQ server source are both present.

`stream_video.py` contains the GEN 11-class label list, edit if you are using different dataset.

## Required files

```text
2_scripts_pynq/CMakeLists.txt
2_scripts_pynq/yolo26_ncnn_server.cpp
2_scripts_pynq/yolo26_ncnn_runner.cpp
2_scripts_pynq/stream_video.py
3_models/<model_name>/model.ncnn.param
3_models/<model_name>/model.ncnn.bin
3_models/<model_name>/metadata.yaml
```

## Step 1 — Build the PYNQ server

On PYNQ:

```bash
cd /path/to/repo/2_scripts_pynq
mkdir -p build
cd build
cmake .. -DNCNN_ROOT=/path/to/ncnn/install
make -j2
```

## Step 2 — Start the PYNQ NCNN TCP server

Example for a 320×320 NCNN model:

```bash
./2_scripts_pynq/build/yolo26_ncnn_server \
  --model-dir 3_models/<model_name> \
  --imgsz 320 \
  --threads 2 \
  --port 5000 \
  --conf 0.25 \
  --iou 0.70 \
  --max-det 300
```

Edit model path / name.

Use the same input size as the exported model metadata (for example 320, 480 or 640).

## Step 3 — Run the PC client

On the PC:

```bash
python 2_scripts_pynq/stream_video.py \
  --host 192.168.2.99 \
  --video /path/to/video.mp4 \
  --port 5000 \
  --send-size 320 \
  --display-width 1280 \
  --frame-skip 4 \
  --jpeg-quality 85
```

`--send-size` should match the server/model input size.

Frame-skip examples:

```text
--frame-skip 0    process every frame
--frame-skip 9    process [0, 10, 20, ...]
```

## Data flow

```text
PC reads video frame
  -> PC resizes/JPEG-encodes frame
  -> TCP sends frame to PYNQ
  -> PYNQ NCNN server performs inference
  -> TCP returns detections
  -> PC rescales boxes to the displayed original frame
  -> PC draws labels/boxes and displays the result
```
