PYNQ-Z2 YOLO26 live-video patch
================================

This patch keeps the existing validation/benchmark runner and adds a separate
TCP inference server. It is designed for these two existing model folders:

  /home/xilinx/yolo26_ps/1_models/2_baseline_full                 imgsz=640
  /home/xilinx/yolo26_ps/1_models/4_layer_replacement_320_FP16    imgsz=320

1) COPY THE PATCH FILES TO PYNQ
--------------------------------
Copy the contents of patch/1_scripts into:

  /home/xilinx/yolo26_ps/1_scripts/

The supplied yolo26_ncnn_runner.cpp restores the source file that CMake expects;
the ZIP backup had the compiled binary but the source only under Legacy/.

2) BUILD ON PYNQ
----------------
cd /home/xilinx/yolo26_ps/1_scripts
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2

Expected binaries:
  /home/xilinx/yolo26_ps/1_scripts/build/yolo26_ncnn_runner
  /home/xilinx/yolo26_ps/1_scripts/build/yolo26_ncnn_server

3) GET THE PYNQ IP ADDRESS
--------------------------
hostname -I

Use the IP reachable from the PC through the same network used by SSH/MobaXterm.

4) START ONE MODEL ON PYNQ
--------------------------
Baseline 640:
  cd /home/xilinx/yolo26_ps
  bash 1_scripts/run_live_server.sh 2_baseline_full

Optimized 320 FP16:
  cd /home/xilinx/yolo26_ps
  bash 1_scripts/run_live_server.sh 4_layer_replacement_320_FP16

The terminal should end with:
  Waiting for PC connection...

5) PREPARE PC PYTHON
--------------------
The PC script needs OpenCV:
  python -m pip install opencv-python

6) FIRST SMOKE TEST FROM PC
---------------------------
Run only 3 frames first:

  python stream_video.py --host <PYNQ_IP> --video "C:\path\video.mp4" --max-frames 3

If this works, run the whole video:

  python stream_video.py --host <PYNQ_IP> --video "C:\path\video.mp4"

Press Q in the video window to stop.

IMPORTANT PERFORMANCE NOTE
--------------------------
This first version is synchronous on purpose: each displayed frame waits for the
matching PYNQ result. It preserves exact frame/box correctness and is easiest to
debug. Based on the measurements already in the backup:

  2_baseline_full:                  ~0.095 FPS, ~10.6 s compute/frame
  4_layer_replacement_320_FP16:     ~0.423 FPS, ~2.36 s compute/frame

Therefore the video window will advance at inference speed rather than the
original video FPS. After correctness is confirmed, an asynchronous/latest-frame
version can be added without changing the PYNQ inference code.
