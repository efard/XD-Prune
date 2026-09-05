# TrafficYOLO

YOLO on PYNQ for Smart Traffic Systems.

## .md files

- **[Model Training](1_tarin.md)** — Convert MIO-TCD dataset to YOLO format and train baseline YOLO26n models.
- **[Profiling](2_profiling.md)** — Layer-wise and end-to-end latency profiling of GEN and SNOW baselines.
- **[Error Injection](3_error_injection.md)** — Gaussian activation error injection experiments for robustness analysis.
- **[(Limited) Global L1 Pruning](4_custom_global_L1.md)** — Incremental target search, replay, recovery, and GEN P56 pipeline.
- **[Layer Replacement Pruning](5_layer_replacement.md)** — Baseline validation, dual-domain sweeps, near-target combinations, and efficiency-greedy runs.
- **[PT → NCNN Export](6_pt_to_ncnn.md)** — Export checkpoints to NCNN in FP32/FP16/INT8 formats at multiple resolutions.
- **[Size, Speed & Accuracy](7_size_speed_accuracy.md)** — Measure NCNN model size, runtime speed, and accuracy on server and PYNQ.
- **[PC ↔ PYNQ Live Video](8_pc_pynq_video.md)** — Run live video inference via TCP between PC client and PYNQ NCNN server.

## How to Use

1. Start with **training** to produce baseline models.  
2. Run **profiling** and **error injection** for performance and robustness insights.  
3. Apply **Global L1** or **Layer Replacement pruning** for pruned models.  
4. Export pruned models to **NCNN** for deployment.  
5. Evaluate **size, speed, and accuracy** on server or PYNQ board.  
6. Optionally, test **live video streaming** from PC to PYNQ.

---

For detailed instructions, please read each corresponding `.md` file listed above.  
