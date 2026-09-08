# NCNN and PYNQ-Z2 Deployment

This repository contains the deployment tooling, not generated NCNN model
binaries. Exported packages are ignored because they are derived, large, and
checkpoint-dependent.

## Export on a PC

Place the external checkpoints according to `../models/README.md`, install the
compatible Ultralytics environment, and run the relevant exporter from the
repository root. Available entry points include:

```text
Pruning_Study/deployment/pc/export_ncnn.py
Pruning_Study/deployment/pc/export_bdd_t7_primary_models.py
Pruning_Study/deployment/pc/export_bdd_t7_isomorphic_taylor_models.py
Pruning_Study/deployment/pc/export_latency_table_56pct_supplement_models.py
```

Each exported package should contain at least `model.ncnn.param`,
`model.ncnn.bin`, `metadata.yaml`, and a provenance manifest. Verify hashes
before transfer.

## Prepare the board

The board utilities assume the default root `/home/xilinx/yolo26_ps`, which can
be overridden with `YOLO26_PS_ROOT`:

```text
1_dataset/       image lists and local evaluation data
1_models/        one directory per exported NCNN model
1_scripts/       board scripts and built C++ runner
2_results/       timestamped evidence directories
```

Build the NCNN tools using `deployment/board/build_ncnn_tools.sh`, then check a
model package with:

```bash
bash Pruning_Study/deployment/board/preflight_model.sh <model_name>
```

Run the fixed research benchmark with:

```bash
bash Pruning_Study/deployment/board/run_research_benchmark.sh <model_name> benchmark
```

The board runner records artifact hashes, model size, runtime configuration,
and timing output in a new result directory. Preserve each result directory as
immutable evidence.

## Interpretation boundary

The committed aggregate latency table reports NCNN compute latency measured on
the PYNQ-Z2 ARM processing system. It does **not** demonstrate FPGA
programmable-logic acceleration. Model load and warm-up are kept separate from
the reported compute path by the benchmark utility.
