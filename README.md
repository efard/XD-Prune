# TrafficYOLO: Efficient YOLO26n Research Workflows

This repository combines the two reproducible research contributions that
form the TrafficYOLO project:

- [`hardware/`](hardware/): YOLO26n profiling, robustness experiments,
  structured-compression studies, NCNN export, and PYNQ-Z2 software-side
  deployment workflow contributed by Ho Yin.
- [`xdprune/`](xdprune/): the XD-Prune dependency-aware structured-pruning
  study, including frozen configurations, ranking and recovery evidence,
  matched baselines, final paper model packages, and deployment utilities
  contributed by Rafed.

The two directories retain their original internal layouts so that their
documentation and scripts remain traceable to the recorded experiments.

## Start here

| Goal | Location |
|---|---|
| Understand the hardware/profiling workflow | [`hardware/README.md`](hardware/README.md) |
| Reproduce the XD-Prune study | [`xdprune/docs/REPRODUCIBILITY.md`](xdprune/docs/REPRODUCIBILITY.md) |
| Read the XD-Prune method and result boundaries | [`xdprune/docs/METHOD.md`](xdprune/docs/METHOD.md), [`xdprune/docs/RESULTS.md`](xdprune/docs/RESULTS.md) |
| Locate approved XD-Prune paper checkpoints and NCNN exports | [`xdprune/models/README.md`](xdprune/models/README.md) |
| Run the PYNQ-Z2 NCNN software-side workflow | [`xdprune/docs/DEPLOYMENT.md`](xdprune/docs/DEPLOYMENT.md) |

## Reproducibility and interpretation

Install a PyTorch and torchvision combination compatible with the target
system, then install the additional packages in
[`requirements.txt`](requirements.txt). The recorded XD-Prune GPU runs used
Ultralytics 8.4.127, torch-pruning 1.6.1, and the NVIDIA PyTorch 25.06
container stack; the exact environment is retained with the experiment
evidence.

Datasets, local paths, temporary outputs, and raw intermediate training
artifacts are intentionally excluded. The `xdprune/models/paper_models/`
directory contains only the curated paper model artifacts with manifests.

PYNQ-Z2 results in this repository refer to NCNN inference on the ARM
processing system unless a result explicitly says otherwise. They must not
be interpreted as programmable-logic accelerator measurements.

## Repository map

```text
hardware/  Ho Yin's profiling, robustness, compression, NCNN, and PYNQ workflow
xdprune/  Rafed's dependency-aware pruning study and approved paper artifacts
```

Each component has its own README and preserves its original evidence paths.
The branches `hoyin/yolo26n` and `rafed/yolo26n-xdprune` remain preserved as
the source histories for this integration.
