# Final Paper Models

`paper_models/` contains only the 17 model configurations used in the final
paper tables: the GEN, SNOW, GEN2, and NGN2 baselines; the corresponding
XD-Prune models; the GEN Global-L1 comparison; and the matched BDD100K R4,
Global-L1, FPGM, and Isomorphic-Taylor models.

Each model directory contains:

- `model.pt`: final PyTorch checkpoint;
- `model.ncnn.param` and `model.ncnn.bin`: NCNN export used for deployment;
- `metadata.yaml`: model metadata and class names;
- `artifact_manifest.json`: role, export versions, sizes, and SHA-256 hashes.

[`paper_models/MODEL_INDEX.csv`](paper_models/MODEL_INDEX.csv) is the central
machine-readable index. Verify any artifact against that index or its local
manifest before using it. Intermediate checkpoints, optimizer state, raw logs,
datasets, and models not reported by the paper remain excluded.

The bundle can be regenerated from the verified private experiment archive
with `scripts/package_paper_models.py`; the script verifies the source
checkpoint hashes and removes workstation-specific paths from public metadata.
