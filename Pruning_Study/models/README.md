# External Model Artifacts

Trained checkpoints are intentionally not committed. Place separately supplied
artifacts at the exact paths expected by the scripts and verify them before use.

## Frozen BDD100K baselines

| Domain | Expected path | SHA-256 |
|---|---|---|
| GEN2 | `Pruning_Study/results/baselines/b_gen2_bdd_clear_alltime_yolo26n_s42_v1/weights/best.pt` | `863520B6AC857D934EFE910035EF3FBFC3DCF927E9D9EAC29407AE1BCE42D4EA` |
| NGN2 | `Pruning_Study/results/baselines/b_ngn2_bdd_adverse_alltime_yolo26n_s42_v1/weights/best.pt` | `6C5F7580651AFD48862E598F649BAF6D4D7CF23CEEADC5C42B105546AF72C27D` |

The corresponding GEN and SNOW paths are specified by the frozen configuration
files under `configs/baselines/` and `configs/pruning/`. Treat any changed
checkpoint as a new experimental input.

## Verification

Linux:

```bash
sha256sum path/to/checkpoint.pt
```

PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 "path\to\checkpoint.pt"
```

Do not commit checkpoints, exported NCNN files, private download links, or
credentials. For a public release, publish permitted model artifacts through a
versioned release or archival service and cite their hashes here.
