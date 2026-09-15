# Reproducibility Guide

Run all commands from the repository root. Install a PyTorch/torchvision pair
appropriate for the target CUDA system first, then install
`requirements.txt`. The recorded BDD100K GPU environment used Python 3,
Ultralytics 8.4.127, Torch-Pruning 1.6.1, and seed 42.

## 1. Supply external artifacts

Populate the dataset views described in `DATA.md` and place verified baseline
checkpoints as described in `../models/README.md`. Do not replace frozen inputs
without creating a new experiment version.

## 2. Inspect the frozen structural catalogues

The committed catalogues and operation manifests are under:

```text
results/depgraph/week1_safe_v3/
results/depgraph/custom_groups_v1/
```

They record 42 generic and 9 custom groups. Regenerating a catalogue is a new
audit, not an in-place edit of the committed evidence.

## 3. BDD100K isolated T1/T2 sweep

```bash
python scripts/run_bdd_t1_t2_sweep.py --preflight --all-ratios
python scripts/run_bdd_t1_t2_sweep.py --all-ratios
```

This stage is computationally expensive: it evaluates 306 isolated
group/domain/ratio combinations. Completed compact evidence is committed in
`results/pruning/bdd_gen2_ngn2_t1_t2_v1/`.

## 4. Build the proposed and R4 rankings

These steps consume completed evidence and do not train models:

```bash
python scripts/build_bdd_updated_t3_t4.py
python scripts/build_bdd_r4_formula_t4.py --variant clipped --ratio all
python scripts/build_bdd_r4_formula_t4.py --variant signed --ratio all
```

## 5. Replay the proposed architecture

```bash
python scripts/prepare_bdd_raw56_updated.py --preflight
python scripts/prepare_bdd_raw56_updated.py --run
```

Use `--resume` instead of `--run` only for a matching interrupted manifest.
The R4 path uses `prepare_bdd_raw56_r4_signed.py` with the same action pattern.

## 6. Build matched comparator rankings and architectures

Global L1 and FPGM:

```bash
python scripts/rank_bdd_l1_fpgm_competitors.py --method global_l1 --preflight
python scripts/rank_bdd_l1_fpgm_competitors.py --method global_l1 --run
python scripts/rank_bdd_l1_fpgm_competitors.py --method fpgm --preflight
python scripts/rank_bdd_l1_fpgm_competitors.py --method fpgm --run
python scripts/prepare_bdd_competitor_raw56.py --method global_l1 --preflight
python scripts/prepare_bdd_competitor_raw56.py --method global_l1 --run
python scripts/prepare_bdd_competitor_raw56.py --method fpgm --preflight
python scripts/prepare_bdd_competitor_raw56.py --method fpgm --run
```

Isomorphic Pruning + Taylor requires a CUDA device for loss-gradient
calibration:

```bash
python scripts/rank_bdd_isomorphic_taylor.py --calibration-batches 50 --preflight
python scripts/rank_bdd_isomorphic_taylor.py --calibration-batches 50 --run
python scripts/prepare_bdd_isomorphic_taylor_raw56.py --preflight
python scripts/prepare_bdd_isomorphic_taylor_raw56.py --run
```

For structural preparation, `--max-steps 1 --run` provides a limited smoke
check before the complete replay.

## 7. Matched T7 recovery

Preflight each domain before the full run. Examples:

```bash
python scripts/run_bdd_t7_recovery.py --domain GEN2 --preflight
python scripts/run_bdd_t7_recovery.py --domain GEN2 --run
python scripts/run_bdd_t7_competitor.py --method global_l1 --domain GEN2 --preflight
python scripts/run_bdd_t7_competitor.py --method global_l1 --domain GEN2 --run
python scripts/run_bdd_t7_isomorphic_taylor.py --domain GEN2 --preflight
python scripts/run_bdd_t7_isomorphic_taylor.py --domain GEN2 --run
```

Repeat for NGN2 and for FPGM. R4 uses the domain-specific
`run_bdd_t7_r4_signed_*.py` wrapper. These are long GPU runs and should be
executed through the site's scheduler or another persistent session.

## 8. Verify and summarize

```bash
python scripts/measure_bdd_isomorphic_taylor_structural_gflops.py
python scripts/measure_bdd_r4_structural_gflops.py
python scripts/build_bdd_t7_full_method_comparison.py
```

Compare generated hashes, parameter counts, pruning reductions, and evaluation
records with the aggregate CSVs. A result should not be reported merely because
a training directory exists; require a completed manifest and final evaluation.
