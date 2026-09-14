# Selected Output Figures

These seven compact figures are included for quick visual inspection. The
authoritative numerical evidence remains in the linked result tables and
experiment manifests.

| Figure | What it shows | Interpretation boundary |
|---|---|---|
| `bdd_gen2_proposed_t7_stage5_training.png` | GEN2 proposed-method curves during the final recovery stage | Shows final-stage optimization behaviour; it is not a direct method comparison. |
| `bdd_ngn2_proposed_t7_stage5_training.png` | NGN2 proposed-method curves during the final recovery stage | Shows final-stage optimization behaviour; it is not a direct method comparison. |
| `bdd_gen2_baseline_precision_recall.png` | Per-class and aggregate precision-recall curve for the GEN2 baseline validation | Baseline validation output only; use the results tables for retained-accuracy comparisons. |
| `bdd_gen2_baseline_validation_predictions.jpg` | GEN2 baseline validation prediction grid with detected bounding boxes | Qualitative baseline output only; it is not a proposed-method or pruned-model comparison. |
| `bdd_ngn2_baseline_validation_predictions.jpg` | NGN2 baseline validation prediction grid with detected bounding boxes | Qualitative baseline output only; it is not a proposed-method or pruned-model comparison. |
| `signed_ad_bn_recalibration_audit.png` | Effect of controlled batch-normalization recalibration on signed isolated-pruning accuracy differences | An audit of the ranking evidence, not a recovery or deployment result. |
| `signed_ad_bootstrap_intervals.png` | Paired 95% bootstrap intervals for the signed accuracy-difference audit | Indicates uncertainty for the listed isolated-pruning cases; it does not establish a whole-model result. |

Relevant machine-readable evidence:

- [`../results/tables/bdd100k_t7_matched_results.csv`](../results/tables/bdd100k_t7_matched_results.csv)
- [`../results/pruning/`](../results/pruning/)
- [`../experiments/`](../experiments/)
