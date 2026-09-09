# Results and Interpretation

## Initial GEN/SNOW study

The initial MIO-TCD GEN and ACDC SNOW experiments established the cross-domain
workflow before the matched BDD100K study. At approximately 56.8% parameter
reduction, XD-Prune retained 94.62% of GEN baseline mAP50:95 and 76.79% of
SNOW baseline mAP50:95. Exact values are in
`results/tables/gen_snow_xdprune_summary.csv`.

GEN and SNOW use different datasets and class taxonomies. Their absolute mAP
values should therefore be interpreted within each domain rather than averaged
into one benchmark score.

## Matched BDD100K study

At the approximately 56% parameter-reduction operating point, XD-Prune achieved
the highest mAP50:95 retention among the proposed ranking and three matched
comparators in both domains:

| Domain | XD-Prune | R4 signed | Global L1 | FPGM | Isomorphic + Taylor |
|---|---:|---:|---:|---:|---:|
| GEN2 mAP50:95 | **0.2499** | 0.2497 | 0.2356 | 0.2395 | 0.2420 |
| GEN2 retention | **85.46%** | 85.40% | 80.57% | 81.92% | 82.78% |
| NGN2 mAP50:95 | **0.2300** | 0.2293 | 0.2060 | 0.2085 | 0.2082 |
| NGN2 retention | **90.00%** | 89.73% | 80.59% | 81.58% | 81.44% |

The strongest supported result is accuracy retention at a closely matched
parameter-reduction target. The methods are not GFLOP-matched: the classical
and Isomorphic-Taylor comparators remove more estimated computation than
XD-Prune. Exact unrounded measurements are in
`results/tables/bdd100k_t7_matched_results.csv`.

## Deployment result

All latency entries in `results/tables/pynq_z2_ps_ncnn_latency.csv` are measured
NCNN compute latency on the PYNQ-Z2 ARM processing system. They are not
programmable-logic acceleration results. The measurements show that structural
pruning reduces practical software-path latency, while also showing that
parameter count alone does not predict latency: operator shapes and runtime
implementation matter.

## Limitations

- The matched BDD100K comparison currently uses one fixed seed (`42`), so it
  does not support variance or statistical-significance claims across runs.
- Comparison is parameter-matched rather than GFLOP- or latency-matched.
- The Isomorphic-Taylor result is a controlled adaptation, not an official
  author implementation for YOLO26n.
- Dataset licenses prevent redistribution of data, and trained checkpoints are
  distributed separately from this source branch.
- PYNQ-Z2 values describe ARM/NCNN execution, not FPGA PL acceleration.
