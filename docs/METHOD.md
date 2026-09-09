# XD-Prune Method

## Dependency-safe candidate space

YOLO26n cannot be pruned safely by treating every convolution independently.
Residual paths, concatenations, distributional regression, and attention
blocks couple channel dimensions. The study therefore begins with a physically
validated catalogue of 51 structural candidates:

- 42 generic dependency-graph groups (`DGxxx`);
- 9 custom C3k2/C2PSA block-aware groups (`CDGxxx`).

The exact operations are in `results/depgraph/`. `DG001` (the stem) remains
protected, and `DG020` is excluded by the completed cumulative-collapse audit.

## Isolated T1/T2 evidence

For each candidate group (i), domain (d), and local pruning fraction
(r\in\{0.125,0.25,0.375\}), the runner reloads a fresh baseline, prunes only
that group, and measures validation mAP50:95 without fine-tuning or BatchNorm
updates. The BDD100K study completed 306 measurements:

```text
51 groups x 2 domains x 3 local pruning fractions = 306 runs
```

Let (M_{d,0}) be baseline mAP50:95 and (M_{d,i}) the isolated-pruning
result. The signed normalized accuracy drop is

```text
NAD_d,i = (M_d,0 - M_d,i) / M_d,0
D_d,i   = max(0, NAD_d,i)
```

Small apparent improvements are retained in the evidence as signed values but
are not treated as pruning gains. The selected negative-AD cases were checked
with 5,000 paired, class-preserving bootstrap resamples; their confidence
intervals crossed zero.

## Proposed ranking

The GEN-protected cross-domain sensitivity is

```text
S_i = max(D_GEN2,i, (D_GEN2,i + D_NGN2,i) / 2)
```

Structural benefit combines parameter and GFLOP reductions:

```text
RP_i = removed_parameters_i / baseline_parameters
RF_i = removed_GFLOPs_i / baseline_GFLOPs
B_i  = sqrt(RP_i * RF_i)
```

The bounded prunability priority is

```text
P_i = B_i / (B_i + S_i)
```

Candidates are ranked by descending priority, then descending benefit,
ascending sensitivity, and group identifier. A shared cumulative order is
frozen before structural replay. The selected 37.5% local-group operating
point yields approximately 56% whole-model parameter reduction.

## R4 ablation

The signed R4 alternative uses an explicit GEN2 preference:

```text
S_i = (2/3) NAD_GEN2,i + (1/3) NAD_NGN2,i
R_i = removed_parameters_i / baseline_parameters
P_i = R_i / (S_i + 0.001)
```

Clipped and signed R4 produced the same 37.5% ranking sequence in this study;
the negative normalized drops were too small to alter the ordering. The final
R4 and proposed results are consequently extremely close and should be
interpreted as an ablation, not as evidence of a meaningful separation.

## Matched comparators

Global L1, FPGM, and Isomorphic Pruning + Taylor use the same dependency-safe
candidate space, protected groups, approximate parameter target, cumulative
structural replay, evaluation protocol, seed, and recovery budget. The ranking
criterion is the controlled variable.

- **Global L1:** group-aware weight magnitude.
- **FPGM:** filter redundancy represented by geometric-median proximity.
- **Isomorphic Pruning + Taylor:** a controlled YOLO26n adaptation of the
  topology-aware Isomorphic Pruning framework using first-order Taylor
  importance. It is labelled an adaptation because the original method targets
  other model families and the custom YOLO26n groups require this project's
  validated coupled rules.

RGP is not reported as a completed comparator. Its advertised source repository
was unavailable during the study, and no author-provided implementation was
received. The retained RGP-named helper supplies shared Taylor-calibration
utilities only; no RGP result is claimed.

## Recovery

All matched BDD100K methods use the same T7 staged recovery schedule:

```text
6 + 6 + 8 + 10 + 40 = 70 epochs
```

This controls recovery opportunity while preserving each method's frozen
architecture.
