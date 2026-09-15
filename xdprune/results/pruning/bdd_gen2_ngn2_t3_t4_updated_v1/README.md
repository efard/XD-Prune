# BDD GEN-2 / NGN-2 Updated T3/T4 Decision Tables

Status: complete. This evidence-only analysis reads the completed, immutable
BDD T1/T2 isolated-pruning records at 12.5%, 25%, and 37.5%. It does not train,
prune, fine-tune, or edit the source evidence.

## Revised formula

```text
NAD_d,i = (mAP_d,baseline - mAP_d,pruned,i) / mAP_d,baseline
D_d,i   = max(0, NAD_d,i)
S_i     = max(D_GEN2,i, (D_GEN2,i + D_NGN2,i) / 2)
RP_i    = removed_parameters_i / baseline_parameters
RF_i    = removed_GFLOPs_i / baseline_GFLOPs
B_i     = sqrt(RP_i * RF_i)
P_i     = B_i / (B_i + S_i)
```

GEN2 is protected by `S`: a high GEN2 harm cannot be offset by a lower NGN2
harm. Signed negative AD/NAD values remain visible in T3 but contribute zero
harm, consistent with the completed paired-bootstrap analysis.

`T3_UPDATED_*.csv` contains signed and nonnegative damage plus sensitivity.
`T4_UPDATED_*.csv` ranks all 51 groups by the revised bounded prunability.
`PRUNING_SEQUENCE_UPDATED_*.csv` freezes the ranking only. It is **not** an
executed cumulative mask plan and is not evidence of a 56% model.

## Next experimental stages

1. Use the 25% T4 sequence for a matched approximately 10.25% raw cumulative
   validation (new BDD T5 pilot).
2. If structurally valid, construct a separate 56% raw cumulative BDD T5 plan
   from a predeclared ratio-specific T4 sequence and freeze its exact masks.
3. Evaluate raw T5, then run end-only T6 and staged T7 recovery on those exact
   same frozen architectures.
