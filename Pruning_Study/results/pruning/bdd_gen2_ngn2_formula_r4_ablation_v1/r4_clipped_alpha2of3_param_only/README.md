# BDD Formula R4 clipped weighted ranking

Status: completed evidence-only Formula R4 ranking. This folder is separate
from the existing `BDD_UPDATED_GEN_PROTECTED_V1` ranking and does not alter
T1/T2 evidence, checkpoints, raw-T5 models, or the completed current T7 runs.

## Scope

The ranking uses paired **GEN2 and NGN2** isolated-pruning evidence. NGN2 is
the planned recovery-only domain for the following formula ablation; it is not
the sole input to this cross-domain score. Generated local-pruning ratios:
12.5%, 25%, 37.5%.

## Formula

```text
NAD_d,i = (mAP_d,baseline - mAP_d,pruned,i) / mAP_d,baseline
D_d,i   = max(0, NAD_d,i)
alpha   = 2/3
S_i     = (2/3) * max(0, NAD_GEN2) + (1/3) * max(0, NAD_NGN2)
R_i     = removed dependency-group parameters_i / baseline parameters
P_i     = R_i / (S_i + epsilon)
epsilon = 0.001
```

Signed negative NAD remains reported but contributes zero to sensitivity.

`epsilon` is recorded explicitly as a predeclared Formula R4 component. It
keeps all signed-formula denominators positive across the completed BDD T1/T2
evidence, including the small negative signed sensitivities at 12.5% and 25%.
Because it can affect the rank of near-zero-sensitivity groups, it must not be
silently changed before the later raw-56% construction.

## Files

- `FORMULA_CONFIG.json`: exact formula, hashes, and baseline references.
- `<ratio>/tables/T3_R4_*.csv`: signed observations and Formula R4 sensitivity.
- `<ratio>/tables/T4_R4_*.csv`: parameter-only Formula R4 prunability ranking.
- `<ratio>/tables/PRUNING_SEQUENCE_R4_*.csv`: ranking order only; not a
  cumulative mask plan and not an accuracy result.
- `VALIDATION_AUDIT.json`: evidence-integrity checks and generated-file hashes.

Before recovery, a separate raw-T5 builder must derive a new dependency-safe
56% architecture from this ranking and freeze the resulting exact masks. That
architecture must not reuse the current-formula masks.
