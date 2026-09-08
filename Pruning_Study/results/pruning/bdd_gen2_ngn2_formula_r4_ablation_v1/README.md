# BDD Formula R4 ranking ablation

This folder contains frozen T3/T4 ranking evidence derived from
`260819_Formula_R4.docx` for BDD GEN2 and NGN2.

- `r4_signed_alpha2of3_param_only` retains signed negative NAD in the weighted
  sensitivity: `(2/3) * NAD_GEN2 + (1/3) * NAD_NGN2`.
- `r4_clipped_alpha2of3_param_only` clips each NAD at zero before the same
  weighting.
- Both use parameter removal only and the declared `epsilon = 0.001`.

At 37.5% local pruning, the signed and clipped T4 sequences are byte-identical,
because no group has a negative NAD in either BDD domain at that local rate.
Consequently, one 56%-target cumulative/recovery arm is sufficient for this
rate: `prepare_bdd_raw56_r4_signed.py` and
`run_bdd_t7_r4_signed_ngn2.py`.  The 12.5% rankings differ and remain as a
separate formula-level sensitivity result; they do not form a direct 56% model
under the predeclared 37.5%-local cumulative procedure.

This is an ablation artifact, not a claim of statistically significant model
improvement.  The final recovery comparison must retain the existing paired
bootstrap analysis protocol.
