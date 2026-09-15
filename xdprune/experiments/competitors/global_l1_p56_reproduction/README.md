# Unrestricted Global L1 at approximately 56% reduction

This is the final GEN Global-L1 comparison reported by the paper. It uses the
42 eligible stock DepGraph roots, ranks channels globally by raw L1 filter
magnitude, rebuilds the dependency graph after every channel removal, applies
no per-root 50% cap, and does not use GEN/SNOW accuracy or domain information
during selection.

The compact public evidence includes the frozen experiment manifest, exact
selection sequence, generation and recovery code, final-best evaluation, and
final results table. The final checkpoint and matching NCNN export are in
`../../../models/paper_models/gen_global_l1_unrestricted_p56_640_fp32/`.

Intermediate training checkpoints, optimizer state, plots, and duplicate model
copies are intentionally excluded.
