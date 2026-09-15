# Recovery Runs

This tree has two distinct purposes:

1. `recovery_10pct/` retains the legacy shared engine modules imported by the
   staged T7 runners. It is implementation support, not a claim that the paper
   reports a 10% result.
2. `recovery_56pct/` contains compact evidence for the 12 completed pruned-model
   recovery runs used by the paper.

For each reported run, the repository preserves the experiment manifest, the
five-stage summary, the final 70-epoch summary, and the final-best evaluation
record. Final checkpoints and NCNN exports are stored separately under
`../../models/paper_models/`. Per-epoch logs, stage checkpoints, optimizer
state, and duplicate model copies are intentionally excluded.

The final GEN unrestricted Global-L1 comparison used a separate 20-epoch
single-stage recovery protocol and is therefore documented under
`../competitors/global_l1_p56_reproduction/`, not misrepresented as a T7 run.

The historical folder name `T7_snow_50ep` is retained so existing source paths
remain valid. Its final manifest and result table record the completed
6-6-8-10-40 schedule (70 total epochs).
