MODEL VALIDATION

GEN full validation images: 11000
SNOW full validation images: 100
GEN NCNN subset images: 550
GEN NCNN subset seed: 42

the same validation images were used for every model.

Global L1 ~10.25%:
- dependency-aware incremental Global L1
- 42 GENERIC roots
- smallest current L1 channel selected globally
- DepGraph rebuilt after each prune
- 50% per-root cap, min 4 channels

Global L1 ~56.5%:
- exact Stage-5 Global L1 pipeline; see collected summary/scripts for actual aggressive cap used

Layer Replacement ~56.5%:
- rank by parameter reduction (%) / prior single-layer GEN accuracy drop
- greedily add layers until total reduction exceeds 50%
- selected order in completed experiment: 22 -> 9 -> 8 -> 20 -> 7

NCNN:
- conf=0.001
- NMS IoU=0.70
- max_det=300
- rect=True
- augment=False
- batch=1
- no statistical mAP confidence interval
