GLOBAL L1 STAGE 5E, MATCHED 20-EPOCH RECOVERY

GEN:
- source parameters: 2,250,880
- full GEN training set
- 20 continuous epochs

SNOW:
- source parameters: 2,250,272
- full SNOW training set
- 20 continuous epochs

Matched settings
- AdamW
- lr0 0.001
- lrf 0.01
- momentum 0.9
- weight decay 0.0005
- warmup 1 epoch
- batch 16
- image size 640
- AMP enabled
- pretrained false
- mosaic 1.0
- close_mosaic 0
- deterministic seed 42

Final evaluation
- full validation split
- FP32
- rect true
- conf 0.001
- IoU 0.70
- max_det 300