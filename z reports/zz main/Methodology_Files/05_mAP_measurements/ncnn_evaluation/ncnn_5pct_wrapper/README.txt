NCNN 5% WRAPPER

Only runtime configuration is changed:
- GEN dataset -> fixed 550-image (5%) validation subset, seed 42
- SNOW -> full 100-image validation set
- NCNN -> 4 threads/model
- sequential evaluation
- minimal confirmed NCNN Python runtime