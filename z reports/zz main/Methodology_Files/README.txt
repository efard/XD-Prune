
This archive is organized around five workflows:

01_baseline_light_training
  Training/data-preparation scripts and available args/results related to the baseline-light model.

02_profiling_and_error_injection
  Profiling, benchmark, error-injection scripts and result/config files.

03_baseline_full_to_global_L1
  p10:
    scripts, pruning plans, incremental channel-selection logs, protocols, args/results, group-scope manifests.
  p56:
    56% Global L1 pipeline, replay/selection files, args/results.

04_baseline_full_to_layer_replacement
  p10:
    Physical layer-replacement scripts, C007 selection evidence, training args/results.
  p56:
    Efficiency-greedy layer-replacement pipeline, layer efficiency ranking, selected layer list, args/results.

05_mAP_measurements
  PT/NCNN evaluation scripts, dataset YAMLs, fixed validation manifest, args/results.