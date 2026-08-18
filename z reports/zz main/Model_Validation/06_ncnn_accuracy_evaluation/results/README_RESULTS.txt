NCNN ACCURACY SWEEP RESULTS

Main report file:
  ncnn_accuracy_results.csv

Sorted comparison:
  ncnn_accuracy_ranked_by_map50_95.csv

Per-class results:
  ncnn_accuracy_per_class.csv

Summary:
  ncnn_accuracy_summary.json

Accuracy metrics:
- mAP50-95: main overall detection accuracy metric
- mAP50: AP at IoU=0.50
- mAP75: stricter localization accuracy
- Precision: fraction of detections that are correct
- Recall: fraction of ground-truth objects detected

The input size is read from each NCNN model's metadata.yaml.
GEN models are evaluated on the fixed 11,000-image GEN val split.
SNOW models are evaluated on the fixed 100-image SNOW val split.
