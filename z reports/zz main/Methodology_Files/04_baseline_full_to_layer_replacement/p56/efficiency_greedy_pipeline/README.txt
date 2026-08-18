GEN >=50% WHOLE-LAYER REPLACEMENT, PARAM/AD GREEDY

Selection rule
For each previously tested GEN layer:
    Efficiency = Parameter reduction (%) / GEN mAP50-95 accuracy drop

Sort from highest efficiency to lowest, then add one layer at a time
until the cumulative parameter reduction reaches or exceeds 50%.

Using the results, the expected ranking begins:
Rank  Layer  Type   Reduction %   GEN AD      Param% / AD   Cumulative %
1     22     C3k2   18.464409     0.210247    87.822334     18.464409
2      9     SPPF    6.563082     0.101337    64.764598     25.027491
3      8     C3k2   13.799824     0.490194    28.151778     38.827315
4     20     Conv    5.889422     0.209843    28.065839     44.716737
5      7     Conv   11.778844     0.508404    23.168292     56.495580

Expected selection order:
    22 -> 9 -> 8 -> 20 -> 7
Expected parameter reduction:
    about 56.4956%

Full workflow
1. Load a fresh unfused GEN baseline.
2. Rank eligible layers by Param% / GEN AD.
3. Greedily select layers until cumulative reduction >= 50%.
4. Build and save the unfused raw replacement model.
5. Verify BatchNorm and YOLO26 training head remain present.
6. Evaluate baseline GEN mAP using a throwaway model instance.
7. Evaluate raw cumulative GEN mAP using a throwaway model instance.
8. Fresh-load the untouched raw checkpoint.
9. Run 20-epoch exact-structure recovery.
10. Run final explicit GEN validation.
11. Save raw/recovered PT models, ranking CSV, selected layer CSV, training CSV, report table and summary.