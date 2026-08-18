GLOBAL L1 STAGE 5C

1. Replays every planned dependency-aware channel removal.
2. Verifies every step against the recorded:
   - root
   - current channel index
   - L1 score
   - root channel count
   - total parameter count
3. Saves raw structurally pruned .pt checkpoints.
4. Reloads each checkpoint.
5. Verifies architecture hash, parameter count and forward pass.
6. Runs raw GEN and SNOW validation.
7. Calculates signed accuracy drop and accuracy retention.

Methodological limit
The 9 custom C3k2/C2PSA groups remain excluded because their implementation
was not included in the reproduction package.
