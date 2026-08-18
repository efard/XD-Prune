GLOBAL L1 STAGE 5B V2

1. calculates current raw L1 filter magnitudes
2. ranks all eligible channels globally
3. chooses the smallest safe candidate
4. obtains the complete DepGraph group
5. physically prunes it
6. rebuilds DepGraph
7. stops at the closest real parameter target

Every eligible root retains at least:
- 50% of its original output channels
- at least 4 output channels
This prevents one layer from being almost completely removed.
