GLOBAL L1 STAGE 5A V3

- 51 T4 groups
- 42 generic roots
- 9 custom roots
- 3,528 L1 channel scores per domain

But every generic root was absent from the dependency graph.

Cause
YOLO26 uses a dual one-to-many / one-to-one detection head.
The default end-to-end evaluation output can expose only the selected inference graph to Torch-Pruning.
tests complete-output and training-output tracing modes while keeping AutoGrad enabled.
