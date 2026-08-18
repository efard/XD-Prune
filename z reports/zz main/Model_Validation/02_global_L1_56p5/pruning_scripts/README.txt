GEN ~56% GLOBAL L1 PIPELINE

- Stage 5B V2 incremental Global L1 search
- Stage 5C deterministic replay/raw model validation
- Stage 5E exact-structure 20-epoch recovery

trace is:
- model.eval()
- all model parameters require gradients
- constant 640x640 input with requires_grad=True
- Torch-Pruning output_transform flattens ALL tensors from YOLO26 output
- DependencyGraph rebuilt after every single channel pruning step

The pipeline first tries the original Stage-5 limit:
  maximum 50% removal per eligible root, minimum 4 channels.
If that cannot reach 56.4956%, it restarts the search from the
untouched GEN baseline using:
  maximum 90% removal per eligible root, minimum 4 channels.
