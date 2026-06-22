import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO


def describe_shape(value):
    # YOLO layers may return Tensor, list, tuple, or dict.
    # This recursive function converts the output structure into readable shapes.
    if isinstance(value, torch.Tensor):
        return str(list(value.shape))

    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(describe_shape(item) for item in value) + "]"

    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {describe_shape(val)}" for key, val in value.items()) + "}"

    return type(value).__name__


def sync_if_needed(device):
    # CUDA execution is asynchronous, so timing without synchronization can be wrong.
    # CPU does not need synchronization.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to YOLO checkpoint, e.g. best.pt")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size used for profiling")
    parser.add_argument("--device", default="cpu", help="Use cpu or cuda:0")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up runs before measurement")
    parser.add_argument("--runs", type=int, default=50, help="Measured forward runs")
    parser.add_argument("--out", default="runs/week4_profile", help="Output folder")
    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.weights)
    model = yolo.model.to(device)
    model.eval()

    # Ultralytics detection models store the top-level YOLO layers in model.model.
    # We profile only these top-level layers to avoid double-counting nested Conv/BN/Act modules.
    try:
        ## top level layers
        layers = list(model.model)
    except TypeError as exc:
        raise TypeError("Could not read top-level YOLO layers from model.model.") from exc

    layer_times = defaultdict(list)
    layer_shapes = {}
    start_times = {}
    handles = []

    for layer_id, layer in enumerate(layers):
        def pre_hook(module, inputs, current_layer_id=layer_id):
            # Mark the starting time immediately before this top-level layer runs.
            sync_if_needed(device)
            ## start
            start_times[current_layer_id] = time.perf_counter()

        def post_hook(module, inputs, output, current_layer_id=layer_id):
            # Measure elapsed time immediately after this top-level layer finishes.
            sync_if_needed(device)
            ## end
            elapsed_ms = (time.perf_counter() - start_times[current_layer_id]) * 1000.0
            layer_times[current_layer_id].append(elapsed_ms)
            layer_shapes[current_layer_id] = describe_shape(output)

        handles.append(layer.register_forward_pre_hook(pre_hook))
        handles.append(layer.register_forward_hook(post_hook))

    ## Use a dummy input [batch, channel, height, width] = [1, 3, 416, 416]
    dummy_input = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)

    with torch.inference_mode():
        # Warm-up is important because the first few runs may include initialization overhead.
        for _ in range(args.warmup):
            model(dummy_input)

        for values in layer_times.values():
            values.clear()

        for _ in range(args.runs):
            model(dummy_input)

    for handle in handles:
        handle.remove()

    rows = []

    for layer_id, layer in enumerate(layers):
        values = layer_times[layer_id]

        if not values:
            raise RuntimeError(f"No timing data was recorded for layer {layer_id}.")

        mean_latency_ms = sum(values) / len(values)
        variance = sum((value - mean_latency_ms) ** 2 for value in values) / len(values)
        std_latency_ms = variance ** 0.5

        layer_type = layer.__class__.__name__
        layer_name = getattr(layer, "type", layer_type)
        from_layer = getattr(layer, "f", "")
        parameter_count = sum(parameter.numel() for parameter in layer.parameters())

        rows.append({
            "layer_id": layer_id,
            "from": from_layer,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "output_shape": layer_shapes.get(layer_id, ""),
            "parameters": parameter_count,
            "mean_latency_ms": mean_latency_ms,
            "std_latency_ms": std_latency_ms,
        })

    total_layer_latency = sum(row["mean_latency_ms"] for row in rows)

    for row in rows:
        row["latency_percent"] = (row["mean_latency_ms"] / total_layer_latency) * 100.0

    profile_csv = output_dir / "layer_latency_table.csv"
    ranking_csv = output_dir / "bottleneck_ranking.csv"

    fieldnames = [
        "layer_id",
        "from",
        "layer_name",
        "layer_type",
        "output_shape",
        "parameters",
        "mean_latency_ms",
        "std_latency_ms",
        "latency_percent",
    ]

    with profile_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with ranking_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["mean_latency_ms"], reverse=True))

    print(f"Layer latency table saved to: {profile_csv}")
    print(f"Bottleneck ranking saved to: {ranking_csv}")
    print(f"Total measured model-layer latency: {total_layer_latency:.3f} ms")


if __name__ == "__main__":
    main()

'''
python scripts/2.1_profile_yolo26n_layers.py `
  --weights runs/mio_tcd_yolo26n/v2_cpu_baseline_5p_10e_416/weights/best.pt `
  --imgsz 416 `
  --device cpu `
  --warmup 10 `
  --runs 50 `
  --out runs/week4_profile
'''