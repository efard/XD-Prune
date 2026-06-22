import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO


def inject_activation_error(value, sigma):
    # Reason:
    # A YOLO layer output may be a Tensor, list, tuple, or dict.
    # The function preserves the original structure and only injects error into floating-point tensors.
    if isinstance(value, torch.Tensor):
        if not torch.is_floating_point(value):
            return value

        # Operation:
        # y_injected = y + sigma * std(y) * N(0, 1)
        # This makes the noise scale relative to the current layer activation range.
        activation_std = value.detach().std()
        noise = torch.randn_like(value)
        return value + sigma * activation_std * noise

    if isinstance(value, list):
        return [inject_activation_error(item, sigma) for item in value]

    if isinstance(value, tuple):
        return tuple(inject_activation_error(item, sigma) for item in value)

    if isinstance(value, dict):
        return {key: inject_activation_error(val, sigma) for key, val in value.items()}

    return value


def read_map_metrics(metrics):
    # Reason:
    # Ultralytics validation returns detection metrics under metrics.box.
    # mAP50-95 is metrics.box.map, and mAP50 is metrics.box.map50.
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
    }


def run_validation(yolo, args):
    metrics = yolo.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
    )

    return read_map_metrics(metrics)


def parse_layer_ids(layer_ids_text, total_layers):
    # Reason:
    # During debugging, you may want to test only a few layers first.
    # Example: --layer_ids 0,1,2,3
    if layer_ids_text == "":
        return list(range(total_layers))

    layer_ids = [int(item.strip()) for item in layer_ids_text.split(",")]

    for layer_id in layer_ids:
        if layer_id < 0 or layer_id >= total_layers:
            raise ValueError(f"Layer id {layer_id} is outside valid range 0 to {total_layers - 1}.")

    return layer_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to YOLO checkpoint, e.g. best.pt")
    parser.add_argument("--data", required=True, help="Dataset YAML used for validation")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.05, help="Error strength")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer_ids", default="", help="Comma-separated layer ids. Empty means all layers.")
    parser.add_argument("--include_detect", action="store_true", help="Also inject error into Detect layer")
    parser.add_argument("--out", default="runs/week4_error_injection")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_yolo = YOLO(args.weights)

    try:
        baseline_layers = list(baseline_yolo.model.model)
    except TypeError as exc:
        raise TypeError("Could not read top-level YOLO layers from model.model.") from exc

    selected_layer_ids = parse_layer_ids(args.layer_ids, len(baseline_layers))

    print("Running clean baseline validation...")
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    baseline_metrics = run_validation(baseline_yolo, args)
    baseline_map50_95 = baseline_metrics["map50_95"]
    baseline_map50 = baseline_metrics["map50"]

    rows = []

    for layer_id in selected_layer_ids:
        reference_layer = baseline_layers[layer_id]
        layer_type = reference_layer.__class__.__name__

        # Detect is skipped by default because injecting error directly into final detection outputs
        # can dominate the result and is less useful for deciding which feature-extraction layers to optimize.
        if not args.include_detect and "Detect" in layer_type:
            continue

        print(f"Injecting error into layer {layer_id}: {layer_type}")

        # Reload the checkpoint for each layer to keep each experiment independent.
        yolo = YOLO(args.weights)
        layers = list(yolo.model.model)
        target_layer = layers[layer_id]

        torch.manual_seed(args.seed)

        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)

        def error_hook(module, inputs, output):
            # The hook returns a modified output, so the rest of YOLO receives the corrupted activation.
            return inject_activation_error(output, args.sigma)

        handle = target_layer.register_forward_hook(error_hook)

        try:
            injected_metrics = run_validation(yolo, args)
        finally:
            handle.remove()

        injected_map50_95 = injected_metrics["map50_95"]
        injected_map50 = injected_metrics["map50"]

        drop_map50_95 = baseline_map50_95 - injected_map50_95
        drop_map50 = baseline_map50 - injected_map50

        rows.append({
            "layer_id": layer_id,
            "layer_type": layer_type,
            "sigma": args.sigma,
            "baseline_map50_95": baseline_map50_95,
            "injected_map50_95": injected_map50_95,
            "drop_map50_95": drop_map50_95,
            "baseline_map50": baseline_map50,
            "injected_map50": injected_map50,
            "drop_map50": drop_map50,
        })

    rows_sorted = sorted(rows, key=lambda row: row["drop_map50_95"], reverse=True)

    output_csv = output_dir / "error_sensitivity_ranking.csv"

    fieldnames = [
        "layer_id",
        "layer_type",
        "sigma",
        "baseline_map50_95",
        "injected_map50_95",
        "drop_map50_95",
        "baseline_map50",
        "injected_map50",
        "drop_map50",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"Error sensitivity ranking saved to: {output_csv}")


if __name__ == "__main__":
    main()

'''
python scripts/2.2_error_injection_yolo26n.py `
  --weights runs/mio_tcd_yolo26n/v2_cpu_baseline_5p_10e_416/weights/best.pt `
  --data data/mio_tcd_yolo/mio_tcd_profile_500.yaml `
  --imgsz 416 `
  --batch 4 `
  --device cpu `
  --workers 2 `
  --sigma 0.05 `
  --out runs/week4_error_injection_test `
  --include_detect `

  --seed 42 `
  --layer_ids 16,22,13,6,19,10,8,9,2,4 `
'''