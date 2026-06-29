import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO


def inject_activation_error(value, sigma):
    # A YOLO layer output may be a Tensor, list, tuple, or dict.
    # This keeps the original output structure while only perturbing floating-point tensors.
    if isinstance(value, torch.Tensor):
        if not torch.is_floating_point(value):
            return value

        # Operation:
        ## y_injected = y + sigma * std(y) * N(0, 1)
        # sigma controls the injected activation noise strength relative to the layer activation scale.
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
    # Ultralytics detection validation metrics are stored under metrics.box.
    # metrics.box.map is mAP50-95, and metrics.box.map50 is mAP50.
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
    }


def run_validation(yolo, args):
    ## mAP
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
    # If --layer_ids is empty, the experiment tests every top-level YOLO layer.
    # Example for selected layers: --layer_ids 0,1,2,3
    if layer_ids_text == "":
        return list(range(total_layers))

    layer_ids = [int(item.strip()) for item in layer_ids_text.split(",") if item.strip() != ""]

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
    parser.add_argument("--sigma", type=float, default=0.2, help="Error strength")
    parser.add_argument("--seeds", default="42,123,999,2026", help="Comma-separated random seeds for repeated runs")
    parser.add_argument("--layer_ids", default="", help="Comma-separated layer ids. Empty means all layers.")
    parser.add_argument("--include_detect", action="store_true", help="Also inject error into Detect layer")
    parser.add_argument("--out", default="runs/week4_error_injection_multiseed")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip() != ""]

    if len(seeds) == 0:
        raise ValueError("--seeds must contain at least one integer seed.")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_yolo = YOLO(args.weights)

    try:
        baseline_layers = list(baseline_yolo.model.model)
    except TypeError as exc:
        raise TypeError("Could not read top-level YOLO layers from model.model.") from exc

    selected_layer_ids = parse_layer_ids(args.layer_ids, len(baseline_layers))

    # Run the clean baseline once.
    # The clean baseline does not use the injected random noise, so repeating it for each seed wastes time.
    print("Running clean baseline validation once...")
    torch.manual_seed(seeds[0])

    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seeds[0])

    ## clean baseline
    baseline_metrics = run_validation(baseline_yolo, args)
    baseline_map50_95 = baseline_metrics["map50_95"]
    baseline_map50 = baseline_metrics["map50"]

    all_rows = []

    for seed in seeds:
        print(f"\nRunning error injection with seed {seed}...")

        for layer_id in selected_layer_ids:
            reference_layer = baseline_layers[layer_id]
            layer_type = reference_layer.__class__.__name__

            # Detect is skipped by default because injecting error directly into final detection outputs
            # can dominate the result and is less useful for feature-layer optimization decisions.
            if not args.include_detect and "Detect" in layer_type:
                continue

            print(f"Injecting error into layer {layer_id}: {layer_type}")

            # Reload the checkpoint for each layer.
            # This keeps each layer experiment independent and avoids state contamination from hooks.
            yolo = YOLO(args.weights)
            layers = list(yolo.model.model)
            target_layer = layers[layer_id]

            # Reset the random seed before each layer.
            # This makes the result reproducible for the same seed and layer.
            torch.manual_seed(seed)

            if args.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

            ## target_layer forward -> add error to output -> pass to next layers
            def error_hook(module, inputs, output):
                # Returning a modified output means all following YOLO layers receive the corrupted activation.
                return inject_activation_error(output, args.sigma)

            handle = target_layer.register_forward_hook(error_hook)

            try:
                injected_metrics = run_validation(yolo, args)
            finally:
                handle.remove()

            injected_map50_95 = injected_metrics["map50_95"]
            injected_map50 = injected_metrics["map50"]

            ## drops
            drop_map50_95 = baseline_map50_95 - injected_map50_95
            drop_map50 = baseline_map50 - injected_map50

            all_rows.append({
                "seed": seed,
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

    if len(all_rows) == 0:
        raise RuntimeError(
            "No error injection rows were created. "
            "If you selected only Detect layer, add --include_detect."
        )

    all_runs_csv = output_dir / "error_sensitivity_all_runs.csv"
    all_fieldnames = [
        "seed",
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

    with all_runs_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_fieldnames)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (row["seed"], row["layer_id"])))

    grouped_rows = defaultdict(list)

    for row in all_rows:
        grouped_rows[(row["layer_id"], row["layer_type"], row["sigma"])].append(row)

    averaged_rows = []

    for (layer_id, layer_type, sigma), rows in grouped_rows.items():
        drops_95 = [row["drop_map50_95"] for row in rows]
        drops_50 = [row["drop_map50"] for row in rows]
        injected_95 = [row["injected_map50_95"] for row in rows]
        injected_50 = [row["injected_map50"] for row in rows]

        mean_drop_95 = sum(drops_95) / len(drops_95)
        mean_drop_50 = sum(drops_50) / len(drops_50)
        mean_injected_95 = sum(injected_95) / len(injected_95)
        mean_injected_50 = sum(injected_50) / len(injected_50)

        # Population standard deviation is used because the listed seeds are the complete set
        # evaluated for this experiment.
        std_drop_95 = (sum((value - mean_drop_95) ** 2 for value in drops_95) / len(drops_95)) ** 0.5
        std_drop_50 = (sum((value - mean_drop_50) ** 2 for value in drops_50) / len(drops_50)) ** 0.5

        averaged_rows.append({
            "layer_id": layer_id,
            "layer_type": layer_type,
            "sigma": sigma,
            "num_runs": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])),
            "baseline_map50_95": baseline_map50_95,
            "mean_injected_map50_95": mean_injected_95,
            "mean_drop_map50_95": mean_drop_95,
            "std_drop_map50_95": std_drop_95,
            "baseline_map50": baseline_map50,
            "mean_injected_map50": mean_injected_50,
            "mean_drop_map50": mean_drop_50,
            "std_drop_map50": std_drop_50,
        })

    averaged_rows_sorted = sorted(
        averaged_rows,
        key=lambda row: row["mean_drop_map50_95"],
        reverse=True,
    )

    average_csv = output_dir / "error_sensitivity_average.csv"
    average_fieldnames = [
        "layer_id",
        "layer_type",
        "sigma",
        "num_runs",
        "seeds",
        "baseline_map50_95",
        "mean_injected_map50_95",
        "mean_drop_map50_95",
        "std_drop_map50_95",
        "baseline_map50",
        "mean_injected_map50",
        "mean_drop_map50",
        "std_drop_map50",
    ]

    with average_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=average_fieldnames)
        writer.writeheader()
        writer.writerows(averaged_rows_sorted)

    print(f"\nAll-run sensitivity results saved to: {all_runs_csv}")
    print(f"Averaged sensitivity ranking saved to: {average_csv}")


if __name__ == "__main__":
    main()

'''
python scripts/2.3_error_injection_yolo26n_multiseed.py `
  --weights runs/mio_tcd_yolo26n/v2_cpu_baseline_5p_10e_416/weights/best.pt `
  --data data/mio_tcd_yolo/mio_tcd_profile_500.yaml `
  --imgsz 416 `
  --batch 4 `
  --device cpu `
  --workers 2 `
  --sigma 0.2 `
  --seeds 42,123,999,2026 `
  --include_detect `
  --out runs/week4_error_injection_sigma020_500_4seeds
'''