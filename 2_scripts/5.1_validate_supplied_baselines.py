from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

EXPECTED = {
    "GEN": {"map50_95": 0.647145, "map50": 0.830967, "classes": 11, "params": 2508090},
    "SNOW": {"map50_95": 0.191981, "map50": 0.326256, "classes": 8, "params": 2506920},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def profile_checkpoint(path: Path, imgsz: int) -> dict:
    from ultralytics.utils.torch_utils import get_flops

    yolo = YOLO(str(path), task="detect")
    model = yolo.model.float().cpu().eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    gflops = float(get_flops(model, imgsz=imgsz))
    result = {
        "classes": len(yolo.names),
        "parameters": parameters,
        "gflops": gflops,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_mib": path.stat().st_size / (1024 * 1024),
        "checkpoint_sha256": sha256(path),
    }
    del yolo, model
    return result


def validate(domain: str, checkpoint: Path, data: Path, args: argparse.Namespace) -> dict:
    print(f"\n===== VALIDATING {domain} BASELINE =====", flush=True)
    profile = profile_checkpoint(checkpoint, args.imgsz)
    expected = EXPECTED[domain]
    if profile["classes"] != expected["classes"]:
        raise RuntimeError(f"{domain} checkpoint class count {profile['classes']} != {expected['classes']}")
    if profile["parameters"] != expected["params"]:
        raise RuntimeError(f"{domain} parameter count {profile['parameters']} != {expected['params']}")

    yolo = YOLO(str(checkpoint), task="detect")
    results = yolo.val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        rect=True,
        conf=0.001,
        iou=0.70,
        max_det=300,
        half=False,
        dnn=False,
        augment=False,
        agnostic_nms=False,
        plots=True,
        save_json=False,
        verbose=True,
        project=str(args.project),
        name=f"baseline_{domain}_fp32",
        exist_ok=True,
    )

    all_ap = np.asarray(results.box.all_ap, dtype=float)
    map75 = float(all_ap[:, 5].mean()) if all_ap.ndim == 2 and all_ap.shape[1] >= 6 else float("nan")
    speed = getattr(results, "speed", {}) or {}
    latency_ms = sum(float(speed.get(key, 0.0)) for key in ("preprocess", "inference", "postprocess"))
    metrics = {
        "domain": domain,
        "dataset_yaml": str(data.resolve()),
        "dataset_yaml_sha256": sha256(data),
        **profile,
        "map50_95": float(results.box.map),
        "map50": float(results.box.map50),
        "map75": map75,
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "latency_ms_per_image": latency_ms,
        "fps_from_reported_latency": 0.0 if latency_ms == 0.0 else 1000.0 / latency_ms,
        "expected_map50_95": expected["map50_95"],
        "expected_map50": expected["map50"],
    }
    metrics["map50_95_abs_difference"] = abs(metrics["map50_95"] - expected["map50_95"])
    metrics["map50_abs_difference"] = abs(metrics["map50"] - expected["map50"])
    metrics["status"] = (
        "PASS"
        if metrics["map50_95_abs_difference"] <= args.tolerance
        and metrics["map50_abs_difference"] <= args.tolerance
        else "FAIL"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate supplied GEN and SNOW baselines with frozen FP32 settings.")
    parser.add_argument("--gen-model", required=True, type=Path)
    parser.add_argument("--snow-model", required=True, type=Path)
    parser.add_argument("--gen-data", required=True, type=Path)
    parser.add_argument("--snow-data", required=True, type=Path)
    parser.add_argument("--gen-manifest", required=True, type=Path)
    parser.add_argument("--snow-manifest", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()

    for path in (args.gen_model, args.snow_model, args.gen_data, args.snow_data, args.gen_manifest, args.snow_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.project.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    rows = [
        validate("GEN", args.gen_model, args.gen_data, args),
        validate("SNOW", args.snow_model, args.snow_data, args),
    ]

    csv_path = args.out_dir / "baseline_validation_fp32.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    evidence = {
        "command": sys.argv,
        "evaluation": {
            "split": "val",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "rect": True,
            "conf": 0.001,
            "iou": 0.70,
            "max_det": 300,
            "half": False,
            "augment": False,
            "agnostic_nms": False,
            "seed": 42,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "ultralytics": package_version("ultralytics"),
            "torchvision": package_version("torchvision"),
            "torch_pruning": package_version("torch-pruning"),
            "ultralytics_thop": package_version("ultralytics-thop"),
        },
        "inputs": {
            "gen_manifest": str(args.gen_manifest.resolve()),
            "gen_manifest_sha256": sha256(args.gen_manifest),
            "snow_manifest": str(args.snow_manifest.resolve()),
            "snow_manifest_sha256": sha256(args.snow_manifest),
        },
        "results": rows,
    }
    json_path = args.out_dir / "baseline_validation_evidence.json"
    json_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("\n===== BASELINE VALIDATION SUMMARY =====")
    for row in rows:
        print(
            f"{row['domain']}: status={row['status']}, "
            f"mAP50-95={row['map50_95']:.6f} (expected {row['expected_map50_95']:.6f}), "
            f"mAP50={row['map50']:.6f} (expected {row['expected_map50']:.6f})"
        )
    print(f"CSV:  {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")

    failed = [row["domain"] for row in rows if row["status"] != "PASS"]
    if failed:
        raise SystemExit(
            "Baseline gate failed for " + ", ".join(failed) + ". Do not start pruning until the split/config mismatch is resolved."
        )


if __name__ == "__main__":
    main()
