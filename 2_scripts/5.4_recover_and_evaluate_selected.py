from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Required so the custom class can be unpickled from Stage-2/3 .pt files.
from layer_replacement_adapter import DynamicNoParamLayerAdapter  # noqa: F401

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

try:
    import torchvision
except Exception:
    torchvision = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_params(yolo: YOLO) -> int:
    return sum(p.numel() for p in yolo.model.parameters())


def file_mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def model_signature(yolo: YOLO) -> list[dict[str, Any]]:
    """Return a stable top-level architecture signature for audit and reload checks."""
    signature: list[dict[str, Any]] = []
    for index, layer in enumerate(yolo.model.model):
        signature.append(
            {
                "index": index,
                "class": layer.__class__.__name__,
                "parameters": sum(p.numel() for p in layer.parameters()),
                "from": getattr(layer, "f", None),
                "layer_index": getattr(layer, "i", index),
            }
        )
    return signature


def signature_sha256(signature: list[dict[str, Any]]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_preserve_architecture_trainer(
    loaded_model: torch.nn.Module,
    expected_params: int,
    expected_signature_hash: str,
):
    """
    Create a DetectionTrainer that preserves the exact already-loaded custom architecture.

    when ``yolo.model`` already contained the required parameter-free adapters,
    trainer captures that loaded model directly and uses it whenever Ultralytics
    does not pass a module.
    """

    if not isinstance(loaded_model, torch.nn.Module):
        raise TypeError("loaded_model must be a PyTorch module")

    class PreserveLoadedArchitectureTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # Ultralytics passes weights=None when pretrained=False. In that case, use the exact
            # model captured from YOLO(raw_replaced_checkpoint.pt), rather than rebuilding cfg/YAML.
            candidate = weights if isinstance(weights, torch.nn.Module) else loaded_model

            actual_params = sum(p.numel() for p in candidate.parameters())
            if actual_params != expected_params:
                raise RuntimeError(
                    f"Loaded model parameter mismatch before training: "
                    f"{actual_params} != {expected_params}"
                )

            wrapper = type("ModelWrapper", (), {"model": candidate})
            actual_signature_hash = signature_sha256(model_signature(wrapper))
            if actual_signature_hash != expected_signature_hash:
                raise RuntimeError(
                    "Loaded model architecture signature changed before training: "
                    f"{actual_signature_hash} != {expected_signature_hash}"
                )

            print(
                "PreserveLoadedArchitectureTrainer: using the exact loaded replacement model "
                f"({actual_params} parameters)."
            )
            return candidate

    PreserveLoadedArchitectureTrainer.__name__ = "PreserveLoadedArchitectureTrainer"
    return PreserveLoadedArchitectureTrainer


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def one(rows: list[dict[str, str]], key: str, value: str, label: str) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {label} row with {key}={value}; found {len(matches)}")
    return matches[0]


def summarize(result: Any) -> dict[str, float]:
    box = result.box
    speed = getattr(result, "speed", {}) or {}
    latency = (
        as_float(speed.get("preprocess", 0.0))
        + as_float(speed.get("inference", 0.0))
        + as_float(speed.get("postprocess", 0.0))
    )
    return {
        "map50_95": as_float(box.map),
        "map50": as_float(box.map50),
        "map75": as_float(box.map75),
        "precision": as_float(getattr(box, "mp", float("nan"))),
        "recall": as_float(getattr(box, "mr", float("nan"))),
        "latency_ms": latency,
        "fps": 0.0 if latency <= 0 else 1000.0 / latency,
    }


def evaluate(model_path: Path, data: Path, device: str, workers: int, project: Path, name: str) -> tuple[dict[str, float], int]:
    yolo = YOLO(str(model_path), task="detect")
    params = count_params(yolo)
    result = yolo.val(
        data=str(data),
        split="val",
        imgsz=640,
        batch=16,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=True,
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
    )
    metrics = summarize(result)
    del yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, params


def train_kwargs(data: Path, project: Path, name: str, device: str, workers: int, smoke: bool) -> dict[str, Any]:
    return {
        "data": str(data),
        "epochs": 1 if smoke else 20,
        "patience": 0,
        "batch": 16,
        "imgsz": 640,
        "save": True,
        "save_period": -1 if smoke else 5,
        "cache": False,
        "device": device,
        "workers": 2 if smoke else workers,
        "project": str(project),
        "name": name,
        "exist_ok": False,
        "pretrained": False,
        "optimizer": "AdamW",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": False,
        "close_mosaic": 0,
        "resume": False,
        "amp": True,
        "fraction": 0.10 if smoke else 1.0,
        "profile": False,
        "freeze": None,
        "multi_scale": 0.0,
        "compile": False,
        "dropout": 0.0,
        "val": False if smoke else True,
        "split": "val",
        "save_json": False,
        "conf": 0.001,
        "iou": 0.70,
        "max_det": 300,
        "half": False,
        "dnn": False,
        "plots": False if smoke else True,
        "augment": False,
        "agnostic_nms": False,
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "nbs": 64,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "copy_paste_mode": "flip",
    }


def source_specs(stage2: list[dict[str, str]], stage3: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    c007 = one(stage3, "candidate_id", "C007", "Stage-3")
    l9 = one(stage2, "layer_id", "9", "Stage-2")
    if c007.get("status") != "ok":
        raise RuntimeError(f"C007 is not successful: {c007.get('error')}")
    if l9.get("status") != "ok":
        raise RuntimeError(f"Layer 9 is not successful: {l9.get('error')}")
    return {
        "C007": {"layers": "9,19", "types": "SPPF,C3k2", "row": c007},
        "L9": {"layers": "9", "types": "SPPF", "row": l9},
    }


def pref(domain: str) -> str:
    return "gen" if domain == "GEN" else "snow"


def source_path(spec: dict[str, Any], domain: str) -> Path:
    return Path(spec["row"][f"{pref(domain)}_model_path"])


def raw_metrics(spec: dict[str, Any], domain: str) -> dict[str, float]:
    p = pref(domain)
    row = spec["row"]
    return {
        "map50_95": as_float(row.get(f"{p}_replaced_map50_95")),
        "map50": as_float(row.get(f"{p}_replaced_map50")),
        "map75": as_float(row.get(f"{p}_replaced_map75")),
        "precision": as_float(row.get(f"{p}_precision")),
        "recall": as_float(row.get(f"{p}_recall")),
        "latency_ms": as_float(row.get(f"{p}_latency_ms")),
        "fps": as_float(row.get(f"{p}_fps")),
    }


def baseline_metrics(stage1: list[dict[str, str]], domain: str) -> tuple[dict[str, float], int]:
    row = one(stage1, "domain", domain, "Stage-1 baseline")
    return (
        {
            "map50_95": as_float(row.get("map50_95")),
            "map50": as_float(row.get("map50")),
            "map75": as_float(row.get("map75")),
            "precision": as_float(row.get("precision")),
            "recall": as_float(row.get("recall")),
            "latency_ms": as_float(row.get("latency_ms_per_image")),
            "fps": as_float(row.get("fps_from_reported_latency")),
        },
        int(float(row["parameters"])),
    )


def verify_source_models(specs: dict[str, dict[str, Any]], baseline_models: dict[str, Path], stage1: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in ("GEN", "SNOW"):
        base_metrics, expected_base_params = baseline_metrics(stage1, domain)
        base_path = baseline_models[domain]
        yolo = YOLO(str(base_path), task="detect")
        actual = count_params(yolo)
        del yolo
        if actual != expected_base_params:
            raise RuntimeError(f"{domain} baseline params mismatch: {actual} != {expected_base_params}")
        rows.append({
            "candidate": "BASELINE", "domain": domain, "layers": "", "model_path": str(base_path),
            "sha256": sha256(base_path), "model_mib": file_mib(base_path), "parameters": actual,
            "parameter_reduction_percent": 0.0,
        })
        for candidate, spec in specs.items():
            path = source_path(spec, domain)
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = spec["row"].get(f"{pref(domain)}_model_sha256", "")
            actual_hash = sha256(path)
            if expected_hash and expected_hash != actual_hash:
                raise RuntimeError(f"{candidate}/{domain} checksum mismatch")
            yolo = YOLO(str(path), task="detect")
            params = count_params(yolo)
            del yolo
            expected_after = spec["row"].get(f"{pref(domain)}_params_after", "")
            if expected_after and params != int(float(expected_after)):
                raise RuntimeError(f"{candidate}/{domain} parameter mismatch")
            rows.append({
                "candidate": candidate, "domain": domain, "layers": spec["layers"], "model_path": str(path),
                "sha256": actual_hash, "model_mib": file_mib(path), "parameters": params,
                "parameter_reduction_percent": (expected_base_params - params) / expected_base_params * 100.0,
            })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def comparison_row(candidate: str, spec: dict[str, Any], domain: str, phase: str, model_path: Path,
                   params: int, base_params: int, metrics: dict[str, float], base: dict[str, float],
                   raw: dict[str, float], run_dir: Path | None = None) -> dict[str, Any]:
    ad = base["map50_95"] - metrics["map50_95"]
    return {
        "candidate": candidate,
        "layers": spec["layers"],
        "layer_types": spec["types"],
        "domain": domain,
        "phase": phase,
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "model_mib": file_mib(model_path),
        "parameters": params,
        "baseline_parameters": base_params,
        "parameters_reduced": base_params - params,
        "parameter_reduction_percent": (base_params - params) / base_params * 100.0,
        **metrics,
        "signed_ad_vs_baseline_map50_95": ad,
        "negative_ad_retained": ad < 0,
        "accuracy_retained_percent": metrics["map50_95"] / base["map50_95"] * 100.0,
        "recovery_gain_over_raw_map50_95": metrics["map50_95"] - raw["map50_95"] if phase == "recovered_best" else "",
        "training_run_dir": str(run_dir.resolve()) if run_dir else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-csv", required=True, type=Path)
    parser.add_argument("--stage2-raw-csv", required=True, type=Path)
    parser.add_argument("--stage3-raw-csv", required=True, type=Path)
    parser.add_argument("--gen-baseline-model", required=True, type=Path)
    parser.add_argument("--snow-baseline-model", required=True, type=Path)
    parser.add_argument("--gen-data", required=True, type=Path)
    parser.add_argument("--snow-data", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--train-workers", default=4, type=int)
    parser.add_argument("--val-workers", default=8, type=int)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()

    for path in [args.stage1_csv, args.stage2_raw_csv, args.stage3_raw_csv,
                 args.gen_baseline_model, args.snow_baseline_model, args.gen_data, args.snow_data]:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    train_project = args.out_dir / "training_runs"
    val_project = args.out_dir / "final_validation_runs"
    train_project.mkdir()
    val_project.mkdir()

    stage1 = read_csv(args.stage1_csv)
    stage2 = read_csv(args.stage2_raw_csv)
    stage3 = read_csv(args.stage3_raw_csv)
    specs = source_specs(stage2, stage3)
    baseline_models = {"GEN": args.gen_baseline_model, "SNOW": args.snow_baseline_model}
    data_paths = {"GEN": args.gen_data, "SNOW": args.snow_data}

    inventory = verify_source_models(specs, baseline_models, stage1)
    write_csv(args.out_dir / "source_model_inventory.csv", inventory)
    inv = {(row["candidate"], row["domain"]): row for row in inventory}

    checks = []
    for label, path in {
        "stage1_csv": args.stage1_csv,
        "stage2_raw_csv": args.stage2_raw_csv,
        "stage3_raw_csv": args.stage3_raw_csv,
        "gen_baseline_model": args.gen_baseline_model,
        "snow_baseline_model": args.snow_baseline_model,
        "gen_data_yaml": args.gen_data,
        "snow_data_yaml": args.snow_data,
        "script": Path(__file__),
        "adapter": Path(__file__).with_name("layer_replacement_adapter.py"),
    }.items():
        checks.append({"label": label, "path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(args.out_dir / "input_checksums.csv", checks)

    settings = {
        "mode": args.mode,
        "selected_candidates": {"C007": "layers 9,19", "L9": "layer 9"},
        "negative_accuracy_drop_policy": "retain_signed",
        "t3_formula_used": False,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torchvision": getattr(torchvision, "__version__", "unavailable"),
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "full_recovery_args": train_kwargs(args.snow_data, train_project, "example", args.device, args.train_workers, False),
        "evaluation": {"imgsz": 640, "batch": 16, "rect": True, "conf": 0.001, "iou": 0.70,
                       "max_det": 300, "half": False, "augment": False, "agnostic_nms": False},
    }
    (args.out_dir / "experiment_environment_and_settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")

    tasks = [("C007", "SNOW"), ("L9", "SNOW")] if args.mode == "smoke" else [
        ("C007", "SNOW"), ("L9", "SNOW"), ("C007", "GEN"), ("L9", "GEN")
    ]

    progress: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []

    for index, (candidate, domain) in enumerate(tasks, 1):
        spec = specs[candidate]
        raw_path = source_path(spec, domain)
        data = data_paths[domain]
        run_name = f"{candidate}_{domain}_{'smoke1e' if args.mode == 'smoke' else 'recovery20e'}"
        run_dir = train_project / run_name
        row: dict[str, Any] = {
            "candidate": candidate, "layers": spec["layers"], "domain": domain,
            "mode": args.mode, "status": "running", "error": "", "run_dir": str(run_dir),
            "source_model": str(raw_path), "started_unix": time.time(),
        }
        progress.append(row)
        write_csv(args.out_dir / "stage4_progress.csv", progress)

        print("\n" + "=" * 80)
        print(f"TASK {index}/{len(tasks)}: {candidate} / {domain} / layers {spec['layers']}")
        print("=" * 80)

        try:
            yolo = YOLO(str(raw_path), task="detect")
            source_params = count_params(yolo)
            if source_params != int(inv[(candidate, domain)]["parameters"]):
                raise RuntimeError("Source parameter count changed before training")

            source_signature = model_signature(yolo)
            source_signature_hash = signature_sha256(source_signature)
            expected_adapter_layers = {int(value) for value in spec["layers"].split(",")}
            actual_adapter_layers = {
                item["index"]
                for item in source_signature
                if item["class"] == "DynamicNoParamLayerAdapter"
            }
            if actual_adapter_layers != expected_adapter_layers:
                raise RuntimeError(
                    f"Unexpected adapter layers before training: "
                    f"{sorted(actual_adapter_layers)} != {sorted(expected_adapter_layers)}"
                )

            architecture_audit_path = args.out_dir / f"architecture_{candidate}_{domain}_source.json"
            architecture_audit_path.write_text(
                json.dumps(
                    {
                        "candidate": candidate,
                        "domain": domain,
                        "source_model": str(raw_path.resolve()),
                        "parameters": source_params,
                        "signature_sha256": source_signature_hash,
                        "top_level_layers": source_signature,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            trainer_class = make_preserve_architecture_trainer(
                loaded_model=yolo.model,
                expected_params=source_params,
                expected_signature_hash=source_signature_hash,
            )
            yolo.train(
                trainer=trainer_class,
                **train_kwargs(
                    data,
                    train_project,
                    run_name,
                    args.device,
                    args.train_workers,
                    args.mode == "smoke",
                ),
            )
            del yolo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            weights = run_dir / "weights"
            if args.mode == "smoke":
                trained = weights / "last.pt"
                if not trained.is_file():
                    raise RuntimeError(f"Smoke last.pt missing: {trained}")
                check_model = YOLO(str(trained), task="detect")
                trained_params = count_params(check_model)
                trained_signature = model_signature(check_model)
                trained_signature_hash = signature_sha256(trained_signature)
                trained_adapter_layers = {
                    item["index"]
                    for item in trained_signature
                    if item["class"] == "DynamicNoParamLayerAdapter"
                }
                del check_model
                if trained_params != source_params:
                    raise RuntimeError(
                        f"Smoke training changed parameter count: {source_params} -> {trained_params}"
                    )
                if trained_signature_hash != source_signature_hash:
                    raise RuntimeError(
                        "Smoke training changed the top-level architecture signature: "
                        f"{source_signature_hash} -> {trained_signature_hash}"
                    )
                if trained_adapter_layers != expected_adapter_layers:
                    raise RuntimeError(
                        f"Smoke checkpoint adapter layers changed: "
                        f"{sorted(trained_adapter_layers)} != {sorted(expected_adapter_layers)}"
                    )
                row.update({
                    "status": "ok", "trained_model": str(trained.resolve()),
                    "trained_model_sha256": sha256(trained), "parameters": trained_params,
                    "source_architecture_sha256": source_signature_hash,
                    "trained_architecture_sha256": trained_signature_hash,
                    "adapter_layers": ",".join(str(v) for v in sorted(trained_adapter_layers)),
                    "elapsed_seconds": time.time() - row["started_unix"],
                })
            else:
                best = weights / "best.pt"
                last = weights / "last.pt"
                if not best.is_file() or not last.is_file():
                    raise RuntimeError(f"Missing best.pt or last.pt in {weights}")
                recovered, recovered_params = evaluate(best, data, args.device, args.val_workers, val_project,
                                                       f"{candidate}_{domain}_recovered_best_fp32")
                recovered_model = YOLO(str(best), task="detect")
                recovered_signature = model_signature(recovered_model)
                recovered_signature_hash = signature_sha256(recovered_signature)
                recovered_adapter_layers = {
                    item["index"]
                    for item in recovered_signature
                    if item["class"] == "DynamicNoParamLayerAdapter"
                }
                del recovered_model
                if recovered_params != source_params:
                    raise RuntimeError(f"Recovery changed parameters: {source_params} -> {recovered_params}")
                if recovered_signature_hash != source_signature_hash:
                    raise RuntimeError(
                        "Recovery changed the top-level architecture signature: "
                        f"{source_signature_hash} -> {recovered_signature_hash}"
                    )
                if recovered_adapter_layers != expected_adapter_layers:
                    raise RuntimeError(
                        f"Recovered checkpoint adapter layers changed: "
                        f"{sorted(recovered_adapter_layers)} != {sorted(expected_adapter_layers)}"
                    )

                base, base_params = baseline_metrics(stage1, domain)
                raw = raw_metrics(spec, domain)
                comparison.extend([
                    comparison_row(candidate, spec, domain, "baseline", baseline_models[domain], base_params,
                                   base_params, base, base, raw),
                    comparison_row(candidate, spec, domain, "raw_replaced", raw_path, source_params,
                                   base_params, raw, base, raw),
                    comparison_row(candidate, spec, domain, "recovered_best", best, recovered_params,
                                   base_params, recovered, base, raw, run_dir),
                ])
                write_csv(args.out_dir / "baseline_raw_recovered_comparison.csv", comparison)
                row.update({
                    "status": "ok", "best_model": str(best.resolve()), "best_model_sha256": sha256(best),
                    "last_model": str(last.resolve()), "last_model_sha256": sha256(last),
                    "parameters": recovered_params,
                    "source_architecture_sha256": source_signature_hash,
                    "recovered_architecture_sha256": recovered_signature_hash,
                    "adapter_layers": ",".join(str(v) for v in sorted(recovered_adapter_layers)),
                    "recovered_map50_95": recovered["map50_95"],
                    "recovered_map50": recovered["map50"],
                    "recovery_gain_map50_95": recovered["map50_95"] - raw["map50_95"],
                    "signed_ad_vs_baseline_map50_95": base["map50_95"] - recovered["map50_95"],
                    "elapsed_seconds": time.time() - row["started_unix"],
                })
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.time() - row["started_unix"]})
            write_csv(args.out_dir / "stage4_progress.csv", progress)
            raise
        finally:
            write_csv(args.out_dir / "stage4_progress.csv", progress)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "mode": args.mode,
        "tasks_ok": sum(row["status"] == "ok" for row in progress),
        "tasks_failed": sum(row["status"] == "failed" for row in progress),
        "selected_candidates": {"C007": "layers 9,19", "L9": "layer 9"},
        "negative_accuracy_drop_policy": "retain_signed",
        "t3_formula_used": False,
        "output_folder": str(args.out_dir.resolve()),
    }
    (args.out_dir / "stage4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== STAGE 4 COMPLETE =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
