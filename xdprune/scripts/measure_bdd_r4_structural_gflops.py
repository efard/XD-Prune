"""Measure Formula R4 raw-T5 structural GFLOPs without using a GPU.

The R4 raw architecture is already frozen before recovery training.  This
script loads the untouched BDD baselines and frozen raw models on CPU, computes
the Ultralytics 640x640 structural GFLOP estimate, and records both domains.
It intentionally does not access recovery checkpoints or CUDA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
DEFAULT_RAW_ROOT = (
    STUDY_ROOT / "results" / "pruning"
    / "bdd_gen2_ngn2_r4_signed_37_5pct_raw56_v2"
)
FREEZE_PATH = STUDY_ROOT / "configs" / "pruning" / "BDD_T1_T2_EXPERIMENT_FREEZE_V1.json"
EXPECTED_FORMULA_ID = "BDD_R4_SIGNED_WEIGHTED_PARAM_ONLY_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def hook_gflops(model: Any) -> float:
    """Count Conv2d/Linear multiply-add FLOPs from an actual CPU forward pass.

    This fallback is used only when the supplied Ultralytics profiler returns
    zero.  Each multiply-add is counted as two FLOPs.  It measures the same
    frozen architecture before and after pruning; it is not GPU throughput.
    """

    import torch

    operations = 0

    def tensor_output(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)) and value:
            return tensor_output(value[0])
        return None

    def count(module: Any, _inputs: Any, output: Any) -> None:
        nonlocal operations
        tensor = tensor_output(output)
        if tensor is None:
            return
        if isinstance(module, torch.nn.Conv2d):
            kernel = int(module.kernel_size[0]) * int(module.kernel_size[1])
            operations += 2 * tensor.numel() * (module.in_channels // module.groups) * kernel
        elif isinstance(module, torch.nn.Linear):
            operations += 2 * tensor.numel() * module.in_features

    handles = [
        module.register_forward_hook(count)
        for module in model.modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, 640, 640))
    finally:
        for handle in handles:
            handle.remove()
    if operations <= 0:
        raise RuntimeError("Independent hook profiler counted no Conv2d/Linear operations")
    return operations / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    raw_root = args.raw_root.resolve()
    manifest_path = raw_root / "experiment_manifest.json"
    plan_path = raw_root / "frozen_pruning_plan.json"
    if not manifest_path.is_file() or not plan_path.is_file() or not FREEZE_PATH.is_file():
        raise FileNotFoundError("R4 raw manifest/plan or BDD freeze configuration is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not manifest.get("target_reached"):
        raise RuntimeError("Formula R4 raw plan is not complete")
    if manifest.get("formula_id") != EXPECTED_FORMULA_ID:
        raise RuntimeError(f"Unexpected formula ID: {manifest.get('formula_id')!r}")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for domain in ("GEN2", "NGN2"):
        baseline_path = PROJECT_ROOT / freeze["domains"][domain]["checkpoint"]
        raw_path = PROJECT_ROOT / manifest["final_models"][domain]["model"]
        if not baseline_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError(f"Missing {domain} baseline or Formula R4 raw model")
        baseline = YOLO(str(baseline_path), task="detect").model.float().cpu().eval()
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            raw = raw.get("model", raw)
        if not isinstance(raw, torch.nn.Module):
            raise TypeError(f"{domain} raw checkpoint does not contain a torch module")
        raw = raw.float().cpu().eval()

        ultralytics_baseline_gflops = float(get_flops(baseline, imgsz=640))
        ultralytics_raw_gflops = float(get_flops(raw, imgsz=640))
        if ultralytics_baseline_gflops > 0.0 and ultralytics_raw_gflops > 0.0:
            estimator = "ultralytics_get_flops"
            baseline_gflops = ultralytics_baseline_gflops
            raw_gflops = ultralytics_raw_gflops
        else:
            estimator = "cpu_forward_hook_conv2d_linear_multiply_add_x2"
            baseline_gflops = hook_gflops(baseline)
            raw_gflops = hook_gflops(raw)
        baseline_parameters = count_parameters(baseline)
        raw_parameters = count_parameters(raw)
        if raw_parameters >= baseline_parameters:
            raise RuntimeError(f"{domain} raw model did not reduce parameters")
        rows.append({
            "domain": domain,
            "imgsz": 640,
            "estimator": estimator,
            "baseline_parameters": baseline_parameters,
            "raw_parameters": raw_parameters,
            "parameter_reduction_percent": 100.0 * (1.0 - raw_parameters / baseline_parameters),
            "baseline_gflops": baseline_gflops,
            "raw_gflops": raw_gflops,
            "ultralytics_baseline_gflops": ultralytics_baseline_gflops,
            "ultralytics_raw_gflops": ultralytics_raw_gflops,
            "gflops_removed": baseline_gflops - raw_gflops,
            "gflops_reduction_percent": 100.0 * (1.0 - raw_gflops / baseline_gflops),
            "baseline_model": str(baseline_path.relative_to(PROJECT_ROOT)),
            "baseline_sha256": sha256(baseline_path),
            "raw_model": str(raw_path.relative_to(PROJECT_ROOT)),
            "raw_sha256": sha256(raw_path),
        })

    output = raw_root / "metrics"
    atomic_csv(output / "R4_RAW56_STRUCTURAL_GFLOPS_CPU.csv", rows)
    atomic_json(output / "R4_RAW56_STRUCTURAL_GFLOPS_CPU.json", {
        "schema": "bdd_r4_raw56_cpu_structural_gflops_v1",
        "formula_id": EXPECTED_FORMULA_ID,
        "measurement": "640x640 architecture-only estimate, not measured GPU throughput. Uses Ultralytics get_flops when nonzero; otherwise an explicit CPU forward-hook Conv2d/Linear multiply-add counter.",
        "raw_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "raw_manifest_sha256": sha256(manifest_path),
        "frozen_plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "frozen_plan_sha256": sha256(plan_path),
        "rows": rows,
    })
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
