"""Measure Isomorphic Pruning + Taylor structural GFLOPs at 640x640.

The calculation uses a CPU forward-hook counter over Conv2d and Linear layers.
Each multiply-add counts as two FLOPs.  It measures architecture-level work,
not measured GPU or PYNQ throughput.  The same estimator is used for the
unpruned BDD checkpoints and the completed Isomorphic-Taylor checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT.parent
COMPETITOR_ROOT = STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct" / "bdd_competitors"
RUNS = {
    "GEN2": COMPETITOR_ROOT / "T7_bdd_gen2_isomorphic_taylor_taylor50b_70ep_v2",
    "NGN2": COMPETITOR_ROOT / "T7_bdd_ngn2_isomorphic_taylor_taylor50b_70ep_v2",
}
OUTPUT_DIR = COMPETITOR_ROOT / "isomorphic_taylor_structural_metrics_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def hook_gflops(model: Any, imgsz: int = 640) -> float:
    """Count Conv2d/Linear FLOPs from a real CPU forward pass."""
    import torch

    operations = 0

    def first_tensor(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)) and value:
            return first_tensor(value[0])
        return None

    def count(module: Any, _inputs: Any, output: Any) -> None:
        nonlocal operations
        tensor = first_tensor(output)
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
            model(torch.zeros(1, 3, imgsz, imgsz))
    finally:
        for handle in handles:
            handle.remove()
    if operations <= 0:
        raise RuntimeError("The structural profiler counted no Conv2d/Linear operations")
    return operations / 1e9


def load_module_checkpoint(path: Path) -> Any:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("model", checkpoint)
    if not isinstance(checkpoint, torch.nn.Module):
        raise TypeError(f"{path} does not contain a torch module")
    return checkpoint.float().cpu().eval()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    from ultralytics import YOLO

    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for domain, run_root in RUNS.items():
        manifest_path = run_root / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest for {domain}: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"{domain} Isomorphic-Taylor run is not PASS")

        baseline_path = PROJECT_ROOT / manifest["baseline_checkpoint"]
        final = manifest["final_results"]["best"]
        final_path = PROJECT_ROOT / final["model"]
        if not baseline_path.is_file() or not final_path.is_file():
            raise FileNotFoundError(f"Missing {domain} baseline or final checkpoint")

        baseline = YOLO(str(baseline_path), task="detect").model.float().cpu().eval()
        pruned = load_module_checkpoint(final_path)
        baseline_parameters = count_parameters(baseline)
        pruned_parameters = count_parameters(pruned)
        if pruned_parameters >= baseline_parameters:
            raise RuntimeError(f"{domain} final model did not reduce parameters")
        if pruned_parameters != int(final["parameters"]):
            raise RuntimeError(f"{domain} final checkpoint parameter count disagrees with its manifest")

        baseline_gflops = hook_gflops(baseline)
        pruned_gflops = hook_gflops(pruned)
        rows.append({
            "domain": domain,
            "imgsz": 640,
            "estimator": "cpu_forward_hook_conv2d_linear_multiply_add_x2",
            "baseline_parameters": baseline_parameters,
            "pruned_parameters": pruned_parameters,
            "parameter_reduction_percent": 100.0 * (1.0 - pruned_parameters / baseline_parameters),
            "baseline_gflops": baseline_gflops,
            "pruned_gflops": pruned_gflops,
            "gflops_removed": baseline_gflops - pruned_gflops,
            "gflops_reduction_percent": 100.0 * (1.0 - pruned_gflops / baseline_gflops),
            "baseline_model": str(baseline_path.relative_to(PROJECT_ROOT)),
            "baseline_sha256": sha256(baseline_path),
            "pruned_model": str(final_path.relative_to(PROJECT_ROOT)),
            "pruned_sha256": sha256(final_path),
        })
        provenance[domain] = {
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256(manifest_path),
            "schema": final.get("schema"),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "ISOMORPHIC_TAYLOR_STRUCTURAL_GFLOPS_CPU.csv", rows)
    (OUTPUT_DIR / "ISOMORPHIC_TAYLOR_STRUCTURAL_GFLOPS_CPU.json").write_text(
        json.dumps({
            "schema": "bdd_isomorphic_taylor_structural_gflops_cpu_v1",
            "measurement": "640x640 architecture-only estimate; CPU Conv2d/Linear forward-hook multiply-add counter, with one multiply-add counted as two FLOPs. This is not GPU or PYNQ throughput.",
            "rows": rows,
            "provenance": provenance,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
