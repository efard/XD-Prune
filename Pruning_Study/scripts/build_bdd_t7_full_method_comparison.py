"""Build a comparable BDD GEN2/NGN2 T7 method table.

The table uses each method's final ``best`` evaluation record and measures
architecture-only GFLOPs from the corresponding raw checkpoint with the same
CPU Conv2d/Linear forward-hook estimator.  It is not a throughput benchmark.
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
RECOVERY_ROOT = STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct"
BASELINE_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_t1_t2_v1" / "baselines"
FREEZE_PATH = STUDY_ROOT / "configs" / "pruning" / "BDD_T1_T2_EXPERIMENT_FREEZE_V1.json"
OUTPUT_DIR = RECOVERY_ROOT / "tables"

METHODS = {
    "Proposed method": {
        "recovery": RECOVERY_ROOT / "T7_bdd_{domain_lower}_70ep",
        "raw": STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_raw56_updated_v1" / "T5" / "models" / "final" / "{domain}_raw_ranked_56pct.pth",
        "method_id": "proposed_bdd_formula",
    },
    "R4 signed (non-clipped)": {
        "recovery": RECOVERY_ROOT / "T7_bdd_{domain_lower}_r4_signed_70ep",
        "raw": STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_r4_signed_37_5pct_raw56_v2" / "T5" / "models" / "final" / "{domain}_raw_ranked_56pct.pth",
        "method_id": "r4_signed_nonclipped",
    },
    "Global L1": {
        "recovery": RECOVERY_ROOT / "bdd_competitors" / "T7_bdd_{domain_lower}_global_l1_70ep",
        "raw": STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_competitors_raw56_v1" / "global_l1_37_5pct_raw56" / "T5" / "models" / "final" / "{domain}_global_l1_raw_ranked_56pct.pth",
        "method_id": "global_l1",
    },
    "FPGM": {
        "recovery": RECOVERY_ROOT / "bdd_competitors" / "T7_bdd_{domain_lower}_fpgm_70ep",
        "raw": STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_competitors_raw56_v1" / "fpgm_37_5pct_raw56" / "T5" / "models" / "final" / "{domain}_fpgm_raw_ranked_56pct.pth",
        "method_id": "fpgm",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def hook_gflops(model: Any) -> float:
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
            operations += 2 * tensor.numel() * (module.in_channels // module.groups) * int(module.kernel_size[0]) * int(module.kernel_size[1])
        elif isinstance(module, torch.nn.Linear):
            operations += 2 * tensor.numel() * module.in_features

    handles = [module.register_forward_hook(count) for module in model.modules() if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, 640, 640))
    finally:
        for handle in handles:
            handle.remove()
    if operations <= 0:
        raise RuntimeError("CPU structural profiler counted no operations")
    return operations / 1e9


def load_model(path: Path) -> Any:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value.get("model", value)
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"Checkpoint does not contain a torch module: {path}")
    return value.float().cpu().eval()


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    import torch
    from ultralytics import YOLO

    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for domain in ("GEN2", "NGN2"):
        baseline_info = json.loads((BASELINE_ROOT / f"{domain}.json").read_text(encoding="utf-8"))
        baseline_path = PROJECT_ROOT / freeze["domains"][domain]["checkpoint"]
        baseline = YOLO(str(baseline_path), task="detect").model.float().cpu().eval()
        baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
        baseline_gflops = hook_gflops(baseline)
        baseline_map = float(baseline_info["metrics"]["map50_95"])
        rows.append({
            "domain": domain, "method": "Unpruned baseline", "method_id": "baseline",
            "checkpoint": "baseline", "status": "PASS", "parameters": baseline_parameters,
            "parameter_reduction_percent": 0.0, "map50_95": baseline_map,
            "retention_vs_baseline_percent": 100.0, "delta_map50_95": 0.0,
            "map50": baseline_info["metrics"]["ap50"], "map75": baseline_info["metrics"]["ap75"],
            "precision": baseline_info["metrics"]["precision"], "recall": baseline_info["metrics"]["recall"],
            "baseline_map50_95": baseline_map, "baseline_parameters": baseline_parameters,
            "baseline_gflops": baseline_gflops, "raw_gflops": baseline_gflops,
            "gflops_removed": 0.0, "gflops_reduction_percent": 0.0,
            "raw_model": "", "raw_sha256": "", "evaluation_record": str((BASELINE_ROOT / f"{domain}.json").relative_to(PROJECT_ROOT)),
        })
        for method, spec in METHODS.items():
            recovery = Path(str(spec["recovery"]).format(domain_lower=domain.lower()))
            evaluation_path = recovery / "records" / f"{domain}_final_best_evaluation.json"
            raw_path = Path(str(spec["raw"]).format(domain=domain))
            if not evaluation_path.is_file() or not raw_path.is_file():
                raise FileNotFoundError(f"Missing {method} {domain} evaluation or raw model")
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation.get("status") != "PASS":
                raise RuntimeError(f"{method} {domain} evaluation is not PASS")
            metrics = evaluation["metrics"]
            raw = load_model(raw_path)
            raw_parameters = sum(parameter.numel() for parameter in raw.parameters())
            raw_gflops = hook_gflops(raw)
            rows.append({
                "domain": domain, "method": method, "method_id": spec["method_id"],
                "checkpoint": "best", "status": evaluation["status"],
                "parameters": evaluation["parameters"], "parameter_reduction_percent": evaluation["parameter_reduction_percent"],
                "map50_95": metrics["map50_95"],
                "retention_vs_baseline_percent": 100.0 * metrics["map50_95"] / baseline_map,
                "delta_map50_95": metrics["map50_95"] - baseline_map,
                "map50": metrics["ap50"], "map75": metrics["ap75"],
                "precision": metrics["precision"], "recall": metrics["recall"],
                "baseline_map50_95": baseline_map, "baseline_parameters": baseline_parameters,
                "baseline_gflops": baseline_gflops, "raw_gflops": raw_gflops,
                "gflops_removed": baseline_gflops - raw_gflops,
                "gflops_reduction_percent": 100.0 * (1.0 - raw_gflops / baseline_gflops),
                "raw_model": str(raw_path.relative_to(PROJECT_ROOT)), "raw_sha256": sha256(raw_path),
                "evaluation_record": str(evaluation_path.relative_to(PROJECT_ROOT)),
            })

    fields = list(rows[0])
    csv_path = OUTPUT_DIR / "BDD_T7_FULL_METHOD_COMPARISON.csv"
    atomic_csv(csv_path, rows)
    md_path = OUTPUT_DIR / "BDD_T7_FULL_METHOD_COMPARISON.md"
    lines = [
        "# BDD GEN2/NGN2 T7 method comparison",
        "",
        "Best-checkpoint results at the completed T7 recovery. GFLOPs are 640x640 architecture-only estimates from the same CPU Conv2d/Linear forward-hook estimator; they are not measured throughput.",
        "",
        "| Domain | Method | mAP50-95 | Retention | Δ mAP50-95 | mAP50 | Precision | Recall | Parameters | Param. reduction | GFLOPs | GFLOPs removed | GFLOP reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {domain} | {method} | {map50_95:.4f} | {retention_vs_baseline_percent:.2f}% | {delta_map50_95:+.4f} | {map50:.4f} | {precision:.4f} | {recall:.4f} | {parameters:,} | {parameter_reduction_percent:.2f}% | {raw_gflops:.4f} | {gflops_removed:.4f} | {gflops_reduction_percent:.2f}% |".format(**row))
    lines.extend([
        "", "Notes:", "",
        "- The R4 row is the trained signed/non-clipped checkpoint. The clipped and signed 37.5% T3/T4 sequences were previously byte-identical, so clipped R4 was not trained as an independent T7 checkpoint.",
        "- Global-L1 and FPGM are DepGraph-constrained structural competitors. All rows use the same BDD validation protocol and 70-epoch 6-6-8-10-40 recovery schedule.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
