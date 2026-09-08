"""Build BDD GEN-2/NGN-2 T3/T4 tables using the revised dual-domain formula.

This is an evidence-only analysis stage.  It reads the completed isolated
T1/T2 paired tables, verifies their frozen baseline references, and writes new
versioned T3 (damage/sensitivity) and T4 (resource/prunability) tables.  It
does not modify checkpoints, source evidence, or historical MIO/SNOW outputs.

Formula ID: BDD_UPDATED_GEN_PROTECTED_V1

  NAD_d,i = (mAP_d,baseline - mAP_d,pruned,i) / mAP_d,baseline
  D_d,i   = max(0, NAD_d,i)
  S_i     = max(D_GEN2,i, (D_GEN2,i + D_NGN2,i) / 2)
  RP_i    = removed_parameters_i / baseline_parameters
  RF_i    = removed_GFLOPs_i / baseline_GFLOPs
  B_i     = sqrt(RP_i * RF_i)
  P_i     = B_i / (B_i + S_i)

Negative isolated AD/NAD values are retained in T3 for transparency but are
not rewarded: they contribute zero to D.  This implements the paired-bootstrap
conclusion that apparent isolated gains were not statistically supported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_t1_t2_v1"
OUTPUT_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_t3_t4_updated_v1"
FORMULA_ID = "BDD_UPDATED_GEN_PROTECTED_V1"
RATIOS = (
    ("12_5pct", "12_5", 12.5),
    ("25pct", "25", 25.0),
    ("37_5pct", "37_5", 37.5),
)

T3_FIELDS = [
    "rank_by_sensitivity_ascending",
    "ratio_percent",
    "group_id",
    "group_kind",
    "representative_root",
    "mAP_GEN2_baseline",
    "mAP_GEN2_pruned",
    "AD_GEN2",
    "NAD_GEN2_signed",
    "D_GEN2_nonnegative_harm",
    "mAP_NGN2_baseline",
    "mAP_NGN2_pruned",
    "AD_NGN2",
    "NAD_NGN2_signed",
    "D_NGN2_nonnegative_harm",
    "sensitivity_S_GEN_PROTECTED",
]
T4_FIELDS = [
    "rank_by_prunability_descending",
    "ratio_percent",
    "group_id",
    "group_kind",
    "representative_root",
    "parameters_removed",
    "gflops_removed",
    "baseline_parameters",
    "baseline_gflops",
    "RP_parameter_reduction_fraction",
    "RF_gflop_reduction_fraction",
    "B_geometric_resource_benefit",
    "sensitivity_S_GEN_PROTECTED",
    "P_updated_prunability",
]
SEQUENCE_FIELDS = [
    "sequence_rank",
    "T4_rank",
    "ratio_percent",
    "group_id",
    "group_kind",
    "representative_root",
    "P_updated_prunability",
    "B_geometric_resource_benefit",
    "sensitivity_S_GEN_PROTECTED",
    "parameters_removed_isolated",
    "gflops_removed_isolated",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite")
    return number


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def relative(path: Path) -> str:
    return path.resolve().relative_to(STUDY_ROOT.parent.resolve()).as_posix()


def baseline(domain: str) -> tuple[Path, dict[str, Any], float, float, float]:
    path = SOURCE_ROOT / "baselines" / f"{domain}.json"
    record = read_json(path)
    if record.get("status") != "PASS" or record.get("domain") != domain:
        raise RuntimeError(f"Invalid baseline record: {path}")
    metric = finite(record["metrics"]["map50_95"], f"{domain} baseline mAP")
    parameters = finite(record["structure"]["parameters"], f"{domain} baseline parameters")
    gflops = finite(record["structure"]["gflops"], f"{domain} baseline GFLOPs")
    if min(metric, parameters, gflops) <= 0:
        raise RuntimeError(f"{domain} baseline has non-positive metric/resource value")
    return path, record, metric, parameters, gflops


def expected_group_ids() -> set[str]:
    return {f"DG{index:03d}" for index in range(1, 43)} | {f"CDG{index:03d}" for index in range(1, 10)}


def build_ratio(
    source_path: Path,
    ratio_percent: float,
    gen_map: float,
    ngn_map: float,
    parameters: float,
    gflops: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(source_path)
    groups = [row["group_id"] for row in source]
    if len(source) != 51 or len(set(groups)) != 51 or set(groups) != expected_group_ids():
        raise RuntimeError(f"{source_path} must contain the 51 canonical paired groups exactly once")

    t3: list[dict[str, Any]] = []
    t4: list[dict[str, Any]] = []
    for row in source:
        group_id = row["group_id"]
        observed_ratio = finite(row["ratio_percent"], f"{group_id} ratio")
        if not close(observed_ratio, ratio_percent):
            raise RuntimeError(f"{group_id} ratio mismatch: {observed_ratio} != {ratio_percent}")
        gen_pruned = finite(row["gen2_pruned_map50_95"], f"{group_id} GEN2 mAP")
        ngn_pruned = finite(row["ngn2_pruned_map50_95"], f"{group_id} NGN2 mAP")
        gen_ad = gen_map - gen_pruned
        ngn_ad = ngn_map - ngn_pruned
        gen_nad = gen_ad / gen_map
        ngn_nad = ngn_ad / ngn_map
        if not close(gen_ad, finite(row["gen2_AD_map50_95"], f"{group_id} GEN2 AD")):
            raise RuntimeError(f"{group_id} GEN2 AD evidence mismatch")
        if not close(ngn_ad, finite(row["ngn2_AD_map50_95"], f"{group_id} NGN2 AD")):
            raise RuntimeError(f"{group_id} NGN2 AD evidence mismatch")
        if not close(gen_nad, finite(row["gen2_NAD_map50_95"], f"{group_id} GEN2 NAD")):
            raise RuntimeError(f"{group_id} GEN2 NAD evidence mismatch")
        if not close(ngn_nad, finite(row["ngn2_NAD_map50_95"], f"{group_id} NGN2 NAD")):
            raise RuntimeError(f"{group_id} NGN2 NAD evidence mismatch")

        d_gen = max(0.0, gen_nad)
        d_ngn = max(0.0, ngn_nad)
        sensitivity = max(d_gen, (d_gen + d_ngn) / 2.0)
        removed_parameters = finite(row["parameters_removed"], f"{group_id} removed parameters")
        removed_gflops = finite(row["gflops_removed"], f"{group_id} removed GFLOPs")
        rp = removed_parameters / parameters
        rf = removed_gflops / gflops
        benefit = math.sqrt(rp * rf)
        prunability = benefit / (benefit + sensitivity) if benefit > 0 or sensitivity > 0 else float("nan")
        values = (gen_pruned, ngn_pruned, gen_ad, ngn_ad, gen_nad, ngn_nad, d_gen, d_ngn, sensitivity, rp, rf, benefit, prunability)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{group_id} generated a non-finite updated-formula value")
        if min(removed_parameters, removed_gflops, rp, rf, benefit, sensitivity) < 0:
            raise RuntimeError(f"{group_id} generated an invalid negative magnitude")

        shared = {
            "ratio_percent": ratio_percent,
            "group_id": group_id,
            "group_kind": row["group_kind"],
            "representative_root": row["representative_root"],
        }
        t3.append({
            **shared,
            "mAP_GEN2_baseline": gen_map,
            "mAP_GEN2_pruned": gen_pruned,
            "AD_GEN2": gen_ad,
            "NAD_GEN2_signed": gen_nad,
            "D_GEN2_nonnegative_harm": d_gen,
            "mAP_NGN2_baseline": ngn_map,
            "mAP_NGN2_pruned": ngn_pruned,
            "AD_NGN2": ngn_ad,
            "NAD_NGN2_signed": ngn_nad,
            "D_NGN2_nonnegative_harm": d_ngn,
            "sensitivity_S_GEN_PROTECTED": sensitivity,
        })
        t4.append({
            **shared,
            "parameters_removed": int(removed_parameters),
            "gflops_removed": removed_gflops,
            "baseline_parameters": int(parameters),
            "baseline_gflops": gflops,
            "RP_parameter_reduction_fraction": rp,
            "RF_gflop_reduction_fraction": rf,
            "B_geometric_resource_benefit": benefit,
            "sensitivity_S_GEN_PROTECTED": sensitivity,
            "P_updated_prunability": prunability,
        })

    t3.sort(key=lambda item: (item["sensitivity_S_GEN_PROTECTED"], item["group_id"]))
    for rank, row in enumerate(t3, start=1):
        row["rank_by_sensitivity_ascending"] = rank
    t4.sort(key=lambda item: (-item["P_updated_prunability"], -item["B_geometric_resource_benefit"], item["sensitivity_S_GEN_PROTECTED"], item["group_id"]))
    for rank, row in enumerate(t4, start=1):
        row["rank_by_prunability_descending"] = rank
    t3_by_group = {row["group_id"]: row for row in t3}
    sequence = []
    for rank, row in enumerate(t4, start=1):
        t3_row = t3_by_group[row["group_id"]]
        sequence.append({
            "sequence_rank": rank,
            "T4_rank": row["rank_by_prunability_descending"],
            "ratio_percent": ratio_percent,
            "group_id": row["group_id"],
            "group_kind": row["group_kind"],
            "representative_root": row["representative_root"],
            "P_updated_prunability": row["P_updated_prunability"],
            "B_geometric_resource_benefit": row["B_geometric_resource_benefit"],
            "sensitivity_S_GEN_PROTECTED": t3_row["sensitivity_S_GEN_PROTECTED"],
            "parameters_removed_isolated": row["parameters_removed"],
            "gflops_removed_isolated": row["gflops_removed"],
        })
    audit = {
        "ratio_percent": ratio_percent,
        "source": relative(source_path),
        "source_sha256": sha256(source_path),
        "rows": len(source),
        "unique_groups": len(set(groups)),
        "all_numeric_values_finite": True,
        "top_five": [row["group_id"] for row in t4[:5]],
    }
    return t3, t4, sequence, audit


def main() -> int:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Completed BDD T1/T2 evidence is missing: {SOURCE_ROOT}")
    gen_path, _, gen_map, gen_parameters, gen_gflops = baseline("GEN2")
    ngn_path, _, ngn_map, ngn_parameters, ngn_gflops = baseline("NGN2")
    if not (close(gen_parameters, ngn_parameters) and close(gen_gflops, ngn_gflops)):
        raise RuntimeError("GEN2 and NGN2 baseline structures differ; a shared resource denominator is invalid")

    formula = {
        "formula_id": FORMULA_ID,
        "status": "FROZEN_FOR_BDD_T3_T4",
        "domains": {"general": "GEN2", "adverse": "NGN2"},
        "equations": {
            "NAD": "(mAP_baseline - mAP_pruned) / mAP_baseline",
            "D": "max(0, NAD)",
            "S": "max(D_GEN2, (D_GEN2 + D_NGN2) / 2)",
            "RP": "removed_parameters / baseline_parameters",
            "RF": "removed_GFLOPs / baseline_GFLOPs",
            "B": "sqrt(RP * RF)",
            "P": "B / (B + S)",
        },
        "rules": {
            "negative_AD_policy": "Retain signed values in T3 but assign zero nonnegative harm D; no reward for apparent isolated accuracy gain.",
            "tie_break": ["larger B", "smaller S", "lexicographic group_id"],
            "no_fixed_accuracy_gate": "The revised formula replaces the prior fixed 5% gate with the bounded B/(B+S) trade-off.",
            "scope": "T3/T4 isolated-evidence scoring only. It is not evidence of cumulative 56% accuracy.",
        },
        "source_t1_t2_root": relative(SOURCE_ROOT),
        "source_baselines": {
            "GEN2": {"path": relative(gen_path), "sha256": sha256(gen_path), "mAP50_95": gen_map},
            "NGN2": {"path": relative(ngn_path), "sha256": sha256(ngn_path), "mAP50_95": ngn_map},
        },
        "baseline_resources": {"parameters": int(gen_parameters), "gflops": gen_gflops},
    }
    atomic_json(OUTPUT_ROOT / "FORMULA_CONFIG.json", formula)

    audits = []
    for source_slug, output_slug, ratio in RATIOS:
        source = SOURCE_ROOT / source_slug / "tables" / f"PAIRED_GEN2_NGN2_{source_slug.upper()}.csv"
        t3, t4, sequence, audit = build_ratio(source, ratio, gen_map, ngn_map, gen_parameters, gen_gflops)
        table_dir = OUTPUT_ROOT / source_slug / "tables"
        t3_path = table_dir / f"T3_UPDATED_GEN_PROTECTED_{output_slug.upper()}PCT.csv"
        t4_path = table_dir / f"T4_UPDATED_PRUNABILITY_{output_slug.upper()}PCT.csv"
        sequence_path = table_dir / f"PRUNING_SEQUENCE_UPDATED_{output_slug.upper()}PCT.csv"
        atomic_csv(t3_path, T3_FIELDS, t3)
        atomic_csv(t4_path, T4_FIELDS, t4)
        atomic_csv(sequence_path, SEQUENCE_FIELDS, sequence)
        audit.update({
            "T3": relative(t3_path), "T3_sha256": sha256(t3_path),
            "T4": relative(t4_path), "T4_sha256": sha256(t4_path),
            "sequence": relative(sequence_path), "sequence_sha256": sha256(sequence_path),
        })
        audits.append(audit)

    audit = {
        "schema": "bdd_updated_t3_t4_audit_v1",
        "status": "PASS",
        "formula_id": FORMULA_ID,
        "ratios": audits,
        "builder": relative(Path(__file__)),
        "builder_sha256": sha256(Path(__file__)),
    }
    atomic_json(OUTPUT_ROOT / "VALIDATION_AUDIT.json", audit)
    readme = """# BDD GEN-2 / NGN-2 Updated T3/T4 Decision Tables

Status: complete. This evidence-only analysis reads the completed, immutable
BDD T1/T2 isolated-pruning records at 12.5%, 25%, and 37.5%. It does not train,
prune, fine-tune, or edit the source evidence.

## Revised formula

```text
NAD_d,i = (mAP_d,baseline - mAP_d,pruned,i) / mAP_d,baseline
D_d,i   = max(0, NAD_d,i)
S_i     = max(D_GEN2,i, (D_GEN2,i + D_NGN2,i) / 2)
RP_i    = removed_parameters_i / baseline_parameters
RF_i    = removed_GFLOPs_i / baseline_GFLOPs
B_i     = sqrt(RP_i * RF_i)
P_i     = B_i / (B_i + S_i)
```

GEN2 is protected by `S`: a high GEN2 harm cannot be offset by a lower NGN2
harm. Signed negative AD/NAD values remain visible in T3 but contribute zero
harm, consistent with the completed paired-bootstrap analysis.

`T3_UPDATED_*.csv` contains signed and nonnegative damage plus sensitivity.
`T4_UPDATED_*.csv` ranks all 51 groups by the revised bounded prunability.
`PRUNING_SEQUENCE_UPDATED_*.csv` freezes the ranking only. It is **not** an
executed cumulative mask plan and is not evidence of a 56% model.

## Next experimental stages

1. Use the 25% T4 sequence for a matched approximately 10.25% raw cumulative
   validation (new BDD T5 pilot).
2. If structurally valid, construct a separate 56% raw cumulative BDD T5 plan
   from a predeclared ratio-specific T4 sequence and freeze its exact masks.
3. Evaluate raw T5, then run end-only T6 and staged T7 recovery on those exact
   same frozen architectures.
"""
    atomic_text(OUTPUT_ROOT / "README.md", readme)
    print(json.dumps({"status": "PASS", "output": str(OUTPUT_ROOT), "formula_id": FORMULA_ID, "ratios": audits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
