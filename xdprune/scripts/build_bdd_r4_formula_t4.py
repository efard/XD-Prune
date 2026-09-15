"""Build BDD Formula R4 weighted T3/T4 tables without changing source evidence.

This is the formula-ablation builder requested after supervisor discussion.  It
uses the completed BDD GEN2/NGN2 isolated T1/T2 records and implements two
Formula R4 variants from ``260819_Formula_R4.docx``:

signed (non-clipped)
    S_i = (2/3) NAD_GEN2,i + (1/3) NAD_NGN2,i

clipped
    D_d,i = max(0, NAD_d,i)
    S_i = (2/3) D_GEN2,i + (1/3) D_NGN2,i

For both variants:

    R_i = removed dependency-group parameters_i / baseline parameters
    P_i = R_i / (S_i + epsilon)

The builder deliberately keeps the current BDD_UPDATED_GEN_PROTECTED_V1
results untouched.  It creates distinct output roots for the two Formula R4
ablations.  It does not train, prune, alter checkpoints, or rerun T1/T2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
SOURCE_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_t1_t2_v1"
OUTPUT_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_formula_r4_ablation_v1"
FORMULA_SOURCE = Path.home() / "Downloads" / "260819_Formula_R4.docx"

ALPHA = 2.0 / 3.0
VARIANTS = {
    "signed": {
        "formula_id": "BDD_R4_SIGNED_WEIGHTED_PARAM_ONLY_V1",
        "folder": "r4_signed_alpha2of3_param_only",
        "description": "Non-clipped Formula R4: signed NAD enters the weighted sensitivity.",
        "sensitivity_equation": "(2/3) * NAD_GEN2 + (1/3) * NAD_NGN2",
        "negative_ad_policy": "Signed negative NAD remains in sensitivity exactly as measured.",
    },
    "clipped": {
        "formula_id": "BDD_R4_CLIPPED_WEIGHTED_PARAM_ONLY_V1",
        "folder": "r4_clipped_alpha2of3_param_only",
        "description": "Clipped Formula R4: only non-negative damage enters weighted sensitivity.",
        "sensitivity_equation": "(2/3) * max(0, NAD_GEN2) + (1/3) * max(0, NAD_NGN2)",
        "negative_ad_policy": "Signed negative NAD remains reported but contributes zero to sensitivity.",
    },
}
RATIOS = {
    "12.5": ("12_5pct", "12_5", 12.5),
    "25": ("25pct", "25", 25.0),
    "37.5": ("37_5pct", "37_5", 37.5),
}

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
    "sensitivity_S_R4",
]
T4_FIELDS = [
    "rank_by_prunability_descending",
    "ratio_percent",
    "group_id",
    "group_kind",
    "representative_root",
    "parameters_removed",
    "baseline_parameters",
    "R_parameter_reduction_fraction",
    "sensitivity_S_R4",
    "epsilon",
    "P_R4_prunability",
]
SEQUENCE_FIELDS = [
    "sequence_rank",
    "T4_rank",
    "ratio_percent",
    "group_id",
    "group_kind",
    "representative_root",
    "P_R4_prunability",
    "sensitivity_S_R4",
    "parameters_removed_isolated",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


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
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def expected_group_ids() -> set[str]:
    return {f"DG{index:03d}" for index in range(1, 43)} | {f"CDG{index:03d}" for index in range(1, 10)}


def baseline(domain: str) -> tuple[Path, float, int]:
    path = SOURCE_ROOT / "baselines" / f"{domain}.json"
    record = read_json(path)
    if record.get("status") != "PASS" or record.get("domain") != domain:
        raise RuntimeError(f"Invalid frozen {domain} baseline: {path}")
    map50_95 = finite(record["metrics"]["map50_95"], f"{domain} mAP50-95")
    parameters = int(finite(record["structure"]["parameters"], f"{domain} parameters"))
    if map50_95 <= 0 or parameters <= 0:
        raise RuntimeError(f"Invalid frozen {domain} baseline metric/resource")
    return path, map50_95, parameters


def source_table(slug: str) -> Path:
    return SOURCE_ROOT / slug / "tables" / f"PAIRED_GEN2_NGN2_{slug.upper()}.csv"


def sensitivity(variant: str, gen_nad: float, ngn_nad: float) -> tuple[float, float, float]:
    d_gen, d_ngn = max(0.0, gen_nad), max(0.0, ngn_nad)
    if variant == "signed":
        result = ALPHA * gen_nad + (1.0 - ALPHA) * ngn_nad
    elif variant == "clipped":
        result = ALPHA * d_gen + (1.0 - ALPHA) * d_ngn
    else:
        raise ValueError(f"Unsupported Formula R4 variant: {variant}")
    if not math.isfinite(result):
        raise RuntimeError("Formula R4 sensitivity is not finite")
    return result, d_gen, d_ngn


def build_tables(
    variant: str,
    ratio_percent: float,
    source_path: Path,
    gen_baseline: float,
    ngn_baseline: float,
    baseline_parameters: int,
    epsilon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(source_path)
    groups = [row["group_id"] for row in source]
    if len(source) != 51 or len(set(groups)) != 51 or set(groups) != expected_group_ids():
        raise RuntimeError(f"{source_path} must contain all 51 canonical groups exactly once")

    t3: list[dict[str, Any]] = []
    t4: list[dict[str, Any]] = []
    for row in source:
        group_id = row["group_id"]
        if not close(finite(row["ratio_percent"], f"{group_id} ratio"), ratio_percent):
            raise RuntimeError(f"{group_id}: source ratio does not match {ratio_percent}")
        gen_pruned = finite(row["gen2_pruned_map50_95"], f"{group_id} GEN2 mAP50-95")
        ngn_pruned = finite(row["ngn2_pruned_map50_95"], f"{group_id} NGN2 mAP50-95")
        gen_ad, ngn_ad = gen_baseline - gen_pruned, ngn_baseline - ngn_pruned
        gen_nad, ngn_nad = gen_ad / gen_baseline, ngn_ad / ngn_baseline
        if not close(gen_ad, finite(row["gen2_AD_map50_95"], f"{group_id} GEN2 AD")):
            raise RuntimeError(f"{group_id}: GEN2 AD disagrees with frozen source evidence")
        if not close(ngn_ad, finite(row["ngn2_AD_map50_95"], f"{group_id} NGN2 AD")):
            raise RuntimeError(f"{group_id}: NGN2 AD disagrees with frozen source evidence")
        if not close(gen_nad, finite(row["gen2_NAD_map50_95"], f"{group_id} GEN2 NAD")):
            raise RuntimeError(f"{group_id}: GEN2 NAD disagrees with frozen source evidence")
        if not close(ngn_nad, finite(row["ngn2_NAD_map50_95"], f"{group_id} NGN2 NAD")):
            raise RuntimeError(f"{group_id}: NGN2 NAD disagrees with frozen source evidence")

        score_sensitivity, d_gen, d_ngn = sensitivity(variant, gen_nad, ngn_nad)
        denominator = score_sensitivity + epsilon
        if denominator <= 0:
            raise RuntimeError(
                f"{group_id}: Formula R4 denominator is non-positive ({denominator:.12g}). "
                "Increase epsilon before ranking; a non-positive denominator reverses the intended ranking."
            )
        removed_parameters = int(finite(row["parameters_removed"], f"{group_id} parameters removed"))
        if removed_parameters <= 0:
            raise RuntimeError(f"{group_id}: no physical parameter reduction in frozen source evidence")
        resource = removed_parameters / baseline_parameters
        prunability = resource / denominator
        values = (gen_pruned, ngn_pruned, gen_ad, ngn_ad, gen_nad, ngn_nad, d_gen, d_ngn, score_sensitivity, resource, prunability)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{group_id}: Formula R4 generated a non-finite value")

        shared = {
            "ratio_percent": ratio_percent,
            "group_id": group_id,
            "group_kind": row["group_kind"],
            "representative_root": row["representative_root"],
        }
        t3.append({
            **shared,
            "mAP_GEN2_baseline": gen_baseline,
            "mAP_GEN2_pruned": gen_pruned,
            "AD_GEN2": gen_ad,
            "NAD_GEN2_signed": gen_nad,
            "D_GEN2_nonnegative_harm": d_gen,
            "mAP_NGN2_baseline": ngn_baseline,
            "mAP_NGN2_pruned": ngn_pruned,
            "AD_NGN2": ngn_ad,
            "NAD_NGN2_signed": ngn_nad,
            "D_NGN2_nonnegative_harm": d_ngn,
            "sensitivity_S_R4": score_sensitivity,
        })
        t4.append({
            **shared,
            "parameters_removed": removed_parameters,
            "baseline_parameters": baseline_parameters,
            "R_parameter_reduction_fraction": resource,
            "sensitivity_S_R4": score_sensitivity,
            "epsilon": epsilon,
            "P_R4_prunability": prunability,
        })

    t3.sort(key=lambda item: (item["sensitivity_S_R4"], item["group_id"]))
    for rank, row in enumerate(t3, start=1):
        row["rank_by_sensitivity_ascending"] = rank
    t4.sort(key=lambda item: (-item["P_R4_prunability"], -item["R_parameter_reduction_fraction"], item["sensitivity_S_R4"], item["group_id"]))
    for rank, row in enumerate(t4, start=1):
        row["rank_by_prunability_descending"] = rank
    t3_by_group = {row["group_id"]: row for row in t3}
    sequence = [{
        "sequence_rank": rank,
        "T4_rank": row["rank_by_prunability_descending"],
        "ratio_percent": ratio_percent,
        "group_id": row["group_id"],
        "group_kind": row["group_kind"],
        "representative_root": row["representative_root"],
        "P_R4_prunability": row["P_R4_prunability"],
        "sensitivity_S_R4": t3_by_group[row["group_id"]]["sensitivity_S_R4"],
        "parameters_removed_isolated": row["parameters_removed"],
    } for rank, row in enumerate(t4, start=1)]
    audit = {
        "ratio_percent": ratio_percent,
        "source": relative(source_path),
        "source_sha256": sha256(source_path),
        "rows": len(source),
        "unique_groups": len(set(groups)),
        "minimum_sensitivity": min(row["sensitivity_S_R4"] for row in t3),
        "minimum_denominator": min(row["sensitivity_S_R4"] + epsilon for row in t3),
        "top_five_groups": [row["group_id"] for row in t4[:5]],
        "all_numeric_values_finite": True,
    }
    return t3, t4, sequence, audit


def write_readme(root: Path, variant: str, epsilon: float, ratio_keys: list[str]) -> None:
    spec = VARIANTS[variant]
    title = "signed/non-clipped" if variant == "signed" else "clipped"
    ratios = ", ".join(RATIOS[key][2].__format__("g") + "%" for key in ratio_keys)
    text = f"""# BDD Formula R4 {title} weighted ranking

Status: completed evidence-only Formula R4 ranking. This folder is separate
from the existing `BDD_UPDATED_GEN_PROTECTED_V1` ranking and does not alter
T1/T2 evidence, checkpoints, raw-T5 models, or the completed current T7 runs.

## Scope

The ranking uses paired **GEN2 and NGN2** isolated-pruning evidence. NGN2 is
the planned recovery-only domain for the following formula ablation; it is not
the sole input to this cross-domain score. Generated local-pruning ratios:
{ratios}.

## Formula

```text
NAD_d,i = (mAP_d,baseline - mAP_d,pruned,i) / mAP_d,baseline
D_d,i   = max(0, NAD_d,i)
alpha   = 2/3
S_i     = {spec['sensitivity_equation']}
R_i     = removed dependency-group parameters_i / baseline parameters
P_i     = R_i / (S_i + epsilon)
epsilon = {epsilon:.12g}
```

{spec['negative_ad_policy']}

`epsilon` is recorded explicitly as a predeclared Formula R4 component. It
keeps all signed-formula denominators positive across the completed BDD T1/T2
evidence, including the small negative signed sensitivities at 12.5% and 25%.
Because it can affect the rank of near-zero-sensitivity groups, it must not be
silently changed before the later raw-56% construction.

## Files

- `FORMULA_CONFIG.json`: exact formula, hashes, and baseline references.
- `<ratio>/tables/T3_R4_*.csv`: signed observations and Formula R4 sensitivity.
- `<ratio>/tables/T4_R4_*.csv`: parameter-only Formula R4 prunability ranking.
- `<ratio>/tables/PRUNING_SEQUENCE_R4_*.csv`: ranking order only; not a
  cumulative mask plan and not an accuracy result.
- `VALIDATION_AUDIT.json`: evidence-integrity checks and generated-file hashes.

Before recovery, a separate raw-T5 builder must derive a new dependency-safe
56% architecture from this ranking and freeze the resulting exact masks. That
architecture must not reuse the current-formula masks.
"""
    atomic_text(root / "README.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--ratio", default="37.5", choices=[*RATIOS, "all"], help="Local pruning evidence ratio to score; default: 37.5.")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Positive Formula R4 denominator guard; default: 1e-3.")
    args = parser.parse_args()
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("--epsilon must be finite and > 0")
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Completed BDD T1/T2 evidence is missing: {SOURCE_ROOT}")
    if not FORMULA_SOURCE.is_file():
        raise FileNotFoundError(f"Formula R4 source document is missing: {FORMULA_SOURCE}")

    gen_path, gen_map, gen_parameters = baseline("GEN2")
    ngn_path, ngn_map, ngn_parameters = baseline("NGN2")
    if gen_parameters != ngn_parameters:
        raise RuntimeError("GEN2 and NGN2 baseline parameter counts differ; shared Formula R4 R_i is invalid")
    selected_keys = list(RATIOS) if args.ratio == "all" else [args.ratio]
    spec = VARIANTS[args.variant]
    root = OUTPUT_ROOT / spec["folder"]
    formula_config = {
        "schema": "bdd_formula_r4_weighted_parameter_only_v1",
        "status": "FROZEN_FOR_FORMULA_ABLATION_T3_T4",
        "formula_id": spec["formula_id"],
        "variant": args.variant,
        "variant_description": spec["description"],
        "domains": {"general": "GEN2", "adverse": "NGN2"},
        "alpha": {"GEN2": ALPHA, "NGN2": 1.0 - ALPHA},
        "epsilon": args.epsilon,
        "equations": {
            "NAD": "(mAP_baseline - mAP_pruned) / mAP_baseline",
            "D": "max(0, NAD)",
            "S": spec["sensitivity_equation"],
            "R": "removed dependency-group parameters / baseline parameters",
            "P": "R / (S + epsilon)",
        },
        "negative_AD_policy": spec["negative_ad_policy"],
        "formula_source_document": str(FORMULA_SOURCE),
        "formula_source_document_sha256": sha256(FORMULA_SOURCE),
        "source_t1_t2_root": relative(SOURCE_ROOT),
        "source_baselines": {
            "GEN2": {"path": relative(gen_path), "sha256": sha256(gen_path), "mAP50_95": gen_map, "parameters": gen_parameters},
            "NGN2": {"path": relative(ngn_path), "sha256": sha256(ngn_path), "mAP50_95": ngn_map, "parameters": ngn_parameters},
        },
        "scope": "T3/T4 isolated-evidence formula ablation only. Not a cumulative or recovery result.",
    }
    atomic_json(root / "FORMULA_CONFIG.json", formula_config)

    audits = []
    for key in selected_keys:
        source_slug, output_slug, ratio_percent = RATIOS[key]
        t3, t4, sequence, audit = build_tables(
            args.variant, ratio_percent, source_table(source_slug), gen_map, ngn_map, gen_parameters, args.epsilon,
        )
        table_root = root / source_slug / "tables"
        variant_upper = args.variant.upper()
        t3_path = table_root / f"T3_R4_{variant_upper}_WEIGHTED_{output_slug.upper()}PCT.csv"
        t4_path = table_root / f"T4_R4_{variant_upper}_PRUNABILITY_{output_slug.upper()}PCT.csv"
        sequence_path = table_root / f"PRUNING_SEQUENCE_R4_{variant_upper}_{output_slug.upper()}PCT.csv"
        atomic_csv(t3_path, T3_FIELDS, t3)
        atomic_csv(t4_path, T4_FIELDS, t4)
        atomic_csv(sequence_path, SEQUENCE_FIELDS, sequence)
        audit.update({
            "T3": relative(t3_path), "T3_sha256": sha256(t3_path),
            "T4": relative(t4_path), "T4_sha256": sha256(t4_path),
            "sequence": relative(sequence_path), "sequence_sha256": sha256(sequence_path),
        })
        audits.append(audit)
    audit_payload = {
        "schema": "bdd_formula_r4_weighted_parameter_only_audit_v1",
        "status": "PASS",
        "formula_id": spec["formula_id"],
        "variant": args.variant,
        "epsilon": args.epsilon,
        "ratios": audits,
        "builder": relative(SELF),
        "builder_sha256": sha256(SELF),
    }
    atomic_json(root / "VALIDATION_AUDIT.json", audit_payload)
    write_readme(root, args.variant, args.epsilon, selected_keys)
    print(json.dumps({"status": "PASS", "output_root": relative(root), "formula_id": spec["formula_id"], "audit": audits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
