#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_ROOT/1_scripts/global_l1_stage5e}"
RESULT_PARENT="${RESULT_PARENT:-$PROJECT_ROOT/5_reproduction/global_l1_stage5e_results}"
IMAGE="${IMAGE:-/opt/software/apptainer-images/pytorch-25.06.sif}"
PYTHONPATH_SERVER="${PYTHONPATH_SERVER:-/project/ko/afm176/yolo-packages}"

RAW_ROOT="${RAW_ROOT:-$PROJECT_ROOT/5_reproduction/global_l1_stage5c_results/latest}"

GEN_MODEL="${GEN_MODEL:-$RAW_ROOT/GEN/GEN_global_L1_42root_raw.pt}"
SNOW_MODEL="${SNOW_MODEL:-$RAW_ROOT/SNOW/SNOW_global_L1_42root_raw.pt}"

GEN_DATA_ROOT="${GEN_DATA_ROOT:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/GEN_MIO_TCD_exact}"
SNOW_DATA_ROOT="${SNOW_DATA_ROOT:-/project/ko/afm176/datasets/1_data/reproduction_exact_v1/SNOW_ACDC_exact}"

resolve_yaml() {
  local root="$1"

  for candidate in \
    "$root/data.yaml" \
    "$root/dataset.yaml" \
    "$root/data.yml" \
    "$root/dataset.yml" \
    "$root/dataset_GEN_local.yaml" \
    "$root/dataset_SNOW_local.yaml"
  do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  find "$root" -maxdepth 2 -type f \
    \( -name '*.yaml' -o -name '*.yml' \) \
    | sort \
    | head -n 1
}

GEN_DATA="${GEN_DATA:-$(resolve_yaml "$GEN_DATA_ROOT")}"
SNOW_DATA="${SNOW_DATA:-$(resolve_yaml "$SNOW_DATA_ROOT")}"

for file in \
  "$GEN_MODEL" \
  "$SNOW_MODEL" \
  "$GEN_DATA" \
  "$SNOW_DATA"
do
  if [[ ! -f "$file" ]]; then
    echo "Required file not found: $file" >&2
    exit 1
  fi
done

echo "GEN raw model:  $GEN_MODEL"
echo "SNOW raw model: $SNOW_MODEL"
echo "GEN data YAML:  $GEN_DATA"
echo "SNOW data YAML: $SNOW_DATA"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$RESULT_PARENT/recovery_$STAMP"
LOG_DIR="$PROJECT_ROOT/logs/global_l1_stage5e"

mkdir -p "$OUT/GEN" "$OUT/SNOW" "$LOG_DIR"

module load apptainer

run_domain() {
  local domain="$1"
  local model="$2"
  local data="$3"
  local domain_out="$OUT/$domain"
  local log="$LOG_DIR/${domain,,}_recovery_$STAMP.log"

  echo
  echo "===== STARTING $domain 20-EPOCH RECOVERY ====="

  # Some Plato GPU jobs return signal 11 during CUDA/UCX cleanup after all files have already been written.
  # Capture the true Apptainer exit code, then accept the run only when its saved summary says it passed.
  set +e
  set +o pipefail

  apptainer exec --nv \
    -B /home/afm176:/home/afm176 \
    -B /project/ko:/project/ko \
    --env PYTHONPATH="$PYTHONPATH_SERVER" \
    "$IMAGE" \
    python "$SCRIPT_DIR/08_full_20epoch_recovery.py" \
      --domain "$domain" \
      --raw-model "$model" \
      --data "$data" \
      --output-dir "$domain_out" \
      --epochs 20 \
      --imgsz 640 \
      --batch 16 \
      --workers 4 \
      --seed 42 \
    2>&1 | tee "$log"

  local app_rc="${PIPESTATUS[0]}"

  set -o pipefail
  set -e

  local summary="$domain_out/domain_recovery_summary.json"

  if [[ -f "$summary" ]] && \
     grep -q '"status": "PASSED_FULL_20E_RECOVERY"' "$summary"
  then
    echo "$domain recovery summary passed."

    if [[ "$app_rc" -ne 0 ]]; then
      echo \
        "Note: Apptainer exited with code $app_rc after the saved run passed; " \
        "treating this as a post-completion CUDA/UCX cleanup issue."
    fi

    return 0
  fi

  echo "$domain recovery failed or did not write a passing summary." >&2
  echo "Apptainer exit code: $app_rc" >&2
  return 1
}

run_domain "GEN" "$GEN_MODEL" "$GEN_DATA"
run_domain "SNOW" "$SNOW_MODEL" "$SNOW_DATA"

python - "$OUT" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

results = []
for domain in ["GEN", "SNOW"]:
    path = out / domain / "domain_recovery_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    results.append(data)

checks = {
    "gen_passed": results[0]["status"] == "PASSED_FULL_20E_RECOVERY",
    "snow_passed": results[1]["status"] == "PASSED_FULL_20E_RECOVERY",
    "both_exact_architectures_preserved": all(
        item["checks"]["best_architecture_preserved"]
        and item["checks"]["last_architecture_preserved"]
        for item in results
    ),
    "both_parameter_counts_preserved": all(
        item["checks"]["best_parameters_preserved"]
        and item["checks"]["last_parameters_preserved"]
        for item in results
    ),
    "both_final_validations_completed": all(
        item["checks"]["final_validation_completed"]
        for item in results
    ),
}

overall = {
    "status": (
        "PASSED_MATCHED_20E_RECOVERY"
        if all(checks.values())
        else "FAILED_MATCHED_20E_RECOVERY"
    ),
    "checks": checks,
    "results": results,
    "safe_next_step": (
        "Compare recovered Global L1 against baseline, teammate T6, and "
        "whole-layer replacement; then export and benchmark on NCNN/PYNQ."
    ),
}

(out / "stage5e_summary.json").write_text(
    json.dumps(overall, indent=2),
    encoding="utf-8",
)

rows = []
for item in results:
    metrics = item["final_metrics"]
    rows.append(
        {
            "domain": item["domain"],
            "status": item["status"],
            "parameters": item["best"]["parameters"],
            "baseline_map50_95": metrics["baseline_map50_95"],
            "raw_map50_95": metrics["raw_map50_95"],
            "recovered_map50_95": metrics["recovered_map50_95"],
            "accuracy_retention_percent": metrics[
                "accuracy_retention_percent"
            ],
            "signed_accuracy_drop_map50_95": metrics[
                "signed_accuracy_drop_map50_95"
            ],
            "recovered_map50": metrics["recovered_map50"],
            "recovered_map75": metrics["recovered_map75"],
            "precision": metrics["recovered_precision"],
            "recall": metrics["recovered_recall"],
            "best_checkpoint": item["recovered_best_copy"]["checkpoint"],
        }
    )

with (out / "matched_recovery_results.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(json.dumps(overall, indent=2))
PY

ln -sfn "$OUT" "$RESULT_PARENT/latest"

echo
echo "===== STAGE 5E SUMMARY ====="
cat "$OUT/stage5e_summary.json"
