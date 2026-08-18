#!/usr/bin/env bash
set -Eeuo pipefail

# Create a compact Stage 1-4 handoff package containing only:
# 1. Files useful for report writing.
# 2. Recovered best.pt candidate models for later PYNQ deployment.
#
# Raw datasets, caches, optimizer checkpoints, last.pt files, and most
# intermediate model checkpoints are intentionally excluded.

PROJECT_ROOT="${PROJECT_ROOT:-/home/afm176/yolo_project}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/5_reproduction}"
OUTPUT_PARENT="${1:-$PROJECT_ROOT/6_exports}"

STAGE2_PREFERRED="$RESULTS_ROOT/stage2_results/layer_replacement_sweep_20260729_151557"
STAGE3_PREFERRED="$RESULTS_ROOT/stage3_results/near_target_20260729_172754"
STAGE4_PREFERRED="$RESULTS_ROOT/stage4_results/recovery_C007_L9_20260729_192450"

timestamp="$(date +%Y%m%d_%H%M%S)"
package_name="YOLO26n_stage5_simplified_${timestamp}"
work_dir="$OUTPUT_PARENT/$package_name"
zip_path="$OUTPUT_PARENT/${package_name}.zip"

mkdir -p "$OUTPUT_PARENT"
rm -rf "$work_dir"
mkdir -p \
  "$work_dir/01_report_materials/stage1_baseline" \
  "$work_dir/01_report_materials/stage2_layer_sweep" \
  "$work_dir/01_report_materials/stage3_candidate_search" \
  "$work_dir/01_report_materials/stage4_recovery" \
  "$work_dir/02_pynq_candidate_models" \
  "$work_dir/00_package_information"

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

latest_dir() {
  local parent="$1"
  if [[ -d "$parent" ]]; then
    find "$parent" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  fi
}

resolve_run_dir() {
  local preferred="$1"
  local parent="$2"

  if [[ -d "$preferred" ]]; then
    printf '%s\n' "$preferred"
  else
    latest_dir "$parent"
  fi
}

copy_if_exists() {
  local src="$1"
  local dest="$2"

  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp -aL "$src" "$dest"
  fi
}

copy_report_files() {
  local src="$1"
  local dest="$2"

  [[ -d "$src" ]] || return 0
  mkdir -p "$dest"

  # Only copy compact files useful for reading, tables, plotting, and reporting.
  # Checkpoints, caches, images, and large binary artifacts are excluded.
  while IFS= read -r -d '' file; do
    local rel target
    rel="${file#"$src"/}"
    target="$dest/$rel"
    mkdir -p "$(dirname "$target")"
    cp -aL "$file" "$target"
  done < <(
    find "$src" -type f \
      \( \
        -name '*.csv' -o \
        -name '*.json' -o \
        -name '*.yaml' -o \
        -name '*.yml' -o \
        -name '*.txt' -o \
        -name '*.md' -o \
        -name '*.log' -o \
        -name '*.png' \
      \) \
      ! -name '*.cache' \
      ! -path '*/__pycache__/*' \
      ! -path '*/weights/*' \
      -print0
  )
}

stage1_dir="$(resolve_run_dir \
  "$RESULTS_ROOT/stage1_results" \
  "$RESULTS_ROOT/stage1_results")"

stage2_dir="$(resolve_run_dir \
  "$STAGE2_PREFERRED" \
  "$RESULTS_ROOT/stage2_results")"

stage3_dir="$(resolve_run_dir \
  "$STAGE3_PREFERRED" \
  "$RESULTS_ROOT/stage3_results")"

stage4_dir="$STAGE4_PREFERRED"
if [[ -L "$RESULTS_ROOT/stage4_results/latest" ]]; then
  linked="$(readlink -f "$RESULTS_ROOT/stage4_results/latest")"
  if [[ -d "$linked" ]]; then
    stage4_dir="$linked"
  fi
elif [[ ! -d "$stage4_dir" ]]; then
  stage4_dir="$(latest_dir "$RESULTS_ROOT/stage4_results")"
fi

log "Stage 1 source: ${stage1_dir:-not found}"
log "Stage 2 source: ${stage2_dir:-not found}"
log "Stage 3 source: ${stage3_dir:-not found}"
log "Stage 4 source: ${stage4_dir:-not found}"

# ---------------------------------------------------------------------------
# 1. Report materials
# ---------------------------------------------------------------------------

if [[ -n "${stage1_dir:-}" && -d "$stage1_dir" ]]; then
  copy_report_files \
    "$stage1_dir" \
    "$work_dir/01_report_materials/stage1_baseline/$(basename "$stage1_dir")"
fi

if [[ -n "${stage2_dir:-}" && -d "$stage2_dir" ]]; then
  copy_report_files \
    "$stage2_dir" \
    "$work_dir/01_report_materials/stage2_layer_sweep/$(basename "$stage2_dir")"
fi

if [[ -n "${stage3_dir:-}" && -d "$stage3_dir" ]]; then
  copy_report_files \
    "$stage3_dir" \
    "$work_dir/01_report_materials/stage3_candidate_search/$(basename "$stage3_dir")"
fi

if [[ -n "${stage4_dir:-}" && -d "$stage4_dir" ]]; then
  copy_report_files \
    "$stage4_dir" \
    "$work_dir/01_report_materials/stage4_recovery/$(basename "$stage4_dir")"
else
  echo "ERROR: successful Stage 4 recovery folder was not found." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Recovered best.pt candidate models for PYNQ
# ---------------------------------------------------------------------------

export STAGE4_DIR="$stage4_dir"
export MODEL_DEST="$work_dir/02_pynq_candidate_models"

python3 - <<'PY'
from pathlib import Path
import csv
import hashlib
import os
import re
import shutil

stage4 = Path(os.environ["STAGE4_DIR"])
dest = Path(os.environ["MODEL_DEST"])
dest.mkdir(parents=True, exist_ok=True)

expected = {
    ("C007", "GEN"),
    ("C007", "SNOW"),
    ("L9", "GEN"),
    ("L9", "SNOW"),
}

found = {}
pattern = re.compile(r"(C007|L9)_(GEN|SNOW)", re.IGNORECASE)

# Prefer best.pt files under Stage 4 training runs.
for path in stage4.rglob("best.pt"):
    match = pattern.search(str(path))
    if not match:
        continue
    candidate = match.group(1).upper()
    domain = match.group(2).upper()
    found[(candidate, domain)] = path

# Fall back to stage4_progress.csv run_dir entries.
progress = stage4 / "stage4_progress.csv"
if progress.is_file():
    with progress.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            candidate = str(row.get("candidate", "")).upper()
            domain = str(row.get("domain", "")).upper()
            key = (candidate, domain)
            if key not in expected or key in found:
                continue

            run_dir = row.get("run_dir") or ""
            if run_dir:
                best = Path(run_dir) / "weights" / "best.pt"
                if best.is_file():
                    found[key] = best

missing = sorted(expected - set(found))
if missing:
    formatted = ", ".join(f"{c}/{d}" for c, d in missing)
    raise SystemExit(f"Missing recovered best.pt model(s): {formatted}")

rows = []
for candidate, domain in sorted(expected):
    src = found[(candidate, domain)]
    name = f"YOLO26n_recovered_{candidate}_{domain}_best.pt"
    dst = dest / name
    shutil.copy2(src, dst)

    h = hashlib.sha256()
    with dst.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    rows.append({
        "candidate": candidate,
        "domain": domain,
        "packaged_model": name,
        "source_model": str(src),
        "size_bytes": dst.stat().st_size,
        "sha256": h.hexdigest(),
    })

manifest = dest / "MODEL_MANIFEST.csv"
with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Packaged recovered models:")
for row in rows:
    print(
        f"- {row['candidate']} / {row['domain']}: "
        f"{row['packaged_model']}"
    )
PY

# ---------------------------------------------------------------------------
# Compact README
# ---------------------------------------------------------------------------

cat > "$work_dir/00_package_information/README.txt" <<EOF
YOLO26n simplified Stage 5 handoff package
Generated: $(date --iso-8601=seconds)

Purpose
-------
This compact package contains only:
1. Information useful for writing the experiment report.
2. Recovered best.pt candidate models for later PYNQ deployment.

Report materials
----------------
- Stage 1 baseline validation and environment records, when found.
- Stage 2 complete per-layer result tables and summaries.
- Stage 3 complete near-target candidate result tables and summaries.
- Stage 4 C007/L9 recovery tables, summaries, settings, architecture audits,
  final validation outputs, and training curves/results.

PYNQ candidate models
---------------------
- YOLO26n_recovered_C007_GEN_best.pt
- YOLO26n_recovered_C007_SNOW_best.pt
- YOLO26n_recovered_L9_GEN_best.pt
- YOLO26n_recovered_L9_SNOW_best.pt

These are the recovered Stage 4 best checkpoints. Stage 5 should select the
final candidate before PYNQ conversion and benchmarking. Until that selection
is complete, both C007 and L9 are retained.

Excluded
--------
- Raw datasets
- Cache files
- last.pt files
- Raw Stage 2/3 replacement checkpoints
- Unused intermediate checkpoints
- Optimizer state
- Full experiment script copies
- Large temporary artifacts

Source runs
-----------
Stage 1: ${stage1_dir:-not found}
Stage 2: ${stage2_dir:-not found}
Stage 3: ${stage3_dir:-not found}
Stage 4: ${stage4_dir:-not found}
EOF

# ---------------------------------------------------------------------------
# Manifest and checksums
# ---------------------------------------------------------------------------

export PACKAGE_ROOT="$work_dir"

python3 - <<'PY'
from pathlib import Path
import hashlib
import os

root = Path(os.environ["PACKAGE_ROOT"])
info = root / "00_package_information"
manifest = info / "MANIFEST.tsv"
checksums = info / "SHA256SUMS"

files = sorted(
    p for p in root.rglob("*")
    if p.is_file() and p not in {manifest, checksums}
)

with manifest.open("w", encoding="utf-8") as f:
    f.write("relative_path\tsize_bytes\n")
    for p in files:
        f.write(f"{p.relative_to(root)}\t{p.stat().st_size}\n")

with checksums.open("w", encoding="utf-8") as f:
    for p in files:
        h = hashlib.sha256()
        with p.open("rb") as src:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                h.update(chunk)
        f.write(f"{h.hexdigest()}  {p.relative_to(root)}\n")

print(f"Manifest entries: {len(files)}")
PY

# ---------------------------------------------------------------------------
# ZIP creation
# ---------------------------------------------------------------------------

export PACKAGE_WORK_DIR="$work_dir"
export PACKAGE_ZIP_PATH="$zip_path"

python3 - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["PACKAGE_WORK_DIR"])
zip_path = Path(os.environ["PACKAGE_ZIP_PATH"])

with zipfile.ZipFile(
    zip_path,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
    allowZip64=True,
) as zf:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            arcname = Path(root.name) / path.relative_to(root)
            zf.write(path, arcname)

print(zip_path)
PY

zip_sha="$(sha256sum "$zip_path" | awk '{print $1}')"
zip_size="$(du -h "$zip_path" | awk '{print $1}')"

printf '%s  %s\n' \
  "$zip_sha" \
  "$(basename "$zip_path")" \
  > "${zip_path}.sha256"

echo
echo "===== SIMPLIFIED PACKAGE COMPLETE ====="
echo "ZIP:       $zip_path"
echo "Size:      $zip_size"
echo "SHA-256:   $zip_sha"
echo "Checksum:  ${zip_path}.sha256"
echo
echo "Uncompressed staging folder:"
echo "$work_dir"
