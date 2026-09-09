"""Generate the unrestricted Global-L1 p56 competitor.

This reuses the independently verified p10 implementation without modifying
the completed p10 experiment.  Only the output paths and target parameter
count change.  Selection remains a global ranking of current raw L1 channel
magnitudes over the 42 stock DepGraph-compatible roots, with DepGraph rebuilt
after every accepted channel removal.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


SELF = Path(__file__).resolve()
EXPERIMENT_ROOT = SELF.parent
P10_GENERATOR = (
    EXPERIMENT_ROOT.parent
    / "global_l1_p10_reproduction"
    / "generate_global_l1_p10.py"
)

SCHEMA = "global_l1_p56_unrestricted_reproduction_v1"
TARGET_PARAMETERS = 1_083_192


def load_reference_generator() -> ModuleType:
    if not P10_GENERATOR.is_file():
        raise FileNotFoundError(P10_GENERATOR)
    specification = importlib.util.spec_from_file_location(
        "global_l1_p10_reference_generator",
        P10_GENERATOR,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load reference generator: {P10_GENERATOR}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def configure(module: ModuleType) -> None:
    raw_root = EXPERIMENT_ROOT / "raw"
    module.OUTPUT_ROOT = EXPERIMENT_ROOT
    module.SCHEMA = SCHEMA
    module.RAW_ROOT = raw_root
    module.RAW_MODEL = raw_root / "global_l1_gen_p56_unrestricted_raw.pth"
    module.SELECTION_LOG = raw_root / "selection_sequence.csv"
    module.MANIFEST = EXPERIMENT_ROOT / "experiment_manifest.json"
    module.TARGET_PARAMETERS = TARGET_PARAMETERS


def add_manifest_compatibility_fields(module: ModuleType) -> dict:
    manifest = json.loads(module.MANIFEST.read_text(encoding="utf-8"))
    manifest["parameter_reduction_percent"] = manifest["final_reduction_percent"]
    manifest["comparison_target"] = (
        "Matched to the proposed T7 GEN p56 target parameter count"
    )
    manifest["per_root_retention_cap"] = None
    manifest["minimum_channels_per_root"] = (
        "DepGraph legality only; no handcrafted minimum-width cap"
    )
    module.atomic_json(module.MANIFEST, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-steps",
        type=int,
        help="Run only N disposable selections without creating persistent outputs.",
    )
    arguments = parser.parse_args()
    if arguments.preflight_steps is not None and arguments.preflight_steps < 1:
        parser.error("--preflight-steps must be positive")

    generator = load_reference_generator()
    configure(generator)
    result = generator.run(max_steps=arguments.preflight_steps)
    if arguments.preflight_steps is None:
        result = add_manifest_compatibility_fields(generator)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
