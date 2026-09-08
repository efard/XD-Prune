"""Run the Formula R4 signed 56%-target GEN2 T7 recovery ablation.

This is the GEN2 counterpart to ``run_bdd_t7_r4_signed_ngn2.py``.  It reuses
the already frozen Formula-R4 signed 37.5% raw-T5 architecture and runs the
same five-stage 6-6-8-10-40 epoch recovery schedule (70 total epochs).
It writes to its own labelled recovery folder and never overwrites either the
completed NGN2 R4 result or the prior current-formula GEN2 result.

Prerequisite: ``prepare_bdd_raw56_r4_signed.py --run`` must have completed
successfully and produced the shared raw frozen plan.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
NGN2_WRAPPER = STUDY_ROOT / "scripts" / "run_bdd_t7_r4_signed_ngn2.py"
RESULT_ROOT = (
    STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct"
    / "T7_bdd_gen2_r4_signed_70ep"
)
DOMAIN = "GEN2"
SCHEMA = "bdd_t7_gen2_r4_signed_frozen_masks_6_6_8_10_40_raw56_v2"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the established R4 implementation, then bind its configuration to the
# GEN2 raw-model column before the shared T7 runner is invoked.  This keeps the
# R4 mask interpretation and matched resource-milestone stage policy identical
# across the two domains.
ablation = load_module("_bdd_t7_r4_signed_ngn2", NGN2_WRAPPER)
ablation.SELF = SELF
ablation.DOMAIN = DOMAIN
ablation.RESULT_ROOT = RESULT_ROOT
ablation.SCHEMA = SCHEMA
(
    ablation.STAGE_GROUPS,
    ablation.STAGE_TARGET_REDUCTIONS,
    ablation.EXPECTED_PARAMETERS,
) = ablation.frozen_schedule()
ablation.FROZEN_GROUP_SEQUENCE = tuple(
    group for stage in ablation.STAGE_GROUPS for group in stage
)
ablation.STAGE_SIZES = tuple(len(stage) for stage in ablation.STAGE_GROUPS)

runner = ablation.runner
runner.SELF = SELF
runner.STAGE_GROUPS = ablation.STAGE_GROUPS
runner.FROZEN_GROUP_SEQUENCE = ablation.FROZEN_GROUP_SEQUENCE
runner.STAGE_TARGET_REDUCTIONS = ablation.STAGE_TARGET_REDUCTIONS
runner.EXPECTED_PARAMETERS = ablation.EXPECTED_PARAMETERS
runner.EXPECTED_BASELINE_PARAMETERS = ablation.BASELINE_PARAMETERS
runner.EXPECTED_GROUP_COUNT = None
runner.EXPECTED_RAW_FORMULA_IDS = frozenset({ablation.FORMULA_ID})
runner.SCHEMA = SCHEMA
runner.DOMAIN = DOMAIN
runner.STAGE_EPOCHS = ablation.STAGE_EPOCHS
runner.STAGE_WARMUP_EPOCHS = ablation.STAGE_WARMUP_EPOCHS
runner.TOTAL_EPOCHS = sum(ablation.STAGE_EPOCHS)
runner.output_root = lambda domain: RESULT_ROOT
runner.build_runtime = ablation.build_runtime


def main() -> int:
    return ablation.main()


if __name__ == "__main__":
    raise SystemExit(main())
