"""Run T7 recovery for the frozen BDD Isomorphic-Taylor raw architecture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
RAW_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_isomorphic_taylor_raw56_taylor50b_v2"
RAW_HELPER = STUDY_ROOT / "scripts" / "prepare_bdd_isomorphic_taylor_raw56.py"
REFERENCE = STUDY_ROOT / "scripts" / "run_bdd_t7_competitor.py"
RECOVERY_BASE = STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct" / "bdd_competitors"
METHOD = "isomorphic_taylor"
DOMAINS = ("GEN2", "NGN2")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t7 = load_module("_bdd_t7_isomorphic_taylor_reference", REFERENCE)
# ``run_bdd_t7_recovery.orchestrate`` launches each structural replay in a
# subprocess using this module-level path.  Point it back to this wrapper so
# that the worker receives the Isomorphic-Taylor frozen-mask implementation,
# rather than the historical BDD recovery worker.
t7.SELF = SELF
t7.METHODS = (METHOD,)
t7.RAW_HELPER = RAW_HELPER
t7.raw_root = lambda method: RAW_ROOT
t7.result_root = lambda method, domain: RECOVERY_BASE / f"T7_bdd_{domain.lower()}_isomorphic_taylor_taylor50b_70ep_v2"


def load_plan(method: str) -> dict[str, Any]:
    if method != METHOD:
        raise ValueError(method)
    path = RAW_ROOT / "frozen_pruning_plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen Isomorphic-Taylor plan: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "bdd_isomorphic_taylor_raw56_frozen_plan_v2" or plan.get("method") != METHOD:
        raise RuntimeError(f"Unexpected Isomorphic-Taylor plan: {path}")
    return plan


t7.load_plan = load_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--resume", action="store_true")
    action.add_argument("--worker-prune", action="store_true")
    action.add_argument("--smoke-worker", action="store_true")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--group")
    parser.add_argument("--input-model", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    runner, context = t7.build_runtime(METHOD, args.domain)
    if args.preflight:
        print(json.dumps(context["preflight"](False), indent=2, sort_keys=True))
        return 0
    if args.worker_prune:
        if args.candidate_index is None or not args.group or args.output_root is None:
            parser.error("worker mode requires --candidate-index, --group, and --output-root")
        return context["worker_prune"](
            args.candidate_index, args.group, args.input_model, args.output_root
        )
    if args.smoke_worker:
        first = load_plan(METHOD)["entries"][0]
        group_id = str(first["group_id"])
        candidate_index = int(first["candidate_index"])
        smoke_root = context["root"] / "smoke_pruning"
        print(
            json.dumps(
                {
                    "mode": "single_frozen_worker_smoke",
                    "domain": args.domain,
                    "candidate_index": candidate_index,
                    "group_id": group_id,
                    "output_root": str(smoke_root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return context["worker_prune"](candidate_index, group_id, None, smoke_root)
    result = runner.orchestrate(context, resume=args.resume)
    t7.finalize_metadata(METHOD, args.domain)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
