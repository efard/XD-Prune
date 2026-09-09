"""Memory-controlled DepGraph audit for the two frozen YOLO26n baselines.

This script never writes into a baseline run folder and never prunes a canonical
checkpoint.  A parent process monitors one tracing worker at a time and stops it
if its resident memory, elapsed time, or system-free-memory boundary is crossed.

The full-graph worker streams one root group at a time to JSONL.  It deliberately
does not materialize ``list(DG.get_all_groups(...))``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import types
from typing import Any, Iterable

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = PROJECT_ROOT / "Pruning_Study"
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "depgraph" / "week1_safe_v1"
CHECKPOINTS = {
    "GEN": STUDY_ROOT / "results" / "baselines" / "b_gen_mio_yolo26n_s42_v1" / "weights" / "best.pt",
    "SNOW": STUDY_ROOT / "results" / "baselines" / "b_snow_acdc_yolo26n_s42_v1" / "weights" / "best.pt",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def flatten_tensors(value: Any) -> Iterable[Any]:
    import torch

    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_tensors(item)


def trace_detect_forward(self, features):
    """Trace-only Detect forward that preserves both feature dependencies.

    YOLO26 normally detaches the one-to-one feature inputs.  That is correct for
    training, but it hides a structural channel dependency from an Autograd-based
    graph tracer.  This adapter is attached only to the in-memory audit model.
    """

    one2many = self.forward_head(features, **self.one2many)
    one2one = self.forward_head(features, **self.one2one)
    return {"one2many": one2many, "one2one": one2one}


def worker_full(
    domain: str,
    checkpoint: Path,
    output: Path,
    trace_size: int,
    root_start: int,
    root_end: int,
) -> int:
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    domain_dir = output / domain.lower()
    domain_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = domain_dir / "heartbeat.json"
    groups_path = domain_dir / f"groups_{root_start:04d}_{root_end:04d}.jsonl"
    inventory_path = domain_dir / "module_inventory.csv"
    if groups_path.exists():
        groups_path.unlink()

    process = psutil.Process(os.getpid())

    def beat(stage: str, **extra: Any) -> None:
        record = {
            "domain": domain,
            "stage": stage,
            "pid": os.getpid(),
            "rss_bytes": process.memory_info().rss,
            "timestamp": time.time(),
            **extra,
        }
        atomic_json(heartbeat, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    beat("loading_checkpoint", root_start=root_start, root_end=root_end)
    yolo = YOLO(str(checkpoint), task="detect")
    model = yolo.model.cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    head = model.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("Expected the YOLO26 end-to-end Detect head")
    head.forward = types.MethodType(trace_detect_forward, head)

    class TraceWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, images):
            output_value = self.inner(images)
            tensors = tuple(flatten_tensors(output_value))
            if not tensors or not all(tensor.requires_grad for tensor in tensors):
                raise RuntimeError("Trace outputs must retain Autograd dependencies")
            return tensors

    wrapper = TraceWrapper(model)
    named_modules = dict(model.named_modules())
    module_to_path = {id(module): path for path, module in named_modules.items()}
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    beat(
        "model_loaded",
        parameter_count=parameter_count,
        top_level_modules=len(model.model),
        named_modules=len(named_modules),
    )

    with inventory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "module_path",
                "module_type",
                "in_channels",
                "out_channels",
                "groups",
                "root_status",
                "reason",
            ],
        )
        writer.writeheader()
        candidates = []
        for path, module in named_modules.items():
            if not isinstance(module, torch.nn.Conv2d):
                continue
            status = "candidate"
            reason = "eligible Conv2d output-channel root"
            if path.startswith("model.23"):
                status = "protected"
                reason = "Detect-head root protected during the first dependency audit"
            elif ".attn." in path:
                status = "protected"
                reason = "packed attention projection requires head-aware structured pruning"
            elif module.groups != 1:
                status = "protected"
                reason = "grouped/depthwise root requires a dedicated divisibility rule"
            writer.writerow(
                {
                    "module_path": path,
                    "module_type": type(module).__name__,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "groups": module.groups,
                    "root_status": status,
                    "reason": reason,
                }
            )
            if status == "candidate" and module.out_channels > 1:
                candidates.append((path, module))

    selected_candidates = [
        (index, path, module)
        for index, (path, module) in enumerate(candidates, start=1)
        if root_start <= index <= root_end
    ]
    if not selected_candidates:
        raise ValueError(
            f"Requested root range {root_start}-{root_end} does not intersect "
            f"the {len(candidates)} candidates"
        )

    beat(
        "building_dependency_graph",
        candidate_roots=len(candidates),
        batch_roots=len(selected_candidates),
        root_start=root_start,
        root_end=root_end,
        trace_size=trace_size,
    )
    example = torch.zeros(1, 3, trace_size, trace_size, dtype=torch.float32)
    started = time.perf_counter()
    dependency_graph = tp.DependencyGraph().build_dependency(wrapper, example_inputs=example)
    beat("dependency_graph_built", build_seconds=time.perf_counter() - started)

    completed = 0
    failed = 0
    for batch_index, (root_index, root_path, root_module) in enumerate(selected_candidates, start=1):
        probe_indices = sorted({0, root_module.out_channels // 2, root_module.out_channels - 1})
        root_record: dict[str, Any] = {
            "domain": domain,
            "root_index": root_index,
            "root_module_path": root_path,
            "root_module_type": type(root_module).__name__,
            "root_in_channels": root_module.in_channels,
            "root_out_channels": root_module.out_channels,
            "probe_indices": probe_indices,
        }
        try:
            group = dependency_graph.get_pruning_group(
                root_module,
                tp.prune_conv_out_channels,
                idxs=probe_indices,
            )
            operations = []
            for dependency, indices in group:
                target = dependency.target.module
                operations.append(
                    {
                        "target_module_path": module_to_path.get(id(target), "<autograd-operation>"),
                        "target_module_type": type(target).__name__,
                        "handler": getattr(dependency.handler, "__name__", str(dependency.handler)),
                        "indices": [int(index) for index in indices],
                    }
                )
            root_record.update(
                {
                    "status": "generated",
                    "depgraph_check": bool(dependency_graph.check_pruning_group(group)),
                    "operation_count": len(operations),
                    "operations": operations,
                }
            )
            completed += 1
            del group
        except Exception as error:  # keep the audit resumable and evidence-preserving
            root_record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            failed += 1
        append_jsonl(groups_path, root_record)
        if batch_index == 1 or batch_index == len(selected_candidates):
            beat(
                "streaming_groups",
                processed_in_batch=batch_index,
                batch_total=len(selected_candidates),
                root_index=root_index,
                candidate_total=len(candidates),
                generated=completed,
                failed=failed,
            )
            gc.collect()

    beat(
        "complete",
        candidate_roots=len(candidates),
        batch_roots=len(selected_candidates),
        root_start=root_start,
        root_end=root_end,
        generated=completed,
        failed=failed,
    )
    return 0 if failed == 0 else 2


def terminate_tree(process: psutil.Process) -> None:
    descendants = process.children(recursive=True)
    for child in descendants:
        try:
            child.terminate()
        except psutil.Error:
            pass
    try:
        process.terminate()
    except psutil.Error:
        pass
    _, alive = psutil.wait_procs([*descendants, process], timeout=5)
    for item in alive:
        try:
            item.kill()
        except psutil.Error:
            pass


def monitored_worker(
    domain: str,
    checkpoint: Path,
    output: Path,
    trace_size: int,
    max_rss_gb: float,
    min_free_gb: float,
    timeout_seconds: int,
    root_start: int,
    root_end: int,
) -> dict[str, Any]:
    domain_dir = output / domain.lower()
    domain_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = domain_dir / f"worker_{root_start:04d}_{root_end:04d}_stdout.log"
    stderr_path = domain_dir / f"worker_{root_start:04d}_{root_end:04d}_stderr.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-full",
        "--domain",
        domain,
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--trace-size",
        str(trace_size),
        "--root-start",
        str(root_start),
        "--root-end",
        str(root_end),
    ]
    started = time.monotonic()
    peak_rss = 0
    reason = "process_exit"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=PROJECT_ROOT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss = process.memory_info().rss + sum(
                    descendant.memory_info().rss for descendant in process.children(recursive=True)
                )
            except psutil.Error:
                rss = 0
            peak_rss = max(peak_rss, rss)
            available = psutil.virtual_memory().available
            elapsed = time.monotonic() - started
            if rss > max_rss_gb * 1024**3:
                reason = "worker_rss_limit"
                terminate_tree(process)
                break
            if available < min_free_gb * 1024**3:
                reason = "system_free_memory_limit"
                terminate_tree(process)
                break
            if elapsed > timeout_seconds:
                reason = "timeout"
                terminate_tree(process)
                break
            time.sleep(1.0)
        return_code = child.poll()

    heartbeat_path = domain_dir / "heartbeat.json"
    heartbeat = None
    if heartbeat_path.exists():
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    return {
        "domain": domain,
        "root_start": root_start,
        "root_end": root_end,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "return_code": return_code,
        "stop_reason": reason,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "last_heartbeat": heartbeat,
        "stdout_log": str(stdout_path.relative_to(PROJECT_ROOT)),
        "stderr_log": str(stderr_path.relative_to(PROJECT_ROOT)),
    }


def candidate_count(checkpoint: Path) -> int:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task="detect").model.cpu().eval()
    count = sum(
        1
        for path, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d)
        and not path.startswith("model.23")
        and ".attn." not in path
        and module.groups == 1
        and module.out_channels > 1
    )
    del model
    gc.collect()
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_domain_batches(domain: str, output: Path, expected: int) -> list[dict[str, Any]]:
    domain_dir = output / domain.lower()
    records: list[dict[str, Any]] = []
    for path in sorted(domain_dir.glob("groups_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].jsonl")):
        records.extend(read_jsonl(path))
    by_index = {int(record["root_index"]): record for record in records}
    missing = [index for index in range(1, expected + 1) if index not in by_index]
    if missing:
        raise RuntimeError(f"{domain} is missing root records: {missing}")
    merged = [by_index[index] for index in range(1, expected + 1)]
    merged_path = domain_dir / "groups_streamed.jsonl"
    with merged_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in merged:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return merged


def operation_signature(record: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted(
        (
            str(operation["target_module_path"]),
            str(operation["handler"]),
        )
        for operation in record.get("operations", [])
    )


def write_matched_manifest(
    output: Path,
    gen_records: list[dict[str, Any]],
    snow_records: list[dict[str, Any]],
) -> None:
    gen_by_path = {record["root_module_path"]: record for record in gen_records}
    snow_by_path = {record["root_module_path"]: record for record in snow_records}
    all_paths = sorted(set(gen_by_path) | set(snow_by_path))
    path = output / "matched_root_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "group_id",
                "root_module_path",
                "gen_status",
                "snow_status",
                "gen_operation_count",
                "snow_operation_count",
                "structural_signature_match",
                "review_status",
            ],
        )
        writer.writeheader()
        for index, root_path in enumerate(all_paths, start=1):
            gen = gen_by_path.get(root_path, {})
            snow = snow_by_path.get(root_path, {})
            match = bool(gen and snow and operation_signature(gen) == operation_signature(snow))
            accepted = (
                gen.get("status") == "generated"
                and snow.get("status") == "generated"
                and gen.get("depgraph_check") is True
                and snow.get("depgraph_check") is True
                and match
            )
            writer.writerow(
                {
                    "group_id": f"G{index:03d}",
                    "root_module_path": root_path,
                    "gen_status": gen.get("status", "missing"),
                    "snow_status": snow.get("status", "missing"),
                    "gen_operation_count": gen.get("operation_count", ""),
                    "snow_operation_count": snow.get("operation_count", ""),
                    "structural_signature_match": str(match).lower(),
                    "review_status": "automatic_match_pending_physical_validation" if accepted else "manual_review",
                }
            )


def environment_record() -> dict[str, Any]:
    import torch
    import ultralytics

    return {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "torch_pruning": importlib.metadata.version("torch-pruning"),
        "psutil": psutil.__version__,
        "physical_memory_bytes": psutil.virtual_memory().total,
        "checkpoints": {
            domain: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for domain, path in CHECKPOINTS.items()
        },
    }


def orchestrate(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in CHECKPOINTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    atomic_json(output / "environment.json", environment_record())

    results = []
    domain_records: dict[str, list[dict[str, Any]]] = {}
    for domain in ("GEN", "SNOW"):
        total_roots = candidate_count(CHECKPOINTS[domain])
        print(f"Starting monitored {domain} trace in batches; candidate roots={total_roots}", flush=True)
        domain_safe = True
        for root_start in range(1, total_roots + 1, args.batch_roots):
            root_end = min(total_roots, root_start + args.batch_roots - 1)
            result = monitored_worker(
                domain=domain,
                checkpoint=CHECKPOINTS[domain],
                output=output,
                trace_size=args.trace_size,
                max_rss_gb=args.max_rss_gb,
                min_free_gb=args.min_free_gb,
                timeout_seconds=args.timeout_seconds,
                root_start=root_start,
                root_end=root_end,
            )
            results.append(result)
            atomic_json(output / "streaming_summary.json", {"workers": results})
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            safe_exit = result["stop_reason"] == "process_exit" and result["return_code"] in (0, 2)
            if not safe_exit:
                domain_safe = False
                print(
                    f"Stopping at {domain} roots {root_start}-{root_end}; the capped worker "
                    "did not complete safely.",
                    flush=True,
                )
                break
        if not domain_safe:
            break
        domain_records[domain] = merge_domain_batches(domain, output, total_roots)

    all_complete = set(domain_records) == {"GEN", "SNOW"}
    if all_complete:
        write_matched_manifest(output, domain_records["GEN"], domain_records["SNOW"])
        atomic_json(
            output / "audit_summary.json",
            {
                "status": "group_generation_complete_pending_physical_validation",
                "gen_roots": len(domain_records["GEN"]),
                "snow_roots": len(domain_records["SNOW"]),
                "matched_manifest": "matched_root_manifest.csv",
                "baseline_checkpoints_modified": False,
            },
        )
    return 0 if all_complete else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-full", action="store_true")
    parser.add_argument("--domain", choices=["GEN", "SNOW"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-size", type=int, default=32)
    parser.add_argument("--max-rss-gb", type=float, default=2.0)
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--batch-roots", type=int, default=5)
    parser.add_argument("--root-start", type=int, default=1)
    parser.add_argument("--root-end", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.worker_full:
            if args.domain is None or args.checkpoint is None:
                raise ValueError("Worker mode requires --domain and --checkpoint")
            return worker_full(
                args.domain,
                args.checkpoint.resolve(),
                args.output.resolve(),
                args.trace_size,
                args.root_start,
                args.root_end,
            )
        return orchestrate(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
