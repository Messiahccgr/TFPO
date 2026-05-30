#!/usr/bin/env python3
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _resolve_project_root()


def _dataset_result_key(
    dataset_name: str,
    dataset_split: str,
    dataset_config_name: Optional[str] = None,
) -> str:
    raw_parts = [Path(str(dataset_name)).name or str(dataset_name)]
    if dataset_config_name:
        raw_parts.append(str(dataset_config_name))
    raw_parts.append(str(dataset_split))
    raw = "_".join(part for part in raw_parts if str(part).strip())
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower() or "eval"


def _safe_relative_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


@dataclass(frozen=True)
class DatasetProgress:
    dataset_key: str
    completed_runs: int
    target_runs: int

    @property
    def done(self) -> bool:
        return self.completed_runs >= self.target_runs


@dataclass(frozen=True)
class CheckpointProgress:
    relative_label: str
    completed_runs: int
    total_runs: int
    completed_datasets: int
    total_datasets: int
    state: str
    next_dataset_key: Optional[str]
    next_run_index: Optional[int]
    error: Optional[str]


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print resume progress for eval_all_checkpoints.py from an existing output root."
        )
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(
            (
                PROJECT_ROOT
                / "experiments"
                / "manual_eval"
                / "all_checkpoints_randomized_5x"
            ).resolve()
        ),
        help="Evaluation output root that contains manifest.json and per-checkpoint results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many remaining checkpoints to print. Use 0 to suppress the list.",
    )
    parser.add_argument(
        "--show-completed",
        action="store_true",
        help="Also print completed checkpoints.",
    )
    return parser


def _planned_dataset_keys(manifest: Dict[str, Any]) -> List[str]:
    dataset_keys: List[str] = []
    for spec in manifest.get("dataset_specs", []):
        dataset_keys.append(
            _dataset_result_key(
                dataset_name=str(spec["dataset_name"]),
                dataset_split=str(spec["dataset_split"]),
                dataset_config_name=spec.get("dataset_config_name"),
            )
        )
    if len(dataset_keys) == 0:
        raise RuntimeError("manifest.json has no dataset_specs; cannot infer dataset output keys.")
    return dataset_keys


def _checkpoint_output_dir(
    *,
    output_root: Path,
    experiments_root: Path,
    checkpoint_info: Dict[str, Any],
) -> Path:
    experiment_dir = Path(str(checkpoint_info["experiment_dir"]))
    checkpoint_name = str(checkpoint_info["checkpoint_name"])
    relative_experiment_dir = _safe_relative_path(experiment_dir, experiments_root)
    return output_root / relative_experiment_dir / checkpoint_name


def _summarize_checkpoint(
    *,
    checkpoint_info: Dict[str, Any],
    output_root: Path,
    experiments_root: Path,
    dataset_keys: List[str],
    repeat_count: int,
) -> CheckpointProgress:
    checkpoint_output_dir = _checkpoint_output_dir(
        output_root=output_root,
        experiments_root=experiments_root,
        checkpoint_info=checkpoint_info,
    )
    relative_label = str(_safe_relative_path(checkpoint_output_dir, output_root))
    error_payload = _load_json(checkpoint_output_dir / "error.json")

    dataset_progresses: List[DatasetProgress] = []
    any_started = False
    for dataset_key in dataset_keys:
        runs_path = checkpoint_output_dir / dataset_key / "runs.jsonl"
        completed_runs = _count_jsonl_records(runs_path)
        if completed_runs > 0:
            any_started = True
        dataset_progresses.append(
            DatasetProgress(
                dataset_key=dataset_key,
                completed_runs=completed_runs,
                target_runs=repeat_count,
            )
        )

    completed_datasets = sum(1 for item in dataset_progresses if item.done)
    completed_runs = sum(min(item.completed_runs, repeat_count) for item in dataset_progresses)
    total_runs = len(dataset_progresses) * repeat_count

    next_dataset_key = None
    next_run_index = None
    for item in dataset_progresses:
        if item.completed_runs < repeat_count:
            next_dataset_key = item.dataset_key
            next_run_index = item.completed_runs + 1
            break

    if completed_datasets == len(dataset_progresses):
        state = "completed"
    elif any_started or checkpoint_output_dir.exists() or error_payload is not None:
        state = "partial"
    else:
        state = "pending"

    error_text = None
    if isinstance(error_payload, dict):
        error_text = str(error_payload.get("error") or "").strip() or "unknown error"

    return CheckpointProgress(
        relative_label=relative_label,
        completed_runs=completed_runs,
        total_runs=total_runs,
        completed_datasets=completed_datasets,
        total_datasets=len(dataset_progresses),
        state=state,
        next_dataset_key=next_dataset_key,
        next_run_index=next_run_index,
        error=error_text,
    )


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{(100.0 * numerator / denominator):.1f}%"


def _print_checkpoint_list(
    *,
    title: str,
    checkpoints: List[CheckpointProgress],
    limit: int,
) -> None:
    print(title)
    if len(checkpoints) == 0:
        print("  none")
        return

    display = checkpoints if limit <= 0 else checkpoints[:limit]
    for item in display:
        line = (
            f"  [{item.state}] {item.relative_label} "
            f"datasets={item.completed_datasets}/{item.total_datasets} "
            f"runs={item.completed_runs}/{item.total_runs}"
        )
        if item.next_dataset_key is not None and item.next_run_index is not None:
            line += f" next={item.next_dataset_key} run={item.next_run_index}"
        if item.error:
            line += f" error={item.error}"
        print(line)

    if limit > 0 and len(checkpoints) > limit:
        print(f"  ... {len(checkpoints) - limit} more")


def main() -> None:
    args = _build_argument_parser().parse_args()

    output_root = Path(args.output_root).resolve()
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"manifest.json not found under output root: {output_root}")

    experiments_root = Path(str(manifest["experiments_root"])).resolve()
    repeat_count = int(manifest["repeat_count"])
    dataset_keys = _planned_dataset_keys(manifest)
    checkpoints_info = list(manifest.get("checkpoints", []))
    if len(checkpoints_info) == 0:
        raise RuntimeError("manifest.json has no checkpoints.")

    checkpoint_progresses = [
        _summarize_checkpoint(
            checkpoint_info=checkpoint_info,
            output_root=output_root,
            experiments_root=experiments_root,
            dataset_keys=dataset_keys,
            repeat_count=repeat_count,
        )
        for checkpoint_info in checkpoints_info
    ]

    completed = [item for item in checkpoint_progresses if item.state == "completed"]
    partial = [item for item in checkpoint_progresses if item.state == "partial"]
    pending = [item for item in checkpoint_progresses if item.state == "pending"]
    remaining = [item for item in checkpoint_progresses if item.state != "completed"]

    completed_runs = sum(item.completed_runs for item in checkpoint_progresses)
    total_runs = sum(item.total_runs for item in checkpoint_progresses)
    next_item = remaining[0] if remaining else None

    print(f"output_root: {output_root}")
    print(f"manifest: {manifest_path}")
    print(f"experiments_root: {experiments_root}")
    print(f"datasets: {', '.join(dataset_keys)}")
    print(f"repeat_count: {repeat_count}")
    print(
        "checkpoints: "
        f"{len(completed)}/{len(checkpoint_progresses)} completed, "
        f"{len(partial)} partial, "
        f"{len(pending)} pending"
    )
    print(f"runs: {completed_runs}/{total_runs} completed ({_format_pct(completed_runs, total_runs)})")

    if next_item is None:
        print("next_resume_point: none")
    else:
        next_line = f"next_resume_point: {next_item.relative_label}"
        if next_item.next_dataset_key is not None and next_item.next_run_index is not None:
            next_line += f" -> {next_item.next_dataset_key} run {next_item.next_run_index}/{repeat_count}"
        print(next_line)

    if args.show_completed:
        _print_checkpoint_list(
            title="completed_checkpoints:",
            checkpoints=completed,
            limit=args.limit,
        )

    if args.limit != 0:
        _print_checkpoint_list(
            title="remaining_checkpoints:",
            checkpoints=remaining,
            limit=args.limit,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
