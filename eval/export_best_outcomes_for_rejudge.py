#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


BEST_OUTCOMES_FILENAME = "best_outcomes.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export each checkpoint's best_outcomes.log together with best_acc "
            "and variance into a separate bundle directory."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory like experiments/manual_eval/all_checkpoints_randomized_5x",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Where to write the export bundle. Defaults to a sibling directory "
            "named <input-root>_rejudge_export."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset result key filter. Repeat the flag to keep multiple datasets.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "relative_checkpoint_dir",
        "experiment_name",
        "checkpoint_name",
        "checkpoint_step",
        "dataset_key",
        "dataset_alias",
        "dataset_split",
        "completed",
        "num_runs",
        "target_num_runs",
        "best_acc",
        "mean_acc",
        "variance",
        "std",
        "best_run_index",
        "best_run_seed",
        "source_summary_path",
        "source_best_outcomes_path",
        "export_metadata_path",
        "export_best_outcomes_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def _maybe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_best_outcomes_path(summary_path: Path, dataset_key: str, dataset_summary: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []

    for raw_path in (
        dataset_summary.get("best_outcomes_copy_path"),
        (dataset_summary.get("best_run") or {}).get("outcomes_log_path"),
    ):
        if not raw_path:
            continue
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = (summary_path.parent / dataset_key / candidate).resolve()
        candidates.append(candidate)

    candidates.append(summary_path.parent / dataset_key / BEST_OUTCOMES_FILENAME)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _build_record(
    *,
    input_root: Path,
    summary_path: Path,
    checkpoint_summary: Dict[str, Any],
    dataset_key: str,
    dataset_summary: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    relative_checkpoint_dir = summary_path.parent.relative_to(input_root)
    export_dir = output_root / relative_checkpoint_dir / dataset_key
    export_log_path = export_dir / BEST_OUTCOMES_FILENAME
    export_metadata_path = export_dir / "metadata.json"

    dataset_spec = dataset_summary.get("dataset") or {}
    best_run = dataset_summary.get("best_run") or {}
    source_best_outcomes_path = _resolve_best_outcomes_path(summary_path, dataset_key, dataset_summary)

    return {
        "record_id": str(relative_checkpoint_dir / dataset_key),
        "relative_checkpoint_dir": str(relative_checkpoint_dir),
        "experiment_name": str(relative_checkpoint_dir.parent),
        "checkpoint_name": checkpoint_summary.get("checkpoint_name"),
        "checkpoint_step": checkpoint_summary.get("checkpoint_step"),
        "checkpoint_type": checkpoint_summary.get("checkpoint_type"),
        "checkpoint_dir": checkpoint_summary.get("checkpoint_dir"),
        "experiment_dir": checkpoint_summary.get("experiment_dir"),
        "dataset_key": dataset_key,
        "dataset_alias": dataset_spec.get("name"),
        "dataset_name": dataset_spec.get("dataset_name"),
        "dataset_config_name": dataset_spec.get("dataset_config_name"),
        "dataset_split": dataset_spec.get("dataset_split"),
        "completed": bool(dataset_summary.get("completed", False)),
        "num_runs": dataset_summary.get("num_runs"),
        "target_num_runs": dataset_summary.get("target_num_runs"),
        "best_acc": _maybe_float(dataset_summary.get("best_acc")),
        "mean_acc": _maybe_float(dataset_summary.get("mean_acc", dataset_summary.get("accuracy_mean"))),
        "variance": _maybe_float(dataset_summary.get("variance", dataset_summary.get("accuracy_variance"))),
        "std": _maybe_float(dataset_summary.get("std", dataset_summary.get("accuracy_std"))),
        "worst_acc": _maybe_float(dataset_summary.get("worst_acc")),
        "best_run_index": best_run.get("run_index"),
        "best_run_seed": best_run.get("seed"),
        "best_run_metrics_path": best_run.get("metrics_path"),
        "best_run_outcomes_log_path": best_run.get("outcomes_log_path"),
        "best_run_per_example_scores_path": best_run.get("per_example_scores_path"),
        "source_summary_path": str(summary_path.parent / dataset_key / "summary.json"),
        "source_checkpoint_summary_path": str(summary_path),
        "source_best_outcomes_path": str(source_best_outcomes_path) if source_best_outcomes_path else None,
        "export_dir": str(export_dir),
        "export_metadata_path": str(export_metadata_path),
        "export_best_outcomes_path": str(export_log_path),
        "export_best_outcomes_exists": False,
    }


def export_bundle(input_root: Path, output_root: Path, selected_datasets: Sequence[str]) -> Dict[str, Any]:
    selected_dataset_set = set(selected_datasets)
    checkpoint_summary_paths = sorted(input_root.rglob("checkpoint_summary.json"))

    index_records: List[Dict[str, Any]] = []
    missing_records: List[Dict[str, Any]] = []
    exported_logs = 0

    for summary_path in checkpoint_summary_paths:
        checkpoint_summary = _load_json(summary_path)
        datasets = checkpoint_summary.get("datasets") or {}
        if not isinstance(datasets, dict):
            continue

        for dataset_key in sorted(datasets):
            if selected_dataset_set and dataset_key not in selected_dataset_set:
                continue

            dataset_summary = datasets.get(dataset_key) or {}
            if not isinstance(dataset_summary, dict):
                continue

            record = _build_record(
                input_root=input_root,
                summary_path=summary_path,
                checkpoint_summary=checkpoint_summary,
                dataset_key=dataset_key,
                dataset_summary=dataset_summary,
                output_root=output_root,
            )

            export_dir = Path(record["export_dir"])
            export_dir.mkdir(parents=True, exist_ok=True)

            source_best_outcomes = record.get("source_best_outcomes_path")
            if source_best_outcomes:
                shutil.copy2(source_best_outcomes, record["export_best_outcomes_path"])
                record["export_best_outcomes_exists"] = True
                exported_logs += 1
            else:
                missing_records.append(
                    {
                        "record_id": record["record_id"],
                        "source_summary_path": record["source_summary_path"],
                        "reason": "best_outcomes.log not found",
                    }
                )

            _write_json(Path(record["export_metadata_path"]), record)
            index_records.append(record)

    _write_jsonl(output_root / "index.jsonl", index_records)
    _write_csv(output_root / "index.csv", index_records)
    if missing_records:
        _write_jsonl(output_root / "missing_best_outcomes.jsonl", missing_records)

    summary = {
        "source_root": str(input_root),
        "output_root": str(output_root),
        "created_at": time.time(),
        "selected_datasets": sorted(selected_dataset_set),
        "num_checkpoint_summaries": len(checkpoint_summary_paths),
        "num_dataset_entries": len(index_records),
        "num_exported_best_outcomes_logs": exported_logs,
        "num_missing_best_outcomes_logs": len(missing_records),
        "index_jsonl_path": str(output_root / "index.jsonl"),
        "index_csv_path": str(output_root / "index.csv"),
    }
    if missing_records:
        summary["missing_best_outcomes_path"] = str(output_root / "missing_best_outcomes.jsonl")
    _write_json(output_root / "export_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if args.output_root is None:
        output_root = input_root.parent / f"{input_root.name}_rejudge_export"
    else:
        output_root = args.output_root.expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"Input root does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 1

    summary = export_bundle(input_root, output_root, args.dataset)
    print(
        "Exported "
        f"{summary['num_dataset_entries']} dataset entries "
        f"from {summary['num_checkpoint_summaries']} checkpoint summaries."
    )
    print(f"Output directory: {summary['output_root']}")
    print(f"Index CSV: {summary['index_csv_path']}")
    print(f"Exported logs: {summary['num_exported_best_outcomes_logs']}")
    print(f"Missing logs: {summary['num_missing_best_outcomes_logs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
