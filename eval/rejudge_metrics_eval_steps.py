#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _bootstrap_python_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


PROJECT_ROOT = _bootstrap_python_path()

from eval.larger_model import BASE_URL, MODEL_NAME
from eval import rejudge_ckpt_accuracy_with_larger_model as legacy_rejudge


STEP_SUMMARY_PATTERN = re.compile(r"^eval_step_(\d+)_summary\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rejudge eval_step best/worst per-example files produced under a metrics-style "
            "evaluation directory."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing eval_step_*_summary.json files, typically a metrics directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where rejudge artifacts and summaries are written.",
    )
    parser.add_argument(
        "--source-run",
        choices=("best", "worst"),
        default="worst",
        help="Which eval_step per-example file to rejudge.",
    )
    parser.add_argument(
        "--step",
        action="append",
        default=[],
        help="Optional step filter. Repeatable and comma-separated values are both supported.",
    )
    parser.add_argument(
        "--dataset-key",
        default=None,
        help="Optional manual dataset key override when auto-inference is unavailable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent stronger-model requests per dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of originally-wrong samples to judge in one stronger-model request.",
    )
    parser.add_argument(
        "--max-cases-per-dataset",
        type=int,
        default=None,
        help="Debug option: only rejudge up to N originally-wrong samples per step.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Rewrite per-step cache every N completed batch requests.",
    )
    return parser.parse_args()


def _parse_step_filters(raw_items: Sequence[str]) -> Set[int]:
    selected: Set[int] = set()
    for raw in raw_items:
        for part in str(raw).split(","):
            value = part.strip()
            if not value:
                continue
            try:
                selected.add(int(value))
            except ValueError as exc:
                raise ValueError(f"Invalid --step value: {value}") from exc
    return selected


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got: {value}")


def _validate_non_negative_optional_int(name: str, value: Optional[int]) -> None:
    if value is None:
        return
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got: {value}")


def _normalize_path_parts(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    normalized = str(raw).replace("\\", "/").rstrip("/")
    if not normalized:
        return []
    return [part for part in normalized.split("/") if part]


def _basename(raw: Optional[str]) -> Optional[str]:
    parts = _normalize_path_parts(raw)
    return parts[-1] if parts else None


def _parent_basename(raw: Optional[str]) -> Optional[str]:
    parts = _normalize_path_parts(raw)
    return parts[-2] if len(parts) >= 2 else None


def _resolve_scan_root(input_root: Path) -> Path:
    candidates = [input_root]
    metrics_dir = input_root / "metrics"
    if metrics_dir != input_root:
        candidates.append(metrics_dir)

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if any(candidate.glob("eval_step_*_summary.json")):
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find any eval_step_*_summary.json under {input_root} or {metrics_dir}"
    )


def _discover_step_summaries(scan_root: Path) -> List[Tuple[int, Path]]:
    discovered: List[Tuple[int, Path]] = []
    for path in scan_root.glob("eval_step_*_summary.json"):
        match = STEP_SUMMARY_PATTERN.match(path.name)
        if match is None:
            continue
        discovered.append((int(match.group(1)), path.resolve()))
    discovered.sort(key=lambda item: item[0])
    return discovered


def _extract_dataset_metadata(summary: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("best_run", "worst_run", "latest_run"):
        run = summary.get(key)
        if not isinstance(run, dict):
            continue
        metrics = run.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    return {}


def _infer_dataset_key(summary: Dict[str, Any], dataset_key_override: Optional[str]) -> Tuple[str, Optional[str]]:
    if dataset_key_override:
        return str(dataset_key_override), None

    dataset_metadata = _extract_dataset_metadata(summary)
    dataset_name = str(dataset_metadata.get("dataset_name") or "")
    dataset_split = str(dataset_metadata.get("dataset_split") or "").strip().lower()
    dataset_basename = _basename(dataset_name)

    if dataset_basename == "MATH-500" and dataset_split == "test":
        return "math_500_test", "math500"

    raise ValueError(
        "Could not infer dataset key from eval_step summary. "
        f"dataset_name={dataset_name or '<missing>'}, dataset_split={dataset_split or '<missing>'}. "
        "Pass --dataset-key explicitly."
    )


def _resolve_source_per_example_path(
    summary_path: Path,
    summary: Dict[str, Any],
    step: int,
    source_run: str,
) -> Optional[Path]:
    root_key = f"{source_run}_per_example_scores_path"
    sibling_name = f"eval_step_{step:06d}_{source_run}_per_example_scores.jsonl"
    candidates: List[Path] = []

    raw = summary.get(root_key)
    if raw:
        candidates.append(Path(str(raw)))
    candidates.append(summary_path.parent / sibling_name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _chunked(records: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if batch_size <= 1:
        return [[record] for record in records]
    return [list(records[i : i + batch_size]) for i in range(0, len(records), batch_size)]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _process_step(
    *,
    output_root: Path,
    summary_path: Path,
    summary: Dict[str, Any],
    step: int,
    dataset_key_override: Optional[str],
    workers: int,
    batch_size: int,
    max_cases_per_dataset: Optional[int],
    save_every: int,
    source_run: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source_path = _resolve_source_per_example_path(summary_path, summary, step, source_run)
    if source_path is None:
        raise FileNotFoundError(
            f"Could not find eval_step_{step:06d}_{source_run}_per_example_scores.jsonl for {summary_path}"
        )

    dataset_key, dataset_alias = _infer_dataset_key(summary, dataset_key_override)
    step_dir_name = f"eval_step_{step:06d}"
    step_output_dir = output_root / step_dir_name
    dataset_output_dir = step_output_dir / dataset_key
    rejudge_results_path = dataset_output_dir / legacy_rejudge.REJUDGE_RESULTS_FILENAME
    rejudge_results_readable_path = dataset_output_dir / legacy_rejudge.REJUDGE_RESULTS_READABLE_FILENAME
    rejudge_summary_path = dataset_output_dir / "rejudge_summary.json"

    checkpoint_dir = str(summary.get("checkpoint_dir") or "")
    checkpoint_name = _basename(checkpoint_dir) or f"checkpoint-{step}"
    experiment_name = _parent_basename(checkpoint_dir) or _parent_basename(str(summary_path.parent)) or "unknown_experiment"

    all_records = _load_jsonl(source_path)
    original_correct = 0
    wrong_records: List[Dict[str, Any]] = []
    for record in all_records:
        if bool((record.get("pass1") or {}).get("is_correct", False)):
            original_correct += 1
        else:
            wrong_records.append(record)

    existing_results = legacy_rejudge._load_existing_results(rejudge_results_path)
    for record in wrong_records:
        index = int(record["index"])
        if index in existing_results:
            existing_results[index] = legacy_rejudge._attach_source_context(existing_results[index], record)

    pending_records = [
        record
        for record in wrong_records
        if int(record["index"]) not in existing_results or existing_results[int(record["index"])].get("status") != "ok"
    ]
    if max_cases_per_dataset is not None:
        pending_records = pending_records[: max_cases_per_dataset]

    print(
        f"[step] {step_dir_name}/{dataset_key} "
        f"wrong={len(wrong_records)} cached_ok={sum(1 for value in existing_results.values() if value.get('status') == 'ok')} "
        f"pending={len(pending_records)}",
        flush=True,
    )

    completed_since_save = 0
    if pending_records:
        pending_batches = _chunked(pending_records, max(1, batch_size))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {
                executor.submit(legacy_rejudge.judge_batch, dataset_key, batch): tuple(int(record["index"]) for record in batch)
                for batch in pending_batches
            }
            finished = 0
            total = len(future_map)
            for future in concurrent.futures.as_completed(future_map):
                batch_results = future.result()
                for result in batch_results:
                    existing_results[int(result["index"])] = result
                finished += 1
                completed_since_save += 1

                if finished == total or completed_since_save >= save_every:
                    legacy_rejudge._write_rejudge_results(dataset_output_dir, existing_results)
                    completed_since_save = 0

                if finished == total or finished % max(1, min(25, total)) == 0:
                    print(
                        f"[progress] {step_dir_name}/{dataset_key} {finished}/{total} batch requests finished",
                        flush=True,
                    )
    else:
        legacy_rejudge._write_rejudge_results(dataset_output_dir, existing_results)

    rejudged_ok = 0
    rejudged_errors = 0
    recovered_correct = 0
    for record in wrong_records:
        result = existing_results.get(int(record["index"]))
        if result is None:
            continue
        if result.get("status") == "ok":
            rejudged_ok += 1
            if bool((result.get("judge") or {}).get("is_correct", False)):
                recovered_correct += 1
        else:
            rejudged_errors += 1

    num_examples = len(all_records)
    original_acc = original_correct / num_examples if num_examples else 0.0
    accurate_num_correct = original_correct + recovered_correct
    accurate_acc = accurate_num_correct / num_examples if num_examples else 0.0
    unresolved_wrong = len(wrong_records) - rejudged_ok

    dataset_summary: Dict[str, Any] = {
        "step": int(step),
        "step_dir_name": step_dir_name,
        "experiment_name": experiment_name,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_name": checkpoint_name,
        "dataset_key": dataset_key,
        "dataset_alias": dataset_alias,
        "source_summary_path": str(summary_path),
        "source_run": source_run,
        "source_per_example_path": str(source_path),
        "rejudge_results_path": str(rejudge_results_path),
        "rejudge_results_readable_path": str(rejudge_results_readable_path),
        "num_examples": num_examples,
        "original_num_correct": original_correct,
        "original_num_wrong": len(wrong_records),
        "original_acc": original_acc,
        "rejudged_wrong_ok": rejudged_ok,
        "rejudged_wrong_errors": rejudged_errors,
        "recovered_correct": recovered_correct,
        "accurate_num_correct": accurate_num_correct,
        "accurate_acc": accurate_acc,
        "acc_delta": accurate_acc - original_acc,
        "unresolved_wrong": unresolved_wrong,
        "model_name": MODEL_NAME,
        "model_base_url": BASE_URL,
        "updated_at": time.time(),
    }
    if source_run == "best":
        dataset_summary["source_best_per_example_path"] = str(source_path)
    elif source_run == "worst":
        dataset_summary["source_worst_per_example_path"] = str(source_path)
    legacy_rejudge._write_json(rejudge_summary_path, dataset_summary)

    step_summary_path = step_output_dir / "step_accurate_summary.json"
    step_summary: Dict[str, Any] = {
        "step": int(step),
        "step_dir_name": step_dir_name,
        "experiment_name": experiment_name,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_name": checkpoint_name,
        "num_datasets": 1,
        "source_run": source_run,
        "macro_original_acc": original_acc,
        "macro_accurate_acc": accurate_acc,
        "macro_acc_delta": accurate_acc - original_acc,
        "micro_original_acc": original_acc,
        "micro_accurate_acc": accurate_acc,
        "micro_acc_delta": accurate_acc - original_acc,
        "total_examples": num_examples,
        "total_original_correct": original_correct,
        "total_recovered_correct": recovered_correct,
        "total_accurate_correct": accurate_num_correct,
        "datasets": {dataset_key: dataset_summary},
        "summary_path": str(step_summary_path),
        "updated_at": time.time(),
    }
    step_summary[f"{dataset_key}_original_acc"] = original_acc
    step_summary[f"{dataset_key}_accurate_acc"] = accurate_acc
    step_summary[f"{dataset_key}_acc_delta"] = accurate_acc - original_acc
    step_summary[f"{dataset_key}_recovered_correct"] = recovered_correct
    legacy_rejudge._write_json(step_summary_path, step_summary)

    return step_summary, dataset_summary


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"Input root does not exist: {input_root}", file=sys.stderr)
        return 1

    _validate_positive_int("workers", args.workers)
    _validate_positive_int("batch_size", args.batch_size)
    _validate_positive_int("save_every", args.save_every)
    _validate_non_negative_optional_int("max_cases_per_dataset", args.max_cases_per_dataset)

    try:
        selected_steps = _parse_step_filters(args.step)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        scan_root = _resolve_scan_root(input_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    discovered_steps = _discover_step_summaries(scan_root)
    if selected_steps:
        discovered_steps = [item for item in discovered_steps if item[0] in selected_steps]

    print(
        f"Using stronger model {MODEL_NAME} via {BASE_URL}; "
        f"scan_root={scan_root}; found {len(discovered_steps)} step summaries.",
        flush=True,
    )

    step_records: List[Dict[str, Any]] = []
    dataset_records: List[Dict[str, Any]] = []
    skipped_steps: List[Dict[str, Any]] = []

    for step, summary_path in discovered_steps:
        summary = _load_json(summary_path)
        try:
            step_summary, dataset_summary = _process_step(
                output_root=output_root,
                summary_path=summary_path,
                summary=summary,
                step=step,
                dataset_key_override=args.dataset_key,
                workers=args.workers,
                batch_size=args.batch_size,
                max_cases_per_dataset=args.max_cases_per_dataset,
                save_every=args.save_every,
                source_run=args.source_run,
            )
        except FileNotFoundError as exc:
            skipped = {
                "step": int(step),
                "step_dir_name": f"eval_step_{step:06d}",
                "source_summary_path": str(summary_path),
                "reason": "missing_source_per_example_scores",
                "error": str(exc),
                "checkpoint_dir": str(summary.get("checkpoint_dir") or ""),
                "checkpoint_name": _basename(str(summary.get("checkpoint_dir") or "")),
            }
            skipped_steps.append(skipped)
            print(f"[skip] eval_step_{step:06d} {exc}", flush=True)
            continue

        step_records.append(step_summary)
        dataset_records.append(dataset_summary)

    dataset_csv_fields = [
        "experiment_name",
        "checkpoint_dir",
        "checkpoint_name",
        "step",
        "step_dir_name",
        "dataset_key",
        "dataset_alias",
        "num_examples",
        "original_num_correct",
        "original_num_wrong",
        "original_acc",
        "rejudged_wrong_ok",
        "rejudged_wrong_errors",
        "recovered_correct",
        "accurate_num_correct",
        "accurate_acc",
        "acc_delta",
        "unresolved_wrong",
        "source_run",
        "source_summary_path",
        "source_per_example_path",
        "rejudge_results_path",
        "rejudge_results_readable_path",
        "source_best_per_example_path",
        "source_worst_per_example_path",
    ]
    step_csv_fields = [
        "experiment_name",
        "checkpoint_dir",
        "checkpoint_name",
        "step",
        "step_dir_name",
        "num_datasets",
        "source_run",
        "macro_original_acc",
        "macro_accurate_acc",
        "macro_acc_delta",
        "micro_original_acc",
        "micro_accurate_acc",
        "micro_acc_delta",
        "total_examples",
        "total_original_correct",
        "total_recovered_correct",
        "total_accurate_correct",
        "summary_path",
    ]
    extra_step_fields = sorted(
        {
            key
            for record in step_records
            for key in record
            if key not in step_csv_fields and key not in {"datasets", "updated_at"}
        }
    )
    step_csv_fields = step_csv_fields + extra_step_fields

    run_summary = {
        "input_root": str(input_root),
        "scan_root": str(scan_root),
        "output_root": str(output_root),
        "model_name": MODEL_NAME,
        "model_base_url": BASE_URL,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "source_run": args.source_run,
        "dataset_key_override": args.dataset_key,
        "selected_steps": sorted(selected_steps),
        "num_discovered_steps": len(discovered_steps),
        "num_processed_steps": len(step_records),
        "num_dataset_records": len(dataset_records),
        "num_skipped_steps": len(skipped_steps),
        "skipped_steps": skipped_steps,
        "updated_at": time.time(),
    }
    legacy_rejudge._write_json(output_root / "run_summary.json", run_summary)
    legacy_rejudge._write_jsonl(output_root / "dataset_accurate_acc_summary.jsonl", dataset_records)
    legacy_rejudge._write_jsonl(output_root / "step_accurate_acc_summary.jsonl", step_records)
    legacy_rejudge._write_csv(output_root / "dataset_accurate_acc_summary.csv", dataset_csv_fields, dataset_records)
    legacy_rejudge._write_csv(output_root / "step_accurate_acc_summary.csv", step_csv_fields, step_records)
    if skipped_steps:
        legacy_rejudge._write_jsonl(output_root / "skipped_steps.jsonl", skipped_steps)

    print(
        f"Finished rejudge pipeline for {len(step_records)} steps / {len(dataset_records)} datasets. "
        f"Step summary CSV: {output_root / 'step_accurate_acc_summary.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
