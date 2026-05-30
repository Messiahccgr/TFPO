#!/usr/bin/env python3

import argparse
import concurrent.futures
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _bootstrap_python_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


PROJECT_ROOT = _bootstrap_python_path()

from eval.larger_model import BASE_URL, MODEL_NAME, chat


REJUDGE_RESULTS_FILENAME = "rejudge_wrong_cases.jsonl"
REJUDGE_RESULTS_READABLE_FILENAME = "rejudge_wrong_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the stronger model API to rejudge originally-wrong samples from selected run "
            "and compute corrected accuracy for each checkpoint."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "manual_eval" / "all_checkpoints_randomized_5x",
        help="Manual-eval root containing checkpoint_summary.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "manual_eval"
        / "all_checkpoints_randomized_5x_larger_model_rejudge",
        help="Where to write rejudge artifacts and summaries.",
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
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset key filter, for example math_500_test. Repeat to keep multiple.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Optional checkpoint name filter, for example iter_0119_actor. Repeat to keep multiple.",
    )
    parser.add_argument(
        "--relative-checkpoint-path-contains",
        action="append",
        default=[],
        help=(
            "Optional relative checkpoint path substring filter (case-insensitive). "
            "Repeat to keep multiple; any match is accepted."
        ),
    )
    parser.add_argument(
        "--exclude-relative-checkpoint-path-contains",
        action="append",
        default=[],
        help=(
            "Optional relative checkpoint path exclusion substring filter (case-insensitive). "
            "Repeat to keep multiple; any match is excluded."
        ),
    )
    parser.add_argument(
        "--max-cases-per-dataset",
        type=int,
        default=None,
        help="Debug option: only rejudge up to N originally-wrong samples per dataset.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Rewrite per-dataset cache every N new rejudge results.",
    )
    parser.add_argument(
        "--source-run",
        choices=("best", "worst"),
        default="best",
        help="Which run's per-example file to rejudge.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    tmp.replace(path)


def _write_csv(path: Path, fieldnames: Sequence[str], records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})
    tmp.replace(path)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    candidates = [text]

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced_match:
        candidates.append(fenced_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse JSON from response: {text[:500]}")


def _build_prompt_intro(dataset_key: str) -> str:
    return f"""You are verifying whether benchmark answers should count as correct.

Judge each candidate response against the question and gold answer.
Count it as correct if the candidate response clearly gives an answer equivalent to the gold answer anywhere in the response, even if formatting differs or answer extraction failed.
Do not count it as correct if the response is incomplete, ambiguous, contradictory, or gives a different final answer.
If there are multiple answers, prefer the final explicit answer. If there is no explicit final answer, only mark correct when the response still unambiguously concludes the correct answer.
The `reason` field in your JSON response must be concise Simplified Chinese.

Dataset key: {dataset_key}
"""


def _build_single_prompt(
    *,
    dataset_key: str,
    question: str,
    gold_answer_raw: str,
    gold_answer_extracted: Optional[str],
    candidate_response: str,
    candidate_answer_only: Optional[str],
) -> str:
    gold_answer_extracted = gold_answer_extracted or ""
    candidate_answer_only = candidate_answer_only or ""
    return f"""{_build_prompt_intro(dataset_key)}

Return exactly one JSON object on a single line with this schema:
{{"is_correct": true, "final_answer": "short string or null", "reason": "short explanation in Simplified Chinese"}}

The `reason` value must be written in concise Simplified Chinese.

Question:
{question}

Gold answer raw:
{gold_answer_raw}

Gold answer extracted:
{gold_answer_extracted}

Candidate extracted final answer:
{candidate_answer_only}

Candidate response:
{candidate_response}
"""


def _build_batch_prompt(dataset_key: str, records: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for record in records:
        pass1 = record["pass1"]
        blocks.append(
            "\n".join(
                [
                    f"Sample index: {record['index']}",
                    "Question:",
                    str(record["question"]),
                    "",
                    "Gold answer raw:",
                    str(record.get("gold_answer_raw", "")),
                    "",
                    "Gold answer extracted:",
                    str(record.get("gold_answer_extracted") or ""),
                    "",
                    "Candidate extracted final answer:",
                    str(pass1.get("answer_only_text") or ""),
                    "",
                    "Candidate response:",
                    str(pass1.get("text", "")),
                ]
            )
        )

    joined_blocks = "\n\n====================\n\n".join(blocks)
    return f"""{_build_prompt_intro(dataset_key)}

Return exactly one JSON object on a single line with this schema:
{{"results": [{{"index": 123, "is_correct": true, "final_answer": "short string or null", "reason": "short explanation in Simplified Chinese"}}]}}

Every `reason` value must be written in concise Simplified Chinese.

You must return exactly one result item for every sample index below, and each index must appear exactly once.

{joined_blocks}
"""


def _judge_once(
    *,
    dataset_key: str,
    question: str,
    gold_answer_raw: str,
    gold_answer_extracted: Optional[str],
    candidate_response: str,
    candidate_answer_only: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    prompt = _build_single_prompt(
        dataset_key=dataset_key,
        question=question,
        gold_answer_raw=gold_answer_raw,
        gold_answer_extracted=gold_answer_extracted,
        candidate_response=candidate_response,
        candidate_answer_only=candidate_answer_only,
    )
    raw_response = chat(prompt)
    parsed = _extract_json_object(raw_response)
    if "is_correct" not in parsed:
        raise ValueError(f"Response JSON missing is_correct: {raw_response[:500]}")
    parsed["is_correct"] = bool(parsed["is_correct"])
    final_answer = parsed.get("final_answer")
    if final_answer is not None and not isinstance(final_answer, str):
        parsed["final_answer"] = str(final_answer)
    reason = parsed.get("reason")
    if reason is None:
        parsed["reason"] = ""
    elif not isinstance(reason, str):
        parsed["reason"] = str(reason)
    return raw_response, parsed


def _judge_batch_once(dataset_key: str, records: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    prompt = _build_batch_prompt(dataset_key, records)
    raw_response = chat(prompt)
    parsed = _extract_json_object(raw_response)
    raw_results = parsed.get("results")
    if not isinstance(raw_results, list):
        raise ValueError(f"Batch response missing results list: {raw_response[:500]}")

    parsed_by_index: Dict[int, Dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError(f"Batch response item is not an object: {raw_response[:500]}")
        if "index" not in item or "is_correct" not in item:
            raise ValueError(f"Batch response item missing required fields: {raw_response[:500]}")
        index = int(item["index"])
        parsed_by_index[index] = {
            "is_correct": bool(item["is_correct"]),
            "final_answer": None if item.get("final_answer") is None else str(item.get("final_answer")),
            "reason": "" if item.get("reason") is None else str(item.get("reason")),
        }

    expected_indices = {int(record["index"]) for record in records}
    if set(parsed_by_index) != expected_indices:
        raise ValueError(
            f"Batch response index mismatch. expected={sorted(expected_indices)} got={sorted(parsed_by_index)}"
        )

    return raw_response, parsed_by_index


def _build_result_payload(
    *,
    record: Dict[str, Any],
    status: str,
    judge_is_correct: bool,
    judge_final_answer: Optional[str],
    judge_reason: str,
    attempts: int,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    pass1 = record["pass1"]
    payload: Dict[str, Any] = {
        "index": int(record["index"]),
        "status": status,
        "question": str(record.get("question", "")),
        "gold": {
            "answer_raw": record.get("gold_answer_raw"),
            "answer_extracted": record.get("gold_answer_extracted"),
        },
        "candidate": {
            "finish_reason": pass1.get("finish_reason"),
            "answer_extracted": pass1.get("answer_only_text"),
            "response": str(pass1.get("text", "")),
        },
        "judge": {
            "is_correct": bool(judge_is_correct),
            "final_answer": judge_final_answer,
            "reason": judge_reason,
            "attempts": int(attempts),
        },
    }
    if error is not None:
        payload["judge"]["error"] = error
    return payload


def judge_case(dataset_key: str, record: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    pass1 = record["pass1"]
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            _, parsed = _judge_once(
                dataset_key=dataset_key,
                question=str(record["question"]),
                gold_answer_raw=str(record.get("gold_answer_raw", "")),
                gold_answer_extracted=record.get("gold_answer_extracted"),
                candidate_response=str(pass1.get("text", "")),
                candidate_answer_only=pass1.get("answer_only_text"),
            )
            return _build_result_payload(
                record=record,
                status="ok",
                judge_is_correct=bool(parsed["is_correct"]),
                judge_final_answer=parsed.get("final_answer"),
                judge_reason=parsed.get("reason", ""),
                attempts=attempt,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 * attempt, 8))

    return _build_result_payload(
        record=record,
        status="error",
        judge_is_correct=False,
        judge_final_answer=None,
        judge_reason="",
        attempts=max_retries,
        error=repr(last_error),
    )


def judge_batch(
    dataset_key: str,
    records: Sequence[Dict[str, Any]],
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    if not records:
        return []

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            _, parsed_by_index = _judge_batch_once(dataset_key, records)
            return [
                _build_result_payload(
                    record=record,
                    status="ok",
                    judge_is_correct=bool(parsed_by_index[int(record["index"])]["is_correct"]),
                    judge_final_answer=parsed_by_index[int(record["index"])]["final_answer"],
                    judge_reason=parsed_by_index[int(record["index"])]["reason"],
                    attempts=attempt,
                )
                for record in records
            ]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 * attempt, 8))

    if len(records) == 1:
        return [judge_case(dataset_key, records[0], max_retries=max_retries)]

    midpoint = len(records) // 2
    left = judge_batch(dataset_key, records[:midpoint], max_retries=max_retries)
    right = judge_batch(dataset_key, records[midpoint:], max_retries=max_retries)
    if left or right:
        return left + right

    return [
        _build_result_payload(
            record=record,
            status="error",
            judge_is_correct=False,
            judge_final_answer=None,
            judge_reason="",
            attempts=max_retries,
            error=repr(last_error),
        )
        for record in records
    ]


def _chunked(records: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if batch_size <= 1:
        return [[record] for record in records]
    return [list(records[i : i + batch_size]) for i in range(0, len(records), batch_size)]


def _locate_per_example_path(
    summary_path: Path,
    dataset_key: str,
    dataset_summary: Dict[str, Any],
    source_run: str,
) -> Optional[Path]:
    if source_run == "worst":
        candidates = [
            dataset_summary.get("worst_per_example_copy_path"),
            (dataset_summary.get("worst_run") or {}).get("per_example_scores_path"),
            str(summary_path.parent / dataset_key / "worst_per_example_scores.jsonl"),
        ]
    else:
        candidates = [
            dataset_summary.get("best_per_example_copy_path"),
            (dataset_summary.get("best_run") or {}).get("per_example_scores_path"),
            str(summary_path.parent / dataset_key / "best_per_example_scores.jsonl"),
        ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw))
        if path.exists() and path.is_file():
            return path
    return None


def _relative_checkpoint_dir(input_root: Path, summary_path: Path) -> Path:
    return summary_path.parent.relative_to(input_root)


def _normalize_filter_tokens(values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for raw in values:
        token = str(raw).strip().lower()
        if token:
            tokens.append(token)
    return tokens


def _matches_relative_checkpoint_dir(
    relative_checkpoint_dir: Path,
    *,
    include_tokens: Sequence[str],
    exclude_tokens: Sequence[str],
) -> bool:
    normalized_relative = str(relative_checkpoint_dir).replace("\\", "/").lower()
    if include_tokens and not any(token in normalized_relative for token in include_tokens):
        return False
    if exclude_tokens and any(token in normalized_relative for token in exclude_tokens):
        return False
    return True


def _normalize_existing_result(record: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(record.get("judge"), dict):
        normalized: Dict[str, Any] = {
            "index": int(record["index"]),
            "status": str(record.get("status", "error")),
            "question": str(record.get("question", "")),
            "gold": {
                "answer_raw": (record.get("gold") or {}).get("answer_raw"),
                "answer_extracted": (record.get("gold") or {}).get("answer_extracted"),
            },
            "candidate": {
                "finish_reason": (record.get("candidate") or {}).get("finish_reason"),
                "answer_extracted": (record.get("candidate") or {}).get("answer_extracted"),
                "response": str((record.get("candidate") or {}).get("response", "")),
            },
            "judge": {
                "is_correct": bool((record.get("judge") or {}).get("is_correct", False)),
                "final_answer": (record.get("judge") or {}).get("final_answer"),
                "reason": "" if (record.get("judge") or {}).get("reason") is None else str((record.get("judge") or {}).get("reason")),
                "attempts": int((record.get("judge") or {}).get("attempts", 0)),
            },
        }
        error = (record.get("judge") or {}).get("error")
        if error is not None:
            normalized["judge"]["error"] = str(error)
        return normalized

    normalized = {
        "index": int(record["index"]),
        "status": str(record.get("status", "error")),
        "question": str(record.get("question", "")),
        "gold": {
            "answer_raw": record.get("gold_answer_raw"),
            "answer_extracted": record.get("gold_answer_extracted"),
        },
        "candidate": {
            "finish_reason": record.get("candidate_finish_reason"),
            "answer_extracted": record.get("candidate_answer_only"),
            "response": str(record.get("candidate_response", "")),
        },
        "judge": {
            "is_correct": bool(record.get("judge_is_correct", False)),
            "final_answer": record.get("judge_final_answer"),
            "reason": "" if record.get("judge_reason") is None else str(record.get("judge_reason")),
            "attempts": int(record.get("attempts", 0)),
        },
    }
    error = record.get("error")
    if error is not None:
        normalized["judge"]["error"] = str(error)
    return normalized


def _attach_source_context(result: Dict[str, Any], source_record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_existing_result(result)
    pass1 = source_record["pass1"]

    if not normalized.get("question"):
        normalized["question"] = str(source_record.get("question", ""))

    gold = normalized.setdefault("gold", {})
    if gold.get("answer_raw") is None:
        gold["answer_raw"] = source_record.get("gold_answer_raw")
    if gold.get("answer_extracted") is None:
        gold["answer_extracted"] = source_record.get("gold_answer_extracted")

    candidate = normalized.setdefault("candidate", {})
    if candidate.get("finish_reason") is None:
        candidate["finish_reason"] = pass1.get("finish_reason")
    if candidate.get("answer_extracted") is None:
        candidate["answer_extracted"] = pass1.get("answer_only_text")
    if not candidate.get("response"):
        candidate["response"] = str(pass1.get("text", ""))

    judge = normalized.setdefault("judge", {})
    judge["is_correct"] = bool(judge.get("is_correct", False))
    judge["reason"] = "" if judge.get("reason") is None else str(judge.get("reason"))
    judge["attempts"] = int(judge.get("attempts", 0))
    return normalized


def _load_existing_results(path: Path) -> Dict[int, Dict[str, Any]]:
    existing: Dict[int, Dict[str, Any]] = {}
    for record in _load_jsonl(path):
        normalized = _normalize_existing_result(record)
        existing[int(normalized["index"])] = normalized
    return existing


def _sorted_result_records(records_by_index: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [records_by_index[index] for index in sorted(records_by_index)]


def _write_rejudge_results(dataset_output_dir: Path, records_by_index: Dict[int, Dict[str, Any]]) -> None:
    sorted_records = _sorted_result_records(records_by_index)
    _write_jsonl(dataset_output_dir / REJUDGE_RESULTS_FILENAME, sorted_records)
    _write_json(dataset_output_dir / REJUDGE_RESULTS_READABLE_FILENAME, sorted_records)


def _process_dataset(
    *,
    input_root: Path,
    output_root: Path,
    summary_path: Path,
    checkpoint_summary: Dict[str, Any],
    dataset_key: str,
    dataset_summary: Dict[str, Any],
    workers: int,
    batch_size: int,
    max_cases_per_dataset: Optional[int],
    save_every: int,
    source_run: str,
) -> Dict[str, Any]:
    source_path = _locate_per_example_path(summary_path, dataset_key, dataset_summary, source_run)
    if source_path is None:
        raise FileNotFoundError(
            f"Could not find {source_run}_per_example_scores for {summary_path} / {dataset_key}"
        )

    relative_checkpoint_dir = _relative_checkpoint_dir(input_root, summary_path)
    dataset_output_dir = output_root / relative_checkpoint_dir / dataset_key
    rejudge_results_path = dataset_output_dir / REJUDGE_RESULTS_FILENAME
    rejudge_results_readable_path = dataset_output_dir / REJUDGE_RESULTS_READABLE_FILENAME
    rejudge_summary_path = dataset_output_dir / "rejudge_summary.json"

    all_records = _load_jsonl(source_path)
    original_correct = 0
    wrong_records: List[Dict[str, Any]] = []
    for record in all_records:
        if bool(record["pass1"]["is_correct"]):
            original_correct += 1
        else:
            wrong_records.append(record)

    existing_results = _load_existing_results(rejudge_results_path)
    for record in wrong_records:
        index = int(record["index"])
        if index in existing_results:
            existing_results[index] = _attach_source_context(existing_results[index], record)
    pending_records = [
        record
        for record in wrong_records
        if int(record["index"]) not in existing_results or existing_results[int(record["index"])].get("status") != "ok"
    ]
    if max_cases_per_dataset is not None:
        pending_records = pending_records[: max(0, max_cases_per_dataset)]

    print(
        f"[dataset] {relative_checkpoint_dir}/{dataset_key} "
        f"wrong={len(wrong_records)} cached_ok={sum(1 for value in existing_results.values() if value.get('status') == 'ok')} "
        f"pending={len(pending_records)}",
        flush=True,
    )

    completed_since_save = 0
    if pending_records:
        pending_batches = _chunked(pending_records, max(1, batch_size))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {
                executor.submit(judge_batch, dataset_key, batch): tuple(int(record["index"]) for record in batch)
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
                    _write_rejudge_results(dataset_output_dir, existing_results)
                    completed_since_save = 0

                if finished == total or finished % max(1, min(25, total)) == 0:
                    print(
                        f"[progress] {relative_checkpoint_dir}/{dataset_key} "
                        f"{finished}/{total} batch requests finished",
                        flush=True,
                    )
    else:
        _write_rejudge_results(dataset_output_dir, existing_results)

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

    rejudge_summary = {
        "checkpoint_name": checkpoint_summary.get("checkpoint_name"),
        "checkpoint_step": checkpoint_summary.get("checkpoint_step"),
        "dataset_key": dataset_key,
        "dataset_alias": (dataset_summary.get("dataset") or {}).get("name"),
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
        rejudge_summary["source_best_per_example_path"] = str(source_path)
    elif source_run == "worst":
        rejudge_summary["source_worst_per_example_path"] = str(source_path)
    _write_json(rejudge_summary_path, rejudge_summary)
    return rejudge_summary


def _checkpoint_summary_record(
    *,
    input_root: Path,
    output_root: Path,
    summary_path: Path,
    checkpoint_summary: Dict[str, Any],
    dataset_rejudge_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    dataset_keys = sorted(dataset_rejudge_summaries)
    macro_original_acc = (
        sum(dataset_rejudge_summaries[key]["original_acc"] for key in dataset_keys) / len(dataset_keys)
        if dataset_keys
        else None
    )
    macro_accurate_acc = (
        sum(dataset_rejudge_summaries[key]["accurate_acc"] for key in dataset_keys) / len(dataset_keys)
        if dataset_keys
        else None
    )
    total_examples = sum(dataset_rejudge_summaries[key]["num_examples"] for key in dataset_keys)
    total_original_correct = sum(dataset_rejudge_summaries[key]["original_num_correct"] for key in dataset_keys)
    total_recovered_correct = sum(dataset_rejudge_summaries[key]["recovered_correct"] for key in dataset_keys)
    total_accurate_correct = total_original_correct + total_recovered_correct
    micro_original_acc = total_original_correct / total_examples if total_examples else None
    micro_accurate_acc = total_accurate_correct / total_examples if total_examples else None

    relative_checkpoint_dir = _relative_checkpoint_dir(input_root, summary_path)
    checkpoint_output_dir = output_root / relative_checkpoint_dir
    checkpoint_output_path = checkpoint_output_dir / "checkpoint_accurate_summary.json"

    record: Dict[str, Any] = {
        "experiment_name": str(relative_checkpoint_dir.parent),
        "relative_checkpoint_dir": str(relative_checkpoint_dir),
        "checkpoint_name": checkpoint_summary.get("checkpoint_name"),
        "checkpoint_step": checkpoint_summary.get("checkpoint_step"),
        "checkpoint_type": checkpoint_summary.get("checkpoint_type"),
        "num_datasets": len(dataset_keys),
        "macro_original_acc": macro_original_acc,
        "macro_accurate_acc": macro_accurate_acc,
        "macro_acc_delta": None if macro_original_acc is None or macro_accurate_acc is None else macro_accurate_acc - macro_original_acc,
        "micro_original_acc": micro_original_acc,
        "micro_accurate_acc": micro_accurate_acc,
        "micro_acc_delta": None if micro_original_acc is None or micro_accurate_acc is None else micro_accurate_acc - micro_original_acc,
        "total_examples": total_examples,
        "total_original_correct": total_original_correct,
        "total_recovered_correct": total_recovered_correct,
        "total_accurate_correct": total_accurate_correct,
        "datasets": dataset_rejudge_summaries,
        "summary_path": str(checkpoint_output_path),
        "updated_at": time.time(),
    }

    for dataset_key, dataset_summary in dataset_rejudge_summaries.items():
        record[f"{dataset_key}_original_acc"] = dataset_summary["original_acc"]
        record[f"{dataset_key}_accurate_acc"] = dataset_summary["accurate_acc"]
        record[f"{dataset_key}_acc_delta"] = dataset_summary["acc_delta"]
        record[f"{dataset_key}_recovered_correct"] = dataset_summary["recovered_correct"]

    _write_json(checkpoint_output_path, record)
    return record


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    selected_datasets = set(args.dataset)
    selected_checkpoints = set(args.checkpoint)
    include_relative_checkpoint_tokens = _normalize_filter_tokens(
        args.relative_checkpoint_path_contains
    )
    exclude_relative_checkpoint_tokens = _normalize_filter_tokens(
        args.exclude_relative_checkpoint_path_contains
    )

    if not input_root.exists() or not input_root.is_dir():
        print(f"Input root does not exist: {input_root}", file=sys.stderr)
        return 1

    checkpoint_summary_paths = sorted(input_root.rglob("checkpoint_summary.json"))
    checkpoint_records: List[Dict[str, Any]] = []
    dataset_records: List[Dict[str, Any]] = []

    print(
        f"Using stronger model {MODEL_NAME} via {BASE_URL}; "
        f"found {len(checkpoint_summary_paths)} checkpoint summaries.",
        flush=True,
    )

    for summary_path in checkpoint_summary_paths:
        checkpoint_summary = _load_json(summary_path)
        checkpoint_name = str(checkpoint_summary.get("checkpoint_name"))
        relative_checkpoint_dir = _relative_checkpoint_dir(input_root, summary_path)
        if not _matches_relative_checkpoint_dir(
            relative_checkpoint_dir,
            include_tokens=include_relative_checkpoint_tokens,
            exclude_tokens=exclude_relative_checkpoint_tokens,
        ):
            continue
        if selected_checkpoints and checkpoint_name not in selected_checkpoints:
            continue

        dataset_summaries: Dict[str, Dict[str, Any]] = {}
        for dataset_key, dataset_summary in sorted((checkpoint_summary.get("datasets") or {}).items()):
            if selected_datasets and dataset_key not in selected_datasets:
                continue
            if not isinstance(dataset_summary, dict):
                continue

            dataset_rejudge_summary = _process_dataset(
                input_root=input_root,
                output_root=output_root,
                summary_path=summary_path,
                checkpoint_summary=checkpoint_summary,
                dataset_key=dataset_key,
                dataset_summary=dataset_summary,
                workers=args.workers,
                batch_size=args.batch_size,
                max_cases_per_dataset=args.max_cases_per_dataset,
                save_every=args.save_every,
                source_run=args.source_run,
            )
            dataset_summaries[dataset_key] = dataset_rejudge_summary
            dataset_records.append(
                {
                    "experiment_name": str(relative_checkpoint_dir.parent),
                    "relative_checkpoint_dir": str(relative_checkpoint_dir),
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_step": checkpoint_summary.get("checkpoint_step"),
                    **dataset_rejudge_summary,
                }
            )

        if dataset_summaries:
            checkpoint_records.append(
                _checkpoint_summary_record(
                    input_root=input_root,
                    output_root=output_root,
                    summary_path=summary_path,
                    checkpoint_summary=checkpoint_summary,
                    dataset_rejudge_summaries=dataset_summaries,
                )
            )

    dataset_csv_fields = [
        "experiment_name",
        "relative_checkpoint_dir",
        "checkpoint_name",
        "checkpoint_step",
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
        "source_per_example_path",
        "rejudge_results_path",
        "rejudge_results_readable_path",
        "source_best_per_example_path",
        "source_worst_per_example_path",
    ]
    checkpoint_csv_fields = [
        "experiment_name",
        "relative_checkpoint_dir",
        "checkpoint_name",
        "checkpoint_step",
        "checkpoint_type",
        "num_datasets",
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

    extra_checkpoint_fields = sorted(
        {
            key
            for record in checkpoint_records
            for key in record
            if key not in checkpoint_csv_fields and key not in {"datasets", "updated_at"}
        }
    )
    checkpoint_csv_fields = checkpoint_csv_fields + extra_checkpoint_fields

    _write_json(output_root / "run_summary.json", {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_name": MODEL_NAME,
        "model_base_url": BASE_URL,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "source_run": args.source_run,
        "selected_datasets": sorted(selected_datasets),
        "selected_checkpoints": sorted(selected_checkpoints),
        "include_relative_checkpoint_path_contains": include_relative_checkpoint_tokens,
        "exclude_relative_checkpoint_path_contains": exclude_relative_checkpoint_tokens,
        "num_checkpoint_records": len(checkpoint_records),
        "num_dataset_records": len(dataset_records),
        "updated_at": time.time(),
    })
    _write_jsonl(output_root / "dataset_accurate_acc_summary.jsonl", dataset_records)
    _write_jsonl(output_root / "checkpoint_accurate_acc_summary.jsonl", checkpoint_records)
    _write_csv(output_root / "dataset_accurate_acc_summary.csv", dataset_csv_fields, dataset_records)
    _write_csv(output_root / "checkpoint_accurate_acc_summary.csv", checkpoint_csv_fields, checkpoint_records)

    print(
        f"Finished rejudge pipeline for {len(checkpoint_records)} checkpoints / {len(dataset_records)} datasets. "
        f"Checkpoint summary CSV: {output_root / 'checkpoint_accurate_acc_summary.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
