from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.reward import extract_gold_answer_text, normalize_answer
from src.tokenization import load_causal_lm_tokenizer
from src.utils import setup_logger


logger = setup_logger("prepare_train_data")

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "deepseek_r1_distill_qwen_1_5b_mainline_rl.jsonnet"
DEFAULT_BIGMATH_INPUT = PROJECT_ROOT / "train_data" / "Big-Math-RL-Verified"
DEFAULT_COMPETITION_INPUT = PROJECT_ROOT / "train_data" / "competition_math"
DEFAULT_BIGMATH_OUTPUT = PROJECT_ROOT / "train_data" / "Big-Math-RL-Verified-2k"
DEFAULT_COMPETITION_OUTPUT = PROJECT_ROOT / "train_data" / "competition_math_bigmath_format_2k"
DEFAULT_LOCAL_TOKENIZER = PROJECT_ROOT / "init_model" / "DeepSeek-R1-Distill-Qwen-1.5B"


def _safe_prompt_format(example: Dict[str, Any], *, question_template: str, question_field: str) -> str:
    class _SafeFormatDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    values = dict(example)
    if question_field in example:
        values.setdefault("problem", example[question_field])
        values.setdefault("query", example[question_field])

    return question_template.format_map(_SafeFormatDict(values))


def _resolve_tokenizer_name_or_path(cfg: Dict[str, Any], explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()

    if DEFAULT_LOCAL_TOKENIZER.exists():
        return str(DEFAULT_LOCAL_TOKENIZER)

    model_cfg = cfg.get("model", {})
    tokenizer_name = model_cfg.get("tokenizer_name_or_path")
    if tokenizer_name is not None and str(tokenizer_name).strip():
        return str(tokenizer_name).strip()

    actor_name = model_cfg.get("actor_name_or_path")
    if actor_name is not None and str(actor_name).strip():
        return str(actor_name).strip()

    raise ValueError("Could not resolve tokenizer_name_or_path.")


def _prepare_bigmath_example(example: Dict[str, Any]) -> Dict[str, Any]:
    domain = example.get("domain")
    if domain is None:
        normalized_domain: list[str] = []
    elif isinstance(domain, list):
        normalized_domain = [str(item) for item in domain]
    else:
        normalized_domain = [str(domain)]

    solve_rate = example.get("llama8b_solve_rate")
    try:
        normalized_solve_rate = float(solve_rate)
    except (TypeError, ValueError):
        normalized_solve_rate = -1.0

    return {
        "problem": str(example.get("problem", "")).strip(),
        "answer": str(example.get("answer", "")).strip(),
        "source": str(example.get("source", "big_math")).strip() or "big_math",
        "domain": normalized_domain,
        "llama8b_solve_rate": normalized_solve_rate,
    }


def _prepare_competition_example(example: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(example)
    normalized["problem"] = str(example.get("problem", "")).strip()
    normalized["solution"] = str(example.get("solution", "")).strip()
    normalized["answer"] = _sanitize_extracted_answer(
        extract_gold_answer_text(normalized["solution"])
    )
    return normalized


def _sanitize_extracted_answer(answer: Any) -> str:
    text = str(answer).strip()
    if not text:
        return ""
    text = text.replace(r"\$", "$")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\$\s+", "$", text)
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    elif text.endswith("$") and text.count("$") == 1:
        text = text[:-1].rstrip()
    return text.strip()


def _looks_like_verbose_explanation(answer: str) -> bool:
    if len(answer) <= 80:
        return False

    matrix_markers = (
        r"\begin{pmatrix}",
        r"\begin{bmatrix}",
        r"\begin{matrix}",
        r"\begin{cases}",
    )
    if any(marker in answer for marker in matrix_markers):
        return False

    natural_text = re.sub(r"\\[A-Za-z]+", " ", answer)
    plain_word_count = len(re.findall(r"[A-Za-z]{2,}", natural_text))
    return plain_word_count >= 8


def _validate_competition_example(example: Dict[str, Any]) -> str | None:
    answer = str(example.get("answer", "")).strip()
    solution = str(example.get("solution", "")).strip()
    lowered = answer.lower()

    if not answer:
        return "empty_answer"
    if normalize_answer(answer) == normalize_answer(solution):
        return "answer_equals_solution"
    if "boxed" in lowered or "final answer" in lowered:
        return "contains_format_marker"
    if _looks_like_verbose_explanation(answer):
        return "verbose_explanation"
    return None


def _normalize_dataset(
    dataset: Dataset,
    *,
    dataset_name: str,
    map_fn,
    validate_fn=None,
) -> tuple[Dataset, Dict[str, Any]]:
    normalized_records = []
    empty_problem = 0
    empty_answer = 0
    invalid_answer = 0
    invalid_answer_breakdown: Dict[str, int] = {}

    for example in dataset:
        normalized = map_fn(example)
        if not normalized["problem"]:
            empty_problem += 1
            continue
        if not normalized["answer"]:
            empty_answer += 1
            continue
        if validate_fn is not None:
            reason = validate_fn(normalized)
            if reason is not None:
                invalid_answer += 1
                invalid_answer_breakdown[reason] = invalid_answer_breakdown.get(reason, 0) + 1
                continue
        normalized_records.append(normalized)

    stats = {
        "kept_examples": len(normalized_records),
        "dropped_empty_problem": empty_problem,
        "dropped_empty_answer": empty_answer,
        "dropped_invalid_answer": invalid_answer,
        "invalid_answer_breakdown": invalid_answer_breakdown,
    }
    logger.info(
        "%s normalization kept=%d dropped_empty_problem=%d dropped_empty_answer=%d dropped_invalid_answer=%d invalid_breakdown=%s",
        dataset_name,
        len(normalized_records),
        empty_problem,
        empty_answer,
        invalid_answer,
        invalid_answer_breakdown,
    )
    return Dataset.from_list(normalized_records), stats


def _truncate_text(value: Any, limit: int = 400) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _log_debug_samples(dataset: Dataset, *, dataset_name: str, debug_samples: int) -> None:
    if debug_samples <= 0 or len(dataset) == 0:
        return

    limit = min(int(debug_samples), len(dataset))
    logger.info("Debug samples for %s: showing %d/%d", dataset_name, limit, len(dataset))
    for idx in range(limit):
        example = dataset[idx]
        preview: Dict[str, Any] = {}
        for key in ("problem", "answer", "solution", "level", "type", "source", "domain"):
            if key in example:
                preview[key] = _truncate_text(example[key])
        logger.info("[%s sample %d] %s", dataset_name, idx, json.dumps(preview, ensure_ascii=False))


def _compute_lengths_and_filter(
    dataset: Dataset,
    *,
    dataset_name: str,
    tokenizer,
    question_field: str,
    question_template: str,
    max_prompt_tokens: int,
    count_target: str,
) -> tuple[Dataset, Dict[str, Any]]:
    def _length_batch(batch: Dict[str, list[Any]]) -> Dict[str, list[int]]:
        batch_size = len(batch[question_field])
        texts: list[str] = []
        for idx in range(batch_size):
            example = {key: batch[key][idx] for key in batch}
            if count_target == "question":
                texts.append(str(example.get(question_field, "")))
            else:
                texts.append(
                    _safe_prompt_format(
                        example,
                        question_template=question_template,
                        question_field=question_field,
                    )
                )

        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        return {"_token_len": [len(ids) for ids in encoded["input_ids"]]}

    with_lengths = dataset.map(
        _length_batch,
        batched=True,
        batch_size=256,
        desc=f"Computing {count_target} token lengths for {dataset_name}",
    )

    token_lengths = list(with_lengths["_token_len"])
    keep_indices = [idx for idx, token_len in enumerate(token_lengths) if token_len <= max_prompt_tokens]
    filtered = with_lengths.select(keep_indices).remove_columns(["_token_len"])

    summary = {
        "dataset_name": dataset_name,
        "count_target": count_target,
        "kept_examples": len(filtered),
        "removed_examples": len(with_lengths) - len(filtered),
        "max_allowed_tokens": max_prompt_tokens,
        "max_seen_tokens": max(token_lengths) if token_lengths else 0,
        "max_kept_tokens": max((token_lengths[idx] for idx in keep_indices), default=0),
        "min_seen_tokens": min(token_lengths) if token_lengths else 0,
    }
    logger.info(
        "%s filtering kept=%d removed=%d max_seen=%d max_kept=%d limit=%d target=%s",
        dataset_name,
        summary["kept_examples"],
        summary["removed_examples"],
        summary["max_seen_tokens"],
        summary["max_kept_tokens"],
        max_prompt_tokens,
        count_target,
    )
    return filtered, summary


def _save_dataset_dict(dataset: Dataset, output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Pass --overwrite to replace it."
            )
        import shutil

        shutil.rmtree(output_dir)

    DatasetDict({"train": dataset}).save_to_disk(str(output_dir))


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_competition_dataset(path: Path) -> Dataset:
    parquet_pattern = str(path / "data" / "train-*.parquet")
    return load_dataset("parquet", data_files=parquet_pattern, split="train")


def _maybe_limit(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add an extracted `answer` column to competition_math and "
            "filter both training datasets by prompt/question token length."
        )
    )
    parser.add_argument("--configs", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--bigmath-input", type=str, default=str(DEFAULT_BIGMATH_INPUT))
    parser.add_argument("--competition-input", type=str, default=str(DEFAULT_COMPETITION_INPUT))
    parser.add_argument("--bigmath-output", type=str, default=str(DEFAULT_BIGMATH_OUTPUT))
    parser.add_argument("--competition-output", type=str, default=str(DEFAULT_COMPETITION_OUTPUT))
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--count-target",
        choices=("prompt", "question"),
        default="prompt",
        help="Filter by rendered training prompt length or raw question/problem length.",
    )
    parser.add_argument("--max-samples-bigmath", type=int, default=None)
    parser.add_argument("--max-samples-competition", type=int, default=None)
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="Print the first N normalized samples for manual inspection.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = load_config(args.configs)
    data_cfg = cfg["data"]
    question_field = str(data_cfg.get("question_field", "problem"))
    question_template = str(data_cfg.get("question_template", "{problem}"))
    trust_remote_code = bool(cfg.get("model", {}).get("trust_remote_code", True))

    tokenizer_name_or_path = _resolve_tokenizer_name_or_path(cfg, args.tokenizer_name_or_path)
    logger.info("Using tokenizer: %s", tokenizer_name_or_path)
    try:
        tokenizer = load_causal_lm_tokenizer(
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tokenizer. If your environment is offline, pass a local "
            "`--tokenizer-name-or-path`, for example the repo's local init_model directory."
        ) from exc

    bigmath_input = Path(args.bigmath_input).resolve()
    competition_input = Path(args.competition_input).resolve()
    bigmath_output = Path(args.bigmath_output).resolve()
    competition_output = Path(args.competition_output).resolve()

    logger.info("Loading Big-Math dataset from %s", bigmath_input)
    bigmath_raw = load_from_disk(str(bigmath_input))["train"]
    bigmath_raw = _maybe_limit(bigmath_raw, args.max_samples_bigmath)
    logger.info("Loaded Big-Math examples=%d", len(bigmath_raw))

    logger.info("Loading competition_math dataset from %s", competition_input)
    competition_raw = _load_competition_dataset(competition_input)
    competition_raw = _maybe_limit(competition_raw, args.max_samples_competition)
    logger.info("Loaded competition_math examples=%d", len(competition_raw))

    bigmath_normalized, bigmath_normalize_stats = _normalize_dataset(
        bigmath_raw,
        dataset_name="Big-Math-RL-Verified",
        map_fn=_prepare_bigmath_example,
    )
    competition_normalized, competition_normalize_stats = _normalize_dataset(
        competition_raw,
        dataset_name="competition_math",
        map_fn=_prepare_competition_example,
        validate_fn=_validate_competition_example,
    )

    _log_debug_samples(
        bigmath_normalized,
        dataset_name="Big-Math-RL-Verified",
        debug_samples=int(args.debug_samples),
    )
    _log_debug_samples(
        competition_normalized,
        dataset_name="competition_math",
        debug_samples=int(args.debug_samples),
    )

    bigmath_filtered, bigmath_summary = _compute_lengths_and_filter(
        bigmath_normalized,
        dataset_name="Big-Math-RL-Verified",
        tokenizer=tokenizer,
        question_field=question_field,
        question_template=question_template,
        max_prompt_tokens=int(args.max_prompt_tokens),
        count_target=args.count_target,
    )
    competition_filtered, competition_summary = _compute_lengths_and_filter(
        competition_normalized,
        dataset_name="competition_math",
        tokenizer=tokenizer,
        question_field=question_field,
        question_template=question_template,
        max_prompt_tokens=int(args.max_prompt_tokens),
        count_target=args.count_target,
    )

    logger.info("Saving filtered Big-Math dataset to %s", bigmath_output)
    _save_dataset_dict(bigmath_filtered, bigmath_output, overwrite=bool(args.overwrite))

    logger.info("Saving converted competition_math dataset to %s", competition_output)
    _save_dataset_dict(competition_filtered, competition_output, overwrite=bool(args.overwrite))

    summary_payload = {
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "count_target": args.count_target,
        "question_field": question_field,
        "question_template_preview": question_template[:300],
        "bigmath": bigmath_summary,
        "competition_math": competition_summary,
        "normalization": {
            "bigmath": bigmath_normalize_stats,
            "competition_math": competition_normalize_stats,
        },
        "outputs": {
            "bigmath_output": str(bigmath_output),
            "competition_output": str(competition_output),
        },
    }

    _write_summary(bigmath_output / "prepare_summary.json", summary_payload["bigmath"])
    _write_summary(competition_output / "prepare_summary.json", summary_payload["competition_math"])
    _write_summary(
        PROJECT_ROOT / "train_data" / "prepare_train_data_summary.json",
        summary_payload,
    )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
