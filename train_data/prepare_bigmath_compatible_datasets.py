from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from datasets import Dataset, DatasetDict, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.reward import extract_gold_answer_text, normalize_answer
from src.tokenization import load_causal_lm_tokenizer
from src.utils import setup_logger


logger = setup_logger("prepare_train_data")

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "deepseek_r1_distill_qwen_1_5b_mainline_rl.jsonnet"
DEFAULT_COMPETITION_INPUT = PROJECT_ROOT / "train_data" / "competition_math"
DEFAULT_NUMINA_INPUT = PROJECT_ROOT / "train_data" / "NuminaMath-1.5"
DEFAULT_COMPETITION_OUTPUT = PROJECT_ROOT / "train_data" / "competition_math_bigmath_format_2k"
DEFAULT_NUMINA_OUTPUT = PROJECT_ROOT / "train_data" / "NuminaMath-1.5-hard-verifiable_2k"
DEFAULT_LOCAL_TOKENIZER = PROJECT_ROOT / "init_model" / "DeepSeek-R1-Distill-Qwen-1.5B"
NUMINA_ALLOWED_SOURCES = (
    "olympiads",
    "olympiads_ref",
    "amc_aime",
    "cn_contest",
    "inequalities",
    "number_theory",
)


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


def _prepare_competition_example(example: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(example)
    normalized["problem"] = str(example.get("problem", "")).strip()
    normalized["solution"] = str(example.get("solution", "")).strip()
    normalized["answer"] = _sanitize_extracted_answer(
        extract_gold_answer_text(normalized["solution"])
    )
    return normalized


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


def _prepare_numina_example(example: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(example)
    for field_name in (
        "problem",
        "solution",
        "answer",
        "problem_type",
        "question_type",
        "problem_is_valid",
        "solution_is_valid",
        "source",
    ):
        normalized[field_name] = str(example.get(field_name, "")).strip()
    return normalized


def _validate_numina_example(
    example: Dict[str, Any],
    *,
    allowed_sources: Sequence[str] = NUMINA_ALLOWED_SOURCES,
) -> str | None:
    answer = str(example.get("answer", "")).strip().lower()
    question_type = str(example.get("question_type", "")).strip()
    source = str(example.get("source", "")).strip()

    if question_type != "math-word-problem":
        return "question_type_mismatch"
    if answer in {"proof", "notfound"}:
        return f"answer_{answer}"
    if source not in set(allowed_sources):
        return "source_not_allowed"
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
    invalid_examples = 0
    invalid_breakdown: Dict[str, int] = {}

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
                invalid_examples += 1
                invalid_breakdown[reason] = invalid_breakdown.get(reason, 0) + 1
                continue
        normalized_records.append(normalized)

    stats = {
        "kept_examples": len(normalized_records),
        "dropped_empty_problem": empty_problem,
        "dropped_empty_answer": empty_answer,
        "dropped_invalid_examples": invalid_examples,
        "invalid_breakdown": invalid_breakdown,
    }
    logger.info(
        "%s normalization kept=%d dropped_empty_problem=%d dropped_empty_answer=%d dropped_invalid_examples=%d invalid_breakdown=%s",
        dataset_name,
        len(normalized_records),
        empty_problem,
        empty_answer,
        invalid_examples,
        invalid_breakdown,
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
        for key in ("problem", "answer", "solution", "level", "type", "source", "question_type"):
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
) -> tuple[Dataset, Dict[str, Any]]:
    def _length_batch(batch: Dict[str, list[Any]]) -> Dict[str, list[int]]:
        batch_size = len(batch[question_field])
        texts: list[str] = []
        for idx in range(batch_size):
            example = {key: batch[key][idx] for key in batch}
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
        desc=f"Computing prompt token lengths for {dataset_name}",
    )
    token_lengths = list(with_lengths["_token_len"])
    keep_indices = [idx for idx, token_len in enumerate(token_lengths) if token_len <= max_prompt_tokens]
    filtered = with_lengths.select(keep_indices).remove_columns(["_token_len"])
    summary = {
        "dataset_name": dataset_name,
        "kept_examples": len(filtered),
        "removed_examples": len(with_lengths) - len(filtered),
        "max_allowed_tokens": max_prompt_tokens,
        "max_seen_tokens": max(token_lengths) if token_lengths else 0,
        "max_kept_tokens": max((token_lengths[idx] for idx in keep_indices), default=0),
        "min_seen_tokens": min(token_lengths) if token_lengths else 0,
    }
    logger.info(
        "%s 2k filtering kept=%d removed=%d max_seen=%d max_kept=%d limit=%d",
        dataset_name,
        summary["kept_examples"],
        summary["removed_examples"],
        summary["max_seen_tokens"],
        summary["max_kept_tokens"],
        max_prompt_tokens,
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


def _load_parquet_dataset(path: Path) -> Dataset:
    parquet_pattern = str(path / "data" / "train-*.parquet")
    return load_dataset("parquet", data_files=parquet_pattern, split="train")


def _maybe_limit(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the datasets used by the current curriculum RL setup: "
            "convert competition_math into a 2k-filtered dataset with extracted answers, "
            "and filter NuminaMath-1.5 into the 2k-filtered hard verifiable subset."
        )
    )
    parser.add_argument("--configs", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--competition-input", type=str, default=str(DEFAULT_COMPETITION_INPUT))
    parser.add_argument("--numina-input", type=str, default=str(DEFAULT_NUMINA_INPUT))
    parser.add_argument("--competition-output", type=str, default=str(DEFAULT_COMPETITION_OUTPUT))
    parser.add_argument("--numina-output", type=str, default=str(DEFAULT_NUMINA_OUTPUT))
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-samples-competition", type=int, default=None)
    parser.add_argument("--max-samples-numina", type=int, default=None)
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

    competition_input = Path(args.competition_input).resolve()
    numina_input = Path(args.numina_input).resolve()
    competition_output = Path(args.competition_output).resolve()
    numina_output = Path(args.numina_output).resolve()

    logger.info("Loading competition_math dataset from %s", competition_input)
    competition_raw = _load_parquet_dataset(competition_input)
    competition_raw = _maybe_limit(competition_raw, args.max_samples_competition)
    logger.info("Loaded competition_math examples=%d", len(competition_raw))

    logger.info("Loading NuminaMath-1.5 dataset from %s", numina_input)
    numina_raw = _load_parquet_dataset(numina_input)
    numina_raw = _maybe_limit(numina_raw, args.max_samples_numina)
    logger.info("Loaded NuminaMath-1.5 examples=%d", len(numina_raw))

    competition_normalized, competition_normalize_stats = _normalize_dataset(
        competition_raw,
        dataset_name="competition_math_bigmath_format",
        map_fn=_prepare_competition_example,
        validate_fn=_validate_competition_example,
    )
    numina_filtered, numina_filter_stats = _normalize_dataset(
        numina_raw,
        dataset_name="NuminaMath-1.5-hard-verifiable",
        map_fn=_prepare_numina_example,
        validate_fn=_validate_numina_example,
    )

    _log_debug_samples(
        competition_normalized,
        dataset_name="competition_math_bigmath_format",
        debug_samples=int(args.debug_samples),
    )
    _log_debug_samples(
        numina_filtered,
        dataset_name="NuminaMath-1.5-hard-verifiable",
        debug_samples=int(args.debug_samples),
    )

    competition_filtered_2k, competition_filter_stats = _compute_lengths_and_filter(
        competition_normalized,
        dataset_name="competition_math_bigmath_format_2k",
        tokenizer=tokenizer,
        question_field=question_field,
        question_template=question_template,
        max_prompt_tokens=int(args.max_prompt_tokens),
    )
    numina_filtered_2k, numina_filter_length_stats = _compute_lengths_and_filter(
        numina_filtered,
        dataset_name="NuminaMath-1.5-hard-verifiable_2k",
        tokenizer=tokenizer,
        question_field=question_field,
        question_template=question_template,
        max_prompt_tokens=int(args.max_prompt_tokens),
    )

    logger.info("Saving 2k-filtered competition_math dataset to %s", competition_output)
    _save_dataset_dict(competition_filtered_2k, competition_output, overwrite=bool(args.overwrite))

    logger.info("Saving 2k-filtered NuminaMath-1.5 subset to %s", numina_output)
    _save_dataset_dict(numina_filtered_2k, numina_output, overwrite=bool(args.overwrite))

    summary_payload = {
        "competition_math": {
            "kept_examples": len(competition_filtered_2k),
            "normalization": competition_normalize_stats,
            "filtering": competition_filter_stats,
            "output": str(competition_output),
        },
        "numina_math": {
            "kept_examples": len(numina_filtered_2k),
            "normalization": numina_filter_stats,
            "filtering": numina_filter_length_stats,
            "allowed_sources": list(NUMINA_ALLOWED_SOURCES),
            "output": str(numina_output),
        },
    }

    _write_summary(competition_output / "prepare_summary.json", summary_payload["competition_math"])
    _write_summary(numina_output / "prepare_summary.json", summary_payload["numina_math"])
    _write_summary(
        PROJECT_ROOT / "train_data" / "prepare_train_data_summary.json",
        summary_payload,
    )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
