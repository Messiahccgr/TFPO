#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _bootstrap_python_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


PROJECT_ROOT = _bootstrap_python_path()

from eval.eval_model import (
    _checkpoint_has_weights,
    _dataset_result_key,
    _format_prompt,
    _infer_supported_context_len,
    _load_eval_dataset,
    _tokenize_text_lengths,
)
from src.grpo_runner import _add_pass_1_sampling_metrics, _resolve_pass_1_generation_cfg
from src.reward import MathRewardFn, extract_gold_answer_text, extract_pred_answer
from src.tokenization import load_causal_lm_tokenizer
from src.utils import ensure_dir, save_json_atomic, setup_logger
from src.vllm import VLLMClient, VLLMServer


logger = setup_logger("eval_all_checkpoints")

ITER_ACTOR_PATTERN = re.compile(r"iter_(\d+)_actor$")
DIRECT_CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
BEST_OUTCOMES_FILENAME = "best_outcomes.log"
WORST_OUTCOMES_FILENAME = "worst_outcomes.log"
BEST_PER_EXAMPLE_FILENAME = "best_per_example_scores.jsonl"
WORST_PER_EXAMPLE_FILENAME = "worst_per_example_scores.jsonl"
DEFAULT_DPSK_QUESTION_TEMPLATE = (
    "<｜begin▁of▁sentence｜><｜User｜>"
    "Solve the following math problem efficiently and clearly. Think step by step before "
    "answering. Put the final answer at the very end of your response using exactly this "
    "format and : Therefore, the final answer is: $\\boxed{{your\\ answer}}$. I hope it is "
    "correct.\n\n"
    "{problem}<｜Assistant｜><think>\n"
)
DEFAULT_DPSK_STOP = ["<｜User｜>"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_name: str
    dataset_config_name: Optional[str]
    dataset_split: str
    question_field: str
    answer_field: str
    question_template: str
    max_tokens: int
    stop: Optional[List[str]]


@dataclass(frozen=True)
class PreparedDataset:
    spec: DatasetSpec
    result_key: str
    batch: List[Dict[str, Any]]
    prompts: List[str]


@dataclass(frozen=True)
class CheckpointRef:
    experiment_dir: Path
    checkpoint_dir: Path
    checkpoint_name: str
    checkpoint_type: str
    checkpoint_step: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_dir": str(self.experiment_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_type": self.checkpoint_type,
            "checkpoint_step": self.checkpoint_step,
        }


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        if path.stat().st_size == 0:
            logger.warning("Skipping empty JSON file: %s", path)
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable JSON file %s: %s", path, exc)
        return None


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _safe_relative_path(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def _population_variance(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = float(sum(values) / len(values))
    return float(sum((value - mean_value) ** 2 for value in values) / len(values))


def _numeric_metric_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metric_keys = sorted(
        {
            key
            for record in records
            for key, value in record.get("metrics", {}).items()
            if isinstance(value, (int, float))
        }
    )
    summary: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = [
            float(record["metrics"][key])
            for record in records
            if isinstance(record.get("metrics", {}).get(key), (int, float))
        ]
        if len(values) == 0:
            continue
        variance = _population_variance(values)
        summary[key] = {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / len(values)),
            "variance": float(variance),
            "std": float(math.sqrt(variance)),
            "latest": float(values[-1]),
        }
    return summary


def _extract_accuracy_stats(metrics_summary: Dict[str, Dict[str, float]]) -> Dict[str, Optional[float]]:
    accuracy_summary = metrics_summary.get("accuracy", {})
    return {
        "best_acc": accuracy_summary.get("max"),
        "worst_acc": accuracy_summary.get("min"),
        "mean_acc": accuracy_summary.get("mean"),
        "variance": accuracy_summary.get("variance"),
        "std": accuracy_summary.get("std"),
    }


def _summarize_dataset_runs(
    *,
    records: List[Dict[str, Any]],
    checkpoint_ref: CheckpointRef,
    prepared_dataset: PreparedDataset,
    repeat_count: int,
) -> Dict[str, Any]:
    metrics_summary = _numeric_metric_summary(records)
    ranking_key = "accuracy"
    best_run = None
    worst_run = None
    if len(records) > 0:
        best_run = max(
            records,
            key=lambda item: float(item.get("metrics", {}).get(ranking_key, float("-inf"))),
        )
        worst_run = min(
            records,
            key=lambda item: float(item.get("metrics", {}).get(ranking_key, float("inf"))),
        )

    accuracy_stats = _extract_accuracy_stats(metrics_summary)
    return {
        "checkpoint_dir": str(checkpoint_ref.checkpoint_dir),
        "checkpoint_name": checkpoint_ref.checkpoint_name,
        "checkpoint_type": checkpoint_ref.checkpoint_type,
        "checkpoint_step": checkpoint_ref.checkpoint_step,
        "dataset": asdict(prepared_dataset.spec),
        "dataset_result_key": prepared_dataset.result_key,
        "num_runs": int(len(records)),
        "target_num_runs": int(repeat_count),
        "completed": bool(len(records) >= repeat_count),
        "metrics_summary": metrics_summary,
        "best_acc": accuracy_stats["best_acc"],
        "worst_acc": accuracy_stats["worst_acc"],
        "mean_acc": accuracy_stats["mean_acc"],
        "variance": accuracy_stats["variance"],
        "std": accuracy_stats["std"],
        "accuracy_mean": accuracy_stats["mean_acc"],
        "accuracy_variance": accuracy_stats["variance"],
        "accuracy_std": accuracy_stats["std"],
        "best_run": best_run,
        "worst_run": worst_run,
        "latest_run": records[-1] if records else None,
    }


def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _materialize_extrema_artifacts(dataset_output_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    best_run = summary.get("best_run")
    worst_run = summary.get("worst_run")

    best_outcomes_copy = None
    best_per_example_copy = None
    worst_outcomes_copy = None
    worst_per_example_copy = None

    if isinstance(best_run, dict):
        best_outcomes_path = Path(str(best_run.get("outcomes_log_path", "")))
        best_per_example_path = Path(str(best_run.get("per_example_scores_path", "")))
        best_outcomes_copy = _copy_if_exists(
            best_outcomes_path,
            dataset_output_dir / BEST_OUTCOMES_FILENAME,
        )
        best_per_example_copy = _copy_if_exists(
            best_per_example_path,
            dataset_output_dir / BEST_PER_EXAMPLE_FILENAME,
        )

    if isinstance(worst_run, dict):
        worst_outcomes_path = Path(str(worst_run.get("outcomes_log_path", "")))
        worst_per_example_path = Path(str(worst_run.get("per_example_scores_path", "")))
        worst_outcomes_copy = _copy_if_exists(
            worst_outcomes_path,
            dataset_output_dir / WORST_OUTCOMES_FILENAME,
        )
        worst_per_example_copy = _copy_if_exists(
            worst_per_example_path,
            dataset_output_dir / WORST_PER_EXAMPLE_FILENAME,
        )

    summary["best_outcomes_copy_path"] = best_outcomes_copy
    summary["best_per_example_copy_path"] = best_per_example_copy
    summary["worst_outcomes_copy_path"] = worst_outcomes_copy
    summary["worst_per_example_copy_path"] = worst_per_example_copy
    return summary


def _default_dataset_registry(project_root: Path, *, max_tokens: int, stop: Optional[List[str]]) -> Dict[str, DatasetSpec]:
    eval_root = project_root / "eval_data"
    common_template = DEFAULT_DPSK_QUESTION_TEMPLATE
    return {
        "math500": DatasetSpec(
            name="math500",
            dataset_name=str((eval_root / "MATH-500").resolve()),
            dataset_config_name=None,
            dataset_split="test",
            question_field="problem",
            answer_field="answer",
            question_template=common_template,
            max_tokens=max_tokens,
            stop=stop,
        ),
        "gsm8k": DatasetSpec(
            name="gsm8k",
            dataset_name=str((eval_root / "GSM8K").resolve()),
            dataset_config_name="socratic",
            dataset_split="test",
            question_field="question",
            answer_field="answer",
            question_template=common_template,
            max_tokens=max_tokens,
            stop=stop,
        ),
        "aime24": DatasetSpec(
            name="aime24",
            dataset_name=str((eval_root / "AIME24").resolve()),
            dataset_config_name=None,
            dataset_split="train",
            question_field="problem",
            answer_field="answer",
            question_template=common_template,
            max_tokens=max_tokens,
            stop=stop,
        ),
    }


def _resolve_dataset_specs(
    *,
    project_root: Path,
    dataset_names_csv: str,
    max_tokens: int,
    stop: Optional[List[str]],
) -> List[DatasetSpec]:
    registry = _default_dataset_registry(project_root, max_tokens=max_tokens, stop=stop)
    names = [name.strip().lower() for name in dataset_names_csv.split(",") if name.strip()]
    if len(names) == 0:
        raise ValueError("`--datasets` resolved to an empty list.")
    specs: List[DatasetSpec] = []
    unknown: List[str] = []
    for name in names:
        spec = registry.get(name)
        if spec is None:
            unknown.append(name)
            continue
        specs.append(spec)
    if unknown:
        raise ValueError(
            f"Unknown dataset aliases: {unknown}. Available: {sorted(registry.keys())}"
        )
    return specs


def _prepare_datasets(
    dataset_specs: Sequence[DatasetSpec],
    *,
    max_samples: Optional[int],
) -> List[PreparedDataset]:
    prepared: List[PreparedDataset] = []
    for spec in dataset_specs:
        logger.info(
            "Loading dataset %s from %s (config=%s split=%s)",
            spec.name,
            spec.dataset_name,
            spec.dataset_config_name,
            spec.dataset_split,
        )
        eval_dataset = _load_eval_dataset(
            dataset_name=spec.dataset_name,
            dataset_split=spec.dataset_split,
            dataset_config_name=spec.dataset_config_name,
        )
        if max_samples is not None and max_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(max_samples))
        batch = [eval_dataset[i] for i in range(len(eval_dataset))]
        prompts = [
            _format_prompt(
                example=sample,
                question_template=spec.question_template,
                question_field=spec.question_field,
            )
            for sample in batch
        ]
        result_key = _dataset_result_key(
            spec.dataset_name,
            spec.dataset_split,
            spec.dataset_config_name,
        )
        prepared.append(
            PreparedDataset(
                spec=spec,
                result_key=result_key,
                batch=batch,
                prompts=prompts,
            )
        )
        logger.info(
            "Prepared dataset %s result_key=%s size=%d",
            spec.name,
            result_key,
            len(batch),
        )
    return prepared


def _retarget_prepared_datasets_max_tokens(
    prepared_datasets: Sequence[PreparedDataset],
    *,
    max_tokens: int,
) -> List[PreparedDataset]:
    adjusted: List[PreparedDataset] = []
    for prepared in prepared_datasets:
        adjusted.append(
            PreparedDataset(
                spec=replace(prepared.spec, max_tokens=int(max_tokens)),
                result_key=prepared.result_key,
                batch=prepared.batch,
                prompts=prepared.prompts,
            )
        )
    return adjusted


def _infer_checkpoint_step(checkpoint_name: str) -> Optional[int]:
    direct_match = DIRECT_CHECKPOINT_PATTERN.match(checkpoint_name)
    if direct_match is not None:
        return int(direct_match.group(1))
    iter_match = ITER_ACTOR_PATTERN.match(checkpoint_name)
    if iter_match is not None:
        return int(iter_match.group(1))
    return None


def _add_checkpoint_ref(
    refs_by_path: Dict[str, CheckpointRef],
    *,
    experiment_dir: Path,
    checkpoint_dir: Path,
    checkpoint_type: str,
) -> None:
    resolved_checkpoint = checkpoint_dir.resolve()
    if not _checkpoint_has_weights(resolved_checkpoint):
        return
    refs_by_path[str(resolved_checkpoint)] = CheckpointRef(
        experiment_dir=experiment_dir.resolve(),
        checkpoint_dir=resolved_checkpoint,
        checkpoint_name=resolved_checkpoint.name,
        checkpoint_type=checkpoint_type,
        checkpoint_step=_infer_checkpoint_step(resolved_checkpoint.name),
    )


def _collect_from_experiment_dir(
    refs_by_path: Dict[str, CheckpointRef],
    experiment_dir: Path,
    *,
    include_anchor_ema: bool,
) -> None:
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        return

    for child in sorted(experiment_dir.iterdir()):
        if not child.is_dir():
            continue
        if DIRECT_CHECKPOINT_PATTERN.match(child.name):
            _add_checkpoint_ref(
                refs_by_path,
                experiment_dir=experiment_dir,
                checkpoint_dir=child,
                checkpoint_type="trainer_checkpoint",
            )

    checkpoints_dir = experiment_dir / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        for child in sorted(checkpoints_dir.iterdir()):
            if not child.is_dir():
                continue
            if ITER_ACTOR_PATTERN.match(child.name):
                _add_checkpoint_ref(
                    refs_by_path,
                    experiment_dir=experiment_dir,
                    checkpoint_dir=child,
                    checkpoint_type="iter_actor",
                )
            elif include_anchor_ema and child.name == "anchor_ema":
                _add_checkpoint_ref(
                    refs_by_path,
                    experiment_dir=experiment_dir,
                    checkpoint_dir=child,
                    checkpoint_type="anchor_ema",
                )


def _collect_from_path(
    refs_by_path: Dict[str, CheckpointRef],
    path: Path,
    *,
    include_anchor_ema: bool,
) -> None:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if _checkpoint_has_weights(path):
        experiment_dir = path.parent
        if path.parent.name == "checkpoints":
            experiment_dir = path.parent.parent
        checkpoint_type = "weight_dir"
        if DIRECT_CHECKPOINT_PATTERN.match(path.name):
            checkpoint_type = "trainer_checkpoint"
        elif ITER_ACTOR_PATTERN.match(path.name):
            checkpoint_type = "iter_actor"
        elif path.name == "anchor_ema":
            checkpoint_type = "anchor_ema"
        _add_checkpoint_ref(
            refs_by_path,
            experiment_dir=experiment_dir,
            checkpoint_dir=path,
            checkpoint_type=checkpoint_type,
        )
        return

    if path.name == "checkpoints":
        _collect_from_experiment_dir(
            refs_by_path,
            path.parent,
            include_anchor_ema=include_anchor_ema,
        )
        return

    before = len(refs_by_path)
    _collect_from_experiment_dir(
        refs_by_path,
        path,
        include_anchor_ema=include_anchor_ema,
    )
    if len(refs_by_path) > before:
        return

    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "manual_eval":
            continue
        _collect_from_path(
            refs_by_path,
            child,
            include_anchor_ema=include_anchor_ema,
        )


def _checkpoint_sort_key(ref: CheckpointRef, *, experiments_root: Path) -> tuple[str, int, str]:
    relative_exp = str(_safe_relative_path(ref.experiment_dir, experiments_root))
    step = ref.checkpoint_step if ref.checkpoint_step is not None else -1
    return (relative_exp, step, ref.checkpoint_name)


def discover_checkpoints(
    *,
    experiments_root: Path,
    paths: Sequence[str],
    include_anchor_ema: bool,
    max_checkpoints: Optional[int],
) -> List[CheckpointRef]:
    refs_by_path: Dict[str, CheckpointRef] = {}
    if len(paths) > 0:
        for raw_path in paths:
            _collect_from_path(
                refs_by_path,
                Path(raw_path),
                include_anchor_ema=include_anchor_ema,
            )
    else:
        for child in sorted(experiments_root.iterdir()):
            if not child.is_dir() or child.name == "manual_eval":
                continue
            _collect_from_path(
                refs_by_path,
                child,
                include_anchor_ema=include_anchor_ema,
            )

    refs = sorted(
        refs_by_path.values(),
        key=lambda ref: _checkpoint_sort_key(ref, experiments_root=experiments_root),
    )
    if max_checkpoints is not None:
        refs = refs[:max_checkpoints]
    return refs


def _is_myalgo_checkpoint(checkpoint_ref: CheckpointRef) -> bool:
    normalized_haystacks = [
        str(checkpoint_ref.checkpoint_dir).lower(),
        str(checkpoint_ref.experiment_dir).lower(),
        str(checkpoint_ref.checkpoint_name).lower(),
    ]
    return any(("myalgo" in value) or ("my_algo" in value) for value in normalized_haystacks)


def _compute_run_seed(
    *,
    base_seed: int,
    checkpoint_dir: Path,
    dataset_key: str,
    run_index: int,
) -> int:
    digest = hashlib.sha256(
        f"{checkpoint_dir.resolve()}::{dataset_key}::{run_index}".encode("utf-8")
    ).hexdigest()
    offset = int(digest[:12], 16) % 1_000_000_000
    return int(base_seed) + int(offset)


def _compute_required_max_model_len(
    *,
    tokenizer: Any,
    tokenizer_source: str,
    prepared_datasets: Sequence[PreparedDataset],
) -> int:
    safety_margin = 16
    max_required = 0
    max_prompt_tokens = 0
    max_dataset_key = None
    max_sample_index = -1

    for prepared in prepared_datasets:
        prompts = prepared.prompts
        if len(prompts) == 0:
            max_required = max(max_required, prepared.spec.max_tokens + safety_margin)
            continue
        batch_size = 64
        dataset_max_prompt_tokens = 0
        dataset_max_prompt_index = -1
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            encoded = tokenizer(
                prompt_batch,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            for offset, input_ids in enumerate(encoded):
                prompt_len = len(input_ids)
                if prompt_len > dataset_max_prompt_tokens:
                    dataset_max_prompt_tokens = prompt_len
                    dataset_max_prompt_index = start + offset
        required = dataset_max_prompt_tokens + int(prepared.spec.max_tokens) + safety_margin
        if required > max_required:
            max_required = required
            max_prompt_tokens = dataset_max_prompt_tokens
            max_dataset_key = prepared.result_key
            max_sample_index = dataset_max_prompt_index

    supported_context_len = _infer_supported_context_len(
        tokenizer,
        tokenizer_source=tokenizer_source,
    )
    if supported_context_len is not None and max_required > supported_context_len:
        raise ValueError(
            "Requested evaluation context exceeds model capacity: "
            f"required={max_required}, supported={supported_context_len}, "
            f"dataset={max_dataset_key}, max_prompt_tokens={max_prompt_tokens}, "
            f"sample_index={max_sample_index}."
        )

    logger.info(
        "Dynamic max_model_len=%d (dataset=%s max_prompt_tokens=%d sample_index=%d supported=%s)",
        max_required,
        max_dataset_key,
        max_prompt_tokens,
        max_sample_index,
        supported_context_len if supported_context_len is not None else "unknown",
    )
    return max_required


def _checkpoint_output_dir(
    *,
    output_root: Path,
    experiments_root: Path,
    checkpoint_ref: CheckpointRef,
) -> Path:
    relative_experiment_dir = _safe_relative_path(checkpoint_ref.experiment_dir, experiments_root)
    return output_root / relative_experiment_dir / checkpoint_ref.checkpoint_name


def _evaluate_dataset_once(
    *,
    checkpoint_ref: CheckpointRef,
    prepared_dataset: PreparedDataset,
    client: VLLMClient,
    tokenizer: Any,
    pass_1_cfg: Dict[str, Any],
    seed: int,
    run_index: int,
    dataset_output_dir: Path,
    gpu_id: int,
) -> Dict[str, Any]:
    start_time = time.time()
    run_prefix = f"run_{run_index:02d}"
    outcomes_log_path = dataset_output_dir / f"{run_prefix}_outcomes.log"
    per_example_scores_path = dataset_output_dir / f"{run_prefix}_per_example_scores.jsonl"
    metrics_path = dataset_output_dir / f"{run_prefix}_metrics.json"
    run_record_path = dataset_output_dir / f"{run_prefix}.json"

    batch = prepared_dataset.batch
    prompts = prepared_dataset.prompts
    spec = prepared_dataset.spec

    generated_pass1 = client.generate_batch(
        prompts=prompts,
        n=1,
        temperature=float(pass_1_cfg["temperature"]),
        top_p=float(pass_1_cfg["top_p"]),
        max_tokens=int(pass_1_cfg["max_tokens"]),
        stop=spec.stop,
        seed=seed,
    )

    reward_fn = MathRewardFn(answer_field=str(spec.answer_field))
    num_total = len(batch)
    num_correct_pass1 = 0
    num_empty_pass1 = 0
    pass1_texts: List[str] = []
    pass1_total_rewards: List[float] = []
    pass1_format_rewards: List[float] = []
    pass1_answer_rewards: List[float] = []
    pass1_length_rewards: List[float] = []

    with outcomes_log_path.open("w", encoding="utf-8") as outcome_f, per_example_scores_path.open(
        "w", encoding="utf-8"
    ) as score_f:
        outcome_f.write("Evaluation Outcomes\n")
        outcome_f.write(f"checkpoint_dir: {checkpoint_ref.checkpoint_dir}\n")
        outcome_f.write(f"dataset_alias: {spec.name}\n")
        outcome_f.write(
            f"dataset: {spec.dataset_name}[{spec.dataset_split}] config={spec.dataset_config_name}\n"
        )
        outcome_f.write(f"run_index: {run_index}\n")
        outcome_f.write(f"seed: {seed}\n")
        outcome_f.write(f"num_examples: {num_total}\n\n")

        for idx, (sample, sampled_choices) in enumerate(zip(batch, generated_pass1)):
            sampled_choice = sampled_choices[0] if sampled_choices else None
            pred = str(sampled_choice.get("text", "")) if sampled_choice else ""
            if not pred:
                num_empty_pass1 += 1
            pass1_texts.append(pred)

            sampled_score = reward_fn.score_completion(sampled_choice or pred, sample)
            pred_answer = extract_pred_answer(pred)
            raw_gold_answer = str(sample.get(spec.answer_field, ""))
            gold_answer = extract_gold_answer_text(raw_gold_answer)
            parsed_gold_answer = reward_fn.describe_gold_math_verify_parse(sample)
            parsed_pred_answer = reward_fn.describe_prediction_math_verify_parse(
                sampled_choice or pred
            )

            pass1_total_rewards.append(float(sampled_score.total_reward))
            pass1_format_rewards.append(float(sampled_score.format_reward))
            pass1_answer_rewards.append(float(sampled_score.answer_reward))
            pass1_length_rewards.append(float(sampled_score.length_reward))

            is_correct_pass1 = sampled_score.total_reward > 0.5
            if is_correct_pass1:
                num_correct_pass1 += 1

            question_text = str(
                sample.get(
                    spec.question_field,
                    _format_prompt(
                        example=sample,
                        question_template=spec.question_template,
                        question_field=spec.question_field,
                    ),
                )
            )

            outcome_f.write(f"[{idx}]\n")
            outcome_f.write(f"Question: {question_text}\n")
            outcome_f.write(f"Complete Answer: {pred}\n")
            if raw_gold_answer.strip() == gold_answer:
                outcome_f.write(f"Gold Answer: {gold_answer}\n")
            else:
                outcome_f.write(f"Gold Answer Raw: {raw_gold_answer}\n")
                outcome_f.write(f"Gold Answer Extracted: {gold_answer}\n")
            outcome_f.write(f"MathVerify Parsed Gold: {parsed_gold_answer}\n")
            outcome_f.write(f"Pass@1 Extracted Answer: {pred_answer}\n")
            outcome_f.write(f"MathVerify Parsed Pred: {parsed_pred_answer}\n")
            outcome_f.write(f"Pass@1 Reward Total: {sampled_score.total_reward:.6f}\n")
            outcome_f.write(f"Pass@1 Reward Format: {sampled_score.format_reward:.6f}\n")
            outcome_f.write(f"Pass@1 Reward Answer: {sampled_score.answer_reward:.6f}\n")
            outcome_f.write(f"Pass@1 Reward Length: {sampled_score.length_reward:.6f}\n")
            outcome_f.write(
                f"Pass@1 Correct: {'Correct' if is_correct_pass1 else 'wrong'}\n\n"
            )

            per_example_record = {
                "index": int(idx),
                "question": question_text,
                "gold_answer_raw": raw_gold_answer,
                "gold_answer_extracted": gold_answer,
                "pass1": {
                    "text": pred,
                    "answer_only_text": sampled_score.answer_only_text,
                    "total_reward": float(sampled_score.total_reward),
                    "format_reward": float(sampled_score.format_reward),
                    "answer_reward": float(sampled_score.answer_reward),
                    "length_reward": float(sampled_score.length_reward),
                    "is_correct": bool(is_correct_pass1),
                    "finish_reason": sampled_score.finish_reason,
                },
            }
            score_f.write(json.dumps(per_example_record, ensure_ascii=False))
            score_f.write("\n")

    elapsed = time.time() - start_time
    pass1_token_lengths = _tokenize_text_lengths(tokenizer, pass1_texts)
    accuracy = float(num_correct_pass1 / max(num_total, 1))
    metrics = {
        "enabled": True,
        "timestamp": int(start_time),
        "checkpoint_dir": str(checkpoint_ref.checkpoint_dir),
        "dataset_alias": spec.name,
        "dataset_name": str(spec.dataset_name),
        "dataset_config_name": (
            str(spec.dataset_config_name) if spec.dataset_config_name is not None else None
        ),
        "dataset_split": str(spec.dataset_split),
        "question_field": str(spec.question_field),
        "answer_field": str(spec.answer_field),
        "num_examples": int(num_total),
        "num_correct": int(num_correct_pass1),
        "num_correct_pass1": int(num_correct_pass1),
        "accuracy": accuracy,
        "pass@1": accuracy,
        "pass_at_1": accuracy,
        "avg_total_reward_pass1": float(
            sum(pass1_total_rewards) / max(len(pass1_total_rewards), 1)
        ),
        "avg_format_reward_pass1": float(
            sum(pass1_format_rewards) / max(len(pass1_format_rewards), 1)
        ),
        "avg_answer_reward_pass1": float(
            sum(pass1_answer_rewards) / max(len(pass1_answer_rewards), 1)
        ),
        "avg_length_reward_pass1": float(
            sum(pass1_length_rewards) / max(len(pass1_length_rewards), 1)
        ),
        "empty_predictions": int(num_empty_pass1),
        "empty_predictions_pass1": int(num_empty_pass1),
        "avg_completion_tokens_pass1": float(
            sum(pass1_token_lengths) / max(len(pass1_token_lengths), 1)
        ),
        "eval_seconds": float(elapsed),
        "model": str(checkpoint_ref.checkpoint_dir),
        "num_inference_gpus": 1,
        "inference_gpu_ids": [int(gpu_id)],
        "pass_k_enabled": False,
        "pass_k_num_samples": 0,
        "outcomes_log_path": str(outcomes_log_path),
        "per_example_scores_path": str(per_example_scores_path),
    }
    _add_pass_1_sampling_metrics(metrics, pass_1_cfg)
    save_json_atomic(metrics, metrics_path)

    run_record = {
        "run_index": int(run_index),
        "seed": int(seed),
        "checkpoint_dir": str(checkpoint_ref.checkpoint_dir),
        "dataset_result_key": prepared_dataset.result_key,
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "outcomes_log_path": str(outcomes_log_path),
        "per_example_scores_path": str(per_example_scores_path),
        "evaluated_at": float(time.time()),
    }
    save_json_atomic(run_record, run_record_path)
    return run_record


def _write_checkpoint_summary(
    *,
    checkpoint_ref: CheckpointRef,
    checkpoint_output_dir: Path,
    prepared_datasets: Sequence[PreparedDataset],
    repeat_count: int,
) -> Dict[str, Any]:
    datasets_summary: Dict[str, Any] = {}
    completed_datasets = 0
    mean_accs: List[float] = []
    for prepared in prepared_datasets:
        summary_path = checkpoint_output_dir / prepared.result_key / "summary.json"
        summary_payload = _load_json(summary_path)
        if summary_payload is None:
            continue
        datasets_summary[prepared.result_key] = summary_payload
        if bool(summary_payload.get("completed", False)):
            completed_datasets += 1
        mean_acc = summary_payload.get("mean_acc", summary_payload.get("accuracy_mean"))
        if isinstance(mean_acc, (int, float)):
            mean_accs.append(float(mean_acc))

    checkpoint_summary = {
        "checkpoint_dir": str(checkpoint_ref.checkpoint_dir),
        "experiment_dir": str(checkpoint_ref.experiment_dir),
        "checkpoint_name": checkpoint_ref.checkpoint_name,
        "checkpoint_type": checkpoint_ref.checkpoint_type,
        "checkpoint_step": checkpoint_ref.checkpoint_step,
        "repeat_count": int(repeat_count),
        "num_datasets": int(len(prepared_datasets)),
        "num_completed_datasets": int(completed_datasets),
        "all_datasets_completed": bool(completed_datasets == len(prepared_datasets)),
        "macro_mean_acc": float(sum(mean_accs) / len(mean_accs)) if mean_accs else None,
        "macro_accuracy_mean": float(sum(mean_accs) / len(mean_accs)) if mean_accs else None,
        "datasets": datasets_summary,
        "updated_at": float(time.time()),
    }
    save_json_atomic(checkpoint_summary, checkpoint_output_dir / "checkpoint_summary.json")
    return checkpoint_summary


def _build_global_summary(
    *,
    output_root: Path,
    experiments_root: Path,
    dataset_specs: Sequence[DatasetSpec],
    repeat_count: int,
) -> Dict[str, Any]:
    compact_checkpoints: List[Dict[str, Any]] = []
    leaderboards: Dict[str, List[Dict[str, Any]]] = {
        _dataset_result_key(spec.dataset_name, spec.dataset_split, spec.dataset_config_name): []
        for spec in dataset_specs
    }

    for summary_path in sorted(output_root.rglob("checkpoint_summary.json")):
        checkpoint_summary = _load_json(summary_path)
        if checkpoint_summary is None:
            continue

        compact_dataset_summary: Dict[str, Any] = {}
        for dataset_key, dataset_summary in checkpoint_summary.get("datasets", {}).items():
            mean_acc = dataset_summary.get("mean_acc", dataset_summary.get("accuracy_mean"))
            variance = dataset_summary.get("variance", dataset_summary.get("accuracy_variance"))
            best_acc = dataset_summary.get("best_acc")
            worst_acc = dataset_summary.get("worst_acc")
            compact_dataset_summary[dataset_key] = {
                "best_acc": best_acc,
                "worst_acc": worst_acc,
                "mean_acc": mean_acc,
                "variance": variance,
                "accuracy_mean": mean_acc,
                "accuracy_variance": variance,
                "accuracy_std": dataset_summary.get("accuracy_std"),
                "best_accuracy": best_acc,
                "worst_accuracy": worst_acc,
                "summary_path": str(summary_path.parent / dataset_key / "summary.json"),
                "best_outcomes_copy_path": dataset_summary.get("best_outcomes_copy_path"),
                "worst_outcomes_copy_path": dataset_summary.get("worst_outcomes_copy_path"),
            }
            if isinstance(mean_acc, (int, float)):
                leaderboards.setdefault(dataset_key, []).append(
                    {
                        "checkpoint_dir": checkpoint_summary["checkpoint_dir"],
                        "experiment_dir": checkpoint_summary["experiment_dir"],
                        "checkpoint_name": checkpoint_summary["checkpoint_name"],
                        "checkpoint_step": checkpoint_summary["checkpoint_step"],
                        "best_acc": best_acc,
                        "worst_acc": worst_acc,
                        "mean_acc": float(mean_acc),
                        "variance": variance,
                        "accuracy_mean": float(mean_acc),
                        "accuracy_variance": variance,
                        "accuracy_std": dataset_summary.get("accuracy_std"),
                        "summary_path": str(summary_path.parent / dataset_key / "summary.json"),
                        "best_outcomes_copy_path": dataset_summary.get("best_outcomes_copy_path"),
                        "worst_outcomes_copy_path": dataset_summary.get("worst_outcomes_copy_path"),
                    }
                )

        compact_checkpoints.append(
            {
                "checkpoint_dir": checkpoint_summary["checkpoint_dir"],
                "experiment_dir": checkpoint_summary["experiment_dir"],
                "checkpoint_name": checkpoint_summary["checkpoint_name"],
                "checkpoint_type": checkpoint_summary["checkpoint_type"],
                "checkpoint_step": checkpoint_summary["checkpoint_step"],
                "repeat_count": checkpoint_summary["repeat_count"],
                "num_datasets": checkpoint_summary["num_datasets"],
                "num_completed_datasets": checkpoint_summary["num_completed_datasets"],
                "all_datasets_completed": checkpoint_summary["all_datasets_completed"],
                "macro_mean_acc": checkpoint_summary.get(
                    "macro_mean_acc",
                    checkpoint_summary.get("macro_accuracy_mean"),
                ),
                "macro_accuracy_mean": checkpoint_summary.get("macro_accuracy_mean"),
                "relative_output_dir": str(
                    _safe_relative_path(summary_path.parent, output_root)
                ),
                "datasets": compact_dataset_summary,
            }
        )

    for dataset_key, rows in leaderboards.items():
        rows.sort(key=lambda item: float(item["mean_acc"]), reverse=True)

    return {
        "generated_at": float(time.time()),
        "output_root": str(output_root.resolve()),
        "experiments_root": str(experiments_root.resolve()),
        "repeat_count": int(repeat_count),
        "dataset_specs": [asdict(spec) for spec in dataset_specs],
        "num_checkpoints_with_summary": int(len(compact_checkpoints)),
        "checkpoints": compact_checkpoints,
        "leaderboards": leaderboards,
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all discovered checkpoints with randomized single-sample decoding "
            "on multiple datasets, repeating each dataset 5 times by default."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Optional experiment/checkpoint paths. If omitted, scan --experiments-root "
            "for all supported checkpoints."
        ),
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default=str((PROJECT_ROOT / "experiments").resolve()),
        help="Experiments root to scan when no explicit paths are provided.",
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
        help="Directory where per-run outcomes and aggregated summaries are written.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="math500,gsm8k,aime24",
        help="Comma-separated built-in dataset aliases. Available: math500,gsm8k,aime24.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=5,
        help="How many randomized runs to execute per checkpoint per dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Each checkpoint/dataset/run derives its own stable seed from it.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=6,
        help="Physical GPU id for the single A100 used by vLLM.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per dataset for quick smoke runs.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Optional cap on the number of discovered checkpoints.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max new tokens for the randomized pass@1 generation.",
    )
    parser.add_argument(
        "--myalgo-extra-max-tokens",
        type=int,
        default=0,
        help=(
            "Extra max_tokens added only for checkpoints whose path/name matches "
            "myalgo or my_algo."
        ),
    )
    parser.add_argument(
        "--increment-max-tokens-per-checkpoint",
        nargs="?",
        const=10,
        default=0,
        type=int,
        metavar="STEP",
        help=(
            "Increase max_tokens by STEP after each checkpoint. "
            "Checkpoint 1 uses --max-tokens, checkpoint 2 uses --max-tokens+STEP, etc. "
            "If provided without STEP, defaults to 10."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for randomized pass@1 decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Sampling top-p for randomized pass@1 decoding.",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help='Optional repeated stop sequence argument. Defaults to "<锝淯ser锝?".',
    )
    parser.add_argument(
        "--request-timeout-s",
        type=int,
        default=300,
        help="HTTP request timeout for each vLLM completion request.",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=128,
        help="Parallel HTTP requests to vLLM. Tuned aggressively for a 1.5B model on one A100.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="vLLM gpu_memory_utilization for the single A100.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="vLLM max_num_seqs for the single A100.",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=16,
        help="vLLM swap-space in GiB.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="vLLM dtype.",
    )
    parser.add_argument(
        "--wait-timeout-s",
        type=int,
        default=800,
        help="How long to wait for the vLLM server to start.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing evaluation outputs for a checkpoint and rerun all datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover checkpoints and datasets, then write the manifest and exit.",
    )
    parser.add_argument(
        "--include-anchor-ema",
        action="store_true",
        help="Also include checkpoints/checkpoints/anchor_ema if present.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first checkpoint failure.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.repeat_count <= 0:
        raise ValueError("--repeat-count must be positive.")
    if args.max_checkpoints is not None and args.max_checkpoints <= 0:
        raise ValueError("--max-checkpoints must be positive.")
    if args.increment_max_tokens_per_checkpoint < 0:
        raise ValueError("--increment-max-tokens-per-checkpoint STEP must be non-negative.")
    if args.myalgo_extra_max_tokens < 0:
        raise ValueError("--myalgo-extra-max-tokens must be non-negative.")

    stop_sequences = args.stop if args.stop else list(DEFAULT_DPSK_STOP)
    experiments_root = Path(args.experiments_root).resolve()
    output_root = ensure_dir(Path(args.output_root).resolve())

    dataset_specs = _resolve_dataset_specs(
        project_root=PROJECT_ROOT,
        dataset_names_csv=args.datasets,
        max_tokens=int(args.max_tokens),
        stop=stop_sequences,
    )
    prepared_datasets = _prepare_datasets(
        dataset_specs,
        max_samples=args.max_samples,
    )

    checkpoints = discover_checkpoints(
        experiments_root=experiments_root,
        paths=args.paths,
        include_anchor_ema=bool(args.include_anchor_ema),
        max_checkpoints=args.max_checkpoints,
    )
    if len(checkpoints) == 0:
        raise RuntimeError("No checkpoints discovered.")

    manifest = {
        "generated_at": float(time.time()),
        "project_root": str(PROJECT_ROOT.resolve()),
        "experiments_root": str(experiments_root),
        "output_root": str(output_root),
        "argv": sys.argv,
        "repeat_count": int(args.repeat_count),
        "gpu_id": int(args.gpu_id),
        "max_parallel_requests": int(args.max_parallel_requests),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": int(args.max_num_seqs),
        "swap_space": int(args.swap_space),
        "dtype": str(args.dtype),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "myalgo_extra_max_tokens": int(args.myalgo_extra_max_tokens),
        "increment_max_tokens_per_checkpoint": bool(
            int(args.increment_max_tokens_per_checkpoint) > 0
        ),
        "max_tokens_increment_step": int(args.increment_max_tokens_per_checkpoint),
        "stop": stop_sequences,
        "dataset_specs": [asdict(spec) for spec in dataset_specs],
        "num_checkpoints": int(len(checkpoints)),
        "total_planned_runs": int(
            len(checkpoints) * len(prepared_datasets) * int(args.repeat_count)
        ),
        "checkpoints": [checkpoint.to_dict() for checkpoint in checkpoints],
    }
    save_json_atomic(manifest, output_root / "manifest.json")

    logger.info(
        "Discovered %d checkpoints, %d datasets, repeat_count=%d, total_planned_runs=%d",
        len(checkpoints),
        len(prepared_datasets),
        args.repeat_count,
        manifest["total_planned_runs"],
    )

    if args.dry_run:
        logger.info("Dry run requested. Manifest written to %s", output_root / "manifest.json")
        return

    errors_path = output_root / "errors.jsonl"
    max_tokens_increment_step = int(args.increment_max_tokens_per_checkpoint)

    for checkpoint_idx, checkpoint_ref in enumerate(checkpoints, start=1):
        checkpoint_max_tokens = int(args.max_tokens)
        if max_tokens_increment_step > 0:
            checkpoint_max_tokens += max_tokens_increment_step * (checkpoint_idx - 1)
        if int(args.myalgo_extra_max_tokens) > 0 and _is_myalgo_checkpoint(checkpoint_ref):
            checkpoint_max_tokens += int(args.myalgo_extra_max_tokens)

        checkpoint_prepared_datasets = _retarget_prepared_datasets_max_tokens(
            prepared_datasets,
            max_tokens=checkpoint_max_tokens,
        )
        eval_cfg_like = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(checkpoint_max_tokens),
            "enable_pass_k": False,
            "pass_k_temperature": float(args.temperature),
            "pass_k_top_p": float(args.top_p),
            "pass_k_max_tokens": int(checkpoint_max_tokens),
            "repeat_until_new_checkpoint": True,
            "randomize_pass_1_when_pass_k_disabled": True,
        }
        pass_1_cfg = _resolve_pass_1_generation_cfg(eval_cfg_like)

        checkpoint_output_dir = _checkpoint_output_dir(
            output_root=output_root,
            experiments_root=experiments_root,
            checkpoint_ref=checkpoint_ref,
        )

        if args.force and checkpoint_output_dir.exists():
            shutil.rmtree(checkpoint_output_dir)

        checkpoint_output_dir = ensure_dir(checkpoint_output_dir)
        logger.info(
            "[%d/%d] checkpoint=%s max_tokens=%d",
            checkpoint_idx,
            len(checkpoints),
            checkpoint_ref.checkpoint_dir,
            checkpoint_max_tokens,
        )

        try:
            dataset_output_dirs = {
                prepared.result_key: ensure_dir(checkpoint_output_dir / prepared.result_key)
                for prepared in checkpoint_prepared_datasets
            }

            pending_prepared: List[PreparedDataset] = []
            for prepared in checkpoint_prepared_datasets:
                dataset_output_dir = dataset_output_dirs[prepared.result_key]
                runs_path = dataset_output_dir / "runs.jsonl"
                existing_records = [] if args.force else _load_jsonl_records(runs_path)
                if len(existing_records) >= int(args.repeat_count):
                    summary = _summarize_dataset_runs(
                        records=existing_records,
                        checkpoint_ref=checkpoint_ref,
                        prepared_dataset=prepared,
                        repeat_count=int(args.repeat_count),
                    )
                    summary["runs_path"] = str(runs_path)
                    summary = _materialize_extrema_artifacts(dataset_output_dir, summary)
                    save_json_atomic(summary, dataset_output_dir / "summary.json")
                    logger.info(
                            "Skipping completed dataset checkpoint=%s dataset=%s existing_runs=%d",
                            checkpoint_ref.checkpoint_name,
                            prepared.result_key,
                            len(existing_records),
                        )
                    continue
                pending_prepared.append(prepared)

            if len(pending_prepared) > 0:
                tokenizer = load_causal_lm_tokenizer(
                    str(checkpoint_ref.checkpoint_dir),
                    trust_remote_code=True,
                )
                dynamic_max_model_len = _compute_required_max_model_len(
                    tokenizer=tokenizer,
                    tokenizer_source=str(checkpoint_ref.checkpoint_dir),
                    prepared_datasets=pending_prepared,
                )
                vllm_cfg = {
                    "host": "127.0.0.1",
                    "port": None,
                    "gpu_idx": int(args.gpu_id),
                    "swap_space": int(args.swap_space),
                    "dtype": str(args.dtype),
                    "trust_remote_code": True,
                    "gpu_memory_utilization": float(args.gpu_memory_utilization),
                    "max_num_seqs": int(args.max_num_seqs),
                    "max_model_len": int(dynamic_max_model_len),
                    "enable_prefix_caching": True,
                    "disable_sliding_window": False,
                    "disable_frontend_multiprocessing": False,
                    "wait_timeout_s": int(args.wait_timeout_s),
                    "log_file": "vllm_server.log",
                }

                server_logs_dir = ensure_dir(checkpoint_output_dir / "_server_logs")
                server = VLLMServer(vllm_cfg, server_logs_dir)
                try:
                    api_base = server.start(
                        model_name_or_path=str(checkpoint_ref.checkpoint_dir),
                        seed=int(args.seed) + checkpoint_idx,
                    )
                    client = VLLMClient(
                        api_base=api_base,
                        model=str(checkpoint_ref.checkpoint_dir),
                        request_timeout_s=int(args.request_timeout_s),
                        max_parallel_requests=int(args.max_parallel_requests),
                    )
                    for prepared in pending_prepared:
                        dataset_output_dir = dataset_output_dirs[prepared.result_key]
                        runs_path = dataset_output_dir / "runs.jsonl"
                        existing_records = [] if args.force else _load_jsonl_records(runs_path)
                        next_run_index = len(existing_records) + 1
                        logger.info(
                            "checkpoint=%s dataset=%s pending_runs=%d/%d",
                            checkpoint_ref.checkpoint_name,
                            prepared.result_key,
                            max(0, args.repeat_count - len(existing_records)),
                            args.repeat_count,
                        )
                        for run_index in range(next_run_index, int(args.repeat_count) + 1):
                            run_seed = _compute_run_seed(
                                base_seed=int(args.seed),
                                checkpoint_dir=checkpoint_ref.checkpoint_dir,
                                dataset_key=prepared.result_key,
                                run_index=run_index,
                            )
                            logger.info(
                                "Evaluating checkpoint=%s dataset=%s run=%d/%d seed=%d",
                                checkpoint_ref.checkpoint_name,
                                prepared.result_key,
                                run_index,
                                args.repeat_count,
                                run_seed,
                            )
                            run_record = _evaluate_dataset_once(
                                checkpoint_ref=checkpoint_ref,
                                prepared_dataset=prepared,
                                client=client,
                                tokenizer=tokenizer,
                                pass_1_cfg=pass_1_cfg,
                                seed=run_seed,
                                run_index=run_index,
                                dataset_output_dir=dataset_output_dir,
                                gpu_id=int(args.gpu_id),
                            )
                            existing_records.append(run_record)
                            _append_jsonl(runs_path, run_record)

                            summary = _summarize_dataset_runs(
                                records=existing_records,
                                checkpoint_ref=checkpoint_ref,
                                prepared_dataset=prepared,
                                repeat_count=int(args.repeat_count),
                            )
                            summary["runs_path"] = str(runs_path)
                            summary = _materialize_extrema_artifacts(dataset_output_dir, summary)
                            save_json_atomic(summary, dataset_output_dir / "summary.json")

                            accuracy = float(run_record["metrics"].get("accuracy", 0.0))
                            logger.info(
                                "Finished checkpoint=%s dataset=%s run=%d accuracy=%.4f",
                                checkpoint_ref.checkpoint_name,
                                prepared.result_key,
                                run_index,
                                accuracy,
                            )
                finally:
                    server.stop()

            _write_checkpoint_summary(
                checkpoint_ref=checkpoint_ref,
                checkpoint_output_dir=checkpoint_output_dir,
                prepared_datasets=checkpoint_prepared_datasets,
                repeat_count=int(args.repeat_count),
            )
        except Exception as exc:
            error_payload = {
                "checkpoint_dir": str(checkpoint_ref.checkpoint_dir),
                "checkpoint_name": checkpoint_ref.checkpoint_name,
                "checkpoint_step": checkpoint_ref.checkpoint_step,
                "max_tokens": int(checkpoint_max_tokens),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": float(time.time()),
            }
            save_json_atomic(error_payload, checkpoint_output_dir / "error.json")
            _append_jsonl(errors_path, error_payload)
            logger.exception("Checkpoint evaluation failed: %s", checkpoint_ref.checkpoint_dir)
            if args.fail_fast:
                raise

        global_summary = _build_global_summary(
            output_root=output_root,
            experiments_root=experiments_root,
            dataset_specs=dataset_specs,
            repeat_count=int(args.repeat_count),
        )
        save_json_atomic(global_summary, output_root / "global_summary.json")

    final_global_summary = _build_global_summary(
        output_root=output_root,
        experiments_root=experiments_root,
        dataset_specs=dataset_specs,
        repeat_count=int(args.repeat_count),
    )
    save_json_atomic(final_global_summary, output_root / "global_summary.json")
    logger.info("All checkpoint evaluation work finished. Summary: %s", output_root / "global_summary.json")


if __name__ == "__main__":
    main()
