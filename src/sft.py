import bisect
import inspect
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from src.distributed import barrier, get_dist_state
from src.reward import MathRewardFn
from src.utils import ensure_dir, save_json


def _safe_format_prompt(example: Dict[str, Any], template: str) -> str:
    class _SafeFormatDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    values = dict(example)
    if "problem" in example:
        values.setdefault("query", example["problem"])
    if "query" in example:
        values.setdefault("problem", example["query"])
    return template.format_map(_SafeFormatDict(values))


def _unwrap_model(model):
    current = model
    while hasattr(current, "module"):
        current = current.module
    return current


def _supported_keyword_params(callable_obj) -> Optional[set[str]]:
    target = callable_obj.__init__ if inspect.isclass(callable_obj) else callable_obj
    signature = inspect.signature(target)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return None
    return {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _filter_supported_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    supported = _supported_keyword_params(callable_obj)
    if supported is None:
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in supported}


def _build_training_arguments(training_kwargs: Dict[str, Any]) -> TrainingArguments:
    return TrainingArguments(**_filter_supported_kwargs(TrainingArguments, training_kwargs))


def _build_trainer(
    *,
    model,
    args,
    train_dataset,
    data_collator,
    tokenizer,
) -> Trainer:
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    supported = _supported_keyword_params(Trainer)
    if supported is None or "processing_class" in supported:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**_filter_supported_kwargs(Trainer, trainer_kwargs))


def _gather_object_logs(
    local_items: List[Dict[str, Any]],
    *,
    world_size: int,
) -> List[Dict[str, Any]]:
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_items)
        merged: List[Dict[str, Any]] = []
        for shard in gathered:
            if shard:
                merged.extend(shard)
        return merged
    return list(local_items)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _extract_question_text(sample: Dict[str, Any], prompt_text: str) -> str:
    for key in ("problem", "query", "question"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(prompt_text)


def _format_sample_eval_text(record: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "=" * 80,
            f"dataset_index: {int(record['dataset_index'])}",
            f"correct: {bool(record['correct'])}",
            "question:",
            str(record["question_text"]),
            "",
            "model_answer:",
            str(record["completion_text"]),
            "",
            "reference_answer:",
            str(record["expected_answer"]),
        ]
    )


def _format_eval_log_text(
    *,
    step: int,
    metrics: Dict[str, Any],
    generation_cfg: Dict[str, Any],
    sample_logs: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        f"evaluation_step: {int(step)}",
        f"num_examples: {int(metrics['num_examples'])}",
        f"num_correct: {int(metrics['num_correct'])}",
        f"accuracy: {float(metrics['accuracy']):.6f}",
        f"is_best: {bool(metrics.get('is_best', False))}",
        f"best_accuracy: {float(metrics.get('best_accuracy', 0.0)):.6f}",
        f"best_step: {int(metrics.get('best_step', step))}",
        f"empty_predictions: {int(metrics['empty_predictions'])}",
        f"elapsed_seconds: {float(metrics.get('elapsed_seconds', 0.0)):.3f}",
        (
            "generation_cfg: "
            f"eval_batch_size={int(generation_cfg['eval_batch_size'])}, "
            f"prompt_max_length={int(generation_cfg['prompt_max_length'])}, "
            f"max_new_tokens={int(generation_cfg['max_new_tokens'])}"
        ),
        "",
        "[Sample Results]",
    ]

    if sample_logs:
        for record in sample_logs:
            lines.extend([_format_sample_eval_text(record), ""])
    else:
        lines.append("No sample logs recorded.")

    return "\n".join(lines).rstrip() + "\n"


@dataclass
class SFTSource:
    name: str
    dataset: Any
    question_field: str
    response_field: str
    question_template: str
    append_final_answer_from_field: Optional[str] = None

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        if self.question_field not in example:
            raise KeyError(
                f"SFT source '{self.name}' missing question field '{self.question_field}'. "
                f"Available keys={list(example.keys())}"
            )
        if self.response_field not in example:
            raise KeyError(
                f"SFT source '{self.name}' missing response field '{self.response_field}'. "
                f"Available keys={list(example.keys())}"
            )

        prompt_text = _safe_format_prompt(
            {
                **example,
                "problem": example[self.question_field],
                "query": example[self.question_field],
            },
            self.question_template,
        )
        response_text = str(example[self.response_field]).strip()

        answer_field = self.append_final_answer_from_field
        if answer_field is not None:
            final_answer = str(example.get(answer_field, "")).strip()
            if final_answer and "final answer:" not in response_text.lower():
                response_text = (
                    f"{response_text}\n\nFinal Answer: {final_answer}"
                    if response_text
                    else f"Final Answer: {final_answer}"
                )

        return {
            "prompt_text": prompt_text,
            "response_text": response_text,
            "source_name": self.name,
        }


class MixedSFTDataset(Dataset):
    def __init__(self, sources: Sequence[SFTSource]):
        self.sources = list(sources)
        self._cumulative_sizes: List[int] = []

        total = 0
        for source in self.sources:
            total += len(source.dataset)
            self._cumulative_sizes.append(total)

    def __len__(self) -> int:
        if not self._cumulative_sizes:
            return 0
        return self._cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Dict[str, str]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        source_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        source_start = 0 if source_idx == 0 else self._cumulative_sizes[source_idx - 1]
        source = self.sources[source_idx]
        example = source.dataset[idx - source_start]
        return source.format_example(example)


class SFTDataCollator:
    def __init__(
        self,
        tokenizer,
        max_sequence_length: int,
        append_eos: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = int(max_sequence_length)
        self.append_eos = bool(append_eos)
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.eos_token_id = tokenizer.eos_token_id

    def _truncate_pair(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        if self.append_eos and self.eos_token_id is not None:
            if len(response_ids) == 0 or response_ids[-1] != int(self.eos_token_id):
                response_ids = list(response_ids) + [int(self.eos_token_id)]

        if len(response_ids) == 0:
            response_ids = [int(self.eos_token_id or self.pad_token_id)]

        if len(prompt_ids) + len(response_ids) <= self.max_sequence_length:
            return prompt_ids, response_ids

        # Prefer keeping the full supervised target and trimming prompt from the left.
        max_prompt_len = max(self.max_sequence_length - len(response_ids), 0)
        if max_prompt_len < len(prompt_ids):
            prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []

        if len(prompt_ids) + len(response_ids) <= self.max_sequence_length:
            return prompt_ids, response_ids

        # If the response alone is too long, keep the earliest target tokens
        # but ensure EOS is preserved at the end.
        response_ids = response_ids[: self.max_sequence_length]
        if self.append_eos and self.eos_token_id is not None:
            if len(response_ids) > 0 and response_ids[-1] != int(self.eos_token_id):
                response_ids[-1] = int(self.eos_token_id)
        prompt_ids = []
        return prompt_ids, response_ids

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            prompt_ids = self.tokenizer(
                feature["prompt_text"],
                add_special_tokens=False,
            ).input_ids
            response_ids = self.tokenizer(
                feature["response_text"],
                add_special_tokens=False,
            ).input_ids
            prompt_ids, response_ids = self._truncate_pair(prompt_ids, response_ids)

            merged_ids = prompt_ids + response_ids
            merged_labels = ([-100] * len(prompt_ids)) + response_ids

            input_ids.append(torch.tensor(merged_ids, dtype=torch.long))
            attention_mask.append(torch.ones(len(merged_ids), dtype=torch.long))
            labels.append(torch.tensor(merged_labels, dtype=torch.long))

        return {
            "input_ids": pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            ),
        }


def evaluate_math_generation(
    *,
    model,
    tokenizer,
    eval_examples: Sequence[Dict[str, Any]],
    answer_field: str,
    batch_size: int,
    prompt_max_length: int,
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    raw_model = _unwrap_model(model)
    device = next(raw_model.parameters()).device
    reward_fn = MathRewardFn(answer_field=answer_field)
    dist_state = get_dist_state()
    world_size = max(1, int(dist_state.num_processes))
    rank = int(dist_state.process_index)
    local_examples = list(eval_examples)[rank::world_size]
    eval_started_at = time.perf_counter()

    was_training = raw_model.training
    prev_use_cache = getattr(raw_model.config, "use_cache", None)
    if prev_use_cache is not None:
        raw_model.config.use_cache = True
    raw_model.eval()

    num_correct = 0
    num_empty = 0
    num_examples = len(local_examples)
    local_sample_logs: List[Dict[str, Any]] = []

    try:
        for start in range(0, num_examples, max(1, int(batch_size))):
            batch_examples = local_examples[start : start + batch_size]
            prompts = [str(ex["prompt_text"]) for ex in batch_examples]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(prompt_max_length),
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = raw_model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_length = int(encoded["input_ids"].shape[1])
            for row_idx, example in enumerate(batch_examples):
                local_index = start + row_idx
                generated_ids = generated[row_idx, prompt_length:]
                completion = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                is_empty = not completion.strip()
                if is_empty:
                    num_empty += 1
                reward = float(reward_fn(completion, example["sample"]))
                is_correct = reward > 0.5
                if is_correct:
                    num_correct += 1
                local_sample_logs.append(
                    {
                        "dataset_index": int(rank + (local_index * world_size)),
                        "correct": bool(is_correct),
                        "question_text": _extract_question_text(
                            example["sample"],
                            str(example["prompt_text"]),
                        ),
                        "expected_answer": str(example["sample"].get(answer_field, "")),
                        "completion_text": completion,
                    }
                )
    finally:
        if prev_use_cache is not None:
            raw_model.config.use_cache = prev_use_cache
        if was_training:
            raw_model.train()

    counts = torch.tensor(
        [int(num_examples), int(num_correct), int(num_empty)],
        dtype=torch.long,
        device=device,
    )
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    elapsed_ms = torch.tensor(
        [int((time.perf_counter() - eval_started_at) * 1000)],
        dtype=torch.long,
        device=device,
    )
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(elapsed_ms, op=dist.ReduceOp.MAX)

    total_examples = int(counts[0].item())
    total_correct = int(counts[1].item())
    total_empty = int(counts[2].item())
    metrics = {
        "num_examples": total_examples,
        "num_correct": total_correct,
        "accuracy": float(total_correct / max(total_examples, 1)),
        "empty_predictions": total_empty,
        "elapsed_seconds": float(elapsed_ms.item() / 1000.0),
    }
    sample_logs = _gather_object_logs(local_sample_logs, world_size=world_size)
    details = None
    if dist_state.is_main_process:
        details = {
            "sample_logs": sorted(
                sample_logs,
                key=lambda item: int(item["dataset_index"]),
            ),
        }
    return metrics, details


class MathGenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer,
        eval_examples: Sequence[Dict[str, Any]],
        answer_field: str,
        metrics_dir: Path,
        best_model_dir: Path,
        generation_cfg: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.eval_examples = list(eval_examples)
        self.answer_field = answer_field
        self.metrics_dir = ensure_dir(metrics_dir)
        self.eval_logs_dir = ensure_dir(metrics_dir / "eval_logs")
        self.best_model_dir = best_model_dir
        self.generation_cfg = generation_cfg
        self.history: List[Dict[str, Any]] = []
        self.best_accuracy: Optional[float] = None
        self.best_step: Optional[int] = None

    def _save_best_model(self, model, metrics: Dict[str, Any]) -> None:
        raw_model = _unwrap_model(model)
        if self.best_model_dir.exists():
            shutil.rmtree(self.best_model_dir, ignore_errors=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(str(self.best_model_dir), safe_serialization=False)
        self.tokenizer.save_pretrained(str(self.best_model_dir))
        save_json(metrics, self.best_model_dir / "best_eval_metrics.json")

    def _evaluate_and_track(self, *, model, step: int) -> None:
        metrics, eval_details = evaluate_math_generation(
            model=model,
            tokenizer=self.tokenizer,
            eval_examples=self.eval_examples,
            answer_field=self.answer_field,
            batch_size=int(self.generation_cfg["eval_batch_size"]),
            prompt_max_length=int(self.generation_cfg["prompt_max_length"]),
            max_new_tokens=int(self.generation_cfg["max_new_tokens"]),
        )
        metrics["step"] = int(step)
        accuracy = float(metrics["accuracy"])
        is_best = self.best_accuracy is None or accuracy > self.best_accuracy
        if is_best:
            self.best_accuracy = accuracy
            self.best_step = int(step)
        metrics["is_best"] = bool(is_best)
        metrics["best_accuracy"] = float(self.best_accuracy or 0.0)
        metrics["best_step"] = int(self.best_step or step)
        if not get_dist_state().is_main_process:
            return

        log_path = self.eval_logs_dir / f"eval_step_{int(step):06d}.log"
        sample_logs = []
        if eval_details is not None:
            sample_logs = list(eval_details.get("sample_logs", []))
        _write_text(
            log_path,
            _format_eval_log_text(
                step=int(step),
                metrics=metrics,
                generation_cfg=self.generation_cfg,
                sample_logs=sample_logs,
            ),
        )
        metrics["log_path"] = str(log_path)
        self.history.append(metrics)
        save_json({"history": self.history}, self.metrics_dir / "eval_history.json")

        if is_best:
            best_metrics = dict(metrics)
            best_metrics["best"] = True
            self._save_best_model(model, best_metrics)

    def on_save(self, args, state, control, **kwargs):
        barrier()
        model = kwargs.get("model")
        if model is None:
            barrier()
            return control

        self._evaluate_and_track(model=model, step=int(state.global_step))
        barrier()
        return control

    def ensure_best_model(self, *, model, step: int) -> None:
        if self.best_step == int(step) and self.best_accuracy is not None:
            return
        self._evaluate_and_track(model=model, step=int(step))


def train_sft_stage(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_examples: Sequence[Dict[str, Any]],
    sft_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    deepspeed_cfg: Optional[Dict[str, Any]],
    output_dir: Path,
    seed: int,
    best_model_path_file: Optional[Path] = None,
) -> Dict[str, Any]:
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    best_model_dir = output_dir / "best_model"

    collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_sequence_length=int(sft_cfg["max_sequence_length"]),
        append_eos=bool(sft_cfg.get("append_eos", True)),
    )
    use_bf16 = bool(sft_cfg.get("bf16", False)) and torch.cuda.is_available()

    save_strategy = str(sft_cfg.get("save_strategy", "epoch")).lower()
    training_kwargs: Dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "per_device_train_batch_size": int(sft_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(sft_cfg["gradient_accumulation_steps"]),
        "num_train_epochs": float(sft_cfg["num_train_epochs"]),
        "learning_rate": float(sft_cfg["learning_rate"]),
        "weight_decay": float(sft_cfg["weight_decay"]),
        "warmup_ratio": float(sft_cfg["warmup_ratio"]),
        "max_grad_norm": float(sft_cfg["max_grad_norm"]),
        "bf16": use_bf16,
        "fp16": False,
        "remove_unused_columns": False,
        "logging_steps": int(sft_cfg["logging_steps"]),
        "dataloader_num_workers": int(sft_cfg["dataloader_num_workers"]),
        "report_to": [],
        "gradient_checkpointing": bool(sft_cfg.get("gradient_checkpointing", False)),
        "seed": seed,
        "data_seed": seed,
        "save_strategy": save_strategy,
        "save_total_limit": int(sft_cfg.get("save_total_limit", 3)),
        "deepspeed": deepspeed_cfg.get("config_path")
        if deepspeed_cfg and deepspeed_cfg.get("enabled")
        else None,
        "ddp_find_unused_parameters": False,
    }
    if save_strategy == "steps":
        training_kwargs["save_steps"] = int(sft_cfg["save_steps"])

    args = _build_training_arguments(training_kwargs)
    trainer = _build_trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    callback = MathGenerationEvalCallback(
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        answer_field=str(eval_cfg["answer_field"]),
        metrics_dir=metrics_dir,
        best_model_dir=best_model_dir,
        generation_cfg={
            "eval_batch_size": int(sft_cfg.get("eval_batch_size", 8)),
            "prompt_max_length": int(
                sft_cfg.get("eval_prompt_max_length", sft_cfg["max_sequence_length"])
            ),
            "max_new_tokens": int(
                sft_cfg.get("eval_max_new_tokens", eval_cfg.get("max_tokens", 1024))
            ),
        },
    )
    trainer.add_callback(callback)

    train_result = trainer.train()
    barrier()
    callback.ensure_best_model(model=trainer.model, step=int(trainer.state.global_step))
    barrier()

    summary: Dict[str, Any] = {
        "train_metrics": {
            k: float(v)
            for k, v in train_result.metrics.items()
            if isinstance(v, (int, float))
        },
        "best_accuracy": float(callback.best_accuracy or 0.0),
        "best_step": int(callback.best_step or 0),
        "best_model_path": str(best_model_dir),
        "output_dir": str(output_dir),
        "num_train_examples": int(len(train_dataset)),
        "num_eval_examples": int(len(eval_examples)),
    }

    if trainer.is_world_process_zero():
        save_json(summary, metrics_dir / "sft_summary.json")
        if best_model_path_file is not None:
            best_model_path_file.parent.mkdir(parents=True, exist_ok=True)
            best_model_path_file.write_text(str(best_model_dir), encoding="utf-8")

    barrier()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
