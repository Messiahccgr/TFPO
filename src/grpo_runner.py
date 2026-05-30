import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, TrainerCallback

from src.config import load_config
from src.distributed import barrier
from src.curriculum import CurriculumSampler
from src.grpo_variants import resolve_grpo_variant
from src.reward import MathRewardFn
from src.tokenization import align_model_special_tokens, load_causal_lm_tokenizer
from src.utils import (
    ensure_dir,
    exclude_dataset_indices,
    resolve_attn_implementation,
    resolve_init_model_path,
    save_json,
    save_json_atomic,
    set_seed,
    setup_logger,
    terminate_process_tree,
)
from src.vllm import VLLMClient, VLLMServer
from src.vllm_multi_gpu import MultiGPUVLLMServer


logger = setup_logger("grpo_runner")
DEFAULT_MATH_QUESTION_TEMPLATE = (
    "You are a careful math solver.\n\n"
    "Solve the problem step by step. Keep the reasoning concise but sufficient.\n"
    "Use exact forms when possible. Simplify the final result.\n\n"
    "Output rules:\n"
    "1. The last non-empty line must be exactly:\n"
    "   Final Answer: \\boxed{{<answer>}}\n"
    '2. Do not output any other line starting with "Final Answer:"\n'
    "3. Do not use \\boxed{{...}} anywhere else in the response.\n"
    "4. Do not give multiple alternative answers.\n"
    "5. If the problem has multiple valid values, output all and only the valid values "
    "in a canonical order.\n"
    "6. After the final answer line, output nothing else.\n"
    "7. Do not include code, markdown fences, or execution traces.\n\n"
    "Problem:\n"
    "{problem}\n\n"
    "Solution:\n"
)


def _apply_numpy_compat() -> None:
    # NumPy 2 removed aliases that older wandb releases still import transitively via TRL.
    alias_map = {
        "float_": "float64",
        "complex_": "complex128",
    }
    for alias_name, target_name in alias_map.items():
        if not hasattr(np, alias_name) and hasattr(np, target_name):
            setattr(np, alias_name, getattr(np, target_name))


def _load_dataset_auto(name_or_path: str, split: str):
    path = Path(name_or_path)
    if path.exists() and path.is_dir():
        logger.info("Loading dataset from local path: %s (split=%s)", path, split)
        ds = load_from_disk(str(path))
        if hasattr(ds, "__getitem__") and split in ds:
            return ds[split]
        return ds
    return load_dataset(name_or_path, split=split)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_prompt(example: Dict[str, Any], data_cfg: Dict[str, Any]) -> str:
    template = data_cfg["question_template"]
    question_field = data_cfg.get("question_field")
    values = dict(example)
    if question_field is not None and question_field in example:
        values.setdefault("problem", example[question_field])
        values.setdefault("query", example[question_field])
    return template.format_map(_SafeFormatDict(values))


def _resolve_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(dtype_name).lower(), torch.float32)


def _reset_accelerate_trainer_wrappers(trainer: Any) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None:
        return

    current_model = getattr(trainer, "model_wrapped", None) or getattr(trainer, "model", None)
    if current_model is None:
        return

    try:
        unwrapped_model = accelerator.unwrap_model(
            current_model,
            keep_fp32_wrapper=False,
        )
    except TypeError:
        unwrapped_model = accelerator.unwrap_model(current_model)
        original_forward = getattr(unwrapped_model, "_original_forward", None)
        if original_forward is not None:
            unwrapped_model.forward = original_forward
            try:
                delattr(unwrapped_model, "_original_forward")
            except AttributeError:
                pass

    trainer.model = unwrapped_model
    trainer.model_wrapped = unwrapped_model


def _build_reusable_trl_trainer_cls(base_cls):
    class ReusableTRLTrainer(base_cls):
        def train(self, *args, **kwargs):
            # Curriculum mode reuses one Trainer across many outer iterations. If we
            # let Accelerator.prepare() keep stacking fp32/autocast wrappers on every
            # train() call, Qwen2 generation eventually hits RecursionError.
            _reset_accelerate_trainer_wrappers(self)
            return super().train(*args, **kwargs)

    ReusableTRLTrainer.__name__ = f"Reusable{base_cls.__name__}"
    ReusableTRLTrainer.__qualname__ = ReusableTRLTrainer.__name__
    return ReusableTRLTrainer


def _filter_kwargs_for_ctor(ctor: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(ctor)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            if parts:
                return "".join(parts)
        if "text" in completion:
            return str(completion["text"])
        return str(completion)

    if isinstance(completion, list):
        if len(completion) == 0:
            return ""
        # Chat-style list of message dicts.
        if all(isinstance(msg, dict) for msg in completion):
            parts: List[str] = []
            for msg in completion:
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(str(item["text"]))
                elif "text" in msg:
                    parts.append(str(msg["text"]))
            return "\n".join(parts)
        return " ".join(str(x) for x in completion)

    return str(completion)


class MathGRPOReward:
    def __init__(self, answer_field: str = "answer"):
        self.answer_field = answer_field
        self.reward_fn = MathRewardFn(answer_field=answer_field)
        # Older TRL trainer versions assume each reward callable exposes __name__.
        self.__name__ = self.__class__.__name__

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        del prompts
        answers = kwargs.get(self.answer_field, kwargs.get("answer"))
        if answers is None:
            raise KeyError(
                "Reward function did not receive answers. "
                f"Expected dataset column `{self.answer_field}` (or `answer`)."
            )
        if not isinstance(answers, list):
            answers = list(answers)
        if not isinstance(completions, list):
            completions = list(completions)

        if len(answers) == 0:
            return [0.0 for _ in completions]

        if len(completions) != len(answers):
            if len(completions) % len(answers) == 0:
                repeat = len(completions) // len(answers)
                answers = [a for a in answers for _ in range(repeat)]
            else:
                raise ValueError(
                    "Length mismatch in reward function: "
                    f"len(completions)={len(completions)} len(answers)={len(answers)}"
                )

        rewards: List[float] = []
        for completion, gold in zip(completions, answers):
            rewards.append(float(self.reward_fn(completion, {self.answer_field: gold})))
        return rewards


def _prepare_grpo_dataset(
    dataset,
    data_cfg: Dict[str, Any],
    max_train_samples: Optional[int],
) -> Dataset:
    answer_field = str(data_cfg.get("answer_field", "answer"))

    if max_train_samples is not None and max_train_samples < len(dataset):
        dataset = dataset.select(range(max_train_samples))

    def _map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt": _format_prompt(example, data_cfg),
            "answer": str(example.get(answer_field, "")),
        }

    mapped = dataset.map(_map_fn, remove_columns=list(dataset.column_names))
    return mapped


def _normalize_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            normalized[key] = float(value)
        elif value is None or isinstance(value, str):
            normalized[key] = value
    return normalized


def _save_trainer_checkpoint(trainer, tokenizer, checkpoint_dir: Path) -> None:
    checkpoint_dir = checkpoint_dir.resolve()
    ensure_dir(checkpoint_dir)
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))


def _cleanup_old_checkpoints(
    saved_checkpoints: List[Path],
    *,
    save_total_limit: Optional[int],
) -> None:
    if save_total_limit is None:
        return
    while len(saved_checkpoints) > save_total_limit:
        stale_checkpoint = saved_checkpoints.pop(0)
        if stale_checkpoint.exists():
            shutil.rmtree(stale_checkpoint, ignore_errors=True)
            logger.info("Removed old checkpoint to enforce save_total_limit: %s", stale_checkpoint)


def _resolve_trainer_model_kwargs(
    *,
    trainer_cls,
    model_cfg: Dict[str, Any],
    initial_model_path: Path,
) -> Dict[str, Any]:
    trainer_sig = inspect.signature(trainer_cls.__init__)
    attn_implementation, attn_source = resolve_attn_implementation(
        model_cfg.get("attn_implementation"),
        model_name_or_path=str(model_cfg["actor_name_or_path"]),
        model_path=initial_model_path,
    )

    model_load_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
    }
    if torch.cuda.is_available():
        model_load_kwargs["torch_dtype"] = _resolve_dtype(
            str(model_cfg.get("torch_dtype", "bfloat16"))
        )
    if attn_implementation is not None:
        model_load_kwargs["attn_implementation"] = attn_implementation
        if attn_source == "configured":
            logger.info("Using attn_implementation=%s", attn_implementation)
        else:
            logger.warning(
                "Auto-selected attn_implementation=%s for %s to avoid the Qwen2 SDPA masking RecursionError seen in this environment.",
                attn_implementation,
                initial_model_path,
            )

    if "model_init_kwargs" in trainer_sig.parameters:
        return {
            "model": str(initial_model_path),
            "model_init_kwargs": model_load_kwargs,
        }

    logger.info(
        "Installed TRL %s does not accept model_init_kwargs; preloading the model in the runner for compatibility.",
        trainer_cls.__name__,
    )
    return {
        "model": AutoModelForCausalLM.from_pretrained(
            str(initial_model_path),
            **model_load_kwargs,
        )
    }


def _resolve_eval_cfg(cfg: Dict[str, Any], *, algo_key: str = "grpo") -> Dict[str, Any]:
    data_cfg = cfg["data"]
    inference_cfg = cfg["inference"]
    algo_cfg = cfg.get(algo_key, {})
    algo_eval_steps = algo_cfg.get("eval_steps")
    default_eval_steps = 10 if algo_eval_steps is None else int(algo_eval_steps)
    default_eval_cfg: Dict[str, Any] = {
        "enabled": True,
        "mode": "trainer",
        "dataset_name": "./eval_data/MATH-500",
        "dataset_split": "test",
        "question_field": data_cfg.get("question_field", "problem"),
        "answer_field": data_cfg.get("answer_field", "answer"),
        "question_template": data_cfg.get(
            "question_template",
            DEFAULT_MATH_QUESTION_TEMPLATE,
        ),
        "max_samples": None,
        "eval_steps": default_eval_steps,
        "per_device_eval_batch_size": int(
            algo_cfg.get(
                "per_device_eval_batch_size",
                cfg["train"]["per_device_train_batch_size"],
            )
        ),
        "temperature": float(algo_cfg.get("temperature", inference_cfg.get("temperature", 0.6))),
        "top_p": float(algo_cfg.get("top_p", inference_cfg.get("top_p", 0.9))),
        "max_completion_length": int(
            algo_cfg.get("max_completion_length", inference_cfg.get("max_tokens", 1024))
        ),
        "max_tokens": int(
            algo_cfg.get("max_completion_length", inference_cfg.get("max_tokens", 1024))
        ),
        "stop": inference_cfg.get("stop"),
        "request_timeout_s": int(inference_cfg.get("request_timeout_s", 300)),
        "max_parallel_requests": int(inference_cfg.get("max_parallel_requests", 64)),
        "enable_pass_k": False,
        "pass_k_num_samples": 8,
        "pass_k_temperature": float(inference_cfg.get("temperature", 0.6)),
        "pass_k_top_p": float(inference_cfg.get("top_p", 0.9)),
        "pass_k_max_tokens": int(inference_cfg.get("max_tokens", 1024)),
        "vllm_gpu_ids": None,
        "keep_last_checkpoint_only": False,
        "repeat_until_new_checkpoint": False,
        "randomize_pass_1_when_pass_k_disabled": False,
        "repeat_pause_s": 0.0,
        "max_repeats_per_checkpoint": None,
    }
    user_eval_cfg = cfg.get("evaluation", {})
    if isinstance(user_eval_cfg, dict):
        default_eval_cfg.update(user_eval_cfg)
    env_eval_mode = os.environ.get("APP_EVAL_MODE")
    if env_eval_mode:
        default_eval_cfg["mode"] = env_eval_mode.strip()
    env_eval_gpu_ids = os.environ.get("APP_EVAL_VLLM_GPU_IDS")
    if env_eval_gpu_ids:
        default_eval_cfg["vllm_gpu_ids"] = env_eval_gpu_ids
    return default_eval_cfg


def _is_pass_k_enabled(eval_cfg: Dict[str, Any]) -> bool:
    return bool(eval_cfg.get("enable_pass_k", False))


def _resolve_pass_1_generation_cfg(eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    pass_k_enabled = _is_pass_k_enabled(eval_cfg)
    use_randomized_pass_1_defaults = (
        bool(eval_cfg.get("randomize_pass_1_when_pass_k_disabled", False))
        and bool(eval_cfg.get("repeat_until_new_checkpoint", False))
        and not pass_k_enabled
    )

    def _resolve_numeric(
        pass_1_key: str,
        default_key: str,
        pass_k_key: str,
        caster: Callable[[Any], Any],
    ) -> tuple[Any, str]:
        explicit_value = eval_cfg.get(pass_1_key)
        if explicit_value is not None:
            return caster(explicit_value), pass_1_key
        if use_randomized_pass_1_defaults:
            return caster(eval_cfg.get(pass_k_key, eval_cfg[default_key])), pass_k_key
        return caster(eval_cfg[default_key]), default_key

    temperature, temperature_source = _resolve_numeric(
        "pass_1_temperature",
        "temperature",
        "pass_k_temperature",
        float,
    )
    top_p, top_p_source = _resolve_numeric(
        "pass_1_top_p",
        "top_p",
        "pass_k_top_p",
        float,
    )
    max_tokens, max_tokens_source = _resolve_numeric(
        "pass_1_max_tokens",
        "max_tokens",
        "pass_k_max_tokens",
        int,
    )
    sampling_randomized = bool(float(temperature) > 0.0 or float(top_p) < 1.0)
    uses_pass_k_sampling_defaults = use_randomized_pass_1_defaults and any(
        source in {"pass_k_temperature", "pass_k_top_p", "pass_k_max_tokens"}
        for source in (temperature_source, top_p_source, max_tokens_source)
    )
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "sampling_randomized": sampling_randomized,
        "uses_pass_k_sampling_defaults": uses_pass_k_sampling_defaults,
        "temperature_source": temperature_source,
        "top_p_source": top_p_source,
        "max_tokens_source": max_tokens_source,
    }


def _add_pass_1_sampling_metrics(
    metrics: Dict[str, Any],
    pass_1_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    metrics["pass_1_temperature_used"] = float(pass_1_cfg["temperature"])
    metrics["pass_1_top_p_used"] = float(pass_1_cfg["top_p"])
    metrics["pass_1_max_tokens_used"] = int(pass_1_cfg["max_tokens"])
    metrics["pass_1_sampling_randomized"] = bool(
        pass_1_cfg["sampling_randomized"]
    )
    return metrics


def _log_eval_summary(eval_metrics: Dict[str, Any], *, prefix: str = "Eval") -> None:
    pass_at_1 = float(eval_metrics.get("pass@1", eval_metrics.get("accuracy", 0.0)))
    if bool(eval_metrics.get("pass_k_enabled", False)):
        pass_k_num_samples = int(eval_metrics.get("pass_k_num_samples", 0))
        logger.info(
            "%s pass@1=%.4f pass@%d=%.4f (%d/%d)",
            prefix,
            pass_at_1,
            pass_k_num_samples,
            float(
                eval_metrics.get(
                    f"pass@{pass_k_num_samples}",
                    eval_metrics.get("accuracy", 0.0),
                )
            ),
            int(eval_metrics.get("num_correct", 0)),
            int(eval_metrics.get("num_examples", 0)),
        )
        return
    logger.info(
        "%s pass@1=%.4f (%d/%d)",
        prefix,
        pass_at_1,
        int(eval_metrics.get("num_correct", 0)),
        int(eval_metrics.get("num_examples", 0)),
    )


def _resolve_grpo_schedule(
    *,
    cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    external_vllm_eval: bool,
) -> Dict[str, Any]:
    runtime_cfg = cfg.get("runtime", {})
    grpo_cfg = cfg.get("grpo", {})

    runtime_policy_interval = max(1, int(runtime_cfg.get("policy_save_interval", 10)))

    if grpo_cfg.get("save_steps") is None:
        save_steps = runtime_policy_interval
        save_steps_source = "runtime.policy_save_interval"
    else:
        save_steps = max(1, int(grpo_cfg["save_steps"]))
        save_steps_source = "grpo.save_steps"

    if grpo_cfg.get("save_total_limit") is None:
        save_total_limit_raw = runtime_cfg.get("save_total_limit")
        save_total_limit_source = "runtime.save_total_limit"
    else:
        save_total_limit_raw = grpo_cfg.get("save_total_limit")
        save_total_limit_source = "grpo.save_total_limit"

    if save_total_limit_raw is None:
        save_total_limit = None
    else:
        save_total_limit = int(save_total_limit_raw)
        if save_total_limit <= 0:
            raise ValueError(
                f"save_total_limit must be positive or null, got {save_total_limit!r}"
            )

    if eval_cfg.get("interval") is not None:
        eval_steps = max(1, int(eval_cfg["interval"]))
        eval_steps_source = "evaluation.interval"
    elif grpo_cfg.get("eval_steps") is not None:
        eval_steps = max(1, int(grpo_cfg["eval_steps"]))
        eval_steps_source = "grpo.eval_steps"
    else:
        eval_steps = runtime_policy_interval
        eval_steps_source = "runtime.policy_save_interval"

    if external_vllm_eval and eval_steps != save_steps:
        logger.warning(
            "evaluation.mode='external_vllm' runs on checkpoint-save events. "
            "Overriding eval interval from %d to save_steps=%d.",
            eval_steps,
            save_steps,
        )
        eval_steps = save_steps
        eval_steps_source = f"save_steps ({save_steps_source})"

    return {
        "save_steps": save_steps,
        "save_steps_source": save_steps_source,
        "save_total_limit": save_total_limit,
        "save_total_limit_source": save_total_limit_source,
        "eval_steps": eval_steps,
        "eval_steps_source": eval_steps_source,
    }


def _use_external_vllm_eval(eval_cfg: Dict[str, Any]) -> bool:
    return bool(eval_cfg.get("enabled", False)) and str(
        eval_cfg.get("mode", "trainer")
    ).strip().lower() == "external_vllm"


def _parse_gpu_ids(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) == 0:
        return None
    return [int(part) for part in parts]


def _tokenize_text_lengths(tokenizer, texts: List[str]) -> List[int]:
    if len(texts) == 0:
        return []
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def _resolve_vllm_eval_resources(
    *,
    eval_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    resolved_vllm_cfg = dict(vllm_cfg)
    gpu_ids = _parse_gpu_ids(eval_cfg.get("vllm_gpu_ids"))
    if gpu_ids is None:
        raise ValueError(
            "evaluation.mode='external_vllm' requires dedicated eval GPUs. "
            "Set evaluation.vllm_gpu_ids in config or export APP_EVAL_VLLM_GPU_IDS, "
            "for example '2,3' or '1'."
        )
    if len(gpu_ids) > 1:
        resolved_vllm_cfg["gpu_ids"] = gpu_ids
    resolved_vllm_cfg["gpu_idx"] = int(gpu_ids[0])
    return {
        "vllm_cfg": resolved_vllm_cfg,
        "num_inference_gpus": len(gpu_ids),
        "inference_gpu_ids": gpu_ids,
    }


def _evaluate_with_vllm(
    *,
    model_name_or_path: str,
    eval_dataset,
    eval_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
    num_inference_gpus: int,
    inference_gpu_ids: List[int],
    log_dir: Path,
    seed: int,
    tokenizer,
) -> Dict[str, Any]:
    pass_k_enabled = _is_pass_k_enabled(eval_cfg)
    pass_1_cfg = _resolve_pass_1_generation_cfg(eval_cfg)
    pass_k_num_samples = (
        max(1, int(eval_cfg.get("pass_k_num_samples", 8))) if pass_k_enabled else 0
    )
    total = len(eval_dataset)
    if total == 0:
        metrics = {
            "enabled": True,
            "dataset_name": str(eval_cfg["dataset_name"]),
            "dataset_split": str(eval_cfg["dataset_split"]),
            "num_examples": 0,
            "num_correct": 0,
            "num_correct_pass1": 0,
            "accuracy": 0.0,
            "pass@1": 0.0,
            "avg_completion_tokens_pass1": 0.0,
            "model": str(model_name_or_path),
            "skipped": True,
            "reason": "empty_eval_dataset",
            "pass_k_enabled": bool(pass_k_enabled),
            "pass_k_num_samples": int(pass_k_num_samples),
        }
        _add_pass_1_sampling_metrics(metrics, pass_1_cfg)
        if pass_k_enabled:
            passk_key = f"pass@{pass_k_num_samples}"
            passk_safe_key = f"pass_at_{pass_k_num_samples}"
            num_correct_passk_key = f"num_correct_pass{pass_k_num_samples}"
            avg_completion_tokens_passk_key = f"avg_completion_tokens_pass{pass_k_num_samples}"
            metrics[passk_key] = 0.0
            metrics[passk_safe_key] = 0.0
            metrics[num_correct_passk_key] = 0
            metrics[avg_completion_tokens_passk_key] = 0.0
            if pass_k_num_samples == 8:
                metrics["pass@8"] = 0.0
                metrics["pass_at_8"] = 0.0
                metrics["num_correct_pass8"] = 0
                metrics["avg_completion_tokens_pass8"] = 0.0
        return metrics

    eval_data_cfg = {
        "question_template": eval_cfg["question_template"],
        "question_field": eval_cfg.get("question_field"),
    }
    batch = [eval_dataset[i] for i in range(total)]
    prompts = [_format_prompt(example, eval_data_cfg) for example in batch]

    if num_inference_gpus > 1:
        vllm_cfg_copy = dict(vllm_cfg)
        vllm_cfg_copy["gpu_ids"] = inference_gpu_ids
        server = MultiGPUVLLMServer(
            vllm_cfg_copy,
            log_dir,
            num_gpus=num_inference_gpus,
        )
    else:
        vllm_cfg_copy = dict(vllm_cfg)
        vllm_cfg_copy["gpu_idx"] = int(inference_gpu_ids[0])
        server = VLLMServer(vllm_cfg_copy, log_dir)

    start = time.time()
    try:
        api_base = server.start(model_name_or_path=model_name_or_path, seed=seed)
        client = VLLMClient(
            api_base=api_base,
            model=str(model_name_or_path),
            request_timeout_s=int(eval_cfg["request_timeout_s"]),
            max_parallel_requests=int(eval_cfg["max_parallel_requests"]),
        )
        generated_pass1 = client.generate_batch(
            prompts=prompts,
            n=1,
            temperature=float(pass_1_cfg["temperature"]),
            top_p=float(pass_1_cfg["top_p"]),
            max_tokens=int(pass_1_cfg["max_tokens"]),
            stop=eval_cfg.get("stop"),
            seed=seed,
        )
        generated_passk = [[] for _ in range(total)]
        if pass_k_enabled:
            generated_passk = client.generate_batch(
                prompts=prompts,
                n=pass_k_num_samples,
                temperature=float(eval_cfg.get("pass_k_temperature", 0.6)),
                top_p=float(eval_cfg.get("pass_k_top_p", 0.9)),
                max_tokens=int(eval_cfg.get("pass_k_max_tokens", eval_cfg["max_tokens"])),
                stop=eval_cfg.get("stop"),
                seed=seed + 100000,
            )
    finally:
        server.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reward_fn = MathRewardFn(answer_field=str(eval_cfg["answer_field"]))
    num_correct_pass1 = 0
    num_correct_passk = 0
    num_empty_pass1 = 0
    num_empty_passk = 0
    pass1_texts: List[str] = []
    passk_texts: List[str] = []
    for example, greedy_choices, sampled_choices in zip(batch, generated_pass1, generated_passk):
        greedy_text = ""
        greedy_choice = None
        if greedy_choices:
            greedy_choice = greedy_choices[0]
            greedy_text = str(greedy_choice.get("text", ""))
        else:
            num_empty_pass1 += 1
        pass1_texts.append(greedy_text)
        if reward_fn(greedy_choice or greedy_text, example) > 0.5:
            num_correct_pass1 += 1

        if pass_k_enabled:
            sample_hit = False
            for choice in sampled_choices[:pass_k_num_samples]:
                sampled_text = str(choice.get("text", ""))
                passk_texts.append(sampled_text)
                if reward_fn(choice, example) > 0.5:
                    sample_hit = True

            missing_samples = max(0, pass_k_num_samples - len(sampled_choices))
            if missing_samples > 0:
                num_empty_passk += missing_samples
                passk_texts.extend([""] * missing_samples)

            if sample_hit:
                num_correct_passk += 1

    elapsed_s = time.time() - start
    pass1_token_lengths = _tokenize_text_lengths(tokenizer, pass1_texts)
    pass_at_1 = float(num_correct_pass1 / max(total, 1))
    metrics = {
        "enabled": True,
        "dataset_name": str(eval_cfg["dataset_name"]),
        "dataset_split": str(eval_cfg["dataset_split"]),
        "num_examples": int(total),
        "num_correct": int(num_correct_pass1),
        "num_correct_pass1": int(num_correct_pass1),
        "accuracy": pass_at_1,
        "pass@1": pass_at_1,
        "pass_at_1": pass_at_1,
        "empty_predictions": int(num_empty_pass1),
        "empty_predictions_pass1": int(num_empty_pass1),
        "avg_completion_tokens_pass1": float(
            sum(pass1_token_lengths) / max(len(pass1_token_lengths), 1)
        ),
        "eval_seconds": float(elapsed_s),
        "model": str(model_name_or_path),
        "num_inference_gpus": int(num_inference_gpus),
        "inference_gpu_ids": [int(gpu_id) for gpu_id in inference_gpu_ids],
        "pass_k_enabled": bool(pass_k_enabled),
        "pass_k_num_samples": int(pass_k_num_samples),
    }
    _add_pass_1_sampling_metrics(metrics, pass_1_cfg)
    if pass_k_enabled:
        passk_token_lengths = _tokenize_text_lengths(tokenizer, passk_texts)
        pass_at_k = float(num_correct_passk / max(total, 1))
        passk_key = f"pass@{pass_k_num_samples}"
        passk_safe_key = f"pass_at_{pass_k_num_samples}"
        num_correct_passk_key = f"num_correct_pass{pass_k_num_samples}"
        empty_predictions_passk_key = f"empty_predictions_pass{pass_k_num_samples}"
        avg_completion_tokens_passk_key = f"avg_completion_tokens_pass{pass_k_num_samples}"
        metrics[passk_key] = pass_at_k
        metrics[passk_safe_key] = pass_at_k
        metrics[num_correct_passk_key] = int(num_correct_passk)
        metrics[empty_predictions_passk_key] = int(num_empty_passk)
        metrics[avg_completion_tokens_passk_key] = float(
            sum(passk_token_lengths) / max(len(passk_token_lengths), 1)
        )
        if pass_k_num_samples == 8:
            metrics["pass@8"] = pass_at_k
            metrics["pass_at_8"] = pass_at_k
            metrics["num_correct_pass8"] = int(num_correct_passk)
            metrics["empty_predictions_pass8"] = int(num_empty_passk)
            metrics["avg_completion_tokens_pass8"] = float(
                sum(passk_token_lengths) / max(len(passk_token_lengths), 1)
            )
    return metrics


class EvalHistoryCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.history: List[Dict[str, Any]] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, control, kwargs
        metrics = metrics or {}
        serializable_metrics: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        self.history.append(
            {
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "metrics": serializable_metrics,
            }
        )
        save_json({"history": self.history}, self.path)
        return control


class ExternalVLLMEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer,
        eval_dataset,
        eval_cfg: Dict[str, Any],
        vllm_cfg: Dict[str, Any],
        num_inference_gpus: int,
        inference_gpu_ids: List[int],
        logs_dir: Path,
        metrics_dir: Path,
        seed: int,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_cfg = dict(eval_cfg)
        self.vllm_cfg = dict(vllm_cfg)
        self.num_inference_gpus = int(num_inference_gpus)
        self.inference_gpu_ids = [int(gpu_id) for gpu_id in inference_gpu_ids]
        self.logs_dir = logs_dir
        self.metrics_dir = metrics_dir
        self.seed = int(seed)
        self.history: List[Dict[str, Any]] = []
        self.repeat_until_new_checkpoint = bool(
            self.eval_cfg.get("repeat_until_new_checkpoint", False)
        )
        max_repeats = self.eval_cfg.get("max_repeats_per_checkpoint")
        self.max_repeats_per_checkpoint = (
            None if max_repeats is None else int(max_repeats)
        )
        if (
            self.max_repeats_per_checkpoint is not None
            and self.max_repeats_per_checkpoint <= 0
        ):
            raise ValueError(
                "evaluation.max_repeats_per_checkpoint must be positive or null."
            )
        self.repeat_pause_s = max(0.0, float(self.eval_cfg.get("repeat_pause_s", 0.0)))
        self.project_root = Path(__file__).resolve().parents[1]
        self.worker_dir = ensure_dir(self.metrics_dir / "external_eval_worker")
        self.worker_config_path = self.worker_dir / "worker_config.json"
        self.control_path = self.worker_dir / "control.json"
        self._worker_process: Optional[subprocess.Popen] = None
        self._submitted_generation = 0
        self._latest_step: Optional[int] = None
        self._latest_epoch: Optional[float] = None
        self._latest_checkpoint_dir: Optional[Path] = None

    def _build_continuous_submission_metrics(
        self,
        *,
        checkpoint_dir: Path,
        step: int,
        epoch: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "enabled": True,
            "mode": "external_vllm",
            "status": "submitted",
            "repeat_until_new_checkpoint": True,
            "step": int(step),
            "epoch": None if epoch is None else float(epoch),
            "checkpoint_dir": str(checkpoint_dir),
            "num_inference_gpus": int(self.num_inference_gpus),
            "inference_gpu_ids": [int(gpu_id) for gpu_id in self.inference_gpu_ids],
            "summary_path": str(self.metrics_dir / f"eval_step_{step:06d}_summary.json"),
            "runs_path": str(self.metrics_dir / f"eval_step_{step:06d}_runs.jsonl"),
            "history_path": str(self.metrics_dir / "eval_history.json"),
        }

    def evaluate_or_submit_checkpoint(
        self,
        *,
        checkpoint_dir: Path,
        step: int,
        epoch: Optional[float],
    ) -> Dict[str, Any]:
        checkpoint_dir = checkpoint_dir.resolve()
        if not checkpoint_dir.exists():
            logger.warning(
                "Skipped external vLLM eval at step=%d because checkpoint path does not exist: %s",
                step,
                checkpoint_dir,
            )
            return {
                "enabled": True,
                "mode": "external_vllm",
                "status": "skipped",
                "reason": "missing_checkpoint_dir",
                "step": int(step),
                "epoch": None if epoch is None else float(epoch),
                "checkpoint_dir": str(checkpoint_dir),
            }

        if self.repeat_until_new_checkpoint:
            self._submit_checkpoint_to_worker(
                checkpoint_dir=checkpoint_dir,
                step=step,
                epoch=epoch,
            )
            return self._build_continuous_submission_metrics(
                checkpoint_dir=checkpoint_dir,
                step=step,
                epoch=epoch,
            )

        logger.info(
            "Running external vLLM eval at step=%d using checkpoint=%s on GPUs=%s",
            step,
            checkpoint_dir,
            self.inference_gpu_ids,
        )
        metrics = _evaluate_with_vllm(
            model_name_or_path=str(checkpoint_dir),
            eval_dataset=self.eval_dataset,
            eval_cfg=self.eval_cfg,
            vllm_cfg=self.vllm_cfg,
            num_inference_gpus=self.num_inference_gpus,
            inference_gpu_ids=self.inference_gpu_ids,
            log_dir=ensure_dir(self.logs_dir / f"eval_step_{step:06d}"),
            seed=self.seed + step,
            tokenizer=self.tokenizer,
        )
        _log_eval_summary(metrics, prefix=f"External eval step={step}")
        return metrics

    def close(self) -> None:
        if self.repeat_until_new_checkpoint:
            self._stop_worker()

    def _cleanup_old_checkpoints(self, output_dir: Path, keep_path: Path) -> None:
        if not bool(self.eval_cfg.get("keep_last_checkpoint_only", False)):
            return
        for candidate in sorted(output_dir.glob("checkpoint-*")):
            if candidate.resolve() == keep_path.resolve():
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)

    def _write_worker_config(self) -> None:
        payload = {
            "eval_cfg": dict(self.eval_cfg),
            "vllm_cfg": dict(self.vllm_cfg),
            "num_inference_gpus": int(self.num_inference_gpus),
            "inference_gpu_ids": [int(gpu_id) for gpu_id in self.inference_gpu_ids],
            "logs_dir": str(self.logs_dir),
            "metrics_dir": str(self.metrics_dir),
            "seed": int(self.seed),
            "tokenizer_name_or_path": str(
                getattr(self.tokenizer, "name_or_path", "") or ""
            ),
        }
        save_json_atomic(payload, self.worker_config_path)

    def _ensure_worker_running(self) -> None:
        if self._worker_process is not None and self._worker_process.poll() is None:
            return
        if self._worker_process is not None and self._worker_process.poll() is not None:
            logger.warning(
                "Background external eval worker exited with code %s. Restarting it.",
                self._worker_process.returncode,
            )
        self._write_worker_config()
        env = os.environ.copy()
        env.setdefault("WANDB_DISABLED", "true")
        self._worker_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "src.external_vllm_eval_worker",
                "--worker-config",
                str(self.worker_config_path),
                "--control-file",
                str(self.control_path),
            ],
            cwd=str(self.project_root),
            env=env,
        )
        logger.info(
            "Started background external eval worker pid=%s",
            self._worker_process.pid,
        )

    def _submit_checkpoint_to_worker(
        self,
        *,
        checkpoint_dir: Path,
        step: int,
        epoch: Optional[float],
    ) -> None:
        self._ensure_worker_running()
        self._submitted_generation += 1
        self._latest_step = int(step)
        self._latest_epoch = float(epoch) if epoch is not None else None
        self._latest_checkpoint_dir = checkpoint_dir.resolve()
        payload = {
            "stop": False,
            "generation": int(self._submitted_generation),
            "step": int(step),
            "epoch": self._latest_epoch,
            "checkpoint_dir": str(self._latest_checkpoint_dir),
            "submitted_at": time.time(),
        }
        save_json_atomic(payload, self.control_path)
        logger.info(
            "Submitted checkpoint=%s step=%d for continuous external vLLM eval",
            checkpoint_dir,
            step,
        )

    def _stop_worker(self) -> None:
        if self._worker_process is None:
            return
        shutdown_timeout_s = float(
            self.eval_cfg.get(
                "worker_shutdown_timeout_s",
                max(300, int(self.eval_cfg.get("request_timeout_s", 300)) + 120),
            )
        )
        payload = {
            "stop": True,
            "generation": int(self._submitted_generation),
            "step": self._latest_step,
            "epoch": self._latest_epoch,
            "checkpoint_dir": str(self._latest_checkpoint_dir)
            if self._latest_checkpoint_dir is not None
            else None,
            "stopped_at": time.time(),
        }
        save_json_atomic(payload, self.control_path)
        try:
            self._worker_process.wait(timeout=shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Background external eval worker pid=%s did not stop in %.1fs; terminating it.",
                self._worker_process.pid,
                shutdown_timeout_s,
            )
            terminate_process_tree(self._worker_process, logger=logger, wait_timeout_s=30.0)
        finally:
            self._worker_process = None

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        barrier()
        if bool(getattr(state, "is_world_process_zero", True)):
            step = int(state.global_step)
            checkpoint_dir = (Path(args.output_dir) / f"checkpoint-{step}").resolve()
            if checkpoint_dir.exists():
                self.tokenizer.save_pretrained(str(checkpoint_dir))
                epoch = float(state.epoch) if state.epoch is not None else None
                metrics = self.evaluate_or_submit_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    epoch=epoch,
                )
                if not self.repeat_until_new_checkpoint:
                    payload = {
                        "step": step,
                        "epoch": epoch,
                        "checkpoint_dir": str(checkpoint_dir),
                        "metrics": metrics,
                    }
                    self.history.append(payload)
                    save_json(payload, self.metrics_dir / f"eval_step_{step:06d}.json")
                    save_json({"history": self.history}, self.metrics_dir / "eval_history.json")
                    self._cleanup_old_checkpoints(Path(args.output_dir), checkpoint_dir)
            else:
                logger.warning(
                    "Skipped external vLLM eval at step=%d because checkpoint path does not exist: %s",
                    step,
                    checkpoint_dir,
                )
        barrier()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        barrier()
        if bool(self.repeat_until_new_checkpoint) and bool(
            getattr(state, "is_world_process_zero", True)
        ):
            self.close()
        barrier()
        return control


def _build_grpo_args(
    *,
    grpo_config_cls,
    cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    schedule_cfg: Dict[str, Any],
    output_dir: Path,
    seed: int,
    manual_curriculum_loop: bool = False,
):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    inference_cfg = cfg["inference"]
    model_cfg = cfg["model"]
    deepspeed_cfg = cfg.get("deepspeed", {})
    grpo_cfg = cfg.get("grpo", {})

    use_bf16 = bool(train_cfg.get("bf16", False)) and torch.cuda.is_available()
    use_fp16 = (
        bool(train_cfg.get("fp16", False))
        and torch.cuda.is_available()
        and not use_bf16
    )
    dtype = _resolve_dtype(str(model_cfg.get("torch_dtype", "bfloat16")))
    (
        requested_loss_type,
        effective_loss_type,
        requested_importance_sampling_level,
    ) = resolve_grpo_variant(grpo_cfg)

    args_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),
        "num_train_epochs": float(grpo_cfg.get("num_train_epochs", 1.0)),
        "learning_rate": float(train_cfg["learning_rate"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "warmup_ratio": float(train_cfg["warmup_ratio"]),
        "max_grad_norm": float(train_cfg["max_grad_norm"]),
        "logging_steps": int(train_cfg["logging_steps"]),
        "dataloader_num_workers": int(train_cfg["dataloader_num_workers"]),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": seed,
        "data_seed": seed,
        "save_strategy": "no" if manual_curriculum_loop else "steps",
        "max_prompt_length": int(
            grpo_cfg.get("max_prompt_length", train_cfg.get("max_sequence_length", 2048))
        ),
        "max_completion_length": int(
            grpo_cfg.get("max_completion_length", inference_cfg.get("max_tokens", 1024))
        ),
        "num_generations": int(
            grpo_cfg.get("num_generations", inference_cfg.get("rollouts_per_question", 8))
        ),
        "temperature": float(grpo_cfg.get("temperature", inference_cfg.get("temperature", 0.6))),
        "top_p": float(grpo_cfg.get("top_p", inference_cfg.get("top_p", 0.9))),
        "beta": float(grpo_cfg.get("beta", 0.0)),
        "dataset_text_field": "prompt",
        "torch_dtype": dtype,
        "do_eval": False
        if manual_curriculum_loop
        else bool(eval_cfg.get("enabled", False)) and not _use_external_vllm_eval(eval_cfg),
    }

    if not manual_curriculum_loop:
        args_kwargs["save_steps"] = int(schedule_cfg["save_steps"])
        args_kwargs["save_total_limit"] = schedule_cfg["save_total_limit"]

    if requested_loss_type == "gspo":
        logger.info(
            "Treating loss_type='gspo' as GRPO with importance_sampling_level='sequence'."
        )
    if effective_loss_type != "grpo":
        args_kwargs["loss_type"] = effective_loss_type
    if requested_importance_sampling_level is not None:
        args_kwargs["importance_sampling_level"] = requested_importance_sampling_level

    if manual_curriculum_loop:
        manual_curriculum_max_steps_raw = grpo_cfg.get("manual_curriculum_max_steps")
        if manual_curriculum_max_steps_raw is not None:
            manual_curriculum_max_steps = int(manual_curriculum_max_steps_raw)
            if manual_curriculum_max_steps <= 0:
                raise ValueError(
                    "grpo.manual_curriculum_max_steps must be positive when provided."
                )
            args_kwargs["max_steps"] = manual_curriculum_max_steps
        else:
            # Some TRL/Transformers combinations expose a length-less curriculum dataloader
            # and require max_steps > 0. Estimate per-outer-iteration optimizer steps.
            per_device_bs = max(1, int(train_cfg["per_device_train_batch_size"]))
            grad_accum = max(1, int(train_cfg["gradient_accumulation_steps"]))
            num_questions = max(1, int(data_cfg["num_questions_per_iteration"]))
            num_train_epochs = float(grpo_cfg.get("num_train_epochs", 1.0))
            steps_per_epoch = int(np.ceil(float(num_questions) / float(per_device_bs)))
            forward_steps = max(1, int(np.ceil(num_train_epochs * float(steps_per_epoch))))
            optimizer_steps = max(1, int(np.ceil(float(forward_steps) / float(grad_accum))))
            args_kwargs["max_steps"] = optimizer_steps
        if grpo_cfg.get("max_steps") is not None:
            logger.warning(
                "Ignoring grpo.max_steps=%s because curriculum mode is driven by runtime.num_iterations outer iterations. "
                "Use grpo.manual_curriculum_max_steps to control per-iteration inner train() steps.",
                grpo_cfg["max_steps"],
            )
    elif grpo_cfg.get("max_steps") is not None:
        args_kwargs["max_steps"] = int(grpo_cfg["max_steps"])

    if (
        not manual_curriculum_loop
        and bool(eval_cfg.get("enabled", False))
        and not _use_external_vllm_eval(eval_cfg)
    ):
        eval_steps = int(schedule_cfg["eval_steps"])
        args_kwargs["eval_steps"] = eval_steps
        # Keep compatibility across HF/TRL versions.
        args_kwargs["evaluation_strategy"] = "steps"
        args_kwargs["eval_strategy"] = "steps"
        args_kwargs["per_device_eval_batch_size"] = int(
            eval_cfg.get(
                "per_device_eval_batch_size",
                train_cfg["per_device_train_batch_size"],
            )
        )

    if deepspeed_cfg.get("enabled", False):
        args_kwargs["deepspeed"] = str(deepspeed_cfg["config_path"])

    filtered = _filter_kwargs_for_ctor(grpo_config_cls.__init__, args_kwargs)
    dropped = sorted(set(args_kwargs.keys()) - set(filtered.keys()))
    if effective_loss_type != "grpo" and "loss_type" in dropped:
        raise RuntimeError(
            "Installed TRL GRPOConfig does not accept `loss_type`, so the requested "
            f"algorithm variant {requested_loss_type!r} cannot be selected safely. "
            "Upgrade/downgrade TRL to a version that explicitly supports this loss_type."
        )
    if requested_importance_sampling_level is not None and "importance_sampling_level" in dropped:
        raise RuntimeError(
            "Installed TRL GRPOConfig does not accept `importance_sampling_level`, so GSPO-style "
            "sequence-level importance sampling cannot be enabled safely. "
            "Upgrade/downgrade TRL to a version that explicitly supports this argument."
        )
    if dropped:
        logger.info("Dropped unsupported GRPOConfig args for installed TRL: %s", dropped)
    return grpo_config_cls(**filtered)


def run(cfg: Dict[str, Any]) -> None:
    seed = int(cfg["seed"])
    set_seed(seed)

    base_output_dir = Path(cfg["output_dir"]).resolve()
    run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = ensure_dir(
        base_output_dir.with_name(f"{base_output_dir.name}_grpo_{run_timestamp}")
    )
    metrics_dir = ensure_dir(output_dir / "metrics")
    logs_dir = ensure_dir(output_dir / "logs")

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    runtime_cfg = cfg.get("runtime", {})
    curriculum_cfg = cfg.get("curriculum", {})
    curriculum_enabled = isinstance(curriculum_cfg, dict) and bool(
        curriculum_cfg.get("enabled", False)
    )
    grpo_cfg = cfg.get("grpo", {})
    eval_cfg = _resolve_eval_cfg(cfg, algo_key="grpo")
    external_vllm_eval = _use_external_vllm_eval(eval_cfg)
    schedule_cfg = _resolve_grpo_schedule(
        cfg=cfg,
        eval_cfg=eval_cfg,
        external_vllm_eval=external_vllm_eval,
    )
    eval_cfg["eval_steps"] = int(schedule_cfg["eval_steps"])
    vllm_eval_resources = None
    if external_vllm_eval:
        vllm_eval_resources = _resolve_vllm_eval_resources(
            eval_cfg=eval_cfg,
            vllm_cfg=cfg.get("vllm", {}),
        )
    project_root = Path(cfg["_meta"]["project_root"]).resolve()

    initial_model_path = resolve_init_model_path(
        str(model_cfg["actor_name_or_path"]),
        project_root=project_root,
    )
    tokenizer_name = model_cfg.get("tokenizer_name_or_path") or model_cfg["actor_name_or_path"]
    initial_tokenizer_path = resolve_init_model_path(
        str(tokenizer_name),
        project_root=project_root,
    )

    logger.info("Experiment=%s | output_dir=%s", cfg["exp_name"], output_dir)
    logger.info("Resolved model path: %s", initial_model_path)
    logger.info("Resolved tokenizer path: %s", initial_tokenizer_path)
    logger.info(
        "Schedule | save_every=%d (%s) | eval_every=%d (%s) | save_total_limit=%s (%s)",
        int(schedule_cfg["save_steps"]),
        str(schedule_cfg["save_steps_source"]),
        int(schedule_cfg["eval_steps"]),
        str(schedule_cfg["eval_steps_source"]),
        "null"
        if schedule_cfg["save_total_limit"] is None
        else int(schedule_cfg["save_total_limit"]),
        str(schedule_cfg["save_total_limit_source"]),
    )
    if curriculum_enabled:
        logger.info(
            "Curriculum mode enabled | num_iterations=%d | questions_per_iteration=%d",
            int(runtime_cfg["num_iterations"]),
            int(data_cfg["num_questions_per_iteration"]),
        )

    tokenizer = load_causal_lm_tokenizer(
        str(initial_tokenizer_path),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    _apply_numpy_compat()
    from trl import GRPOConfig, GRPOTrainer

    ReusableGRPOTrainer = _build_reusable_trl_trainer_cls(GRPOTrainer)

    max_train_samples = grpo_cfg.get("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    if curriculum_enabled and max_train_samples is not None:
        logger.warning(
            "Ignoring grpo.max_train_samples=%d because curriculum mode uses data.num_questions_per_iteration for outer-loop sampling.",
            max_train_samples,
        )
        max_train_samples = None

    train_dataset = None
    curriculum_sampler: Optional[CurriculumSampler] = None
    if curriculum_enabled:
        curriculum_sampler = CurriculumSampler(
            curriculum_cfg=curriculum_cfg,
            data_cfg=data_cfg,
            num_iterations=int(runtime_cfg["num_iterations"]),
            project_root=project_root,
            seed=seed,
        )
    else:
        dataset = _load_dataset_auto(str(data_cfg["dataset_name"]), split=str(data_cfg["dataset_split"]))
        excluded_question_indices = data_cfg.get("excluded_question_indices") or []
        if len(excluded_question_indices) > 0:
            dataset, num_excluded = exclude_dataset_indices(
                dataset,
                excluded_question_indices,
            )
            logger.info(
                "Excluded %d dataset rows via data.excluded_question_indices",
                num_excluded,
            )
        if data_cfg.get("max_dataset_size") is not None:
            max_size = int(data_cfg["max_dataset_size"])
            if max_size < len(dataset):
                dataset = dataset.shuffle(seed=seed).select(range(max_size))
        logger.info("Loaded training dataset size: %d", len(dataset))

        train_dataset = _prepare_grpo_dataset(
            dataset=dataset,
            data_cfg=data_cfg,
            max_train_samples=max_train_samples,
        )
        logger.info("Prepared GRPO dataset size: %d", len(train_dataset))

    eval_dataset = None
    if bool(eval_cfg.get("enabled", False)):
        eval_raw = _load_dataset_auto(
            str(eval_cfg["dataset_name"]),
            split=str(eval_cfg["dataset_split"]),
        )
        if eval_cfg.get("max_samples") is not None:
            max_eval_samples = int(eval_cfg["max_samples"])
            if max_eval_samples < len(eval_raw):
                eval_raw = eval_raw.select(range(max_eval_samples))

        if external_vllm_eval:
            eval_dataset = eval_raw
        else:
            eval_data_cfg = {
                "question_template": eval_cfg["question_template"],
                "question_field": eval_cfg.get("question_field"),
                "answer_field": eval_cfg.get("answer_field", "answer"),
            }
            eval_dataset = _prepare_grpo_dataset(
                dataset=eval_raw,
                data_cfg=eval_data_cfg,
                max_train_samples=None,
            )
        logger.info(
            "Loaded eval dataset size: %d | %s every %d steps",
            len(eval_dataset),
            "external-vllm eval" if external_vllm_eval else "auto-eval",
            int(eval_cfg.get("eval_steps", 10)),
        )
        if external_vllm_eval:
            logger.info(
                "External vLLM eval will use GPUs=%s and keep_last_checkpoint_only=%s",
                vllm_eval_resources["inference_gpu_ids"],
                bool(eval_cfg.get("keep_last_checkpoint_only", False)),
            )
            if bool(eval_cfg.get("repeat_until_new_checkpoint", False)):
                pass_1_cfg = _resolve_pass_1_generation_cfg(eval_cfg)
                logger.info(
                    "Continuous external eval enabled: repeat current checkpoint until a newer one is saved."
                )
                if bool(pass_1_cfg["uses_pass_k_sampling_defaults"]):
                    logger.info(
                        "Continuous external eval will randomize pass@1 when pass@k is disabled "
                        "(temperature=%.3f, top_p=%.3f, max_tokens=%d).",
                        float(pass_1_cfg["temperature"]),
                        float(pass_1_cfg["top_p"]),
                        int(pass_1_cfg["max_tokens"]),
                    )
                elif (
                    not _is_pass_k_enabled(eval_cfg)
                    and not bool(pass_1_cfg["sampling_randomized"])
                ):
                    logger.warning(
                        "Continuous external eval is configured with deterministic pass@1 "
                        "and pass@k disabled, so repeated runs will usually be identical."
                    )

    grpo_args = _build_grpo_args(
        grpo_config_cls=GRPOConfig,
        cfg=cfg,
        eval_cfg=eval_cfg,
        schedule_cfg=schedule_cfg,
        output_dir=output_dir,
        seed=seed,
        manual_curriculum_loop=curriculum_enabled,
    )
    reward_adapter = MathGRPOReward(answer_field="answer")

    trainer_sig = inspect.signature(ReusableGRPOTrainer.__init__)
    common_trainer_kwargs: Dict[str, Any] = {
        "args": grpo_args,
    }
    if "reward_funcs" in trainer_sig.parameters:
        common_trainer_kwargs["reward_funcs"] = [reward_adapter]
    elif "reward_func" in trainer_sig.parameters:
        common_trainer_kwargs["reward_func"] = reward_adapter
    else:
        raise RuntimeError(
            "Installed TRL GRPOTrainer does not expose reward_func(s) argument. "
            "Please upgrade/downgrade TRL to a GRPO-compatible version."
        )

    if "processing_class" in trainer_sig.parameters:
        common_trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        common_trainer_kwargs["tokenizer"] = tokenizer

    common_trainer_kwargs.update(
        _resolve_trainer_model_kwargs(
            trainer_cls=ReusableGRPOTrainer,
            model_cfg=model_cfg,
            initial_model_path=initial_model_path,
        )
    )

    if curriculum_enabled:
        if curriculum_sampler is None:
            raise RuntimeError("Curriculum sampler was not initialized.")

        trainer = None
        external_eval_manager: Optional[ExternalVLLMEvalCallback] = None
        history: List[Dict[str, Any]] = []
        saved_checkpoints: List[Path] = []
        last_train_metrics: Dict[str, Any] = {}
        total_train_examples = 0
        num_iterations = int(runtime_cfg["num_iterations"])
        start_iteration = int(runtime_cfg.get("start_iteration", 0))
        if start_iteration < 0 or start_iteration >= num_iterations:
            raise ValueError(
                f"runtime.start_iteration must be in [0, {num_iterations - 1}], "
                f"got {start_iteration!r}"
            )
        if start_iteration > 0:
            logger.info(
                "Curriculum resume enabled | starting from outer iteration %d/%d",
                start_iteration + 1,
                num_iterations,
            )

        continuous_external_vllm_eval = bool(
            eval_dataset is not None
            and external_vllm_eval
            and eval_cfg.get("repeat_until_new_checkpoint", False)
        )
        if continuous_external_vllm_eval:
            assert vllm_eval_resources is not None
            external_eval_manager = ExternalVLLMEvalCallback(
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                eval_cfg=eval_cfg,
                vllm_cfg=vllm_eval_resources["vllm_cfg"],
                num_inference_gpus=int(vllm_eval_resources["num_inference_gpus"]),
                inference_gpu_ids=list(vllm_eval_resources["inference_gpu_ids"]),
                logs_dir=logs_dir,
                metrics_dir=metrics_dir,
                seed=seed,
            )
            if schedule_cfg["save_total_limit"] is not None:
                logger.info(
                    "Deferring save_total_limit checkpoint cleanup until the continuous external eval worker stops."
                )

        try:
            for iteration in range(start_iteration, num_iterations):
                curriculum_sample = curriculum_sampler.sample(iteration)
                curriculum_info = dict(curriculum_sample.info)
                logger.info(
                    "Curriculum | phase=%s stage=%s sampling_mode=%s source_counts=%s group_counts=%s subset_sizes=%s",
                    curriculum_info["phase"],
                    curriculum_info["stage"],
                    curriculum_info.get("sampling_mode"),
                    curriculum_info["source_counts"],
                    curriculum_info.get("source_group_counts"),
                    curriculum_info["source_subset_sizes"],
                )

                if len(curriculum_sample.examples) == 0:
                    payload = {
                        "iteration": iteration,
                        "outer_iteration": iteration + 1,
                        "skipped": True,
                        "curriculum": curriculum_info,
                    }
                    history.append(payload)
                    save_json(payload, metrics_dir / f"iter_{iteration:04d}.json")
                    save_json({"history": history}, metrics_dir / "history.json")
                    logger.warning("Skipped outer iteration %d because no examples were sampled.", iteration + 1)
                    continue

                iter_raw_dataset = Dataset.from_list(curriculum_sample.examples)
                iter_train_dataset = _prepare_grpo_dataset(
                    dataset=iter_raw_dataset,
                    data_cfg=data_cfg,
                    max_train_samples=max_train_samples,
                )
                total_train_examples += int(len(iter_train_dataset))

                if trainer is None:
                    trainer_kwargs = dict(common_trainer_kwargs)
                    trainer_kwargs["train_dataset"] = iter_train_dataset
                    if eval_dataset is not None and not external_vllm_eval:
                        trainer_kwargs["eval_dataset"] = eval_dataset
                    trainer = ReusableGRPOTrainer(**trainer_kwargs)
                    align_model_special_tokens(trainer.model, tokenizer)
                else:
                    trainer.train_dataset = iter_train_dataset
                    if eval_dataset is not None and not external_vllm_eval:
                        trainer.eval_dataset = eval_dataset

                logger.info(
                    "Outer iteration %d/%d | sampled_questions=%d | train_examples=%d",
                    iteration + 1,
                    num_iterations,
                    len(curriculum_sample.question_ids),
                    len(iter_train_dataset),
                )
                train_result = trainer.train()
                train_metrics = _normalize_metrics(train_result.metrics)
                last_train_metrics = train_metrics

                is_final_iteration = iteration == (num_iterations - 1)
                should_save = ((iteration + 1) % int(schedule_cfg["save_steps"]) == 0) or is_final_iteration
                should_eval = eval_dataset is not None and (
                    ((iteration + 1) % int(schedule_cfg["eval_steps"]) == 0) or is_final_iteration
                )

                checkpoint_dir: Optional[Path] = None
                if should_save:
                    checkpoint_dir = output_dir / f"checkpoint-{iteration + 1}"
                    _save_trainer_checkpoint(trainer, tokenizer, checkpoint_dir)
                    saved_checkpoints.append(checkpoint_dir)
                    if not continuous_external_vllm_eval:
                        _cleanup_old_checkpoints(
                            saved_checkpoints,
                            save_total_limit=schedule_cfg["save_total_limit"],
                        )

                eval_metrics: Dict[str, Any] = {}
                if should_eval:
                    if checkpoint_dir is None:
                        checkpoint_dir = output_dir / f"checkpoint-{iteration + 1}"
                        _save_trainer_checkpoint(trainer, tokenizer, checkpoint_dir)
                        if checkpoint_dir not in saved_checkpoints:
                            saved_checkpoints.append(checkpoint_dir)
                            if not continuous_external_vllm_eval:
                                _cleanup_old_checkpoints(
                                    saved_checkpoints,
                                    save_total_limit=schedule_cfg["save_total_limit"],
                                )

                    if external_vllm_eval:
                        assert checkpoint_dir is not None
                        if continuous_external_vllm_eval:
                            assert external_eval_manager is not None
                            logger.info(
                                "Submitting checkpoint for continuous external vLLM eval at outer_iteration=%d using checkpoint=%s on GPUs=%s",
                                iteration + 1,
                                checkpoint_dir,
                                vllm_eval_resources["inference_gpu_ids"],
                            )
                            eval_metrics = external_eval_manager.evaluate_or_submit_checkpoint(
                                checkpoint_dir=checkpoint_dir,
                                step=iteration + 1,
                                epoch=None,
                            )
                        else:
                            logger.info(
                                "Running external vLLM eval at outer_iteration=%d using checkpoint=%s on GPUs=%s",
                                iteration + 1,
                                checkpoint_dir,
                                vllm_eval_resources["inference_gpu_ids"],
                            )
                            eval_metrics = _evaluate_with_vllm(
                                model_name_or_path=str(checkpoint_dir),
                                eval_dataset=eval_dataset,
                                eval_cfg=eval_cfg,
                                vllm_cfg=vllm_eval_resources["vllm_cfg"],
                                num_inference_gpus=int(vllm_eval_resources["num_inference_gpus"]),
                                inference_gpu_ids=list(vllm_eval_resources["inference_gpu_ids"]),
                                log_dir=ensure_dir(logs_dir / f"eval_iter_{iteration + 1:04d}"),
                                seed=seed + 17 + iteration,
                                tokenizer=tokenizer,
                            )
                            _log_eval_summary(
                                eval_metrics,
                                prefix=f"External eval outer_iteration={iteration + 1}",
                            )
                    else:
                        eval_metrics = _normalize_metrics(trainer.evaluate(eval_dataset=eval_dataset))

                    save_json(
                        {
                            "iteration": iteration,
                            "outer_iteration": iteration + 1,
                            "checkpoint_dir": str(checkpoint_dir),
                            "metrics": eval_metrics,
                        },
                        metrics_dir / f"eval_step_{iteration + 1:06d}.json",
                    )

                payload = {
                    "iteration": iteration,
                    "outer_iteration": iteration + 1,
                    "curriculum": curriculum_info,
                    "num_sampled_questions": int(len(curriculum_sample.question_ids)),
                    "num_train_examples": int(len(iter_train_dataset)),
                    "train": train_metrics,
                    "evaluation": eval_metrics,
                    "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
                }
                history.append(payload)
                save_json(payload, metrics_dir / f"iter_{iteration:04d}.json")
                save_json({"history": history}, metrics_dir / "history.json")
        finally:
            if external_eval_manager is not None:
                external_eval_manager.close()

        if trainer is None:
            raise RuntimeError("Curriculum mode did not produce any trainable dataset.")

        if continuous_external_vllm_eval:
            _cleanup_old_checkpoints(
                saved_checkpoints,
                save_total_limit=schedule_cfg["save_total_limit"],
            )

        final_model_dir = output_dir / "final_model"
        _save_trainer_checkpoint(trainer, tokenizer, final_model_dir)
        save_json(
            {
                "train": last_train_metrics,
                "output_dir": str(output_dir),
                "model_source": str(initial_model_path),
                "start_iteration": int(start_iteration),
                "num_iterations": int(num_iterations),
                "total_train_examples": int(total_train_examples),
            },
            metrics_dir / "final_metrics.json",
        )
        logger.info(
            "GRPO curriculum training finished. Metrics saved to %s",
            metrics_dir / "final_metrics.json",
        )
        return

    trainer_kwargs = dict(common_trainer_kwargs)
    trainer_kwargs["train_dataset"] = train_dataset
    if eval_dataset is not None and not external_vllm_eval:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = ReusableGRPOTrainer(**trainer_kwargs)
    align_model_special_tokens(trainer.model, tokenizer)
    if eval_dataset is not None and not external_vllm_eval:
        trainer.add_callback(EvalHistoryCallback(metrics_dir / "eval_history.json"))
    if eval_dataset is not None and external_vllm_eval:
        trainer.add_callback(
            ExternalVLLMEvalCallback(
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                eval_cfg=eval_cfg,
                vllm_cfg=vllm_eval_resources["vllm_cfg"],
                num_inference_gpus=int(vllm_eval_resources["num_inference_gpus"]),
                inference_gpu_ids=list(vllm_eval_resources["inference_gpu_ids"]),
                logs_dir=logs_dir,
                metrics_dir=metrics_dir,
                seed=seed + 17,
            )
        )
    train_result = trainer.train()

    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    metrics = _normalize_metrics(train_result.metrics)
    save_json(
        {
            "train": metrics,
            "output_dir": str(output_dir),
            "model_source": str(initial_model_path),
            "num_train_examples": int(len(train_dataset)),
        },
        metrics_dir / "final_metrics.json",
    )
    logger.info("GRPO training finished. Metrics saved to %s", metrics_dir / "final_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help='Comma-separated jsonnet configs, e.g. "a.jsonnet,b.jsonnet"',
    )
    args = parser.parse_args()
    cfg = load_config(args.configs)
    run(cfg)


if __name__ == "__main__":
    main()
