import argparse
import math
import os
import random
import shutil
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk


def _load_dataset_auto(name_or_path: str, split: str):
    """Load dataset from local disk if path exists, otherwise from HuggingFace Hub."""
    path = Path(name_or_path)
    if path.exists() and path.is_dir():
        logger.info("Loading dataset from local path: %s (split=%s)", path, split)
        # Arrow format saved by `save_to_disk`
        if (path / "dataset_info.json").exists() or (path / "state.json").exists():
            ds = load_from_disk(str(path))
            if hasattr(ds, "__getitem__") and split in ds:
                return ds[split]
            return ds

        # Check for split subdirectory (e.g. ./data/MetaMathQA/train/)
        split_dir = path / split
        if split_dir.exists() and split_dir.is_dir():
            if (split_dir / "dataset_info.json").exists() or (split_dir / "state.json").exists():
                return load_from_disk(str(split_dir))
            parquet_files = list(split_dir.glob("*.parquet"))
            if parquet_files:
                return load_dataset("parquet", data_files=[str(f) for f in sorted(parquet_files)], split="train")

        # Check for data/ subdirectory with split-prefixed parquet files
        # e.g. data/train-00000-of-00001.parquet
        data_dir = path / "data"
        if data_dir.exists() and data_dir.is_dir():
            split_parquets = sorted(data_dir.glob(f"{split}-*.parquet"))
            if split_parquets:
                return load_dataset("parquet", data_files=[str(f) for f in split_parquets], split="train")
            # Fallback: any parquet in data/
            all_parquets = sorted(data_dir.glob("*.parquet"))
            if all_parquets:
                return load_dataset("parquet", data_files=[str(f) for f in all_parquets], split="train")

        # Parquet files directly in the directory
        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            return load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")

        # JSON files directly in the directory
        json_files = sorted(path.glob("*.json"))
        if json_files:
            return load_dataset("json", data_files=[str(f) for f in json_files], split="train")

        # Fallback to load_from_disk
        ds = load_from_disk(str(path))
        if hasattr(ds, "__getitem__") and split in ds:
            return ds[split]
        return ds
    return load_dataset(name_or_path, split=split)
from transformers import AutoModelForCausalLM

from src.config import load_config
from src.curriculum import CurriculumSampler
from src.distributed import barrier, get_dist_state
from src.reward import MathRewardFn, extract_gold_answer_text, extract_pred_answer
from src.rejudge import Rejudger
from src.teacher import ClosedFormTeacherBuilder
from src.tokenization import align_model_special_tokens, load_causal_lm_tokenizer
from src.training import (
    create_trainer,
    train_one_iteration,
)
from src.utils import (
    ensure_dir,
    exclude_dataset_indices,
    resolve_attn_implementation,
    resolve_init_model_path,
    save_json,
    set_seed,
    setup_logger,
)
from src.vllm import VLLMClient, VLLMServer
from src.vllm_multi_gpu import MultiGPUVLLMServer


logger = setup_logger("runner")


class QuestionSampler:
    def __init__(
        self,
        dataset_size: int,
        num_iterations: int,
        num_questions_per_iteration: int,
        sample_with_replacement: bool,
        shuffle_on_each_iteration: bool,
        seed: int,
    ):
        self.dataset_size = dataset_size
        self.num_iterations = num_iterations
        self.num_questions_per_iteration = num_questions_per_iteration
        self.sample_with_replacement = sample_with_replacement
        self.shuffle_on_each_iteration = shuffle_on_each_iteration
        self.seed = seed

        self._expanded_indices: List[int] = []
        if not sample_with_replacement:
            repeats = (
                math.ceil(
                    (num_iterations * num_questions_per_iteration) / max(dataset_size, 1)
                )
                + 1
            )
            for i in range(repeats):
                indices = list(range(dataset_size))
                if shuffle_on_each_iteration:
                    rng = random.Random(seed + i)
                    rng.shuffle(indices)
                self._expanded_indices.extend(indices)

    def sample(self, iteration: int) -> List[int]:
        n = self.num_questions_per_iteration
        if self.sample_with_replacement:
            indices = list(range(self.dataset_size))
            if self.shuffle_on_each_iteration:
                rng = random.Random(self.seed + iteration)
                rng.shuffle(indices)
            return indices[:n]
        start = iteration * n
        end = start + n
        return self._expanded_indices[start:end]


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


def _format_prompt(example: Dict[str, Any], data_cfg: Dict[str, Any]) -> str:
    class _SafeFormatDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    template = data_cfg["question_template"]
    question_field = data_cfg.get("question_field")
    values = dict(example)
    if question_field is not None and question_field in example:
        values.setdefault("problem", example[question_field])
        values.setdefault("query", example[question_field])
    try:
        return template.format_map(_SafeFormatDict(values))
    except KeyError as exc:
        raise KeyError(
            f"Prompt template missing key {exc} for example keys={list(example.keys())}"
        ) from exc


def _resolve_deepspeed_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    deepspeed_cfg = dict(cfg["deepspeed"])
    if not deepspeed_cfg.get("enabled", False):
        return deepspeed_cfg
    path = Path(deepspeed_cfg["config_path"])
    if not path.is_absolute():
        project_root = Path(cfg["_meta"]["project_root"])
        candidates = [
            Path.cwd() / path,
            project_root / path,
            project_root / "configs" / path,
            project_root / path.name,
            project_root / "configs" / path.name,
        ]
        uniq_candidates = []
        for candidate in candidates:
            if candidate not in uniq_candidates:
                uniq_candidates.append(candidate)
        found = None
        for candidate in uniq_candidates:
            if candidate.exists():
                found = candidate.resolve()
                break
        if found is None:
            raise FileNotFoundError(
                f"DeepSpeed config file not found. Tried: {uniq_candidates}"
            )
        path = found
    deepspeed_cfg["config_path"] = str(path)
    return deepspeed_cfg


def _build_batch_rollouts(
    batch,
    prompts: List[str],
    generated: List[List[Dict[str, Any]]],
    reward_fn: MathRewardFn,
    data_cfg: Dict[str, Any],
    sampled_indices: Optional[List[int]] = None,
    sampled_question_ids: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    qid_field = data_cfg.get("question_id_field")
    for local_idx, (example, prompt, choices) in enumerate(zip(batch, prompts, generated)):
        if qid_field is not None and qid_field in example:
            question_idx = example[qid_field]
        elif sampled_question_ids is not None:
            question_idx = sampled_question_ids[local_idx]
        elif sampled_indices is not None:
            question_idx = int(sampled_indices[local_idx])
        else:
            question_idx = int(local_idx)

        rollouts = []
        for choice in choices:
            completion = str(choice.get("text", ""))
            reward_score = reward_fn.score_completion(choice, example)
            rollout_entry = {
                "response_text": completion,
                "reward": float(reward_score.total_reward),
                "finish_reason": choice.get("finish_reason"),
                "answer_correct": bool(reward_score.is_answer_correct),
                "answer_only_text": reward_score.answer_only_text,
                "format_reward": float(reward_score.format_reward),
                "answer_reward": float(reward_score.answer_reward),
                "length_reward": float(reward_score.length_reward),
            }
            token_logprobs = choice.get("token_logprobs")
            if token_logprobs is not None:
                # Stored as fp32 list; downstream re-tokenization (teacher.py)
                # verifies length alignment with response_token_ids and falls back
                # to None (ratio=1) when vLLM string-tokens disagree with the HF
                # tokenizer (handled in the GRPO example builder).
                rollout_entry["token_logprobs"] = [float(x) for x in token_logprobs]
            rollouts.append(rollout_entry)

        gold_answer_text = ""
        answer_field = data_cfg.get("answer_field")
        if answer_field is not None and answer_field in example:
            gold_answer_text = extract_gold_answer_text(
                str(example.get(answer_field, ""))
            )
        output.append(
            {
                "question_idx": question_idx,
                "query_text": prompt,
                "gold_answer_text": gold_answer_text,
                "rollouts": rollouts,
            }
        )
    return output


def _save_jsonl_examples(examples: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write("\n")


def _load_jsonl_examples(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Teacher examples file not found: {path}")
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _resolve_eval_cfg(
    cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    inference_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    default_eval_cfg: Dict[str, Any] = {
        "enabled": True,
        "dataset_name": "./eval_data/MATH-500",
        "dataset_split": "test",
        "question_field": data_cfg.get("question_field", "problem"),
        "answer_field": data_cfg.get("answer_field", "answer"),
        "question_template": data_cfg.get("question_template"),
        "max_samples": None,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(inference_cfg.get("max_tokens", 1024)),
        "stop": inference_cfg.get("stop"),
        "request_timeout_s": int(inference_cfg.get("request_timeout_s", 300)),
        "max_parallel_requests": int(inference_cfg.get("max_parallel_requests", 64)),
        "enable_pass_k": False,
        "pass_k_num_samples": 8,
        "pass_k_temperature": float(inference_cfg.get("temperature", 0.6)),
        "pass_k_top_p": float(inference_cfg.get("top_p", 0.9)),
        "pass_k_max_tokens": int(inference_cfg.get("max_tokens", 1024)),
        "interval": None,
    }
    user_eval_cfg = cfg.get("evaluation", {})
    if isinstance(user_eval_cfg, dict):
        default_eval_cfg.update(user_eval_cfg)
    return default_eval_cfg


def _should_run_evaluation(
    iteration: int,
    eval_cfg: Dict[str, Any],
) -> bool:
    if not bool(eval_cfg.get("enabled", False)):
        return False
    interval = max(1, int(eval_cfg["interval"]))
    return (iteration + 1) % interval == 0


def _is_pass_k_enabled(eval_cfg: Dict[str, Any]) -> bool:
    return bool(eval_cfg.get("enable_pass_k", False))


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


def _replace_checkpoint_dir(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir, ignore_errors=True)
    shutil.copytree(src_dir, dst_dir)


def _eval_details_dir(
    *,
    eval_outcomes_dir: Path,
    iteration: int,
    eval_cfg: Dict[str, Any],
) -> Path:
    dataset_name = Path(str(eval_cfg["dataset_name"])).name or "eval"
    dataset_split = str(eval_cfg["dataset_split"])
    raw_key = f"{dataset_name}_{dataset_split}".lower()
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in raw_key).strip("_")
    return eval_outcomes_dir / f"iter_{iteration:04d}" / (safe_key or "eval")


def _unwrap_trainer_model(trainer):
    model = getattr(trainer, "model", None)
    accelerator = getattr(trainer, "accelerator", None)
    if model is None or accelerator is None:
        return model
    try:
        return accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    except TypeError:
        return accelerator.unwrap_model(model)
    except Exception:
        return model


def _compute_bad_action_stats(
    *,
    trainer,
    bad_action_examples: List[Dict[str, Any]],
    pad_token_id: int,
    batch_size: int,
    max_edges: Optional[int] = None,
) -> Dict[str, Any]:
    total_edges = len(bad_action_examples)
    if total_edges == 0:
        return {
            "p_bad": None,
            "num_bad_edges": 0,
            "num_bad_edges_total": 0,
            "num_bad_edges_evaluated": 0,
            "bad_action_stats_capped": False,
        }

    examples = bad_action_examples
    capped = False
    if max_edges is not None and int(max_edges) > 0 and total_edges > int(max_edges):
        examples = bad_action_examples[: int(max_edges)]
        capped = True

    model = _unwrap_trainer_model(trainer)
    if model is None:
        raise RuntimeError("Unable to access trainer model for bad-action statistics.")
    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()

    prob_sum = 0.0
    prob_count = 0
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            max_len = max(len(ex["query_token_ids"]) for ex in chunk)
            input_ids = torch.full(
                (len(chunk), max_len),
                int(pad_token_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(chunk), max_len),
                dtype=torch.long,
                device=device,
            )
            negative_token_ids = torch.tensor(
                [int(ex["negative_token_id"]) for ex in chunk],
                dtype=torch.long,
                device=device,
            )
            for row_idx, ex in enumerate(chunk):
                ids = torch.tensor(
                    [int(tok) for tok in ex["query_token_ids"]],
                    dtype=torch.long,
                    device=device,
                )
                input_ids[row_idx, : ids.numel()] = ids
                attention_mask[row_idx, : ids.numel()] = 1

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_indices = torch.arange(len(chunk), device=device)
            last_token_index = attention_mask.sum(dim=1).clamp(min=1) - 1
            next_token_logits = outputs.logits[batch_indices, last_token_index]
            next_token_log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)
            probs = torch.exp(
                next_token_log_probs.gather(
                    dim=-1,
                    index=negative_token_ids.unsqueeze(-1),
                ).squeeze(-1)
            )
            prob_sum += float(probs.sum().detach().cpu())
            prob_count += int(probs.numel())

    if was_training:
        model.train()

    return {
        "p_bad": float(prob_sum / max(prob_count, 1)),
        "num_bad_edges": int(total_edges),
        "num_bad_edges_total": int(total_edges),
        "num_bad_edges_evaluated": int(prob_count),
        "bad_action_stats_capped": bool(capped),
    }


def _log_eval_summary(eval_metrics: Dict[str, Any]) -> None:
    pass_at_1 = float(eval_metrics.get("pass@1", eval_metrics.get("accuracy", 0.0)))
    if bool(eval_metrics.get("pass_k_enabled", False)):
        pass_k_num_samples = int(eval_metrics.get("pass_k_num_samples", 0))
        logger.info(
            "Eval pass@1=%.4f pass@%d=%.4f (%d/%d)",
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
        "Eval pass@1=%.4f (%d/%d)",
        pass_at_1,
        int(eval_metrics.get("num_correct", 0)),
        int(eval_metrics.get("num_examples", 0)),
    )


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
    details_dir: Optional[Path] = None,
    rejudger: Optional[Rejudger] = None,
) -> Dict[str, Any]:
    pass_k_enabled = _is_pass_k_enabled(eval_cfg)
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
        server = VLLMServer(vllm_cfg, log_dir)

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
            temperature=float(eval_cfg.get("pass_1_temperature", eval_cfg["temperature"])),
            top_p=float(eval_cfg.get("pass_1_top_p", eval_cfg["top_p"])),
            max_tokens=int(eval_cfg.get("pass_1_max_tokens", eval_cfg["max_tokens"])),
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

    # Optional LLM-as-judge rescue of rule-wrong pass@1 responses. Mirrors the
    # training-time rejudge: only responses graded WRONG by the rule are sent to
    # the judge (full response in the prompt; the answer may live in truncated
    # reasoning); correct ones are left untouched. Shares the same Rejudger
    # (prompt/cache/audit). Off unless evaluation.use_llm_judge is set.
    rescued_pass1: set = set()
    use_llm_judge = bool(eval_cfg.get("use_llm_judge", False))
    llm_judge_active = use_llm_judge and rejudger is not None and rejudger.enabled
    if llm_judge_active:
        judge_batch: List[Dict[str, Any]] = []
        for j_idx, (example, prompt, greedy_choices) in enumerate(
            zip(batch, prompts, generated_pass1)
        ):
            greedy_text = (
                str(greedy_choices[0].get("text", "")) if greedy_choices else ""
            )
            rule_score = reward_fn.score_completion(
                greedy_choices[0] if greedy_choices else greedy_text, example
            )
            rule_correct = bool(rule_score.total_reward > 0.5)
            gold = extract_gold_answer_text(
                str(example.get(eval_cfg["answer_field"], ""))
            )
            judge_batch.append({
                "question_idx": j_idx,
                "query_text": prompt,
                "gold_answer_text": gold,
                # reward>0 => rule-correct => Rejudger skips it; only wrong ones judged.
                "rollouts": [{
                    "reward": 1.0 if rule_correct else 0.0,
                    "response_text": greedy_text,
                }],
            })
        judge_metrics = rejudger.rejudge_batch(judge_batch)
        for sample in judge_batch:
            if bool(sample["rollouts"][0].get("rejudged")):
                rescued_pass1.add(sample["question_idx"])
        logger.info(
            "Eval LLM-judge: rule_wrong=%.0f judged=%.0f rescued=%d",
            judge_metrics.get("rejudge_total_zero_reward", 0),
            judge_metrics.get("rejudge_called", 0),
            len(rescued_pass1),
        )

    num_correct_pass1 = 0
    num_rule_correct_pass1 = 0
    num_correct_passk = 0
    num_empty_pass1 = 0
    num_empty_passk = 0
    pass1_texts: List[str] = []
    passk_texts: List[str] = []
    pass1_total_rewards: List[float] = []
    pass1_format_rewards: List[float] = []
    pass1_answer_rewards: List[float] = []
    pass1_length_rewards: List[float] = []
    passk_total_rewards: List[float] = []
    passk_format_rewards: List[float] = []
    passk_answer_rewards: List[float] = []
    passk_length_rewards: List[float] = []

    outcomes_log_path: Optional[Path] = None
    per_example_scores_path: Optional[Path] = None
    outcome_f = None
    score_f = None
    if details_dir is not None:
        ensure_dir(details_dir)
        outcomes_log_path = details_dir / "outcomes.log"
        per_example_scores_path = details_dir / "per_example_scores.jsonl"
        outcome_f = outcomes_log_path.open("w", encoding="utf-8")
        score_f = per_example_scores_path.open("w", encoding="utf-8")
        outcome_f.write("Evaluation Outcomes\n")
        outcome_f.write(f"model_source: {model_name_or_path}\n")
        outcome_f.write(
            f"dataset: {eval_cfg['dataset_name']}[{eval_cfg['dataset_split']}]\n"
        )
        outcome_f.write(f"num_examples: {total}\n")
        outcome_f.write(f"pass_k_enabled: {pass_k_enabled}\n")
        outcome_f.write(f"pass_k_num_samples: {pass_k_num_samples}\n\n")

    try:
        for idx, (example, prompt, greedy_choices, sampled_choices) in enumerate(
            zip(batch, prompts, generated_pass1, generated_passk)
        ):
            greedy_text = ""
            greedy_choice = None
            if greedy_choices:
                greedy_choice = greedy_choices[0]
                greedy_text = str(greedy_choice.get("text", ""))
            else:
                num_empty_pass1 += 1
            pass1_texts.append(greedy_text)

            greedy_score = reward_fn.score_completion(
                greedy_choice or greedy_text,
                example,
            )
            pass1_total_rewards.append(float(greedy_score.total_reward))
            pass1_format_rewards.append(float(greedy_score.format_reward))
            pass1_answer_rewards.append(float(greedy_score.answer_reward))
            pass1_length_rewards.append(float(greedy_score.length_reward))
            rule_correct_pass1 = bool(greedy_score.total_reward > 0.5)
            llm_rescued_pass1 = (not rule_correct_pass1) and (idx in rescued_pass1)
            is_correct_pass1 = rule_correct_pass1 or llm_rescued_pass1
            if rule_correct_pass1:
                num_rule_correct_pass1 += 1
            if is_correct_pass1:
                num_correct_pass1 += 1

            sampled_pred_answers: List[str] = []
            sampled_reward_records: List[Dict[str, Any]] = []
            sample_hit = False
            if pass_k_enabled:
                for choice in sampled_choices[:pass_k_num_samples]:
                    sampled_text = str(choice.get("text", ""))
                    sampled_score = reward_fn.score_completion(choice, example)
                    passk_texts.append(sampled_text)
                    passk_total_rewards.append(float(sampled_score.total_reward))
                    passk_format_rewards.append(float(sampled_score.format_reward))
                    passk_answer_rewards.append(float(sampled_score.answer_reward))
                    passk_length_rewards.append(float(sampled_score.length_reward))
                    sampled_pred_answers.append(extract_pred_answer(sampled_text))
                    sampled_reward_records.append(
                        {
                            "text": sampled_text,
                            "answer_only_text": sampled_score.answer_only_text,
                            "total_reward": float(sampled_score.total_reward),
                            "format_reward": float(sampled_score.format_reward),
                            "answer_reward": float(sampled_score.answer_reward),
                            "length_reward": float(sampled_score.length_reward),
                            "is_correct": bool(sampled_score.total_reward > 0.5),
                            "finish_reason": sampled_score.finish_reason,
                        }
                    )
                    if sampled_score.total_reward > 0.5:
                        sample_hit = True

                missing_samples = max(0, pass_k_num_samples - len(sampled_choices))
                if missing_samples > 0:
                    num_empty_passk += missing_samples
                    passk_texts.extend([""] * missing_samples)
                    sampled_pred_answers.extend([""] * missing_samples)

                if sample_hit:
                    num_correct_passk += 1

            raw_gold_answer = str(example.get(eval_cfg["answer_field"], ""))
            gold_answer = extract_gold_answer_text(raw_gold_answer)
            question_text = str(
                example.get(
                    eval_cfg.get("question_field"),
                    prompt,
                )
            )
            pred_answer = extract_pred_answer(greedy_text)

            if outcome_f is not None:
                outcome_f.write(f"[{idx}]\n")
                outcome_f.write(f"Question: {question_text}\n")
                outcome_f.write(f"Prompt: {prompt}\n")
                outcome_f.write(f"Complete Answer: {greedy_text}\n")
                if raw_gold_answer.strip() == gold_answer:
                    outcome_f.write(f"Gold Answer: {gold_answer}\n")
                else:
                    outcome_f.write(f"Gold Answer Raw: {raw_gold_answer}\n")
                    outcome_f.write(f"Gold Answer Extracted: {gold_answer}\n")
                outcome_f.write(f"Pass@1 Extracted Answer: {pred_answer}\n")
                outcome_f.write(
                    f"Pass@1 Reward Total: {greedy_score.total_reward:.6f}\n"
                )
                outcome_f.write(
                    f"Pass@1 Reward Format: {greedy_score.format_reward:.6f}\n"
                )
                outcome_f.write(
                    f"Pass@1 Reward Answer: {greedy_score.answer_reward:.6f}\n"
                )
                outcome_f.write(
                    f"Pass@1 Reward Length: {greedy_score.length_reward:.6f}\n"
                )
                outcome_f.write(
                    f"Pass@1 Correct: {'Correct' if is_correct_pass1 else 'wrong'}"
                    f"{' (llm_rescued)' if llm_rescued_pass1 else ''}\n"
                )
                if pass_k_enabled:
                    outcome_f.write(
                        f"Pass@{pass_k_num_samples} Sample Hit: "
                        f"{'Correct' if sample_hit else 'wrong'}\n"
                    )
                    outcome_f.write(
                        f"Pass@{pass_k_num_samples} Extracted Answers: "
                        f"{sampled_pred_answers}\n"
                    )
                outcome_f.write("\n")

            if score_f is not None:
                record = {
                    "index": int(idx),
                    "question": question_text,
                    "prompt": prompt,
                    "gold_answer_raw": raw_gold_answer,
                    "gold_answer_extracted": gold_answer,
                    "pass1": {
                        "text": greedy_text,
                        "extracted_answer": pred_answer,
                        "answer_only_text": greedy_score.answer_only_text,
                        "total_reward": float(greedy_score.total_reward),
                        "format_reward": float(greedy_score.format_reward),
                        "answer_reward": float(greedy_score.answer_reward),
                        "length_reward": float(greedy_score.length_reward),
                        "is_correct": bool(is_correct_pass1),
                        "rule_correct": bool(rule_correct_pass1),
                        "llm_rescued": bool(llm_rescued_pass1),
                        "finish_reason": greedy_score.finish_reason,
                    },
                    "pass_k_enabled": bool(pass_k_enabled),
                    "pass_k_num_samples": int(pass_k_num_samples),
                    "pass_k_hit": bool(sample_hit) if pass_k_enabled else None,
                    "pass_k_samples": sampled_reward_records,
                }
                score_f.write(json.dumps(record, ensure_ascii=False))
                score_f.write("\n")
    finally:
        if outcome_f is not None:
            outcome_f.close()
        if score_f is not None:
            score_f.close()

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
        # Transparency: rule-only vs LLM-judge-rescued breakdown.
        "llm_judge_enabled": bool(llm_judge_active),
        "num_correct_rule_pass1": int(num_rule_correct_pass1),
        "num_llm_rescued_pass1": int(len(rescued_pass1)),
        "accuracy_rule_only": float(num_rule_correct_pass1 / max(total, 1)),
        "empty_predictions": int(num_empty_pass1),
        "empty_predictions_pass1": int(num_empty_pass1),
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
        "avg_completion_tokens_pass1": float(
            sum(pass1_token_lengths) / max(len(pass1_token_lengths), 1)
        ),
        "eval_seconds": float(elapsed_s),
        "model": str(model_name_or_path),
        "pass_k_enabled": bool(pass_k_enabled),
        "pass_k_num_samples": int(pass_k_num_samples),
        "max_tokens": int(eval_cfg["max_tokens"]),
        "pass_1_max_tokens": int(
            eval_cfg.get("pass_1_max_tokens", eval_cfg["max_tokens"])
        ),
        "pass_k_max_tokens": int(
            eval_cfg.get("pass_k_max_tokens", eval_cfg["max_tokens"])
        ),
    }
    if outcomes_log_path is not None:
        metrics["outcomes_log_path"] = str(outcomes_log_path)
    if per_example_scores_path is not None:
        metrics["per_example_scores_path"] = str(per_example_scores_path)
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
        metrics[f"avg_total_reward_pass{pass_k_num_samples}"] = float(
            sum(passk_total_rewards) / max(len(passk_total_rewards), 1)
        )
        metrics[f"avg_format_reward_pass{pass_k_num_samples}"] = float(
            sum(passk_format_rewards) / max(len(passk_format_rewards), 1)
        )
        metrics[f"avg_answer_reward_pass{pass_k_num_samples}"] = float(
            sum(passk_answer_rewards) / max(len(passk_answer_rewards), 1)
        )
        metrics[f"avg_length_reward_pass{pass_k_num_samples}"] = float(
            sum(passk_length_rewards) / max(len(passk_length_rewards), 1)
        )
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
    if details_dir is not None:
        save_json(metrics, details_dir / "metrics.json")
    return metrics


def run(cfg: Dict[str, Any]) -> None:
    dist_state = get_dist_state()
    rank = int(dist_state.process_index)
    world_size = int(dist_state.num_processes)
    is_main = bool(dist_state.is_main_process)

    seed = int(cfg["seed"])
    set_seed(seed)

    base_output_dir = Path(cfg["output_dir"]).resolve()
    # All ranks MUST resolve to the SAME run dir: the disk-backed teacher-pairs
    # transport (world_size>1) has rank 0 write and other ranks read the same
    # files. Each rank computing its own time.strftime() would diverge by a
    # second, so the launcher sets APP_RUN_TAG once and every rank reuses it.
    # Fallback to a local timestamp is only safe for single-process runs.
    run_tag = os.environ.get("APP_RUN_TAG", "").strip() or time.strftime(
        "%Y%m%d_%H%M%S", time.localtime()
    )
    output_dir = ensure_dir(
        base_output_dir.with_name(f"{base_output_dir.name}_{run_tag}")
    )
    cfg["output_dir"] = str(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    rollouts_dir = ensure_dir(output_dir / "rollouts")
    metrics_dir = ensure_dir(output_dir / "metrics")
    logs_dir = ensure_dir(output_dir / "logs")
    eval_outcomes_dir = output_dir / "eval_outcomes"

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    inference_cfg = cfg["inference"]
    vllm_cfg = cfg["vllm"]
    algorithm_cfg = cfg["algorithm"]
    train_cfg = cfg["train"]
    runtime_cfg = cfg["runtime"]
    curriculum_cfg = cfg.get("curriculum", {})
    curriculum_enabled = isinstance(curriculum_cfg, dict) and bool(
        curriculum_cfg.get("enabled", False)
    )

    # GPU allocation: prefer explicit config (vllm.num_inference_gpus +
    # vllm.inference_gpu_ids) so 14B can pin 2-train + 2-infer regardless of
    # world_size. Fall back to a heuristic when not configured.
    configured_num_infer_gpus = vllm_cfg.get("num_inference_gpus")
    configured_infer_gpu_ids = vllm_cfg.get("inference_gpu_ids")
    if configured_num_infer_gpus is not None:
        num_inference_gpus = int(configured_num_infer_gpus)
        if configured_infer_gpu_ids:
            inference_gpu_ids = [int(x) for x in configured_infer_gpu_ids]
        else:
            inference_gpu_ids = list(range(num_inference_gpus))
        logger.info(
            "Inference GPUs from config: num=%d ids=%s",
            num_inference_gpus,
            inference_gpu_ids,
        )
    elif world_size >= 4:
        num_inference_gpus = max(1, world_size // 4)
        inference_gpu_ids = list(range(num_inference_gpus))
        logger.info(
            "Multi-GPU inference mode: using %d GPUs %s for vLLM tensor parallelism",
            num_inference_gpus,
            inference_gpu_ids,
        )
    else:
        num_inference_gpus = 1
        inference_gpu_ids = [int(vllm_cfg.get("gpu_idx", 0))]
    logger.info(
        "Experiment=%s | output_dir=%s | rank=%d/%d",
        cfg["exp_name"],
        output_dir,
        rank,
        world_size,
    )

    eval_cfg = _resolve_eval_cfg(cfg, data_cfg=data_cfg, inference_cfg=inference_cfg)
    # Single unified schedule: every `save_interval` iterations (and the final
    # iteration) we materialize + keep a checkpoint, sync the rollout vLLM to it,
    # and evaluate on it. policy_save_interval is the one knob driving all three.
    save_interval = max(1, int(runtime_cfg.get("policy_save_interval", 10)))
    eval_cfg["interval"] = save_interval
    save_eval_outcomes = bool(runtime_cfg.get("save_eval_outcomes", False))
    save_best_math500 = bool(runtime_cfg.get("save_best_math500", False))
    save_final_checkpoint_alias = bool(
        runtime_cfg.get("save_final_checkpoint_alias", False)
    )
    record_bad_action_stats = bool(runtime_cfg.get("record_bad_action_stats", False))
    bad_action_stats_interval = max(
        1,
        int(runtime_cfg.get("bad_action_stats_interval", save_interval)),
    )
    bad_action_stats_batch_size = max(
        1,
        int(runtime_cfg.get("bad_action_stats_batch_size", 8)),
    )
    bad_action_stats_max_edges_raw = runtime_cfg.get("bad_action_stats_max_edges")
    bad_action_stats_max_edges = (
        None
        if bad_action_stats_max_edges_raw is None
        else int(bad_action_stats_max_edges_raw)
    )
    if save_eval_outcomes:
        ensure_dir(eval_outcomes_dir)
    logger.info(
        "Schedule | save+eval+rollout_sync_every=%d",
        save_interval,
    )
    logger.info(
        "Artifacts | save_eval_outcomes=%s | save_best_math500=%s | record_bad_action_stats=%s",
        save_eval_outcomes,
        save_best_math500,
        record_bad_action_stats,
    )
    deepspeed_cfg = _resolve_deepspeed_cfg(cfg)
    project_root = Path(cfg["_meta"]["project_root"]).resolve()
    num_iterations = int(runtime_cfg["num_iterations"])
    num_questions_per_iteration = int(data_cfg["num_questions_per_iteration"])

    dataset = None
    sampler = None
    curriculum_sampler: Optional[CurriculumSampler] = None
    if curriculum_enabled:
        logger.info(
            "Curriculum sampling enabled for phase=%s",
            str(curriculum_cfg.get("phase_name", "curriculum")),
        )
        if is_main:
            curriculum_sampler = CurriculumSampler(
                curriculum_cfg=curriculum_cfg,
                data_cfg=data_cfg,
                num_iterations=num_iterations,
                project_root=project_root,
                seed=seed,
            )
    else:
        dataset = _load_dataset_auto(
            data_cfg["dataset_name"],
            split=data_cfg["dataset_split"],
        )
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
        logger.info("Loaded dataset size: %d", len(dataset))

    initial_model_path = resolve_init_model_path(
        str(model_cfg["actor_name_or_path"]),
        project_root=project_root,
    )
    tokenizer_name = model_cfg.get("tokenizer_name_or_path") or model_cfg["actor_name_or_path"]
    initial_tokenizer_path = resolve_init_model_path(
        str(tokenizer_name),
        project_root=project_root,
    )
    logger.info("Resolved initial model path: %s", initial_model_path)
    logger.info("Resolved initial tokenizer path: %s", initial_tokenizer_path)
    tokenizer = load_causal_lm_tokenizer(
        str(initial_tokenizer_path),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    dtype = _resolve_dtype(model_cfg.get("torch_dtype", "bfloat16"))
    if not torch.cuda.is_available():
        dtype = torch.float32
    model_load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
    }
    attn_implementation, attn_source = resolve_attn_implementation(
        model_cfg.get("attn_implementation"),
        model_name_or_path=str(model_cfg["actor_name_or_path"]),
        model_path=initial_model_path,
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

    model = AutoModelForCausalLM.from_pretrained(
        str(initial_model_path),
        **model_load_kwargs,
    )
    align_model_special_tokens(model, tokenizer)
    if bool(train_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    teacher_builder = ClosedFormTeacherBuilder(
        tokenizer=tokenizer,
        algorithm_cfg=algorithm_cfg,
        max_sequence_length=train_cfg.get("max_sequence_length"),
    )
    reward_fn = MathRewardFn(answer_field=data_cfg["answer_field"])

    rejudge_cfg = dict(cfg.get("rejudge") or {})
    rejudge_cfg.setdefault(
        "cache_path", str(output_dir / "rejudge_cache.jsonl")
    )
    rejudge_cfg.setdefault(
        "audit_path", str(output_dir / "rejudge_log.jsonl")
    )
    rejudger = Rejudger(rejudge_cfg) if is_main else None
    eval_dataset = None
    if is_main and bool(eval_cfg.get("enabled", False)):
        eval_dataset = _load_dataset_auto(
            str(eval_cfg["dataset_name"]),
            split=str(eval_cfg["dataset_split"]),
        )
        if eval_cfg.get("max_samples") is not None:
            max_samples = int(eval_cfg["max_samples"])
            if max_samples < len(eval_dataset):
                eval_dataset = eval_dataset.select(range(max_samples))
        logger.info("Loaded evaluation dataset size: %d", len(eval_dataset))

    if not curriculum_enabled:
        sampler = QuestionSampler(
            dataset_size=len(dataset),
            num_iterations=num_iterations,
            num_questions_per_iteration=num_questions_per_iteration,
            sample_with_replacement=bool(data_cfg["sample_with_replacement"]),
            shuffle_on_each_iteration=bool(data_cfg["shuffle_on_each_iteration"]),
            seed=seed,
        )

    sampling_model_source: str = str(initial_model_path)
    best_math500_dir = checkpoints_dir / "best_math500"
    final_alias_dir = checkpoints_dir / "final_actor"
    latest_saved_actor_ckpt: Optional[Path] = None
    best_math500_accuracy = float("-inf")
    history: List[Dict[str, Any]] = []
    save_total_limit: Optional[int] = runtime_cfg.get("save_total_limit")
    saved_actor_ckpts: List[Path] = []
    persist_teacher_pairs = bool(
        runtime_cfg.get("persist_teacher_pairs", world_size > 1)
    )

    teacher_pairs_dir = ensure_dir(output_dir / "teacher_pairs")
    if is_main:
        logger.info(
            "Teacher pairs transport: %s",
            "disk-backed sync/persist" if persist_teacher_pairs else "in-memory only",
        )
    barrier()

    # Create the trainer once; reuse across iterations to preserve optimizer state.
    rl_trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        algorithm_cfg=algorithm_cfg,
        deepspeed_cfg=deepspeed_cfg,
        output_dir=checkpoints_dir,
        seed=seed,
    )

    # Persistent rollout vLLM server — only restarted when the sampling model changes.
    vllm_server: Optional[VLLMServer] = None
    vllm_client: Optional[VLLMClient] = None
    vllm_loaded_model: Optional[str] = None  # tracks which model the server has loaded

    def _create_vllm_server():
        if num_inference_gpus > 1:
            vllm_cfg_copy = dict(vllm_cfg)
            vllm_cfg_copy["gpu_ids"] = inference_gpu_ids
            return MultiGPUVLLMServer(
                vllm_cfg_copy,
                logs_dir / "vllm_server",
                num_gpus=num_inference_gpus,
            )
        return VLLMServer(vllm_cfg, logs_dir / "vllm_server")

    def _stop_vllm():
        nonlocal vllm_server, vllm_client, vllm_loaded_model
        if vllm_server is not None:
            vllm_server.stop()
            vllm_server = None
        vllm_client = None
        vllm_loaded_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_vllm_for_model(model_path: str, seed: int):
        """Start or reuse the rollout vLLM server for the given model."""
        nonlocal vllm_server, vllm_client, vllm_loaded_model
        if vllm_server is not None and vllm_loaded_model == model_path:
            return
        if vllm_server is not None:
            logger.info("Restarting vLLM server for new model: %s", model_path)
            _stop_vllm()
        else:
            logger.info("Starting vLLM server with model: %s", model_path)
        vllm_server = _create_vllm_server()
        api_base = vllm_server.start(model_name_or_path=model_path, seed=seed)
        vllm_client = VLLMClient(
            api_base=api_base,
            model=str(model_path),
            request_timeout_s=int(inference_cfg["request_timeout_s"]),
            max_parallel_requests=int(inference_cfg["max_parallel_requests"]),
        )
        vllm_loaded_model = model_path

    for iteration in range(num_iterations):
        iter_seed = seed + iteration * 1000
        if is_main:
            logger.info("========== Iteration %d ==========", iteration)

        teacher_examples_path = teacher_pairs_dir / f"iter_{iteration:04d}.jsonl"
        teacher_examples: List[Dict[str, Any]] = []
        teacher_metrics: Dict[str, Any] = {}
        eval_metrics: Dict[str, Any] = {}
        bad_action_stats: Dict[str, Any] = {}
        curriculum_info: Dict[str, Any] = {}
        batch_rollouts_for_stats: List[Dict[str, Any]] = []
        grpo_build_metrics: Dict[str, Any] = {}
        rejudge_metrics: Dict[str, Any] = {}

        if is_main:
            sampled_indices = None
            sampled_question_ids = None
            if curriculum_enabled:
                if curriculum_sampler is None:
                    raise RuntimeError("Curriculum sampler is not initialized on the main process.")
                curriculum_sample = curriculum_sampler.sample(iteration)
                batch = curriculum_sample.examples
                sampled_question_ids = curriculum_sample.question_ids
                curriculum_info = dict(curriculum_sample.info)
                logger.info(
                    "Curriculum | phase=%s stage=%s sampling_mode=%s stage_weights=%s source_counts=%s group_counts=%s subset_sizes=%s",
                    curriculum_info["phase"],
                    curriculum_info["stage"],
                    curriculum_info.get("sampling_mode"),
                    curriculum_info.get("stage_group_weights"),
                    curriculum_info["source_counts"],
                    curriculum_info.get("source_group_counts"),
                    curriculum_info["source_subset_sizes"],
                )
            else:
                sampled_indices = sampler.sample(iteration)
                batch = dataset.select(sampled_indices)
            prompts = [_format_prompt(example, data_cfg) for example in batch]

            # Reuse rollout vLLM server if the sampling model hasn't changed.
            _ensure_vllm_for_model(sampling_model_source, seed=iter_seed)

            generated = vllm_client.generate_batch(
                prompts=prompts,
                n=int(inference_cfg["rollouts_per_question"]),
                temperature=float(inference_cfg["temperature"]),
                top_p=float(inference_cfg["top_p"]),
                max_tokens=int(inference_cfg["max_tokens"]),
                stop=inference_cfg.get("stop"),
                seed=iter_seed,
                # Per-token logprobs for GRPO importance ratio.
                logprobs=1,
            )

            batch_rollouts = _build_batch_rollouts(
                batch=batch,
                prompts=prompts,
                generated=generated,
                reward_fn=reward_fn,
                data_cfg=data_cfg,
                sampled_indices=sampled_indices,
                sampled_question_ids=sampled_question_ids,
            )

            rejudge_metrics: Dict[str, Any] = {}
            if rejudger is not None:
                rejudge_metrics = rejudger.rejudge_batch(batch_rollouts)
                logger.info(
                    "Rejudge: zero=%.0f called=%.0f cache_hits=%.0f flipped=%.0f failures=%.0f short_empty=%.0f latency=%.1fs",
                    rejudge_metrics.get("rejudge_total_zero_reward", 0),
                    rejudge_metrics.get("rejudge_called", 0),
                    rejudge_metrics.get("rejudge_cache_hits", 0),
                    rejudge_metrics.get("rejudge_flipped", 0),
                    rejudge_metrics.get("rejudge_api_failures", 0),
                    rejudge_metrics.get("rejudge_short_circuit_empty", 0),
                    rejudge_metrics.get("rejudge_latency_s", 0),
                )

            batch_rollouts_for_stats = batch_rollouts

            if (
                int(runtime_cfg.get("save_rollouts_every", 0)) > 0
                and iteration % int(runtime_cfg["save_rollouts_every"]) == 0
            ):
                rollout_payload: Dict[str, Any] = {
                    "iteration": iteration,
                    "samples": batch_rollouts,
                }
                if curriculum_info:
                    rollout_payload["curriculum"] = curriculum_info
                save_json(
                    rollout_payload,
                    rollouts_dir / f"rollouts_iter_{iteration:04d}.json",
                )

            teacher_examples, teacher_metrics, per_question_aux = (
                teacher_builder.build_for_batch(batch_rollouts)
            )

            grpo_loss_weight = float(algorithm_cfg.get("grpo_loss_weight", 0.0))
            grpo_build_metrics: Dict[str, float] = {}
            if grpo_loss_weight > 0:
                grpo_examples, grpo_build_metrics = teacher_builder.build_grpo_examples(
                    batch_rollouts=batch_rollouts,
                    per_question_aux=per_question_aux,
                    grpo_skip_after_frontier=bool(
                        algorithm_cfg.get("grpo_skip_after_frontier", False)
                    ),
                    advantage_eps=float(
                        algorithm_cfg.get("grpo_advantage_eps", 1e-6)
                    ),
                )
                teacher_examples = teacher_examples + grpo_examples
                teacher_metrics.update(
                    {f"grpo_build/{k}": v for k, v in grpo_build_metrics.items()}
                )
            if persist_teacher_pairs:
                _save_jsonl_examples(teacher_examples, teacher_examples_path)
            logger.info(
                "Teacher pairs: total=%d pos=%.0f neg=%.0f",
                len(teacher_examples),
                teacher_metrics["num_positive_pairs_total"],
                teacher_metrics["num_negative_pairs_total"],
            )

        barrier()
        if persist_teacher_pairs:
            teacher_examples = _load_jsonl_examples(teacher_examples_path)

        if len(teacher_examples) == 0:
            if is_main:
                if eval_dataset is not None and _should_run_evaluation(
                    iteration=iteration,
                    eval_cfg=eval_cfg,
                ):
                    logger.info(
                        "Running evaluation on %s[%s] at iteration %d with max_tokens=%d",
                        eval_cfg["dataset_name"],
                        eval_cfg["dataset_split"],
                        iteration,
                        int(eval_cfg["max_tokens"]),
                    )
                    eval_model_source = (
                        str(latest_saved_actor_ckpt)
                        if latest_saved_actor_ckpt is not None
                        else str(sampling_model_source)
                    )
                    # Stop rollout server, run eval, then let next iteration restart rollout server.
                    _stop_vllm()
                    eval_details_dir = (
                        _eval_details_dir(
                            eval_outcomes_dir=eval_outcomes_dir,
                            iteration=iteration,
                            eval_cfg=eval_cfg,
                        )
                        if save_eval_outcomes
                        else None
                    )
                    eval_metrics = _evaluate_with_vllm(
                        model_name_or_path=eval_model_source,
                        eval_dataset=eval_dataset,
                        eval_cfg=eval_cfg,
                        vllm_cfg=vllm_cfg,
                        num_inference_gpus=num_inference_gpus,
                        inference_gpu_ids=inference_gpu_ids,
                        log_dir=logs_dir / f"eval_iter_{iteration:04d}",
                        seed=iter_seed + 17,
                        tokenizer=tokenizer,
                        details_dir=eval_details_dir,
                        rejudger=rejudger,
                    )
                    _log_eval_summary(eval_metrics)
                if (
                    record_bad_action_stats
                    and (iteration + 1) % bad_action_stats_interval == 0
                    and len(batch_rollouts_for_stats) > 0
                ):
                    bad_edges, bad_edge_metrics = teacher_builder.build_bad_frontier_edges(
                        batch_rollouts_for_stats
                    )
                    pad_token_id = tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = (
                            tokenizer.eos_token_id
                            if tokenizer.eos_token_id is not None
                            else 0
                        )
                    bad_action_stats = {
                        "iteration": int(iteration),
                        "outer_iteration": int(iteration + 1),
                        **bad_edge_metrics,
                        **_compute_bad_action_stats(
                            trainer=rl_trainer,
                            bad_action_examples=bad_edges,
                            pad_token_id=int(pad_token_id),
                            batch_size=bad_action_stats_batch_size,
                            max_edges=bad_action_stats_max_edges,
                        ),
                    }
                    save_json(
                        bad_action_stats,
                        metrics_dir / f"bad_action_stats_iter_{iteration:04d}.json",
                    )
                logger.warning(
                    "No teacher examples generated at iteration %d. Skipping training.",
                    iteration,
                )
                skip_metrics = {
                    "iteration": iteration,
                    "skipped": True,
                    "teacher": teacher_metrics,
                    "grpo_build": grpo_build_metrics,
                    "rejudge": rejudge_metrics,
                    "evaluation": eval_metrics,
                    "reference_model_source": str(sampling_model_source),
                    "latest_policy_checkpoint": str(latest_saved_actor_ckpt)
                    if latest_saved_actor_ckpt is not None
                    else None,
                    "reference_model_source_next_iter": str(sampling_model_source),
                    "rollout_model_source_next_iter": str(sampling_model_source),
                    "sampling_model_source": str(sampling_model_source),
                    "sampling_model_source_next_iter": str(sampling_model_source),
                    "rollout_model_updated": False,
                    "save_interval": int(save_interval),
                    "bad_action_stats": bad_action_stats,
                }
                if curriculum_info:
                    skip_metrics["curriculum"] = curriculum_info
                history.append(skip_metrics)
                save_json(skip_metrics, metrics_dir / f"iter_{iteration:04d}.json")
                save_json({"history": history}, metrics_dir / "history.json")
            barrier()
            continue

        # vLLM runs on a separate GPU, no need to stop it during training.

        eval_is_active = eval_dataset is not None
        is_final_iteration = iteration == (num_iterations - 1)
        # One unified checkpoint boundary: every `save_interval` iterations (and
        # the final one) we save+keep a ckpt, sync the rollout vLLM to it, and eval.
        should_checkpoint = ((iteration + 1) % save_interval == 0) or is_final_iteration
        should_run_eval = eval_is_active and should_checkpoint

        actor_ckpt_dir = checkpoints_dir / f"iter_{iteration:04d}_actor"
        if is_main and actor_ckpt_dir.exists():
            shutil.rmtree(actor_ckpt_dir, ignore_errors=True)
        barrier()
        train_metrics = train_one_iteration(
            trainer=rl_trainer,
            tokenizer=tokenizer,
            examples=teacher_examples,
            train_cfg=train_cfg,
            output_dir=actor_ckpt_dir,
            save_checkpoint=should_checkpoint,
            num_iterations=num_iterations,
        )
        barrier()

        # At each checkpoint boundary, sync the rollout vLLM to the just-saved
        # weights (next iteration's _ensure_vllm_for_model reloads them).
        rollout_model_updated_this_iter = False
        if is_main and should_checkpoint:
            if not actor_ckpt_dir.exists():
                logger.warning(
                    "Skipping rollout model update at iteration %d because no actor checkpoint was materialized.",
                    iteration,
                )
            else:
                sampling_model_source = str(actor_ckpt_dir)
                rollout_model_updated_this_iter = True
        barrier()

        if is_main:
            if should_run_eval:
                # When eval runs we always have a freshly saved ckpt to score.
                eval_model_source = str(actor_ckpt_dir)
                # Stop rollout vLLM so eval can use the same GPU.
                _stop_vllm()
                logger.info(
                    "Running evaluation on %s[%s] at iteration %d with max_tokens=%d",
                    eval_cfg["dataset_name"],
                    eval_cfg["dataset_split"],
                    iteration,
                    int(eval_cfg["max_tokens"]),
                )
                eval_details_dir = (
                    _eval_details_dir(
                        eval_outcomes_dir=eval_outcomes_dir,
                        iteration=iteration,
                        eval_cfg=eval_cfg,
                    )
                    if save_eval_outcomes
                    else None
                )
                eval_metrics = _evaluate_with_vllm(
                    model_name_or_path=eval_model_source,
                    eval_dataset=eval_dataset,
                    eval_cfg=eval_cfg,
                    vllm_cfg=vllm_cfg,
                    num_inference_gpus=num_inference_gpus,
                    inference_gpu_ids=inference_gpu_ids,
                    log_dir=logs_dir / f"eval_iter_{iteration:04d}",
                    seed=iter_seed + 17,
                    tokenizer=tokenizer,
                    details_dir=eval_details_dir,
                    rejudger=rejudger,
                )
                _log_eval_summary(eval_metrics)

            if (
                record_bad_action_stats
                and (iteration + 1) % bad_action_stats_interval == 0
                and len(batch_rollouts_for_stats) > 0
            ):
                bad_edges, bad_edge_metrics = teacher_builder.build_bad_frontier_edges(
                    batch_rollouts_for_stats
                )
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = (
                        tokenizer.eos_token_id
                        if tokenizer.eos_token_id is not None
                        else 0
                    )
                bad_action_stats = {
                    "iteration": int(iteration),
                    "outer_iteration": int(iteration + 1),
                    **bad_edge_metrics,
                    **_compute_bad_action_stats(
                        trainer=rl_trainer,
                        bad_action_examples=bad_edges,
                        pad_token_id=int(pad_token_id),
                        batch_size=bad_action_stats_batch_size,
                        max_edges=bad_action_stats_max_edges,
                    ),
                }
                save_json(
                    bad_action_stats,
                    metrics_dir / f"bad_action_stats_iter_{iteration:04d}.json",
                )

            best_math500_checkpoint = None
            if save_best_math500 and should_run_eval and eval_metrics:
                math500_acc = float(eval_metrics.get("accuracy", float("-inf")))
                if math500_acc > best_math500_accuracy and actor_ckpt_dir.exists():
                    best_math500_accuracy = math500_acc
                    _replace_checkpoint_dir(actor_ckpt_dir, best_math500_dir)
                    best_math500_checkpoint = str(best_math500_dir)
                    save_json(
                        {
                            "iteration": int(iteration),
                            "outer_iteration": int(iteration + 1),
                            "accuracy": float(best_math500_accuracy),
                            "checkpoint_dir": str(best_math500_dir),
                            "source_checkpoint_dir": str(actor_ckpt_dir),
                            "evaluation": eval_metrics,
                        },
                        metrics_dir / "best_math500.json",
                    )

            final_checkpoint_alias = None
            if (
                is_final_iteration
                and save_final_checkpoint_alias
                and actor_ckpt_dir.exists()
            ):
                _replace_checkpoint_dir(actor_ckpt_dir, final_alias_dir)
                final_checkpoint_alias = str(final_alias_dir)

            actor_checkpoint_kept = False
            if should_checkpoint and actor_ckpt_dir.exists():
                actor_checkpoint_kept = True
                latest_saved_actor_ckpt = actor_ckpt_dir
                if actor_ckpt_dir not in saved_actor_ckpts:
                    saved_actor_ckpts.append(actor_ckpt_dir)
                if save_total_limit is not None and len(saved_actor_ckpts) > int(save_total_limit):
                    oldest = saved_actor_ckpts.pop(0)
                    if oldest.exists() and oldest != latest_saved_actor_ckpt:
                        logger.info("Removing old actor checkpoint: %s", oldest)
                        shutil.rmtree(oldest, ignore_errors=True)
            elif actor_ckpt_dir.exists():
                shutil.rmtree(actor_ckpt_dir, ignore_errors=True)

            iter_metrics = {
                "iteration": iteration,
                "skipped": False,
                "num_teacher_examples": len(teacher_examples),
                "teacher": teacher_metrics,
                "grpo_build": grpo_build_metrics,
                "rejudge": rejudge_metrics,
                "train": train_metrics,
                "evaluation": eval_metrics,
                "reference_model_source_next_iter": str(sampling_model_source),
                "rollout_model_source_next_iter": str(sampling_model_source),
                "sampling_model_source_next_iter": str(sampling_model_source),
                "evaluation_model_source": eval_model_source if should_run_eval else None,
                "actor_checkpoint": str(actor_ckpt_dir) if actor_checkpoint_kept else None,
                "actor_checkpoint_materialized": bool(should_checkpoint),
                "actor_checkpoint_saved": bool(actor_checkpoint_kept),
                "policy_checkpoint_saved": bool(actor_checkpoint_kept),
                "best_math500_accuracy": (
                    None
                    if best_math500_accuracy == float("-inf")
                    else float(best_math500_accuracy)
                ),
                "best_math500_checkpoint": best_math500_checkpoint,
                "final_checkpoint_alias": final_checkpoint_alias,
                "bad_action_stats": bad_action_stats,
                "save_interval": int(save_interval),
                "rollout_model_updated": bool(rollout_model_updated_this_iter),
            }
            if curriculum_info:
                iter_metrics["curriculum"] = curriculum_info
            history.append(iter_metrics)
            save_json(iter_metrics, metrics_dir / f"iter_{iteration:04d}.json")
            save_json({"history": history}, metrics_dir / "history.json")
        barrier()

    # Cleanup: stop vLLM server after all iterations.
    if is_main:
        _stop_vllm()

    if is_main:
        logger.info("All iterations finished. Final checkpoint: %s", checkpoints_dir)


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
