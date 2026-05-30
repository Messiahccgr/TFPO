import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.grpo_runner import (
    _add_pass_1_sampling_metrics,
    _format_prompt,
    _is_pass_k_enabled,
    _load_dataset_auto,
    _resolve_pass_1_generation_cfg,
    _tokenize_text_lengths,
)
from src.reward import MathRewardFn, extract_gold_answer_text, extract_pred_answer
from src.tokenization import load_causal_lm_tokenizer
from src.utils import ensure_dir, save_json_atomic, setup_logger
from src.vllm import VLLMClient, VLLMServer
from src.vllm_multi_gpu import MultiGPUVLLMServer


logger = setup_logger("external_vllm_eval_worker")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _collect_eval_batch(eval_cfg: Dict[str, Any]):
    dataset = _load_dataset_auto(
        str(eval_cfg["dataset_name"]),
        split=str(eval_cfg["dataset_split"]),
    )
    max_samples = eval_cfg.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
    eval_data_cfg = {
        "question_template": eval_cfg["question_template"],
        "question_field": eval_cfg.get("question_field"),
    }
    batch = [dataset[i] for i in range(len(dataset))]
    prompts = [_format_prompt(example, eval_data_cfg) for example in batch]
    return batch, prompts


def _numeric_metric_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if len(records) == 0:
        return {}
    metric_keys = sorted(
        {
            key
            for record in records
            for key, value in record["metrics"].items()
            if isinstance(value, (int, float))
        }
    )
    summary: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = [float(record["metrics"][key]) for record in records if key in record["metrics"]]
        if len(values) == 0:
            continue
        summary[key] = {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / len(values)),
            "latest": float(values[-1]),
        }
    return summary


def _build_summary(records: List[Dict[str, Any]], checkpoint_dir: str, step: int, epoch: Optional[float]):
    if len(records) == 0:
        return {
            "step": int(step),
            "epoch": epoch,
            "checkpoint_dir": str(checkpoint_dir),
            "num_runs": 0,
            "metrics_summary": {},
        }
    ranking_key = "accuracy" if "accuracy" in records[0]["metrics"] else None
    best_run = None
    worst_run = None
    if ranking_key is not None:
        best_run = max(records, key=lambda item: float(item["metrics"].get(ranking_key, float("-inf"))))
        worst_run = min(records, key=lambda item: float(item["metrics"].get(ranking_key, float("inf"))))
    return {
        "step": int(step),
        "epoch": epoch,
        "checkpoint_dir": str(checkpoint_dir),
        "num_runs": int(len(records)),
        "metrics_summary": _numeric_metric_summary(records),
        "best_run": best_run,
        "worst_run": worst_run,
        "latest_run": records[-1],
    }


def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _materialize_extrema_artifacts(metrics_dir: Path, step: int, summary: Dict[str, Any]) -> Dict[str, Any]:
    step_prefix = f"eval_step_{int(step):06d}"
    best_run = summary.get("best_run")
    worst_run = summary.get("worst_run")

    best_outcomes_path = None
    best_per_example_scores_path = None
    worst_outcomes_path = None
    worst_per_example_scores_path = None

    if isinstance(best_run, dict):
        best_outcomes_path = _copy_if_exists(
            Path(str(best_run.get("outcomes_log_path", ""))),
            metrics_dir / f"{step_prefix}_best_outcomes.log",
        )
        best_per_example_scores_path = _copy_if_exists(
            Path(str(best_run.get("per_example_scores_path", ""))),
            metrics_dir / f"{step_prefix}_best_per_example_scores.jsonl",
        )

    if isinstance(worst_run, dict):
        worst_outcomes_path = _copy_if_exists(
            Path(str(worst_run.get("outcomes_log_path", ""))),
            metrics_dir / f"{step_prefix}_worst_outcomes.log",
        )
        worst_per_example_scores_path = _copy_if_exists(
            Path(str(worst_run.get("per_example_scores_path", ""))),
            metrics_dir / f"{step_prefix}_worst_per_example_scores.jsonl",
        )

    summary["best_outcomes_path"] = best_outcomes_path
    summary["best_per_example_scores_path"] = best_per_example_scores_path
    summary["worst_outcomes_path"] = worst_outcomes_path
    summary["worst_per_example_scores_path"] = worst_per_example_scores_path
    return summary


def _update_history(path: Path, summary: Dict[str, Any]) -> None:
    payload = _load_json(path) or {"history": []}
    history = payload.get("history", [])
    filtered = [item for item in history if int(item.get("step", -1)) != int(summary["step"])]
    filtered.append(summary)
    filtered.sort(key=lambda item: int(item.get("step", -1)))
    save_json_atomic({"history": filtered}, path)


class _PersistentEvaluator:
    def __init__(
        self,
        *,
        vllm_cfg: Dict[str, Any],
        num_inference_gpus: int,
        inference_gpu_ids: List[int],
        log_root: Path,
        metrics_root: Path,
        eval_cfg: Dict[str, Any],
        batch: List[Dict[str, Any]],
        prompts: List[str],
        tokenizer,
    ):
        self.vllm_cfg = dict(vllm_cfg)
        self.num_inference_gpus = int(num_inference_gpus)
        self.inference_gpu_ids = [int(gpu_id) for gpu_id in inference_gpu_ids]
        self.log_root = log_root
        self.metrics_root = metrics_root
        self.eval_cfg = dict(eval_cfg)
        self.batch = list(batch)
        self.prompts = list(prompts)
        self.reward_fn = MathRewardFn(answer_field=str(eval_cfg["answer_field"]))
        self.tokenizer = tokenizer
        self.server = None
        self.client: Optional[VLLMClient] = None
        self.loaded_model: Optional[str] = None

    def _stop(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
            self.client = None
            self.loaded_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_name_or_path: str, seed: int, step: int) -> None:
        if self.loaded_model == model_name_or_path and self.client is not None:
            return
        self._stop()
        if self.num_inference_gpus > 1:
            vllm_cfg_copy = dict(self.vllm_cfg)
            vllm_cfg_copy["gpu_ids"] = self.inference_gpu_ids
            self.server = MultiGPUVLLMServer(
                vllm_cfg_copy,
                ensure_dir(self.log_root / f"eval_step_{step:06d}"),
                num_gpus=self.num_inference_gpus,
            )
        else:
            vllm_cfg_copy = dict(self.vllm_cfg)
            vllm_cfg_copy["gpu_idx"] = int(self.inference_gpu_ids[0])
            self.server = VLLMServer(
                vllm_cfg_copy,
                ensure_dir(self.log_root / f"eval_step_{step:06d}"),
            )
        api_base = self.server.start(model_name_or_path=model_name_or_path, seed=seed)
        self.client = VLLMClient(
            api_base=api_base,
            model=str(model_name_or_path),
            request_timeout_s=int(self.eval_cfg["request_timeout_s"]),
            max_parallel_requests=int(self.eval_cfg["max_parallel_requests"]),
        )
        self.loaded_model = str(model_name_or_path)

    def evaluate_once(
        self,
        *,
        model_name_or_path: str,
        step: int,
        seed: int,
        run_index: int,
    ) -> Dict[str, Any]:
        self.load_model(model_name_or_path=model_name_or_path, seed=seed, step=step)
        assert self.client is not None
        start = time.time()
        pass_k_enabled = _is_pass_k_enabled(self.eval_cfg)
        pass_1_cfg = _resolve_pass_1_generation_cfg(self.eval_cfg)
        pass_k_num_samples = (
            max(1, int(self.eval_cfg.get("pass_k_num_samples", 8)))
            if pass_k_enabled
            else 0
        )
        generated_pass1 = self.client.generate_batch(
            prompts=self.prompts,
            n=1,
            temperature=float(pass_1_cfg["temperature"]),
            top_p=float(pass_1_cfg["top_p"]),
            max_tokens=int(pass_1_cfg["max_tokens"]),
            stop=self.eval_cfg.get("stop"),
            seed=seed,
        )
        generated_passk = [[] for _ in range(len(self.batch))]
        if pass_k_enabled:
            generated_passk = self.client.generate_batch(
                prompts=self.prompts,
                n=pass_k_num_samples,
                temperature=float(self.eval_cfg.get("pass_k_temperature", 0.6)),
                top_p=float(self.eval_cfg.get("pass_k_top_p", 0.9)),
                max_tokens=int(self.eval_cfg.get("pass_k_max_tokens", self.eval_cfg["max_tokens"])),
                stop=self.eval_cfg.get("stop"),
                seed=seed + 100000,
            )

        num_correct_pass1 = 0
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

        run_prefix = f"eval_step_{int(step):06d}_run_{int(run_index):04d}"
        outcomes_log_path = self.metrics_root / f"{run_prefix}_outcomes.log"
        per_example_scores_path = self.metrics_root / f"{run_prefix}_per_example_scores.jsonl"
        metrics_path = self.metrics_root / f"{run_prefix}_metrics.json"
        run_record_path = self.metrics_root / f"{run_prefix}.json"
        answer_field = str(self.eval_cfg["answer_field"])
        question_field = str(self.eval_cfg.get("question_field") or "problem")
        question_template = str(self.eval_cfg["question_template"])

        with outcomes_log_path.open("w", encoding="utf-8") as outcome_f, per_example_scores_path.open(
            "w", encoding="utf-8"
        ) as score_f:
            outcome_f.write("Evaluation Outcomes\n")
            outcome_f.write(f"checkpoint_dir: {model_name_or_path}\n")
            outcome_f.write(
                f"dataset: {self.eval_cfg['dataset_name']}[{self.eval_cfg['dataset_split']}] "
                f"config={self.eval_cfg.get('dataset_config_name')}\n"
            )
            outcome_f.write(f"step: {int(step)}\n")
            outcome_f.write(f"run_index: {int(run_index)}\n")
            outcome_f.write(f"seed: {int(seed)}\n")
            outcome_f.write(f"num_examples: {len(self.batch)}\n\n")

            for idx, (example, greedy_choices, sampled_choices) in enumerate(
                zip(self.batch, generated_pass1, generated_passk)
            ):
                greedy_text = ""
                greedy_choice = None
                if greedy_choices:
                    greedy_choice = greedy_choices[0]
                    greedy_text = str(greedy_choice.get("text", ""))
                else:
                    num_empty_pass1 += 1
                pass1_texts.append(greedy_text)

                greedy_score = self.reward_fn.score_completion(greedy_choice or greedy_text, example)
                pred_answer = extract_pred_answer(greedy_text)
                raw_gold_answer = str(example.get(answer_field, ""))
                gold_answer = extract_gold_answer_text(raw_gold_answer)
                parsed_gold_answer = self.reward_fn.describe_gold_math_verify_parse(example)
                parsed_pred_answer = self.reward_fn.describe_prediction_math_verify_parse(
                    greedy_choice or greedy_text
                )

                pass1_total_rewards.append(float(greedy_score.total_reward))
                pass1_format_rewards.append(float(greedy_score.format_reward))
                pass1_answer_rewards.append(float(greedy_score.answer_reward))
                pass1_length_rewards.append(float(greedy_score.length_reward))

                is_correct_pass1 = greedy_score.total_reward > 0.5
                if is_correct_pass1:
                    num_correct_pass1 += 1

                sampled_pred_answers: List[str] = []
                sampled_reward_records: List[Dict[str, Any]] = []
                sampled_hit = False
                if pass_k_enabled:
                    for choice in sampled_choices[:pass_k_num_samples]:
                        sampled_text = str(choice.get("text", ""))
                        sampled_score = self.reward_fn.score_completion(choice, example)
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
                            sampled_hit = True

                    missing_samples = max(0, pass_k_num_samples - len(sampled_choices))
                    if missing_samples > 0:
                        num_empty_passk += missing_samples
                        passk_texts.extend([""] * missing_samples)
                        sampled_pred_answers.extend([""] * missing_samples)

                    if sampled_hit:
                        num_correct_passk += 1

                question_text = str(
                    example.get(
                        question_field,
                        _format_prompt(
                            example,
                            {
                                "question_template": question_template,
                                "question_field": question_field,
                            },
                        ),
                    )
                )

                outcome_f.write(f"[{idx}]\n")
                outcome_f.write(f"Question: {question_text}\n")
                outcome_f.write(f"Complete Answer: {greedy_text}\n")
                if raw_gold_answer.strip() == gold_answer:
                    outcome_f.write(f"Gold Answer: {gold_answer}\n")
                else:
                    outcome_f.write(f"Gold Answer Raw: {raw_gold_answer}\n")
                    outcome_f.write(f"Gold Answer Extracted: {gold_answer}\n")
                outcome_f.write(f"MathVerify Parsed Gold: {parsed_gold_answer}\n")
                outcome_f.write(f"Pass@1 Extracted Answer: {pred_answer}\n")
                outcome_f.write(f"MathVerify Parsed Pred: {parsed_pred_answer}\n")
                outcome_f.write(f"Pass@1 Reward Total: {greedy_score.total_reward:.6f}\n")
                outcome_f.write(f"Pass@1 Reward Format: {greedy_score.format_reward:.6f}\n")
                outcome_f.write(f"Pass@1 Reward Answer: {greedy_score.answer_reward:.6f}\n")
                outcome_f.write(f"Pass@1 Reward Length: {greedy_score.length_reward:.6f}\n")
                outcome_f.write(
                    f"Pass@1 Correct: {'Correct' if is_correct_pass1 else 'wrong'}\n"
                )
                if pass_k_enabled:
                    outcome_f.write(
                        f"Pass@{pass_k_num_samples} Sample Hit: "
                        f"{'Correct' if sampled_hit else 'wrong'}\n"
                    )
                    outcome_f.write(
                        f"Pass@{pass_k_num_samples} Extracted Answers: {sampled_pred_answers}\n"
                    )
                    outcome_f.write(
                        f"Pass@{pass_k_num_samples} Reward Totals: "
                        f"{[round(rec['total_reward'], 6) for rec in sampled_reward_records]}\n\n"
                    )
                else:
                    outcome_f.write("Pass@k Disabled: true\n\n")

                per_example_record = {
                    "index": int(idx),
                    "question": question_text,
                    "gold_answer_raw": raw_gold_answer,
                    "gold_answer_extracted": gold_answer,
                    "pass1": {
                        "text": greedy_text,
                        "answer_only_text": greedy_score.answer_only_text,
                        "total_reward": float(greedy_score.total_reward),
                        "format_reward": float(greedy_score.format_reward),
                        "answer_reward": float(greedy_score.answer_reward),
                        "length_reward": float(greedy_score.length_reward),
                        "is_correct": bool(is_correct_pass1),
                        "finish_reason": greedy_score.finish_reason,
                    },
                    "pass_k_enabled": bool(pass_k_enabled),
                    "pass_k_num_samples": int(pass_k_num_samples),
                    "pass_k_hit": bool(sampled_hit) if pass_k_enabled else None,
                    "pass_k_samples": sampled_reward_records,
                }
                score_f.write(json.dumps(per_example_record, ensure_ascii=False))
                score_f.write("\n")

        total = len(self.batch)
        elapsed_s = time.time() - start
        pass1_token_lengths = _tokenize_text_lengths(self.tokenizer, pass1_texts)
        pass_at_1 = float(num_correct_pass1 / max(total, 1))
        metrics = {
            "enabled": True,
            "timestamp": int(start),
            "dataset_name": str(self.eval_cfg["dataset_name"]),
            "dataset_config_name": (
                str(self.eval_cfg["dataset_config_name"])
                if self.eval_cfg.get("dataset_config_name") is not None
                else None
            ),
            "dataset_split": str(self.eval_cfg["dataset_split"]),
            "question_field": question_field,
            "answer_field": answer_field,
            "num_examples": int(total),
            "num_correct": int(num_correct_pass1),
            "num_correct_pass1": int(num_correct_pass1),
            "accuracy": pass_at_1,
            "pass@1": pass_at_1,
            "pass_at_1": pass_at_1,
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
            "eval_seconds": float(elapsed_s),
            "model": str(model_name_or_path),
            "num_inference_gpus": int(self.num_inference_gpus),
            "inference_gpu_ids": [int(gpu_id) for gpu_id in self.inference_gpu_ids],
            "pass_k_enabled": bool(pass_k_enabled),
            "pass_k_num_samples": int(pass_k_num_samples),
            "outcomes_log_path": str(outcomes_log_path),
            "per_example_scores_path": str(per_example_scores_path),
        }
        _add_pass_1_sampling_metrics(metrics, pass_1_cfg)
        if pass_k_enabled:
            passk_token_lengths = _tokenize_text_lengths(self.tokenizer, passk_texts)
            pass_at_k = float(num_correct_passk / max(total, 1))
            passk_key = f"pass@{pass_k_num_samples}"
            passk_safe_key = f"pass_at_{pass_k_num_samples}"
            num_correct_passk_key = f"num_correct_pass{pass_k_num_samples}"
            empty_predictions_passk_key = f"empty_predictions_pass{pass_k_num_samples}"
            avg_completion_tokens_passk_key = f"avg_completion_tokens_pass{pass_k_num_samples}"
            avg_total_reward_passk_key = f"avg_total_reward_pass{pass_k_num_samples}"
            avg_format_reward_passk_key = f"avg_format_reward_pass{pass_k_num_samples}"
            avg_answer_reward_passk_key = f"avg_answer_reward_pass{pass_k_num_samples}"
            avg_length_reward_passk_key = f"avg_length_reward_pass{pass_k_num_samples}"
            metrics[passk_key] = pass_at_k
            metrics[passk_safe_key] = pass_at_k
            metrics[num_correct_passk_key] = int(num_correct_passk)
            metrics[empty_predictions_passk_key] = int(num_empty_passk)
            metrics[avg_completion_tokens_passk_key] = float(
                sum(passk_token_lengths) / max(len(passk_token_lengths), 1)
            )
            metrics[avg_total_reward_passk_key] = float(
                sum(passk_total_rewards) / max(len(passk_total_rewards), 1)
            )
            metrics[avg_format_reward_passk_key] = float(
                sum(passk_format_rewards) / max(len(passk_format_rewards), 1)
            )
            metrics[avg_answer_reward_passk_key] = float(
                sum(passk_answer_rewards) / max(len(passk_answer_rewards), 1)
            )
            metrics[avg_length_reward_passk_key] = float(
                sum(passk_length_rewards) / max(len(passk_length_rewards), 1)
            )
            if pass_k_num_samples == 8:
                metrics["pass@8"] = pass_at_k
                metrics["pass_at_8"] = pass_at_k
                metrics["num_correct_pass8"] = int(num_correct_passk)
                metrics["empty_predictions_pass8"] = int(num_empty_passk)
                metrics["avg_completion_tokens_pass8"] = float(
                    sum(passk_token_lengths) / max(len(passk_token_lengths), 1)
                )

        save_json_atomic(metrics, metrics_path)
        return {
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "outcomes_log_path": str(outcomes_log_path),
            "per_example_scores_path": str(per_example_scores_path),
            "run_record_path": str(run_record_path),
        }

    def close(self) -> None:
        self._stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True, type=str)
    parser.add_argument("--control-file", required=True, type=str)
    args = parser.parse_args()

    worker_config_path = Path(args.worker_config).resolve()
    control_path = Path(args.control_file).resolve()
    worker_cfg = _load_json(worker_config_path)
    if worker_cfg is None:
        raise FileNotFoundError(f"Missing worker config: {worker_config_path}")

    eval_cfg = dict(worker_cfg["eval_cfg"])
    vllm_cfg = dict(worker_cfg["vllm_cfg"])
    num_inference_gpus = int(worker_cfg["num_inference_gpus"])
    inference_gpu_ids = [int(gpu_id) for gpu_id in worker_cfg["inference_gpu_ids"]]
    logs_dir = ensure_dir(Path(worker_cfg["logs_dir"]).resolve())
    metrics_dir = ensure_dir(Path(worker_cfg["metrics_dir"]).resolve())
    seed = int(worker_cfg["seed"])
    tokenizer_name_or_path = str(worker_cfg.get("tokenizer_name_or_path") or "").strip()
    repeat_pause_s = max(0.0, float(eval_cfg.get("repeat_pause_s", 0.0)))
    max_repeats_raw = eval_cfg.get("max_repeats_per_checkpoint")
    max_repeats = None if max_repeats_raw is None else int(max_repeats_raw)
    if max_repeats is not None and max_repeats <= 0:
        raise ValueError("max_repeats_per_checkpoint must be positive or null.")

    if not tokenizer_name_or_path:
        raise ValueError("Missing tokenizer_name_or_path in worker config.")
    tokenizer = load_causal_lm_tokenizer(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )

    batch, prompts = _collect_eval_batch(eval_cfg)
    evaluator = _PersistentEvaluator(
        vllm_cfg=vllm_cfg,
        num_inference_gpus=num_inference_gpus,
        inference_gpu_ids=inference_gpu_ids,
        log_root=logs_dir,
        metrics_root=metrics_dir,
        eval_cfg=eval_cfg,
        batch=batch,
        prompts=prompts,
        tokenizer=tokenizer,
    )

    current_generation = -1
    current_step: Optional[int] = None
    current_epoch: Optional[float] = None
    current_checkpoint_dir: Optional[str] = None
    current_records: List[Dict[str, Any]] = []

    try:
        while True:
            control = _load_json(control_path)
            if control is None:
                time.sleep(1.0)
                continue

            stop_requested = bool(control.get("stop", False))
            target_checkpoint_dir = control.get("checkpoint_dir")
            target_generation = int(control.get("generation", current_generation))

            if stop_requested and not target_checkpoint_dir:
                break

            if target_checkpoint_dir is None:
                time.sleep(1.0)
                continue

            if target_generation != current_generation or target_checkpoint_dir != current_checkpoint_dir:
                current_generation = target_generation
                current_checkpoint_dir = str(target_checkpoint_dir)
                current_step = int(control["step"])
                epoch_raw = control.get("epoch")
                current_epoch = None if epoch_raw is None else float(epoch_raw)
                runs_path = metrics_dir / f"eval_step_{current_step:06d}_runs.jsonl"
                current_records = _load_jsonl_records(runs_path)
                logger.info(
                    "Switched to checkpoint=%s step=%d generation=%d existing_runs=%d",
                    current_checkpoint_dir,
                    current_step,
                    current_generation,
                    len(current_records),
                )

            if current_step is None or current_checkpoint_dir is None:
                time.sleep(1.0)
                continue

            if max_repeats is not None and len(current_records) >= max_repeats:
                if stop_requested:
                    break
                time.sleep(max(repeat_pause_s, 1.0))
                continue

            run_index = len(current_records) + 1
            eval_seed = seed + current_generation * 100000 + run_index
            eval_result = evaluator.evaluate_once(
                model_name_or_path=current_checkpoint_dir,
                step=current_step,
                seed=eval_seed,
                run_index=run_index,
            )
            metrics = dict(eval_result["metrics"])
            payload = {
                "step": int(current_step),
                "epoch": current_epoch,
                "checkpoint_dir": str(current_checkpoint_dir),
                "generation": int(current_generation),
                "run_index": int(run_index),
                "seed": int(eval_seed),
                "metrics": metrics,
                "metrics_path": str(eval_result["metrics_path"]),
                "outcomes_log_path": str(eval_result["outcomes_log_path"]),
                "per_example_scores_path": str(eval_result["per_example_scores_path"]),
                "run_record_path": str(eval_result["run_record_path"]),
                "evaluated_at": time.time(),
            }
            save_json_atomic(payload, Path(str(eval_result["run_record_path"])))
            current_records.append(payload)
            runs_path = metrics_dir / f"eval_step_{current_step:06d}_runs.jsonl"
            _append_jsonl(runs_path, payload)
            summary = _build_summary(
                current_records,
                checkpoint_dir=current_checkpoint_dir,
                step=current_step,
                epoch=current_epoch,
            )
            summary = _materialize_extrema_artifacts(metrics_dir, int(current_step), summary)
            summary["runs_path"] = str(runs_path)
            save_json_atomic(summary, metrics_dir / f"eval_step_{current_step:06d}_summary.json")
            _update_history(metrics_dir / "eval_history.json", summary)
            logger.info(
                "Continuous external eval accuracy=%.4f (%d/%d) step=%d run=%d",
                float(metrics.get("accuracy", 0.0)),
                int(metrics.get("num_correct", 0)),
                int(metrics.get("num_examples", 0)),
                int(current_step),
                int(run_index),
            )

            control_after = _load_json(control_path) or {}
            new_generation = int(control_after.get("generation", current_generation))
            stop_after = bool(control_after.get("stop", False))
            if stop_after and new_generation == current_generation:
                break
            if new_generation != current_generation:
                continue
            if repeat_pause_s > 0.0:
                time.sleep(repeat_pause_s)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
