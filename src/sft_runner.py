import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM

from src.config import load_config
from src.distributed import get_dist_state
from src.runner import (
    DEFAULT_MATH_QUESTION_TEMPLATE,
    _format_prompt,
    _load_dataset_auto,
    _resolve_deepspeed_cfg,
    _resolve_dtype,
    _resolve_eval_cfg,
)
from src.sft import MixedSFTDataset, SFTSource, train_sft_stage
from src.tokenization import align_model_special_tokens, load_causal_lm_tokenizer
from src.utils import (
    ensure_dir,
    resolve_attn_implementation,
    resolve_init_model_path,
    set_seed,
    setup_logger,
)


logger = setup_logger("sft_runner")


def _load_sft_sources(
    *,
    sft_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    seed: int,
) -> List[SFTSource]:
    sources: List[SFTSource] = []
    for source_cfg in sft_cfg.get("datasets", []):
        dataset_name = str(source_cfg["dataset_name"])
        dataset_split = str(source_cfg.get("dataset_split", "train"))
        dataset = _load_dataset_auto(dataset_name, split=dataset_split)

        max_samples = source_cfg.get("max_samples")
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples < len(dataset):
                dataset = dataset.shuffle(seed=seed).select(range(max_samples))

        source = SFTSource(
            name=str(source_cfg.get("name", dataset_name)),
            dataset=dataset,
            question_field=str(source_cfg["question_field"]),
            response_field=str(source_cfg["response_field"]),
            question_template=str(
                source_cfg.get(
                    "question_template",
                    data_cfg.get("question_template", DEFAULT_MATH_QUESTION_TEMPLATE),
                )
            ),
            append_final_answer_from_field=source_cfg.get("append_final_answer_from_field"),
        )
        sources.append(source)
        logger.info(
            "Loaded SFT source=%s split=%s size=%d",
            source.name,
            dataset_split,
            len(dataset),
        )
    return sources


def _build_eval_examples(
    *,
    eval_dataset,
    eval_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    eval_data_cfg = {
        "question_template": str(
            eval_cfg.get("question_template", DEFAULT_MATH_QUESTION_TEMPLATE)
        ),
        "question_field": str(eval_cfg.get("question_field", "problem")),
    }
    examples: List[Dict[str, Any]] = []
    for idx in range(len(eval_dataset)):
        sample = eval_dataset[idx]
        prompt_text = _format_prompt(sample, eval_data_cfg)
        examples.append(
            {
                "prompt_text": prompt_text,
                "sample": sample,
            }
        )
    return examples


def run_sft(cfg: Dict[str, Any], *, best_model_path_file: Optional[Path] = None) -> str:
    sft_cfg = cfg.get("sft", {})
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    inference_cfg = cfg["inference"]
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

    if not bool(sft_cfg.get("enabled", False)):
        logger.info("SFT disabled. Using initial model path: %s", initial_model_path)
        if best_model_path_file is not None:
            best_model_path_file.parent.mkdir(parents=True, exist_ok=True)
            best_model_path_file.write_text(str(initial_model_path), encoding="utf-8")
        return str(initial_model_path)

    dist_state = get_dist_state()
    world_size = int(dist_state.num_processes)
    if world_size < 2:
        raise ValueError(
            "Cold-start SFT is configured to run before RL and should be launched with "
            "at least 2 processes on the 2xH100 node."
        )

    deepspeed_cfg = _resolve_deepspeed_cfg(cfg)

    seed = int(cfg["seed"])
    set_seed(seed)

    base_output_dir = Path(sft_cfg["output_dir"]).resolve()
    run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = ensure_dir(
        base_output_dir.with_name(f"{base_output_dir.name}_{run_timestamp}")
    )

    if dist_state.is_main_process:
        logger.info(
            "Starting SFT cold start | output_dir=%s | world_size=%d | model=%s",
            output_dir,
            world_size,
            initial_model_path,
        )

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
    if bool(sft_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    sources = _load_sft_sources(sft_cfg=sft_cfg, data_cfg=data_cfg, seed=seed)
    train_dataset = MixedSFTDataset(sources)
    if len(train_dataset) == 0:
        raise ValueError("SFT is enabled but the combined SFT dataset is empty.")
    logger.info("Combined SFT dataset size: %d", len(train_dataset))

    eval_cfg = _resolve_eval_cfg(cfg, data_cfg=data_cfg, inference_cfg=inference_cfg)
    eval_dataset = _load_dataset_auto(
        str(eval_cfg["dataset_name"]),
        split=str(eval_cfg["dataset_split"]),
    )
    if eval_cfg.get("max_samples") is not None:
        max_samples = int(eval_cfg["max_samples"])
        if max_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(max_samples))
    eval_examples = _build_eval_examples(eval_dataset=eval_dataset, eval_cfg=eval_cfg)
    logger.info("Loaded SFT eval dataset size: %d", len(eval_examples))

    summary = train_sft_stage(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_examples=eval_examples,
        sft_cfg=sft_cfg,
        eval_cfg=eval_cfg,
        deepspeed_cfg=deepspeed_cfg,
        output_dir=output_dir,
        seed=seed,
        best_model_path_file=best_model_path_file,
    )
    if dist_state.is_main_process:
        logger.info(
            "Finished SFT cold start | best_accuracy=%.4f | best_model=%s",
            float(summary["best_accuracy"]),
            summary["best_model_path"],
        )
    return str(summary["best_model_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help='Comma-separated jsonnet configs, e.g. "a.jsonnet,b.jsonnet"',
    )
    parser.add_argument(
        "--best-model-path-file",
        type=str,
        default=None,
        help="Optional file used to export the best SFT checkpoint path for the RL stage.",
    )
    args = parser.parse_args()
    cfg = load_config(args.configs)
    best_model_path_file = (
        Path(args.best_model_path_file).resolve()
        if args.best_model_path_file is not None
        else None
    )
    run_sft(cfg, best_model_path_file=best_model_path_file)
