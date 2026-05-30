import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset, load_from_disk


def _bootstrap_python_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


PROJECT_ROOT = _bootstrap_python_path()

from src.reward import MathRewardFn, extract_gold_answer_text, extract_pred_answer
from src.tokenization import load_causal_lm_tokenizer
from src.utils import ensure_dir, resolve_init_model_path, save_json, set_seed, setup_logger
from src.vllm import VLLMClient, VLLMServer
from src.vllm_multi_gpu import MultiGPUVLLMServer


logger = setup_logger("eval_model")

DEFAULT_EVAL_DATASET = str((PROJECT_ROOT / "eval_data" / "MATH-500").resolve())
ITER_ACTOR_PATTERN = re.compile(r"iter_(\d+)_actor$")


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_eval_config(configs_csv: str) -> Dict[str, Any]:
    config_paths = [p.strip() for p in configs_csv.split(",") if p.strip()]
    if len(config_paths) == 0:
        raise ValueError("`--configs` is empty.")

    abs_paths = [Path(p).resolve() for p in config_paths]
    project_root = PROJECT_ROOT
    ext_vars = {k: v for k, v in os.environ.items() if k.startswith("APP_")}
    ext_vars["APP_SEED"] = ext_vars.get("APP_SEED", "42")
    jsonnet_expr = "+".join([f'(import "{p.as_posix()}")' for p in abs_paths])

    try:
        import _jsonnet

        cfg = json.loads(_jsonnet.evaluate_snippet("eval_cfg", jsonnet_expr, ext_vars=ext_vars))
    except Exception as exc:
        raise RuntimeError(f"Failed to load eval jsonnet config: {exc}") from exc

    cfg.setdefault("seed", int(ext_vars["APP_SEED"]))
    cfg.setdefault("output_dir", "experiments/closed_form")
    cfg.setdefault("model", {})
    cfg["model"].setdefault("actor_name_or_path", "Qwen/Qwen2.5-Math-7B-Instruct")
    cfg.setdefault("data", {})
    cfg.setdefault("inference", {})
    cfg.setdefault("vllm", {})
    cfg.setdefault("evaluation", {})

    cfg["_meta"] = {
        "config_paths": [str(p) for p in abs_paths],
        "project_root": str(project_root),
    }
    return cfg


def _format_prompt(example: Dict[str, Any], question_template: str, question_field: str) -> str:
    class _SafeFormatDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    values = dict(example)
    if question_field in example:
        values.setdefault("problem", example[question_field])
        values.setdefault("query", example[question_field])
    return question_template.format_map(_SafeFormatDict(values))


def _tokenize_text_lengths(tokenizer: Any, texts: List[str]) -> List[int]:
    if len(texts) == 0:
        return []
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def _checkpoint_has_weights(ckpt_dir: Path) -> bool:
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return False
    weight_candidates = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    if any((ckpt_dir / name).exists() for name in weight_candidates):
        return True
    if any(ckpt_dir.glob("pytorch_model-*.bin")):
        return True
    if any(ckpt_dir.glob("model-*.safetensors")):
        return True
    return False


def _find_latest_actor_checkpoint(checkpoints_dir: Path) -> Optional[Path]:
    if not checkpoints_dir.exists():
        return None
    best_iter = -1
    best_dir: Optional[Path] = None
    for candidate in checkpoints_dir.iterdir():
        if not candidate.is_dir():
            continue
        matched = ITER_ACTOR_PATTERN.match(candidate.name)
        if matched is None:
            continue
        if not _checkpoint_has_weights(candidate):
            continue
        iter_idx = int(matched.group(1))
        if iter_idx > best_iter:
            best_iter = iter_idx
            best_dir = candidate
    return best_dir


def _dataset_result_key(
    dataset_name: str,
    dataset_split: str,
    dataset_config_name: Optional[str] = None,
) -> str:
    raw_parts = [Path(str(dataset_name)).name or str(dataset_name)]
    if dataset_config_name:
        raw_parts.append(str(dataset_config_name))
    raw_parts.append(str(dataset_split))
    raw = "_".join(part for part in raw_parts if str(part).strip())
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower() or "eval"


def _format_run_label(timestamp_s: float) -> str:
    return datetime.fromtimestamp(timestamp_s).strftime("%Y-%m-%d_%H-%M-%S")


def _load_eval_dataset(
    *,
    dataset_name: str,
    dataset_split: str,
    dataset_config_name: Optional[str],
):
    ds_path = Path(dataset_name)

    def _load_local_parquet_files(parquet_files: List[Path]):
        logger.info(
            "Loading eval dataset from local parquet files: %s",
            [str(p) for p in parquet_files],
        )
        tables = [pq.read_table(str(path)) for path in parquet_files]
        if len(tables) == 1:
            table = tables[0]
        else:
            import pyarrow as pa

            table = pa.concat_tables(tables)
        return Dataset.from_list(table.to_pylist())

    effective_config_name = dataset_config_name
    if (
        ds_path.exists()
        and ds_path.is_dir()
        and dataset_config_name
        and (ds_path / dataset_config_name).exists()
        and (ds_path / dataset_config_name).is_dir()
    ):
        ds_path = ds_path / dataset_config_name
        effective_config_name = None

    if ds_path.exists() and ds_path.is_dir() and (ds_path / "dataset_dict.json").exists():
        logger.info("Loading eval dataset from local load_from_disk path: %s", ds_path)
        ds = load_from_disk(str(ds_path))
        if hasattr(ds, "__getitem__") and dataset_split in ds:
            return ds[dataset_split]
        return ds

    if ds_path.exists() and ds_path.is_dir():
        logger.info("Loading eval dataset from local path: %s (split=%s)", ds_path, dataset_split)

        if (ds_path / "dataset_info.json").exists() or (ds_path / "state.json").exists():
            ds = load_from_disk(str(ds_path))
            if hasattr(ds, "__getitem__") and dataset_split in ds:
                return ds[dataset_split]
            return ds

        split_dir = ds_path / dataset_split
        if split_dir.exists() and split_dir.is_dir():
            if (split_dir / "dataset_info.json").exists() or (split_dir / "state.json").exists():
                return load_from_disk(str(split_dir))
            parquet_files = sorted(split_dir.glob("*.parquet"))
            if parquet_files:
                return _load_local_parquet_files(parquet_files)

        data_dir = ds_path / "data"
        if data_dir.exists() and data_dir.is_dir():
            split_parquets = sorted(data_dir.glob(f"{dataset_split}-*.parquet"))
            if split_parquets:
                return _load_local_parquet_files(split_parquets)
            all_parquets = sorted(data_dir.glob("*.parquet"))
            if all_parquets:
                return _load_local_parquet_files(all_parquets)

        split_parquets = sorted(ds_path.glob(f"{dataset_split}-*.parquet"))
        if split_parquets:
            return _load_local_parquet_files(split_parquets)

        parquet_files = sorted(ds_path.glob("*.parquet"))
        if parquet_files:
            return _load_local_parquet_files(parquet_files)

        json_files = sorted(ds_path.glob("*.json"))
        if json_files:
            return load_dataset(
                "json",
                data_files=[str(f) for f in json_files],
                split="train",
            )

    load_kwargs: Dict[str, Any] = {
        "path": str(ds_path if ds_path.exists() and ds_path.is_dir() else dataset_name),
        "split": dataset_split,
    }
    if effective_config_name:
        load_kwargs["name"] = str(effective_config_name)

    if ds_path.exists() and ds_path.is_dir():
        logger.info(
            "Loading eval dataset from local dataset repo: %s (config=%s, split=%s)",
            ds_path,
            effective_config_name,
            dataset_split,
        )
    else:
        logger.info(
            "Loading eval dataset from Hugging Face: %s (config=%s, split=%s)",
            dataset_name,
            dataset_config_name,
            dataset_split,
        )
    return load_dataset(**load_kwargs)


def _resolve_eval_model_source(cfg: Dict[str, Any]) -> Tuple[str, str]:
    output_dir = Path(cfg["output_dir"]).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    project_root = Path(cfg["_meta"]["project_root"]).resolve()

    anchor_dir = checkpoints_dir / "anchor_ema"
    if _checkpoint_has_weights(anchor_dir):
        return str(anchor_dir), "anchor_ema"

    latest_actor = _find_latest_actor_checkpoint(checkpoints_dir)
    if latest_actor is not None:
        return str(latest_actor), latest_actor.name

    initial_model_path = resolve_init_model_path(
        str(cfg["model"]["actor_name_or_path"]),
        project_root=project_root,
    )
    return str(initial_model_path), "initial_actor"


def _resolve_local_init_or_raw_path(raw_value: str, *, project_root: Path) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError("Path-like config value must not be empty.")

    path = Path(value)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        project_relative = Path(project_root) / path
        if project_relative not in candidates:
            candidates.append(project_relative)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    try:
        local_candidate = resolve_init_model_path(
            value,
            project_root=project_root,
            must_exist=False,
        )
    except Exception:
        local_candidate = None

    if local_candidate is not None and Path(local_candidate).exists():
        return str(Path(local_candidate).resolve())

    return value


def _infer_supported_context_len(
    tokenizer: Any,
    *,
    tokenizer_source: str,
) -> Optional[int]:
    candidates: List[int] = []

    tokenizer_model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_model_max_length, int) and 0 < tokenizer_model_max_length < 1_000_000:
        candidates.append(int(tokenizer_model_max_length))

    tokenizer_path = Path(str(tokenizer_source))
    metadata_paths = [
        tokenizer_path / "config.json",
        tokenizer_path / "tokenizer_config.json",
    ]
    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue

        for key in ("max_position_embeddings", "model_max_length", "max_seq_len", "seq_length"):
            value = metadata.get(key)
            if isinstance(value, int) and 0 < value < 1_000_000:
                candidates.append(int(value))

        rope_scaling = metadata.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            rope_max = rope_scaling.get("max_position_embeddings")
            if isinstance(rope_max, int) and 0 < rope_max < 1_000_000:
                candidates.append(int(rope_max))

            rope_original = rope_scaling.get("original_max_position_embeddings")
            rope_factor = rope_scaling.get("factor")
            if (
                isinstance(rope_original, int)
                and 0 < rope_original < 1_000_000
                and isinstance(rope_factor, (int, float))
                and float(rope_factor) > 0
            ):
                candidates.append(int(float(rope_original) * float(rope_factor)))

    if not candidates:
        return None
    return max(candidates)


def _compute_dynamic_vllm_max_model_len(
    cfg: Dict[str, Any],
    *,
    model_source: str,
    prompts: List[str],
    max_tokens: int,
) -> Optional[int]:
    if len(prompts) == 0:
        return cfg.get("vllm", {}).get("max_model_len")

    project_root = Path(cfg["_meta"]["project_root"]).resolve()
    model_cfg = cfg.get("model", {})
    tokenizer_source_raw = (
        model_cfg.get("tokenizer_name_or_path")
        or model_cfg.get("actor_name_or_path")
        or model_source
    )
    tokenizer_source = _resolve_local_init_or_raw_path(
        str(tokenizer_source_raw),
        project_root=project_root,
    )

    tokenizer = load_causal_lm_tokenizer(
        tokenizer_source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    max_prompt_tokens = 0
    max_prompt_index = -1
    batch_size = 64
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            prompt_batch,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        for offset, input_ids in enumerate(encoded):
            prompt_len = len(input_ids)
            if prompt_len > max_prompt_tokens:
                max_prompt_tokens = prompt_len
                max_prompt_index = start + offset

    logger.info(
        "Longest prompt token length in eval batch: %d (sample_index=%d, num_prompts=%d)",
        max_prompt_tokens,
        max_prompt_index,
        len(prompts),
    )

    safety_margin = 16
    required_context_len = int(max_prompt_tokens) + int(max_tokens) + safety_margin
    supported_context_len = _infer_supported_context_len(
        tokenizer,
        tokenizer_source=tokenizer_source,
    )
    if supported_context_len is not None and required_context_len > supported_context_len:
        raise ValueError(
            "Requested generation length exceeds model context capacity: "
            f"max_prompt_tokens={max_prompt_tokens}, max_tokens={max_tokens}, "
            f"safety_margin={safety_margin}, required={required_context_len}, "
            f"supported={supported_context_len}. Reduce MAX_TOKENS or shorten prompts."
        )

    logger.info(
        "Dynamic vLLM max_model_len: max_prompt_tokens=%d max_tokens=%d "
        "safety_margin=%d -> %d (supported=%s)",
        max_prompt_tokens,
        max_tokens,
        safety_margin,
        required_context_len,
        supported_context_len if supported_context_len is not None else "unknown",
    )
    return required_context_len


def _build_eval_cfg(
    cfg: Dict[str, Any],
    *,
    max_samples_override: Optional[int],
    enable_pass_k_override: Optional[bool] = None,
    max_tokens_override: Optional[int] = None,
    pass_k_num_samples_override: Optional[int] = None,
    pass_k_temperature_override: Optional[float] = None,
    pass_k_top_p_override: Optional[float] = None,
    pass_k_max_tokens_override: Optional[int] = None,
    dataset_name_override: Optional[str] = None,
    dataset_config_override: Optional[str] = None,
    dataset_split_override: Optional[str] = None,
    question_field_override: Optional[str] = None,
    answer_field_override: Optional[str] = None,
    question_template_override: Optional[str] = None,
) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    inference_cfg = cfg.get("inference", {})
    user_eval_cfg = cfg.get("evaluation", {})
    defaults: Dict[str, Any] = {
        "dataset_name": user_eval_cfg.get("dataset_name", DEFAULT_EVAL_DATASET),
        "dataset_config_name": user_eval_cfg.get("dataset_config_name"),
        "dataset_split": user_eval_cfg.get("dataset_split", "test"),
        "question_field": user_eval_cfg.get(
            "question_field",
            data_cfg.get("question_field", "problem"),
        ),
        "answer_field": user_eval_cfg.get(
            "answer_field",
            data_cfg.get("answer_field", "answer"),
        ),
        "question_template": user_eval_cfg.get(
            "question_template",
            data_cfg.get("question_template"),
        ),
        "max_samples": user_eval_cfg.get("max_samples"),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(inference_cfg.get("max_tokens", 1024)),
        "enable_pass_k": _coerce_bool(user_eval_cfg.get("enable_pass_k", False)),
        "pass_1_temperature": float(user_eval_cfg.get("pass_1_temperature", 0.0)),
        "pass_1_top_p": float(user_eval_cfg.get("pass_1_top_p", 1.0)),
        "pass_1_max_tokens": int(
            user_eval_cfg.get("pass_1_max_tokens", inference_cfg.get("max_tokens", 1024))
        ),
        "pass_k_num_samples": int(user_eval_cfg.get("pass_k_num_samples", 8)),
        "pass_k_temperature": float(
            user_eval_cfg.get("pass_k_temperature", inference_cfg.get("temperature", 0.6))
        ),
        "pass_k_top_p": float(
            user_eval_cfg.get("pass_k_top_p", inference_cfg.get("top_p", 0.9))
        ),
        "pass_k_max_tokens": int(
            user_eval_cfg.get("pass_k_max_tokens", inference_cfg.get("max_tokens", 1024))
        ),
        "stop": inference_cfg.get("stop"),
        "request_timeout_s": int(inference_cfg.get("request_timeout_s", 300)),
        "max_parallel_requests": int(inference_cfg.get("max_parallel_requests", 64)),
    }
    if isinstance(user_eval_cfg, dict):
        defaults.update({k: v for k, v in user_eval_cfg.items() if k in defaults})
    if dataset_name_override is not None:
        defaults["dataset_name"] = str(dataset_name_override)
    if dataset_config_override is not None:
        defaults["dataset_config_name"] = str(dataset_config_override)
    if dataset_split_override is not None:
        defaults["dataset_split"] = str(dataset_split_override)
    if question_field_override is not None:
        defaults["question_field"] = str(question_field_override)
    if answer_field_override is not None:
        defaults["answer_field"] = str(answer_field_override)
    if question_template_override is not None:
        defaults["question_template"] = str(question_template_override)
    if max_samples_override is not None:
        defaults["max_samples"] = int(max_samples_override)
    if enable_pass_k_override is not None:
        defaults["enable_pass_k"] = bool(enable_pass_k_override)
    if max_tokens_override is not None:
        defaults["max_tokens"] = int(max_tokens_override)
    if pass_k_num_samples_override is not None:
        defaults["pass_k_num_samples"] = int(pass_k_num_samples_override)
    if pass_k_temperature_override is not None:
        defaults["pass_k_temperature"] = float(pass_k_temperature_override)
    if pass_k_top_p_override is not None:
        defaults["pass_k_top_p"] = float(pass_k_top_p_override)
    if pass_k_max_tokens_override is not None:
        defaults["pass_k_max_tokens"] = int(pass_k_max_tokens_override)
    return defaults


def _run_eval(
    *,
    model_source: str,
    source_type: str,
    cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    num_inference_gpus: int,
    gpu_ids: Optional[List[int]],
) -> Dict[str, Any]:
    set_seed(int(cfg["seed"]))
    output_dir = Path(cfg["output_dir"]).resolve()
    run_started_at = time.time()
    run_label = _format_run_label(run_started_at)
    logs_dir = ensure_dir(output_dir / "logs" / run_label)
    metrics_dir = ensure_dir(output_dir / "metrics")

    name_or_path = str(eval_cfg["dataset_name"])
    dataset_config_name = eval_cfg.get("dataset_config_name")
    split = str(eval_cfg["dataset_split"])
    eval_dataset = _load_eval_dataset(
        dataset_name=name_or_path,
        dataset_split=split,
        dataset_config_name=(
            str(dataset_config_name) if dataset_config_name is not None else None
        ),
    )
    if eval_cfg.get("max_samples") is not None:
        max_samples = int(eval_cfg["max_samples"])
        if max_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(max_samples))
    logger.info("Loaded eval dataset size=%d", len(eval_dataset))

    question_field = str(eval_cfg["question_field"])
    question_template = str(eval_cfg["question_template"])
    batch = [eval_dataset[i] for i in range(len(eval_dataset))]
    prompts = [
        _format_prompt(
            example=sample,
            question_template=question_template,
            question_field=question_field,
        )
        for sample in batch
    ]
    project_root = Path(cfg["_meta"]["project_root"]).resolve()
    model_cfg = cfg.get("model", {})
    tokenizer_source_raw = (
        model_cfg.get("tokenizer_name_or_path")
        or model_cfg.get("actor_name_or_path")
        or model_source
    )
    tokenizer_source = _resolve_local_init_or_raw_path(
        str(tokenizer_source_raw),
        project_root=project_root,
    )
    tokenizer = load_causal_lm_tokenizer(
        tokenizer_source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    vllm_cfg = dict(cfg["vllm"])
    dynamic_max_model_len = _compute_dynamic_vllm_max_model_len(
        cfg,
        model_source=model_source,
        prompts=prompts,
        max_tokens=int(eval_cfg["max_tokens"]),
    )
    if dynamic_max_model_len is not None:
        vllm_cfg["max_model_len"] = int(dynamic_max_model_len)
    if num_inference_gpus > 1:
        if gpu_ids is None:
            gpu_ids = list(range(num_inference_gpus))
        vllm_cfg["gpu_ids"] = gpu_ids
        server = MultiGPUVLLMServer(vllm_cfg, logs_dir, num_gpus=num_inference_gpus)
    else:
        server = VLLMServer(vllm_cfg, logs_dir)

    def _read_log_tail(path: Path, max_lines: int = 120) -> str:
        if not path.exists():
            return ""
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])

    def _raise_if_server_exited(stage: str) -> None:
        process = getattr(server, "process", None)
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            return
        raise RuntimeError(
            f"vLLM server exited during {stage} with code {return_code}. "
            f"See log: {server.log_path}"
        )

    start_time = run_started_at
    try:
        try:
            api_base = server.start(model_name_or_path=model_source, seed=int(cfg["seed"]) + 7)
            client = VLLMClient(
                api_base=api_base,
                model=str(model_source),
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
                seed=int(cfg["seed"]) + 11,
            )
            _raise_if_server_exited("pass@1 generation")
            enable_pass_k = bool(eval_cfg.get("enable_pass_k", False))
            configured_pass_k_num_samples = max(
                1, int(eval_cfg.get("pass_k_num_samples", 8))
            )
            if enable_pass_k:
                pass_k_num_samples = configured_pass_k_num_samples
                generated_passk = client.generate_batch(
                    prompts=prompts,
                    n=pass_k_num_samples,
                    temperature=float(eval_cfg.get("pass_k_temperature", 0.6)),
                    top_p=float(eval_cfg.get("pass_k_top_p", 0.9)),
                    max_tokens=int(
                        eval_cfg.get("pass_k_max_tokens", eval_cfg["max_tokens"])
                    ),
                    stop=eval_cfg.get("stop"),
                    seed=int(cfg["seed"]) + 100011,
                )
                _raise_if_server_exited(f"pass@{pass_k_num_samples} generation")
            else:
                pass_k_num_samples = 0
                generated_passk = [[] for _ in prompts]
        except Exception:
            log_tail = _read_log_tail(Path(server.log_path))
            if log_tail:
                logger.error("vLLM server log tail:\n%s", log_tail)
            raise
    finally:
        server.stop()

    reward_fn = MathRewardFn(answer_field=str(eval_cfg["answer_field"]))
    num_total = len(batch)
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
    outcomes_log_path = logs_dir / "evaluate_outcomes.log"
    per_example_scores_path = metrics_dir / f"per_example_scores_{run_label}.jsonl"
    with outcomes_log_path.open("w", encoding="utf-8") as outcome_f, per_example_scores_path.open(
        "w", encoding="utf-8"
    ) as score_f:
        outcome_f.write("Evaluation Outcomes\n")
        outcome_f.write(f"model_source: {model_source}\n")
        outcome_f.write(
            f"dataset: {eval_cfg['dataset_name']}"
            f"[{eval_cfg['dataset_split']}]"
            f" config={eval_cfg.get('dataset_config_name')}\n"
        )
        outcome_f.write(f"num_examples: {num_total}\n\n")

        for idx, (sample, greedy_choices, sampled_choices) in enumerate(
            zip(batch, generated_pass1, generated_passk)
        ):
            greedy_choice = greedy_choices[0] if greedy_choices else None
            pred = str(greedy_choice.get("text", "")) if greedy_choice else ""
            if not pred:
                num_empty_pass1 += 1
            pass1_texts.append(pred)
            greedy_score = reward_fn.score_completion(greedy_choice or pred, sample)
            pred_answer = extract_pred_answer(pred)
            raw_gold_answer = str(sample.get(eval_cfg["answer_field"], ""))
            gold_answer = extract_gold_answer_text(raw_gold_answer)
            parsed_gold_answer = reward_fn.describe_gold_math_verify_parse(sample)
            parsed_pred_answer = reward_fn.describe_prediction_math_verify_parse(
                greedy_choice or pred
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
            if pass_k_num_samples > 0:
                for choice in sampled_choices[:pass_k_num_samples]:
                    sampled_text = str(choice.get("text", ""))
                    sampled_score = reward_fn.score_completion(choice, sample)
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
                sample.get(
                    question_field,
                    _format_prompt(
                        example=sample,
                        question_template=question_template,
                        question_field=question_field,
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
            outcome_f.write(f"Pass@1 Reward Total: {greedy_score.total_reward:.6f}\n")
            outcome_f.write(f"Pass@1 Reward Format: {greedy_score.format_reward:.6f}\n")
            outcome_f.write(f"Pass@1 Reward Answer: {greedy_score.answer_reward:.6f}\n")
            outcome_f.write(f"Pass@1 Reward Length: {greedy_score.length_reward:.6f}\n")
            outcome_f.write(
                f"Pass@1 Correct: {'Correct' if is_correct_pass1 else 'wrong'}\n"
            )
            if pass_k_num_samples > 0:
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
                    "text": pred,
                    "answer_only_text": greedy_score.answer_only_text,
                    "total_reward": float(greedy_score.total_reward),
                    "format_reward": float(greedy_score.format_reward),
                    "answer_reward": float(greedy_score.answer_reward),
                    "length_reward": float(greedy_score.length_reward),
                    "is_correct": bool(is_correct_pass1),
                    "finish_reason": greedy_score.finish_reason,
                },
                "pass_k_enabled": bool(pass_k_num_samples > 0),
                "pass_k_num_samples": int(pass_k_num_samples),
                "pass_k_hit": bool(sampled_hit) if pass_k_num_samples > 0 else None,
                "pass_k_samples": sampled_reward_records,
            }
            score_f.write(json.dumps(per_example_record, ensure_ascii=False))
            score_f.write("\n")

    elapsed = time.time() - start_time
    pass1_token_lengths = _tokenize_text_lengths(tokenizer, pass1_texts)
    passk_token_lengths = _tokenize_text_lengths(tokenizer, passk_texts)
    accuracy = float(num_correct_pass1 / max(num_total, 1))
    pass_at_k = (
        float(num_correct_passk / max(num_total, 1)) if pass_k_num_samples > 0 else None
    )

    result = {
        "timestamp": int(run_started_at),
        "timestamp_readable": run_label,
        "model_source_type": source_type,
        "model_source": model_source,
        "dataset_name": str(eval_cfg["dataset_name"]),
        "dataset_config_name": (
            str(eval_cfg["dataset_config_name"])
            if eval_cfg.get("dataset_config_name") is not None
            else None
        ),
        "dataset_split": str(eval_cfg["dataset_split"]),
        "question_field": str(eval_cfg["question_field"]),
        "answer_field": str(eval_cfg["answer_field"]),
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
        "enable_pass_k": bool(pass_k_num_samples > 0),
        "pass_k_num_samples": int(pass_k_num_samples),
        "eval_seconds": float(elapsed),
        "num_inference_gpus": int(num_inference_gpus),
        "outcomes_log_path": str(outcomes_log_path),
        "per_example_scores_path": str(per_example_scores_path),
    }
    if pass_k_num_samples > 0:
        passk_key = f"pass@{pass_k_num_samples}"
        passk_safe_key = f"pass_at_{pass_k_num_samples}"
        num_correct_passk_key = f"num_correct_pass{pass_k_num_samples}"
        empty_predictions_passk_key = f"empty_predictions_pass{pass_k_num_samples}"
        avg_completion_tokens_passk_key = (
            f"avg_completion_tokens_pass{pass_k_num_samples}"
        )
        avg_total_reward_passk_key = f"avg_total_reward_pass{pass_k_num_samples}"
        avg_format_reward_passk_key = f"avg_format_reward_pass{pass_k_num_samples}"
        avg_answer_reward_passk_key = f"avg_answer_reward_pass{pass_k_num_samples}"
        avg_length_reward_passk_key = f"avg_length_reward_pass{pass_k_num_samples}"
        result[passk_key] = pass_at_k
        result[passk_safe_key] = pass_at_k
        result[num_correct_passk_key] = int(num_correct_passk)
        result[empty_predictions_passk_key] = int(num_empty_passk)
        result[avg_completion_tokens_passk_key] = float(
            sum(passk_token_lengths) / max(len(passk_token_lengths), 1)
        )
        result[avg_total_reward_passk_key] = float(
            sum(passk_total_rewards) / max(len(passk_total_rewards), 1)
        )
        result[avg_format_reward_passk_key] = float(
            sum(passk_format_rewards) / max(len(passk_format_rewards), 1)
        )
        result[avg_answer_reward_passk_key] = float(
            sum(passk_answer_rewards) / max(len(passk_answer_rewards), 1)
        )
        result[avg_length_reward_passk_key] = float(
            sum(passk_length_rewards) / max(len(passk_length_rewards), 1)
        )
        if pass_k_num_samples == 8:
            result["pass@8"] = pass_at_k
            result["pass_at_8"] = pass_at_k
            result["num_correct_pass8"] = int(num_correct_passk)
            result["empty_predictions_pass8"] = int(num_empty_passk)
            result["avg_completion_tokens_pass8"] = float(
                sum(passk_token_lengths) / max(len(passk_token_lengths), 1)
            )

    result_path = metrics_dir / f"eval_{run_label}.json"
    save_json(result, result_path)
    logger.info("Saved eval result to %s", result_path)
    logger.info("Saved per-example outcomes to %s", outcomes_log_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="General math evaluation entrypoint.")
    parser.add_argument(
        "--configs",
        "--config",
        dest="configs",
        type=str,
        required=True,
        help='Comma-separated jsonnet configs, e.g. "a.jsonnet,b.jsonnet"',
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick evaluation.",
    )
    pass_k_mode_group = parser.add_mutually_exclusive_group()
    pass_k_mode_group.add_argument(
        "--enable-pass-k",
        action="store_true",
        help="Enable pass@k sampling and metrics for this run.",
    )
    pass_k_mode_group.add_argument(
        "--disable-pass-k",
        action="store_true",
        help="Disable pass@k sampling and metrics for this run.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max new tokens override for generation length.",
    )
    parser.add_argument(
        "--pass-k-num-samples",
        type=int,
        default=None,
        help="Optional pass@k sample count override.",
    )
    parser.add_argument(
        "--pass-k-temperature",
        type=float,
        default=None,
        help="Optional pass@k temperature override.",
    )
    parser.add_argument(
        "--pass-k-top-p",
        type=float,
        default=None,
        help="Optional pass@k top-p override.",
    )
    parser.add_argument(
        "--pass-k-max-tokens",
        type=int,
        default=None,
        help="Optional pass@k max new tokens override.",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default=None,
        help="Optional explicit model path/name. If provided, skip auto-detection.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset name or local dataset path for evaluation.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset config name, for example main for GSM8K.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        help="Optional dataset split, for example test or validation.",
    )
    parser.add_argument(
        "--question-field",
        type=str,
        default=None,
        help="Optional question field name in the evaluation dataset.",
    )
    parser.add_argument(
        "--answer-field",
        type=str,
        default=None,
        help="Optional answer field name in the evaluation dataset.",
    )
    parser.add_argument(
        "--question-template",
        type=str,
        default=None,
        help="Optional prompt template override. It should contain {problem} or the chosen question field.",
    )
    parser.add_argument(
        "--num-inference-gpus",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallel evaluation.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help='Optional GPU ids for vLLM, e.g. "0,1".',
    )
    parser.add_argument(
        "--vllm-gpu-idx",
        type=int,
        default=None,
        help="Optional physical GPU index for single-GPU vLLM evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override for logs and metrics.",
    )
    args = parser.parse_args()

    cfg = load_eval_config(args.configs)
    if args.vllm_gpu_idx is not None:
        cfg.setdefault("vllm", {})
        cfg["vllm"]["gpu_idx"] = int(args.vllm_gpu_idx)
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)

    enable_pass_k_override: Optional[bool] = None
    if args.enable_pass_k:
        enable_pass_k_override = True
    elif args.disable_pass_k:
        enable_pass_k_override = False

    eval_cfg = _build_eval_cfg(
        cfg,
        max_samples_override=args.max_samples,
        enable_pass_k_override=enable_pass_k_override,
        max_tokens_override=args.max_tokens,
        pass_k_num_samples_override=args.pass_k_num_samples,
        pass_k_temperature_override=args.pass_k_temperature,
        pass_k_top_p_override=args.pass_k_top_p,
        pass_k_max_tokens_override=args.pass_k_max_tokens,
        dataset_name_override=args.dataset_name,
        dataset_config_override=args.dataset_config,
        dataset_split_override=args.dataset_split,
        question_field_override=args.question_field,
        answer_field_override=args.answer_field,
        question_template_override=args.question_template,
    )
    project_root = Path(cfg["_meta"]["project_root"]).resolve()

    if args.model_source is not None:
        model_source = str(
            resolve_init_model_path(
                args.model_source,
                project_root=project_root,
            )
        )
        source_type = "explicit"
    else:
        model_source, source_type = _resolve_eval_model_source(cfg)
    logger.info("Using model source: %s (%s)", model_source, source_type)

    gpu_ids: Optional[List[int]] = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]

    result = _run_eval(
        model_source=model_source,
        source_type=source_type,
        cfg=cfg,
        eval_cfg=eval_cfg,
        num_inference_gpus=max(1, int(args.num_inference_gpus)),
        gpu_ids=gpu_ids,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
