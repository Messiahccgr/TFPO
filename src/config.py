import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        actor_override = os.environ.get("APP_ACTOR_NAME_OR_PATH")
        if actor_override is not None and actor_override.strip():
            model_cfg["actor_name_or_path"] = actor_override.strip()

        tokenizer_override = os.environ.get("APP_TOKENIZER_NAME_OR_PATH")
        if tokenizer_override is not None and tokenizer_override.strip():
            model_cfg["tokenizer_name_or_path"] = tokenizer_override.strip()

    grpo_cfg = cfg.get("grpo")
    if isinstance(grpo_cfg, dict):
        loss_type_override = os.environ.get("APP_GRPO_LOSS_TYPE")
        if loss_type_override is not None and loss_type_override.strip():
            grpo_cfg["loss_type"] = loss_type_override.strip()

        importance_sampling_level_override = os.environ.get(
            "APP_GRPO_IMPORTANCE_SAMPLING_LEVEL"
        )
        if (
            importance_sampling_level_override is not None
            and importance_sampling_level_override.strip()
        ):
            grpo_cfg["importance_sampling_level"] = (
                importance_sampling_level_override.strip()
            )

    vllm_cfg = cfg.get("vllm")
    if not isinstance(vllm_cfg, dict):
        return

    numeric_overrides = {
        "APP_VLLM_GPU_IDX": ("gpu_idx", int),
        "APP_VLLM_GPU_MEMORY_UTILIZATION": ("gpu_memory_utilization", float),
        "APP_VLLM_MAX_NUM_SEQS": ("max_num_seqs", int),
        "APP_VLLM_MAX_MODEL_LEN": ("max_model_len", int),
        "APP_VLLM_SWAP_SPACE": ("swap_space", int),
    }
    for env_name, (cfg_key, parser) in numeric_overrides.items():
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        vllm_cfg[cfg_key] = parser(raw)

    bool_overrides = {
        "APP_VLLM_ENFORCE_EAGER": "enforce_eager",
        "APP_VLLM_ENABLE_PREFIX_CACHING": "enable_prefix_caching",
        "APP_VLLM_DISABLE_SLIDING_WINDOW": "disable_sliding_window",
        "APP_VLLM_DISABLE_FRONTEND_MULTIPROCESSING": "disable_frontend_multiprocessing",
        "APP_VLLM_USE_V1_ENGINE": "use_v1_engine",
    }
    for env_name, cfg_key in bool_overrides.items():
        parsed = _parse_optional_bool(os.environ.get(env_name))
        if parsed is not None:
            vllm_cfg[cfg_key] = parsed

    string_overrides = {
        "APP_VLLM_ATTENTION_BACKEND": "attention_backend",
    }
    for env_name, cfg_key in string_overrides.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip():
            vllm_cfg[cfg_key] = raw.strip()


def _detect_project_root(config_path: str) -> str:
    resolved_path = Path(config_path).resolve()
    for parent in resolved_path.parents:
        if parent.name == "configs":
            return str(parent.parent)
    if len(resolved_path.parents) >= 2:
        return str(resolved_path.parents[1])
    return str(resolved_path.parent)


def load_config(configs_csv: str) -> Dict[str, Any]:
    config_paths = [p.strip() for p in configs_csv.split(",") if p.strip()]
    if len(config_paths) == 0:
        raise ValueError("`--configs` is empty.")

    abs_paths: List[str] = [Path(p).resolve().as_posix() for p in config_paths]
    ext_vars = {k: v for k, v in os.environ.items() if k.startswith("APP_")}
    ext_vars["APP_SEED"] = ext_vars.get("APP_SEED", "42")

    jsonnet_expr = "+".join([f'(import "{p}")' for p in abs_paths])

    try:
        import _jsonnet

        json_str = _jsonnet.evaluate_snippet("cfg", jsonnet_expr, ext_vars=ext_vars)
        cfg = json.loads(json_str)
    except Exception as exc:
        raise RuntimeError(f"Failed to load jsonnet config: {exc}") from exc

    _apply_env_overrides(cfg)
    cfg["_meta"] = {
        "config_paths": abs_paths,
        "project_root": _detect_project_root(abs_paths[0]),
    }
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    required = [
        "exp_name",
        "seed",
        "output_dir",
        "model",
        "data",
        "inference",
        "vllm",
        "algorithm",
        "train",
        "runtime",
        "deepspeed",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
