import json
import logging
import os
import random
import socket
import subprocess
import signal
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch


def setup_logger(name: str = "TriPO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_json_atomic(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def exclude_dataset_indices(dataset: Any, indices_to_exclude: Sequence[int]):
    normalized = sorted({int(idx) for idx in indices_to_exclude})
    if len(normalized) == 0:
        return dataset, 0

    dataset_len = len(dataset)
    invalid = [idx for idx in normalized if idx < 0 or idx >= dataset_len]
    if invalid:
        raise ValueError(
            "Excluded dataset indices out of range: "
            f"{invalid[:10]} (dataset size={dataset_len})"
        )

    excluded = set(normalized)
    keep_indices = [idx for idx in range(dataset_len) if idx not in excluded]
    return dataset.select(keep_indices), len(normalized)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def get_init_model_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / "init_model"


def local_model_dir_name(model_name_or_path: str) -> str:
    normalized = str(model_name_or_path).strip().rstrip("/\\")
    if not normalized:
        raise ValueError("`model_name_or_path` must not be empty.")
    return normalized.replace("\\", "/").split("/")[-1]


def resolve_init_model_path(
    model_name_or_path: str,
    project_root: Path,
    *,
    must_exist: bool = True,
) -> Path:
    raw_value = str(model_name_or_path).strip()
    if not raw_value:
        raise ValueError("`model_name_or_path` must not be empty.")

    path = Path(raw_value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        project_relative = Path(project_root) / path
        if project_relative not in candidates:
            candidates.append(project_relative)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    local_candidate = get_init_model_root(project_root) / local_model_dir_name(raw_value)
    if must_exist and not local_candidate.exists():
        raise FileNotFoundError(
            "Local init model directory not found: "
            f"{local_candidate}. Download it first with "
            f"`python init_model/download_model.py --model-id {raw_value}`."
        )
    return local_candidate.resolve()


def _load_model_config_metadata(model_path: Path) -> Dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def _is_qwen2_family_model(model_name_or_path: str, model_path: Path) -> bool:
    raw_name = str(model_name_or_path).strip().lower()
    if any(token in raw_name for token in ("qwen2", "qwen2.5", "deepseek-r1-distill-qwen")):
        return True

    metadata = _load_model_config_metadata(Path(model_path))
    model_type = str(metadata.get("model_type", "")).strip().lower()
    if model_type == "qwen2":
        return True

    architectures = metadata.get("architectures")
    if isinstance(architectures, list):
        for architecture in architectures:
            if "qwen2" in str(architecture).strip().lower():
                return True

    return False


def resolve_attn_implementation(
    requested_attn_implementation: Any,
    *,
    model_name_or_path: str,
    model_path: Path,
) -> tuple[str | None, str]:
    if requested_attn_implementation is not None:
        normalized = str(requested_attn_implementation).strip()
        if normalized:
            return normalized, "configured"

    if _is_qwen2_family_model(model_name_or_path, model_path):
        return "eager", "auto:qwen2_stability"

    return None, "default"


def terminate_process_tree(
    process: subprocess.Popen,
    logger: logging.Logger,
    wait_timeout_s: float = 30.0,
) -> None:
    if process.poll() is not None:
        try:
            process.wait(timeout=wait_timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Process pid=%s already exited but wait() timed out.",
                process.pid,
            )
        return

    pid = int(process.pid)
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=wait_timeout_s,
            )
            if result.returncode != 0 and process.poll() is None:
                logger.warning(
                    "taskkill returned code %s for pid=%s; falling back to kill().",
                    result.returncode,
                    pid,
                )
                process.kill()
                process.wait(timeout=wait_timeout_s)
                return
            process.wait(timeout=wait_timeout_s)
            return

        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=min(wait_timeout_s, 10.0))
            return
        except subprocess.TimeoutExpired:
            logger.warning(
                "Process group pgid=%s did not exit after SIGTERM; sending SIGKILL.",
                pgid,
            )
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=max(wait_timeout_s - 10.0, 1.0))
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        logger.error("Timed out waiting for pid=%s to exit.", pid)
    except Exception as exc:
        logger.warning(
            "Failed to terminate process tree for pid=%s cleanly: %s",
            pid,
            exc,
        )
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=wait_timeout_s)
            except subprocess.TimeoutExpired:
                logger.error(
                    "Timed out waiting for pid=%s after fallback kill().",
                    pid,
                )
