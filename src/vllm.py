import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils import find_free_port, setup_logger, terminate_process_tree


logger = setup_logger("vllm")


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _is_legacy_vllm_gpu_name(name: str) -> bool:
    normalized = str(name).strip().upper()
    return "V100" in normalized or "T4" in normalized


def _query_gpu_names(cuda_visible_devices: str) -> List[str]:
    selected_ids = [part.strip() for part in str(cuda_visible_devices).split(",") if part.strip()]
    if len(selected_ids) == 0:
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if result.returncode != 0:
        return []

    gpu_name_by_idx: Dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pieces = line.split(",", 1)
        if len(pieces) != 2:
            continue
        gpu_name_by_idx[pieces[0].strip()] = pieces[1].strip()

    return [gpu_name_by_idx[idx] for idx in selected_ids if idx in gpu_name_by_idx]


def _resolve_vllm_runtime_env(
    cfg: Dict[str, Any],
    *,
    inherited_env: Dict[str, str],
    cuda_visible_devices: str,
) -> Tuple[Dict[str, str], List[str], Optional[str]]:
    env = dict(inherited_env)
    # Strip torch-distributed rendezvous vars inherited from the parent. Under
    # multi-GPU training, accelerate/torchrun sets RANK/WORLD_SIZE/MASTER_PORT
    # etc. for the training ranks; if they leak into the vLLM subprocess, vLLM's
    # own tensor-parallel NCCL init (env:// rendezvous) collides with the
    # training process group and hangs at startup before loading weights.
    for _dist_var in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "ROLE_NAME",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCHELASTIC_ERROR_FILE",
    ):
        env.pop(_dist_var, None)
    gpu_names = _query_gpu_names(cuda_visible_devices)

    requested_use_v1 = _parse_optional_bool(cfg.get("use_v1_engine"))
    if requested_use_v1 is None and any(
        _is_legacy_vllm_gpu_name(name) for name in gpu_names
    ):
        env["VLLM_USE_V1"] = "0"
        note = (
            "Auto-set VLLM_USE_V1=0 for legacy GPU(s): "
            f"{', '.join(gpu_names)}"
        )
    elif requested_use_v1 is not None:
        env["VLLM_USE_V1"] = "1" if requested_use_v1 else "0"
        note = None
    else:
        note = None

    attention_backend = str(cfg.get("attention_backend", "")).strip()
    if attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = attention_backend

    return env, gpu_names, note


class VLLMServer:
    def __init__(self, cfg: Dict[str, Any], log_dir: Path):
        self.cfg = cfg
        self.log_dir = log_dir
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = cfg.get("port")
        self.host: str = cfg.get("host", "127.0.0.1")
        self.log_path: Path = self.log_dir / cfg.get("log_file", "vllm_server.log")
        self.log_file_handle = None

    @property
    def api_base(self) -> str:
        assert self.port is not None
        return f"http://{self.host}:{self.port}/v1"

    def start(self, model_name_or_path: str, seed: int) -> str:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("vLLM server is already running.")

        if self.port is None:
            self.port = find_free_port()

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_name_or_path),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--seed",
            str(seed),
            "--swap-space",
            str(self.cfg.get("swap_space", 16)),
            "--dtype",
            str(self.cfg.get("dtype", "bfloat16")),
            "--gpu-memory-utilization",
            str(self.cfg.get("gpu_memory_utilization", 0.9)),
            "--max-num-seqs",
            str(self.cfg.get("max_num_seqs", 256)),
        ]

        if self.cfg.get("trust_remote_code", True):
            cmd.append("--trust-remote-code")
        if self.cfg.get("enable_prefix_caching", False):
            cmd.append("--enable-prefix-caching")
        if self.cfg.get("disable_sliding_window", False):
            cmd.append("--disable-sliding-window")
        if self.cfg.get("disable_frontend_multiprocessing", False):
            cmd.append("--disable-frontend-multiprocessing")
        if self.cfg.get("enforce_eager", False):
            cmd.append("--enforce-eager")
        if self.cfg.get("max_model_len") is not None:
            cmd.extend(["--max-model-len", str(self.cfg["max_model_len"])])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("gpu_idx", 0))
        env, gpu_names, runtime_env_note = _resolve_vllm_runtime_env(
            self.cfg,
            inherited_env=env,
            cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"],
        )
        if runtime_env_note:
            logger.info("%s", runtime_env_note)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_handle = self.log_path.open("w", encoding="utf-8")
        popen_kwargs = {
            "env": env,
            "stdout": self.log_file_handle,
            "stderr": self.log_file_handle,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0,
            )
        else:
            popen_kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            cmd,
            **popen_kwargs,
        )
        logger.info(
            "Launched vLLM server pid=%s on port=%s model=%s gpu=%s gpu_names=%s "
            "VLLM_USE_V1=%s VLLM_ATTENTION_BACKEND=%s log=%s",
            self.process.pid,
            self.port,
            model_name_or_path,
            env["CUDA_VISIBLE_DEVICES"],
            ",".join(gpu_names) if gpu_names else "<unknown>",
            env.get("VLLM_USE_V1", "<inherit>"),
            env.get("VLLM_ATTENTION_BACKEND", "<inherit>"),
            self.log_path,
        )
        self._wait_until_ready(timeout_s=int(self.cfg.get("wait_timeout_s", 800)))
        return self.api_base

    def _wait_until_ready(self, timeout_s: int) -> None:
        start = time.time()
        url = f"http://{self.host}:{self.port}/v1/models"
        while True:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early with code {self.process.returncode}. "
                    f"See log: {self.log_path}"
                )

            try:
                resp = requests.get(
                    url,
                    timeout=5,
                    proxies={"http": None, "https": None},
                )
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass

            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for vLLM server at {url}")
            time.sleep(1.0)

    def stop(self) -> None:
        if self.process is None:
            return
        terminate_process_tree(self.process, logger=logger, wait_timeout_s=30.0)
        self.process = None
        if self.log_file_handle is not None:
            self.log_file_handle.close()
            self.log_file_handle = None


class VLLMClient:
    def __init__(
        self,
        api_base: str,
        model: str,
        request_timeout_s: int = 300,
        max_parallel_requests: int = 16,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        retry_backoff_max_s: float = 8.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.request_timeout_s = request_timeout_s
        self.max_parallel_requests = max_parallel_requests
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self.retry_backoff_max_s = float(retry_backoff_max_s)

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError):
            response = getattr(exc, "response", None)
            if response is None:
                return False
            status_code = int(getattr(response, "status_code", 0))
            # 429/5xx are generally transient.
            return status_code == 429 or status_code >= 500
        return False

    def _generate_one(
        self,
        prompt: str,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
        seed: int,
        logprobs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if stop is not None:
            payload["stop"] = stop
        if logprobs is not None:
            # vLLM /completions: integer >=0; we only need sampled-token logprobs.
            payload["logprobs"] = int(logprobs)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.api_base}/completions",
                    json=payload,
                    timeout=self.request_timeout_s,
                    proxies={"http": None, "https": None},
                )
                resp.raise_for_status()
                body = resp.json()
                outputs: List[Dict[str, Any]] = []
                for choice in body.get("choices", []):
                    out: Dict[str, Any] = {
                        "text": choice.get("text", ""),
                        "finish_reason": choice.get("finish_reason"),
                    }
                    lp = choice.get("logprobs")
                    if lp is not None:
                        # token_logprobs[i] may be None for the very first token in
                        # some vLLM builds; coerce to 0.0 to keep length aligned.
                        raw = lp.get("token_logprobs") or []
                        out["token_logprobs"] = [
                            float(x) if x is not None else 0.0 for x in raw
                        ]
                        out["logprob_tokens"] = list(lp.get("tokens") or [])
                    outputs.append(out)
                return outputs
            except Exception as exc:
                last_exc = exc
                retryable = self._is_retryable_exception(exc)
                if attempt >= self.max_retries or not retryable:
                    break
                backoff_s = min(
                    self.retry_backoff_s * (2**attempt),
                    self.retry_backoff_max_s,
                )
                logger.warning(
                    "vLLM request failed (attempt %d/%d, retry in %.1fs): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    backoff_s,
                    exc,
                )
                time.sleep(backoff_s)

        assert last_exc is not None
        raise last_exc

    def generate_batch(
        self,
        prompts: List[str],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
        seed: int,
        logprobs: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        if len(prompts) == 0:
            return []

        results: List[Optional[List[Dict[str, Any]]]] = [None for _ in prompts]
        max_workers = min(self.max_parallel_requests, len(prompts))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_one,
                    prompt=prompt,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed + i,
                    logprobs=logprobs,
                ): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error(
                        "vLLM generation failed for prompt_idx=%d after retries: %s",
                        idx,
                        exc,
                    )
                    results[idx] = []

        return [r if r is not None else [] for r in results]
