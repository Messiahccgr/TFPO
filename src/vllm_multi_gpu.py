import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.utils import find_free_port, setup_logger, terminate_process_tree
from src.vllm import VLLMClient, _resolve_vllm_runtime_env


logger = setup_logger("vllm_multi_gpu")


class MultiGPUVLLMServer:
    """支持tensor并行的vLLM服务器，可以利用多GPU加速推理"""

    def __init__(self, cfg: Dict[str, Any], log_dir: Path, num_gpus: int = 1):
        self.cfg = cfg
        self.log_dir = log_dir
        self.num_gpus = max(1, int(num_gpus))
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
            str(self.cfg.get("gpu_memory_utilization", 0.85)),
            "--max-num-seqs",
            str(self.cfg.get("max_num_seqs", 256)),
        ]

        if self.cfg.get("trust_remote_code", True):
            cmd.append("--trust-remote-code")

        # 添加tensor并行支持
        if self.num_gpus > 1:
            cmd.extend([
                "--tensor-parallel-size",
                str(self.num_gpus),
            ])
            logger.info("Enabling tensor parallelism with %d GPUs", self.num_gpus)
            # The custom all-reduce CUDA kernel relies on direct P2P/IPC between
            # the TP GPUs. On many shared/SLURM nodes that handoff fails at
            # runtime ("custom_all_reduce.cuh:453 'invalid argument'") even when
            # the driver claims P2P works, killing the worker after CUDA-graph
            # capture. Default to disabling it so vLLM falls back to the robust
            # NCCL all-reduce; set vllm.disable_custom_all_reduce=false to re-enable.
            if self.cfg.get("disable_custom_all_reduce", True):
                cmd.append("--disable-custom-all-reduce")

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
        # 设置多GPU
        if self.num_gpus > 1:
            gpu_ids = self.cfg.get("gpu_ids", list(range(self.num_gpus)))
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        else:
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
            "Launched vLLM server pid=%s on port=%s model=%s gpus=%s gpu_names=%s "
            "tensor_parallel=%d VLLM_USE_V1=%s VLLM_ATTENTION_BACKEND=%s log=%s",
            self.process.pid,
            self.port,
            model_name_or_path,
            env["CUDA_VISIBLE_DEVICES"],
            ",".join(gpu_names) if gpu_names else "<unknown>",
            self.num_gpus,
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
                    logger.info("vLLM server is ready at %s", url)
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
        logger.info("vLLM server stopped")
