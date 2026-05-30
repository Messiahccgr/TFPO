from pathlib import Path
import sys


def _bootstrap_python_path() -> None:
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def _init_distributed_with_long_timeout() -> None:
    """Initialize the NCCL process group early with a long timeout.

    Under multi-GPU, rank 0 spends the first iteration on (slow) vLLM startup +
    rollout generation while the other ranks block on the next collective. The
    default 10-min NCCL watchdog timeout can fire during that window and abort
    the run. We init the PG here (accelerate's PartialState then reuses it) so
    the timeout covers a cold first iteration. No-op for single-process runs.
    """
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size <= 1:
        return
    try:
        from datetime import timedelta

        import torch
        import torch.distributed as dist

        if dist.is_available() and not dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            timeout_s = int(os.environ.get("APP_DDP_TIMEOUT_S", "36000") or "36000")
            dist.init_process_group(
                backend="nccl", timeout=timedelta(seconds=timeout_s)
            )
    except Exception as exc:  # fall back to accelerate's default init
        print(f"[run.py] early process-group init skipped: {exc}", flush=True)


def main() -> None:
    _bootstrap_python_path()
    _init_distributed_with_long_timeout()
    from src.runner import main as run_main

    run_main()


if __name__ == "__main__":
    main()
