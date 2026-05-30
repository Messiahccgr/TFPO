from typing import Any

try:
    from accelerate import PartialState
except Exception:  # pragma: no cover
    PartialState = None  # type: ignore


_DIST_STATE: Any = None


class _SingleProcessState:
    process_index = 0
    num_processes = 1
    is_main_process = True
    is_local_main_process = True

    @staticmethod
    def wait_for_everyone() -> None:
        return


def get_dist_state() -> Any:
    global _DIST_STATE
    if _DIST_STATE is not None:
        return _DIST_STATE
    if PartialState is None:
        _DIST_STATE = _SingleProcessState()
        return _DIST_STATE
    try:
        _DIST_STATE = PartialState()
    except Exception:
        _DIST_STATE = _SingleProcessState()
    return _DIST_STATE


def is_main_process() -> bool:
    return bool(get_dist_state().is_main_process)


def process_index() -> int:
    return int(get_dist_state().process_index)


def num_processes() -> int:
    return int(get_dist_state().num_processes)


def barrier() -> None:
    get_dist_state().wait_for_everyone()

