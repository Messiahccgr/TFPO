from pathlib import Path
import sys


def _bootstrap_python_path() -> None:
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def main() -> None:
    _bootstrap_python_path()
    from src.sft_runner import main as run_main

    run_main()


if __name__ == "__main__":
    main()
