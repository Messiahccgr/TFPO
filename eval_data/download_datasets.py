from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

try:
    from datasets import DatasetDict, load_dataset
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - used for operator guidance.
    raise SystemExit(
        "Missing data dependencies. Install `datasets` and `huggingface_hub` first."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent

ALL_TARGETS = ("math500", "gsm8k", "aime24")

SNAPSHOT_SPECS = {
    "gsm8k": {
        "repo_id": "openai/gsm8k",
        "revision": "main",
        "dest": EVAL_ROOT / "GSM8K",
        "allow_patterns": [
            ".gitattributes",
            "README.md",
            "eval.yaml",
            "main/*.parquet",
            "socratic/*.parquet",
        ],
    },
    "aime24": {
        "repo_id": "HuggingFaceH4/aime_2024",
        "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "dest": EVAL_ROOT / "AIME24",
        "allow_patterns": [
            ".gitattributes",
            "README.md",
            "data/*.parquet",
        ],
    },
}


def _replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_snapshot(
    *,
    repo_id: str,
    revision: str,
    dest: Path,
    allow_patterns: Iterable[str],
) -> None:
    _replace_dir(dest)
    print(f"Downloading {repo_id}@{revision} -> {dest.relative_to(PROJECT_ROOT)}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=list(allow_patterns),
    )


def _download_math500() -> None:
    dest = EVAL_ROOT / "MATH-500"
    revision = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    _replace_dir(dest)
    print(
        "Downloading HuggingFaceH4/MATH-500@"
        f"{revision} -> {dest.relative_to(PROJECT_ROOT)}"
    )
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=revision)
    DatasetDict({"test": dataset}).save_to_disk(str(dest))


def _selected_targets(args: argparse.Namespace) -> list[str]:
    if not args.dataset or "all" in args.dataset:
        return list(ALL_TARGETS)
    return [target for target in ALL_TARGETS if target in set(args.dataset)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download local evaluation datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("all", *ALL_TARGETS),
        help="Dataset target to download. Repeat for multiple targets. Defaults to all.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    for target in _selected_targets(args):
        if target == "math500":
            _download_math500()
            continue
        spec = SNAPSHOT_SPECS[target]
        _download_snapshot(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            dest=spec["dest"],
            allow_patterns=spec["allow_patterns"],
        )


if __name__ == "__main__":
    main()
