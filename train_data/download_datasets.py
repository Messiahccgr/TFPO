from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
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
TRAIN_ROOT = Path(__file__).resolve().parent

ALL_TARGETS = (
    "bigmath_processed",
    "bigmath",
    "competition_math",
    "numina",
)
DEFAULT_TARGETS = (
    "bigmath_processed",
    "bigmath",
    "competition_math",
)

SNAPSHOT_SPECS = {
    "bigmath_processed": {
        "repo_id": "open-r1/Big-Math-RL-Verified-Processed",
        "revision": "c79efbb6d3b75e3a2bcc27a5c569119918132345",
        "dest": TRAIN_ROOT / "Big-Math-RL-Verified-Processed",
        "allow_patterns": [
            ".gitattributes",
            "README.md",
            "create_dataset.py",
            "level_1/*.parquet",
            "level_2/*.parquet",
            "level_3/*.parquet",
            "level_4/*.parquet",
            "level_5/*.parquet",
        ],
    },
    "competition_math": {
        "repo_id": "hendrycks/competition_math",
        "revision": "e839825f9ec5c6cfa585c654a59610969ec13993",
        "dest": TRAIN_ROOT / "competition_math",
        "allow_patterns": [
            ".gitattributes",
            "README.md",
            "data/*.parquet",
        ],
    },
    "numina": {
        "repo_id": "AI-MO/NuminaMath-1.5",
        "revision": "main",
        "dest": TRAIN_ROOT / "NuminaMath-1.5",
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


def _download_bigmath() -> None:
    dest = TRAIN_ROOT / "Big-Math-RL-Verified"
    revision = "c75d2f117cddfecb6bd08756e61e508e59732b21"
    _replace_dir(dest)
    print(
        "Downloading SynthLabsAI/Big-Math-RL-Verified@"
        f"{revision} -> {dest.relative_to(PROJECT_ROOT)}"
    )
    dataset = load_dataset(
        "SynthLabsAI/Big-Math-RL-Verified",
        split="train",
        revision=revision,
    )
    DatasetDict({"train": dataset}).save_to_disk(str(dest))


def _selected_targets(args: argparse.Namespace) -> list[str]:
    if args.dataset:
        selected = set(args.dataset)
        targets = set(ALL_TARGETS) if "all" in selected else selected
    else:
        targets = set(DEFAULT_TARGETS)

    if args.prepare_derived:
        targets.update({"bigmath", "competition_math", "numina"})

    return [target for target in ALL_TARGETS if target in targets]


def _run_prepare_script(script_name: str, tokenizer_name_or_path: str | None) -> None:
    cmd = [sys.executable, str(TRAIN_ROOT / script_name), "--overwrite"]
    if tokenizer_name_or_path:
        cmd.extend(["--tokenizer-name-or-path", tokenizer_name_or_path])
    print(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or rebuild local training datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("all", *ALL_TARGETS),
        help=(
            "Dataset target to download. Repeat this option for multiple targets. "
            "Defaults to the datasets used by current training configs."
        ),
    )
    parser.add_argument(
        "--prepare-derived",
        action="store_true",
        help="Rebuild Big-Math-RL-Verified-2k, competition_math_bigmath_format_2k, and NuminaMath-1.5-hard-verifiable_2k.",
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        default=os.environ.get("TOKENIZER_NAME_OR_PATH"),
        help="Tokenizer path/name used by the derived dataset preparation scripts.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    for target in _selected_targets(args):
        if target == "bigmath":
            _download_bigmath()
            continue
        spec = SNAPSHOT_SPECS[target]
        _download_snapshot(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            dest=spec["dest"],
            allow_patterns=spec["allow_patterns"],
        )

    if args.prepare_derived:
        _run_prepare_script("process.py", args.tokenizer_name_or_path)
        _run_prepare_script("prepare_bigmath_compatible_datasets.py", args.tokenizer_name_or_path)


if __name__ == "__main__":
    main()
