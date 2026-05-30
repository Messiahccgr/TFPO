import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compare_curriculum import (
    build_stage_records,
    get_compare_stage_names,
    load_compare_curriculum_spec,
)
from src.utils import setup_logger


logger = setup_logger("prepare_deepseek_compare_curriculum")


DEFAULT_BIGMATH_INPUT = PROJECT_ROOT / "train_data" / "Big-Math-RL-Verified-2k"
DEFAULT_COMPETITION_INPUT = (
    PROJECT_ROOT / "train_data" / "competition_math_bigmath_format_2k"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "train_data" / "deepseek_compare_curriculum"


def _load_train_split(path: Path):
    dataset_dict = load_from_disk(str(path))
    if "train" in dataset_dict:
        return dataset_dict["train"]
    return dataset_dict


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _has_complete_cache(output_root: Path, stage_names: list[str]) -> bool:
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        return False
    return all((output_root / stage_name).exists() for stage_name in stage_names)


def _save_stage_dataset(*, stage_dir: Path, records: list[Dict[str, Any]], overwrite: bool) -> None:
    if stage_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Stage dataset already exists: {stage_dir}. Pass --overwrite to rebuild."
            )
        shutil.rmtree(stage_dir)
    dataset = Dataset.from_list(records)
    DatasetDict({"train": dataset}).save_to_disk(str(stage_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare static DeepSeek compare curriculum stage datasets."
    )
    parser.add_argument("--bigmath-input", type=str, default=str(DEFAULT_BIGMATH_INPUT))
    parser.add_argument(
        "--competition-input",
        type=str,
        default=str(DEFAULT_COMPETITION_INPUT),
    )
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bigmath_input = Path(args.bigmath_input).resolve()
    competition_input = Path(args.competition_input).resolve()
    output_root = Path(args.output_root).resolve()

    spec = load_compare_curriculum_spec()
    stage_names = get_compare_stage_names(spec)
    if output_root.exists() and _has_complete_cache(output_root, stage_names) and not args.overwrite:
        logger.info("Reusing existing compare curriculum cache at %s", output_root)
        summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bigmath_dataset = _load_train_split(bigmath_input)
    competition_dataset = _load_train_split(competition_input)
    logger.info(
        "Loaded compare curriculum inputs | bigmath=%d competition=%d",
        len(bigmath_dataset),
        len(competition_dataset),
    )

    stages_summary: Dict[str, Any] = {}
    for stage_name in stage_names:
        stage_payload = build_stage_records(
            bigmath_dataset=bigmath_dataset,
            competition_dataset=competition_dataset,
            stage_name=stage_name,
            seed=int(args.seed),
            spec=spec,
        )
        stage_dir = output_root / stage_name
        _save_stage_dataset(
            stage_dir=stage_dir,
            records=stage_payload["records"],
            overwrite=True,
        )
        stage_summary = {
            "stage_name": stage_name,
            "num_examples": len(stage_payload["records"]),
            "source_counts": dict(stage_payload["source_counts"]),
            "bigmath_bucket": stage_payload["bigmath_bucket"],
            "competition_group": stage_payload["competition_group"],
            "stage_spec": dict(stage_payload["stage_spec"]),
            "output_dir": str(stage_dir),
        }
        stages_summary[stage_name] = stage_summary
        _write_json(stage_dir / "stage_summary.json", stage_summary)

    summary = {
        "output_root": str(output_root),
        "seed": int(args.seed),
        "bigmath_input": str(bigmath_input),
        "competition_input": str(competition_input),
        "stages": stages_summary,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
