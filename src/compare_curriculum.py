import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.curriculum import group_indices_by_value, ranked_bucket_indices
from src.utils import setup_logger


logger = setup_logger("compare_curriculum")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_CURRICULUM_SPEC_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "deepseek_compare_curriculum_spec.json"
)


def load_compare_curriculum_spec() -> Dict[str, Any]:
    with COMPARE_CURRICULUM_SPEC_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_compare_stage_names(spec: Dict[str, Any] | None = None) -> List[str]:
    resolved_spec = load_compare_curriculum_spec() if spec is None else spec
    return [str(stage_name) for stage_name in resolved_spec["stage_order"]]


def get_compare_stage_spec(stage_name: str, spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resolved_spec = load_compare_curriculum_spec() if spec is None else spec
    stages = dict(resolved_spec["stages"])
    if stage_name not in stages:
        raise KeyError(
            f"Unknown compare curriculum stage={stage_name!r}. "
            f"Available: {sorted(stages)}"
        )
    return dict(stages[stage_name])


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _get_dataset_column(dataset: Any, column_name: str) -> List[Any]:
    if hasattr(dataset, "__getitem__"):
        try:
            column = dataset[column_name]
            return list(column)
        except (TypeError, KeyError, IndexError):
            pass
    return [row.get(column_name) for row in dataset]


def _get_dataset_record(dataset: Any, index: int) -> Dict[str, Any]:
    record = dataset[int(index)]
    return dict(record)


def _normalize_bigmath_record(example: Dict[str, Any], *, bucket_name: str, stage_name: str) -> Dict[str, Any] | None:
    problem = _normalize_text(example.get("problem"))
    answer = _normalize_text(example.get("answer"))
    if not problem or not answer:
        return None
    return {
        "problem": problem,
        "answer": answer,
        "source_dataset": "bigmath",
        "bigmath_bucket": str(bucket_name),
        "competition_group": None,
        "curriculum_stage": str(stage_name),
    }


def _normalize_competition_record(
    example: Dict[str, Any],
    *,
    group_name: str,
    stage_name: str,
) -> Dict[str, Any] | None:
    problem = _normalize_text(example.get("problem"))
    answer = _normalize_text(example.get("answer"))
    if not problem or not answer:
        return None
    return {
        "problem": problem,
        "answer": answer,
        "source_dataset": "competition_math",
        "bigmath_bucket": None,
        "competition_group": str(group_name),
        "curriculum_stage": str(stage_name),
    }


def build_stage_records(
    *,
    bigmath_dataset,
    competition_dataset,
    stage_name: str,
    seed: int,
    spec: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_spec = load_compare_curriculum_spec() if spec is None else spec
    stage_spec = get_compare_stage_spec(stage_name, resolved_spec)

    bigmath_bucket_indices = ranked_bucket_indices(
        _get_dataset_column(bigmath_dataset, "llama8b_solve_rate"),
        dict(resolved_spec["bigmath_bucket_weights"]),
        descending=True,
    )
    competition_group_indices = group_indices_by_value(
        _get_dataset_column(competition_dataset, "level"),
        dict(resolved_spec["competition_level_groups"]),
    )

    bigmath_bucket_name = str(stage_spec["bigmath_bucket"])
    competition_group_name = str(stage_spec["competition_group"])
    selected_bigmath_indices = list(bigmath_bucket_indices.get(bigmath_bucket_name, []))
    selected_competition_indices = list(
        competition_group_indices.get(competition_group_name, [])
    )

    records: List[Dict[str, Any]] = []
    for dataset_idx in selected_bigmath_indices:
        record = _normalize_bigmath_record(
            _get_dataset_record(bigmath_dataset, int(dataset_idx)),
            bucket_name=bigmath_bucket_name,
            stage_name=stage_name,
        )
        if record is not None:
            records.append(record)

    for dataset_idx in selected_competition_indices:
        record = _normalize_competition_record(
            _get_dataset_record(competition_dataset, int(dataset_idx)),
            group_name=competition_group_name,
            stage_name=stage_name,
        )
        if record is not None:
            records.append(record)

    if len(records) > 1:
        rng = random.Random(int(seed) + get_compare_stage_names(resolved_spec).index(stage_name) * 1009)
        rng.shuffle(records)

    source_counts = {
        "bigmath": sum(1 for record in records if record["source_dataset"] == "bigmath"),
        "competition_math": sum(
            1 for record in records if record["source_dataset"] == "competition_math"
        ),
    }
    logger.info(
        "Prepared compare curriculum stage=%s size=%d source_counts=%s",
        stage_name,
        len(records),
        source_counts,
    )
    return {
        "records": records,
        "source_counts": source_counts,
        "bigmath_bucket": bigmath_bucket_name,
        "competition_group": competition_group_name,
        "stage_spec": stage_spec,
    }
