import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils import exclude_dataset_indices, setup_logger


logger = setup_logger("curriculum")


@dataclass
class CurriculumStage:
    name: str
    iteration_start: int
    iteration_end: int
    group_weights: Dict[str, float] = field(default_factory=dict)
    sampling_mode: str = "uniform_with_replacement"
    bigmath_bucket: Optional[str] = None
    competition_group: Optional[str] = None

    def contains(self, iteration_1based: int) -> bool:
        return self.iteration_start <= iteration_1based <= self.iteration_end

    @property
    def is_legacy(self) -> bool:
        return self.bigmath_bucket is not None and self.competition_group is not None


@dataclass
class CurriculumSample:
    examples: List[Dict[str, Any]]
    question_ids: List[str]
    info: Dict[str, Any]


def _load_dataset_auto(name_or_path: str, split: str):
    from datasets import load_dataset, load_from_disk

    path = Path(name_or_path)
    if path.exists() and path.is_dir():
        logger.info("Loading curriculum dataset from local path: %s (split=%s)", path, split)
        if (path / "dataset_info.json").exists() or (path / "state.json").exists():
            ds = load_from_disk(str(path))
            if hasattr(ds, "__getitem__") and split in ds:
                return ds[split]
            return ds

        split_dir = path / split
        if split_dir.exists() and split_dir.is_dir():
            if (split_dir / "dataset_info.json").exists() or (split_dir / "state.json").exists():
                return load_from_disk(str(split_dir))
            parquet_files = list(split_dir.glob("*.parquet"))
            if parquet_files:
                return load_dataset(
                    "parquet",
                    data_files=[str(f) for f in sorted(parquet_files)],
                    split="train",
                )

        data_dir = path / "data"
        if data_dir.exists() and data_dir.is_dir():
            split_parquets = sorted(data_dir.glob(f"{split}-*.parquet"))
            if split_parquets:
                return load_dataset(
                    "parquet",
                    data_files=[str(f) for f in split_parquets],
                    split="train",
                )
            all_parquets = sorted(data_dir.glob("*.parquet"))
            if all_parquets:
                return load_dataset(
                    "parquet",
                    data_files=[str(f) for f in all_parquets],
                    split="train",
                )

        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            return load_dataset(
                "parquet",
                data_files=[str(f) for f in parquet_files],
                split="train",
            )

        json_files = sorted(path.glob("*.json"))
        if json_files:
            return load_dataset(
                "json",
                data_files=[str(f) for f in json_files],
                split="train",
            )

        ds = load_from_disk(str(path))
        if hasattr(ds, "__getitem__") and split in ds:
            return ds[split]
        return ds

    return load_dataset(name_or_path, split=split)


def allocate_proportional_counts(total_count: int, weights: Dict[str, float]) -> Dict[str, int]:
    counts = {key: 0 for key in weights}
    if total_count <= 0:
        return counts

    positive_keys = [key for key, weight in weights.items() if float(weight) > 0]
    if len(positive_keys) == 0:
        return counts

    total_weight = sum(float(weights[key]) for key in positive_keys)
    raw_counts: Dict[str, float] = {}
    for key in positive_keys:
        raw_counts[key] = float(total_count) * float(weights[key]) / max(total_weight, 1e-12)
        counts[key] = int(math.floor(raw_counts[key]))

    remainder = int(total_count - sum(counts.values()))
    if remainder > 0:
        ranked_remainders = sorted(
            positive_keys,
            key=lambda key: (raw_counts[key] - counts[key], float(weights[key]), key),
            reverse=True,
        )
        for idx in range(remainder):
            counts[ranked_remainders[idx % len(ranked_remainders)]] += 1

    return counts


def ranked_bucket_indices(
    scores: Sequence[Any],
    bucket_weights: Dict[str, float],
    *,
    descending: bool = True,
) -> Dict[str, List[int]]:
    normalized_scores: List[float] = []
    for score in scores:
        try:
            normalized_scores.append(float(score))
        except (TypeError, ValueError):
            normalized_scores.append(float("-inf"))

    ranked_indices = sorted(
        range(len(normalized_scores)),
        key=lambda idx: normalized_scores[idx],
        reverse=descending,
    )
    bucket_sizes = allocate_proportional_counts(len(ranked_indices), bucket_weights)

    buckets: Dict[str, List[int]] = {}
    cursor = 0
    for bucket_name in bucket_weights:
        bucket_size = int(bucket_sizes.get(bucket_name, 0))
        buckets[bucket_name] = ranked_indices[cursor : cursor + bucket_size]
        cursor += bucket_size
    return buckets


def group_indices_by_value(
    values: Sequence[Any],
    groups: Dict[str, Sequence[str]],
) -> Dict[str, List[int]]:
    normalized_lookup: Dict[str, str] = {}
    for group_name, group_values in groups.items():
        for group_value in group_values:
            normalized_lookup[str(group_value).strip()] = group_name

    grouped_indices: Dict[str, List[int]] = {group_name: [] for group_name in groups}
    unmatched = 0
    for idx, value in enumerate(values):
        normalized_value = str(value).strip()
        group_name = normalized_lookup.get(normalized_value)
        if group_name is None:
            unmatched += 1
            continue
        grouped_indices[group_name].append(idx)

    if unmatched > 0:
        logger.warning("Ignored %d curriculum examples with unmatched group values.", unmatched)
    return grouped_indices


def format_source_group_key(source_name: str, group_name: str) -> str:
    return f"{str(source_name).strip()}:{str(group_name).strip()}"


def parse_source_group_key(key: str) -> Tuple[str, str]:
    normalized = str(key).strip()
    source_name, sep, group_name = normalized.partition(":")
    if sep == "" or not source_name or not group_name:
        raise ValueError(
            f"Invalid curriculum group key {normalized!r}. Expected format 'source:group'."
        )
    return source_name, group_name


def parse_curriculum_stages(
    stage_cfgs: Sequence[Dict[str, Any]],
    *,
    num_iterations: int,
) -> List[CurriculumStage]:
    stages: List[CurriculumStage] = []
    for stage_cfg in stage_cfgs:
        stage_name = str(stage_cfg["name"])
        iteration_start = int(stage_cfg["iteration_start"])
        iteration_end = int(stage_cfg["iteration_end"])

        if "group_weights" in stage_cfg:
            group_weights: Dict[str, float] = {}
            for raw_key, raw_weight in dict(stage_cfg["group_weights"]).items():
                weight = float(raw_weight)
                if weight < 0:
                    raise ValueError(
                        f"Curriculum stage {stage_name!r} has negative weight for {raw_key!r}."
                    )
                if weight == 0:
                    continue
                key = str(raw_key)
                parse_source_group_key(key)
                group_weights[key] = weight
            if len(group_weights) == 0:
                raise ValueError(
                    f"Curriculum stage {stage_name!r} must define at least one positive group weight."
                )
            stages.append(
                CurriculumStage(
                    name=stage_name,
                    iteration_start=iteration_start,
                    iteration_end=iteration_end,
                    group_weights=group_weights,
                    sampling_mode=str(
                        stage_cfg.get("sampling_mode", "uniform_with_replacement")
                    ).strip(),
                )
            )
            continue

        if "bigmath_bucket" in stage_cfg and "competition_group" in stage_cfg:
            stages.append(
                CurriculumStage(
                    name=stage_name,
                    iteration_start=iteration_start,
                    iteration_end=iteration_end,
                    sampling_mode=str(
                        stage_cfg.get("sampling_mode", "legacy_proportional_to_subset_size")
                    ).strip(),
                    bigmath_bucket=str(stage_cfg["bigmath_bucket"]),
                    competition_group=str(stage_cfg["competition_group"]),
                )
            )
            continue

        raise ValueError(
            f"Curriculum stage {stage_name!r} must define either group_weights or "
            "the legacy bigmath_bucket / competition_group fields."
        )

    stages.sort(key=lambda stage: stage.iteration_start)

    expected_start = 1
    for stage in stages:
        if stage.iteration_start != expected_start:
            raise ValueError(
                "Curriculum stages must be contiguous and gap-free. "
                f"Expected start={expected_start}, got {stage.iteration_start} for {stage.name!r}."
            )
        if stage.iteration_end < stage.iteration_start:
            raise ValueError(
                f"Curriculum stage {stage.name!r} has invalid range "
                f"{stage.iteration_start}-{stage.iteration_end}."
            )
        expected_start = stage.iteration_end + 1

    if stages and stages[-1].iteration_end != int(num_iterations):
        raise ValueError(
            "Curriculum stages must cover the full run. "
            f"Last stage ends at {stages[-1].iteration_end}, expected {num_iterations}."
        )
    if len(stages) == 0 and int(num_iterations) > 0:
        raise ValueError("Curriculum is enabled but no curriculum stages are configured.")
    return stages


def resolve_curriculum_stage(
    iteration_1based: int,
    stages: Sequence[CurriculumStage],
) -> CurriculumStage:
    for stage in stages:
        if stage.contains(iteration_1based):
            return stage
    raise ValueError(f"No curriculum stage found for iteration {iteration_1based}.")


class _IndexPool:
    def __init__(
        self,
        indices: Sequence[int],
        *,
        sample_with_replacement: bool,
        shuffle_on_each_iteration: bool,
        seed: int,
    ):
        self.indices = [int(idx) for idx in indices]
        self.sample_with_replacement = bool(sample_with_replacement)
        self.shuffle_on_each_iteration = bool(shuffle_on_each_iteration)
        self.seed = int(seed)

        self._epoch = 0
        self._cursor = 0
        self._ordered_indices = list(self.indices)
        self._refresh_order()

    def _refresh_order(self) -> None:
        self._ordered_indices = list(self.indices)
        if self.shuffle_on_each_iteration and len(self._ordered_indices) > 1:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(self._ordered_indices)
        self._epoch += 1

    def sample(self, n: int, iteration: int) -> List[int]:
        if n <= 0 or len(self.indices) == 0:
            return []

        if self.sample_with_replacement:
            result: List[int] = []
            repeat_idx = 0
            while len(result) < n:
                ordered = list(self.indices)
                if self.shuffle_on_each_iteration and len(ordered) > 1:
                    rng = random.Random(self.seed + iteration + repeat_idx)
                    rng.shuffle(ordered)
                take = min(n - len(result), len(ordered))
                result.extend(ordered[:take])
                repeat_idx += 1
            return result

        result = []
        while len(result) < n:
            if self._cursor >= len(self._ordered_indices):
                self._cursor = 0
                self._refresh_order()
            take = min(n - len(result), len(self._ordered_indices) - self._cursor)
            result.extend(self._ordered_indices[self._cursor : self._cursor + take])
            self._cursor += take
        return result


class CurriculumSampler:
    def __init__(
        self,
        *,
        curriculum_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any],
        num_iterations: int,
        project_root: Path,
        seed: int,
    ):
        self.curriculum_cfg = dict(curriculum_cfg)
        self.data_cfg = dict(data_cfg)
        self.num_questions_per_iteration = int(data_cfg["num_questions_per_iteration"])
        self.sample_with_replacement = bool(data_cfg.get("sample_with_replacement", True))
        self.shuffle_on_each_iteration = bool(data_cfg.get("shuffle_on_each_iteration", True))
        self.phase_name = str(curriculum_cfg.get("phase_name", "curriculum"))
        self.project_root = Path(project_root).resolve()
        self.seed = int(seed)
        self.stages = parse_curriculum_stages(
            curriculum_cfg.get("stages", []),
            num_iterations=int(num_iterations),
        )

        self.legacy_mode = "sources" not in curriculum_cfg
        self.source_datasets: Dict[str, Any] = {}
        self.source_group_indices: Dict[str, Dict[str, List[int]]] = {}
        self.source_pools: Dict[str, Dict[str, _IndexPool]] = {}
        self.mixture_mode = str(
            curriculum_cfg.get("mixture_mode", "proportional_to_subset_size")
        ).strip()

        if self.legacy_mode:
            self._initialize_legacy_sources(curriculum_cfg)
        else:
            self._initialize_generic_sources(curriculum_cfg)

    def _resolve_dataset_name_or_path(self, name_or_path: str) -> str:
        raw_path = Path(str(name_or_path))
        if raw_path.is_absolute():
            return str(raw_path)

        project_candidate = (self.project_root / raw_path).resolve()
        if project_candidate.exists():
            return str(project_candidate)
        return str(name_or_path)

    def _load_source_dataset(
        self,
        *,
        source_name: str,
        source_cfg: Dict[str, Any],
        default_max_size: Optional[int],
        excluded_indices: Sequence[int],
        seed_offset: int,
    ):
        dataset_name = self._resolve_dataset_name_or_path(str(source_cfg["dataset_name"]))
        dataset_split = str(source_cfg.get("dataset_split", "train"))
        dataset = _load_dataset_auto(dataset_name, split=dataset_split)

        if excluded_indices:
            dataset, num_excluded = exclude_dataset_indices(dataset, excluded_indices)
            logger.info("Curriculum source=%s excluded %d rows by index.", source_name, num_excluded)

        max_dataset_size = source_cfg.get("max_dataset_size", default_max_size)
        if max_dataset_size is not None:
            max_dataset_size = int(max_dataset_size)
            if max_dataset_size < len(dataset):
                dataset = dataset.shuffle(seed=self.seed + seed_offset).select(range(max_dataset_size))
        logger.info(
            "Loaded curriculum source=%s dataset=%s split=%s size=%d",
            source_name,
            dataset_name,
            dataset_split,
            len(dataset),
        )
        return dataset

    def _initialize_legacy_sources(self, curriculum_cfg: Dict[str, Any]) -> None:
        if self.mixture_mode != "proportional_to_subset_size":
            raise ValueError(
                f"Unsupported curriculum.mixture_mode={self.mixture_mode!r} in legacy curriculum mode."
            )

        dataset_cfg = curriculum_cfg.get("datasets", {})
        bigmath_dataset_cfg = dataset_cfg.get("bigmath", {})
        competition_dataset_cfg = dataset_cfg.get("competition_math", {})
        if not bigmath_dataset_cfg or not competition_dataset_cfg:
            raise ValueError("Legacy curriculum config must define both bigmath and competition_math datasets.")

        bigmath_dataset = self._load_source_dataset(
            source_name="bigmath",
            source_cfg=bigmath_dataset_cfg,
            default_max_size=self.data_cfg.get("max_dataset_size"),
            excluded_indices=self.data_cfg.get("excluded_question_indices") or [],
            seed_offset=17,
        )
        competition_dataset = self._load_source_dataset(
            source_name="competition_math",
            source_cfg=competition_dataset_cfg,
            default_max_size=self.data_cfg.get("max_dataset_size"),
            excluded_indices=[],
            seed_offset=37,
        )

        bigmath_cfg = dict(curriculum_cfg.get("bigmath", {}))
        solve_rate_order = str(bigmath_cfg.get("solve_rate_order", "desc")).strip().lower()
        if solve_rate_order not in {"desc", "ascending", "asc", "descending"}:
            raise ValueError(
                "curriculum.bigmath.solve_rate_order must be one of "
                "'desc', 'descending', 'asc', or 'ascending'."
            )
        bigmath_bucket_weights = dict(bigmath_cfg.get("buckets", {}))
        if len(bigmath_bucket_weights) == 0:
            raise ValueError("curriculum.bigmath.buckets must not be empty.")

        competition_cfg = dict(curriculum_cfg.get("competition_math", {}))
        competition_groups_cfg = dict(competition_cfg.get("level_groups", {}))
        if len(competition_groups_cfg) == 0:
            raise ValueError("curriculum.competition_math.level_groups must not be empty.")

        bigmath_scores = bigmath_dataset["llama8b_solve_rate"]
        bigmath_group_indices = ranked_bucket_indices(
            bigmath_scores,
            bigmath_bucket_weights,
            descending=solve_rate_order in {"desc", "descending"},
        )
        competition_levels = competition_dataset["level"]
        initial_competition_group_indices = group_indices_by_value(
            competition_levels,
            competition_groups_cfg,
        )
        matched_competition_indices = sorted(
            {
                int(idx)
                for group_indices in initial_competition_group_indices.values()
                for idx in group_indices
            }
        )
        if len(matched_competition_indices) < len(competition_dataset):
            dropped_count = len(competition_dataset) - len(matched_competition_indices)
            logger.warning(
                "Dropping %d competition_math rows with unsupported level values from curriculum sampling.",
                dropped_count,
            )
            competition_dataset = competition_dataset.select(matched_competition_indices)
            competition_levels = competition_dataset["level"]
            competition_group_indices = group_indices_by_value(
                competition_levels,
                competition_groups_cfg,
            )
        else:
            competition_group_indices = initial_competition_group_indices

        self.source_datasets = {
            "bigmath": bigmath_dataset,
            "competition_math": competition_dataset,
        }
        self.source_group_indices = {
            "bigmath": bigmath_group_indices,
            "competition_math": competition_group_indices,
        }
        self.source_pools = {
            "bigmath": {
                bucket_name: _IndexPool(
                    indices=indices,
                    sample_with_replacement=self.sample_with_replacement,
                    shuffle_on_each_iteration=self.shuffle_on_each_iteration,
                    seed=self.seed + 1000 + bucket_idx * 101,
                )
                for bucket_idx, (bucket_name, indices) in enumerate(bigmath_group_indices.items())
            },
            "competition_math": {
                group_name: _IndexPool(
                    indices=indices,
                    sample_with_replacement=self.sample_with_replacement,
                    shuffle_on_each_iteration=self.shuffle_on_each_iteration,
                    seed=self.seed + 2000 + group_idx * 101,
                )
                for group_idx, (group_name, indices) in enumerate(competition_group_indices.items())
            },
        }

    def _initialize_generic_sources(self, curriculum_cfg: Dict[str, Any]) -> None:
        sources_cfg = dict(curriculum_cfg.get("sources", {}))
        if len(sources_cfg) == 0:
            raise ValueError("Curriculum config must define curriculum.sources in generic mode.")

        for source_idx, (source_name, raw_source_cfg) in enumerate(sources_cfg.items()):
            source_cfg = dict(raw_source_cfg)
            dataset = self._load_source_dataset(
                source_name=source_name,
                source_cfg=source_cfg,
                default_max_size=self.data_cfg.get("max_dataset_size"),
                excluded_indices=source_cfg.get("excluded_question_indices", []),
                seed_offset=500 + source_idx * 101,
            )
            grouping_cfg = dict(source_cfg.get("grouping", {}))
            grouping_type = str(grouping_cfg.get("type", "")).strip().lower()
            if grouping_type == "value_groups":
                group_field = str(grouping_cfg["field"])
                group_values = dict(grouping_cfg["groups"])
                group_indices = group_indices_by_value(dataset[group_field], group_values)
            elif grouping_type == "all":
                group_name = str(grouping_cfg.get("group_name", "all"))
                group_indices = {group_name: list(range(len(dataset)))}
            else:
                raise ValueError(
                    f"Unsupported curriculum.sources.{source_name}.grouping.type={grouping_type!r}."
                )

            if sum(len(indices) for indices in group_indices.values()) <= 0:
                raise ValueError(f"Curriculum source {source_name!r} has no usable grouped examples.")

            self.source_datasets[source_name] = dataset
            self.source_group_indices[source_name] = group_indices
            self.source_pools[source_name] = {
                group_name: _IndexPool(
                    indices=indices,
                    sample_with_replacement=self.sample_with_replacement,
                    shuffle_on_each_iteration=self.shuffle_on_each_iteration,
                    seed=self.seed + 4000 + source_idx * 1000 + group_idx * 101,
                )
                for group_idx, (group_name, indices) in enumerate(group_indices.items())
            }

        for stage in self.stages:
            if stage.is_legacy:
                raise ValueError(
                    f"Curriculum stage {stage.name!r} uses legacy fields but curriculum.sources is configured."
                )
            if stage.sampling_mode != "uniform_with_replacement":
                raise ValueError(
                    f"Unsupported stage sampling_mode={stage.sampling_mode!r} in generic curriculum mode."
                )
            for group_key in stage.group_weights:
                source_name, group_name = parse_source_group_key(group_key)
                if source_name not in self.source_pools:
                    raise KeyError(
                        f"Curriculum stage {stage.name!r} references unknown source {source_name!r}."
                    )
                if group_name not in self.source_pools[source_name]:
                    raise KeyError(
                        f"Curriculum stage {stage.name!r} references unknown group "
                        f"{group_name!r} for source {source_name!r}."
                    )
                if len(self.source_pools[source_name][group_name].indices) == 0:
                    raise ValueError(
                        f"Curriculum stage {stage.name!r} references empty group "
                        f"{group_name!r} for source {source_name!r}."
                    )

        logger.info(
            "Generic curriculum phase=%s | sources=%s",
            self.phase_name,
            {
                source_name: {
                    "size": len(self.source_datasets[source_name]),
                    "groups": {
                        group_name: len(indices)
                        for group_name, indices in self.source_group_indices[source_name].items()
                    },
                }
                for source_name in self.source_datasets
            },
        )

    def _aggregate_source_counts(self, source_group_counts: Dict[str, int]) -> Dict[str, int]:
        source_counts: Dict[str, int] = {}
        for group_key, count in source_group_counts.items():
            source_name, _ = parse_source_group_key(group_key)
            source_counts[source_name] = source_counts.get(source_name, 0) + int(count)
        return source_counts

    def _aggregate_source_subset_sizes(self, source_group_subset_sizes: Dict[str, int]) -> Dict[str, int]:
        source_subset_sizes: Dict[str, int] = {}
        for group_key, subset_size in source_group_subset_sizes.items():
            source_name, _ = parse_source_group_key(group_key)
            source_subset_sizes[source_name] = source_subset_sizes.get(source_name, 0) + int(subset_size)
        return source_subset_sizes

    def _shuffle_batch(
        self,
        *,
        batch_examples: List[Dict[str, Any]],
        question_ids: List[str],
        iteration: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not self.shuffle_on_each_iteration or len(batch_examples) <= 1:
            return batch_examples, question_ids
        combined = list(zip(batch_examples, question_ids))
        rng = random.Random(self.seed + 3000 + iteration)
        rng.shuffle(combined)
        return (
            [example for example, _ in combined],
            [question_id for _, question_id in combined],
        )

    def _sample_legacy(self, *, stage: CurriculumStage, iteration: int) -> CurriculumSample:
        bigmath_key = format_source_group_key("bigmath", str(stage.bigmath_bucket))
        competition_key = format_source_group_key("competition_math", str(stage.competition_group))
        bigmath_pool = self.source_pools["bigmath"][str(stage.bigmath_bucket)]
        competition_pool = self.source_pools["competition_math"][str(stage.competition_group)]
        source_group_subset_sizes = {
            bigmath_key: len(bigmath_pool.indices),
            competition_key: len(competition_pool.indices),
        }
        source_subset_sizes = self._aggregate_source_subset_sizes(source_group_subset_sizes)
        source_counts = allocate_proportional_counts(
            self.num_questions_per_iteration,
            source_subset_sizes,
        )
        if sum(source_counts.values()) <= 0:
            raise RuntimeError(
                f"Curriculum stage {stage.name!r} has no available samples in either source."
            )

        bigmath_indices = bigmath_pool.sample(source_counts.get("bigmath", 0), iteration)
        competition_indices = competition_pool.sample(
            source_counts.get("competition_math", 0),
            iteration,
        )
        source_group_counts = {
            bigmath_key: len(bigmath_indices),
            competition_key: len(competition_indices),
        }

        batch_examples: List[Dict[str, Any]] = []
        question_ids: List[str] = []
        for dataset_idx in bigmath_indices:
            example = dict(self.source_datasets["bigmath"][int(dataset_idx)])
            example["_curriculum_source"] = "bigmath"
            example["_curriculum_group"] = str(stage.bigmath_bucket)
            batch_examples.append(example)
            question_ids.append(f"bigmath:{int(dataset_idx)}")

        for dataset_idx in competition_indices:
            example = dict(self.source_datasets["competition_math"][int(dataset_idx)])
            example["_curriculum_source"] = "competition_math"
            example["_curriculum_group"] = str(stage.competition_group)
            batch_examples.append(example)
            question_ids.append(f"competition_math:{int(dataset_idx)}")

        batch_examples, question_ids = self._shuffle_batch(
            batch_examples=batch_examples,
            question_ids=question_ids,
            iteration=iteration,
        )

        return CurriculumSample(
            examples=batch_examples,
            question_ids=question_ids,
            info={
                "phase": self.phase_name,
                "stage": stage.name,
                "iteration": int(iteration) + 1,
                "sampling_mode": stage.sampling_mode,
                "mixture_mode": self.mixture_mode,
                "bigmath_bucket": stage.bigmath_bucket,
                "competition_group": stage.competition_group,
                "stage_group_weights": {bigmath_key: 1.0, competition_key: 1.0},
                "source_counts": source_counts,
                "source_subset_sizes": source_subset_sizes,
                "source_group_counts": source_group_counts,
                "source_group_subset_sizes": source_group_subset_sizes,
            },
        )

    def _sample_generic(self, *, stage: CurriculumStage, iteration: int) -> CurriculumSample:
        source_group_counts = allocate_proportional_counts(
            self.num_questions_per_iteration,
            stage.group_weights,
        )
        source_group_subset_sizes: Dict[str, int] = {}
        for group_key in stage.group_weights:
            source_name, group_name = parse_source_group_key(group_key)
            source_group_subset_sizes[group_key] = len(self.source_pools[source_name][group_name].indices)

        batch_examples: List[Dict[str, Any]] = []
        question_ids: List[str] = []
        for group_key, requested_count in source_group_counts.items():
            source_name, group_name = parse_source_group_key(group_key)
            sampled_indices = self.source_pools[source_name][group_name].sample(requested_count, iteration)
            for dataset_idx in sampled_indices:
                example = dict(self.source_datasets[source_name][int(dataset_idx)])
                example["_curriculum_source"] = source_name
                example["_curriculum_group"] = group_name
                batch_examples.append(example)
                question_ids.append(f"{source_name}:{int(dataset_idx)}")

        batch_examples, question_ids = self._shuffle_batch(
            batch_examples=batch_examples,
            question_ids=question_ids,
            iteration=iteration,
        )
        return CurriculumSample(
            examples=batch_examples,
            question_ids=question_ids,
            info={
                "phase": self.phase_name,
                "stage": stage.name,
                "iteration": int(iteration) + 1,
                "sampling_mode": stage.sampling_mode,
                "stage_group_weights": dict(stage.group_weights),
                "source_counts": self._aggregate_source_counts(source_group_counts),
                "source_subset_sizes": self._aggregate_source_subset_sizes(source_group_subset_sizes),
                "source_group_counts": source_group_counts,
                "source_group_subset_sizes": source_group_subset_sizes,
            },
        )

    def sample(self, iteration: int) -> CurriculumSample:
        stage = resolve_curriculum_stage(int(iteration) + 1, self.stages)
        if self.legacy_mode:
            return self._sample_legacy(stage=stage, iteration=iteration)
        return self._sample_generic(stage=stage, iteration=iteration)
