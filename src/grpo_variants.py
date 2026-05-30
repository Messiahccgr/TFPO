from typing import Any, Dict, Optional


SUPPORTED_GRPO_LOSS_TYPES = {"grpo", "dr_grpo", "dapo", "sapo", "reinforce"}
SUPPORTED_GRPO_ALIASES = {"gspo"}
SUPPORTED_IMPORTANCE_SAMPLING_LEVELS = {"token", "sequence"}


def resolve_grpo_variant(
    grpo_cfg: Dict[str, Any],
) -> tuple[str, str, Optional[str]]:
    requested_loss_type = str(grpo_cfg.get("loss_type", "grpo")).strip().lower()
    requested_importance_sampling_level = grpo_cfg.get("importance_sampling_level")
    if requested_importance_sampling_level is not None:
        requested_importance_sampling_level = (
            str(requested_importance_sampling_level).strip().lower()
        )

    if requested_loss_type == "rloo":
        raise ValueError(
            "loss_type='rloo' is not supported in run_grpo.py. "
            "RLOO is no longer part of the active config surface in this repo. "
            "Use run_rloo.py only with an explicit RLOO config that you manage separately."
        )

    effective_loss_type = requested_loss_type
    if requested_loss_type == "gspo":
        if requested_importance_sampling_level not in (None, "sequence"):
            raise ValueError(
                "loss_type='gspo' requires grpo.importance_sampling_level='sequence' "
                f"when it is set explicitly, got {requested_importance_sampling_level!r}."
            )
        effective_loss_type = "grpo"
        requested_importance_sampling_level = "sequence"

    if (
        requested_loss_type not in SUPPORTED_GRPO_LOSS_TYPES
        and requested_loss_type not in SUPPORTED_GRPO_ALIASES
    ):
        supported_values = sorted(SUPPORTED_GRPO_LOSS_TYPES | SUPPORTED_GRPO_ALIASES)
        raise ValueError(
            f"Unsupported GRPO loss_type={requested_loss_type!r}. "
            f"Supported values: {supported_values}."
        )

    if (
        requested_importance_sampling_level is not None
        and requested_importance_sampling_level
        not in SUPPORTED_IMPORTANCE_SAMPLING_LEVELS
    ):
        raise ValueError(
            "Unsupported GRPO importance_sampling_level="
            f"{requested_importance_sampling_level!r}. "
            f"Supported values: {sorted(SUPPORTED_IMPORTANCE_SAMPLING_LEVELS)}."
        )

    return (
        requested_loss_type,
        effective_loss_type,
        requested_importance_sampling_level,
    )
