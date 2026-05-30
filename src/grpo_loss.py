"""GRPO-family token-level loss (5 variants) for the TFPO complementary objective.

This module provides a single ``compute_grpo_loss`` that produces a scalar loss
for a batch of (rollout, response_mask, valid_mask, advantage, old_logprob)
tuples, with branch logic for the variants reviewed in the paper Appendix D:

  - ``grpo``     : token-level ratio + PPO-style symmetric clip.
  - ``dr_grpo``  : same ratio/clip; aggregation divides by a constant L_max
                    instead of per-sequence length (removes length bias).
  - ``dapo``     : token-level ratio + decoupled clip (clip_low != clip_high).
  - ``gspo``     : sequence-level ratio (geometric mean of token ratios) +
                    symmetric clip; one ratio shared by all tokens in a sequence.
  - ``sapo``     : soft adaptive gate replaces hard clip; no min-of-two-surrogates,
                    a smooth re-weighting curve.

Only the tokens with ``valid_mask=True`` contribute to the loss. The caller is
responsible for computing ``valid_mask`` (TFPO covers some tokens; GRPO covers
the rest) and ``advantage`` (group-relative across rollouts of the same prompt).
"""

from __future__ import annotations

from typing import Optional

import torch


SUPPORTED_VARIANTS = {"grpo", "dr_grpo", "dapo", "gspo", "sapo"}


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


def _masked_sum_div(
    x: torch.Tensor, mask: torch.Tensor, constant: float
) -> torch.Tensor:
    """Sum over valid tokens, divide by a fixed constant (Dr.GRPO aggregation)."""
    mask = mask.to(x.dtype)
    denom = max(float(constant), 1.0)
    return (x * mask).sum() / denom


def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    variant: str = "grpo",
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    sapo_alpha: float = 1.0,
    dr_grpo_norm_constant: Optional[float] = None,
) -> torch.Tensor:
    """Returns a scalar loss tensor.

    Shapes:
      new_logprobs:  [B, L]   log pi_theta(a_t | s_t)  at response positions
      old_logprobs:  [B, L]   log pi_old(a_t | s_t)    from rollout time
      advantages:    [B] or [B, L]   group-relative advantage (broadcast-ok)
      valid_mask:    [B, L]   bool, True where this token feeds the GRPO loss
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported GRPO variant {variant!r}. "
            f"Supported: {sorted(SUPPORTED_VARIANTS)}"
        )

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1).expand_as(new_logprobs)
    advantages = advantages.detach()
    valid_mask_f = valid_mask.to(new_logprobs.dtype)

    log_ratio = new_logprobs - old_logprobs.detach()

    if variant == "gspo":
        # One ratio per sequence: geometric mean of token ratios over valid tokens.
        denom = valid_mask_f.sum(dim=-1).clamp(min=1.0)
        seq_log_ratio = (log_ratio * valid_mask_f).sum(dim=-1) / denom  # [B]
        seq_ratio = seq_log_ratio.exp().unsqueeze(-1).expand_as(new_logprobs)
        surr1 = seq_ratio * advantages
        clipped = torch.clamp(seq_ratio, 1.0 - clip_low, 1.0 + clip_high)
        surr2 = clipped * advantages
        per_token_loss = -torch.minimum(surr1, surr2)
        return _masked_mean(per_token_loss, valid_mask)

    if variant == "sapo":
        # Soft adaptive gate (smooth replacement for hard clip).
        ratio = log_ratio.exp()
        # 1 + tanh(alpha * (r-1)) / alpha keeps slope=1 at r=1 and saturates at
        # 1 +/- 1/alpha for large |r-1|.
        alpha = max(float(sapo_alpha), 1e-6)
        soft_ratio = 1.0 + torch.tanh(alpha * (ratio - 1.0)) / alpha
        per_token_loss = -soft_ratio * advantages
        return _masked_mean(per_token_loss, valid_mask)

    # grpo / dr_grpo / dapo : token-level ratio + (decoupled) clip.
    ratio = log_ratio.exp()
    surr1 = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    surr2 = clipped * advantages
    per_token_loss = -torch.minimum(surr1, surr2)

    if variant == "dr_grpo":
        # Dr.GRPO: divide by a constant rather than per-sequence length.
        # If unspecified, fall back to the batch's max valid length.
        if dr_grpo_norm_constant is None:
            with torch.no_grad():
                dr_grpo_norm_constant = float(
                    valid_mask_f.sum(dim=-1).max().clamp(min=1.0).item()
                )
        # Average across batch sequences after sum-then-divide-by-L_max.
        per_seq_loss = (per_token_loss * valid_mask_f).sum(dim=-1) / max(
            float(dr_grpo_norm_constant), 1.0
        )
        return per_seq_loss.mean()

    # grpo / dapo : standard masked mean.
    return _masked_mean(per_token_loss, valid_mask)


def compute_clip_fraction(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_low: float,
    clip_high: float,
) -> float:
    """Fraction of valid tokens whose importance ratio falls outside the clip band."""
    with torch.no_grad():
        ratio = (new_logprobs - old_logprobs).exp()
        out = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
        mask = valid_mask.bool()
        n_valid = int(mask.sum().item())
        if n_valid == 0:
            return 0.0
        return float((out & mask).sum().item()) / float(n_valid)
