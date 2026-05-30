import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments

from src.distributed import barrier
from src.grpo_loss import compute_clip_fraction, compute_grpo_loss
from src.teacher import (
    EXAMPLE_KIND_GRPO,
    EXAMPLE_KIND_NEGATIVE,
    EXAMPLE_KIND_POSITIVE,
    KEY_CANDIDATE_PROBS,
    KEY_CANDIDATE_TOKEN_IDS,
    KEY_EXAMPLE_KIND,
    KEY_GRPO_ADVANTAGE,
    KEY_GRPO_HAS_OLD_LOGPROB,
    KEY_GRPO_OLD_LOGPROBS,
    KEY_GRPO_RESPONSE_MASK,
    KEY_GRPO_VALID_MASK,
    KEY_IS_NEGATIVE,
    KEY_NEGATIVE_TOKEN_ID,
    KEY_NEGATIVE_WEIGHT,
    KEY_PREFIX_WEIGHT,
    KEY_SAMPLE_WEIGHT,
)
from src.utils import setup_logger


logger = setup_logger("training")


class _CacheFlushWarningFilter(logging.Filter):
    """Drop DeepSpeed's per-step 'pytorch allocator cache flushes' WARNING.

    DeepSpeed emits it whenever the allocator flushes its cache during a step; with
    our comfortable per-GPU headroom it is pure noise (one line every step), so we
    filter just that message and keep every other DeepSpeed log intact.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "pytorch allocator cache flushes" not in record.getMessage()


logging.getLogger("DeepSpeed").addFilter(_CacheFlushWarningFilter())

KEY_CANDIDATE_MASK = "candidate_mask"
KEY_KIND_ID = "kind_id"

KIND_ID_POSITIVE = 0
KIND_ID_NEGATIVE = 1
KIND_ID_GRPO = 2

_KIND_TO_ID = {
    EXAMPLE_KIND_POSITIVE: KIND_ID_POSITIVE,
    EXAMPLE_KIND_NEGATIVE: KIND_ID_NEGATIVE,
    EXAMPLE_KIND_GRPO: KIND_ID_GRPO,
}


class TeacherPairsDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        max_sequence_length: Optional[int] = None,
    ):
        self.examples = examples
        self.max_sequence_length = max_sequence_length
        self.num_questions = len({ex["question_idx"] for ex in examples})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        kind_str = ex.get(KEY_EXAMPLE_KIND, EXAMPLE_KIND_POSITIVE)
        kind_id = _KIND_TO_ID[kind_str]

        query_ids = list(ex["query_token_ids"])
        if (
            self.max_sequence_length is not None
            and len(query_ids) > self.max_sequence_length
        ):
            # Keep the tail so the action state remains aligned for pos/neg.
            query_ids = query_ids[-self.max_sequence_length :]

        # Per-position auxiliary tensors for the GRPO branch; for pos/neg they are
        # zero-filled with the same length as input_ids so the collator can pad.
        seq_len = len(query_ids)
        if kind_id == KIND_ID_GRPO:
            response_mask = list(ex.get(KEY_GRPO_RESPONSE_MASK, [0] * seq_len))
            valid_mask = list(ex.get(KEY_GRPO_VALID_MASK, [0] * seq_len))
            old_logprobs = list(ex.get(KEY_GRPO_OLD_LOGPROBS, [0.0] * seq_len))
            # Truncation may shorten input_ids; align aux lengths too.
            response_mask = response_mask[: seq_len] + [0] * max(
                0, seq_len - len(response_mask)
            )
            valid_mask = valid_mask[: seq_len] + [0] * max(
                0, seq_len - len(valid_mask)
            )
            old_logprobs = old_logprobs[: seq_len] + [0.0] * max(
                0, seq_len - len(old_logprobs)
            )
        else:
            response_mask = [0] * seq_len
            valid_mask = [0] * seq_len
            old_logprobs = [0.0] * seq_len

        return {
            "input_ids": query_ids,
            "attention_mask": [1] * seq_len,
            KEY_KIND_ID: kind_id,
            KEY_PREFIX_WEIGHT: float(ex.get(KEY_PREFIX_WEIGHT, 1.0)),
            KEY_IS_NEGATIVE: float(ex.get(KEY_IS_NEGATIVE, 0.0)),
            KEY_NEGATIVE_WEIGHT: float(ex.get(KEY_NEGATIVE_WEIGHT, 1.0)),
            KEY_SAMPLE_WEIGHT: float(ex.get(KEY_SAMPLE_WEIGHT, 1.0)),
            KEY_CANDIDATE_TOKEN_IDS: list(ex.get(KEY_CANDIDATE_TOKEN_IDS, [])),
            KEY_CANDIDATE_PROBS: list(ex.get(KEY_CANDIDATE_PROBS, [])),
            KEY_NEGATIVE_TOKEN_ID: int(ex.get(KEY_NEGATIVE_TOKEN_ID, -100)),
            KEY_GRPO_RESPONSE_MASK: response_mask,
            KEY_GRPO_VALID_MASK: valid_mask,
            KEY_GRPO_OLD_LOGPROBS: old_logprobs,
            KEY_GRPO_ADVANTAGE: float(ex.get(KEY_GRPO_ADVANTAGE, 0.0)),
            KEY_GRPO_HAS_OLD_LOGPROB: bool(ex.get(KEY_GRPO_HAS_OLD_LOGPROB, False)),
        }


class TeacherDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
        ]
        response_mask = [
            torch.tensor(f[KEY_GRPO_RESPONSE_MASK], dtype=torch.long) for f in features
        ]
        valid_mask = [
            torch.tensor(f[KEY_GRPO_VALID_MASK], dtype=torch.bool) for f in features
        ]
        old_logprobs = [
            torch.tensor(f[KEY_GRPO_OLD_LOGPROBS], dtype=torch.float32) for f in features
        ]

        max_candidates = max(1, max(len(f[KEY_CANDIDATE_TOKEN_IDS]) for f in features))
        candidate_token_ids = torch.zeros(
            len(features), max_candidates, dtype=torch.long
        )
        candidate_probs = torch.zeros(
            len(features), max_candidates, dtype=torch.float32
        )
        candidate_mask = torch.zeros(
            len(features), max_candidates, dtype=torch.bool
        )
        negative_token_ids: List[int] = []

        for row_idx, feature in enumerate(features):
            toks = list(feature[KEY_CANDIDATE_TOKEN_IDS])
            probs = list(feature[KEY_CANDIDATE_PROBS])
            if toks:
                limit = min(len(toks), max_candidates)
                candidate_token_ids[row_idx, :limit] = torch.tensor(
                    toks[:limit], dtype=torch.long
                )
                candidate_probs[row_idx, :limit] = torch.tensor(
                    probs[:limit], dtype=torch.float32
                )
                candidate_mask[row_idx, :limit] = True

            neg_token_id = int(feature[KEY_NEGATIVE_TOKEN_ID])
            negative_token_ids.append(max(neg_token_id, 0))

        return {
            "input_ids": pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            ),
            "attention_mask": pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
            KEY_GRPO_RESPONSE_MASK: pad_sequence(
                response_mask, batch_first=True, padding_value=0
            ),
            KEY_GRPO_VALID_MASK: pad_sequence(
                valid_mask, batch_first=True, padding_value=False
            ),
            KEY_GRPO_OLD_LOGPROBS: pad_sequence(
                old_logprobs, batch_first=True, padding_value=0.0
            ),
            KEY_KIND_ID: torch.tensor(
                [int(f[KEY_KIND_ID]) for f in features], dtype=torch.long
            ),
            KEY_PREFIX_WEIGHT: torch.tensor(
                [f[KEY_PREFIX_WEIGHT] for f in features], dtype=torch.float32
            ),
            KEY_IS_NEGATIVE: torch.tensor(
                [f[KEY_IS_NEGATIVE] for f in features], dtype=torch.float32
            ),
            KEY_NEGATIVE_WEIGHT: torch.tensor(
                [f[KEY_NEGATIVE_WEIGHT] for f in features], dtype=torch.float32
            ),
            KEY_SAMPLE_WEIGHT: torch.tensor(
                [f[KEY_SAMPLE_WEIGHT] for f in features], dtype=torch.float32
            ),
            KEY_GRPO_ADVANTAGE: torch.tensor(
                [f[KEY_GRPO_ADVANTAGE] for f in features], dtype=torch.float32
            ),
            KEY_GRPO_HAS_OLD_LOGPROB: torch.tensor(
                [f[KEY_GRPO_HAS_OLD_LOGPROB] for f in features], dtype=torch.bool
            ),
            KEY_CANDIDATE_TOKEN_IDS: candidate_token_ids,
            KEY_CANDIDATE_PROBS: candidate_probs,
            KEY_CANDIDATE_MASK: candidate_mask,
            KEY_NEGATIVE_TOKEN_ID: torch.tensor(
                negative_token_ids, dtype=torch.long
            ),
        }


class ClosedFormTokenTrainer(Trainer):
    """Mixed-loss trainer for TFPO (pos + neg) plus the GRPO complementary objective.

    A single forward pass produces full-sequence logits. Loss is split by example
    kind: positive prefixes contribute D_KL(q_hat || pi_theta) (Eq.11); failure-
    frontier edges contribute -log(1 - pi_theta(a|s)) (Eq.12); GRPO rollouts
    contribute the variant-specific token-level ratio loss on tokens not covered
    by TFPO. The total is L_pos + gamma * L_neg + lambda * L_grpo.
    """

    def __init__(self, *args, algorithm_cfg: Dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_prob_floor = float(algorithm_cfg["teacher_prob_floor"])
        self.negative_loss_weight = float(algorithm_cfg["negative_loss_weight"])
        self.negative_prob_clamp_eps = float(algorithm_cfg["negative_prob_clamp_eps"])

        # GRPO-family parameters.
        self.grpo_loss_weight = float(algorithm_cfg.get("grpo_loss_weight", 0.0))
        self.grpo_variant = str(algorithm_cfg.get("grpo_variant", "grpo")).lower()
        self.grpo_clip_low = float(algorithm_cfg.get("grpo_clip_low", 0.2))
        self.grpo_clip_high = float(algorithm_cfg.get("grpo_clip_high", 0.2))
        self.grpo_sapo_alpha = float(algorithm_cfg.get("grpo_sapo_alpha", 1.0))
        self.grpo_dr_norm_constant = algorithm_cfg.get("grpo_dr_norm_constant")
        if self.grpo_dr_norm_constant is not None:
            self.grpo_dr_norm_constant = float(self.grpo_dr_norm_constant)

        self.debug_loss_print = bool(algorithm_cfg.get("debug_loss_print", True))
        self.debug_loss_print_max_items = int(
            algorithm_cfg.get("debug_loss_print_max_items", 5)
        )
        self.last_loss_breakdown: Dict[str, float] = {}
        self.dataset_loss_scale = 1.0
        # LR schedule floor as a fraction of the peak lr: 0.0 reproduces HF's
        # warmup-from-0 / decay-to-0; >0 warms up FROM and decays TO floor*peak
        # (e.g. 0.1 with peak 1e-5 keeps lr in [1e-6, 1e-5]).
        self.lr_min_ratio = float(algorithm_cfg.get("lr_min_ratio", 0.0))
        # Persistent-engine design: the DeepSpeed engine / optimizer is built once
        # (ensure_prepared) and reused across every RL iteration.
        self._engine_prepared = False
        self._reset_epoch_stats()

    def _estimate_optimizer_steps_per_iteration(self, dataset_len: int) -> int:
        """Optimizer steps one RL iteration (1 epoch over the dataset) will take.

        Only used to size the global LR-schedule horizon. The data is sharded across
        `num_processes` ranks; each rank then accumulates `gas` micro-batches of
        `micro` examples per optimizer step.
        """
        micro = max(1, int(self.args.per_device_train_batch_size))
        gas = max(1, int(self.args.gradient_accumulation_steps))
        world = max(1, int(getattr(self.accelerator, "num_processes", 1)))
        per_rank = math.ceil(max(1, dataset_len) / world)
        return max(1, math.ceil(per_rank / (micro * gas)))

    def _build_lr_scheduler(self, optimizer, num_training_steps: int) -> LambdaLR:
        """Linear warmup to the peak lr, then linear decay -- both bounded by a floor.

        With peak lr = optimizer base lr and floor = self.lr_min_ratio, the lr ramps
        floor*peak -> peak over warmup_ratio of the run, then decays peak -> floor*peak.
        lr_min_ratio=0.0 reproduces HF's get_linear_schedule_with_warmup (0 -> peak -> 0).
        """
        warmup_steps = int(float(self.args.warmup_ratio) * num_training_steps)
        warmup_steps = max(0, min(warmup_steps, num_training_steps))
        min_ratio = float(self.lr_min_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return min_ratio + (1.0 - min_ratio) * (step / float(warmup_steps))
            denom = max(1, num_training_steps - warmup_steps)
            frac = min(max((step - warmup_steps) / float(denom), 0.0), 1.0)
            return 1.0 - (1.0 - min_ratio) * frac

        return LambdaLR(optimizer, lr_lambda)

    def _build_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.args.per_device_train_batch_size),
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=int(self.args.dataloader_num_workers),
            drop_last=False,
            pin_memory=False,
        )

    def _drop_prepared_dataloaders(self) -> None:
        """Release accelerate's references to prepared DataLoaders.

        accelerate appends every prepared loader to Accelerator._dataloaders; since
        we prepare a fresh loader each iteration and never reuse old ones, clearing
        the list avoids slowly accumulating references to past iterations' datasets.
        """
        loaders = getattr(self.accelerator, "_dataloaders", None)
        if isinstance(loaders, list):
            loaders.clear()

    def ensure_prepared(self, dataset, num_iterations: int) -> None:
        """Build the optimizer + LR scheduler + (DeepSpeed) engine exactly ONCE.

        This is the whole point of the persistent-engine design: Accelerator.prepare()
        -- and therefore deepspeed.initialize() -- runs a single time for the entire
        RL run. The ~56GB/GPU ZeRO-3 optimizer state is allocated once, Adam momentum
        carries across iterations, and there is no per-iteration rebuild and thus no
        memory doubling / OOM.
        """
        if self._engine_prepared:
            return

        steps_per_iter = self._estimate_optimizer_steps_per_iteration(len(dataset))
        num_training_steps = max(1, int(num_iterations) * steps_per_iter)

        # Gradient checkpointing must be enabled on the base module before prepare.
        if bool(self.args.gradient_checkpointing):
            gc_kwargs = getattr(self.args, "gradient_checkpointing_kwargs", None) or {}
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs
            )
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

        # HF builds the AdamW optimizer (our ds config has no "optimizer" section)
        # honoring lr / weight_decay / betas, and a single global LR scheduler that
        # spans the whole run -- warmup once at the start, then decay -- instead of
        # restarting warmup every iteration.
        self.create_optimizer()
        self.lr_scheduler = self._build_lr_scheduler(
            self.optimizer, num_training_steps
        )

        # We deliberately do NOT pass the scheduler to prepare(): keeping it as a
        # plain torch scheduler on the base AdamW lets us step it once per optimizer
        # step over `num_training_steps`, bypassing AcceleratedScheduler's world-size
        # multiplier. DeepSpeed reads the lr straight from the base optimizer's
        # param_groups, which the scheduler mutates.
        #
        # train_micro_batch_size_per_gpu is concrete in the ds config, so prepare()
        # builds the engine without needing a probe DataLoader; per-iteration loaders
        # are prepared in run_epoch via prepare_data_loader (which never re-enters the
        # DeepSpeed engine-init path).
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.model_wrapped = self.model
        if getattr(self, "is_deepspeed_enabled", False):
            self.deepspeed = self.model

        self._engine_prepared = True
        if self.accelerator.is_main_process:
            logger.info(
                "Prepared persistent engine once | lr_schedule_steps=%d "
                "(%d iterations x ~%d optimizer steps) | deepspeed=%s",
                num_training_steps,
                int(num_iterations),
                steps_per_iter,
                bool(getattr(self, "is_deepspeed_enabled", False)),
            )

    def _maybe_log_step(self, opt_step: int, log_every: int) -> None:
        """Print the running loss breakdown every `log_every` optimizer steps.

        The custom training loop replaces HF Trainer.train(), which used to emit the
        periodic {'loss': ...} lines; this restores that visibility. last_loss_breakdown
        is the running epoch average, updated on every compute_loss call.
        """
        if opt_step % log_every != 0:
            return
        if not self.accelerator.is_main_process:
            return
        b = self.last_loss_breakdown
        try:
            lr = float(self.lr_scheduler.get_last_lr()[0])
        except Exception:
            lr = float(self.args.learning_rate)
        logger.info(
            "  step %d | loss=%.4f pos=%.4f neg=%.4f grpo=%.4f | lr=%.2e | "
            "n_pos=%d n_neg=%d n_grpo=%d",
            opt_step,
            float(b.get("loss", 0.0)),
            float(b.get("pos_loss", 0.0)),
            float(b.get("neg_loss", 0.0)),
            float(b.get("grpo_loss", 0.0)),
            lr,
            int(b.get("num_positive_examples", 0)),
            int(b.get("num_negative_examples", 0)),
            int(b.get("num_grpo_examples", 0)),
        )

    def run_epoch(self, dataset) -> Dict[str, float]:
        """Run one epoch over `dataset` on the persistent engine (manual loop).

        DeepSpeed owns gradient accumulation (gas in the config) and gradient
        clipping, so on that path we feed raw micro-batch losses and call step() every
        micro-batch -- the engine performs the real update / zeroing only at its own
        accumulation boundary. On the plain DDP / single-GPU path the Accelerator's
        accumulation plugin is 1, so we accumulate gas micro-batches ourselves and
        step at the boundary.
        """
        self._reset_epoch_stats()
        self.model.train()
        gas = max(1, int(self.args.gradient_accumulation_steps))
        log_every = max(1, int(self.args.logging_steps))
        deepspeed = bool(getattr(self, "is_deepspeed_enabled", False))

        dataloader = self.accelerator.prepare_data_loader(self._build_dataloader(dataset))
        n_batches = len(dataloader)
        start = time.time()
        num_micro = 0
        num_opt_steps = 0
        for i, batch in enumerate(dataloader):
            if deepspeed:
                # DeepSpeed owns accumulation: feed raw loss + step every micro-batch;
                # the engine performs the real update only on its own gas boundary, so
                # we advance the LR on the same gas-aligned boundary (a partial final
                # group is carried over by the engine, not force-flushed here).
                loss = self.compute_loss(self.model, batch)
                self.accelerator.backward(loss)
                self.optimizer.step()
                if (i + 1) % gas == 0:
                    self.lr_scheduler.step()
                    num_opt_steps += 1
                    self._maybe_log_step(num_opt_steps, log_every)
            else:
                # Plain DDP / single-GPU: accumulate gas micro-batches ourselves and
                # step at the boundary, flushing any partial final group.
                is_boundary = ((i + 1) % gas == 0) or ((i + 1) == n_batches)
                loss = self.compute_loss(self.model, batch) / gas
                self.accelerator.backward(loss)
                if is_boundary:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    num_opt_steps += 1
                    self._maybe_log_step(num_opt_steps, log_every)
            num_micro += 1

        self._drop_prepared_dataloaders()
        runtime = time.time() - start
        try:
            last_lr = float(self.lr_scheduler.get_last_lr()[0])
        except Exception:
            last_lr = float(self.args.learning_rate)
        return {
            "train_runtime": runtime,
            "train_micro_steps": float(num_micro),
            "train_optimizer_steps": float(num_opt_steps),
            "train_loss": float(self.last_loss_breakdown.get("loss", 0.0)),
            "learning_rate": last_lr,
        }

    def save_actor_checkpoint(self, output_dir: Path, tokenizer) -> None:
        """Gather ZeRO-3 shards and write an HF-format checkpoint for vLLM.

        accelerator.get_state_dict() is collective (every rank must call it); with
        stage3_gather_16bit_weights_on_model_save it returns the full bf16 weights on
        the main process, which we save_pretrained alongside config + tokenizer so the
        rollout / eval vLLM server can load it directly.
        """
        state_dict = self.accelerator.get_state_dict(self.model)
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(
                str(output_dir),
                state_dict=state_dict,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(str(output_dir))
        self.accelerator.wait_for_everyone()

    def _reset_epoch_stats(self) -> None:
        self._epoch_loss_sum = 0.0
        self._epoch_pos_loss_sum = 0.0
        self._epoch_neg_loss_sum = 0.0
        self._epoch_grpo_loss_sum = 0.0
        self._epoch_prefix_ce_sum = 0.0
        self._epoch_negative_prob_sum = 0.0
        self._epoch_num_positive = 0.0
        self._epoch_num_negative = 0.0
        self._epoch_num_grpo = 0.0
        self._epoch_grpo_valid_tokens = 0.0
        self._epoch_clip_fraction_sum = 0.0
        self._epoch_num_batches = 0

    def _normalize_candidate_probs(
        self,
        candidate_probs: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.where(
            candidate_mask, candidate_probs, torch.zeros_like(candidate_probs)
        )
        if self.teacher_prob_floor > 0:
            probs = torch.where(
                candidate_mask,
                probs.clamp(min=self.teacher_prob_floor),
                probs,
            )
        denom = probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return torch.where(candidate_mask, probs / denom, torch.zeros_like(probs))

    def _compute_loss_impl(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        *,
        update_stats: bool,
    ):
        # Pop auxiliary tensors.
        kind_id = inputs.pop(KEY_KIND_ID).long()
        inputs.pop(KEY_PREFIX_WEIGHT)
        is_negative = inputs.pop(KEY_IS_NEGATIVE).float()
        negative_weight = inputs.pop(KEY_NEGATIVE_WEIGHT).float()
        inputs.pop(KEY_SAMPLE_WEIGHT)
        candidate_token_ids = inputs.pop(KEY_CANDIDATE_TOKEN_IDS).long()
        candidate_probs = inputs.pop(KEY_CANDIDATE_PROBS).float()
        candidate_mask = inputs.pop(KEY_CANDIDATE_MASK).bool()
        negative_token_id = inputs.pop(KEY_NEGATIVE_TOKEN_ID).long()
        response_mask = inputs.pop(KEY_GRPO_RESPONSE_MASK).long()
        valid_mask = inputs.pop(KEY_GRPO_VALID_MASK).bool()
        old_logprobs_full = inputs.pop(KEY_GRPO_OLD_LOGPROBS).float()
        advantages = inputs.pop(KEY_GRPO_ADVANTAGE).float()
        has_old_logprob = inputs.pop(KEY_GRPO_HAS_OLD_LOGPROB).bool()

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits  # [B, T, V]

        batch_size = inputs["attention_mask"].shape[0]
        device = logits.device

        pos_kind = kind_id == KIND_ID_POSITIVE
        neg_kind = kind_id == KIND_ID_NEGATIVE
        grpo_kind = kind_id == KIND_ID_GRPO
        is_pos_neg = pos_kind | neg_kind

        # ----- TFPO positive + negative (last-token logits) -------------------
        # Use last non-pad token position for each example.
        last_token_index = inputs["attention_mask"].sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(batch_size, device=device)
        next_token_logits = logits[batch_indices, last_token_index]
        next_token_log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)

        normalized_candidate_probs = self._normalize_candidate_probs(
            candidate_probs=candidate_probs,
            candidate_mask=candidate_mask,
        )
        safe_candidate_token_ids = candidate_token_ids.masked_fill(~candidate_mask, 0)
        candidate_log_probs = next_token_log_probs.gather(
            dim=-1, index=safe_candidate_token_ids
        )
        candidate_log_probs = candidate_log_probs.masked_fill(~candidate_mask, 0.0)
        prefix_ce = -(normalized_candidate_probs * candidate_log_probs).sum(dim=1)

        positive_mask_eff = pos_kind & is_negative.lt(0.5)
        pos_terms = prefix_ce.masked_select(positive_mask_eff)
        if pos_terms.numel() > 0:
            pos_loss = pos_terms.mean()
        else:
            pos_loss = prefix_ce.new_zeros(())

        safe_negative_token_id = negative_token_id.clamp(min=0)
        neg_index = safe_negative_token_id.unsqueeze(-1)
        negative_log_prob = next_token_log_probs.gather(dim=-1, index=neg_index).squeeze(-1)
        negative_action_prob = torch.exp(negative_log_prob).clamp(
            min=self.negative_prob_clamp_eps,
            max=1.0 - self.negative_prob_clamp_eps,
        )
        neg_term = -negative_weight * torch.log1p(-negative_action_prob)

        negative_mask_eff = neg_kind & is_negative.gt(0.5)
        neg_terms = neg_term.masked_select(negative_mask_eff)
        if neg_terms.numel() > 0:
            neg_loss = neg_terms.mean()
        else:
            neg_loss = neg_term.new_zeros(())

        # ----- GRPO (full-sequence shifted logprobs) --------------------------
        grpo_loss = logits.new_zeros(())
        grpo_clip_frac = 0.0
        n_grpo_valid_tokens = 0.0

        if grpo_kind.any() and self.grpo_loss_weight > 0:
            # Shifted: logits[:, :-1, :] predicts input_ids[:, 1:].
            shifted_log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            shifted_targets = inputs["input_ids"][:, 1:].long()
            new_logprobs = shifted_log_probs.gather(
                -1, shifted_targets.unsqueeze(-1)
            ).squeeze(-1)  # [B, T-1]

            # Auxiliary tensors align via [:, 1:] (mask at full_ids index i becomes
            # mask at shifted index i-1 which corresponds to predicting input_ids[i]).
            valid_shifted = valid_mask[:, 1:]
            old_lp_shifted = old_logprobs_full[:, 1:]

            # Restrict to GRPO examples and to rows that actually have rollout
            # logprobs; for rows without logprobs, fall back to ratio=1 by using
            # new_logprobs.detach() as the old logprob (REINFORCE-style).
            kind_row_mask = grpo_kind.unsqueeze(-1)  # [B, 1]
            valid_shifted = valid_shifted & kind_row_mask

            has_lp_row = has_old_logprob.unsqueeze(-1)  # [B, 1]
            old_lp_effective = torch.where(
                has_lp_row,
                old_lp_shifted,
                new_logprobs.detach(),
            )

            # Per-token advantage broadcast from per-sequence scalar.
            adv_per_token = advantages.unsqueeze(-1).expand_as(new_logprobs)

            grpo_loss = compute_grpo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=old_lp_effective,
                advantages=adv_per_token,
                valid_mask=valid_shifted,
                variant=self.grpo_variant,
                clip_low=self.grpo_clip_low,
                clip_high=self.grpo_clip_high,
                sapo_alpha=self.grpo_sapo_alpha,
                dr_grpo_norm_constant=self.grpo_dr_norm_constant,
            )
            grpo_clip_frac = compute_clip_fraction(
                new_logprobs=new_logprobs,
                old_logprobs=old_lp_effective,
                valid_mask=valid_shifted,
                clip_low=self.grpo_clip_low,
                clip_high=self.grpo_clip_high,
            )
            n_grpo_valid_tokens = float(valid_shifted.sum().detach().cpu())

        loss = pos_loss + self.negative_loss_weight * neg_loss
        if self.grpo_loss_weight > 0:
            loss = loss + self.grpo_loss_weight * grpo_loss

        # ----- bookkeeping ----------------------------------------------------
        if update_stats:
            num_pos = float(positive_mask_eff.sum().detach().cpu())
            num_neg = float(negative_mask_eff.sum().detach().cpu())
            num_grpo = float(grpo_kind.sum().detach().cpu())

            self._epoch_loss_sum += float(loss.detach().cpu())
            self._epoch_pos_loss_sum += float(pos_loss.detach().cpu())
            self._epoch_neg_loss_sum += float(neg_loss.detach().cpu())
            self._epoch_grpo_loss_sum += float(
                grpo_loss.detach().cpu()
                if isinstance(grpo_loss, torch.Tensor)
                else grpo_loss
            )
            self._epoch_prefix_ce_sum += float(
                prefix_ce.masked_select(positive_mask_eff).sum().detach().cpu()
            )
            self._epoch_negative_prob_sum += float(
                negative_action_prob.masked_select(negative_mask_eff)
                .sum()
                .detach()
                .cpu()
            )
            self._epoch_num_positive += num_pos
            self._epoch_num_negative += num_neg
            self._epoch_num_grpo += num_grpo
            self._epoch_grpo_valid_tokens += n_grpo_valid_tokens
            self._epoch_clip_fraction_sum += float(grpo_clip_frac)
            self._epoch_num_batches += 1

            denom_batches = max(self._epoch_num_batches, 1)
            self.last_loss_breakdown = {
                "loss": self._epoch_loss_sum / denom_batches,
                "pos_loss": self._epoch_pos_loss_sum / denom_batches,
                "neg_loss": self._epoch_neg_loss_sum / denom_batches,
                "grpo_loss": self._epoch_grpo_loss_sum / denom_batches,
                "avg_prefix_ce": self._epoch_prefix_ce_sum
                / max(self._epoch_num_positive, 1.0),
                "avg_negative_action_prob": self._epoch_negative_prob_sum
                / max(self._epoch_num_negative, 1.0),
                "avg_grpo_valid_tokens_per_example": self._epoch_grpo_valid_tokens
                / max(self._epoch_num_grpo, 1.0),
                "grpo_clip_fraction": self._epoch_clip_fraction_sum / denom_batches,
                "num_positive_examples": self._epoch_num_positive,
                "num_negative_examples": self._epoch_num_negative,
                "num_grpo_examples": self._epoch_num_grpo,
            }

        return (loss, outputs) if return_outputs else loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        del kwargs
        return self._compute_loss_impl(
            model,
            inputs,
            return_outputs=return_outputs,
            update_stats=True,
        )


def create_trainer(
    model,
    tokenizer,
    train_cfg: Dict[str, Any],
    algorithm_cfg: Dict[str, Any],
    deepspeed_cfg: Optional[Dict[str, Any]],
    output_dir: Path,
    seed: int,
) -> ClosedFormTokenTrainer:
    """Create a persistent ClosedFormTokenTrainer (called once before the RL loop)."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    placeholder = TeacherPairsDataset(
        [], max_sequence_length=train_cfg.get("max_sequence_length")
    )
    collator = TeacherDataCollator(pad_token_id=pad_token_id)

    use_bf16 = bool(train_cfg.get("bf16", False)) and torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(output_dir / "_hf_trainer_tmp"),
        overwrite_output_dir=True,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_epochs_per_iteration"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        bf16=use_bf16,
        fp16=False,
        remove_unused_columns=False,
        logging_steps=int(train_cfg["logging_steps"]),
        dataloader_num_workers=int(train_cfg["dataloader_num_workers"]),
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        seed=seed,
        data_seed=seed,
        deepspeed=deepspeed_cfg.get("config_path")
        if deepspeed_cfg and deepspeed_cfg.get("enabled")
        else None,
    )

    trainer = ClosedFormTokenTrainer(
        model=model,
        args=args,
        train_dataset=placeholder,
        data_collator=collator,
        tokenizer=tokenizer,
        algorithm_cfg=algorithm_cfg,
    )
    return trainer


def train_one_iteration(
    trainer: ClosedFormTokenTrainer,
    tokenizer,
    examples: List[Dict[str, Any]],
    train_cfg: Dict[str, Any],
    output_dir: Path,
    save_checkpoint: bool = True,
    num_iterations: int = 1,
) -> Dict[str, Any]:
    """Run one RL iteration on the persistent engine (one engine for the whole run).

    The DeepSpeed engine + optimizer are built once (ensure_prepared) and reused, so
    Adam momentum carries across iterations and there is no per-iteration rebuild.
    """
    dataset = TeacherPairsDataset(
        examples=examples,
        max_sequence_length=train_cfg.get("max_sequence_length"),
    )

    trainer.ensure_prepared(dataset=dataset, num_iterations=num_iterations)
    loop_metrics = trainer.run_epoch(dataset)

    if save_checkpoint:
        # Collective ZeRO-3 gather + save (all ranks call get_state_dict inside).
        trainer.save_actor_checkpoint(output_dir, tokenizer)
    barrier()

    metrics = {
        k: float(v) for k, v in loop_metrics.items() if isinstance(v, (int, float))
    }
    metrics.update({f"train/{k}": v for k, v in trainer.last_loss_breakdown.items()})
    metrics["skipped"] = False
    barrier()
    return metrics
