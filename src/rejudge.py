"""LLM-as-judge rejudging of rule-based reward=0 rollouts.

The local math-verify rule sometimes marks a response wrong due to formatting
slack (e.g. `\\frac{1}{2}` vs `0.5`, parentheses, units), or because the stated
final answer was truncated by the 2K/4K response-length cap even though the
correct value is present in the reasoning. We pass the model's FULL response to
a stronger LLM and overwrite reward/answer_correct if the judge replies YES,
where YES means the correct answer appears anywhere in the response (final
answer OR reasoning).

Protocol selection: env var ``LLM_PROVIDER`` ∈ {anthropic, openai}.
- anthropic (default for MiniMax-M2.7 gateway): POST {BASE_URL}/v1/messages
  with headers ``x-api-key`` and ``anthropic-version``.
- openai: POST {BASE_URL}/chat/completions with ``Authorization: Bearer``.

Concurrency is bounded by an asyncio Semaphore; outcomes are cached on disk
keyed by (normalized_gold, response_fingerprint) — a sha1 of the full response —
so byte-identical responses only call the API once. An audit log records every
call with the original/rejudged reward so API quality can be spot-checked.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from src.utils import setup_logger


logger = setup_logger("rejudge")


JUDGE_SYSTEM_PROMPT = (
    "You are a lenient but careful math grader. You are given a math question, "
    "the reference answer, and a model's full response (which may contain "
    "reasoning, working, and a final answer). The response may have been cut off "
    "by a length limit, so an explicit final answer may be missing or truncated. "
    "Grade the response as correct if the correct answer — mathematically "
    "equivalent to the reference answer — appears ANYWHERE in the response, "
    "whether as the stated final answer OR inside the reasoning/working. Allow "
    "trivial differences in form (fraction vs decimal, parentheses, ordering, "
    "units, or LaTeX formatting). Judge only mathematical correctness of the "
    "value, not its formatting or whether it was clearly marked as final. "
    "Answer with exactly one word: YES if the correct answer appears anywhere, "
    "otherwise NO."
)


JUDGE_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Reference answer: {gold}\n\n"
    "Model response (may be truncated):\n{pred}\n\n"
    "Does the correct answer (mathematically equivalent to the reference answer) "
    "appear anywhere in the response, including within the reasoning? "
    "Answer with exactly one word: YES or NO."
)


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LATEX_DELIM_RE = re.compile(r"\$+|\\\(|\\\)|\\\[|\\\]")
_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\.,;:!?]+$")
# Whole-word YES/NO, case-insensitive. \b boundaries avoid false hits inside
# words like NONE / KNOW / NOT. Wrapping markup (**YES**, "YES.", `NO`) is fine
# because the surrounding chars are non-word and still form a boundary.
_VERDICT_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def normalize_answer_for_cache(text: Optional[str]) -> str:
    """Cheap, deterministic normalization used only for cache keying."""
    if text is None:
        return ""
    s = str(text)
    # Repeatedly unwrap \boxed{...} (sometimes nested).
    while True:
        m = _BOXED_RE.search(s)
        if not m:
            break
        s = s[: m.start()] + m.group(1) + s[m.end() :]
    s = _LATEX_DELIM_RE.sub("", s)
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = _WS_RE.sub("", s).strip()
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s.lower()


def response_fingerprint(text: Optional[str]) -> str:
    """Compact, deterministic fingerprint of a full model response, used for
    cache keying. The judge now reads the whole response (answer may live in the
    reasoning), so the verdict depends on the full text — we hash a
    whitespace-normalized copy to keep cache keys/file size bounded while still
    deduping byte-identical responses (e.g. repeated/greedy rollouts)."""
    normalized = _WS_RE.sub(" ", str(text or "")).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


@dataclass
class RejudgeMetrics:
    called: int = 0
    cache_hits: int = 0
    flipped: int = 0
    api_failures: int = 0
    short_circuit_empty: int = 0
    total_zero_reward: int = 0
    latency_s: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "rejudge_total_zero_reward": float(self.total_zero_reward),
            "rejudge_called": float(self.called),
            "rejudge_cache_hits": float(self.cache_hits),
            "rejudge_flipped": float(self.flipped),
            "rejudge_api_failures": float(self.api_failures),
            "rejudge_short_circuit_empty": float(self.short_circuit_empty),
            "rejudge_latency_s": float(self.latency_s),
        }


@dataclass
class _PendingCall:
    qid: Any
    rollout_index: int
    question: str
    gold: str
    pred: str
    cache_key: str
    rollout_ref: Dict[str, Any]
    flip_to_reward: float


class Rejudger:
    def __init__(self, cfg: Dict[str, Any]):
        self.enabled = bool(cfg.get("enabled", True))
        self.provider = str(cfg.get("provider") or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        self.base_url = str(cfg.get("base_url") or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.model = str(cfg.get("model") or os.getenv("LLM_MODEL", ""))
        self.api_key = str(cfg.get("api_key") or os.getenv("LLM_API_KEY", ""))
        self.anthropic_version = str(
            cfg.get("anthropic_version") or os.getenv("ANTHROPIC_VERSION", "2023-06-01")
        )
        self.max_concurrency = int(cfg.get("max_concurrency", 32))
        # Output budget for the judge. Reasoning/thinking judge models emit a
        # chain-of-thought before the YES/NO, so this must be large enough to
        # reach the verdict (the parser takes the LAST YES/NO in the output).
        self.max_tokens = int(cfg.get("max_tokens", 2048))
        # Larger output budgets need a longer wall-clock timeout.
        self.timeout_s = float(cfg.get("timeout_s", 120.0))
        self.max_retries = int(cfg.get("max_retries", 1))
        self.flip_to_reward = float(cfg.get("flip_to_reward", 1.0))
        self.cache_path: Optional[Path] = (
            Path(cfg["cache_path"]) if cfg.get("cache_path") else None
        )
        self.audit_path: Optional[Path] = (
            Path(cfg["audit_path"]) if cfg.get("audit_path") else None
        )

        if self.enabled and not self.base_url:
            logger.warning("Rejudger enabled but LLM_BASE_URL is empty; disabling.")
            self.enabled = False
        if self.enabled and not self.model:
            logger.warning("Rejudger enabled but LLM_MODEL is empty; disabling.")
            self.enabled = False
        if self.enabled and not self.api_key:
            logger.warning("Rejudger enabled but LLM_API_KEY is empty; disabling.")
            self.enabled = False

        self._cache: Dict[str, bool] = {}
        self._cache_dirty = False
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._cache[entry["key"]] = bool(entry["verdict"])
            logger.info("Rejudge cache loaded: %d entries", len(self._cache))
        except Exception as exc:
            logger.warning("Failed to load rejudge cache %s: %s", self.cache_path, exc)

    def _persist_cache(self) -> None:
        if not self.cache_path or not self._cache_dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Append-only writes keep the file useful as a resumable log.
        try:
            with self.cache_path.open("w", encoding="utf-8") as f:
                for key, verdict in self._cache.items():
                    f.write(json.dumps({"key": key, "verdict": bool(verdict)}) + "\n")
            self._cache_dirty = False
        except Exception as exc:
            logger.warning("Failed to persist rejudge cache: %s", exc)

    def _audit(self, entry: Dict[str, Any]) -> None:
        if not self.audit_path:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to append rejudge audit log: %s", exc)

    # ----- HTTP calls --------------------------------------------------------

    def _build_request(self, question: str, gold: str, pred: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        user = JUDGE_USER_TEMPLATE.format(question=question, gold=gold, pred=pred)
        if self.provider == "anthropic":
            url = f"{self.base_url}/v1/messages"
            headers = {
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            }
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "system": JUDGE_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user}],
            }
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            }
        return url, headers, payload

    @staticmethod
    def _parse_verdict(provider: str, body: Dict[str, Any]) -> Optional[bool]:
        try:
            if provider == "anthropic":
                content = body.get("content") or []
                text = next(
                    (chunk.get("text", "") for chunk in content if chunk.get("type") == "text"),
                    "",
                )
            else:
                choices = body.get("choices") or []
                text = choices[0]["message"]["content"] if choices else ""
        except (KeyError, IndexError, TypeError):
            return None

        if not text or not text.strip():
            return None
        # Whole-word YES/NO only; if the judge emits more than one
        # (e.g. "NO, wait... YES"), the LAST one is its final verdict.
        # No bare-prefix fallback, so NONE / NOT / NOPE are never misread.
        matches = _VERDICT_RE.findall(text)
        if not matches:
            return None
        return matches[-1].lower() == "yes"

    async def _call_one(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        question: str,
        gold: str,
        pred: str,
    ) -> Optional[bool]:
        url, headers, payload = self._build_request(question, gold, pred)
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        for attempt in range(self.max_retries + 1):
            try:
                async with sem, session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as resp:
                    if resp.status >= 400:
                        body_text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body_text[:200]}")
                    body = await resp.json()
                verdict = self._parse_verdict(self.provider, body)
                return verdict
            except Exception as exc:
                if attempt >= self.max_retries:
                    logger.warning(
                        "Rejudge API call failed (final): %s", str(exc)[:200]
                    )
                    return None
                await asyncio.sleep(0.5 * (attempt + 1))
        return None

    # ----- public entrypoint -------------------------------------------------

    def rejudge_batch(self, batch_rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Mutates rollouts in place; returns metrics dict."""
        metrics = RejudgeMetrics()
        if not self.enabled:
            for sample in batch_rollouts:
                for r in sample.get("rollouts", []):
                    if float(r.get("reward", 0.0)) <= 0.0:
                        metrics.total_zero_reward += 1
            return metrics.to_dict()

        # Build the list of pending calls; consult cache up front.
        pending: List[_PendingCall] = []
        key_to_pending_indices: Dict[str, List[int]] = {}
        t0 = time.monotonic()

        for sample in batch_rollouts:
            qid = sample["question_idx"]
            question = sample.get("query_text", "")
            gold = sample.get("gold_answer_text", "") or ""
            for rollout_index, rollout in enumerate(sample.get("rollouts", [])):
                if float(rollout.get("reward", 0.0)) > 0.0:
                    continue
                metrics.total_zero_reward += 1

                # Judge the model's FULL response (reasoning + final answer),
                # letting the LLM extract and grade the final answer itself.
                pred = rollout.get("response_text") or ""
                if not pred.strip():
                    metrics.short_circuit_empty += 1
                    continue
                if not gold.strip():
                    metrics.short_circuit_empty += 1
                    continue

                key = (
                    f"{normalize_answer_for_cache(gold)}||"
                    f"{response_fingerprint(pred)}"
                )
                if key in self._cache:
                    metrics.cache_hits += 1
                    if self._cache[key]:
                        self._apply_flip(rollout, metrics)
                        self._audit({
                            "qid": str(qid),
                            "rollout_index": rollout_index,
                            "source": "cache",
                            "verdict": True,
                            "original_reward": 0.0,
                            "new_reward": self.flip_to_reward,
                            "normalized_gold": normalize_answer_for_cache(gold),
                            "pred_fingerprint": response_fingerprint(pred),
                            "raw_pred": pred[:512],
                        })
                    continue

                pc = _PendingCall(
                    qid=qid,
                    rollout_index=rollout_index,
                    question=question,
                    gold=gold,
                    pred=pred,
                    cache_key=key,
                    rollout_ref=rollout,
                    flip_to_reward=self.flip_to_reward,
                )
                key_to_pending_indices.setdefault(key, []).append(len(pending))
                pending.append(pc)

        if pending:
            asyncio.run(self._run_pending(pending, key_to_pending_indices, metrics))

        metrics.latency_s = time.monotonic() - t0
        self._persist_cache()
        return metrics.to_dict()

    async def _run_pending(
        self,
        pending: List[_PendingCall],
        key_to_indices: Dict[str, List[int]],
        metrics: RejudgeMetrics,
    ) -> None:
        sem = asyncio.Semaphore(self.max_concurrency)
        # Within a single batch, dedupe by key: first occurrence calls, others wait.
        unique_keys = list(key_to_indices.keys())
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call_one(
                    session,
                    sem,
                    question=pending[key_to_indices[k][0]].question,
                    gold=pending[key_to_indices[k][0]].gold,
                    pred=pending[key_to_indices[k][0]].pred,
                )
                for k in unique_keys
            ]
            verdicts = await asyncio.gather(*tasks, return_exceptions=False)

        for key, verdict in zip(unique_keys, verdicts):
            metrics.called += 1
            if verdict is None:
                metrics.api_failures += 1
                continue
            self._cache[key] = bool(verdict)
            self._cache_dirty = True
            for idx in key_to_indices[key]:
                pc = pending[idx]
                if verdict:
                    self._apply_flip(pc.rollout_ref, metrics)
                self._audit({
                    "qid": str(pc.qid),
                    "rollout_index": pc.rollout_index,
                    "source": "api",
                    "verdict": bool(verdict),
                    "original_reward": 0.0,
                    "new_reward": pc.flip_to_reward if verdict else 0.0,
                    "normalized_gold": normalize_answer_for_cache(pc.gold),
                    "pred_fingerprint": response_fingerprint(pc.pred),
                    "raw_pred": pc.pred[:512],
                })

    def _apply_flip(self, rollout: Dict[str, Any], metrics: RejudgeMetrics) -> None:
        rollout["reward"] = float(self.flip_to_reward)
        rollout["answer_correct"] = True
        rollout["rejudged"] = True
        metrics.flipped += 1
