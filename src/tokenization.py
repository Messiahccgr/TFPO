from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _unwrap_model(model: Any) -> Any:
    current = model
    while hasattr(current, "module"):
        current = current.module
    return current


def _normalize_token_id(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_token_id(item)
            if normalized is not None:
                return normalized
        return None

    try:
        token_id = int(value)
    except (TypeError, ValueError):
        return None

    return token_id if token_id >= 0 else None


def _resolve_token_id(tokenizer: Any, token_attr: str, token_id_attr: str) -> int | None:
    token_id = _normalize_token_id(getattr(tokenizer, token_id_attr, None))
    if token_id is not None:
        return token_id

    token = getattr(tokenizer, token_attr, None)
    if token is None:
        return None

    try:
        resolved = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None

    return _normalize_token_id(resolved)


def _set_tokenizer_token_by_id(
    tokenizer: Any,
    *,
    token_attr: str,
    token_id_attr: str,
    token_id: int | None,
) -> None:
    normalized_token_id = _normalize_token_id(token_id)
    if normalized_token_id is None:
        return

    token_str = None
    try:
        token_str = tokenizer.convert_ids_to_tokens(normalized_token_id)
    except Exception:
        token_str = None

    if token_str is not None:
        try:
            setattr(tokenizer, token_attr, token_str)
        except Exception:
            pass

    try:
        setattr(tokenizer, token_id_attr, int(normalized_token_id))
    except Exception:
        pass


def load_causal_lm_tokenizer(
    name_or_path: str | Path,
    *,
    trust_remote_code: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        str(name_or_path),
        trust_remote_code=bool(trust_remote_code),
    )

    eos_token_id = _resolve_token_id(tokenizer, "eos_token", "eos_token_id")
    if eos_token_id is None:
        raise ValueError(
            f"Tokenizer at {name_or_path} must define an EOS token for causal LM usage."
        )

    # 永远保留 left padding
    tokenizer.padding_side = "left"

    pad_token_id = _resolve_token_id(tokenizer, "pad_token", "pad_token_id")

    # 对 DeepSeek / Qwen / 其他 causal LM 做兜底：
    # 没有 pad 时，用 eos 作为 pad，避免 batch generation / collator 出问题
    if pad_token_id is None:
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass

        try:
            tokenizer.pad_token_id = int(eos_token_id)
        except Exception:
            pass

    return tokenizer


def align_model_special_tokens(model: Any, tokenizer: Any) -> Any:
    raw_model = _unwrap_model(model)
    model_config = raw_model.config
    generation_config = getattr(raw_model, "generation_config", None)

    tokenizer_eos_token_id = _resolve_token_id(tokenizer, "eos_token", "eos_token_id")
    if tokenizer_eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id before aligning special tokens.")

    tokenizer_pad_token_id = _resolve_token_id(tokenizer, "pad_token", "pad_token_id")
    tokenizer_bos_token_id = _resolve_token_id(tokenizer, "bos_token", "bos_token_id")

    config_eos_token_id = _normalize_token_id(getattr(model_config, "eos_token_id", None))
    config_pad_token_id = _normalize_token_id(getattr(model_config, "pad_token_id", None))
    config_bos_token_id = _normalize_token_id(getattr(model_config, "bos_token_id", None))

    generation_eos_token_id = None
    generation_pad_token_id = None
    generation_bos_token_id = None
    if generation_config is not None:
        generation_eos_token_id = _normalize_token_id(getattr(generation_config, "eos_token_id", None))
        generation_pad_token_id = _normalize_token_id(getattr(generation_config, "pad_token_id", None))
        generation_bos_token_id = _normalize_token_id(getattr(generation_config, "bos_token_id", None))

    # canonical 选择策略：
    # 1. 优先 tokenizer（最接近实际词表）
    # 2. 再看 generation_config
    # 3. 再看 model.config
    canonical_eos_token_id = (
        tokenizer_eos_token_id
        or generation_eos_token_id
        or config_eos_token_id
    )

    canonical_pad_token_id = (
        tokenizer_pad_token_id
        or generation_pad_token_id
        or config_pad_token_id
        or canonical_eos_token_id
    )

    canonical_bos_token_id = (
        tokenizer_bos_token_id
        or generation_bos_token_id
        or config_bos_token_id
    )

    # 先把 tokenizer 本身补齐，尤其是 pad
    _set_tokenizer_token_by_id(
        tokenizer,
        token_attr="eos_token",
        token_id_attr="eos_token_id",
        token_id=canonical_eos_token_id,
    )
    _set_tokenizer_token_by_id(
        tokenizer,
        token_attr="pad_token",
        token_id_attr="pad_token_id",
        token_id=canonical_pad_token_id,
    )

    if canonical_bos_token_id is not None:
        _set_tokenizer_token_by_id(
            tokenizer,
            token_attr="bos_token",
            token_id_attr="bos_token_id",
            token_id=canonical_bos_token_id,
        )

    # 再统一 model.config
    if canonical_eos_token_id is not None:
        model_config.eos_token_id = int(canonical_eos_token_id)
    if canonical_pad_token_id is not None:
        model_config.pad_token_id = int(canonical_pad_token_id)
    if canonical_bos_token_id is not None:
        model_config.bos_token_id = int(canonical_bos_token_id)

    # 最后统一 generation_config
    if generation_config is not None:
        if canonical_eos_token_id is not None:
            generation_config.eos_token_id = int(canonical_eos_token_id)
        if canonical_pad_token_id is not None:
            generation_config.pad_token_id = int(canonical_pad_token_id)
        if canonical_bos_token_id is not None:
            generation_config.bos_token_id = int(canonical_bos_token_id)

    return model